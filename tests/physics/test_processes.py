from collections import namedtuple
import ctypes
import os
import sys
import time
import tracemalloc
import unittest

import hepunits as units
import numpy as np
from numba import njit

from core.other.typing_definitions import Charge, Float, ID, Index, ProcessID
from core.particles.kinematic_state import KinematicState
from core.particles.particles import ParticleBank
from core.particles.particles_kernels import _rotate_particle
from core.physics.g4coherent import _generate_coherent_theta_scalar
from core.physics.g4compton import _generate_compton_theta_scalar
from core.physics.interaction_buffers import InteractionBuffer, RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.processes_kernels import (
    make_coherent_kernel,
    make_compton_kernel,
    make_photoelectric_kernel,
)


class TestProcessesSoA(unittest.TestCase):
    """Тесты физических ядер взаимодействия фотонов (SoA) и генераторов углов рассеяния."""

    def setUp(self) -> None:
        self.capacity = 1000
        self.bank = ParticleBank.allocate(self.capacity)
        self.rng = np.random.default_rng(42)
        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)

        energy = np.full(self.capacity, 0.5 * units.MeV)

        pos = self.bank.state.position
        pos.x[:] = self.rng.random(self.capacity)
        pos.y[:] = self.rng.random(self.capacity)
        pos.z[:] = self.rng.random(self.capacity)

        dir_vec = self.bank.state.direction
        dir_vec.x[:] = 0.0
        dir_vec.y[:] = 0.0
        dir_vec.z[:] = 1.0

        self.bank.state.is_active[:] = True
        self.bank.initial_state.ID[:] = np.arange(self.capacity, dtype=np.uint64)
        self.bank.state.energy[:] = energy
        self.target_indices = np.arange(self.capacity, dtype=np.int64)

        self.buffer = InteractionBuffer.allocate(self.capacity)

    def test_photoelectric_kernel(self) -> None:
        """Проверка фотоэлектрического поглощения в SoA-ядре."""
        kernel = make_photoelectric_kernel(process_id=ProcessID(1))
        material_ids = np.full(self.capacity, 0, dtype=Index)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        mock_physics = self._create_mock_physics_buffer()

        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            mock_physics,
            self.rng_ctx,
        )

        self.assertEqual(self.buffer.cursor[0], self.capacity)
        np.testing.assert_array_equal(self.bank.state.energy, 0.0)
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.5 * units.MeV)
        np.testing.assert_array_equal(self.buffer.process_id, 1)
        np.testing.assert_array_equal(self.buffer.Z, 13)

    def test_photoelectric_effect(self) -> None:
        """Проверка фотоэффекта с изолированным генератором случайных чисел."""
        self.test_photoelectric_kernel()

    def test_compton_kernel_equivalence_and_performance(self) -> None:
        """Проверка воспроизводимости и инвариантов генерации углов Комптона."""
        material_ids = np.full(self.capacity, 13, dtype=Charge)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV, dtype=Float)

        new_rng = np.random.default_rng(1234)
        new_rng_ctx = RNGContext.from_numpy_rng(new_rng)

        @njit(cache=True)
        def _get_new_thetas(cap: int, energies: np.ndarray, z_vals: np.ndarray, ctx: RNGContext) -> np.ndarray:
            out = np.empty(cap, dtype=np.float64)
            for idx in range(cap):
                out[idx] = _generate_compton_theta_scalar(energies[idx], z_vals[idx], ctx)
            return out

        theta_new = _get_new_thetas(self.capacity, energy_arr, material_ids, new_rng_ctx)

        self.assertTrue(np.all(theta_new >= 0.0))
        self.assertTrue(np.all(theta_new <= np.pi))
        self.assertTrue(0.1 < np.mean(theta_new) < 2.5)

    def test_coherent_kernel_equivalence(self) -> None:
        """Проверка воспроизводимости и инвариантов генерации углов когерентного рассеяния."""
        material_ids = np.full(self.capacity, 82, dtype=Charge)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV, dtype=Float)

        new_rng = np.random.default_rng(777)
        new_rng_ctx = RNGContext.from_numpy_rng(new_rng)

        @njit(cache=True)
        def _get_new_thetas_coh(cap: int, energies: np.ndarray, z_vals: np.ndarray, ctx: RNGContext) -> np.ndarray:
            out = np.empty(cap, dtype=np.float64)
            for idx in range(cap):
                out[idx] = _generate_coherent_theta_scalar(energies[idx], z_vals[idx], ctx)
            return out

        theta_new = _get_new_thetas_coh(self.capacity, energy_arr, material_ids, new_rng_ctx)

        self.assertTrue(np.all(theta_new >= 0.0))
        self.assertTrue(np.all(theta_new <= np.pi))
        self.assertTrue(np.mean(theta_new) < 1.0)

    def test_coherent_scattering_de_morgan_and_nan_guard(self) -> None:
        """
        Регрессионный тест _generate_coherent_theta_scalar (g4coherent.py):
        1. Защита от NaN при нулевой/неположительной энергии (xx <= 0.0) или Z <= 0 (возврат 0.0).
        2. Корректная работа цикла отбора (rejection sampling) по закону де Моргана
           (выход при cost >= -1.0 и 2*r3 < 1 + cost^2) в широком диапазоне энергий.
        """
        rng = np.random.default_rng(2026)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        @njit(cache=True)
        def _eval_edge_cases(ctx: RNGContext) -> np.ndarray:
            res = np.empty(4, dtype=np.float64)
            res[0] = _generate_coherent_theta_scalar(Float(0.0), Charge(82), ctx)
            res[1] = _generate_coherent_theta_scalar(Float(-0.1), Charge(82), ctx)
            res[2] = _generate_coherent_theta_scalar(Float(0.14 * units.MeV), Charge(0), ctx)
            res[3] = _generate_coherent_theta_scalar(Float(1e-18), Charge(13), ctx)
            return res

        edge_thetas = _eval_edge_cases(rng_ctx)
        self.assertTrue(np.all(np.isfinite(edge_thetas)))
        np.testing.assert_array_equal(edge_thetas[:3], 0.0)
        self.assertGreaterEqual(edge_thetas[3], 0.0)

        # Проверка выборки при низких и средних энергиях (где возможны широкие углы рассеяния и cost вблизи -1)
        sample_count = 5000
        low_energies = np.full(sample_count, 0.005 * units.MeV, dtype=Float)
        z_carbon = np.full(sample_count, 6, dtype=Charge)

        @njit(cache=True)
        def _sample_many(energies: np.ndarray, z_vals: np.ndarray, ctx: RNGContext) -> np.ndarray:
            count = energies.shape[0]
            out = np.empty(count, dtype=np.float64)
            for idx in range(count):
                out[idx] = _generate_coherent_theta_scalar(energies[idx], z_vals[idx], ctx)
            return out

        sampled_thetas = _sample_many(low_energies, z_carbon, rng_ctx)
        self.assertTrue(np.all(np.isfinite(sampled_thetas)))
        self.assertTrue(np.all(sampled_thetas >= 0.0))
        self.assertTrue(np.all(sampled_thetas <= np.pi))
        cos_thetas = np.cos(sampled_thetas)
        self.assertTrue(np.all(cos_thetas >= -1.0))
        self.assertTrue(np.all(cos_thetas <= 1.0))
        # Убеждаемся, что при малых энергиях углы не вырождаются в константу и покрывают как переднюю, так и заднюю полусферу
        self.assertLess(float(np.min(cos_thetas)), 0.0)
        self.assertGreater(float(np.max(cos_thetas)), 0.9)

    def _create_mock_physics_buffer(self):
        class MockCSR(namedtuple("MockCSR", ["element_offsets", "element_Z", "element_fraction"])):
            pass

        csr = MockCSR(
            element_offsets=np.array([0, 1, 2], dtype=Index),
            element_Z=np.array([13, 82], dtype=Charge),
            element_fraction=np.array([1.0, 1.0], dtype=Float),
        )

        class MockPhysicsBuffer(namedtuple("MockPhysicsBuffer", ["element_csr"])):
            pass

        return MockPhysicsBuffer(element_csr=csr)

    def test_compton_kernel(self) -> None:
        kernel = make_compton_kernel(process_id=ProcessID(2))
        material_ids = np.full(self.capacity, 0, dtype=Index)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        physics_buffer = self._create_mock_physics_buffer()

        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            physics_buffer,
            self.rng_ctx,
        )

        tracemalloc.start()
        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            physics_buffer,
            self.rng_ctx,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertTrue(peak < 100_000)

        for _ in range(100):
            kernel(
                self.bank.state,
                particle_ids,
                self.target_indices,
                current_volumes,
                material_ids,
                self.buffer,
                physics_buffer,
                self.rng_ctx,
            )

        self.assertEqual(self.buffer.cursor[0], self.capacity * 102)
        self.assertTrue(np.all(self.buffer.energy_deposit > 0.0))
        self.assertTrue(np.all(self.bank.state.energy < 0.5 * units.MeV))
        np.testing.assert_array_equal(self.buffer.process_id, 2)
        dir_norms = np.sqrt(
            self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2
        )
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_coherent_kernel(self) -> None:
        kernel = make_coherent_kernel(process_id=ProcessID(3))
        material_ids = np.full(self.capacity, 1, dtype=Index)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        physics_buffer = self._create_mock_physics_buffer()

        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            physics_buffer,
            self.rng_ctx,
        )

        tracemalloc.start()
        kernel(
            self.bank.state,
            particle_ids,
            self.target_indices,
            current_volumes,
            material_ids,
            self.buffer,
            physics_buffer,
            self.rng_ctx,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertTrue(peak < 100_000)

        for _ in range(100):
            kernel(
                self.bank.state,
                particle_ids,
                self.target_indices,
                current_volumes,
                material_ids,
                self.buffer,
                physics_buffer,
                self.rng_ctx,
            )

        self.assertEqual(self.buffer.cursor[0], self.capacity * 102)
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.0)
        np.testing.assert_array_equal(self.bank.state.energy, 0.5 * units.MeV)
        np.testing.assert_array_equal(self.buffer.process_id, 3)

        dir_norms = np.sqrt(
            self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2
        )
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_rotate_particle(self) -> None:
        state = KinematicState.allocate(1)
        state.direction.x[0] = 0.0
        state.direction.y[0] = 0.0
        state.direction.z[0] = 1.0

        theta, phi = np.pi / 4, np.pi / 2

        @njit
        def _wrapper(kin_state, p_idx, angle_theta, angle_phi):
            _rotate_particle(kin_state, p_idx, angle_theta, angle_phi)

        _wrapper(state, 0, theta, phi)

        dir_x = state.direction.x[0]
        dir_y = state.direction.y[0]
        dir_z = state.direction.z[0]

        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertAlmostEqual(dir_x, 0.0, places=5)
        self.assertAlmostEqual(dir_y, np.sin(theta), places=5)
        self.assertAlmostEqual(dir_z, np.cos(theta), places=5)


if __name__ == "__main__":
    unittest.main()

