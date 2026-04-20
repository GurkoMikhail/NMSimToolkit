import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tracemalloc
import unittest
import numpy as np
import hepunits as units
import time
from numba import njit
import ctypes

from core.particles.particles_soa import ParticleBank
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.processes_soa_kernels import make_photoelectric_kernel, make_compton_kernel, make_coherent_kernel

# Old implementations for testing math correctness
import core.physics.g4compton as old_compton
import core.physics.g4coherent as old_coherent
from core.particles.particles_soa_kernels import _rotate_particle
from core.particles.kinematic_state import KinematicState
from core.other.typing_definitions import ID, Index


class TestProcessesSoA(unittest.TestCase):
    def setUp(self):
        self.capacity = 1000
        self.bank = ParticleBank.allocate(self.capacity)

        # Inject some particles
        species = np.full(self.capacity, 1, dtype=np.uint8)
        energy = np.full(self.capacity, 0.5 * units.MeV)

        pos = self.bank.state.position
        pos.x[:] = np.random.rand(self.capacity)
        pos.y[:] = np.random.rand(self.capacity)
        pos.z[:] = np.random.rand(self.capacity)

        dir_vec = self.bank.state.direction
        dir_vec.x[:] = 0.0
        dir_vec.y[:] = 0.0
        dir_vec.z[:] = 1.0

        # Simulate active
        self.bank.state.is_active[:] = True
        self.bank.initial_state.ID[:] = np.arange(self.capacity, dtype=np.uint64)
        self.bank.state.energy[:] = energy
        self.target_indices = np.arange(self.capacity, dtype=np.int64)

        self.buffer = InteractionBuffer.allocate(self.capacity)

        self.rng = np.random.default_rng(42)

        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)

    def test_photoelectric_kernel(self):
        kernel = make_photoelectric_kernel(process_id=1)
        material_ids = np.full(self.capacity, 13)
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        # Need a dummy physics buffer just so Numba doesn't crash on type inference for other kernels
        mock_physics = self._create_mock_physics_buffer()

        kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, mock_physics, self.rng_ctx)

        self.assertEqual(self.buffer.cursor[0], self.capacity)

        # Verify energy in particle state is 0
        np.testing.assert_array_equal(self.bank.state.energy, 0.0)

        # Verify energy deposit
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.5 * units.MeV)

        # Verify process_id
        np.testing.assert_array_equal(self.buffer.process_id, 1)

    def test_compton_kernel_equivalence_and_performance(self):
        material_ids = np.full(self.capacity, 13)  # Aluminum

        # 1. Physics Equivalence on Large Sample
        old_rng = np.random.default_rng(1234)
        old_theta_generator = old_compton.initialize(old_rng)

        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        material_ids = np.full(self.capacity, 13)
        theta_old = old_theta_generator(energy_arr, material_ids)

        # New implementation with identically seeded RNG
        new_rng = np.random.default_rng(1234)
        new_rng_ctx = RNGContext.from_numpy_rng(new_rng)

        # Define a wrapper to extract exactly N thetas to avoid shifting RNG via phi
        from core.physics.g4compton_soa import _generate_compton_theta_scalar

        @njit(cache=True)
        def _get_new_thetas(cap, energies, z_vals, ctx):
            out = np.empty(cap, dtype=np.float64)
            for i in range(cap):
                out[i] = _generate_compton_theta_scalar(energies[i], z_vals[i], ctx)
            return out

        theta_new = _get_new_thetas(self.capacity, energy_arr, material_ids, new_rng_ctx)

        np.testing.assert_allclose(theta_old, theta_new, rtol=1e-5)

    def test_coherent_kernel_equivalence(self):
        material_ids = np.full(self.capacity, 82)  # Lead

        old_rng = np.random.default_rng(777)
        old_theta_generator = old_coherent.initialize(old_rng)

        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        material_ids = np.full(self.capacity, 82)
        theta_old = old_theta_generator(energy_arr, material_ids)

        new_rng = np.random.default_rng(777)
        new_rng_ctx = RNGContext.from_numpy_rng(new_rng)

        from core.physics.g4coherent_soa import _generate_coherent_theta_scalar

        @njit(cache=True)
        def _get_new_thetas_coh(cap, energies, z_vals, ctx):
            out = np.empty(cap, dtype=np.float64)
            for i in range(cap):
                out[i] = _generate_coherent_theta_scalar(energies[i], z_vals[i], ctx)
            return out

        theta_new = _get_new_thetas_coh(self.capacity, energy_arr, material_ids, new_rng_ctx)

        np.testing.assert_allclose(theta_old, theta_new, rtol=1e-5)

    def _create_mock_physics_buffer(self):
        # Create a mock physics buffer with a single material Z=13
        from collections import namedtuple
        from core.other.typing_definitions import Charge, Float, Index

        class MockCSR(namedtuple('MockCSR', ['element_offsets', 'element_Z', 'element_fraction'])):
            pass

        csr = MockCSR(
            element_offsets=np.array([0, 1, 2], dtype=Index),
            element_Z=np.array([13, 82], dtype=Charge),
            element_fraction=np.array([1.0, 1.0], dtype=Float)
        )

        class MockPhysicsBuffer(namedtuple('MockPhysicsBuffer', ['element_csr'])):
            pass

        return MockPhysicsBuffer(element_csr=csr)

    def test_compton_kernel(self):
        kernel = make_compton_kernel(process_id=2)
        # Material index 0 is Z=13
        material_ids = np.full(self.capacity, 0, dtype=Index)  # Aluminum
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        physics_buffer = self._create_mock_physics_buffer()

        # 1. Memory Consumption Benchmark
        # First run to compile the kernel and allocate Numba overhead
        kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, physics_buffer, self.rng_ctx)

        tracemalloc.start()
        kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, physics_buffer, self.rng_ctx)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Since we're in Numba njit, memory should be zero (or near-zero from python wrapper overhead)
        print(f"\n[RAM] SoA Compton Scattering Peak Memory: {peak / 10**6:.6f} MB")
        # Assert minimal memory overhead (under 100 KB is safe for wrapper/dispatcher overhead)
        self.assertTrue(peak < 100_000)

        # 2. Speed Benchmark
        start = time.perf_counter()
        for _ in range(100):
            kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, physics_buffer, self.rng_ctx)
        end = time.perf_counter()
        print(f"[BENCHMARK] NEW SoA Compton Scattering 100x (N={self.capacity}): {end - start:.5f}s")

        old_rng = np.random.default_rng(1234)
        old_theta_generator = old_compton.initialize(old_rng)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        material_ids = np.full(self.capacity, 13)
        # compile old
        old_theta_generator(energy_arr, material_ids)

        tracemalloc.start()
        start = time.perf_counter()
        for _ in range(100):
            theta_old = old_theta_generator(energy_arr, material_ids)
            phi = np.pi * (old_rng.random(self.capacity) * 2 - 1)
        end = time.perf_counter()
        current, old_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[BENCHMARK] OLD vectorize Compton Scattering 100x (N={self.capacity}): {end - start:.5f}s")
        print(f"[RAM] OLD vectorize Compton Scattering Peak Memory: {old_peak / 10**6:.6f} MB")

        # 3. Correctness
        self.assertEqual(self.buffer.cursor[0], self.capacity * 102)
        self.assertTrue(np.all(self.buffer.energy_deposit > 0.0))
        self.assertTrue(np.all(self.bank.state.energy < 0.5 * units.MeV))
        np.testing.assert_array_equal(self.buffer.process_id, 2)
        dir_norms = np.sqrt(self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2)
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_coherent_kernel(self):
        kernel = make_coherent_kernel(process_id=3)
        # Material index 1 is Z=82
        material_ids = np.full(self.capacity, 1, dtype=Index)  # Lead
        particle_ids = np.arange(self.capacity, dtype=ID)
        current_volumes = np.zeros(self.capacity, dtype=Index)

        physics_buffer = self._create_mock_physics_buffer()

        # Warmup
        kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, physics_buffer, self.rng_ctx)

        tracemalloc.start()
        kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, physics_buffer, self.rng_ctx)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n[RAM] SoA Coherent Scattering Peak Memory: {peak / 10**6:.6f} MB")
        self.assertTrue(peak < 100_000)

        start = time.perf_counter()
        for _ in range(100):
            kernel(self.bank.state, particle_ids, self.target_indices, current_volumes, material_ids, self.buffer, physics_buffer, self.rng_ctx)
        end = time.perf_counter()
        print(f"[BENCHMARK] NEW SoA Coherent Scattering 100x (N={self.capacity}): {end - start:.5f}s")

        old_rng = np.random.default_rng(1234)
        old_theta_generator = old_coherent.initialize(old_rng)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        material_ids = np.full(self.capacity, 82)
        # compile old
        old_theta_generator(energy_arr, material_ids)

        tracemalloc.start()
        start = time.perf_counter()
        for _ in range(100):
            theta_old = old_theta_generator(energy_arr, material_ids)
            phi = np.pi * (old_rng.random(self.capacity) * 2 - 1)
        end = time.perf_counter()
        current, old_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[BENCHMARK] OLD vectorize Coherent Scattering 100x (N={self.capacity}): {end - start:.5f}s")
        print(f"[RAM] OLD vectorize Coherent Scattering Peak Memory: {old_peak / 10**6:.6f} MB")

        self.assertEqual(self.buffer.cursor[0], self.capacity * 102)
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.0)
        np.testing.assert_array_equal(self.bank.state.energy, 0.5 * units.MeV)
        np.testing.assert_array_equal(self.buffer.process_id, 3)

        dir_norms = np.sqrt(self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2)
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_rotate_particle(self):
        # Base vector
        state = KinematicState.allocate(1)
        state.direction.x[0] = 0.0
        state.direction.y[0] = 0.0
        state.direction.z[0] = 1.0

        theta, phi = np.pi/4, np.pi/2

        @njit
        def _wrapper(state, p_idx, theta, phi):
            _rotate_particle(state, p_idx, theta, phi)

        _wrapper(state, 0, theta, phi)

        nx = state.direction.x[0]
        ny = state.direction.y[0]
        nz = state.direction.z[0]

        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        # For an initial (0, 0, 1) and rotating theta=pi/4, phi=pi/2
        # x' ~ 0
        # y' = sin(pi/4) ~ 0.707
        # z' = cos(pi/4) ~ 0.707
        self.assertAlmostEqual(nx, 0.0, places=5)
        self.assertAlmostEqual(ny, np.sin(theta), places=5)
        self.assertAlmostEqual(nz, np.cos(theta), places=5)


if __name__ == '__main__':
    unittest.main()
