"""
Модульные и статистические тесты источников излучения (Phase 4):
- Таблицы метода псевдонимов Воза (build_alias_tables);
- Неизменяемый буфер источников (SourceBuffer) и компилятор (SourceCompiler);
- JIT-примитивы выборки и инжекции (source_kernels, inject_source_particles_kernel);
- Точечные, конические и воксельные источники (PointSource, ConePointSource, Source).
"""
from __future__ import annotations

import math
import unittest

import hepunits as units
from numba import njit
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Energy, Float, ID, Index, Length, Species, Time
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext
from core.physics.processes import ComptonScattering, PhotoelectricEffect
import core.source as source_pkg
from core.source.alias_tables import build_alias_tables
from core.source.source_buffer import SourceBuffer
from core.source.source_compiler import SourceCompiler
from core.source.source_kernels import (
    _sample_alias,
    _sample_cone_direction,
    _sample_isotropic_direction,
    _sample_voxel_position,
    _write_primary,
    inject_poisson_sources_kernel,
    inject_source_particles_kernel,
)
from core.source.sources import PointSource, Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
import settings.database_setting as database_setting
from verification.common.sources import ConePointSource


@njit(cache=True)
def _batch_sample_alias_kernel(
    offset: Index,
    count: Index,
    prob_table: NDArray[Float],
    alias_table: NDArray[np.int32],
    num_samples: int,
    rng_context: RNGContext,
) -> NDArray[np.int32]:
    """Пакетная выборка индексов из таблицы Воза для статистической верификации."""
    samples = np.empty(num_samples, dtype=np.int32)
    for sample_idx in range(num_samples):
        chosen = _sample_alias(prob_table, alias_table, offset, count, rng_context)
        samples[sample_idx] = np.int32(chosen - offset)
    return samples


@njit(cache=True)
def _batch_sample_isotropic_directions_kernel(
    num_samples: int,
    rng_context: RNGContext,
) -> tuple[NDArray[Float], NDArray[Float], NDArray[Float]]:
    """Пакетная выборка изотропных направлений для KS-теста."""
    dir_x = np.empty(num_samples, dtype=Float)
    dir_y = np.empty(num_samples, dtype=Float)
    dir_z = np.empty(num_samples, dtype=Float)
    for sample_idx in range(num_samples):
        dx, dy, dz = _sample_isotropic_direction(rng_context)
        dir_x[sample_idx] = dx
        dir_y[sample_idx] = dy
        dir_z[sample_idx] = dz
    return dir_x, dir_y, dir_z


@njit(cache=True)
def _batch_sample_single_voxel_positions_kernel(
    flat_idx: Index,
    ny: Index,
    nz: Index,
    voxel_size: Length,
    half_x: Length,
    half_y: Length,
    half_z: Length,
    local_to_global: NDArray[Float],
    num_samples: int,
    rng_context: RNGContext,
) -> tuple[NDArray[Length], NDArray[Length], NDArray[Length]]:
    """Пакетная выборка координат внутри одного вокселя для KS-теста."""
    out_x = np.empty(num_samples, dtype=Length)
    out_y = np.empty(num_samples, dtype=Length)
    out_z = np.empty(num_samples, dtype=Length)
    for sample_idx in range(num_samples):
        px, py, pz = _sample_voxel_position(
            flat_idx,
            ny,
            nz,
            voxel_size,
            half_x,
            half_y,
            half_z,
            local_to_global,
            Index(0),
            rng_context,
            Length(0.0),
            Length(0.0),
        )
        out_x[sample_idx] = px
        out_y[sample_idx] = py
        out_z[sample_idx] = pz
    return out_x, out_y, out_z


class TestVoseAliasTables(unittest.TestCase):
    """Тестирование построения и инвариантов таблиц Воза (build_alias_tables)."""

    def _reconstruct_probabilities(
        self,
        prob_table: NDArray[Float],
        alias_table: NDArray[np.int32],
    ) -> NDArray[Float]:
        """Восстанавливает исходное распределение вероятностей из таблиц prob и alias."""
        num_items = prob_table.shape[0]
        reconstructed = prob_table.astype(np.float64).copy()
        for idx in range(num_items):
            alias_idx = int(alias_table[idx])
            reconstructed[alias_idx] += 1.0 - float(prob_table[idx])
        return reconstructed / float(num_items)

    def test_alias_tables_exact_reconstruction_and_bounds(self) -> None:
        """Проверка диапазонов prob in [0, 1], alias in [0, K) и точного восстановления вероятностей (<= 1e-12)."""
        rng = np.random.default_rng(101)
        for num_items in (2, 7, 64, 513):
            raw_weights = rng.exponential(scale=2.5, size=num_items)
            probabilities = (raw_weights / np.sum(raw_weights)).astype(Float)

            prob_table, alias_table = build_alias_tables(probabilities)

            self.assertEqual(prob_table.shape, (num_items,))
            self.assertEqual(alias_table.shape, (num_items,))
            self.assertEqual(prob_table.dtype, Float)
            self.assertEqual(alias_table.dtype, np.int32)

            self.assertTrue(np.all(prob_table >= 0.0))
            self.assertTrue(np.all(prob_table <= 1.0))
            self.assertTrue(np.all(alias_table >= 0))
            self.assertTrue(np.all(alias_table < num_items))

            reconstructed = self._reconstruct_probabilities(prob_table, alias_table)
            max_abs_err = float(np.max(np.abs(reconstructed - probabilities)))
            self.assertLessEqual(max_abs_err, 1e-12)

    def test_alias_tables_degenerate_cases(self) -> None:
        """Проверка вырожденных распределений: K=1, равные вероятности, одна доминирующая компонента."""
        # 1. Вырожденный случай K = 1
        prob_single, alias_single = build_alias_tables(np.array([1.0], dtype=Float))
        np.testing.assert_allclose(prob_single, [1.0], atol=1e-15)
        np.testing.assert_array_equal(alias_single, [0])

        # 2. Равномерное распределение
        num_uniform = 16
        uniform_probs = np.full(num_uniform, 1.0 / num_uniform, dtype=Float)
        prob_unif, alias_unif = build_alias_tables(uniform_probs)
        np.testing.assert_allclose(prob_unif, np.ones(num_uniform, dtype=Float), atol=1e-12)
        reconstructed_unif = self._reconstruct_probabilities(prob_unif, alias_unif)
        np.testing.assert_allclose(reconstructed_unif, uniform_probs, atol=1e-12)

        # 3. Сильно перекошенное распределение с одной доминирующей вероятностью
        skewed_probs = np.array([1e-6, 1e-6, 1.0 - 3e-6, 1e-6], dtype=Float)
        prob_skew, alias_skew = build_alias_tables(skewed_probs)
        reconstructed_skew = self._reconstruct_probabilities(prob_skew, alias_skew)
        np.testing.assert_allclose(reconstructed_skew, skewed_probs, atol=1e-12)

    def test_alias_tables_lbyl_validation(self) -> None:
        """Проверка отказоустойчивости LBYL при некорректных входных массивах."""
        with self.assertRaises(ValueError):
            build_alias_tables(np.array([], dtype=Float))

        with self.assertRaises(ValueError):
            build_alias_tables(np.ones((2, 2), dtype=Float) * 0.25)

        with self.assertRaises(ValueError):
            build_alias_tables(np.array([-0.2, 1.2], dtype=Float))

        with self.assertRaises(ValueError):
            build_alias_tables(np.array([0.5, np.nan], dtype=Float))

        with self.assertRaises(ValueError):
            build_alias_tables(np.array([0.2, 0.3], dtype=Float))


class TestSourcesAndSourceBuffer(unittest.TestCase):
    """Тестирование Source, PointSource, ConePointSource, SourceBuffer и inject_source_particles_kernel."""

    def setUp(self) -> None:
        self.capacity = 10
        self.bank = ParticleBank.allocate(self.capacity)
        self.rng = np.random.default_rng(42)
        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)
        self.source = PointSource(
            activity=Float(1000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
        )
        self.source_buffer = SourceCompiler().compile([self.source])

    def test_inject_basic(self) -> None:
        """Проверка базовой инжекции пакета первичных частиц через inject_source_particles_kernel."""
        batch_size = 5
        time_start = Float(0.0)
        time_end = Float(1.0)
        injected_count, next_cursor = inject_source_particles_kernel(
            self.bank,
            self.source_buffer,
            Index(0),
            Index(batch_size),
            time_start,
            time_end,
            self.rng_ctx,
            Index(0),
        )

        self.assertEqual(int(injected_count), batch_size)
        self.assertEqual(int(next_cursor), batch_size)
        self.assertEqual(self.bank.count, batch_size)
        self.assertEqual(len(self.bank.active_indices), batch_size)

        active_mask = self.bank.state.is_active
        np.testing.assert_array_equal(
            self.bank.state.species[active_mask],
            np.zeros(batch_size, dtype=Species),
        )
        np.testing.assert_array_equal(
            self.bank.state.distance_traveled[active_mask],
            np.zeros(batch_size, dtype=Length),
        )
        np.testing.assert_array_equal(
            self.bank.state.position.x[active_mask],
            self.bank.initial_state.emission_position.x[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.position.y[active_mask],
            self.bank.initial_state.emission_position.y[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.position.z[active_mask],
            self.bank.initial_state.emission_position.z[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.direction.x[active_mask],
            self.bank.initial_state.emission_direction.x[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.direction.y[active_mask],
            self.bank.initial_state.emission_direction.y[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.direction.z[active_mask],
            self.bank.initial_state.emission_direction.z[active_mask],
        )
        self.assertTrue(np.all(self.bank.initial_state.emission_time[active_mask] >= time_start))
        self.assertTrue(np.all(self.bank.initial_state.emission_time[active_mask] <= time_end))

    def test_inject_exceeds_capacity_and_full_bank(self) -> None:
        """Проверка ограничения инжекции остаточной емкостью ParticleBank и поведения при полном банке."""
        inj1, cur1 = inject_source_particles_kernel(
            self.bank,
            self.source_buffer,
            Index(0),
            Index(7),
            Float(0.0),
            Float(1.0),
            self.rng_ctx,
            Index(0),
        )
        self.assertEqual(int(inj1), 7)
        self.assertEqual(self.bank.count, 7)

        inj2, cur2 = inject_source_particles_kernel(
            self.bank,
            self.source_buffer,
            Index(0),
            Index(5),
            Float(1.0),
            Float(2.0),
            self.rng_ctx,
            cur1,
        )
        self.assertEqual(int(inj2), 3)
        self.assertEqual(self.bank.count, 10)
        self.assertEqual(len(self.bank.active_indices), 10)

        inj3, _ = inject_source_particles_kernel(
            self.bank,
            self.source_buffer,
            Index(0),
            Index(5),
            Float(2.0),
            Float(3.0),
            self.rng_ctx,
            cur2,
        )
        self.assertEqual(int(inj3), 0)
        self.assertEqual(self.bank.count, 10)

    def test_fragmented_bank_slot_cursor_scanning(self) -> None:
        """Проверка записи в разреженные неактивные слоты банка с продвижением slot_cursor."""
        inject_source_particles_kernel(
            self.bank,
            self.source_buffer,
            Index(0),
            Index(10),
            Float(0.0),
            Float(1.0),
            self.rng_ctx,
            Index(0),
        )
        # Освобождаем слоты 2, 5, 8
        self.bank.state.is_active[[2, 5, 8]] = False
        self.assertEqual(len(self.bank.active_indices), 7)

        inj, next_cur = inject_source_particles_kernel(
            self.bank,
            self.source_buffer,
            Index(0),
            Index(2),
            Float(1.0),
            Float(2.0),
            self.rng_ctx,
            Index(0),
        )
        self.assertEqual(int(inj), 2)
        self.assertEqual(int(next_cur), 6)  # Заняты слоты 2 и 5, курсор остановился на 6
        self.assertTrue(bool(self.bank.state.is_active[2]))
        self.assertTrue(bool(self.bank.state.is_active[5]))
        self.assertFalse(bool(self.bank.state.is_active[8]))
        self.assertEqual(int(self.bank.initial_state.ID[2]), 10)
        self.assertEqual(int(self.bank.initial_state.ID[5]), 11)

    def test_point_source_zero_rng_overhead_for_position_and_voxel(self) -> None:
        """
        Проверка: PointSource (K=1, L=1, voxel_size=0.0, t_end == t_start) расходует ровно 2
        случайных числа на частицу (только cos_theta и phi направления, 0 чисел на воксель/позицию/энергию/время).
        """
        rng_a = np.random.default_rng(999)
        rng_b = np.random.default_rng(999)
        ctx_a = RNGContext.from_numpy_rng(rng_a)
        ctx_b = RNGContext.from_numpy_rng(rng_b)

        bank = ParticleBank.allocate(16)
        num_particles = 10
        inject_source_particles_kernel(
            bank,
            self.source_buffer,
            Index(0),
            Index(num_particles),
            Float(0.5),
            Float(0.5),
            ctx_a,
            Index(0),
        )
        # Для эталона: вызываем _sample_isotropic_direction ровно num_particles раз (по 2 числа на частицу)
        dir_x_ref, dir_y_ref, dir_z_ref = _batch_sample_isotropic_directions_kernel(num_particles, ctx_b)

        self.assertEqual(rng_a.bit_generator.state, rng_b.bit_generator.state)
        np.testing.assert_allclose(bank.state.direction.x[:num_particles], dir_x_ref, atol=1e-15)
        np.testing.assert_allclose(bank.state.direction.y[:num_particles], dir_y_ref, atol=1e-15)
        np.testing.assert_allclose(bank.state.direction.z[:num_particles], dir_z_ref, atol=1e-15)

    def test_partition_independence_in_inject_kernel(self) -> None:
        """
        Проверка независимости от разбиения батча: инжекция N частиц за 1 вызов против 2 вызовов
        K1 + K2 = N с одинаковым начальным RNGContext дает побитово идентичное состояние ParticleBank.
        """
        dist = np.array(
            [
                [[0.0, 2.0], [5.0, 1.0]],
                [[3.0, 0.0], [4.0, 7.0]],
            ],
            dtype=Float,
        )
        spectrum = [
            [Float(0.0489), Float(0.35)],
            [Float(0.0746), Float(0.50)],
            [Float(0.1405), Float(0.15)],
        ]
        src = Source(
            distribution=dist,
            activity=Float(5000.0 * units.Bq),
            voxel_size=Float(3.0 * units.mm),
            energy=spectrum,
            half_life=Float(100.0 * units.s),
        )
        src.rotate(alpha=Float(0.3), beta=Float(-0.2), gamma=Float(0.7))
        src.translate(x=Float(12.5), y=Float(-8.0), z=Float(4.2))
        sbuf = SourceCompiler().compile([src])

        total_n = 250
        part_k1 = 97
        part_k2 = total_n - part_k1

        bank_single = ParticleBank.allocate(total_n)
        bank_split = ParticleBank.allocate(total_n)
        rng_single = np.random.default_rng(123456)
        rng_split = np.random.default_rng(123456)
        ctx_single = RNGContext.from_numpy_rng(rng_single)
        ctx_split = RNGContext.from_numpy_rng(rng_split)

        t_start = Float(0.1 * units.s)
        t_end = Float(0.9 * units.s)

        inj_all, _ = inject_source_particles_kernel(
            bank_single,
            sbuf,
            Index(0),
            Index(total_n),
            t_start,
            t_end,
            ctx_single,
            Index(0),
        )
        inj_1, cur_1 = inject_source_particles_kernel(
            bank_split,
            sbuf,
            Index(0),
            Index(part_k1),
            t_start,
            t_end,
            ctx_split,
            Index(0),
        )
        inj_2, _ = inject_source_particles_kernel(
            bank_split,
            sbuf,
            Index(0),
            Index(part_k2),
            t_start,
            t_end,
            ctx_split,
            cur_1,
        )

        self.assertEqual(int(inj_all), total_n)
        self.assertEqual(int(inj_1 + inj_2), total_n)
        self.assertEqual(bank_single.count, bank_split.count)

        # Побитовое совпадение состояния ГСЧ
        self.assertEqual(rng_single.bit_generator.state, rng_split.bit_generator.state)

        # Побитовое совпадение KinematicState
        np.testing.assert_array_equal(bank_single.state.position.x, bank_split.state.position.x)
        np.testing.assert_array_equal(bank_single.state.position.y, bank_split.state.position.y)
        np.testing.assert_array_equal(bank_single.state.position.z, bank_split.state.position.z)
        np.testing.assert_array_equal(bank_single.state.direction.x, bank_split.state.direction.x)
        np.testing.assert_array_equal(bank_single.state.direction.y, bank_split.state.direction.y)
        np.testing.assert_array_equal(bank_single.state.direction.z, bank_split.state.direction.z)
        np.testing.assert_array_equal(bank_single.state.energy, bank_split.state.energy)
        np.testing.assert_array_equal(bank_single.state.is_active, bank_split.state.is_active)

        # Побитовое совпадение InitialState и NavigationState
        np.testing.assert_array_equal(
            bank_single.initial_state.ID,
            bank_split.initial_state.ID,
        )
        np.testing.assert_array_equal(
            bank_single.initial_state.emission_time,
            bank_split.initial_state.emission_time,
        )
        np.testing.assert_array_equal(
            bank_single.initial_state.emission_energy,
            bank_split.initial_state.emission_energy,
        )
        np.testing.assert_array_equal(
            bank_single.navigation_state.current_volume,
            bank_split.navigation_state.current_volume,
        )

    def test_chi2_voxel_and_spectral_line_frequencies(self) -> None:
        """
        Проверка критерия согласия хи-квадрат (N = 10^6 выборок) для частот вокселей
        и энергетических линий спектра.
        """
        num_samples = 1_000_000
        rng_local = np.random.default_rng(202604)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)

        # 1. Выборка вокселей
        dist = np.zeros((3, 3, 2), dtype=Float)
        weights = np.array([5.0, 12.0, 23.0, 8.0, 31.0, 14.0, 7.0], dtype=Float)
        coords = [(0, 0, 0), (0, 2, 1), (1, 0, 1), (1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 1)]
        for (ix, iy, iz), weight_val in zip(coords, weights):
            dist[ix, iy, iz] = weight_val

        line_weights = np.array([0.18, 0.47, 0.25, 0.10], dtype=Float)
        line_energies = np.array([0.025, 0.049, 0.075, 0.140], dtype=Float)
        spectrum = [[float(en), float(pr)] for en, pr in zip(line_energies, line_weights)]

        src = Source(
            distribution=dist,
            voxel_size=Float(2.0 * units.mm),
            energy=spectrum,
        )
        sbuf = SourceCompiler().compile([src])

        voxel_samples = _batch_sample_alias_kernel(
            Index(0),
            Index(len(weights)),
            sbuf.voxel_prob,
            sbuf.voxel_alias,
            num_samples,
            rng_ctx,
        )
        observed_vox_counts = np.bincount(voxel_samples, minlength=len(weights))
        expected_vox_counts = (weights / np.sum(weights)) * num_samples
        chi2_vox, p_value_vox = stats.chisquare(observed_vox_counts, f_exp=expected_vox_counts)
        self.assertGreater(
            float(p_value_vox),
            1e-3,
            f"Критерий хи-квадрат для вокселей отклонен: chi2={chi2_vox:.2f}, p={p_value_vox:.4g}",
        )

        # 2. Выборка спектральных линий
        line_samples = _batch_sample_alias_kernel(
            Index(0),
            Index(len(line_weights)),
            sbuf.line_prob,
            sbuf.line_alias,
            num_samples,
            rng_ctx,
        )
        observed_line_counts = np.bincount(line_samples, minlength=len(line_weights))
        expected_line_counts = (line_weights / np.sum(line_weights)) * num_samples
        chi2_line, p_value_line = stats.chisquare(observed_line_counts, f_exp=expected_line_counts)
        self.assertGreater(
            float(p_value_line),
            1e-3,
            f"Критерий хи-квадрат для спектральных линий отклонен: chi2={chi2_line:.2f}, p={p_value_line:.4g}",
        )

    def test_ks_uniformity_inside_voxel_and_direction_isotropy(self) -> None:
        """
        Проверка критерия Колмогорова-Смирнова (KS) для равномерности координат внутри вокселя
        по каждой оси и изотропии направлений (cos_theta и phi).
        """
        num_samples = 100_000
        rng_local = np.random.default_rng(777111)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)

        # Активен угловой воксель (3, 2, 1) в сетке (4, 4, 4) с voxel_size = 2.0 мм:
        # half_size = (4.0, 4.0, 4.0); границы вокселя:
        # X in [2.0, 4.0], Y in [0.0, 2.0], Z in [-2.0, 0.0]
        ny, nz = Index(4), Index(4)
        flat_idx = Index(3 * 16 + 2 * 4 + 1)
        voxel_sz = Length(2.0)
        identity_mat = np.eye(4, dtype=Float).reshape(1, 4, 4)

        px, py, pz = _batch_sample_single_voxel_positions_kernel(
            flat_idx,
            ny,
            nz,
            voxel_sz,
            Length(4.0),
            Length(4.0),
            Length(4.0),
            identity_mat,
            num_samples,
            rng_ctx,
        )
        self.assertTrue(np.all((px >= 2.0) & (px <= 4.0)))
        self.assertTrue(np.all((py >= 0.0) & (py <= 2.0)))
        self.assertTrue(np.all((pz >= -2.0) & (pz <= 0.0)))

        ks_x = stats.kstest((px - 2.0) / 2.0, "uniform")
        ks_y = stats.kstest((py - 0.0) / 2.0, "uniform")
        ks_z = stats.kstest((pz - (-2.0)) / 2.0, "uniform")
        self.assertGreater(float(ks_x.pvalue), 1e-3)
        self.assertGreater(float(ks_y.pvalue), 1e-3)
        self.assertGreater(float(ks_z.pvalue), 1e-3)

        # Изотропия направлений
        dx, dy, dz = _batch_sample_isotropic_directions_kernel(num_samples, rng_ctx)
        norms = np.sqrt(dx * dx + dy * dy + dz * dz)
        np.testing.assert_allclose(norms, np.ones(num_samples, dtype=Float), atol=1e-14)

        cos_theta_norm = (dz + 1.0) / 2.0
        phi_angles = np.arctan2(dy, dx)
        phi_norm = (phi_angles + math.pi) / (2.0 * math.pi)

        ks_cos = stats.kstest(cos_theta_norm, "uniform")
        ks_phi = stats.kstest(phi_norm, "uniform")
        self.assertGreater(float(ks_cos.pvalue), 1e-3)
        self.assertGreater(float(ks_phi.pvalue), 1e-3)

    def test_global_coordinates_with_translation_and_rotation(self) -> None:
        """
        Проверка перевода локальных координат вокселя в глобальные при ненулевой матрице
        local_to_global (поворот на 90 градусов вокруг оси Z + сдвиг).
        """
        world = Volume(
            geometry=Box(Float(200.0), Float(200.0), Float(200.0)),
            material=Material(name="Air, Dry (near sea level)", density=Float(0.0012)),
            name="World",
        )
        dist = np.zeros((2, 2, 2), dtype=Float)
        # Только воксель (1, 0, 1): локально X in [0, 2], Y in [-2, 0], Z in [0, 2]
        dist[1, 0, 1] = 1.0
        src = Source(distribution=dist, voxel_size=Float(2.0 * units.mm))
        # Поворот вокруг Z на +90 градусов: X_rot = -Y_loc, Y_rot = X_loc, Z_rot = Z_loc
        src.rotate(alpha=Float(0.0), beta=Float(0.0), gamma=Float(math.pi / 2.0))
        src.translate(x=Float(10.0), y=Float(20.0), z=Float(-5.0))
        world.add_child(src)

        active_sources = SourceCompiler().compile_scene(world)
        sbuf = SourceCompiler().compile(active_sources)

        num_particles = 500
        bank = ParticleBank.allocate(num_particles)
        rng_local = np.random.default_rng(555)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)
        inject_source_particles_kernel(
            bank,
            sbuf,
            Index(0),
            Index(num_particles),
            Float(0.0),
            Float(1.0),
            rng_ctx,
            Index(0),
        )

        # Локальные углы вокселя (1, 0, 1): X_loc in [0, 2], Y_loc in [-2, 0], Z_loc in [0, 2]
        local_corners = np.array(
            [
                [0.0, -2.0, 0.0],
                [2.0, 0.0, 2.0],
            ],
            dtype=Float,
        )
        global_corners = src.convert_to_global_position(local_corners)
        min_glob = np.min(global_corners, axis=0)
        max_glob = np.max(global_corners, axis=0)

        gx = bank.state.position.x[:num_particles]
        gy = bank.state.position.y[:num_particles]
        gz = bank.state.position.z[:num_particles]

        self.assertTrue(np.all((gx >= min_glob[0] - 1e-12) & (gx <= max_glob[0] + 1e-12)))
        self.assertTrue(np.all((gy >= min_glob[1] - 1e-12) & (gy <= max_glob[1] + 1e-12)))
        self.assertTrue(np.all((gz >= min_glob[2] - 1e-12) & (gz <= max_glob[2] + 1e-12)))

    def test_cone_point_source_direction_and_jitter(self) -> None:
        """Проверка поддержки ConePointSource (конус направлений + джиттер элементарной ячейки) в SourceBuffer."""
        max_theta = 0.25
        pitch_x = 1.6
        pitch_y = 2.8
        cone_src = ConePointSource(
            activity=Float(1000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            max_polar_angle=max_theta,
            cell_jitter_span=(pitch_x, pitch_y),
        )
        sbuf = SourceCompiler().compile([cone_src])

        num_particles = 2000
        bank = ParticleBank.allocate(num_particles)
        rng_local = np.random.default_rng(31415)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)
        inject_source_particles_kernel(
            bank,
            sbuf,
            Index(0),
            Index(num_particles),
            Float(0.0),
            Float(1.0),
            rng_ctx,
            Index(0),
        )

        px = bank.state.position.x[:num_particles]
        py = bank.state.position.y[:num_particles]
        pz = bank.state.position.z[:num_particles]
        dx = bank.state.direction.x[:num_particles]
        dy = bank.state.direction.y[:num_particles]
        dz = bank.state.direction.z[:num_particles]

        self.assertTrue(np.all((px >= -pitch_x / 2.0) & (px <= pitch_x / 2.0)))
        self.assertTrue(np.all((py >= -pitch_y / 2.0) & (py <= pitch_y / 2.0)))
        np.testing.assert_allclose(pz, np.zeros(num_particles, dtype=Length), atol=1e-15)

        norms = np.sqrt(dx * dx + dy * dy + dz * dz)
        np.testing.assert_allclose(norms, np.ones(num_particles, dtype=Float), atol=1e-14)
        self.assertTrue(np.all(dz >= math.cos(max_theta) - 1e-14))
        self.assertTrue(np.all(dz <= 1.0 + 1e-14))

    def test_source_unnormalized_distribution_and_activity(self) -> None:
        """Проверка автоматической нормировки распределения и независимости общей активности."""
        raw_dist = np.array(
            [
                [[1000.0, 2000.0], [3000.0, 4000.0]],
                [[5000.0, 6000.0], [7000.0, 22000.0]],
            ],
            dtype=float,
        )
        self.assertEqual(np.sum(raw_dist), 50000.0)

        src = Source(distribution=raw_dist, activity=Float(10 * units.MBq))
        self.assertAlmostEqual(float(np.sum(src.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.voxel_probabilities)), 1.0, places=7)
        self.assertAlmostEqual(float(src.initial_activity), float(10 * units.MBq))

        new_dist = np.ones((8, 8, 8)) * 42.0
        src.distribution = new_dist
        self.assertAlmostEqual(float(np.sum(src.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.voxel_probabilities)), 1.0, places=7)

        src2 = Source(distribution=raw_dist)
        self.assertAlmostEqual(float(src2.initial_activity), 50000.0)
        self.assertAlmostEqual(float(np.sum(src2.distribution)), 1.0, places=7)

    def test_empty_distribution_and_source_buffer_bounds_validation(self) -> None:
        """Проверка Fail-Fast LBYL на пустом распределении и выходе индексов SourceBuffer за границы."""
        with self.assertRaises(ValueError):
            Source(distribution=[])

        with self.assertRaises(ValueError):
            Source(distribution=np.empty((0, 0, 0), dtype=Float))

        valid_src = Source(distribution=np.ones((2, 2, 2), dtype=Float))
        with self.assertRaises(ValueError):
            valid_src.distribution = []

        # Проверка, что изменение voxel_size не перестраивает таблицы вероятностей/алиасов заново
        orig_prob_ref = valid_src.voxel_prob
        orig_alias_ref = valid_src.voxel_alias
        valid_src.voxel_size = Float(5.0 * units.mm)
        self.assertIs(valid_src.voxel_prob, orig_prob_ref)
        self.assertIs(valid_src.voxel_alias, orig_alias_ref)
        np.testing.assert_allclose(valid_src.size, [10.0, 10.0, 10.0])

        # Проверка границ индексов в SourceBuffer.validate()
        base_buf = SourceCompiler().compile([valid_src])
        with self.assertRaises(ValueError):
            bad_vox_idx = base_buf._replace(voxel_index=np.full_like(base_buf.voxel_index, 999))
            bad_vox_idx.validate()

        with self.assertRaises(ValueError):
            bad_vox_alias = base_buf._replace(voxel_alias=np.full_like(base_buf.voxel_alias, 999))
            bad_vox_alias.validate()

        with self.assertRaises(ValueError):
            bad_line_alias = base_buf._replace(line_alias=np.full_like(base_buf.line_alias, 5))
            bad_line_alias.validate()

        with self.assertRaises(ValueError):
            bad_cos = base_buf._replace(cos_theta_min=np.array([1.0], dtype=Float))
            bad_cos.validate()

        with self.assertRaises(ValueError):
            bad_jitter = base_buf._replace(cell_jitter=np.array([[-1.0, 0.0]], dtype=Float))
            bad_jitter.validate()

    def test_primitives_and_voxel_position_bit_identity(self) -> None:
        """
        Прямое тестирование примитивов _write_primary, _sample_cone_direction и побитового
        совпадения координат между _sample_voxel_position и inject_source_particles_kernel.
        """
        # 1. Побитовое совпадение _sample_voxel_position и inject_source_particles_kernel
        # для одновоксельного (K=1) моноэнергетического (L=1) источника (без затрат ГСЧ на алиасы)
        dist = np.zeros((3, 3, 3), dtype=Float)
        dist[2, 1, 0] = 1.0
        single_vox_src = Source(
            distribution=dist,
            voxel_size=Float(3.5 * units.mm),
            energy=Float(140.5 * units.keV),
            half_life=Float(np.inf),
        )
        sbuf = SourceCompiler().compile([single_vox_src])

        num_samples = 64
        bank = ParticleBank.allocate(num_samples)
        rng_a = np.random.default_rng(987654)
        ctx_a = RNGContext.from_numpy_rng(rng_a)
        # При time_start == time_end ГСЧ не тратится на время, только 3 числа на воксель и 2 на направление
        inject_source_particles_kernel(
            bank,
            sbuf,
            Index(0),
            Index(num_samples),
            Float(0.0),
            Float(0.0),
            ctx_a,
            Index(0),
        )

        rng_b = np.random.default_rng(987654)
        ctx_b = RNGContext.from_numpy_rng(rng_b)
        flat_idx = Index(sbuf.voxel_index[0])
        shape_y = Index(sbuf.grid_shape[0, 1])
        shape_z = Index(sbuf.grid_shape[0, 2])
        vox_sz = Float(sbuf.voxel_size[0])
        hx = Float(sbuf.half_size[0, 0])
        hy = Float(sbuf.half_size[0, 1])
        hz = Float(sbuf.half_size[0, 2])

        for idx in range(num_samples):
            px, py, pz = _sample_voxel_position(
                flat_idx,
                shape_y,
                shape_z,
                vox_sz,
                hx,
                hy,
                hz,
                sbuf.local_to_global,
                Index(0),
                ctx_b,
            )
            dx, dy, dz = _sample_cone_direction(
                Float(-1.0),
                sbuf.local_to_global,
                Index(0),
                ctx_b,
            )
            self.assertEqual(float(bank.state.position.x[idx]), float(px))
            self.assertEqual(float(bank.state.position.y[idx]), float(py))
            self.assertEqual(float(bank.state.position.z[idx]), float(pz))
            self.assertEqual(float(bank.state.direction.x[idx]), float(dx))
            self.assertEqual(float(bank.state.direction.y[idx]), float(dy))
            self.assertEqual(float(bank.state.direction.z[idx]), float(dz))

        # 2. Прямая проверка _write_primary
        test_bank = ParticleBank.allocate(2)
        _write_primary(
            test_bank,
            Index(1),
            ID(77),
            Species(0),
            Energy(0.511),
            Time(12.5),
            Float(1.0),
            Float(2.0),
            Float(3.0),
            Float(0.0),
            Float(0.0),
            Float(1.0),
        )
        self.assertTrue(bool(test_bank.state.is_active[1]))
        self.assertEqual(int(test_bank.initial_state.ID[1]), 77)
        self.assertAlmostEqual(float(test_bank.state.energy[1]), 0.511)
        self.assertAlmostEqual(float(test_bank.initial_state.emission_time[1]), 12.5)
        self.assertEqual(int(test_bank.navigation_state.current_volume[1]), -1)

    def test_facade_does_not_reexport_kernels(self) -> None:
        """Проверка соблюдения архитектурного контракта: core.source не реэкспортирует Numba-ядра."""
        for kernel_name in (
            "_sample_alias",
            "_sample_voxel_position",
            "_sample_isotropic_direction",
            "_sample_cone_direction",
            "_write_primary",
            "_emit_primary_from_source",
            "inject_source_particles_kernel",
            "inject_poisson_sources_kernel",
        ):
            self.assertNotIn(kernel_name, source_pkg.__all__)


class TestPoissonSourceStreamPhase5(unittest.TestCase):
    """
    Статистические и детерминированные тесты пуассоновского потока распада
    с прореживанием Льюиса–Шедлера и точным усечением по stop_time (Phase 5).
    """

    @staticmethod
    def _collect_stream_initial_states(
        sbuf: SourceBuffer,
        t_ref: float,
        start_time: float,
        stop_time: float,
        batch_capacity: int,
        rng_seed: int | None = None,
        rng_state: dict | None = None,
        t_last_init: float | None = None,
        pending_init: float = -1.0,
        initial_id: int = 0,
    ) -> tuple[dict[str, NDArray], dict, float, float]:
        """
        Прогоняет inject_poisson_sources_kernel батчами заданной емкости batch_capacity
        до достижения stop_time и собирает все сгенерированные InitialState.
        """
        rng = np.random.default_rng(rng_seed)
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        rng_ctx = RNGContext.from_numpy_rng(rng)

        bank = ParticleBank.allocate(batch_capacity)
        bank.count_array[0] = Index(initial_id)
        t_last = Float(start_time if t_last_init is None else t_last_init)
        pending = Float(pending_init)
        stop_val = Float(stop_time)
        ref_val = Float(t_ref)

        field_getters = (
            ("ID", bank.initial_state.ID, ID),
            ("pos_x", bank.initial_state.emission_position.x, Length),
            ("pos_y", bank.initial_state.emission_position.y, Length),
            ("pos_z", bank.initial_state.emission_position.z, Length),
            ("dir_x", bank.initial_state.emission_direction.x, Float),
            ("dir_y", bank.initial_state.emission_direction.y, Float),
            ("dir_z", bank.initial_state.emission_direction.z, Float),
            ("energy", bank.initial_state.emission_energy, Energy),
            ("emission_time", bank.initial_state.emission_time, Time),
        )
        chunks: dict[str, list[NDArray]] = {name: [] for name, _, _ in field_getters}

        while True:
            bank.state.is_active[:] = False
            injected_cnt, _, new_t_last, new_pending = inject_poisson_sources_kernel(
                bank, sbuf, stop_val, ref_val, t_last, pending, rng_ctx, Index(-1), Index(0)
            )
            t_last, pending = Float(new_t_last), Float(new_pending)
            cnt = int(injected_cnt)
            if cnt > 0:
                for name, arr, _ in field_getters:
                    chunks[name].append(arr[:cnt].copy())
            if float(pending) >= float(stop_val) or cnt == 0:
                break

        collected = {
            name: (np.concatenate(chunks[name]) if chunks[name] else np.empty(0, dtype=dt))
            for name, _, dt in field_getters
        }
        return collected, dict(rng.bit_generator.state), float(t_last), float(pending)

    def test_short_lived_source_terminates_when_exhausted_before_max_particles(self) -> None:
        """
        Проверка отсутствия бесконечного цикла в inject_poisson_sources_kernel при stop_time=inf,
        когда короткоживущий источник полностью распадается раньше достижения квоты max_particles > A_0 / lambda.
        """
        src = PointSource(
            activity=Float(20.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            half_life=Float(1.0 * units.s),
        )
        sbuf = SourceCompiler().compile([src])
        bank = ParticleBank.allocate(200)
        rng = np.random.default_rng(2026)
        rng_ctx = RNGContext.from_numpy_rng(rng)
        injected_cnt, _, _, pending = inject_poisson_sources_kernel(
            bank, sbuf, Float(np.inf), Float(0.0), Float(0.0), Float(-1.0), rng_ctx, Index(200), Index(0)
        )
        self.assertGreater(int(injected_cnt), 0)
        self.assertLess(int(injected_cnt), 200)
        self.assertTrue(math.isinf(float(pending)))

    def test_poisson_decay_count_mean_and_variance(self) -> None:
        """
        Проверка распределения общего числа распадов за интервал [0, T]:
        среднее и дисперсия по независимым сидам должны соответствовать закону Пуассона
        с параметром Lambda(T) = A_0 * (1 - exp(-lambda * T)) / lambda (а также A_0 * T при lambda = 0).
        """
        # 1. Распадающийся источник: T_1/2 = 1.0 с, T = 1.0 с, A_0 = 400 Бк
        half_life = Float(1.0 * units.s)
        stop_time = Float(1.0 * units.s)
        initial_activity = Float(400.0 * units.Bq)
        decay_const = float(np.log(2.0) / half_life)
        expected_lambda = float(initial_activity) * (1.0 - math.exp(-decay_const * float(stop_time))) / decay_const

        decay_src = PointSource(
            activity=initial_activity,
            energy=Float(140.5 * units.keV),
            half_life=half_life,
        )
        sbuf = SourceCompiler().compile([decay_src])

        num_trials = 400
        bank = ParticleBank.allocate(1000)
        counts = np.zeros(num_trials, dtype=np.int64)
        for trial_idx in range(num_trials):
            bank.state.is_active[:] = False
            bank.count_array[0] = Index(0)
            rng = np.random.default_rng(10_000 + trial_idx)
            rng_ctx = RNGContext.from_numpy_rng(rng)
            injected_cnt, _, _, pending = inject_poisson_sources_kernel(
                bank,
                sbuf,
                stop_time,
                Float(0.0),
                Float(0.0),
                Float(-1.0),
                rng_ctx,
                Index(-1),
                Index(0),
            )
            counts[trial_idx] = int(injected_cnt)
            self.assertGreaterEqual(float(pending), float(stop_time))

        sample_mean = float(np.mean(counts))
        sample_var = float(np.var(counts, ddof=1))
        std_err_mean = math.sqrt(expected_lambda / num_trials)
        std_err_var = expected_lambda * math.sqrt(2.0 / (num_trials - 1))

        self.assertAlmostEqual(sample_mean, expected_lambda, delta=3.5 * std_err_mean)
        self.assertAlmostEqual(sample_var, expected_lambda, delta=3.5 * std_err_var)

        # 2. Источник с бесконечным периодом полураспада (decay_constant == 0.0)
        const_src = PointSource(
            activity=Float(300.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            half_life=Float(np.inf),
        )
        sbuf_const = SourceCompiler().compile([const_src])
        self.assertEqual(float(sbuf_const.decay_constant[0]), 0.0)
        expected_const_lambda = float(300.0 * units.Bq) * float(stop_time)

        const_counts = np.zeros(num_trials, dtype=np.int64)
        for trial_idx in range(num_trials):
            bank.state.is_active[:] = False
            bank.count_array[0] = Index(0)
            rng = np.random.default_rng(50_000 + trial_idx)
            rng_ctx = RNGContext.from_numpy_rng(rng)
            injected_cnt, _, _, _ = inject_poisson_sources_kernel(
                bank,
                sbuf_const,
                stop_time,
                Float(0.0),
                Float(0.0),
                Float(-1.0),
                rng_ctx,
                Index(-1),
                Index(0),
            )
            const_counts[trial_idx] = int(injected_cnt)

        self.assertAlmostEqual(
            float(np.mean(const_counts)),
            expected_const_lambda,
            delta=3.5 * math.sqrt(expected_const_lambda / num_trials),
        )

    def test_emission_time_distribution_and_multi_source_proportions(self) -> None:
        """
        Проверка распределения времен распада ~ A(t) (критерий Колмогорова–Смирнова)
        и пропорций вкладов нескольких источников с различными периодами полураспада.
        """
        stop_time = Float(2.0 * units.s)
        hl_fast = Float(0.8 * units.s)
        hl_slow = Float(4.0 * units.s)
        act_fast = Float(8000.0 * units.Bq)
        act_slow = Float(5000.0 * units.Bq)
        e_fast = Float(0.1405 * units.MeV)
        e_slow = Float(0.511 * units.MeV)

        src_fast = PointSource(activity=act_fast, energy=e_fast, half_life=hl_fast)
        src_slow = PointSource(activity=act_slow, energy=e_slow, half_life=hl_slow)
        sbuf = SourceCompiler().compile([src_fast, src_slow])

        collected, _, _, pending = self._collect_stream_initial_states(
            sbuf=sbuf,
            t_ref=0.0,
            start_time=0.0,
            stop_time=float(stop_time),
            batch_capacity=2048,
            rng_seed=202605,
        )

        times = collected["emission_time"]
        energies = collected["energy"]
        total_emitted = len(times)
        self.assertGreater(total_emitted, 10_000)
        self.assertTrue(np.all((times >= 0.0) & (times < float(stop_time))))
        self.assertGreaterEqual(pending, float(stop_time))

        lam_fast = float(np.log(2.0) / hl_fast)
        lam_slow = float(np.log(2.0) / hl_slow)
        integral_fast = float(act_fast) * (1.0 - math.exp(-lam_fast * float(stop_time))) / lam_fast
        integral_slow = float(act_slow) * (1.0 - math.exp(-lam_slow * float(stop_time))) / lam_slow
        expected_total = integral_fast + integral_slow

        # Проверка общего числа и долей каждого источника
        self.assertAlmostEqual(float(total_emitted), expected_total, delta=4.0 * math.sqrt(expected_total))

        mask_fast = np.isclose(energies, float(e_fast), rtol=0.0, atol=1e-12)
        mask_slow = np.isclose(energies, float(e_slow), rtol=0.0, atol=1e-12)
        self.assertEqual(int(np.count_nonzero(mask_fast)) + int(np.count_nonzero(mask_slow)), total_emitted)

        observed_frac_fast = float(np.count_nonzero(mask_fast)) / float(total_emitted)
        expected_frac_fast = integral_fast / expected_total
        frac_std = math.sqrt(expected_frac_fast * (1.0 - expected_frac_fast) / float(total_emitted))
        self.assertAlmostEqual(observed_frac_fast, expected_frac_fast, delta=3.5 * frac_std)

        # Критерий Колмогорова-Смирнова для времен эмиссии каждого источника в отдельности
        times_fast = times[mask_fast]
        cdf_fast = (1.0 - np.exp(-lam_fast * times_fast)) / (1.0 - math.exp(-lam_fast * float(stop_time)))
        ks_fast = stats.kstest(cdf_fast, "uniform")
        self.assertGreater(float(ks_fast.pvalue), 1e-3)

        times_slow = times[mask_slow]
        cdf_slow = (1.0 - np.exp(-lam_slow * times_slow)) / (1.0 - math.exp(-lam_slow * float(stop_time)))
        ks_slow = stats.kstest(cdf_slow, "uniform")
        self.assertGreater(float(ks_slow.pvalue), 1e-3)

        # И для суммарного потока обоих источников
        cdf_mix = (
            float(act_fast) * (1.0 - np.exp(-lam_fast * times)) / lam_fast
            + float(act_slow) * (1.0 - np.exp(-lam_slow * times)) / lam_slow
        ) / expected_total
        ks_mix = stats.kstest(cdf_mix, "uniform")
        self.assertGreater(float(ks_mix.pvalue), 1e-3)

    def test_strict_stop_time_and_full_bank_no_rng_consumption(self) -> None:
        """
        Проверка граничных условий inject_poisson_sources_kernel:
        - Когда банк полностью занят (нет свободных слотов), ГСЧ не расходуется и состояние потока не меняется;
        - Когда pending_arrival_time >= stop_time, ядро мгновенно завершается без расхода ГСЧ;
        - Все сгенерированные частицы строго удовлетворяют emission_time < stop_time.
        """
        src = PointSource(
            activity=Float(5000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            half_life=Float(2.0 * units.s),
        )
        sbuf = SourceCompiler().compile([src])

        bank = ParticleBank.allocate(16)
        bank.state.is_active[:] = True  # Все слоты заняты

        rng = np.random.default_rng(777)
        rng_ctx = RNGContext.from_numpy_rng(rng)
        state_before = rng.bit_generator.state["state"]["state"]

        injected_cnt, _, new_t_last, new_pending = inject_poisson_sources_kernel(
            bank,
            sbuf,
            Float(1.0 * units.s),
            Float(0.0),
            Float(123.0),
            Float(150.0),
            rng_ctx,
            Index(-1),
            Index(0),
        )
        self.assertEqual(int(injected_cnt), 0)
        self.assertEqual(float(new_t_last), 123.0)
        self.assertEqual(float(new_pending), 150.0)
        self.assertEqual(rng.bit_generator.state["state"]["state"], state_before)

        # Случай, когда отложенный распад уже лежит за пределами stop_time
        bank.state.is_active[:] = False
        stop_ns = Float(500.0)
        injected_cnt2, _, new_t_last2, new_pending2 = inject_poisson_sources_kernel(
            bank,
            sbuf,
            stop_ns,
            Float(0.0),
            Float(450.0),
            Float(500.0),
            rng_ctx,
            Index(-1),
            Index(0),
        )
        self.assertEqual(int(injected_cnt2), 0)
        self.assertEqual(float(new_t_last2), 450.0)
        self.assertEqual(float(new_pending2), 500.0)
        self.assertEqual(rng.bit_generator.state["state"]["state"], state_before)

    def test_initial_state_bitwise_independence_from_batch_capacity(self) -> None:
        """
        Проверка побитовой независимости всех сгенерированных InitialState на интервале [0, T]
        от выбранного размера батча batch_capacity (например, 100, 350, 2000).
        """
        dist = np.arange(1.0, 28.0, dtype=Float).reshape((3, 3, 3))
        src1 = Source(
            distribution=dist,
            voxel_size=Float(2.0 * units.mm),
            activity=Float(6000.0 * units.Bq),
            energy=[(0.1405, 0.8), (0.511, 0.2)],
            half_life=Float(1.5 * units.s),
        )
        src2 = PointSource(
            activity=Float(4000.0 * units.Bq),
            energy=Float(0.364 * units.MeV),
            half_life=Float(3.0 * units.s),
        )
        sbuf = SourceCompiler().compile([src1, src2])

        stop_time = float(0.15 * units.s)
        seed = 918273

        ref_states, ref_rng_state, ref_t_last, ref_pending = self._collect_stream_initial_states(
            sbuf=sbuf,
            t_ref=0.0,
            start_time=0.0,
            stop_time=stop_time,
            batch_capacity=2000,
            rng_seed=seed,
        )
        self.assertGreater(len(ref_states["ID"]), 1000)

        for cap in (100, 350):
            states_cap, rng_state_cap, t_last_cap, pending_cap = self._collect_stream_initial_states(
                sbuf=sbuf,
                t_ref=0.0,
                start_time=0.0,
                stop_time=stop_time,
                batch_capacity=cap,
                rng_seed=seed,
            )
            self.assertEqual(t_last_cap, ref_t_last)
            self.assertEqual(pending_cap, ref_pending)
            self.assertEqual(rng_state_cap["state"]["state"], ref_rng_state["state"]["state"])
            for key in ref_states:
                self.assertTrue(
                    np.array_equal(states_cap[key], ref_states[key]),
                    f"Поле InitialState '{key}' отличается при batch_capacity={cap}!",
                )

    def test_stateful_resume_two_stage_vs_continuous_bitwise_identity(self) -> None:
        """
        Проверка побитового совпадения (Stateful Resume):
        1. Двухэтапный прогон [0, T1] + [T1, T2] дает побитово идентичный массив InitialState
           и состояние ГСЧ как непрерывный прогон [0, T2] при любом batch_capacity (100, 350, 2000);
        2. Полная симуляция в SimulationManager с выровненным буфером в точке T1 дает 100%
           побитово идентичные взаимодействия и начальные состояния между [0, T2] и [0, T1] + [T1, T2].
        """
        src1 = PointSource(
            activity=Float(5000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            half_life=Float(1.2 * units.s),
        )
        src2 = PointSource(
            activity=Float(3000.0 * units.Bq),
            energy=Float(511.0 * units.keV),
            half_life=Float(4.0 * units.s),
        )
        sbuf = SourceCompiler().compile([src1, src2])

        t1 = float(0.08 * units.s)
        t2 = float(0.16 * units.s)
        seed = 424242

        continuous_states, cont_rng_state, cont_t_last, cont_pending = self._collect_stream_initial_states(
            sbuf=sbuf,
            t_ref=0.0,
            start_time=0.0,
            stop_time=t2,
            batch_capacity=2000,
            rng_seed=seed,
        )

        for cap in (100, 350, 2000):
            stage1_states, s1_rng_state, s1_t_last, s1_pending = self._collect_stream_initial_states(
                sbuf=sbuf,
                t_ref=0.0,
                start_time=0.0,
                stop_time=t1,
                batch_capacity=cap,
                rng_seed=seed,
            )
            self.assertTrue(np.all(stage1_states["emission_time"] < t1))
            self.assertGreaterEqual(s1_pending, t1)

            stage2_states, s2_rng_state, s2_t_last, s2_pending = self._collect_stream_initial_states(
                sbuf=sbuf,
                t_ref=0.0,
                start_time=t1,
                stop_time=t2,
                batch_capacity=cap,
                rng_state=s1_rng_state,
                t_last_init=s1_t_last,
                pending_init=s1_pending,
                initial_id=len(stage1_states["ID"]),
            )
            self.assertTrue(np.all((stage2_states["emission_time"] >= t1) & (stage2_states["emission_time"] < t2)))
            self.assertEqual(s2_t_last, cont_t_last)
            self.assertEqual(s2_pending, cont_pending)
            self.assertEqual(s2_rng_state["state"]["state"], cont_rng_state["state"]["state"])

            for key in continuous_states:
                combined = np.concatenate([stage1_states[key], stage2_states[key]])
                self.assertTrue(
                    np.array_equal(combined, continuous_states[key]),
                    f"Двухэтапный поток InitialState '{key}' не совпал с непрерывным при batch_capacity={cap}!",
                )

        # Полная сквозная проверка SimulationManager (транспорт + источники) с единым ГСЧ
        # при фиксированном batch_capacity и режима stop_time=None
        nai_mat = database_setting.material_database["Sodium Iodide"]
        world = Volume(
            geometry=Box(Float(1000.0), Float(1000.0), Float(1000.0)),
            material=nai_mat,
            name="World",
        )
        world.add_child(
            PointSource(
                activity=Float(5000.0 * units.Bq),
                energy=Float(140.5 * units.keV),
                half_life=Float(2.0 * units.s),
            )
        )

        # Режим stop_time=None (ограничение по квоте particles_number с малым batch_capacity)
        mgr_quota = SimulationManager(
            scene=world,
            stop_time=None,
            particles_number=125,
            batch_capacity=40,
            seed=777,
        )
        mgr_quota.start()
        mgr_quota.join()
        self.assertEqual(int(mgr_quota.bank.count_array[0]), 125)
        self.assertGreater(float(mgr_quota.global_timer), 0.0)

        def _drain_queue_chunks(mgr: SimulationManager) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
            init_chunks: list[dict[str, NDArray]] = []
            inter_chunks: list[dict[str, NDArray]] = []
            while True:
                item = mgr.queue.get_nowait()
                if item is None or item == "stop":
                    break
                if item["type"].value == "initial_states":
                    init_chunks.append(item["data"])
                elif item["type"].value == "interactions":
                    inter_chunks.append(item["data"])
            init_merged = {
                k: np.concatenate([c[k] for c in init_chunks]) for k in init_chunks[0]
            }
            inter_merged = {
                k: np.concatenate([c[k] for c in inter_chunks]) for k in inter_chunks[0]
            }
            return init_merged, inter_merged

        # Побитовый досчёт [0, T1] + [T1, T2] == [0, T2] с полным циклом транспорта и эмиссии
        # при фиксированном batch_capacity и едином rng
        t_split = Float(0.012 * units.s)
        t_end = Float(0.020 * units.s)
        fixed_batch_cap = 1

        mgr_s1 = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(processes_list=[PhotoelectricEffect, ComptonScattering]),
            start_time=Float(0.0),
            stop_time=t_split,
            particles_number=500,
            batch_capacity=fixed_batch_cap,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=888,
        )
        mgr_s1.start()
        mgr_s1.join()
        s1_count = int(mgr_s1.bank.count_array[0])
        self.assertGreater(s1_count, 10)
        init_s1, inter_s1 = _drain_queue_chunks(mgr_s1)

        mgr_s2 = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(processes_list=[PhotoelectricEffect, ComptonScattering]),
            start_time=t_split,
            stop_time=t_end,
            particles_number=500,
            batch_capacity=fixed_batch_cap,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=888,
            resume_state={
                "initial_particle_id": s1_count,
                "rng_state": mgr_s1.rng.bit_generator.state,
                "t_ref": mgr_s1.t_ref,
                "t_last": mgr_s1.t_last,
                "pending_arrival_time": mgr_s1.pending_arrival_time,
            },
        )
        mgr_s2.start()
        mgr_s2.join()
        init_s2, inter_s2 = _drain_queue_chunks(mgr_s2)

        mgr_cont = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(processes_list=[PhotoelectricEffect, ComptonScattering]),
            start_time=Float(0.0),
            stop_time=t_end,
            particles_number=500,
            batch_capacity=fixed_batch_cap,
            buffer_capacity=2000,
            min_energy=Float(1.0 * units.keV),
            seed=888,
        )
        mgr_cont.start()
        mgr_cont.join()
        init_cont, inter_cont = _drain_queue_chunks(mgr_cont)

        self.assertEqual(int(mgr_cont.bank.count_array[0]), int(mgr_s2.bank.count_array[0]))
        self.assertEqual(
            mgr_cont.rng.bit_generator.state["state"]["state"],
            mgr_s2.rng.bit_generator.state["state"]["state"],
        )
        for key in init_cont:
            combined_init = np.concatenate([init_s1[key], init_s2[key]])
            self.assertTrue(
                np.array_equal(combined_init, init_cont[key]),
                f"Канал initial_states['{key}'] не совпал побитово между [0, T2] и [0, T1]+[T1, T2]!",
            )
        for key in inter_cont:
            combined_inter = np.concatenate([inter_s1[key], inter_s2[key]])
            self.assertTrue(
                np.array_equal(combined_inter, inter_cont[key]),
                f"Канал interactions['{key}'] не совпал побитово между [0, T2] и [0, T1]+[T1, T2]!",
            )


if __name__ == "__main__":
    unittest.main()

