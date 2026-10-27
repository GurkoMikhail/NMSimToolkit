"""
Модульные и статистические тесты источников излучения:
- Таблицы метода псевдонимов Воза (build_alias_tables);
- Неизменяемый буфер источников (SourceBuffer) и компилятор (SourceCompiler);
- JIT-примитивы выборки и инжекции (source_kernels, inject_poisson_sources_kernel, inject_secondary_sources_kernel);
- Композиционная модель источников (Source, SpatialDistribution, RadiationChannel, ParticleSource).
"""
from __future__ import annotations

import math
import unittest

import hepunits as units
from numba import njit
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.models import SimulationManagerConfig
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import (
    Charge,
    Energy,
    Float,
    ID,
    Index,
    Length,
    ProcessID,
    Species,
    Time,
)
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext, SecondaryActivationBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.processes import ComptonScattering, PhotoelectricEffect
import core.source as source_pkg
from core.source.alias_tables import build_alias_tables
from core.source.primary_source import PrimarySource
from core.source.secondary_source import SecondarySource
from core.source.source_buffer import SourceBuffer, SpatialType, SpectrumType
from core.source.source_compiler import SourceCompiler
from core.source.source_kernels import (
    _emit_primary_from_source,
    _sample_alias,
    _sample_cone_direction,
    _sample_isotropic_direction,
    _sample_voxel_position,
    _write_primary,
    inject_poisson_sources_kernel,
    inject_secondary_sources_kernel,
)
from core.source.sources import (
    ConeAngular,
    ContinuousSpectrum,
    DiscreteSpectrum,
    IsotropicAngular,
    ParticleSource,
    ParticleSpecies,
    PointDistribution,
    RadiationChannel,
    Source,
    VoxelDistribution,
)
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
import settings.database_setting as database_setting


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
    source_buffer: SourceBuffer,
    channel_idx: Index,
    flat_idx: Index,
    num_samples: int,
    rng_context: RNGContext,
) -> tuple[NDArray[Length], NDArray[Length], NDArray[Length]]:
    """Пакетная выборка координат внутри одного вокселя для KS-теста."""
    out_x = np.empty(num_samples, dtype=Length)
    out_y = np.empty(num_samples, dtype=Length)
    out_z = np.empty(num_samples, dtype=Length)
    for sample_idx in range(num_samples):
        px, py, pz = _sample_voxel_position(
            source_buffer,
            channel_idx,
            flat_idx,
            rng_context,
        )
        out_x[sample_idx] = px
        out_y[sample_idx] = py
        out_z[sample_idx] = pz
    return out_x, out_y, out_z


@njit(cache=True)
def _batch_emit_primary_kernel(
    bank: ParticleBank,
    source_buffer: SourceBuffer,
    channel_idx: Index,
    num_to_inject: Index,
    emission_time: Time,
    rng_ctx: RNGContext,
    start_slot_cursor: Index = Index(0),
) -> tuple[Index, Index]:
    """Пакетный вызов _emit_primary_from_source для почастичной статистической проверки примитива эмиссии."""
    is_active = bank.state.is_active
    count_array = bank.count_array
    capacity = Index(is_active.shape[0])
    slot_cursor = start_slot_cursor if start_slot_cursor >= Index(0) else Index(0)
    next_id = ID(count_array[0])
    injected = Index(0)
    while injected < num_to_inject:
        while slot_cursor < capacity and is_active[slot_cursor]:
            slot_cursor += Index(1)
        if slot_cursor >= capacity:
            break
        _emit_primary_from_source(
            bank,
            source_buffer,
            channel_idx,
            slot_cursor,
            next_id,
            emission_time,
            rng_ctx,
        )
        next_id += ID(1)
        slot_cursor += Index(1)
        injected += Index(1)
    count_array[0] = Index(next_id)
    return injected, slot_cursor


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
        self.source = Source.from_point(
            activity=Float(1000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
        )
        self.source_buffer = SourceCompiler().compile([self.source])

    def test_inject_basic(self) -> None:
        """Проверка базовой инжекции пакета первичных частиц через inject_poisson_sources_kernel."""
        batch_size = 5
        time_start = Float(0.0)
        injected_count, next_cursor, _, _ = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            time_start,
            time_start,
            Float(np.nan),
            self.rng_ctx,
            Index(batch_size),
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
        np.testing.assert_array_equal(
            self.bank.initial_state.parent_ID[active_mask],
            np.full(batch_size, -1, dtype=ID),
        )

    def test_inject_exceeds_capacity_and_full_bank(self) -> None:
        """Проверка ограничения инжекции остаточной емкостью ParticleBank и поведения при полном банке."""
        inj1, cur1, t_last1, pend1 = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            Float(0.0),
            Float(0.0),
            Float(np.nan),
            self.rng_ctx,
            Index(7),
            Index(0),
        )
        self.assertEqual(int(inj1), 7)
        self.assertEqual(self.bank.count, 7)

        inj2, cur2, t_last2, pend2 = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            Float(0.0),
            t_last1,
            pend1,
            self.rng_ctx,
            Index(5),
            cur1,
        )
        self.assertEqual(int(inj2), 3)
        self.assertEqual(self.bank.count, 10)
        self.assertEqual(len(self.bank.active_indices), 10)

        inj3, _, _, _ = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            Float(0.0),
            t_last2,
            pend2,
            self.rng_ctx,
            Index(5),
            cur2,
        )
        self.assertEqual(int(inj3), 0)
        self.assertEqual(self.bank.count, 10)

    def test_fragmented_bank_slot_cursor_scanning(self) -> None:
        """Проверка записи в разреженные неактивные слоты банка с продвижением slot_cursor."""
        _, _, t_last, pend = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            Float(0.0),
            Float(0.0),
            Float(np.nan),
            self.rng_ctx,
            Index(10),
            Index(0),
        )
        # Освобождаем слоты 2, 5, 8
        self.bank.state.is_active[[2, 5, 8]] = False
        self.assertEqual(len(self.bank.active_indices), 7)

        inj, next_cur, _, _ = inject_poisson_sources_kernel(
            self.bank,
            self.source_buffer,
            Float(np.inf),
            Float(0.0),
            t_last,
            pend,
            self.rng_ctx,
            Index(2),
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
        Проверка: PointSource (K=1, L=1, voxel_size=0.0) в _emit_primary_from_source расходует ровно 2
        случайных числа на частицу (только cos_theta и phi направления, 0 чисел на воксель/позицию/энергию).
        """
        rng_a = np.random.default_rng(999)
        rng_b = np.random.default_rng(999)
        ctx_a = RNGContext.from_numpy_rng(rng_a)
        ctx_b = RNGContext.from_numpy_rng(rng_b)

        bank = ParticleBank.allocate(16)
        num_particles = 10
        _batch_emit_primary_kernel(
            bank,
            self.source_buffer,
            Index(0),
            Index(num_particles),
            Time(0.5),
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
        src = Source.from_voxels(
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

        inj_all, _, _, _ = inject_poisson_sources_kernel(
            bank_single,
            sbuf,
            Float(np.inf),
            t_start,
            t_start,
            Float(np.nan),
            ctx_single,
            Index(total_n),
            Index(0),
        )
        inj_1, cur_1, t_last_1, pend_1 = inject_poisson_sources_kernel(
            bank_split,
            sbuf,
            Float(np.inf),
            t_start,
            t_start,
            Float(np.nan),
            ctx_split,
            Index(part_k1),
            Index(0),
        )
        inj_2, _, _, _ = inject_poisson_sources_kernel(
            bank_split,
            sbuf,
            Float(np.inf),
            t_start,
            t_last_1,
            pend_1,
            ctx_split,
            Index(part_k2),
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

        src = Source.from_voxels(
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
        flat_idx = Index(3 * 16 + 2 * 4 + 1)
        voxel_sz = Length(2.0)
        corner_dist = np.zeros((4, 4, 4), dtype=Float)
        corner_dist[3, 2, 1] = 1.0
        corner_sbuf = SourceCompiler.compile([Source.from_voxels(corner_dist, voxel_size=voxel_sz)])

        px, py, pz = _batch_sample_single_voxel_positions_kernel(
            corner_sbuf,
            Index(0),
            flat_idx,
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
        src = Source.from_voxels(distribution=dist, voxel_size=Float(2.0 * units.mm))
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
        inject_poisson_sources_kernel(
            bank,
            sbuf,
            Float(np.inf),
            Float(0.0),
            Float(0.0),
            Float(np.nan),
            rng_ctx,
            Index(num_particles),
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
        """Проверка поддержки конического точечного источника (ConeAngular + cell_jitter) в SourceBuffer."""
        max_theta = 0.25
        pitch_x = 1.6
        pitch_y = 2.8
        cone_src = Source.from_point(
            activity=Float(1000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            max_polar_angle=max_theta,
            cell_jitter=(pitch_x, pitch_y),
        )
        sbuf = SourceCompiler().compile([cone_src])

        num_particles = 2000
        bank = ParticleBank.allocate(num_particles)
        rng_local = np.random.default_rng(31415)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)
        inject_poisson_sources_kernel(
            bank,
            sbuf,
            Float(np.inf),
            Float(0.0),
            Float(0.0),
            Float(np.nan),
            rng_ctx,
            Index(num_particles),
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

        src = Source.from_voxels(distribution=raw_dist, activity=Float(10 * units.MBq))
        assert isinstance(src.spatial, VoxelDistribution)
        self.assertAlmostEqual(float(np.sum(src.spatial.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.spatial.voxel_probabilities)), 1.0, places=7)
        self.assertAlmostEqual(float(src.initial_activity), float(10 * units.MBq))

        new_dist = np.ones((8, 8, 8)) * 42.0
        src.spatial = VoxelDistribution(distribution=new_dist, voxel_size=src.spatial.voxel_size)
        self.assertAlmostEqual(float(np.sum(src.spatial.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.spatial.voxel_probabilities)), 1.0, places=7)

        src2 = Source.from_voxels(distribution=raw_dist)
        assert isinstance(src2.spatial, VoxelDistribution)
        self.assertAlmostEqual(float(src2.initial_activity), 50000.0)
        self.assertAlmostEqual(float(np.sum(src2.spatial.distribution)), 1.0, places=7)

    def _assert_source_buffer_invariants(self, buf: SourceBuffer) -> None:
        """Проверяет структурные и физические инварианты скомпилированного SourceBuffer."""
        num_channels = int(buf.initial_emission_rate.shape[0])
        total_voxels = int(buf.voxel_index.shape[0])
        total_lines = int(buf.line_energy.shape[0])

        for field_name, arr in zip(buf._fields, buf):
            self.assertTrue(arr.flags.c_contiguous, f"Массив {field_name} должен быть C-contiguous")

        self.assertEqual(buf.source_id.shape, (num_channels,))
        self.assertEqual(buf.source_id.dtype, Index)
        self.assertEqual(buf.species.shape, (num_channels,))
        self.assertEqual(buf.species.dtype, Species)
        self.assertEqual(buf.initial_emission_rate.shape, (num_channels,))
        self.assertEqual(buf.initial_emission_rate.dtype, Float)
        self.assertEqual(buf.decay_constant.shape, (num_channels,))
        self.assertEqual(buf.decay_constant.dtype, Float)
        self.assertEqual(buf.spatial_type.shape, (num_channels,))
        self.assertEqual(buf.spatial_type.dtype, Index)
        self.assertEqual(buf.spectrum_type.shape, (num_channels,))
        self.assertEqual(buf.spectrum_type.dtype, Index)
        self.assertEqual(buf.cos_theta_min.shape, (num_channels,))
        self.assertEqual(buf.cos_theta_min.dtype, Float)
        self.assertEqual(buf.cell_jitter.shape, (num_channels, 2))
        self.assertEqual(buf.cell_jitter.dtype, Float)
        self.assertEqual(buf.voxel_size.shape, (num_channels,))
        self.assertEqual(buf.voxel_size.dtype, Float)
        self.assertEqual(buf.half_size.shape, (num_channels, 3))
        self.assertEqual(buf.half_size.dtype, Float)
        self.assertEqual(buf.grid_shape.shape, (num_channels, 3))
        self.assertEqual(buf.grid_shape.dtype, Index)
        self.assertEqual(buf.voxel_range.shape, (num_channels, 2))
        self.assertEqual(buf.voxel_range.dtype, Index)
        self.assertEqual(buf.local_to_global.shape, (num_channels, 4, 4))
        self.assertEqual(buf.local_to_global.dtype, Float)

        self.assertEqual(buf.voxel_index.dtype, np.int32)
        self.assertEqual(buf.voxel_prob.shape, (total_voxels,))
        self.assertEqual(buf.voxel_prob.dtype, Float)
        self.assertEqual(buf.voxel_alias.shape, (total_voxels,))
        self.assertEqual(buf.voxel_alias.dtype, np.int32)

        self.assertEqual(buf.spectrum_offsets.shape, (num_channels + 1,))
        self.assertEqual(buf.spectrum_offsets.dtype, Index)
        self.assertEqual(int(buf.spectrum_offsets[0]), 0)
        self.assertTrue(np.all(np.diff(buf.spectrum_offsets) >= 0))
        self.assertEqual(int(buf.spectrum_offsets[-1]), total_lines)

        self.assertEqual(buf.line_energy.dtype, Float)
        self.assertEqual(buf.line_grid_high.shape, (total_lines,))
        self.assertEqual(buf.line_grid_high.dtype, Float)
        self.assertEqual(buf.line_pdf_ratio.shape, (total_lines,))
        self.assertEqual(buf.line_pdf_ratio.dtype, Float)
        self.assertEqual(buf.line_prob.shape, (total_lines,))
        self.assertEqual(buf.line_prob.dtype, Float)
        self.assertEqual(buf.line_alias.shape, (total_lines,))
        self.assertEqual(buf.line_alias.dtype, np.int32)

        if num_channels > 0:
            self.assertTrue(np.all((buf.source_id >= 0) & (buf.source_id < num_channels)))
            self.assertTrue(np.all(buf.initial_emission_rate >= 0.0))
            self.assertTrue(np.all(buf.decay_constant >= 0.0))
            self.assertTrue(np.all((buf.cos_theta_min >= -1.0) & (buf.cos_theta_min <= 1.0)))
            self.assertTrue(np.all(buf.cell_jitter >= 0.0))
            self.assertTrue(np.all(buf.voxel_size >= 0.0))

        for ch_idx in range(num_channels):
            v_start = int(buf.voxel_range[ch_idx, 0])
            v_end = int(buf.voxel_range[ch_idx, 1])
            self.assertTrue(0 <= v_start <= v_end <= total_voxels)
            if int(buf.spatial_type[ch_idx]) == int(SpatialType.POINT):
                self.assertEqual(v_start, v_end)
            else:
                v_cnt = v_end - v_start
                self.assertGreater(v_cnt, 0)
                grid_total = int(np.prod(buf.grid_shape[ch_idx]))
                self.assertTrue(np.all((buf.voxel_index[v_start:v_end] >= 0) & (buf.voxel_index[v_start:v_end] < grid_total)))
                self.assertTrue(np.all((buf.voxel_prob[v_start:v_end] >= 0.0) & (buf.voxel_prob[v_start:v_end] <= 1.0)))
                self.assertTrue(np.all((buf.voxel_alias[v_start:v_end] >= 0) & (buf.voxel_alias[v_start:v_end] < v_cnt)))

            l_start = int(buf.spectrum_offsets[ch_idx])
            l_end = int(buf.spectrum_offsets[ch_idx + 1])
            l_cnt = l_end - l_start
            self.assertGreater(l_cnt, 0)
            self.assertTrue(np.all(buf.line_energy[l_start:l_end] > 0.0))
            self.assertTrue(np.all((buf.line_prob[l_start:l_end] >= 0.0) & (buf.line_prob[l_start:l_end] <= 1.0)))
            self.assertTrue(np.all((buf.line_alias[l_start:l_end] >= 0) & (buf.line_alias[l_start:l_end] < l_cnt)))
            if int(buf.spectrum_type[ch_idx]) == int(SpectrumType.CONTINUOUS_TRAPEZOID):
                self.assertTrue(np.all(buf.line_grid_high[l_start:l_end] > buf.line_energy[l_start:l_end]))
                self.assertTrue(np.all((buf.line_pdf_ratio[l_start:l_end] >= 0.0) & (buf.line_pdf_ratio[l_start:l_end] <= 1.0)))

    def test_empty_distribution_and_source_buffer_bounds_validation(self) -> None:
        """Проверка Fail-Fast LBYL на пустом распределении и структурных инвариантов SourceBuffer."""
        with self.assertRaises(ValueError):
            VoxelDistribution(distribution=[])

        with self.assertRaises(ValueError):
            VoxelDistribution(distribution=np.empty((0, 0, 0), dtype=Float))

        valid_src = Source.from_voxels(distribution=np.ones((2, 2, 2), dtype=Float))
        with self.assertRaises(ValueError):
            Source(spatial=PointDistribution(), channels=())

        assert isinstance(valid_src.spatial, VoxelDistribution)
        np.testing.assert_allclose(valid_src.size, [8.0, 8.0, 8.0])

        self._assert_source_buffer_invariants(SourceBuffer.empty())
        self._assert_source_buffer_invariants(self.source_buffer)
        self._assert_source_buffer_invariants(SourceCompiler().compile([valid_src]))

    def test_primitives_and_voxel_position_bit_identity(self) -> None:
        """
        Прямое тестирование примитивов _write_primary, _sample_cone_direction и побитового
        совпадения координат между _sample_voxel_position и _emit_primary_from_source.
        """
        # 1. Побитовое совпадение _sample_voxel_position и _emit_primary_from_source
        # для одновоксельного (K=1) моноэнергетического (L=1) источника (без затрат ГСЧ на алиасы)
        dist = np.zeros((3, 3, 3), dtype=Float)
        dist[2, 1, 0] = 1.0
        single_vox_src = Source.from_voxels(
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
        _batch_emit_primary_kernel(
            bank,
            sbuf,
            Index(0),
            Index(num_samples),
            Time(0.0),
            ctx_a,
            Index(0),
        )

        rng_b = np.random.default_rng(987654)
        ctx_b = RNGContext.from_numpy_rng(rng_b)
        flat_idx = Index(sbuf.voxel_index[0])

        for idx in range(num_samples):
            px, py, pz = _sample_voxel_position(
                sbuf,
                Index(0),
                flat_idx,
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
            "inject_poisson_sources_kernel",
            "inject_secondary_sources_kernel",
        ):
            self.assertNotIn(kernel_name, source_pkg.__all__)

    def test_point_distribution_zero_voxel_table_allocation(self) -> None:
        """
        Критерий приёмки Фазы 1 и Фазы 4:
        1. Для Source(spatial=PointDistribution()) компилятор SourceCompiler выделяет 0 элементов (0 байт)
           в воксельных массивах SourceBuffer (voxel_index, voxel_prob, voxel_alias).
        2. При cell_jitter == (0.0, 0.0) и произвольной матрице local_to_global (поворот + сдвиг)
           мировая координата рождения побитово совпадает со столбцом сдвига (m03, m13, m23).
        3. При ненулевом cell_jitter = (jitter_x, jitter_y) координаты равномерно распределены
           в прямоугольнике [-jitter_x/2, +jitter_x/2] x [-jitter_y/2, +jitter_y/2] (критерий Колмогорова–Смирнова).
        """
        exact_pt_src = Source(
            spatial=PointDistribution(cell_jitter=(Float(0.0), Float(0.0))),
            channels=(
                RadiationChannel(
                    species=ParticleSpecies.PHOTON,
                    spectrum=DiscreteSpectrum.from_energies(Float(140.5 * units.keV)),
                    angular=IsotropicAngular(),
                ),
            ),
            activity=Float(500.0 * units.Bq),
        )
        exact_pt_src.rotate(alpha=Float(0.45), beta=Float(-0.31), gamma=Float(1.12))
        exact_pt_src.translate(x=Float(17.25 * units.mm), y=Float(-9.5 * units.mm), z=Float(42.125 * units.mm))

        sbuf_exact = SourceCompiler().compile([exact_pt_src])
        self._assert_source_buffer_invariants(sbuf_exact)
        self.assertEqual(sbuf_exact.num_channels, 1)
        self.assertEqual(int(sbuf_exact.spatial_type[0]), int(SpatialType.POINT))
        self.assertEqual(sbuf_exact.voxel_index.shape, (0,))
        self.assertEqual(sbuf_exact.voxel_prob.shape, (0,))
        self.assertEqual(sbuf_exact.voxel_alias.shape, (0,))
        self.assertEqual(sbuf_exact.voxel_index.nbytes, 0)
        self.assertEqual(sbuf_exact.voxel_prob.nbytes, 0)
        self.assertEqual(sbuf_exact.voxel_alias.nbytes, 0)
        np.testing.assert_array_equal(sbuf_exact.voxel_range[0], [0, 0])

        num_exact = 256
        bank_exact = ParticleBank.allocate(num_exact)
        rng_exact = np.random.default_rng(202641)
        ctx_exact = RNGContext.from_numpy_rng(rng_exact)
        _batch_emit_primary_kernel(
            bank_exact,
            sbuf_exact,
            Index(0),
            Index(num_exact),
            Time(0.0),
            ctx_exact,
            Index(0),
        )
        expected_shift_x = float(sbuf_exact.local_to_global[0, 0, 3])
        expected_shift_y = float(sbuf_exact.local_to_global[0, 1, 3])
        expected_shift_z = float(sbuf_exact.local_to_global[0, 2, 3])
        np.testing.assert_array_equal(
            bank_exact.state.position.x[:num_exact],
            np.full(num_exact, expected_shift_x, dtype=Length),
        )
        np.testing.assert_array_equal(
            bank_exact.state.position.y[:num_exact],
            np.full(num_exact, expected_shift_y, dtype=Length),
        )
        np.testing.assert_array_equal(
            bank_exact.state.position.z[:num_exact],
            np.full(num_exact, expected_shift_z, dtype=Length),
        )

        # Проверка ненулевого cell_jitter по критерию Колмогорова–Смирнова
        jitter_x = Float(1.6 * units.mm)
        jitter_y = Float(2.8 * units.mm)
        jitter_pt_src = Source(
            spatial=PointDistribution(cell_jitter=(jitter_x, jitter_y)),
            channels=(
                RadiationChannel(
                    species=ParticleSpecies.PHOTON,
                    spectrum=DiscreteSpectrum.from_energies(Float(140.5 * units.keV)),
                    angular=IsotropicAngular(),
                ),
            ),
            activity=Float(500.0 * units.Bq),
        )
        jitter_pt_src.translate(x=Float(5.0 * units.mm), y=Float(-3.0 * units.mm), z=Float(12.0 * units.mm))
        sbuf_jitter = SourceCompiler().compile([jitter_pt_src])
        self._assert_source_buffer_invariants(sbuf_jitter)
        self.assertEqual(sbuf_jitter.voxel_index.nbytes, 0)

        num_jitter = 50_000
        bank_jitter = ParticleBank.allocate(num_jitter)
        rng_jitter = np.random.default_rng(202642)
        ctx_jitter = RNGContext.from_numpy_rng(rng_jitter)
        _batch_emit_primary_kernel(
            bank_jitter,
            sbuf_jitter,
            Index(0),
            Index(num_jitter),
            Time(0.0),
            ctx_jitter,
            Index(0),
        )
        pos_x = bank_jitter.state.position.x[:num_jitter] - 5.0
        pos_y = bank_jitter.state.position.y[:num_jitter] - (-3.0)
        pos_z = bank_jitter.state.position.z[:num_jitter]
        self.assertTrue(np.all((pos_x >= -jitter_x / 2.0) & (pos_x <= jitter_x / 2.0)))
        self.assertTrue(np.all((pos_y >= -jitter_y / 2.0) & (pos_y <= jitter_y / 2.0)))
        np.testing.assert_array_equal(pos_z, np.full(num_jitter, 12.0, dtype=Length))

        ks_res_x = stats.kstest((pos_x + jitter_x / 2.0) / jitter_x, "uniform")
        ks_res_y = stats.kstest((pos_y + jitter_y / 2.0) / jitter_y, "uniform")
        self.assertGreater(float(ks_res_x.pvalue), 1e-3)
        self.assertGreater(float(ks_res_y.pvalue), 1e-3)

    def test_continuous_trapezoid_spectrum_chi2_and_ks(self) -> None:
        """
        Критерий приёмки Фазы 1 и Фазы 4: выборка из ContinuousSpectrum в ядре эмиссии
        воспроизводит аналитическую кусочно-линейную плотность как по критерию согласия
        хи-квадрат, так и по критерию Колмогорова–Смирнова (точные аналитические значения CDF).
        """
        energy_grid = np.array([0.05, 0.15, 0.35, 0.60, 1.00], dtype=Float)
        pdf_values = np.array([0.20, 1.40, 0.80, 1.90, 0.10], dtype=Float)
        cont_spec = ContinuousSpectrum(energy_grid=energy_grid, pdf_values=pdf_values)
        cont_src = Source(
            spatial=PointDistribution(),
            channels=(
                RadiationChannel(
                    species=ParticleSpecies.ELECTRON,
                    spectrum=cont_spec,
                    angular=IsotropicAngular(),
                ),
            ),
            activity=Float(1000.0 * units.Bq),
        )
        sbuf = SourceCompiler().compile([cont_src])
        self._assert_source_buffer_invariants(sbuf)
        self.assertEqual(int(sbuf.spectrum_type[0]), int(SpectrumType.CONTINUOUS_TRAPEZOID))

        num_samples = 1_000_000
        bank = ParticleBank.allocate(num_samples)
        rng_local = np.random.default_rng(202610)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)
        _batch_emit_primary_kernel(
            bank,
            sbuf,
            Index(0),
            Index(num_samples),
            Time(0.0),
            rng_ctx,
            Index(0),
        )
        sampled_energies = bank.state.energy[:num_samples]
        self.assertTrue(np.all((sampled_energies >= energy_grid[0]) & (sampled_energies <= energy_grid[-1])))
        np.testing.assert_array_equal(
            bank.state.species[:num_samples],
            np.full(num_samples, int(ParticleSpecies.ELECTRON), dtype=Species),
        )

        # 1. Критерий Колмогорова–Смирнова по точной аналитической кусочно-квадратичной CDF трапеций
        seg_widths = energy_grid[1:] - energy_grid[:-1]
        seg_areas = 0.5 * (pdf_values[:-1] + pdf_values[1:]) * seg_widths
        total_area = float(np.sum(seg_areas))
        cum_areas = np.concatenate([[0.0], np.cumsum(seg_areas)])

        seg_indices = np.clip(
            np.searchsorted(energy_grid, sampled_energies, side="right") - 1,
            0,
            len(seg_widths) - 1,
        )
        e_left_arr = energy_grid[seg_indices]
        width_arr = seg_widths[seg_indices]
        p_left_arr = pdf_values[seg_indices]
        p_right_arr = pdf_values[seg_indices + 1]
        local_frac = (sampled_energies - e_left_arr) / width_arr
        local_integral = width_arr * (
            p_left_arr * local_frac + 0.5 * (p_right_arr - p_left_arr) * local_frac * local_frac
        )
        analytical_cdf_vals = (cum_areas[seg_indices] + local_integral) / total_area
        ks_stat = stats.kstest(analytical_cdf_vals, "uniform")
        self.assertGreater(
            float(ks_stat.pvalue),
            1e-3,
            f"Критерий Колмогорова–Смирнова для ContinuousSpectrum отклонен: D={ks_stat.statistic:.5f}, p={ks_stat.pvalue:.4g}",
        )

        # 2. Разбиваем каждый трапециевидный интервал на 5 подынтервалов (всего 20 бинов) для критерия хи-квадрат
        bin_edges_list: list[float] = [float(energy_grid[0])]
        expected_probs: list[float] = []
        for seg_idx in range(len(energy_grid) - 1):
            e_left = float(energy_grid[seg_idx])
            e_right = float(energy_grid[seg_idx + 1])
            p_left = float(pdf_values[seg_idx])
            p_right = float(pdf_values[seg_idx + 1])
            sub_edges = np.linspace(e_left, e_right, 6)
            for sub_idx in range(5):
                sub_a = float(sub_edges[sub_idx])
                sub_b = float(sub_edges[sub_idx + 1])
                fa = p_left + (p_right - p_left) * (sub_a - e_left) / (e_right - e_left)
                fb = p_left + (p_right - p_left) * (sub_b - e_left) / (e_right - e_left)
                expected_probs.append(0.5 * (fa + fb) * (sub_b - sub_a) / total_area)
                bin_edges_list.append(sub_b)

        observed_counts, _ = np.histogram(sampled_energies, bins=np.array(bin_edges_list, dtype=Float))
        expected_counts = np.array(expected_probs, dtype=Float) * num_samples
        chi2_stat, p_val = stats.chisquare(observed_counts, f_exp=expected_counts)
        self.assertGreater(
            float(p_val),
            1e-3,
            f"Критерий хи-квадрат для ContinuousSpectrum отклонен: chi2={chi2_stat:.2f}, p={p_val:.4g}",
        )

    def test_multi_channel_voxel_source_shared_alias_table_and_branching_ratios(self) -> None:
        """
        Критерий приёмки Фазы 1 и Фазы 4: многоканальный Source с несколькими RadiationChannel на одном
        VoxelDistribution разделяет одну и ту же воксельную таблицу Воза (общий voxel_range)
        и испускает частицы разных видов в пропорции branching_ratio, в том числе при совместной
        компиляции с другими источниками сцены.
        """
        dist = np.array([[[1.0, 3.0], [2.0, 4.0]], [[0.0, 5.0], [1.0, 0.0]]], dtype=Float)
        active_voxels_count = int(np.count_nonzero(dist > 0.0))
        vox_spatial = VoxelDistribution(distribution=dist, voxel_size=Float(2.0 * units.mm))

        br_photon = Float(0.8)
        br_electron = Float(1.2)
        multi_src = Source(
            spatial=vox_spatial,
            channels=(
                RadiationChannel(
                    species=ParticleSpecies.PHOTON,
                    spectrum=DiscreteSpectrum.from_energies(Float(0.1405 * units.MeV)),
                    angular=IsotropicAngular(),
                    branching_ratio=br_photon,
                ),
                RadiationChannel(
                    species=ParticleSpecies.ELECTRON,
                    spectrum=ContinuousSpectrum(
                        energy_grid=np.array([0.05, 0.25, 0.50], dtype=Float),
                        pdf_values=np.array([1.0, 2.0, 0.2], dtype=Float),
                    ),
                    angular=ConeAngular(max_polar_angle=Float(0.5)),
                    branching_ratio=br_electron,
                ),
            ),
            activity=Float(10_000.0 * units.Bq),
            half_life=Float(np.inf),
        )

        sbuf = SourceCompiler().compile([multi_src])
        self._assert_source_buffer_invariants(sbuf)
        self.assertEqual(sbuf.num_channels, 2)
        # Оба канала ссылаются на один и тот же срез воксельной таблицы без дублирования памяти
        self.assertEqual(len(sbuf.voxel_index), active_voxels_count)
        self.assertEqual(len(sbuf.voxel_prob), active_voxels_count)
        self.assertEqual(len(sbuf.voxel_alias), active_voxels_count)
        np.testing.assert_array_equal(sbuf.voxel_range[0], [0, active_voxels_count])
        np.testing.assert_array_equal(sbuf.voxel_range[1], [0, active_voxels_count])
        self.assertAlmostEqual(float(sbuf.initial_emission_rate[0]), float(10_000.0 * units.Bq * br_photon))
        self.assertAlmostEqual(float(sbuf.initial_emission_rate[1]), float(10_000.0 * units.Bq * br_electron))

        # Проверка совместной компиляции с точечным и вторым воксельным источниками
        pt_companion = Source.from_point(
            activity=Float(5000.0 * units.Bq),
            energy=Float(0.511 * units.MeV),
            half_life=Float(np.inf),
        )
        dist_second = np.array([[[0.0, 2.0], [3.0, 0.0]]], dtype=Float)
        second_vox_count = int(np.count_nonzero(dist_second > 0.0))
        vox_companion = Source.from_voxels(
            distribution=dist_second,
            voxel_size=Float(4.0 * units.mm),
            activity=Float(3000.0 * units.Bq),
            energy=Float(0.364 * units.MeV),
            half_life=Float(np.inf),
        )
        sbuf_combined = SourceCompiler().compile([pt_companion, multi_src, vox_companion])
        self._assert_source_buffer_invariants(sbuf_combined)
        self.assertEqual(sbuf_combined.num_channels, 4)
        self.assertEqual(len(sbuf_combined.voxel_index), active_voxels_count + second_vox_count)
        np.testing.assert_array_equal(sbuf_combined.source_id, [0, 1, 1, 2])
        np.testing.assert_array_equal(sbuf_combined.voxel_range[0], [0, 0])
        np.testing.assert_array_equal(sbuf_combined.voxel_range[1], [0, active_voxels_count])
        np.testing.assert_array_equal(sbuf_combined.voxel_range[2], [0, active_voxels_count])
        np.testing.assert_array_equal(
            sbuf_combined.voxel_range[3],
            [active_voxels_count, active_voxels_count + second_vox_count],
        )

        # Проверяем пуассоновский поток обоих каналов multi_src
        num_quanta = 20_000
        bank = ParticleBank.allocate(num_quanta)
        rng_local = np.random.default_rng(202611)
        rng_ctx = RNGContext.from_numpy_rng(rng_local)
        inj_cnt, _, _, _ = inject_poisson_sources_kernel(
            bank,
            sbuf,
            Float(np.inf),
            Float(0.0),
            Float(0.0),
            Float(-1.0),
            rng_ctx,
            Index(num_quanta),
            Index(0),
        )
        self.assertEqual(int(inj_cnt), num_quanta)
        emitted_species = bank.state.species[:num_quanta]
        photon_cnt = int(np.count_nonzero(emitted_species == int(ParticleSpecies.PHOTON)))
        electron_cnt = int(np.count_nonzero(emitted_species == int(ParticleSpecies.ELECTRON)))
        self.assertEqual(photon_cnt + electron_cnt, num_quanta)

        total_br = float(br_photon + br_electron)
        exp_photons = num_quanta * (float(br_photon) / total_br)
        exp_electrons = num_quanta * (float(br_electron) / total_br)
        _, p_val = stats.chisquare([photon_cnt, electron_cnt], f_exp=[exp_photons, exp_electrons])
        self.assertGreater(float(p_val), 1e-3)


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
        src = Source.from_point(
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

        decay_src = Source.from_point(
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
        const_src = Source.from_point(
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

        src_fast = Source.from_point(activity=act_fast, energy=e_fast, half_life=hl_fast)
        src_slow = Source.from_point(activity=act_slow, energy=e_slow, half_life=hl_slow)
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
        src = Source.from_point(
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
        stop_limit = Float(500.0)
        injected_cnt2, _, new_t_last2, new_pending2 = inject_poisson_sources_kernel(
            bank,
            sbuf,
            stop_limit,
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
        src1 = Source.from_voxels(
            distribution=dist,
            voxel_size=Float(2.0 * units.mm),
            activity=Float(6000.0 * units.Bq),
            energy=[(0.1405, 0.8), (0.511, 0.2)],
            half_life=Float(1.5 * units.s),
        )
        src2 = Source.from_point(
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
        src1 = Source.from_point(
            activity=Float(5000.0 * units.Bq),
            energy=Float(140.5 * units.keV),
            half_life=Float(1.2 * units.s),
        )
        src2 = Source.from_point(
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
            Source.from_point(
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

    def test_inject_secondary_sources_kernel_tail_shift_and_poisson_delay(self) -> None:
        """
        Проверка Фазы 3 (inject_secondary_sources_kernel и SecondarySource):
        1. Пуассоновский розыгрыш времени испускания t_emit = t_act - ln(1 - u) / lambda_sec > t_act
           и отсечка частиц с t_emit >= stop_time.
        2. Сдвиг необработанного хвоста в начало SecondaryActivationBuffer при нехватке
           свободных слотов в ParticleBank и последующая полная разрядка.
        3. Сохранение parent_ID и выдача уникальных particle_ID из bank.count_array[0].
        """
        relax_bank = PhysicsCompiler()._build_atomic_relaxation_bank(
            [PhotoelectricEffect()],
            unique_atomic_numbers=(82,),
        )
        act_buf = SecondaryActivationBuffer.allocate(capacity=10, enable_atomic_relaxation=True)

        # Записываем 5 точек активации (K-оболочка свинца Z=82, local_shell=0)
        for idx in range(5):
            act_buf.position.x[idx] = Float(idx + 1.0)
            act_buf.position.y[idx] = Float(idx + 2.0)
            act_buf.position.z[idx] = Float(idx + 3.0)
            act_buf.direction.x[idx] = Float(0.0)
            act_buf.direction.y[idx] = Float(0.0)
            act_buf.direction.z[idx] = Float(1.0)
            act_buf.activation_time[idx] = Time(10.0 + idx)
            act_buf.energy[idx] = Energy(0.075 * units.MeV)
            act_buf.parent_ID[idx] = ID(100 + idx)
            act_buf.species[idx] = Species(ParticleSpecies.PHOTON)
            act_buf.target_z[idx] = Charge(82)
            act_buf.shell_idx[idx] = Index(0)
            act_buf.process_id[idx] = ProcessID(0)
        act_buf.cursor[0] = Index(5)

        sec_source = SecondarySource(act_buf, relax_bank, stop_time=Float(1000.0 * units.ns))
        self.assertTrue(sec_source.has_pending)
        self.assertEqual(act_buf.cursor_value, 5)

        # Банк емкостью всего 3 слота -> за первый вызов войдут только 3 частицы, а 2 сдвинутся в хвост
        bank = ParticleBank.allocate(3)
        bank.count_array[0] = Index(500)
        rng = np.random.default_rng(12345)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        injected_1, next_slot_1 = sec_source.replenish_bank(
            bank,
            rng_ctx=rng_ctx,
        )
        self.assertEqual(int(injected_1), 3)
        self.assertEqual(int(next_slot_1), 3)
        self.assertEqual(int(bank.count_array[0]), 503)
        self.assertEqual(act_buf.cursor_value, 2)
        np.testing.assert_array_equal(act_buf.parent_ID[:2], np.array([103, 104], dtype=ID))
        np.testing.assert_array_equal(bank.initial_state.ID[:3], np.array([500, 501, 502], dtype=ID))
        np.testing.assert_array_equal(bank.initial_state.parent_ID[:3], np.array([100, 101, 102], dtype=ID))
        self.assertTrue(np.all(bank.initial_state.emission_time[:3] > np.array([10.0, 11.0, 12.0])))

        # Освобождаем банк и разряжаем оставшиеся 2 записи
        bank.state.is_active[:] = False
        injected_2, _ = sec_source.replenish_bank(
            bank,
            rng_ctx=rng_ctx,
        )
        self.assertEqual(int(injected_2), 2)
        self.assertEqual(int(bank.count_array[0]), 505)
        self.assertFalse(sec_source.has_pending)
        np.testing.assert_array_equal(bank.initial_state.parent_ID[:2], np.array([103, 104], dtype=ID))

    def test_particle_source_unified_interface_and_slot_competition(self) -> None:
        """
        Проверка единого базового интерфейса ParticleSource для PrimarySource и SecondarySource:
        1. Полиморфное наследование от ParticleSource и передача эстафетного курсора slot_cursor;
        2. Квотированная конкуренция за слоты ParticleBank через параметр max_slots;
        3. Сохранение и восстановление состояния (export_state / restore_state) с автоматическим
           перевыделением SecondaryActivationBuffer большей емкости без resize().
        """
        self.assertTrue(issubclass(PrimarySource, ParticleSource))
        self.assertTrue(issubclass(SecondarySource, ParticleSource))

        relax_bank = PhysicsCompiler()._build_atomic_relaxation_bank(
            [PhotoelectricEffect()],
            unique_atomic_numbers=(82,),
        )
        act_buf = SecondaryActivationBuffer.allocate(capacity=8, enable_atomic_relaxation=True)
        for idx in range(5):
            act_buf.position.x[idx] = Float(idx)
            act_buf.position.y[idx] = Float(0.0)
            act_buf.position.z[idx] = Float(0.0)
            act_buf.direction.x[idx] = Float(0.0)
            act_buf.direction.y[idx] = Float(0.0)
            act_buf.direction.z[idx] = Float(1.0)
            act_buf.activation_time[idx] = Time(1.0)
            act_buf.energy[idx] = Energy(0.075 * units.MeV)
            act_buf.parent_ID[idx] = ID(200 + idx)
            act_buf.species[idx] = Species(ParticleSpecies.PHOTON)
            act_buf.target_z[idx] = Charge(82)
            act_buf.shell_idx[idx] = Index(0)
            act_buf.process_id[idx] = ProcessID(0)
        act_buf.cursor[0] = Index(5)

        src_node = Source.from_point(
            activity=Float(1.0e6 * units.Bq),
            energy=Float(140.5 * units.keV),
            half_life=Float(np.inf),
        )
        src_buf = SourceCompiler.compile([src_node])
        prim_source = PrimarySource(
            source_buffer=src_buf,
            start_time=Float(0.0),
            stop_time=Float(10.0 * units.s),
            max_particles=4,
        )
        sec_source = SecondarySource(
            activation_buffer=act_buf,
            relaxation_bank=relax_bank,
            stop_time=Float(10.0 * units.s),
        )
        sources: tuple[ParticleSource, ...] = (sec_source, prim_source)

        bank = ParticleBank.allocate(6)
        rng = np.random.default_rng(777)
        rng_ctx = RNGContext.from_numpy_rng(rng)

        # Ограничиваем каждый источник квотой max_slots = 2 и 3 соответственно
        inj_sec, cursor_after_sec = sources[0].replenish_bank(
            bank,
            rng_ctx,
            start_slot_cursor=Index(0),
            max_slots=Index(2),
        )
        inj_prim, cursor_after_prim = sources[1].replenish_bank(
            bank,
            rng_ctx,
            start_slot_cursor=cursor_after_sec,
            max_slots=Index(3),
        )
        self.assertEqual(int(inj_sec), 2)
        self.assertEqual(int(cursor_after_sec), 2)
        self.assertEqual(int(inj_prim), 3)
        self.assertEqual(int(cursor_after_prim), 5)
        self.assertEqual(sec_source.activation_buffer.cursor_value, 3)
        self.assertEqual(prim_source.emitted_particles, 3)
        self.assertTrue(sec_source.has_pending)
        self.assertTrue(prim_source.has_pending)
        np.testing.assert_array_equal(bank.initial_state.parent_ID[:2], np.array([200, 201], dtype=ID))
        np.testing.assert_array_equal(bank.initial_state.parent_ID[2:5], np.array([-1, -1, -1], dtype=ID))
        self.assertFalse(bool(bank.state.is_active[5]))

        # Проверяем export_state / restore_state на обоих источниках (включая расширение емкости буфера)
        sec_state = sec_source.export_state()
        prim_state = prim_source.export_state()
        self.assertIsNotNone(sec_state)
        self.assertIsNotNone(prim_state)

        small_sec = SecondarySource(
            activation_buffer=SecondaryActivationBuffer.allocate(capacity=1, enable_atomic_relaxation=True),
            relaxation_bank=relax_bank,
            stop_time=Float(10.0 * units.s),
        )
        small_sec.restore_state(sec_state)
        self.assertGreaterEqual(small_sec.activation_buffer.capacity, 3)
        self.assertEqual(small_sec.activation_buffer.cursor_value, 3)
        np.testing.assert_array_equal(small_sec.activation_buffer.parent_ID[:3], np.array([202, 203, 204], dtype=ID))

        restored_prim = PrimarySource(
            source_buffer=src_buf,
            start_time=Float(0.0),
            stop_time=Float(10.0 * units.s),
            max_particles=4,
        )
        restored_prim.restore_state(prim_state)
        self.assertEqual(restored_prim.emitted_particles, 3)
        self.assertAlmostEqual(float(restored_prim.t_last), float(prim_source.t_last), places=12)

    def test_two_phase_replenish_bank_and_emitted_primaries_quota(self) -> None:
        """
        Проверка Фазы 3 в SimulationManager:
        1. Двухфазное пополнение банка частиц (Фаза 1: SecondarySource, Фаза 2: первичные источники).
        2. Точное соблюдение квоты первичных историй emitted_primaries == particles_number
           независимо от числа рожденных вторичных флуоресцентных фотонов.
        3. Круговой цикл конфигурации (Roundtrip) для новых полей SimulationManagerConfig.
        """
        world = Volume(
            geometry=Box(200.0 * units.mm, 200.0 * units.mm, 200.0 * units.mm),
            material=database_setting.material_database["Pb"],
            name="World",
        )
        src = Source.from_point(
            activity=Float(50_000.0 * units.Bq),
            energy=Float(140.0 * units.keV),
            half_life=Float(np.inf),
            name="Tc99m_Point",
        )
        world.add_child(src)

        target_primaries = 80
        mgr = SimulationManager(
            scene=world,
            propagator=ParticlePropagator(processes_list=[PhotoelectricEffect, ComptonScattering]),
            start_time=Float(0.0),
            stop_time=None,
            particles_number=target_primaries,
            batch_capacity=16,
            buffer_capacity=1000,
            min_energy=Float(1.0 * units.keV),
            seed=2026,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(1.0 * units.keV),
            electron_production_cut=Float(np.inf),
        )
        mgr.start()
        mgr.join()

        init_list: list[dict[str, NDArray]] = []
        while True:
            item = mgr.queue.get_nowait()
            if item is None or item == "stop":
                break
            if item["type"].value == "initial_states":
                init_list.append(item["data"])
        init_chunks = {
            key: np.concatenate([chunk[key] for chunk in init_list])
            for key in init_list[0]
        }
        self.assertEqual(mgr.emitted_primaries, target_primaries)
        total_histories = int(mgr.bank.count_array[0])
        self.assertGreater(total_histories, target_primaries + 20)

        parent_ids = init_chunks["parent_ID"]
        primary_mask = parent_ids == -1
        secondary_mask = parent_ids >= 0
        self.assertEqual(int(np.sum(primary_mask)), target_primaries)
        self.assertEqual(int(np.sum(secondary_mask)), total_histories - target_primaries)
        self.assertEqual(len(np.unique(init_chunks["particle_ID"])), total_histories)

        # Проверка Roundtrip: SimulationManager -> SimulationManagerConfig -> SimulationManager
        exported_cfg = SceneExporter.export_simulation_manager(mgr, seed=2026)
        self.assertTrue(exported_cfg.enable_atomic_relaxation)
        self.assertAlmostEqual(float(exported_cfg.photon_production_cut), float(1.0 * units.keV), places=12)
        self.assertTrue(math.isinf(float(exported_cfg.electron_production_cut)))

        cfg_from_str = SimulationManagerConfig(
            enable_atomic_relaxation=True,
            photon_production_cut="2.5 keV",
            electron_production_cut="inf",
        )
        self.assertAlmostEqual(float(cfg_from_str.photon_production_cut), float(2.5 * units.keV), places=12)
        self.assertTrue(math.isinf(float(cfg_from_str.electron_production_cut)))

        rebuilt_mgr = SceneBuilder().build_simulation_manager(exported_cfg, scene=world)
        self.assertEqual(rebuilt_mgr.enable_atomic_relaxation, mgr.enable_atomic_relaxation)
        self.assertAlmostEqual(float(rebuilt_mgr.photon_production_cut), float(mgr.photon_production_cut), places=12)
        self.assertEqual(math.isinf(float(rebuilt_mgr.electron_production_cut)), True)


if __name__ == "__main__":
    unittest.main()

