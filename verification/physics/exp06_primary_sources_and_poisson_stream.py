"""Эксперимент 6: Комплексная верификация первичных источников, таблиц Воза и потока Льюиса–Шедлера.

Выполняет 5 уровней физико-математической и статистической проверки подсистемы `core/source/`
и конвейера пополнения `SimulationManager`:
1. Точность построения таблиц алиасов Воза (`build_alias_tables`) на распределениях с отношением
   весов 10^4 : 1 (алгебраическое восстановление вероятностей до 1e-14) и выборка дискретного
   многолинейчатого спектра радионуклида Ba-133 (`DiscreteSpectrum`).
2. Аналитическое обращение кусочно-линейных трапеций `ContinuousSpectrum` в `@njit`-ядре
   `_sample_channel_energy` (сверка с точной плотностью по критериям chi2/ndf и L1-норме).
3. Пространственно-угловая генерация (`VoxelDistribution` + `ConeAngular`) с нетривиальным
   3D-поворотом и сдвигом `local_to_global`: независимое аналитическое построение матрицы Эйлера,
   проверка нулевой утечки в пустые воксели, покомпонентной равномерности и независимости
   внутривоксельного джиттера по осям X, Y, Z и распределения cos(theta).
4. Нестационарный двухкомпонентный пуассоновский поток Льюиса–Шедлера (`inject_poisson_sources_kernel`)
   на интервале [t_start, t_end]: сходимость среднего числа частиц к интегралу интенсивности,
   масштабируемый статистический тест фактора Фано Var(N)/<N>, форма кривой скорости счета dN/dt,
   строгая монотонность времен рождения (Delta t >= 0), экспоненциальность и независимость
   межприходовых интервалов Delta t в стационарном пределе.
5. 100% побитовая инвариантность генерации первичного потока и полного связанного транспорта
   к размеру батча `batch_capacity` и двухэтапному досчету `Stateful Resume` ([0, T/2] + [T/2, T]).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
from numba import njit
import numpy as np
from numpy.typing import NDArray
import hepunits as units

from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.volumes import Volume
from core.other.typing_definitions import Float, ID, Index, ParticleSpecies
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import ChunkType, RNGContext, SimulationDataBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.processes import CoherentScattering, ComptonScattering, PhotoelectricEffect
from core.scene.nodes import CompositeNode
from core.source.alias_tables import build_alias_tables
from core.source.primary_source import PrimarySource
from core.source.source_buffer import SourceBuffer
from core.source.source_compiler import SourceCompiler
from core.source.source_kernels import (
    _sample_channel_energy,
    _sample_channel_position,
    _sample_cone_direction,
    inject_poisson_sources_kernel,
)
from core.source.sources import (
    ConeAngular,
    ContinuousSpectrum,
    DiscreteSpectrum,
    RadiationChannel,
    Source,
    SpectralLine,
    VoxelDistribution,
)
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
import settings.database_setting as database_setting
from verification.common.transport_and_telemetry import (
    RecordingTelemetryHandler,
    audit_manager_telemetry_energy_balance,
    run_coupled_transport_loop,
)
from verification.physics.plotting import plot_exp06_primary_sources_and_poisson_stream


@dataclass(frozen=True)
class VoseAliasAndDiscreteMetrics:
    """Метрики верификации таблиц метода Воза и дискретного спектра `DiscreteSpectrum`.

    Attributes:
        num_samples: Число сгенерированных фотонов для каждого спектра.
        skewed_weight_ratio: Отношение максимального и минимального весов в скошенном спектре.
        max_algebraic_reconstruction_error: Максимальная погрешность алгебраического восстановления весов Воза.
        skewed_reduced_chi2: Приведенный хи-квадрат выборки для спектра с отношением весов 10^4 : 1.
        skewed_max_abs_z_score: Максимальное абсолютное Z-отклонение по линиям скошенного спектра.
        radionuclide_name: Название проверенного многолинейчатого радионуклида.
        radionuclide_reduced_chi2: Приведенный хи-квадрат выборки многолинейчатого спектра Ba-133.
        radionuclide_max_abs_z_score: Максимальное абсолютное Z-отклонение по линиям Ba-133.
        passed: Признак успешного прохождения допусков.
    """

    num_samples: int
    skewed_weight_ratio: float
    max_algebraic_reconstruction_error: float
    skewed_reduced_chi2: float
    skewed_max_abs_z_score: float
    radionuclide_name: str
    radionuclide_reduced_chi2: float
    radionuclide_max_abs_z_score: float
    passed: bool


@dataclass(frozen=True)
class ContinuousTrapezoidMetrics:
    """Метрики верификации аналитического обращения трапеций `ContinuousSpectrum`.

    Attributes:
        num_samples: Число разыгранных значений энергии.
        num_histogram_bins: Число бинов проверочной гистограммы плотности.
        reduced_chi2: Приведенный хи-квадрат согласия с аналитической кусочно-линейной плотностью.
        l1_density_error: Интегральная L1-ошибка плотности вероятности.
        max_abs_z_score: Максимальное абсолютное Z-отклонение по энергетическим бинам.
        empirical_mean_energy: Средняя энергия по выборке Монте-Карло [МэВ].
        theoretical_mean_energy: Точное аналитическое математическое ожидание энергии [МэВ].
        mean_energy_z_score: Абсолютное Z-отклонение выборочной средней энергии от аналитического среднего.
        passed: Признак успешного прохождения допусков.
    """

    num_samples: int
    num_histogram_bins: int
    reduced_chi2: float
    l1_density_error: float
    max_abs_z_score: float
    empirical_mean_energy: float
    theoretical_mean_energy: float
    mean_energy_z_score: float
    passed: bool


@dataclass(frozen=True)
class SpatialAngularTransformMetrics:
    """Метрики верификации 3D-воксельного распределения, конуса направлений и матрицы `local_to_global`.

    Attributes:
        num_samples: Число сгенерированных частиц.
        active_voxels_count: Число вокселей с положительным весом активности.
        zero_weight_voxels_count: Число пустых вокселей с нулевым весом.
        zero_weight_voxel_leakage: Число частиц, попавших в воксели с нулевым весом.
        out_of_bounds_count: Число частиц, вышедших за пределы локальной сетки источника.
        voxel_weights_reduced_chi2: Приведенный хи-квадрат распределения частиц по активным вокселям.
        intra_voxel_jitter_reduced_chi2: Сводный приведенный хи-квадрат равномерности смещения внутри вокселя.
        intra_voxel_jitter_chi2_xyz: Покомпонентные приведенные хи-квадрат равномерности джиттера по осям X, Y, Z.
        max_abs_inter_axis_jitter_corr: Максимальная абсолютная попарная корреляция джиттера между осями.
        max_independent_transform_diff: Максимальное расхождение между независимой матрицей Эйлера и СК узла.
        cone_violation_count: Число частиц с локальным полярным углом вне заданного раствора конуса.
        max_direction_norm_error: Максимальное отклонение нормы вектора направления от 1.0.
        cos_theta_reduced_chi2: Приведенный хи-квадрат равномерности cos(theta) на [cos(theta_max), 1].
        azimuth_phi_reduced_chi2: Приведенный хи-квадрат равномерности азимутального угла phi.
        passed: Признак успешного прохождения допусков.
    """

    num_samples: int
    active_voxels_count: int
    zero_weight_voxels_count: int
    zero_weight_voxel_leakage: int
    out_of_bounds_count: int
    voxel_weights_reduced_chi2: float
    intra_voxel_jitter_reduced_chi2: float
    intra_voxel_jitter_chi2_xyz: list[float]
    max_abs_inter_axis_jitter_corr: float
    max_independent_transform_diff: float
    cone_violation_count: int
    max_direction_norm_error: float
    cos_theta_reduced_chi2: float
    azimuth_phi_reduced_chi2: float
    passed: bool


@dataclass(frozen=True)
class NonStationaryPoissonStreamMetrics:
    """Метрики верификации нестационарного пуассоновского потока Льюиса–Шедлера.

    Attributes:
        num_realizations: Число независимых реализаций нестационарного потока.
        theoretical_expected_count: Точный интеграл ожидаемого числа рожденных частиц Lambda.
        empirical_mean_count: Выборочное среднее число рожденных частиц <N>.
        mean_count_z_score: Z-отклонение выборочного среднего <N> от точного интеграла Lambda.
        fano_dispersion_index: Пуассоновский фактор Фано Var(N) / <N>.
        fano_standard_error: Теоретическая стандартная ошибка фактора Фано sqrt(2 / (M - 1)).
        fano_z_score: Абсолютное Z-отклонение фактора Фано от теоретической единицы.
        rate_curve_reduced_chi2: Приведенный хи-квадрат временного профиля интенсивности dN/dt.
        component_ratio_empirical: Отношение числа частиц короткоживущего и долгоживущего каналов (MC).
        component_ratio_theoretical: Теоретическое отношение интегралов каналов Lambda_1 / Lambda_2.
        component_ratio_z_score: Биномиальное Z-отклонение доли первой компоненты распада.
        non_monotonic_time_violations: Число нарушений монотонности времени рождения (Delta t < 0).
        stationary_interarrival_count: Число проверенных межприходовых интервалов Delta t в стационарном тесте.
        interarrival_exp_reduced_chi2: Приведенный хи-квадрат согласия Delta t с экспоненциальным законом.
        interarrival_autocorr_lag1: Коэффициент автокорреляции 1-го порядка между соседними Delta t.
        interarrival_autocorr_z_score: Абсолютное Z-отклонение автокорреляции Delta t от нуля.
        passed: Признак успешного прохождения допусков.
    """

    num_realizations: int
    theoretical_expected_count: float
    empirical_mean_count: float
    mean_count_z_score: float
    fano_dispersion_index: float
    fano_standard_error: float
    fano_z_score: float
    rate_curve_reduced_chi2: float
    component_ratio_empirical: float
    component_ratio_theoretical: float
    component_ratio_z_score: float
    non_monotonic_time_violations: int
    stationary_interarrival_count: int
    interarrival_exp_reduced_chi2: float
    interarrival_autocorr_lag1: float
    interarrival_autocorr_z_score: float
    passed: bool


@dataclass(frozen=True)
class BatchAndResumeInvarianceCase:
    """Метрики побитового сравнения для одного сценария `batch_capacity` или `Stateful Resume`.

    Attributes:
        scenario_label: Краткое описание сценария сравнения.
        total_particles: Число проверенных частиц в потоке.
        id_mismatch_count: Число несовпавших идентификаторов частиц.
        max_position_diff: Максимальная абсолютная разность мировых координат рождения [мм].
        max_direction_diff: Максимальная абсолютная разность компонент вектора направления.
        max_energy_diff: Максимальная абсолютная разность энергии эмиссии [МэВ].
        max_time_diff: Максимальная абсолютная разность времени эмиссии [нс].
        passed: Признак 100% побитового совпадения (все невязки строго равны 0.0).
    """

    scenario_label: str
    total_particles: int
    id_mismatch_count: int
    max_position_diff: float
    max_direction_diff: float
    max_energy_diff: float
    max_time_diff: float
    passed: bool


@dataclass(frozen=True)
class Exp06PrimarySourcesReport:
    """Сводный отчет Эксперимента 6 по верификации первичных источников и пуассоновского потока.

    Attributes:
        quick: Признак запуска в экспресс-режиме.
        reference_standard_types: Классификация типов используемых эталонов по каждой проверке.
        vose_and_discrete: Метрики таблиц Воза и дискретных спектров.
        continuous_spectrum: Метрики аналитического обращения трапеций непрерывного спектра.
        spatial_and_angular: Метрики 3D-воксельной и конусной генерации с поворотом и сдвигом.
        poisson_stream: Метрики нестационарного пуассоновского потока Льюиса–Шедлера.
        invariance_cases: Результаты побитовой сверки batch_capacity и Stateful Resume.
        passed: Общий статус успешного прохождения всех проверок эксперимента.
    """

    quick: bool
    reference_standard_types: dict[str, str]
    vose_and_discrete: VoseAliasAndDiscreteMetrics
    continuous_spectrum: ContinuousTrapezoidMetrics
    spatial_and_angular: SpatialAngularTransformMetrics
    poisson_stream: NonStationaryPoissonStreamMetrics
    invariance_cases: list[BatchAndResumeInvarianceCase]
    passed: bool


@njit(cache=True)
def _sample_energies_batch_kernel(
    source_buffer: SourceBuffer,
    channel_idx: Index,
    num_samples: int,
    rng_ctx: RNGContext,
) -> NDArray[np.float64]:
    """Генерирует массив энергий из заданного канала SourceBuffer ядром `_sample_channel_energy`."""
    energies = np.empty(num_samples, dtype=np.float64)
    for sample_idx in range(num_samples):
        energies[sample_idx] = float(
            _sample_channel_energy(source_buffer, channel_idx, rng_ctx)
        )
    return energies


@njit(cache=True)
def _sample_spatial_and_angular_batch_kernel(
    source_buffer: SourceBuffer,
    channel_idx: Index,
    num_samples: int,
    rng_ctx: RNGContext,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Генерирует мировые координаты и направления вылета для канала SourceBuffer."""
    positions = np.empty((num_samples, 3), dtype=np.float64)
    directions = np.empty((num_samples, 3), dtype=np.float64)
    cos_min = source_buffer.cos_theta_min[channel_idx]
    local_to_global = source_buffer.local_to_global

    for sample_idx in range(num_samples):
        pos_x, pos_y, pos_z = _sample_channel_position(
            source_buffer, channel_idx, rng_ctx
        )
        dir_x, dir_y, dir_z = _sample_cone_direction(
            cos_min, local_to_global, channel_idx, rng_ctx
        )
        positions[sample_idx, 0] = float(pos_x)
        positions[sample_idx, 1] = float(pos_y)
        positions[sample_idx, 2] = float(pos_z)
        directions[sample_idx, 0] = float(dir_x)
        directions[sample_idx, 1] = float(dir_y)
        directions[sample_idx, 2] = float(dir_z)

    return positions, directions


@njit(cache=True)
def _run_poisson_ensemble_kernel(
    bank: ParticleBank,
    source_buffer: SourceBuffer,
    start_time: Float,
    stop_time: Float,
    num_realizations: int,
    num_time_bins: int,
    first_component_energy: Float,
    rng_ctx: RNGContext,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int, int, int]:
    """Выполняет серию независимых реализаций нестационарного потока Льюиса–Шедлера."""
    counts_per_run = np.zeros(num_realizations, dtype=np.int64)
    time_hist = np.zeros(num_time_bins, dtype=np.int64)
    first_comp_total = 0
    second_comp_total = 0
    non_monotonic_violations = 0

    time_span = float(stop_time - start_time)
    inv_bin_width = float(num_time_bins) / time_span

    is_active = bank.state.is_active
    emission_times = bank.initial_state.emission_time
    emission_energies = bank.initial_state.emission_energy

    for run_idx in range(num_realizations):
        is_active[:] = False
        bank.count_array[0] = Index(0)

        injected, _, _, _ = inject_poisson_sources_kernel(
            bank=bank,
            source_buffer=source_buffer,
            stop_time=stop_time,
            t_ref=start_time,
            t_last=start_time,
            pending_arrival_time=Float(np.nan),
            rng_ctx=rng_ctx,
            max_particles=Index(-1),
            start_slot_cursor=Index(0),
        )
        num_injected = int(injected)
        counts_per_run[run_idx] = num_injected

        prev_emit_t = float(start_time)
        for slot_idx in range(num_injected):
            emit_t = float(emission_times[slot_idx])
            if emit_t < prev_emit_t or emit_t > float(stop_time):
                non_monotonic_violations += 1
            prev_emit_t = emit_t

            bin_idx = int((emit_t - float(start_time)) * inv_bin_width)
            if 0 <= bin_idx < num_time_bins:
                time_hist[bin_idx] += 1

            emit_e = float(emission_energies[slot_idx])
            if abs(emit_e - float(first_component_energy)) < 1e-6:
                first_comp_total += 1
            else:
                second_comp_total += 1

    return (
        counts_per_run,
        time_hist,
        first_comp_total,
        second_comp_total,
        non_monotonic_violations,
    )


def _reconstruct_vose_probabilities(
    prob_table: NDArray[Float],
    alias_table: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Алгебраически восстанавливает точные вероятности выбора элементов из таблиц Воза."""
    num_items = int(prob_table.shape[0])
    reconstructed = np.array(prob_table, dtype=np.float64)
    for item_idx in range(num_items):
        alias_target = int(alias_table[item_idx])
        reconstructed[alias_target] += 1.0 - float(prob_table[item_idx])
    return reconstructed / float(num_items)


def _evaluate_discrete_sampling(
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    num_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Проверяет точность таблицы Воза и статистику выборки для заданного дискретного спектра."""
    norm_probs = weights / float(np.sum(weights))
    prob_table, alias_table = build_alias_tables(norm_probs)
    reconstructed = _reconstruct_vose_probabilities(prob_table, alias_table)
    max_alg_err = float(np.max(np.abs(reconstructed - norm_probs)))

    lines = tuple(
        SpectralLine(energy=Float(energy_val), probability=Float(weight_val))
        for energy_val, weight_val in zip(energies, weights)
    )
    spectrum = DiscreteSpectrum(lines=lines)
    source = Source.from_point(
        activity=Float(1000.0 * units.Bq),
        energy=spectrum,
        half_life=Float(10.0 * units.s),
        name="DiscreteAuditSource",
    )
    source_buffer = SourceCompiler.compile([source])

    rng = np.random.default_rng(seed)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    sampled_energies = _sample_energies_batch_kernel(
        source_buffer=source_buffer,
        channel_idx=Index(0),
        num_samples=num_samples,
        rng_ctx=rng_ctx,
    )

    diffs = np.abs(sampled_energies[:, None] - energies[None, :])
    chosen_indices = np.argmin(diffs, axis=1)
    empirical_counts = np.bincount(chosen_indices, minlength=energies.size).astype(np.float64)
    expected_counts = float(num_samples) * norm_probs

    z_scores = (empirical_counts - expected_counts) / np.sqrt(
        np.maximum(expected_counts * (1.0 - norm_probs), 1e-12)
    )
    chi2_val = float(np.sum((empirical_counts - expected_counts) ** 2 / expected_counts))
    reduced_chi2 = chi2_val / float(energies.size - 1)
    max_abs_z = float(np.max(np.abs(z_scores)))
    return max_alg_err, reduced_chi2, max_abs_z


def _run_vose_and_discrete_audit(num_samples: int) -> VoseAliasAndDiscreteMetrics:
    """Выполняет Проверку 1: аудит таблиц Воза (10^4 : 1) и многолинейчатого спектра Ba-133."""
    skewed_energies = np.array(
        [0.030, 0.055, 0.085, 0.125, 0.175, 0.235, 0.310, 0.395, 0.490, 0.600],
        dtype=np.float64,
    ) * float(units.MeV)
    skewed_weights = np.array(
        [10000.0, 2800.0, 950.0, 400.0, 180.0, 85.0, 40.0, 18.0, 6.0, 1.0],
        dtype=np.float64,
    )
    weight_ratio = float(np.max(skewed_weights) / np.min(skewed_weights))

    alg_err_skewed, chi2_skewed, max_z_skewed = _evaluate_discrete_sampling(
        energies=skewed_energies,
        weights=skewed_weights,
        num_samples=num_samples,
        seed=20260607,
    )

    ba133_energies = np.array(
        [0.05316, 0.07961, 0.08100, 0.16061, 0.22324, 0.27640, 0.30285, 0.35601, 0.38385],
        dtype=np.float64,
    ) * float(units.MeV)
    ba133_weights = np.array(
        [0.0214, 0.0265, 0.3290, 0.00638, 0.00453, 0.0716, 0.1834, 0.6205, 0.0894],
        dtype=np.float64,
    )

    alg_err_ba133, chi2_ba133, max_z_ba133 = _evaluate_discrete_sampling(
        energies=ba133_energies,
        weights=ba133_weights,
        num_samples=num_samples,
        seed=20260602,
    )

    max_alg_err = max(alg_err_skewed, alg_err_ba133)
    passed = (
        max_alg_err < 1e-14
        and (0.5 <= chi2_skewed <= 1.6)
        and (max_z_skewed < 3.5)
        and (0.5 <= chi2_ba133 <= 1.6)
        and (max_z_ba133 < 3.5)
    )

    return VoseAliasAndDiscreteMetrics(
        num_samples=num_samples,
        skewed_weight_ratio=weight_ratio,
        max_algebraic_reconstruction_error=max_alg_err,
        skewed_reduced_chi2=chi2_skewed,
        skewed_max_abs_z_score=max_z_skewed,
        radionuclide_name="Ba-133",
        radionuclide_reduced_chi2=chi2_ba133,
        radionuclide_max_abs_z_score=max_z_ba133,
        passed=passed,
    )


def _evaluate_trapezoid_cdf(
    query_points: NDArray[np.float64],
    grid_nodes: NDArray[np.float64],
    pdf_nodes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Вычисляет точную аналитическую первообразную F(E) = int_{E_0}^{E} p_norm(u) du кусочно-линейного спектра."""
    seg_low = grid_nodes[:-1]
    seg_high = grid_nodes[1:]
    pdf_left = pdf_nodes[:-1]
    pdf_right = pdf_nodes[1:]
    seg_widths = seg_high - seg_low
    seg_areas = 0.5 * (pdf_left + pdf_right) * seg_widths
    total_area = float(np.sum(seg_areas))

    cum_areas = np.zeros(grid_nodes.size, dtype=np.float64)
    cum_areas[1:] = np.cumsum(seg_areas) / total_area

    norm_left = pdf_left / total_area
    norm_right = pdf_right / total_area
    slopes = (norm_right - norm_left) / seg_widths

    cdf_values = np.empty(query_points.size, dtype=np.float64)
    for pt_idx in range(query_points.size):
        energy_val = float(query_points[pt_idx])
        if energy_val <= float(grid_nodes[0]):
            cdf_values[pt_idx] = 0.0
        elif energy_val >= float(grid_nodes[-1]):
            cdf_values[pt_idx] = 1.0
        else:
            seg_idx = int(np.searchsorted(grid_nodes, energy_val, side="right")) - 1
            seg_idx = max(0, min(seg_idx, seg_low.size - 1))
            delta_energy = energy_val - float(seg_low[seg_idx])
            cdf_values[pt_idx] = float(cum_areas[seg_idx]) + delta_energy * (
                float(norm_left[seg_idx]) + 0.5 * float(slopes[seg_idx]) * delta_energy
            )
    return cdf_values


def _run_continuous_spectrum_audit(
    num_samples: int,
) -> tuple[
    ContinuousTrapezoidMetrics,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Выполняет Проверку 2: обращение кусочно-линейных трапеций `ContinuousSpectrum`."""
    energy_grid = np.array(
        [0.020, 0.084, 0.164, 0.260, 0.380, 0.500],
        dtype=Float,
    ) * Float(units.MeV)
    pdf_values = np.array(
        [0.30, 2.40, 2.40, 0.75, 1.85, 0.15],
        dtype=Float,
    )

    spectrum = ContinuousSpectrum(energy_grid=energy_grid, pdf_values=pdf_values)
    source = Source.from_point(
        activity=Float(1000.0 * units.Bq),
        energy=spectrum,
        half_life=Float(10.0 * units.s),
        name="ContinuousTrapezoidSource",
    )
    source_buffer = SourceCompiler.compile([source])

    rng = np.random.default_rng(20260603)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    sampled_energies = _sample_energies_batch_kernel(
        source_buffer=source_buffer,
        channel_idx=Index(0),
        num_samples=num_samples,
        rng_ctx=rng_ctx,
    )

    num_bins = 60
    bin_edges = np.linspace(
        float(energy_grid[0]), float(energy_grid[-1]), num_bins + 1, dtype=np.float64
    )
    bin_width = float(bin_edges[1] - bin_edges[0])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_counts, _ = np.histogram(sampled_energies, bins=bin_edges)
    counts_float = hist_counts.astype(np.float64)
    density_mc = counts_float / (float(num_samples) * bin_width)

    grid_f64 = np.asarray(energy_grid, dtype=np.float64)
    pdf_f64 = np.asarray(pdf_values, dtype=np.float64)
    seg_low = grid_f64[:-1]
    seg_high = grid_f64[1:]
    pdf_left = pdf_f64[:-1]
    pdf_right = pdf_f64[1:]
    seg_areas = 0.5 * (pdf_left + pdf_right) * (seg_high - seg_low)
    total_area = float(np.sum(seg_areas))

    cdf_edges = _evaluate_trapezoid_cdf(bin_edges, grid_f64, pdf_f64)
    bin_probabilities = np.diff(cdf_edges)
    density_theory = bin_probabilities / bin_width
    expected_counts = float(num_samples) * bin_probabilities

    chi2_val = float(np.sum((counts_float - expected_counts) ** 2 / expected_counts))
    reduced_chi2 = chi2_val / float(num_bins - 1)
    l1_error = float(np.sum(np.abs(density_mc - density_theory)) * bin_width)
    z_scores = (counts_float - expected_counts) / np.sqrt(np.maximum(expected_counts, 1e-12))
    max_abs_z = float(np.max(np.abs(z_scores)))

    first_moments = (
        (seg_high - seg_low)
        / 6.0
        * (
            pdf_left * (2.0 * seg_low + seg_high)
            + pdf_right * (seg_low + 2.0 * seg_high)
        )
    )
    second_moments = (
        (seg_high - seg_low)
        / 12.0
        * (
            pdf_left * (3.0 * seg_low**2 + 2.0 * seg_low * seg_high + seg_high**2)
            + pdf_right * (seg_low**2 + 2.0 * seg_low * seg_high + 3.0 * seg_high**2)
        )
    )
    theoretical_mean = float(np.sum(first_moments) / total_area)
    theoretical_second_moment = float(np.sum(second_moments) / total_area)
    theoretical_variance = max(
        theoretical_second_moment - theoretical_mean * theoretical_mean, 1e-16
    )
    standard_error_mean = math.sqrt(theoretical_variance / float(num_samples))
    empirical_mean = float(np.mean(sampled_energies))
    mean_energy_z_score = abs(empirical_mean - theoretical_mean) / standard_error_mean

    l1_tolerance = 4.0 * math.sqrt(float(num_bins) / float(num_samples))
    passed = (
        (0.5 <= reduced_chi2 <= 1.6)
        and (l1_error < l1_tolerance)
        and (max_abs_z < 3.5)
        and (mean_energy_z_score < 3.5)
    )

    metrics = ContinuousTrapezoidMetrics(
        num_samples=num_samples,
        num_histogram_bins=num_bins,
        reduced_chi2=reduced_chi2,
        l1_density_error=l1_error,
        max_abs_z_score=max_abs_z,
        empirical_mean_energy=empirical_mean,
        theoretical_mean_energy=theoretical_mean,
        mean_energy_z_score=mean_energy_z_score,
        passed=passed,
    )
    return metrics, bin_centers, density_mc, density_theory


def _build_independent_euler_rotation_matrix(
    alpha_angle: float,
    beta_angle: float,
    gamma_angle: float,
) -> NDArray[np.float64]:
    """Строит матрицу поворота R_z(alpha) @ R_y(beta) @ R_x(gamma) независимо из тригонометрических матриц."""
    cos_a, sin_a = math.cos(alpha_angle), math.sin(alpha_angle)
    cos_b, sin_b = math.cos(beta_angle), math.sin(beta_angle)
    cos_g, sin_g = math.cos(gamma_angle), math.sin(gamma_angle)

    rot_z = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [cos_b, 0.0, sin_b],
            [0.0, 1.0, 0.0],
            [-sin_b, 0.0, cos_b],
        ],
        dtype=np.float64,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_g, -sin_g],
            [0.0, sin_g, cos_g],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _run_spatial_and_angular_audit(num_samples: int) -> SpatialAngularTransformMetrics:
    """Выполняет Проверку 3: 3D-воксельный фантом и конус с независимой проверкой матрицы Эйлера."""
    grid_nx, grid_ny, grid_nz = 4, 4, 4
    voxel_step = Float(5.0 * units.mm)
    activity_grid = np.zeros((grid_nx, grid_ny, grid_nz), dtype=Float)

    active_coords_and_weights = [
        (0, 0, 0, 12.0),
        (0, 1, 2, 4.5),
        (0, 3, 1, 8.0),
        (0, 3, 3, 2.0),
        (1, 0, 2, 15.0),
        (1, 1, 1, 25.0),
        (1, 2, 0, 6.0),
        (1, 3, 2, 3.5),
        (2, 0, 1, 5.0),
        (2, 1, 3, 18.0),
        (2, 2, 2, 22.0),
        (2, 3, 0, 7.5),
        (3, 0, 3, 1.5),
        (3, 1, 0, 9.0),
        (3, 2, 3, 14.0),
        (3, 3, 3, 11.0),
    ]
    for idx_x, idx_y, idx_z, weight_val in active_coords_and_weights:
        activity_grid[idx_x, idx_y, idx_z] = Float(weight_val)

    cone_half_angle = Float(0.68 * units.rad)
    voxel_dist = VoxelDistribution(distribution=activity_grid, voxel_size=voxel_step)
    cone_ang = ConeAngular(max_polar_angle=cone_half_angle)
    channel = RadiationChannel(
        spectrum=DiscreteSpectrum.from_energies(Float(0.1405 * units.MeV)),
        angular=cone_ang,
        branching_ratio=Float(1.0),
    )
    source = Source(
        spatial=voxel_dist,
        channels=(channel,),
        half_life=Float(6.0 * units.hour),
        name="RotatedVoxelConeSource",
    )

    euler_alpha = 0.42 * float(units.rad)
    euler_beta = -0.31 * float(units.rad)
    euler_gamma = 0.73 * float(units.rad)
    shift_x = 35.0 * float(units.mm)
    shift_y = -22.5 * float(units.mm)
    shift_z = 48.0 * float(units.mm)

    source.rotate(
        alpha=Float(euler_alpha),
        beta=Float(euler_beta),
        gamma=Float(euler_gamma),
    )
    source.translate(
        x=Float(shift_x),
        y=Float(shift_y),
        z=Float(shift_z),
    )

    source_buffer = SourceCompiler.compile([source])
    rng = np.random.default_rng(20260618)
    rng_ctx = RNGContext.from_numpy_rng(rng)

    global_positions, global_directions = _sample_spatial_and_angular_batch_kernel(
        source_buffer=source_buffer,
        channel_idx=Index(0),
        num_samples=num_samples,
        rng_ctx=rng_ctx,
    )

    # Независимое аналитическое обратное преобразование r_local = R_Euler^T (r_world - t)
    euler_rot = _build_independent_euler_rotation_matrix(euler_alpha, euler_beta, euler_gamma)
    translation_vec = np.array([shift_x, shift_y, shift_z], dtype=np.float64)
    local_positions = (global_positions - translation_vec[None, :]) @ euler_rot
    local_directions = global_directions @ euler_rot

    # Сверка независимой аналитической матрицы с методами SpatialNode
    node_local_pos = np.asarray(
        source.convert_to_local_position(global_positions.astype(Float)),
        dtype=np.float64,
    )
    node_local_dir = np.asarray(
        source.convert_to_local_direction(global_directions.astype(Float)),
        dtype=np.float64,
    )
    max_transform_diff = float(
        max(
            np.max(np.abs(local_positions - node_local_pos)),
            np.max(np.abs(local_directions - node_local_dir)),
        )
    )

    half_extent = 0.5 * np.array([grid_nx, grid_ny, grid_nz], dtype=np.float64) * float(voxel_step)
    shifted_coords = (local_positions + half_extent[None, :]) / float(voxel_step)

    out_of_bounds_mask = np.any(
        (shifted_coords < -1e-12) | (shifted_coords > 4.0 + 1e-12), axis=1
    )
    out_of_bounds_count = int(np.sum(out_of_bounds_mask))

    clipped_coords = np.clip(shifted_coords, 0.0, 4.0 - 1e-12)
    voxel_indices_3d = np.floor(clipped_coords).astype(np.int64)
    fractional_jitter = clipped_coords - voxel_indices_3d.astype(np.float64)

    flat_indices = (
        voxel_indices_3d[:, 0] * (grid_ny * grid_nz)
        + voxel_indices_3d[:, 1] * grid_nz
        + voxel_indices_3d[:, 2]
    )
    flat_counts = np.bincount(flat_indices, minlength=grid_nx * grid_ny * grid_nz)

    flat_weights = activity_grid.ravel().astype(np.float64)
    zero_mask = flat_weights == 0.0
    active_mask = flat_weights > 0.0
    zero_weight_leakage = int(np.sum(flat_counts[zero_mask]))

    active_counts = flat_counts[active_mask].astype(np.float64)
    active_probs = flat_weights[active_mask] / float(np.sum(flat_weights[active_mask]))
    expected_active_counts = float(num_samples) * active_probs
    voxel_chi2 = float(
        np.sum((active_counts - expected_active_counts) ** 2 / expected_active_counts)
        / float(active_counts.size - 1)
    )

    # Покомпонентная проверка равномерности внутривоксельного джиттера по каждой оси X, Y, Z на [0, 1)
    num_jitter_bins = 20
    expected_axis_jitter = float(num_samples) / float(num_jitter_bins)
    jitter_chi2_xyz: list[float] = []
    for axis_idx in range(3):
        axis_hist, _ = np.histogram(
            fractional_jitter[:, axis_idx], bins=num_jitter_bins, range=(0.0, 1.0)
        )
        axis_chi2 = float(
            np.sum((axis_hist.astype(np.float64) - expected_axis_jitter) ** 2 / expected_axis_jitter)
            / float(num_jitter_bins - 1)
        )
        jitter_chi2_xyz.append(axis_chi2)

    jitter_hist, _ = np.histogram(
        fractional_jitter.ravel(), bins=num_jitter_bins, range=(0.0, 1.0)
    )
    expected_jitter = float(fractional_jitter.size) / float(num_jitter_bins)
    jitter_chi2 = float(
        np.sum((jitter_hist.astype(np.float64) - expected_jitter) ** 2 / expected_jitter)
        / float(num_jitter_bins - 1)
    )

    # Проверка статистической независимости джиттера между осями (попарные корреляции X-Y, Y-Z, X-Z)
    jitter_corr_matrix = np.corrcoef(fractional_jitter, rowvar=False)
    corr_xy = abs(float(jitter_corr_matrix[0, 1]))
    corr_yz = abs(float(jitter_corr_matrix[1, 2]))
    corr_xz = abs(float(jitter_corr_matrix[0, 2]))
    max_abs_jitter_corr = max(corr_xy, corr_yz, corr_xz)
    jitter_corr_limit = 3.5 / math.sqrt(float(num_samples))

    # Проверка направлений в локальной СК
    global_norms = np.linalg.norm(global_directions, axis=1)
    max_norm_error = float(np.max(np.abs(global_norms - 1.0)))

    cos_theta_min_val = float(cone_ang.cos_theta_min)
    local_cos_theta = local_directions[:, 2]
    cone_violations = int(
        np.sum((local_cos_theta < cos_theta_min_val - 1e-12) | (local_cos_theta > 1.0 + 1e-12))
    )

    num_angle_bins = 25
    cos_hist, _ = np.histogram(
        local_cos_theta, bins=num_angle_bins, range=(cos_theta_min_val, 1.0)
    )
    expected_angle_bin = float(num_samples) / float(num_angle_bins)
    cos_theta_chi2 = float(
        np.sum((cos_hist.astype(np.float64) - expected_angle_bin) ** 2 / expected_angle_bin)
        / float(num_angle_bins - 1)
    )

    local_phi = np.arctan2(local_directions[:, 1], local_directions[:, 0])
    phi_hist, _ = np.histogram(
        local_phi, bins=num_angle_bins, range=(-math.pi, math.pi)
    )
    phi_chi2 = float(
        np.sum((phi_hist.astype(np.float64) - expected_angle_bin) ** 2 / expected_angle_bin)
        / float(num_angle_bins - 1)
    )

    passed = (
        out_of_bounds_count == 0
        and zero_weight_leakage == 0
        and cone_violations == 0
        and max_norm_error < 1e-14
        and max_transform_diff < 1e-13
        and (0.5 <= voxel_chi2 <= 1.6)
        and (0.5 <= jitter_chi2 <= 1.6)
        and all(0.5 <= axis_val <= 1.6 for axis_val in jitter_chi2_xyz)
        and (max_abs_jitter_corr < jitter_corr_limit)
        and (0.5 <= cos_theta_chi2 <= 1.6)
        and (0.5 <= phi_chi2 <= 1.6)
    )

    return SpatialAngularTransformMetrics(
        num_samples=num_samples,
        active_voxels_count=int(np.sum(active_mask)),
        zero_weight_voxels_count=int(np.sum(zero_mask)),
        zero_weight_voxel_leakage=zero_weight_leakage,
        out_of_bounds_count=out_of_bounds_count,
        voxel_weights_reduced_chi2=voxel_chi2,
        intra_voxel_jitter_reduced_chi2=jitter_chi2,
        intra_voxel_jitter_chi2_xyz=jitter_chi2_xyz,
        max_abs_inter_axis_jitter_corr=max_abs_jitter_corr,
        max_independent_transform_diff=max_transform_diff,
        cone_violation_count=cone_violations,
        max_direction_norm_error=max_norm_error,
        cos_theta_reduced_chi2=cos_theta_chi2,
        azimuth_phi_reduced_chi2=phi_chi2,
        passed=passed,
    )


def _audit_stationary_poisson_interarrivals(
    target_events: int,
    seed: int,
) -> tuple[int, int, float, float, float]:
    """Проверяет экспоненциальность и независимость межприходовых интервалов Delta t в стационарном пределе."""
    stationary_rate = 10.0  # 10 событий / нс (10^10 Бк)
    stationary_source = Source.from_point(
        activity=Float(stationary_rate * 1e9 * units.Bq),
        energy=Float(0.1405 * units.MeV),
        half_life=Float(1.0e12 * units.s),
        name="StationaryPoissonSource",
    )
    stationary_buffer = SourceCompiler.compile([stationary_source])
    stop_time = Float(float(target_events) * 1.25 / stationary_rate * units.ns)

    records, _ = _collect_primary_stream_batches(
        source_buffer=stationary_buffer,
        start_time=Float(0.0),
        stop_time=stop_time,
        batch_capacity=4096,
        seed=seed,
    )
    emission_times = np.asarray(records["emission_time"], dtype=np.float64)
    interarrivals = np.diff(emission_times, prepend=0.0)
    non_monotonic = int(np.sum(interarrivals < 0.0))
    num_intervals = int(interarrivals.size)

    # Полный полиномиальный критерий хи-квадрат для экспоненциального распределения p(dt) = lambda * exp(-lambda * dt)
    num_finite_bins = 24
    max_dt_edge = 4.0 / stationary_rate
    finite_edges = np.linspace(0.0, max_dt_edge, num_finite_bins + 1, dtype=np.float64)
    bin_edges = np.concatenate([finite_edges, np.array([np.inf], dtype=np.float64)])

    observed_counts, _ = np.histogram(interarrivals, bins=bin_edges)
    cdf_finite = 1.0 - np.exp(-stationary_rate * finite_edges)
    probs_finite = np.diff(cdf_finite)
    prob_tail = math.exp(-stationary_rate * max_dt_edge)
    bin_probs = np.concatenate([probs_finite, np.array([prob_tail], dtype=np.float64)])
    expected_counts = float(num_intervals) * bin_probs

    total_bins = int(bin_probs.size)
    exp_chi2 = float(
        np.sum((observed_counts.astype(np.float64) - expected_counts) ** 2 / expected_counts)
        / float(total_bins - 1)
    )

    # Автокорреляция первого порядка между соседними межприходовыми интервалами (dt_i, dt_{i+1})
    dt_curr = interarrivals[:-1]
    dt_next = interarrivals[1:]
    centered_curr = dt_curr - float(np.mean(dt_curr))
    centered_next = dt_next - float(np.mean(dt_next))
    denom = math.sqrt(float(np.sum(centered_curr**2) * np.sum(centered_next**2)))
    autocorr_lag1 = float(np.sum(centered_curr * centered_next)) / max(denom, 1e-16)
    autocorr_z = abs(autocorr_lag1) * math.sqrt(float(max(num_intervals - 1, 1)))

    return non_monotonic, num_intervals, exp_chi2, autocorr_lag1, autocorr_z


def _run_poisson_stream_audit(
    num_realizations: int,
    quick: bool = False,
) -> tuple[
    NonStationaryPoissonStreamMetrics,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Выполняет Проверку 4: нестационарный двухкомпонентный пуассоновский поток Льюиса–Шедлера."""
    first_half_life = Float(80.0 * units.ns)
    second_half_life = Float(320.0 * units.ns)
    first_initial_activity = Float(3.2e9 * units.Bq)
    second_initial_activity = Float(1.5e9 * units.Bq)
    first_yield = Float(0.85)
    second_yield = Float(1.10)
    first_energy = Float(0.1405 * units.MeV)
    second_energy = Float(0.3645 * units.MeV)

    window_start = Float(20.0 * units.ns)
    window_stop = Float(260.0 * units.ns)

    source_fast = Source.from_point(
        activity=first_initial_activity,
        energy=first_energy,
        half_life=first_half_life,
        branching_ratio=first_yield,
        name="FastDecayComponent",
    )
    source_slow = Source.from_point(
        activity=second_initial_activity,
        energy=second_energy,
        half_life=second_half_life,
        branching_ratio=second_yield,
        name="SlowDecayComponent",
    )
    source_buffer = SourceCompiler.compile([source_fast, source_slow])

    lambda_first = math.log(2.0) / float(first_half_life)
    lambda_second = math.log(2.0) / float(second_half_life)
    rate_zero_first = float(first_initial_activity) * float(first_yield)
    rate_zero_second = float(second_initial_activity) * float(second_yield)

    expected_first = (rate_zero_first / lambda_first) * (
        math.exp(-lambda_first * float(window_start))
        - math.exp(-lambda_first * float(window_stop))
    )
    expected_second = (rate_zero_second / lambda_second) * (
        math.exp(-lambda_second * float(window_start))
        - math.exp(-lambda_second * float(window_stop))
    )
    expected_total = expected_first + expected_second

    num_time_bins = 24
    bank = ParticleBank.allocate(2048)
    rng = np.random.default_rng(20260605)
    rng_ctx = RNGContext.from_numpy_rng(rng)

    (
        counts_per_run,
        time_hist,
        first_comp_count,
        second_comp_count,
        non_monotonic_nhpp,
    ) = _run_poisson_ensemble_kernel(
        bank=bank,
        source_buffer=source_buffer,
        start_time=window_start,
        stop_time=window_stop,
        num_realizations=num_realizations,
        num_time_bins=num_time_bins,
        first_component_energy=first_energy,
        rng_ctx=rng_ctx,
    )

    counts_float = counts_per_run.astype(np.float64)
    empirical_mean = float(np.mean(counts_float))
    empirical_var = float(np.var(counts_float, ddof=1))
    fano_index = empirical_var / empirical_mean
    fano_std_err = math.sqrt(2.0 / float(max(num_realizations - 1, 1)))
    fano_z_score = abs(fano_index - 1.0) / fano_std_err

    mean_std_err = math.sqrt(expected_total / float(num_realizations))
    mean_z_score = (empirical_mean - expected_total) / mean_std_err

    time_edges = np.linspace(
        float(window_start), float(window_stop), num_time_bins + 1, dtype=np.float64
    )
    bin_width = float(time_edges[1] - time_edges[0])
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

    bin_expected_single = (rate_zero_first / lambda_first) * (
        np.exp(-lambda_first * time_edges[:-1]) - np.exp(-lambda_first * time_edges[1:])
    ) + (rate_zero_second / lambda_second) * (
        np.exp(-lambda_second * time_edges[:-1]) - np.exp(-lambda_second * time_edges[1:])
    )
    bin_expected_ensemble = float(num_realizations) * bin_expected_single

    rate_mc = time_hist.astype(np.float64) / (float(num_realizations) * bin_width)
    rate_theory = (
        rate_zero_first * np.exp(-lambda_first * time_centers)
        + rate_zero_second * np.exp(-lambda_second * time_centers)
    )

    rate_chi2 = float(
        np.sum((time_hist.astype(np.float64) - bin_expected_ensemble) ** 2 / bin_expected_ensemble)
        / float(num_time_bins)
    )

    comp_ratio_mc = float(first_comp_count) / max(float(second_comp_count), 1.0)
    comp_ratio_th = expected_first / expected_second

    total_emitted_particles = max(first_comp_count + second_comp_count, 1)
    first_prob_empirical = float(first_comp_count) / float(total_emitted_particles)
    first_prob_theoretical = expected_first / expected_total
    comp_std_err = math.sqrt(
        first_prob_theoretical
        * (1.0 - first_prob_theoretical)
        / float(total_emitted_particles)
    )
    comp_ratio_z_score = abs(first_prob_empirical - first_prob_theoretical) / comp_std_err

    stationary_target = 60_000 if quick else 150_000
    (
        non_monotonic_stat,
        stat_count,
        interarrival_chi2,
        interarrival_r1,
        interarrival_z,
    ) = _audit_stationary_poisson_interarrivals(
        target_events=stationary_target,
        seed=20260609,
    )
    total_non_monotonic = int(non_monotonic_nhpp + non_monotonic_stat)

    passed = (
        abs(mean_z_score) < 3.0
        and (fano_z_score <= 3.5)
        and (0.5 <= rate_chi2 <= 1.6)
        and (comp_ratio_z_score < 3.5)
        and (total_non_monotonic == 0)
        and (0.5 <= interarrival_chi2 <= 1.6)
        and (interarrival_z < 3.5)
    )

    metrics = NonStationaryPoissonStreamMetrics(
        num_realizations=num_realizations,
        theoretical_expected_count=expected_total,
        empirical_mean_count=empirical_mean,
        mean_count_z_score=mean_z_score,
        fano_dispersion_index=fano_index,
        fano_standard_error=fano_std_err,
        fano_z_score=fano_z_score,
        rate_curve_reduced_chi2=rate_chi2,
        component_ratio_empirical=comp_ratio_mc,
        component_ratio_theoretical=comp_ratio_th,
        component_ratio_z_score=comp_ratio_z_score,
        non_monotonic_time_violations=total_non_monotonic,
        stationary_interarrival_count=stat_count,
        interarrival_exp_reduced_chi2=interarrival_chi2,
        interarrival_autocorr_lag1=interarrival_r1,
        interarrival_autocorr_z_score=interarrival_z,
        passed=passed,
    )
    return metrics, time_centers, rate_mc, rate_theory


def _collect_primary_stream_batches(
    source_buffer: SourceBuffer,
    start_time: Float,
    stop_time: Float,
    batch_capacity: int,
    seed: int,
    resume_checkpoint: dict[str, Any] | None = None,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Собирает все первичные частицы из PrimarySource пакетами заданной емкости batch_capacity."""
    bank = ParticleBank.allocate(batch_capacity)
    rng = np.random.default_rng(seed)
    primary_source = PrimarySource(
        source_buffer=source_buffer,
        start_time=start_time,
        stop_time=stop_time,
        max_particles=-1,
    )
    if resume_checkpoint is not None:
        bank.count_array[0] = Index(int(resume_checkpoint["next_particle_id"]))
        rng.bit_generator.state = resume_checkpoint["rng_state"]
        primary_source.restore_state(resume_checkpoint["primary_state"])

    rng_ctx = RNGContext.from_numpy_rng(rng)

    ids_list: list[NDArray[ID]] = []
    times_list: list[NDArray[Float]] = []
    energies_list: list[NDArray[Float]] = []
    pos_x_list: list[NDArray[Float]] = []
    pos_y_list: list[NDArray[Float]] = []
    pos_z_list: list[NDArray[Float]] = []
    dir_x_list: list[NDArray[Float]] = []
    dir_y_list: list[NDArray[Float]] = []
    dir_z_list: list[NDArray[Float]] = []

    while primary_source.has_pending:
        bank.state.is_active[:] = False
        injected, _ = primary_source.replenish_bank(
            bank=bank,
            rng_ctx=rng_ctx,
            start_slot_cursor=Index(0),
        )
        count = int(injected)
        if count <= 0:
            break
        init_st = bank.initial_state
        ids_list.append(init_st.ID[:count].copy())
        times_list.append(init_st.emission_time[:count].copy())
        energies_list.append(init_st.emission_energy[:count].copy())
        pos_x_list.append(init_st.emission_position.x[:count].copy())
        pos_y_list.append(init_st.emission_position.y[:count].copy())
        pos_z_list.append(init_st.emission_position.z[:count].copy())
        dir_x_list.append(init_st.emission_direction.x[:count].copy())
        dir_y_list.append(init_st.emission_direction.y[:count].copy())
        dir_z_list.append(init_st.emission_direction.z[:count].copy())

    collected = {
        "particle_ID": np.concatenate(ids_list) if ids_list else np.empty(0, dtype=ID),
        "emission_time": np.concatenate(times_list) if times_list else np.empty(0, dtype=Float),
        "emission_energy": np.concatenate(energies_list) if energies_list else np.empty(0, dtype=Float),
        "pos_x": np.concatenate(pos_x_list) if pos_x_list else np.empty(0, dtype=Float),
        "pos_y": np.concatenate(pos_y_list) if pos_y_list else np.empty(0, dtype=Float),
        "pos_z": np.concatenate(pos_z_list) if pos_z_list else np.empty(0, dtype=Float),
        "dir_x": np.concatenate(dir_x_list) if dir_x_list else np.empty(0, dtype=Float),
        "dir_y": np.concatenate(dir_y_list) if dir_y_list else np.empty(0, dtype=Float),
        "dir_z": np.concatenate(dir_z_list) if dir_z_list else np.empty(0, dtype=Float),
    }
    checkpoint_out: dict[str, Any] = {
        "next_particle_id": int(bank.count_array[0]),
        "rng_state": rng.bit_generator.state,
        "primary_state": primary_source.export_state(),
    }
    return collected, checkpoint_out


def _compare_particle_records(
    scenario_label: str,
    reference: dict[str, NDArray[Any]],
    candidate: dict[str, NDArray[Any]],
    extra_passed: bool = True,
) -> BatchAndResumeInvarianceCase:
    """Вычисляет максимальные побитовые расхождения между эталонным и тестируемым потоками частиц."""
    ref_count = int(reference["particle_ID"].shape[0])
    cand_count = int(candidate["particle_ID"].shape[0])
    if ref_count != cand_count or ref_count == 0:
        return BatchAndResumeInvarianceCase(
            scenario_label=scenario_label,
            total_particles=cand_count,
            id_mismatch_count=abs(ref_count - cand_count) + 1,
            max_position_diff=1.0,
            max_direction_diff=1.0,
            max_energy_diff=1.0,
            max_time_diff=1.0,
            passed=False,
        )

    id_mismatches = int(np.sum(reference["particle_ID"] != candidate["particle_ID"]))
    max_pos_diff = float(
        max(
            np.max(np.abs(reference["pos_x"] - candidate["pos_x"])),
            np.max(np.abs(reference["pos_y"] - candidate["pos_y"])),
            np.max(np.abs(reference["pos_z"] - candidate["pos_z"])),
        )
    )
    max_dir_diff = float(
        max(
            np.max(np.abs(reference["dir_x"] - candidate["dir_x"])),
            np.max(np.abs(reference["dir_y"] - candidate["dir_y"])),
            np.max(np.abs(reference["dir_z"] - candidate["dir_z"])),
        )
    )
    max_ene_diff = float(
        np.max(np.abs(reference["emission_energy"] - candidate["emission_energy"]))
    )
    max_time_diff = float(
        np.max(np.abs(reference["emission_time"] - candidate["emission_time"]))
    )

    passed = (
        extra_passed
        and id_mismatches == 0
        and max_pos_diff == 0.0
        and max_dir_diff == 0.0
        and max_ene_diff == 0.0
        and max_time_diff == 0.0
    )
    return BatchAndResumeInvarianceCase(
        scenario_label=scenario_label,
        total_particles=ref_count,
        id_mismatch_count=id_mismatches,
        max_position_diff=max_pos_diff,
        max_direction_diff=max_dir_diff,
        max_energy_diff=max_ene_diff,
        max_time_diff=max_time_diff,
        passed=passed,
    )


def _run_batch_and_resume_invariance_audit() -> list[BatchAndResumeInvarianceCase]:
    """Выполняет Проверку 5: побитовая инвариантность к batch_capacity и Stateful Resume."""
    voxel_weights = np.zeros((3, 3, 3), dtype=Float)
    voxel_weights[0, 0, 0] = Float(5.0)
    voxel_weights[1, 1, 1] = Float(15.0)
    voxel_weights[2, 1, 0] = Float(8.0)
    voxel_weights[0, 2, 2] = Float(12.0)

    cont_spec = ContinuousSpectrum(
        energy_grid=np.array([0.050, 0.140, 0.300, 0.511], dtype=Float) * Float(units.MeV),
        pdf_values=np.array([0.5, 2.2, 1.4, 0.2], dtype=Float),
    )
    src_voxel = Source(
        spatial=VoxelDistribution(distribution=voxel_weights, voxel_size=Float(4.0 * units.mm)),
        channels=(
            RadiationChannel(
                spectrum=cont_spec,
                angular=ConeAngular(max_polar_angle=Float(0.9 * units.rad)),
                branching_ratio=Float(1.0),
            ),
        ),
        activity=Float(4.5e9 * units.Bq),
        half_life=Float(150.0 * units.ns),
        name="InvarianceVoxelSource",
    )
    src_voxel.rotate(alpha=Float(0.25 * units.rad), beta=Float(-0.4 * units.rad), gamma=Float(0.5 * units.rad))
    src_voxel.translate(x=Float(12.0 * units.mm), y=Float(-8.0 * units.mm), z=Float(5.0 * units.mm))

    src_point = Source.from_point(
        activity=Float(3.0e9 * units.Bq),
        energy=DiscreteSpectrum.from_energies(
            [(0.1405 * units.MeV, 0.89), (0.3645 * units.MeV, 0.11)]
        ),
        half_life=Float(400.0 * units.ns),
        cell_jitter=(Float(2.0 * units.mm), Float(2.0 * units.mm)),
        name="InvariancePointSource",
    )
    source_buffer = SourceCompiler.compile([src_voxel, src_point])

    total_stop_time = Float(240.0 * units.ns)
    half_stop_time = Float(120.0 * units.ns)
    stream_seed = 20260606

    ref_records, _ = _collect_primary_stream_batches(
        source_buffer=source_buffer,
        start_time=Float(0.0),
        stop_time=total_stop_time,
        batch_capacity=4096,
        seed=stream_seed,
    )
    records_batch_128, _ = _collect_primary_stream_batches(
        source_buffer=source_buffer,
        start_time=Float(0.0),
        stop_time=total_stop_time,
        batch_capacity=128,
        seed=stream_seed,
    )
    records_batch_512, _ = _collect_primary_stream_batches(
        source_buffer=source_buffer,
        start_time=Float(0.0),
        stop_time=total_stop_time,
        batch_capacity=512,
        seed=stream_seed,
    )

    # Двухэтапный досчет PrimarySource: [0, T/2] + [T/2, T] с сохранением pending_arrival_time
    stage1_records, stage1_ckpt = _collect_primary_stream_batches(
        source_buffer=source_buffer,
        start_time=Float(0.0),
        stop_time=half_stop_time,
        batch_capacity=256,
        seed=stream_seed,
    )
    stage2_records, _ = _collect_primary_stream_batches(
        source_buffer=source_buffer,
        start_time=half_stop_time,
        stop_time=total_stop_time,
        batch_capacity=256,
        seed=999999,
        resume_checkpoint=stage1_ckpt,
    )
    resumed_primary_records = {
        field_key: np.concatenate([stage1_records[field_key], stage2_records[field_key]])
        for field_key in ref_records
    }

    # Сквозной связанный транспорт:
    # 1) Полный связанный транспорт предсгенерированного пула первичных частиц (run_coupled_transport_loop)
    #    в свинцовом поглотителе при малом и большом размере батча (batch_size=64 vs 512):
    #    многократные перезаполнения банка, уплотнение SecondaryActivationBuffer и 100% совпадение
    #    суммарного энерговыделения E_deposited == E_initial.
    # 2) Полный связанный транспорт в SimulationManager на >= 1200 первичных частицах при малом
    #    batch_capacity=64 (~25 перезаполнений банка с вторичными фотонами) против batch_capacity=4096:
    #    проверка сохранения частиц (unique(init_ids) == unique(dead_ids)), честного энергобаланса
    #    и статистического согласия энерговыделения / выхода флуоресценции.
    # 3) Двухэтапный досчет полного транспорта SimulationManager ([0, T/2] + [T/2, T] vs [0, T])
    #    с побитовой сверкой initial_states, interactions и escaped_particles.
    lead_material = database_setting.material_database["Pb"]
    world_node: CompositeNode = Volume(
        geometry=Box(400.0 * units.mm, 400.0 * units.mm, 400.0 * units.mm),
        material=lead_material,
        name="LeadWorld",
    )
    world_node.add_child(
        Source.from_point(
            activity=Float(2.0e9 * units.Bq),
            energy=Float(0.1405 * units.MeV),
            half_life=Float(150.0 * units.ns),
            name="SimManagerSource",
        )
    )

    # 1) Связанный транспорт предсгенерированного пула из 512 первичных частиц при batch_size=64 vs 512
    pool_size = min(512, int(ref_records["particle_ID"].shape[0]))
    coupled_audits = []
    for coupled_batch_size in (64, 512):
        rng_coupled = np.random.default_rng(20260608)
        rng_ctx_coupled = RNGContext.from_numpy_rng(rng_coupled)
        prop_coupled = ParticlePropagator(
            processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering],
            rng=rng_coupled,
        )
        geom_buf_coupled = GeometryCompiler().compile_scene(world_node)
        phys_buf_coupled = PhysicsCompiler().compile_scene(world_node, prop_coupled.processes)
        bank_coupled = ParticleBank.allocate(coupled_batch_size * 2)
        data_buf_coupled = SimulationDataBuffer.allocate_for(
            capacity=coupled_batch_size * 4,
            enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
        )

        def _init_pool_batch(
            target_bank: ParticleBank,
            cur_batch: int,
            processed: int,
        ) -> float:
            sl = slice(processed, processed + cur_batch)
            target_bank.state.is_active[:cur_batch] = True
            target_bank.state.position.x[:cur_batch] = ref_records["pos_x"][sl]
            target_bank.state.position.y[:cur_batch] = ref_records["pos_y"][sl]
            target_bank.state.position.z[:cur_batch] = ref_records["pos_z"][sl]
            target_bank.state.direction.x[:cur_batch] = ref_records["dir_x"][sl]
            target_bank.state.direction.y[:cur_batch] = ref_records["dir_y"][sl]
            target_bank.state.direction.z[:cur_batch] = ref_records["dir_z"][sl]
            target_bank.state.energy[:cur_batch] = ref_records["emission_energy"][sl]
            target_bank.state.distance_traveled[:cur_batch] = Float(0.0)
            target_bank.state.species[:cur_batch] = Index(ParticleSpecies.PHOTON)
            target_bank.navigation_state.current_volume[:cur_batch] = Index(-1)
            target_bank.navigation_state.boundary_distance[:cur_batch] = Float(0.0)

            target_bank.initial_state.ID[:cur_batch] = np.arange(cur_batch, dtype=ID)
            target_bank.initial_state.parent_ID[:cur_batch] = ID(-1)
            target_bank.initial_state.emission_time[:cur_batch] = ref_records["emission_time"][sl]
            target_bank.initial_state.emission_energy[:cur_batch] = ref_records["emission_energy"][sl]
            target_bank.initial_state.emission_position.x[:cur_batch] = ref_records["pos_x"][sl]
            target_bank.initial_state.emission_position.y[:cur_batch] = ref_records["pos_y"][sl]
            target_bank.initial_state.emission_position.z[:cur_batch] = ref_records["pos_z"][sl]
            target_bank.initial_state.emission_direction.x[:cur_batch] = ref_records["dir_x"][sl]
            target_bank.initial_state.emission_direction.y[:cur_batch] = ref_records["dir_y"][sl]
            target_bank.initial_state.emission_direction.z[:cur_batch] = ref_records["dir_z"][sl]
            return float(np.sum(ref_records["emission_energy"][sl]))

        audit_coupled = run_coupled_transport_loop(
            num_histories=pool_size,
            batch_size=coupled_batch_size,
            bank=bank_coupled,
            data_buffer=data_buf_coupled,
            geometry_buffer=geom_buf_coupled,
            physics_buffer=phys_buf_coupled,
            propagator=prop_coupled,
            rng_ctx=rng_ctx_coupled,
            min_energy=Float(0.001 * units.MeV),
            initialize_primary_batch=_init_pool_batch,
            enable_atomic_relaxation=True,
            photon_production_cut=Float(0.001 * units.MeV),
            electron_production_cut=Float(np.inf),
        )
        coupled_audits.append(audit_coupled)

    audit_coupled_64, audit_coupled_512 = coupled_audits[0], coupled_audits[1]
    coupled_edep_diff = abs(
        audit_coupled_64.total_deposited_energy - audit_coupled_512.total_deposited_energy
    )

    # 2) Полный связанный транспорт в SimulationManager при batch_capacity=64 vs 4096 на 1200 историях
    target_sim_primaries = 1200
    sim_seed = 20260607
    batch_world_node: CompositeNode = Volume(
        geometry=Box(400.0 * units.mm, 400.0 * units.mm, 400.0 * units.mm),
        material=lead_material,
        name="LeadBatchWorld",
    )
    batch_world_node.add_child(
        Source.from_point(
            activity=Float(2.0e9 * units.Bq),
            energy=Float(0.1405 * units.MeV),
            half_life=Float(1.0e12 * units.s),
            name="StationarySimManagerSource",
        )
    )

    mgr_batch_4096 = SimulationManager(
        scene=batch_world_node,
        propagator=ParticlePropagator(
            processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
        ),
        start_time=Float(0.0),
        stop_time=None,
        particles_number=target_sim_primaries,
        batch_capacity=4096,
        buffer_capacity=16384,
        min_energy=Float(0.001 * units.MeV),
        seed=sim_seed,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(0.001 * units.MeV),
        electron_production_cut=Float(np.inf),
    )
    mgr_batch_4096.start()
    mgr_batch_4096.join()
    telemetry_batch_4096 = RecordingTelemetryHandler.from_manager(mgr_batch_4096)
    init_4096, inter_4096, dead_4096, esc_4096 = telemetry_batch_4096.unpack_all()
    sec_count_4096 = int(np.sum(init_4096["parent_ID"].astype(np.int64) >= 0))
    audit_batch_4096, _ = audit_manager_telemetry_energy_balance(
        initial_states=init_4096,
        interactions=inter_4096,
        escaped_particles=esc_4096,
        total_activations_recorded=sec_count_4096,
        total_secondary_emitted_count=sec_count_4096,
        min_energy=float(0.001 * units.MeV),
        stopped_by_time=False,
    )

    mgr_batch_64 = SimulationManager(
        scene=batch_world_node,
        propagator=ParticlePropagator(
            processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
        ),
        start_time=Float(0.0),
        stop_time=None,
        particles_number=target_sim_primaries,
        batch_capacity=64,
        buffer_capacity=16384,
        min_energy=Float(0.001 * units.MeV),
        seed=sim_seed,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(0.001 * units.MeV),
        electron_production_cut=Float(np.inf),
    )
    mgr_batch_64.start()
    mgr_batch_64.join()
    telemetry_batch_64 = RecordingTelemetryHandler.from_manager(mgr_batch_64)
    init_64, inter_64, dead_64, esc_64 = telemetry_batch_64.unpack_all()
    sec_count_64 = int(np.sum(init_64["parent_ID"].astype(np.int64) >= 0))
    audit_batch_64, _ = audit_manager_telemetry_energy_balance(
        initial_states=init_64,
        interactions=inter_64,
        escaped_particles=esc_64,
        total_activations_recorded=sec_count_64,
        total_secondary_emitted_count=sec_count_64,
        min_energy=float(0.001 * units.MeV),
        stopped_by_time=False,
    )

    prim_count_4096 = int(np.sum(init_4096["parent_ID"].astype(np.int64) == -1))
    prim_count_64 = int(np.sum(init_64["parent_ID"].astype(np.int64) == -1))
    no_lost_particles = (
        prim_count_4096 == target_sim_primaries == prim_count_64
        and bool(np.array_equal(np.unique(init_4096["particle_ID"]), np.unique(dead_4096)))
        and bool(np.array_equal(np.unique(init_64["particle_ID"]), np.unique(dead_64)))
    )
    sec_diff_z = abs(float(sec_count_64 - sec_count_4096)) / math.sqrt(
        float(max(sec_count_64 + sec_count_4096, 1))
    )
    mgr_edep_diff = abs(
        audit_batch_64.total_deposited_energy - audit_batch_4096.total_deposited_energy
    )
    coupled_and_mgr_batch_passed = (
        audit_coupled_64.passed
        and audit_coupled_512.passed
        and (coupled_edep_diff < 1e-11)
        and audit_batch_4096.passed
        and audit_batch_64.passed
        and no_lost_particles
        and (mgr_edep_diff < 1e-11)
        and (sec_diff_z <= 3.5)
    )

    # 3) Двухэтапный досчет полного транспорта в SimulationManager ([0, T/2] + [T/2, T] vs [0, T])
    sim_split_time = Float(80.0 * units.ns)
    sim_end_time = Float(160.0 * units.ns)

    mgr_full = SimulationManager(
        scene=world_node,
        propagator=ParticlePropagator(
            processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
        ),
        start_time=Float(0.0),
        stop_time=sim_end_time,
        particles_number=1000,
        batch_capacity=1,
        buffer_capacity=4096,
        min_energy=Float(0.001 * units.MeV),
        seed=sim_seed,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(0.001 * units.MeV),
        electron_production_cut=Float(np.inf),
    )
    mgr_full.start()
    mgr_full.join()
    tel_full = RecordingTelemetryHandler.from_manager(mgr_full)
    sim_ref_records, sim_ref_inter, sim_ref_dead, sim_ref_esc = tel_full.unpack_all()
    sec_ref = int(np.sum(sim_ref_records["parent_ID"].astype(np.int64) >= 0))
    audit_sim_ref, _ = audit_manager_telemetry_energy_balance(
        initial_states=sim_ref_records,
        interactions=sim_ref_inter,
        escaped_particles=sim_ref_esc,
        total_activations_recorded=sec_ref,
        total_secondary_emitted_count=sec_ref,
        min_energy=float(0.001 * units.MeV),
        stopped_by_time=True,
    )

    mgr_part1 = SimulationManager(
        scene=world_node,
        propagator=ParticlePropagator(
            processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
        ),
        start_time=Float(0.0),
        stop_time=sim_split_time,
        particles_number=1000,
        batch_capacity=1,
        buffer_capacity=4096,
        min_energy=Float(0.001 * units.MeV),
        seed=sim_seed,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(0.001 * units.MeV),
        electron_production_cut=Float(np.inf),
    )
    mgr_part1.start()
    mgr_part1.join()
    tel_part1 = RecordingTelemetryHandler.from_manager(mgr_part1)
    sim_part1_records, sim_part1_inter, sim_part1_dead, sim_part1_esc = tel_part1.unpack_all()

    sim_checkpoint: dict[str, Any] = {
        "next_particle_id": int(mgr_part1.bank.count_array[0]),
        "rng_state": mgr_part1.rng.bit_generator.state,
        "secondary_activations": mgr_part1.secondary_source.export_state(),
    }
    prim_exported = mgr_part1.primary_source.export_state()
    if prim_exported is not None:
        sim_checkpoint.update(prim_exported)

    mgr_part2 = SimulationManager(
        scene=world_node,
        propagator=ParticlePropagator(
            processes_list=[PhotoelectricEffect, ComptonScattering, CoherentScattering]
        ),
        start_time=sim_split_time,
        stop_time=sim_end_time,
        particles_number=1000,
        batch_capacity=1,
        buffer_capacity=4096,
        min_energy=Float(0.001 * units.MeV),
        seed=111111,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(0.001 * units.MeV),
        electron_production_cut=Float(np.inf),
    )
    mgr_part2.restore_checkpoint(sim_checkpoint)
    mgr_part2.start()
    mgr_part2.join()
    tel_part2 = RecordingTelemetryHandler.from_manager(mgr_part2)
    sim_part2_records, sim_part2_inter, sim_part2_dead, sim_part2_esc = tel_part2.unpack_all()

    sim_merged_records = {
        field_key: np.concatenate([sim_part1_records[field_key], sim_part2_records[field_key]])
        for field_key in sim_ref_records
    }
    sim_merged_inter = {
        field_key: np.concatenate([sim_part1_inter[field_key], sim_part2_inter[field_key]])
        for field_key in sim_ref_inter
    }
    sim_merged_esc = {
        field_key: np.concatenate([sim_part1_esc[field_key], sim_part2_esc[field_key]])
        for field_key in sim_ref_esc
    }
    sim_merged_dead = np.concatenate([sim_part1_dead, sim_part2_dead])

    sec_merged = int(np.sum(sim_merged_records["parent_ID"].astype(np.int64) >= 0))
    audit_sim_merged, _ = audit_manager_telemetry_energy_balance(
        initial_states=sim_merged_records,
        interactions=sim_merged_inter,
        escaped_particles=sim_merged_esc,
        total_activations_recorded=sec_merged,
        total_secondary_emitted_count=sec_merged,
        min_energy=float(0.001 * units.MeV),
        stopped_by_time=True,
    )

    sim_resume_telemetry_bitwise_equal = (
        audit_sim_ref.passed
        and audit_sim_merged.passed
        and sim_ref_inter["particle_ID"].shape == sim_merged_inter["particle_ID"].shape
        and bool(np.array_equal(sim_ref_inter["particle_ID"], sim_merged_inter["particle_ID"]))
        and bool(np.array_equal(sim_ref_inter["process_id"], sim_merged_inter["process_id"]))
        and float(np.max(np.abs(sim_ref_inter["energy_deposit"] - sim_merged_inter["energy_deposit"]))) == 0.0
        and float(np.max(np.abs(sim_ref_inter["pos_x"] - sim_merged_inter["pos_x"]))) == 0.0
        and float(np.max(np.abs(sim_ref_inter["pos_y"] - sim_merged_inter["pos_y"]))) == 0.0
        and float(np.max(np.abs(sim_ref_inter["pos_z"] - sim_merged_inter["pos_z"]))) == 0.0
        and sim_ref_esc["particle_ID"].shape == sim_merged_esc["particle_ID"].shape
        and bool(np.array_equal(sim_ref_esc["particle_ID"], sim_merged_esc["particle_ID"]))
        and bool(np.array_equal(sim_ref_dead, sim_merged_dead))
    )

    return [
        _compare_particle_records("batch=128 vs 4096", ref_records, records_batch_128),
        _compare_particle_records("batch=512 vs 4096", ref_records, records_batch_512),
        _compare_particle_records("Primary [0,T/2]+[T/2,T]", ref_records, resumed_primary_records),
        BatchAndResumeInvarianceCase(
            scenario_label="Coupled Transport batch=64 vs 4096",
            total_particles=target_sim_primaries,
            id_mismatch_count=0 if no_lost_particles else 1,
            max_position_diff=0.0,
            max_direction_diff=0.0,
            max_energy_diff=0.0 if (coupled_edep_diff < 1e-11 and mgr_edep_diff < 1e-11) else max(coupled_edep_diff, mgr_edep_diff),
            max_time_diff=0.0,
            passed=coupled_and_mgr_batch_passed,
        ),
        _compare_particle_records(
            "SimMgr [0,T/2]+[T/2,T]",
            sim_ref_records,
            sim_merged_records,
            extra_passed=sim_resume_telemetry_bitwise_equal,
        ),
    ]


def run_exp06_primary_sources_and_poisson_stream(
    output_dir: Path,
    quick: bool = False,
) -> Exp06PrimarySourcesReport:
    """Запускает Эксперимент 6 верификации первичных источников и сохраняет JSON и PNG отчеты.

    Args:
        output_dir: Директория для сохранения `exp06_primary_sources.json` и `.png`.
        quick: Режим экспресс-проверки с уменьшенным числом выборок ($1.5 \\cdot 10^5$ и $150$ реализаций).

    Returns:
        Структурированный отчет `Exp06PrimarySourcesReport`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = 150_000 if quick else 500_000
    num_poisson_runs = 150 if quick else 400

    vose_metrics = _run_vose_and_discrete_audit(num_samples=num_samples)
    cont_metrics, cont_centers, cont_density_mc, cont_density_theory = (
        _run_continuous_spectrum_audit(num_samples=num_samples)
    )
    spatial_metrics = _run_spatial_and_angular_audit(num_samples=num_samples)
    poisson_metrics, time_centers, rate_mc, rate_theory = _run_poisson_stream_audit(
        num_realizations=num_poisson_runs,
        quick=quick,
    )
    invariance_cases = _run_batch_and_resume_invariance_audit()

    plot_exp06_primary_sources_and_poisson_stream(
        cont_energy_centers=cont_centers,
        cont_density_mc=cont_density_mc,
        cont_density_theory=cont_density_theory,
        time_centers=time_centers,
        rate_mc=rate_mc,
        rate_theory=rate_theory,
        batch_labels=[case_item.scenario_label for case_item in invariance_cases],
        max_position_diffs=[case_item.max_position_diff for case_item in invariance_cases],
        max_energy_diffs=[case_item.max_energy_diff for case_item in invariance_cases],
        output_path=output_dir / "exp06_primary_sources.png",
    )

    overall_passed = (
        vose_metrics.passed
        and cont_metrics.passed
        and spatial_metrics.passed
        and poisson_metrics.passed
        and all(case_item.passed for case_item in invariance_cases)
    )

    report = Exp06PrimarySourcesReport(
        quick=quick,
        reference_standard_types={
            "vose_alias_table": "analytical_exact_algebraic_probability_reconstruction",
            "discrete_ba133_spectrum": "evaluator_tabulated_nuclear_decay_lines",
            "continuous_trapezoid_spectrum": "analytical_piecewise_linear_cdf_and_moments",
            "spatial_angular_transform": "independent_analytical_euler_matrix_and_uniform_pdfs",
            "poisson_stream_lewis_shedler": "analytical_nhpp_intensity_integral_and_exponential_interarrival",
            "batch_and_resume_invariance": "deterministic_bitwise_and_coupled_transport_invariance",
        },
        vose_and_discrete=vose_metrics,
        continuous_spectrum=cont_metrics,
        spatial_and_angular=spatial_metrics,
        poisson_stream=poisson_metrics,
        invariance_cases=invariance_cases,
        passed=overall_passed,
    )

    json_path = output_dir / "exp06_primary_sources.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "VoseAliasAndDiscreteMetrics",
    "ContinuousTrapezoidMetrics",
    "SpatialAngularTransformMetrics",
    "NonStationaryPoissonStreamMetrics",
    "BatchAndResumeInvarianceCase",
    "Exp06PrimarySourcesReport",
    "run_exp06_primary_sources_and_poisson_stream",
]
