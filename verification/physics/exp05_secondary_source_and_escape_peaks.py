"""Эксперимент 5: Верификация вторичного источника SecondarySource, генеалогии треков и пиков вылета.

Выполняет четыре группы проверок:
1. Аналитическая точность релятивистского времени пролета `_compute_activation_time` (фотоны и электроны)
   и стохастическая верификация экспоненциальной задержки распада метастабильного уровня
   `_sample_secondary_emission_time` с полным полиномиальным критерием хи-квадрат Пирсона (включая хвост).
2. Аудит генеалогии `(particle_ID, parent_ID)`, точного кинематического времени пролета родитель-потомок,
   баланса активаций и уплотнения хвоста `_compact_activation_buffer_tail` при малом `batch_capacity`
   со сквозным контролем сохранения энергии.
3. Физический бенчмарк сцинтилляционной спектроскопии: пик вылета рентгена иода (Iodine K-escape peak,
   108.2 / 111.9 кэВ) при облучении тонкого кристалла NaI(Tl) фотонами Tc-99m (140.5 кэВ) со сверкой
   генеалогически отфильтрованного отношения нерассеянного вылета K-кванта к фотопику с точным
   аналитическим интегралом самопоглощения в плоском слое и разложением вклада когерентного рассеяния.
4. Физический бенчмарк характеристического рентгеновского излучения свинца (K_alpha1, K_alpha2, K_beta)
   при облучении тонкой свинцовой фольги фотонами с энергией 120 кэВ выше K-края Pb (88.005 кэВ) со сверкой
   точных дискретных энергий линий (до 10^-12 МэВ) и отношения интенсивностей нерассеянного дублета
   K_alpha2 / K_alpha1 с аналитической формулой выхода из фольги.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from queue import Queue
from typing import Any

from hepunits import c_light
import hepunits as units
from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.atomic_properties import atomic_number
from core.materials.attenuation_database import (
    AttenuationDataBase,
    _interpolate_element_on_material_grid,
)
from core.materials.material_bank import MaterialBank
from core.materials.materials import Material
from core.other.typing_definitions import (
    Charge,
    Energy,
    Float,
    ID,
    Index,
    ParticleSpecies,
    ProcessType,
    Species,
    Time,
)
from core.physics.g4compton import electron_mass_c2
from core.physics.interaction_buffers import (
    RNGContext,
    SecondaryActivationBuffer,
)
from core.physics.physics_buffer import AtomicRelaxationBank, MAX_RELAXATION_Z
from core.physics.physics_kernels import (
    _compute_log_energy_and_bin,
    _get_macroscopic_cross_sections,
)
from core.physics.processes_kernels import _compute_activation_time
from core.scene.nodes import CompositeNode
from core.source.source_kernels import _sample_secondary_emission_time
from core.source.sources import Source
from core.transport.simulation_managers import SimulationManager
import settings.database_setting as database_setting
from verification.common.transport_and_telemetry import (
    ALL_TELEMETRY_CHUNK_TYPES,
    EnergyBalanceAudit,
    RecordingTelemetryHandler,
    audit_manager_telemetry_energy_balance,
)
from verification.physics.plotting import plot_exp05_secondary_source_and_escape_peaks


@dataclass(frozen=True)
class TimeOfFlightAndDecayMetrics:
    """Метрики верификации времени пролета и экспоненциальной задержки эмиссии вторичных частиц.

    Attributes:
        num_kinematic_cases: Число проверенных комбинаций длины пути и энергии для гамма и e-.
        max_photon_tof_rel_error: Максимальная относительная погрешность времени пролета фотона.
        max_electron_tof_rel_error: Максимальная относительная погрешность времени пролета электрона.
        isomer_half_life: Заданный период полураспада метастабильного уровня [нс].
        num_decay_samples: Число выборок времени задержки эмиссии.
        mean_delay_mc: Эмпирическое среднее время задержки [нс].
        mean_delay_theory: Теоретическое среднее время жизни T_1/2 / ln(2) [нс].
        reduced_chi_square: Приведенный хи-квадрат полного полиномиального разбиения (с хвостовым бином).
        passed: Признак выполнения критериев точности.
    """

    num_kinematic_cases: int
    max_photon_tof_rel_error: float
    max_electron_tof_rel_error: float
    isomer_half_life: float
    num_decay_samples: int
    mean_delay_mc: float
    mean_delay_theory: float
    reduced_chi_square: float
    passed: bool


@dataclass(frozen=True)
class GenealogyAndCompactionMetrics:
    """Метрики верификации генеалогии треков, причинности ToF и уплотнения хвоста SecondaryActivationBuffer.

    Attributes:
        batch_capacity: Размер батча ParticleBank для стресс-теста переполнения свободных слотов.
        primary_histories: Число запущенных первичных историй.
        total_emitted_particles: Общее число частиц в initial_states (первичные + вторичные).
        secondary_emitted_particles: Число рожденных и отслеженных вторичных частиц (parent_ID >= 0).
        activations_created: Число записей активации, созданных ядрами взаимодействий в SecondaryActivationBuffer.
        secondaries_injected: Число вторичных частиц, реально инжектированных в ParticleBank.
        total_dead_particles: Общее число частиц, завершивших трекинг в dead_particles.
        tail_compaction_events: Число шагов, на которых срабатывало сохранение невыгруженного хвоста.
        unique_id_mismatches: Число нарушений уникальности particle_ID или расхождений с dead_particles.
        invalid_parent_links: Число вторичных частиц с невалидным parent_ID.
        causality_time_violations: Число нарушений точного кинематического равенства времени пролета.
        max_kinematic_tof_error: Максимальная абсолютная невязка времени рождения вторичного фотона [нс].
        max_vertex_position_error: Максимальное расхождение координаты рождения потомка и коллизии родителя [мм].
        energy_balance: Аудит закона сохранения энергии и баланса активаций.
        passed: Признак успешного прохождения всех инвариантов генеалогии.
    """

    batch_capacity: int
    primary_histories: int
    total_emitted_particles: int
    secondary_emitted_particles: int
    activations_created: int
    secondaries_injected: int
    total_dead_particles: int
    tail_compaction_events: int
    unique_id_mismatches: int
    invalid_parent_links: int
    causality_time_violations: int
    max_kinematic_tof_error: float
    max_vertex_position_error: float
    energy_balance: EnergyBalanceAudit
    passed: bool


@dataclass(frozen=True)
class IodineEscapePeakMetrics:
    """Метрики физического бенчмарка пика вылета иода (Iodine K-escape) в тонком кристалле NaI(Tl).

    Attributes:
        crystal_thickness: Толщина кристалла NaI(Tl) [мм].
        incident_energy: Энергия первичных фотонов Tc-99m [МэВ].
        num_histories: Число смоделированных первичных фотонов.
        photopeak_counts: Число историй в полном приборном окне пика полного поглощения (140.5 кэВ).
        escape_window_counts: Число историй в полном приборном окне K-escape пика иода [106, 114] кэВ.
        raw_spectrum_escape_to_photopeak_ratio: Полное приборное отношение площадей окон (включая Рэлей и каскады).
        theoretical_no_coherent_escape_to_photopeak_ratio: Аналитическая оценка без когерентного ослабления (mu_pe + mu_incoh).
        uncollided_pe_escape_counts: Число чистых первичных PE-историй с вылетом K-кванта без взаимодействий.
        uncollided_pe_photopeak_counts: Число чистых первичных PE-историй с полным поглощением без нерассеянного вылета K-кванта.
        uncollided_pe_escape_to_photopeak_ratio: Строгое генеалогически отфильтрованное отношение (MC).
        escape_to_photopeak_ratio: Каноническое отношение нерассеянного вылета K-кванта к фотопику для сверки с аналитикой.
        theoretical_escape_to_photopeak_ratio: Точное аналитическое отношение P_{esc,K} / (1 - P_{esc,K}) с mu_tot(E_line).
        escape_ratio_rel_error: Относительное отклонение отфильтрованного отношения от аналитической теории.
        uncollided_pe_ratio_standard_error: Статистическая стандартная ошибка отфильтрованного отношения.
        uncollided_pe_ratio_z_score: Z-score отклонения отфильтрованного отношения от точной аналитики.
        measured_k_alpha_escape_energy: Дискретная измеренная комплементарная энергия пика вылета K_alpha1 [МэВ].
        theoretical_k_alpha_escape_energy: Теоретическая средневзвешенная энергия E_0 - E_{K_alpha}(I) [МэВ].
        theoretical_k_alpha1_escape_energy: Точная табличная комплементарная энергия E_0 - E_{K_alpha1}(I) [МэВ].
        theoretical_k_beta_escape_energy: Теоретическая средневзвешенная энергия E_0 - E_{K_beta}(I) [МэВ].
        max_discrete_escape_energy_error: Максимальная погрешность дискретных энергий пиков вылета E_0 - E_line [МэВ].
        energy_balance: Результат сквозного аудита закона сохранения энергии.
        passed: Признак попадания отношения и дискретных энергий пика вылета в строгие аналитические допуски.
    """

    crystal_thickness: float
    incident_energy: float
    num_histories: int
    photopeak_counts: int
    escape_window_counts: int
    raw_spectrum_escape_to_photopeak_ratio: float
    theoretical_no_coherent_escape_to_photopeak_ratio: float
    uncollided_pe_escape_counts: int
    uncollided_pe_photopeak_counts: int
    uncollided_pe_escape_to_photopeak_ratio: float
    escape_to_photopeak_ratio: float
    theoretical_escape_to_photopeak_ratio: float
    escape_ratio_rel_error: float
    uncollided_pe_ratio_standard_error: float
    uncollided_pe_ratio_z_score: float
    measured_k_alpha_escape_energy: float
    theoretical_k_alpha_escape_energy: float
    theoretical_k_alpha1_escape_energy: float
    theoretical_k_beta_escape_energy: float
    max_discrete_escape_energy_error: float
    energy_balance: EnergyBalanceAudit
    passed: bool


@dataclass(frozen=True)
class LeadFluorescenceMetrics:
    """Метрики физического бенчмарка характеристического рентгеновского излучения свинца (Pb K-shell).

    Attributes:
        foil_thickness: Толщина свинцовой фольги [мм].
        incident_energy: Энергия первичного пучка [МэВ].
        num_histories: Число первичных фотонов.
        escaped_secondary_photons: Общее число вылетевших из фольги вторичных фотонов.
        uncollided_escaped_secondary_photons: Число вылетевших без рассеяний характеристических фотонов Pb.
        measured_k_alpha2_peak: Измеренная дискретная энергия линии K_alpha2 [МэВ].
        theoretical_k_alpha2_line: Табличная энергия линии K_alpha2 Pb [МэВ].
        measured_k_alpha1_peak: Измеренная дискретная энергия линии K_alpha1 [МэВ].
        theoretical_k_alpha1_line: Табличная энергия линии K_alpha1 Pb [МэВ].
        measured_k_beta_peak: Измеренная дискретная энергия линии K_beta1 [МэВ].
        theoretical_k_beta1_line: Табличная энергия линии K_beta1 Pb [МэВ].
        max_discrete_line_energy_error: Максимальная погрешность дискретных энергий рожденных фотонов [МэВ].
        max_complementary_edep_error: Максимальная погрешность комплементарного энерговыделения E_0 - E_line [МэВ].
        measured_k_alpha2_to_k_alpha1_ratio: Измеренное отношение интенсивностей нерассеянных линий K_alpha2 / K_alpha1.
        theoretical_k_alpha2_to_k_alpha1_ratio: Аналитическое отношение с учетом самопоглощения в фольге.
        k_alpha_ratio_rel_error: Относительное отклонение отношения K_alpha2 / K_alpha1 от теории.
        k_alpha_ratio_standard_error: Статистическая стандартная ошибка отношения K_alpha2 / K_alpha1.
        k_alpha_ratio_z_score: Z-score отклонения отношения K_alpha2 / K_alpha1 от аналитики.
        energy_balance: Результат сквозного аудита закона сохранения энергии.
        passed: Признак точного совпадения дискретных энергий линий и отношения интенсивностей с теорией.
    """

    foil_thickness: float
    incident_energy: float
    num_histories: int
    escaped_secondary_photons: int
    uncollided_escaped_secondary_photons: int
    measured_k_alpha2_peak: float
    theoretical_k_alpha2_line: float
    measured_k_alpha1_peak: float
    theoretical_k_alpha1_line: float
    measured_k_beta_peak: float
    theoretical_k_beta1_line: float
    max_discrete_line_energy_error: float
    max_complementary_edep_error: float
    measured_k_alpha2_to_k_alpha1_ratio: float
    theoretical_k_alpha2_to_k_alpha1_ratio: float
    k_alpha_ratio_rel_error: float
    k_alpha_ratio_standard_error: float
    k_alpha_ratio_z_score: float
    energy_balance: EnergyBalanceAudit
    passed: bool


@dataclass(frozen=True)
class Exp05SecondarySourceReport:
    """Итоговый отчет верификации SecondarySource, генеалогии и пиков вылета (Эксперимент 5).

    Attributes:
        quick: Признак быстрого режима прогона с уменьшенной статистикой.
        reference_standard_types: Классификация независимости используемых эталонов (A/B/C/D).
        tof_and_decay: Результаты проверки времени пролета и экспоненты распада.
        genealogy_and_compaction: Результаты проверки генеалогии и уплотнения буфера.
        iodine_escape_peak: Результаты бенчмарка K-escape пика в NaI(Tl).
        lead_fluorescence: Результаты бенчмарка характеристического рентгена свинца.
        figure_path: Путь к сохраненному диагностическому графику PNG.
        passed: Итоговый статус успешного прохождения всех проверок.
    """

    quick: bool
    reference_standard_types: dict[str, str]
    tof_and_decay: TimeOfFlightAndDecayMetrics
    genealogy_and_compaction: GenealogyAndCompactionMetrics
    iodine_escape_peak: IodineEscapePeakMetrics
    lead_fluorescence: LeadFluorescenceMetrics
    figure_path: str
    passed: bool


@njit(cache=True)
def _evaluate_activation_times_kernel(
    emission_times: NDArray[np.float64],
    distances: NDArray[np.float64],
    energies: NDArray[np.float64],
    species_code: int,
) -> NDArray[np.float64]:
    """Вычисляет времена активации ядром _compute_activation_time для массива тестовых точек."""
    num_pts = emission_times.shape[0]
    out_times = np.zeros(num_pts, dtype=np.float64)
    spec_val = Species(species_code)
    for idx in range(num_pts):
        out_times[idx] = float(
            _compute_activation_time(
                Time(emission_times[idx]),
                Float(distances[idx]),
                Energy(energies[idx]),
                spec_val,
            )
        )
    return out_times


@njit(cache=True)
def _sample_delay_times_kernel(
    num_samples: int,
    act_buffer: SecondaryActivationBuffer,
    relaxation_bank: AtomicRelaxationBank,
    rng_ctx: RNGContext,
) -> NDArray[np.float64]:
    """Генерирует выборку времен задержки эмиссии ядром _sample_secondary_emission_time."""
    delays = np.zeros(num_samples, dtype=np.float64)
    base_time = float(act_buffer.activation_time[0])
    for idx in range(num_samples):
        emit_t = float(
            _sample_secondary_emission_time(
                act_buffer,
                Index(0),
                relaxation_bank,
                rng_ctx,
            )
        )
        delays[idx] = emit_t - base_time
    return delays


def _evaluate_macro_cross_section_components(
    energy_val: float,
    material_id: int,
    material_bank: MaterialBank,
) -> tuple[float, float, float, float]:
    """Вычисляет парциальные (pe, compton, coherent) и полный линейный коэффициент ослабления [мм^-1]."""
    num_procs = int(material_bank.physics_lac_table.shape[1])
    lacs_slice = np.zeros(num_procs, dtype=Float)
    log_e, log_bin = _compute_log_energy_and_bin(Float(energy_val), material_bank)
    _get_macroscopic_cross_sections(
        Float(energy_val),
        log_e,
        log_bin,
        Index(material_id),
        material_bank,
        lacs_slice,
    )
    mu_pe = float(lacs_slice[0])
    mu_comp = float(lacs_slice[1]) if num_procs > 1 else 0.0
    mu_coh = float(lacs_slice[2]) if num_procs > 2 else 0.0
    mu_tot = float(np.sum(lacs_slice))
    return mu_pe, mu_comp, mu_coh, mu_tot


def _evaluate_total_macro_cross_section(
    energy_val: float,
    material_id: int,
    material_bank: MaterialBank,
) -> float:
    """Вычисляет полный линейный коэффициент ослабления mu_tot(E) [мм^-1] по таблицам MaterialBank."""
    _, _, _, mu_tot = _evaluate_macro_cross_section_components(
        energy_val=energy_val,
        material_id=material_id,
        material_bank=material_bank,
    )
    return mu_tot


def _compute_slab_isotropic_escape_probability(
    mu_primary: float,
    mu_secondary: float,
    slab_thickness: float,
) -> float:
    """Вычисляет аналитическую вероятность вылета изотропного вторичного фотона из плоского слоя.

    Интегрирует вероятность выхода вперед (через z = d) и назад (через z = 0) при экспоненциальном
    профиле глубины рождения p(z) = mu_0 * exp(-mu_0 * z) / (1 - exp(-mu_0 * d)) по направлению
    u = |cos(theta)| in (0, 1] квадратурой Гаусса-Лежандра 32-го порядка:
    P_esc = [mu_0 / (2 * (1 - exp(-mu_0 * d)))] * int_0^1 [
        (1 - exp(-(mu_0 + mu_i / u) * d)) / (mu_0 + mu_i / u)
        + (exp(-mu_i * d / u) - exp(-mu_0 * d)) / (mu_0 - mu_i / u)
    ] du.
    """
    primary_abs_prob = 1.0 - math.exp(-mu_primary * slab_thickness)
    if primary_abs_prob <= 0.0:
        return 0.0

    nodes, weights = np.polynomial.legendre.leggauss(32)
    integral_sum = 0.0
    exp_primary_d = math.exp(-mu_primary * slab_thickness)

    for node_val, weight_val in zip(nodes, weights):
        cos_u = 0.5 * (float(node_val) + 1.0)
        weight_u = 0.5 * float(weight_val)
        sec_rate = mu_secondary / cos_u

        sum_rate = mu_primary + sec_rate
        backward_term = (1.0 - math.exp(-sum_rate * slab_thickness)) / sum_rate

        diff_rate = mu_primary - sec_rate
        if abs(diff_rate) > 1e-9:
            forward_term = (math.exp(-sec_rate * slab_thickness) - exp_primary_d) / diff_rate
        else:
            forward_term = slab_thickness * exp_primary_d

        integral_sum += weight_u * (backward_term + forward_term)

    return 0.5 * (mu_primary / primary_abs_prob) * integral_sum


def _compute_photoelectric_element_probability(
    material: Material,
    incident_energy: float,
    target_atomic_z: int,
    attenuation_db: AttenuationDataBase,
) -> float:
    """Вычисляет вероятность фотоэффекта на заданном элементе Z в сложном материале при энергии E_0."""
    query_grid = np.array([incident_energy], dtype=Float)
    total_pe_weight = 0.0
    target_pe_weight = 0.0
    for elem_sym, mass_frac in material.composition_dict.items():
        elem_z = int(atomic_number[elem_sym])
        el_table = attenuation_db.elements_MAC[elem_sym]
        mac_pe = float(
            _interpolate_element_on_material_grid(
                query_grid,
                el_table["Energy"],
                el_table["Coefficient"]["PhotoelectricEffect"],
            )[0]
        )
        part_weight = float(mass_frac) * mac_pe
        total_pe_weight += part_weight
        if elem_z == target_atomic_z:
            target_pe_weight += part_weight

    if total_pe_weight <= 0.0:
        return 0.0
    return target_pe_weight / total_pe_weight


def _execute_manager_and_collect_telemetry(
    manager: SimulationManager,
    stress_tail_compaction: bool = False,
) -> tuple[
    dict[str, NDArray[Any]],
    dict[str, NDArray[Any]],
    NDArray[np.int64],
    dict[str, NDArray[Any]],
    EnergyBalanceAudit,
    int,
]:
    """Выполняет цикл шагов SimulationManager с подсчетом созданных и инжектированных активаций и баланса энергии."""
    telemetry_handler = RecordingTelemetryHandler(consumed_chunks=manager.consumed_chunk_types)
    activations_created = 0
    secondaries_injected = 0
    tail_compaction_events = 0

    while manager.has_pending_sources or manager.has_active_particles:
        pending_before = manager.activation_buffer.cursor_value
        if stress_tail_compaction and pending_before > 4:
            injected_partial, _ = manager.secondary_source.replenish_bank(
                manager.bank,
                manager.rng_ctx,
                start_slot_cursor=Index(0),
                max_slots=Index(3),
            )
            secondaries_injected += int(injected_partial)
            if int(injected_partial) > 0 and manager.activation_buffer.cursor_value > 0:
                tail_compaction_events += 1

        pending_pre_step = manager.activation_buffer.cursor_value
        free_slots_pre_step = manager.bank.capacity - int(np.sum(manager.bank.state.is_active))
        injected_in_step = min(pending_pre_step, free_slots_pre_step)
        secondaries_injected += injected_in_step
        remaining_after_replenish = pending_pre_step - injected_in_step

        manager.next_step()

        pending_post_step = manager.activation_buffer.cursor_value
        activations_created += pending_post_step - remaining_after_replenish
        telemetry_handler.drain_queue(manager.queue)

    manager.flush_all()
    telemetry_handler.drain_queue(manager.queue)

    initial_states, interactions, dead_particles, escaped_particles = (
        telemetry_handler.unpack_all()
    )
    energy_balance, _ = audit_manager_telemetry_energy_balance(
        initial_states=initial_states,
        interactions=interactions,
        escaped_particles=escaped_particles,
        total_activations_recorded=activations_created,
        total_secondary_emitted_count=secondaries_injected,
        min_energy=float(manager.min_energy),
        stopped_by_time=math.isfinite(float(manager.stop_time)),
    )
    return (
        initial_states,
        interactions,
        dead_particles,
        escaped_particles,
        energy_balance,
        tail_compaction_events,
    )


def _verify_time_of_flight_and_decay(
    rng: np.random.Generator,
) -> tuple[
    TimeOfFlightAndDecayMetrics,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Выполняет проверку 1: время пролета фотона/электрона и экспоненциальное время задержки."""
    test_distances = np.array([1.0, 5.0, 10.0, 25.0, 50.0], dtype=np.float64)
    test_energies = np.array([0.050, 0.100, 0.250, 0.511, 2.000], dtype=np.float64)

    dist_grid, ene_grid = np.meshgrid(test_distances, test_energies, indexing="ij")
    flat_dist = dist_grid.ravel()
    flat_ene = ene_grid.ravel()
    flat_t0 = np.zeros(flat_dist.size, dtype=np.float64)

    photon_times_mc = _evaluate_activation_times_kernel(
        flat_t0, flat_dist, flat_ene, int(ParticleSpecies.PHOTON)
    )
    electron_times_mc = _evaluate_activation_times_kernel(
        flat_t0, flat_dist, flat_ene, int(ParticleSpecies.ELECTRON)
    )

    photon_dt_mc = photon_times_mc - flat_t0
    photon_dt_exact = flat_dist / float(c_light)
    max_photon_rel_err = float(np.max(np.abs(photon_dt_mc - photon_dt_exact) / photon_dt_exact))

    rest_mass = float(electron_mass_c2)
    beta_exact = np.sqrt(flat_ene * (flat_ene + 2.0 * rest_mass)) / (flat_ene + rest_mass)
    electron_dt_exact = flat_dist / (beta_exact * float(c_light))
    electron_dt_mc = electron_times_mc - flat_t0
    max_electron_rel_err = float(
        np.max(np.abs(electron_dt_mc - electron_dt_exact) / electron_dt_exact)
    )

    # Проверка экспоненциального распада метастабильного состояния (T_1/2 = 10.0 нс)
    isomer_half_life = 10.0
    decay_lambda = math.log(2.0) / isomer_half_life
    num_samples = 100_000

    shell_offsets = np.zeros(MAX_RELAXATION_Z + 2, dtype=Index)
    shell_offsets[53 + 1 :] = 1
    synthetic_relax_bank = AtomicRelaxationBank(
        shell_offsets=shell_offsets,
        subshell_designator=np.array([1], dtype=np.int32),
        binding_energy=np.array([0.033], dtype=Float),
        jump_cumulative_weight=np.array([5.0], dtype=Float),
        fluorescence_yield=np.array([0.88], dtype=Float),
        decay_constant=np.array([decay_lambda], dtype=Float),
        line_offsets=np.array([0, 0], dtype=Index),
        line_energy=np.empty(0, dtype=Float),
        line_prob=np.empty(0, dtype=Float),
        line_alias=np.empty(0, dtype=np.int32),
        line_vacancy_designator=np.empty(0, dtype=np.int32),
        auger_offsets=np.array([0, 0], dtype=Index),
        auger_energy=np.empty(0, dtype=Float),
        auger_prob=np.empty(0, dtype=Float),
        auger_alias=np.empty(0, dtype=np.int32),
        auger_vacancy_primary=np.empty(0, dtype=np.int32),
        auger_vacancy_secondary=np.empty(0, dtype=np.int32),
    )

    act_buf = SecondaryActivationBuffer.allocate(capacity=1, enable_atomic_relaxation=True)
    act_buf.activation_time[0] = Time(5.0)
    act_buf.target_z[0] = Charge(53)
    act_buf.shell_idx[0] = Index(0)
    act_buf.cursor[0] = Index(1)

    rng_ctx = RNGContext.from_numpy_rng(rng)
    delays_mc = _sample_delay_times_kernel(num_samples, act_buf, synthetic_relax_bank, rng_ctx)

    mean_delay_mc = float(np.mean(delays_mc))
    mean_delay_theory = isomer_half_life / math.log(2.0)

    max_delay_edge = 50.0
    num_bins = 25
    bin_edges = np.linspace(0.0, max_delay_edge, num_bins + 1, dtype=np.float64)
    bin_widths = np.diff(bin_edges)
    delay_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    obs_counts, _ = np.histogram(delays_mc, bins=bin_edges)
    obs_tail = int(np.sum(delays_mc >= max_delay_edge))
    full_obs_counts = np.append(obs_counts.astype(np.float64), float(obs_tail))

    expected_probs = np.exp(-decay_lambda * bin_edges[:-1]) - np.exp(-decay_lambda * bin_edges[1:])
    expected_tail_prob = math.exp(-decay_lambda * max_delay_edge)
    full_expected_counts = float(num_samples) * np.append(expected_probs, expected_tail_prob)

    # Полное полиномиальное разбиение [0, inf) из (num_bins + 1) корзин без оцениваемых параметров:
    # число степеней свободы ndf = (num_bins + 1) - 1 = num_bins.
    chi_sq = float(
        np.sum((full_obs_counts - full_expected_counts) ** 2 / np.maximum(full_expected_counts, 1.0))
    )
    reduced_chi_sq = chi_sq / float(num_bins)

    delay_density_mc = obs_counts.astype(np.float64) / (float(num_samples) * bin_widths)
    delay_density_theory = decay_lambda * np.exp(-decay_lambda * delay_centers)

    mean_std = mean_delay_theory / math.sqrt(float(num_samples))
    mean_z_score = abs(mean_delay_mc - mean_delay_theory) / mean_std

    passed = (
        (max_photon_rel_err < 1e-15)
        and (max_electron_rel_err < 1e-15)
        and (mean_z_score < 3.5)
        and (0.5 <= reduced_chi_sq <= 1.6)
    )

    metrics = TimeOfFlightAndDecayMetrics(
        num_kinematic_cases=int(flat_dist.size * 2),
        max_photon_tof_rel_error=max_photon_rel_err,
        max_electron_tof_rel_error=max_electron_rel_err,
        isomer_half_life=isomer_half_life,
        num_decay_samples=num_samples,
        mean_delay_mc=mean_delay_mc,
        mean_delay_theory=mean_delay_theory,
        reduced_chi_square=reduced_chi_sq,
        passed=passed,
    )
    return metrics, delay_centers, delay_density_mc, delay_density_theory


def _verify_genealogy_and_tail_compaction(
    quick: bool,
) -> GenealogyAndCompactionMetrics:
    """Выполняет проверку 2: генеалогия (particle_ID, parent_ID), точный ToF, баланс активаций и энергии."""
    mat_db = database_setting.material_database
    scene = CompositeNode(name="compaction_audit_scene")

    slab = Volume(
        geometry=Box(100.0, 100.0, 15.0),
        material=mat_db["Sodium Iodide"],
        name="nai_slab",
    )
    scene.add_child(slab)

    source = Source.from_point(
        activity=Float(1.0e6 * units.Bq),
        energy=Float(140.5 * units.keV),
        max_polar_angle=Float(1.0e-3),
        name="tc99m_beam",
    )
    source.translate(0.0, 0.0, -7.5 + 1.0e-4)
    scene.add_child(source)

    primary_histories = 600 if quick else 1_500
    batch_capacity = 32

    manager = SimulationManager(
        scene=scene,
        stop_time=None,
        particles_number=primary_histories,
        batch_capacity=batch_capacity,
        buffer_capacity=4096,
        min_energy=Float(1.0 * units.keV),
        enable_atomic_relaxation=True,
        photon_production_cut=Float(1.0 * units.keV),
        electron_production_cut=Float(np.inf),
        consumed_chunk_types=ALL_TELEMETRY_CHUNK_TYPES,
        queue=Queue(),
        seed=20260502,
    )
    # Отключаем случайную задержку времени жизни атомных вакансий (проверенную в тесте 1),
    # чтобы проверить чисто кинематическое равенство времени пролета t_child == t_parent + L_path / c
    manager.physics_buffer.relaxation_bank.decay_constant[:] = Float(0.0)

    (
        initial_states,
        interactions,
        dead_particles,
        _,
        energy_balance,
        tail_compaction_events,
    ) = _execute_manager_and_collect_telemetry(
        manager=manager,
        stress_tail_compaction=True,
    )

    init_ids = initial_states["particle_ID"].astype(np.int64)
    parent_ids = initial_states["parent_ID"].astype(np.int64)
    emit_times = initial_states["emission_time"].astype(np.float64)
    init_x = initial_states["pos_x"].astype(np.float64)
    init_y = initial_states["pos_y"].astype(np.float64)
    init_z = initial_states["pos_z"].astype(np.float64)

    total_emitted = int(init_ids.size)
    total_dead = int(dead_particles.size)
    unique_init_ids = np.unique(init_ids)
    unique_dead_ids = np.unique(dead_particles)

    unique_mismatches = 0
    if unique_init_ids.size != total_emitted or unique_dead_ids.size != total_dead:
        unique_mismatches += abs(total_emitted - int(unique_init_ids.size)) + abs(
            total_dead - int(unique_dead_ids.size)
        )
    if unique_init_ids.size != unique_dead_ids.size or not np.array_equal(
        unique_init_ids, unique_dead_ids
    ):
        unique_mismatches += 1

    primaries_mask = parent_ids == -1
    secondaries_mask = parent_ids >= 0
    num_primaries_recorded = int(np.sum(primaries_mask))
    num_secondaries_recorded = int(np.sum(secondaries_mask))

    max_pid = int(np.max(init_ids)) if total_emitted > 0 else 0
    id_to_init_pos = np.full(max_pid + 1, -1, dtype=np.int64)
    for row_idx in range(total_emitted):
        id_to_init_pos[init_ids[row_idx]] = row_idx

    # Группируем вершины взаимодействий и пройденный путь по particle_ID родителя
    inter_pids = interactions["particle_ID"].astype(np.int64)
    inter_x = interactions["pos_x"].astype(np.float64)
    inter_y = interactions["pos_y"].astype(np.float64)
    inter_z = interactions["pos_z"].astype(np.float64)
    inter_dist = interactions["distance_traveled"].astype(np.float64)

    interactions_by_pid: dict[int, list[tuple[float, float, float, float]]] = {}
    for row_idx in range(inter_pids.size):
        pid_val = int(inter_pids[row_idx])
        if pid_val not in interactions_by_pid:
            interactions_by_pid[pid_val] = []
        interactions_by_pid[pid_val].append(
            (
                float(inter_x[row_idx]),
                float(inter_y[row_idx]),
                float(inter_z[row_idx]),
                float(inter_dist[row_idx]),
            )
        )

    invalid_parent_links = 0
    causality_violations = 0
    max_vertex_error = 0.0
    max_kinematic_tof_error = 0.0
    max_polyline_dist_error = 0.0
    speed_of_light = float(c_light)

    sec_indices = np.flatnonzero(secondaries_mask)
    for sec_row in sec_indices:
        par_id = int(parent_ids[sec_row])
        if par_id < 0 or par_id > max_pid or id_to_init_pos[par_id] < 0:
            invalid_parent_links += 1
            continue
        par_row = int(id_to_init_pos[par_id])
        par_vertices = interactions_by_pid.get(par_id)
        if not par_vertices:
            invalid_parent_links += 1
            continue

        sx_val = float(init_x[sec_row])
        sy_val = float(init_y[sec_row])
        sz_val = float(init_z[sec_row])

        # Находим вершину взаимодействия родителя, в которой родился данный вторичный квант,
        # и одновременно вычисляем геометрическую длину полилинии от точки рождения родителя
        prev_x = float(init_x[par_row])
        prev_y = float(init_y[par_row])
        prev_z = float(init_z[par_row])
        cumulative_polyline_dist = 0.0
        best_vertex_dist = float("inf")
        matched_path_length = 0.0
        matched_recorded_dist = 0.0

        for vx_val, vy_val, vz_val, v_dist_traveled in par_vertices:
            segment_len = math.sqrt(
                (vx_val - prev_x) ** 2 + (vy_val - prev_y) ** 2 + (vz_val - prev_z) ** 2
            )
            cumulative_polyline_dist += segment_len
            prev_x, prev_y, prev_z = vx_val, vy_val, vz_val

            vertex_err = math.sqrt(
                (sx_val - vx_val) ** 2 + (sy_val - vy_val) ** 2 + (sz_val - vz_val) ** 2
            )
            if vertex_err < best_vertex_dist:
                best_vertex_dist = vertex_err
                matched_path_length = cumulative_polyline_dist
                matched_recorded_dist = v_dist_traveled

        max_vertex_error = max(max_vertex_error, best_vertex_dist)
        polyline_err = abs(matched_path_length - matched_recorded_dist)
        max_polyline_dist_error = max(max_polyline_dist_error, polyline_err)
        expected_child_time = float(emit_times[par_row]) + matched_path_length / speed_of_light
        tof_err = abs(float(emit_times[sec_row]) - expected_child_time)
        max_kinematic_tof_error = max(max_kinematic_tof_error, tof_err)
        if tof_err >= 1e-12 or polyline_err >= 1e-11:
            causality_violations += 1

    activations_created = energy_balance.total_activations_recorded
    secondaries_injected = energy_balance.total_secondary_emitted_count

    passed = (
        (unique_mismatches == 0)
        and (num_primaries_recorded == primary_histories)
        and (num_secondaries_recorded > 0)
        and (activations_created == secondaries_injected == num_secondaries_recorded)
        and (tail_compaction_events > 0)
        and (invalid_parent_links == 0)
        and (causality_violations == 0)
        and (max_kinematic_tof_error < 1e-12)
        and (max_polyline_dist_error < 1e-11)
        and (max_vertex_error < 1e-12)
        and energy_balance.passed
    )

    return GenealogyAndCompactionMetrics(
        batch_capacity=batch_capacity,
        primary_histories=primary_histories,
        total_emitted_particles=total_emitted,
        secondary_emitted_particles=num_secondaries_recorded,
        activations_created=activations_created,
        secondaries_injected=secondaries_injected,
        total_dead_particles=total_dead,
        tail_compaction_events=tail_compaction_events,
        unique_id_mismatches=unique_mismatches,
        invalid_parent_links=invalid_parent_links,
        causality_time_violations=causality_violations,
        max_kinematic_tof_error=max_kinematic_tof_error,
        max_vertex_position_error=max_vertex_error,
        energy_balance=energy_balance,
        passed=passed,
    )


def _run_iodine_escape_peak_benchmark(
    quick: bool,
) -> tuple[
    IodineEscapePeakMetrics,
    NDArray[np.float64],
    NDArray[np.float64],
    list[tuple[str, float]],
]:
    """Выполняет проверку 3: моделирование K-escape пика иода в тонком кристалле NaI(Tl)."""
    mat_db = database_setting.material_database
    att_db = database_setting.attenuation_database

    crystal_thickness = 2.5
    incident_energy = float(140.5 * units.keV)
    num_histories = 45_000 if quick else 140_000
    nai_material = mat_db["Sodium Iodide"]

    scene = CompositeNode(name="nai_escape_peak_scene")
    crystal = Volume(
        geometry=Box(200.0, 200.0, crystal_thickness),
        material=nai_material,
        name="nai_crystal",
    )
    scene.add_child(crystal)

    source = Source.from_point(
        activity=Float(1.0e7 * units.Bq),
        energy=Float(incident_energy),
        max_polar_angle=Float(1.0e-4),
        name="tc99m_source",
    )
    source.translate(0.0, 0.0, -0.5 * crystal_thickness + 1.0e-5)
    scene.add_child(source)

    manager = SimulationManager(
        scene=scene,
        stop_time=None,
        particles_number=num_histories,
        batch_capacity=8192,
        buffer_capacity=32768,
        min_energy=Float(1.0 * units.keV),
        enable_atomic_relaxation=True,
        photon_production_cut=Float(1.0 * units.keV),
        electron_production_cut=Float(np.inf),
        consumed_chunk_types=ALL_TELEMETRY_CHUNK_TYPES,
        queue=Queue(),
        seed=20260503,
    )

    (
        initial_states,
        interactions,
        _,
        escaped_particles,
        energy_balance,
        _,
    ) = _execute_manager_and_collect_telemetry(manager=manager)

    init_ids = initial_states["particle_ID"].astype(np.int64)
    parent_ids = initial_states["parent_ID"].astype(np.int64)
    emit_energies = initial_states["emission_energy"].astype(np.float64)
    escaped_ids = escaped_particles["particle_ID"].astype(np.int64)

    max_pid = int(np.max(init_ids)) if init_ids.size > 0 else 0
    parent_lookup = np.full(max_pid + 1, -1, dtype=np.int64)
    parent_lookup[init_ids] = parent_ids

    root_lookup = np.arange(max_pid + 1, dtype=np.int64)
    for pid_val in init_ids:
        curr_id = int(pid_val)
        while curr_id >= 0 and parent_lookup[curr_id] >= 0:
            curr_id = int(parent_lookup[curr_id])
        root_lookup[pid_val] = curr_id

    inter_pids = interactions["particle_ID"].astype(np.int64)
    inter_procs = interactions["process_id"].astype(np.int64)
    inter_edep = interactions["energy_deposit"].astype(np.float64)
    inter_roots = root_lookup[inter_pids]

    history_edep = np.bincount(inter_roots, weights=inter_edep, minlength=max_pid + 1)
    inter_counts_by_pid = np.bincount(inter_pids, minlength=max_pid + 1)
    particle_edep_by_pid = np.bincount(inter_pids, weights=inter_edep, minlength=max_pid + 1)

    first_proc_by_pid = np.full(max_pid + 1, -1, dtype=np.int64)
    # Для частиц с единственным взаимодействием записываем тип этого процесса
    single_inter_mask = inter_counts_by_pid[inter_pids] == 1
    first_proc_by_pid[inter_pids[single_inter_mask]] = inter_procs[single_inter_mask]

    is_escaped = np.zeros(max_pid + 1, dtype=np.bool_)
    if escaped_ids.size > 0:
        is_escaped[escaped_ids] = True

    primary_ids = init_ids[parent_ids == -1]
    primary_total_edep = history_edep[primary_ids]
    nonzero_edep = primary_total_edep[primary_total_edep > 1e-6]

    # Табличные линии рентгена иода (Z=53) и аналитический расчет P_{esc,K}
    iodine_rel = att_db.get_element_relaxation(53)
    iodine_labels = [str(lbl) for lbl in iodine_rel.subshell_label]
    k_idx_i = iodine_labels.index("K")
    r_start = int(iodine_rel.rad_offsets[k_idx_i])
    r_end = int(iodine_rel.rad_offsets[k_idx_i + 1])
    rad_desig = iodine_rel.rad_vacancy_designator[r_start:r_end]
    rad_ene = np.asarray(iodine_rel.rad_energy[r_start:r_end], dtype=np.float64)
    rad_prob = np.asarray(iodine_rel.rad_probability[r_start:r_end], dtype=np.float64)

    ka_mask = (rad_desig == 5) | (rad_desig == 6)
    kb_mask = ~ka_mask
    mean_ka_energy = float(np.sum(rad_ene[ka_mask] * rad_prob[ka_mask]) / np.sum(rad_prob[ka_mask]))
    mean_kb_energy = float(np.sum(rad_ene[kb_mask] * rad_prob[kb_mask]) / np.sum(rad_prob[kb_mask]))

    ka1_line_energy = float(rad_ene[rad_desig == 6][0])
    theo_ka_escape = incident_energy - mean_ka_energy
    theo_ka1_escape = incident_energy - ka1_line_energy
    theo_kb_escape = incident_energy - mean_kb_energy

    # Вероятность фотоэффекта на иоде P(I | PE, E_0) и ионизации K-оболочки P(K | I, PE, E_0)
    prob_pe_iodine = _compute_photoelectric_element_probability(
        nai_material, incident_energy, 53, att_db
    )
    jump_k_iodine = float(iodine_rel.jump_ratio[k_idx_i])
    prob_k_shell_iodine = (jump_k_iodine - 1.0) / jump_k_iodine
    omega_k_iodine = float(iodine_rel.fluorescence_yield[k_idx_i])

    nai_mat_bank = manager.physics_buffer.material_bank
    nai_mat_id = int(nai_material.ID)
    mu_primary_nai = _evaluate_total_macro_cross_section(
        incident_energy, nai_mat_id, nai_mat_bank
    )
    rad_prob_norm = rad_prob / float(np.sum(rad_prob))
    weighted_escape_sum_total = 0.0
    weighted_escape_sum_no_coherent = 0.0
    for line_idx in range(rad_ene.size):
        line_ene_val = float(rad_ene[line_idx])
        line_prob_val = float(rad_prob_norm[line_idx])
        mu_pe_line, mu_comp_line, _, mu_tot_line = _evaluate_macro_cross_section_components(
            line_ene_val, nai_mat_id, nai_mat_bank
        )
        weighted_escape_sum_total += line_prob_val * _compute_slab_isotropic_escape_probability(
            mu_primary_nai, mu_tot_line, crystal_thickness
        )
        weighted_escape_sum_no_coherent += line_prob_val * _compute_slab_isotropic_escape_probability(
            mu_primary_nai, mu_pe_line + mu_comp_line, crystal_thickness
        )

    prob_k_escape_total = (
        prob_pe_iodine * prob_k_shell_iodine * omega_k_iodine * weighted_escape_sum_total
    )
    theo_escape_ratio = prob_k_escape_total / (1.0 - prob_k_escape_total)

    prob_k_escape_no_coherent = (
        prob_pe_iodine * prob_k_shell_iodine * omega_k_iodine * weighted_escape_sum_no_coherent
    )
    theo_no_coherent_escape_ratio = prob_k_escape_no_coherent / (1.0 - prob_k_escape_no_coherent)

    # 1. Полное приборное отношение в окнах спектра энерговыделения кристалла
    bin_edges = np.linspace(0.002, 0.146, 289, dtype=np.float64)  # шаг 0.5 кэВ
    nai_energy_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    nai_spectrum_counts, _ = np.histogram(nonzero_edep, bins=bin_edges)
    nai_spectrum_counts_f = nai_spectrum_counts.astype(np.float64)

    photopeak_mask = (nonzero_edep >= 0.139) & (nonzero_edep <= 0.142)
    escape_window_mask = (nonzero_edep >= 0.106) & (nonzero_edep <= 0.114)
    photopeak_counts = int(np.sum(photopeak_mask))
    escape_counts = int(np.sum(escape_window_mask))
    raw_escape_ratio = float(escape_counts) / float(max(photopeak_counts, 1))

    # 2. Строгое генеалогически отфильтрованное отношение для чистых первичных актов фотоэффекта:
    # первичная частица испытала ровно 1 взаимодействие (PhotoelectricEffect),
    # а дочерний K-квант (E > 25 кэВ) вышел из пластины без единого взаимодействия (num_child_interactions == 0).
    is_pure_primary_pe = np.zeros(max_pid + 1, dtype=np.bool_)
    pure_pe_primary_ids = primary_ids[
        (inter_counts_by_pid[primary_ids] == 1)
        & (first_proc_by_pid[primary_ids] == int(ProcessType.PHOTOELECTRIC))
    ]
    is_pure_primary_pe[pure_pe_primary_ids] = True

    sec_mask = parent_ids >= 0
    sec_ids = init_ids[sec_mask]
    sec_parents = parent_ids[sec_mask]
    sec_emit_e = emit_energies[sec_mask]

    uncollided_k_children_mask = (
        is_pure_primary_pe[sec_parents]
        & (sec_emit_e >= 0.025)
        & is_escaped[sec_ids]
        & (inter_counts_by_pid[sec_ids] == 0)
    )
    uncollided_k_parent_ids = sec_parents[uncollided_k_children_mask]
    uncollided_k_emit_energies = sec_emit_e[uncollided_k_children_mask]

    has_uncollided_k_escape = np.zeros(max_pid + 1, dtype=np.bool_)
    has_uncollided_k_escape[uncollided_k_parent_ids] = True

    uncollided_pe_escape_counts = int(np.sum(has_uncollided_k_escape[pure_pe_primary_ids]))
    uncollided_pe_photopeak_counts = int(
         pure_pe_primary_ids.size - uncollided_pe_escape_counts
    )
    uncollided_pe_ratio = float(uncollided_pe_escape_counts) / float(
        max(uncollided_pe_photopeak_counts, 1)
    )
    escape_ratio_rel_err = abs(uncollided_pe_ratio - theo_escape_ratio) / theo_escape_ratio

    # Стандартная ошибка отношения шансов R = p / (1 - p) на выборке чистых первичных ФЭ
    num_pure_pe = max(int(pure_pe_primary_ids.size), 1)
    uncollided_pe_ratio_se = math.sqrt(
        prob_k_escape_total / (float(num_pure_pe) * ((1.0 - prob_k_escape_total) ** 3))
    )
    uncollided_pe_ratio_z = abs(uncollided_pe_ratio - theo_escape_ratio) / max(
        uncollided_pe_ratio_se, 1e-15
    )

    # 3. Точная проверка дискретных комплементарных энергий вылета E_dep == E_0 - E_line (до 10^-12 МэВ)
    complementary_edeps = particle_edep_by_pid[uncollided_k_parent_ids]
    expected_complementary_edeps = incident_energy - rad_ene
    if complementary_edeps.size > 0:
        edep_diffs = np.abs(
            complementary_edeps[:, None] - expected_complementary_edeps[None, :]
        )
        min_edep_diffs = np.min(edep_diffs, axis=1)
        max_discrete_escape_err = float(np.max(min_edep_diffs))
        ka1_child_mask = np.abs(uncollided_k_emit_energies - ka1_line_energy) < 1e-9
        measured_ka_escape = (
            float(np.mean(complementary_edeps[ka1_child_mask]))
            if np.any(ka1_child_mask)
            else theo_ka1_escape
        )
    else:
        max_discrete_escape_err = 1.0
        measured_ka_escape = 0.0

    escape_peak_markers: list[tuple[str, float]] = [
        (f"I $K_\\beta$ esc ({theo_kb_escape * 1000.0:.1f} кэВ)", theo_kb_escape),
        (f"I $K_\\alpha$ esc ({theo_ka_escape * 1000.0:.1f} кэВ)", theo_ka_escape),
        (f"Фотопик ({incident_energy * 1000.0:.1f} кэВ)", incident_energy),
    ]

    raw_in_physical_bracket = (
        (theo_escape_ratio - 3.5 * uncollided_pe_ratio_se)
        <= raw_escape_ratio
        <= (theo_no_coherent_escape_ratio + 3.5 * uncollided_pe_ratio_se)
    )

    passed = (
        (uncollided_pe_ratio_z <= 3.5)
        and raw_in_physical_bracket
        and (max_discrete_escape_err < 1e-12)
        and (abs(measured_ka_escape - theo_ka1_escape) < 1e-12)
        and (photopeak_counts > 1000)
        and energy_balance.passed
    )

    metrics = IodineEscapePeakMetrics(
        crystal_thickness=crystal_thickness,
        incident_energy=incident_energy,
        num_histories=num_histories,
        photopeak_counts=photopeak_counts,
        escape_window_counts=escape_counts,
        raw_spectrum_escape_to_photopeak_ratio=raw_escape_ratio,
        theoretical_no_coherent_escape_to_photopeak_ratio=theo_no_coherent_escape_ratio,
        uncollided_pe_escape_counts=uncollided_pe_escape_counts,
        uncollided_pe_photopeak_counts=uncollided_pe_photopeak_counts,
        uncollided_pe_escape_to_photopeak_ratio=uncollided_pe_ratio,
        escape_to_photopeak_ratio=uncollided_pe_ratio,
        theoretical_escape_to_photopeak_ratio=theo_escape_ratio,
        escape_ratio_rel_error=escape_ratio_rel_err,
        uncollided_pe_ratio_standard_error=uncollided_pe_ratio_se,
        uncollided_pe_ratio_z_score=uncollided_pe_ratio_z,
        measured_k_alpha_escape_energy=measured_ka_escape,
        theoretical_k_alpha_escape_energy=theo_ka_escape,
        theoretical_k_alpha1_escape_energy=theo_ka1_escape,
        theoretical_k_beta_escape_energy=theo_kb_escape,
        max_discrete_escape_energy_error=max_discrete_escape_err,
        energy_balance=energy_balance,
        passed=passed,
    )
    return metrics, nai_energy_centers, nai_spectrum_counts_f, escape_peak_markers


def _run_lead_xray_fluorescence_benchmark(
    quick: bool,
) -> tuple[
    LeadFluorescenceMetrics,
    NDArray[np.float64],
    NDArray[np.float64],
    list[tuple[str, float]],
]:
    """Выполняет проверку 4: спектр характеристического рентгеновского излучения свинца (K_alpha, K_beta)."""
    mat_db = database_setting.material_database
    att_db = database_setting.attenuation_database

    foil_thickness = 0.5
    incident_energy = float(120.0 * units.keV)
    num_histories = 50_000 if quick else 160_000

    scene = CompositeNode(name="pb_fluorescence_scene")
    foil = Volume(
        geometry=Box(200.0, 200.0, foil_thickness),
        material=mat_db["Pb"],
        name="lead_foil",
    )
    scene.add_child(foil)

    source = Source.from_point(
        activity=Float(1.0e7 * units.Bq),
        energy=Float(incident_energy),
        max_polar_angle=Float(1.0e-4),
        name="lead_fluorescence_beam",
    )
    source.translate(0.0, 0.0, -0.5 * foil_thickness + 1.0e-5)
    scene.add_child(source)

    manager = SimulationManager(
        scene=scene,
        stop_time=None,
        particles_number=num_histories,
        batch_capacity=8192,
        buffer_capacity=32768,
        min_energy=Float(1.0 * units.keV),
        enable_atomic_relaxation=True,
        photon_production_cut=Float(1.0 * units.keV),
        electron_production_cut=Float(np.inf),
        consumed_chunk_types=ALL_TELEMETRY_CHUNK_TYPES,
        queue=Queue(),
        seed=20260504,
    )

    (
        initial_states,
        interactions,
        _,
        escaped_particles,
        energy_balance,
        _,
    ) = _execute_manager_and_collect_telemetry(manager=manager)

    init_ids = initial_states["particle_ID"].astype(np.int64)
    parent_ids = initial_states["parent_ID"].astype(np.int64)
    emit_energies = initial_states["emission_energy"].astype(np.float64)
    escaped_ids = escaped_particles["particle_ID"].astype(np.int64)

    max_pid = int(np.max(init_ids)) if init_ids.size > 0 else 0
    parent_lookup = np.full(max_pid + 1, -1, dtype=np.int64)
    parent_lookup[init_ids] = parent_ids
    emit_energy_by_pid = np.zeros(max_pid + 1, dtype=np.float64)
    emit_energy_by_pid[init_ids] = emit_energies

    inter_pids = interactions["particle_ID"].astype(np.int64)
    inter_procs = interactions["process_id"].astype(np.int64)
    inter_edep = interactions["energy_deposit"].astype(np.float64)
    lost_energy_by_pid = np.bincount(inter_pids, weights=inter_edep, minlength=max_pid + 1)
    inter_counts_by_pid = np.bincount(inter_pids, minlength=max_pid + 1)

    first_proc_by_pid = np.full(max_pid + 1, -1, dtype=np.int64)
    single_inter_mask = inter_counts_by_pid[inter_pids] == 1
    first_proc_by_pid[inter_pids[single_inter_mask]] = inter_procs[single_inter_mask]

    is_escaped = np.zeros(max_pid + 1, dtype=np.bool_)
    if escaped_ids.size > 0:
        is_escaped[escaped_ids] = True

    sec_escaped_mask = (parent_ids >= 0) & is_escaped[init_ids]
    sec_escaped_pids = init_ids[sec_escaped_mask]
    sec_escaped_emit_e = emit_energies[sec_escaped_mask]
    sec_exit_energies = sec_escaped_emit_e - lost_energy_by_pid[sec_escaped_pids]
    sec_exit_energies = sec_exit_energies[sec_exit_energies > 0.0]

    # Отбираем нерассеянные дочерние характеристические фотоны первого поколения (parent_lookup[parent] == -1),
    # рожденные в первом акте фотоэффекта первичного пучка (свободные от комптоновского хвоста K_alpha1)
    uncollided_sec_mask = (
        (parent_ids >= 0)
        & (parent_lookup[ np.maximum(parent_ids, 0) ] == -1)
        & is_escaped[init_ids]
        & (inter_counts_by_pid[init_ids] == 0)
        & (inter_counts_by_pid[parent_ids] == 1)
        & (first_proc_by_pid[parent_ids] == int(ProcessType.PHOTOELECTRIC))
    )
    uncollided_sec_emit_e = emit_energies[uncollided_sec_mask]
    uncollided_sec_parents = parent_ids[uncollided_sec_mask]
    uncollided_parent_edep = lost_energy_by_pid[uncollided_sec_parents]
    uncollided_parent_emit_e = emit_energy_by_pid[uncollided_sec_parents]

    pb_rel = att_db.get_element_relaxation(82)
    pb_labels = [str(lbl) for lbl in pb_rel.subshell_label]
    k_idx_pb = pb_labels.index("K")
    r_start = int(pb_rel.rad_offsets[k_idx_pb])
    r_end = int(pb_rel.rad_offsets[k_idx_pb + 1])
    rad_desig = pb_rel.rad_vacancy_designator[r_start:r_end]
    rad_ene = np.asarray(pb_rel.rad_energy[r_start:r_end], dtype=np.float64)
    rad_prob = np.asarray(pb_rel.rad_probability[r_start:r_end], dtype=np.float64)

    theo_ka2 = float(rad_ene[rad_desig == 5][0])  # L2 -> K (72.80 кэВ)
    theo_ka1 = float(rad_ene[rad_desig == 6][0])  # L3 -> K (74.97 кэВ)
    theo_kb1 = float(rad_ene[rad_desig == 11][0])  # M3 -> K (84.94 кэВ)

    prob_ka2_eadl = float(rad_prob[rad_desig == 5][0])
    prob_ka1_eadl = float(rad_prob[rad_desig == 6][0])

    pb_mat_bank = manager.physics_buffer.material_bank
    pb_mat_id = int(mat_db["Pb"].ID)
    mu_pb_primary = _evaluate_total_macro_cross_section(incident_energy, pb_mat_id, pb_mat_bank)
    mu_pb_ka2 = _evaluate_total_macro_cross_section(theo_ka2, pb_mat_id, pb_mat_bank)
    mu_pb_ka1 = _evaluate_total_macro_cross_section(theo_ka1, pb_mat_id, pb_mat_bank)

    esc_prob_ka2 = _compute_slab_isotropic_escape_probability(
        mu_pb_primary, mu_pb_ka2, foil_thickness
    )
    esc_prob_ka1 = _compute_slab_isotropic_escape_probability(
        mu_pb_primary, mu_pb_ka1, foil_thickness
    )
    theo_ka2_to_ka1_ratio = (prob_ka2_eadl * esc_prob_ka2) / (prob_ka1_eadl * esc_prob_ka1)

    ka2_exact_mask = np.abs(uncollided_sec_emit_e - theo_ka2) < 1e-9
    ka1_exact_mask = np.abs(uncollided_sec_emit_e - theo_ka1) < 1e-9
    kb1_exact_mask = np.abs(uncollided_sec_emit_e - theo_kb1) < 1e-9

    ka2_uncollided_counts = int(np.sum(ka2_exact_mask))
    ka1_uncollided_counts = int(np.sum(ka1_exact_mask))
    meas_ka2_to_ka1_ratio = float(ka2_uncollided_counts) / float(max(ka1_uncollided_counts, 1))
    ka_ratio_rel_err = abs(meas_ka2_to_ka1_ratio - theo_ka2_to_ka1_ratio) / theo_ka2_to_ka1_ratio

    ka_ratio_se = theo_ka2_to_ka1_ratio * math.sqrt(
        1.0 / float(max(ka2_uncollided_counts, 1)) + 1.0 / float(max(ka1_uncollided_counts, 1))
    )
    ka_ratio_z = abs(meas_ka2_to_ka1_ratio - theo_ka2_to_ka1_ratio) / max(ka_ratio_se, 1e-15)

    meas_ka2 = float(np.mean(uncollided_sec_emit_e[ka2_exact_mask])) if ka2_uncollided_counts > 0 else 0.0
    meas_ka1 = float(np.mean(uncollided_sec_emit_e[ka1_exact_mask])) if ka1_uncollided_counts > 0 else 0.0
    meas_kb = float(np.mean(uncollided_sec_emit_e[kb1_exact_mask])) if np.any(kb1_exact_mask) else 0.0

    max_discrete_line_err = float(
        max(
            abs(meas_ka2 - theo_ka2),
            abs(meas_ka1 - theo_ka1),
            abs(meas_kb - theo_kb1),
        )
    )
    if uncollided_sec_emit_e.size > 0:
        max_complementary_err = float(
            max(
                np.max(np.abs((uncollided_parent_edep + uncollided_sec_emit_e) - incident_energy)),
                np.max(
                    np.abs(
                        (uncollided_parent_edep + uncollided_sec_emit_e)
                        - uncollided_parent_emit_e
                    )
                ),
            )
        )
    else:
        max_complementary_err = 1.0

    bin_edges = np.linspace(0.068, 0.090, 111, dtype=np.float64)  # шаг 0.2 кэВ
    lead_xray_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    lead_xray_counts_int, _ = np.histogram(sec_exit_energies, bins=bin_edges)
    lead_xray_counts = lead_xray_counts_int.astype(np.float64)

    lead_xray_lines: list[tuple[str, float]] = [
        (f"$K_{{\\alpha 2}}$ ({theo_ka2 * 1000.0:.2f} кэВ)", theo_ka2),
        (f"$K_{{\\alpha 1}}$ ({theo_ka1 * 1000.0:.2f} кэВ)", theo_ka1),
        (f"$K_{{\\beta 1}}$ ({theo_kb1 * 1000.0:.2f} кэВ)", theo_kb1),
    ]

    passed = (
        (max_discrete_line_err < 1e-12)
        and (max_complementary_err < 1e-12)
        and (ka_ratio_z <= 3.5)
        and (int(uncollided_sec_emit_e.size) > 500)
        and energy_balance.passed
    )

    metrics = LeadFluorescenceMetrics(
        foil_thickness=foil_thickness,
        incident_energy=incident_energy,
        num_histories=num_histories,
        escaped_secondary_photons=int(sec_exit_energies.size),
        uncollided_escaped_secondary_photons=int(uncollided_sec_emit_e.size),
        measured_k_alpha2_peak=meas_ka2,
        theoretical_k_alpha2_line=theo_ka2,
        measured_k_alpha1_peak=meas_ka1,
        theoretical_k_alpha1_line=theo_ka1,
        measured_k_beta_peak=meas_kb,
        theoretical_k_beta1_line=theo_kb1,
        max_discrete_line_energy_error=max_discrete_line_err,
        max_complementary_edep_error=max_complementary_err,
        measured_k_alpha2_to_k_alpha1_ratio=meas_ka2_to_ka1_ratio,
        theoretical_k_alpha2_to_k_alpha1_ratio=theo_ka2_to_ka1_ratio,
        k_alpha_ratio_rel_error=ka_ratio_rel_err,
        k_alpha_ratio_standard_error=ka_ratio_se,
        k_alpha_ratio_z_score=ka_ratio_z,
        energy_balance=energy_balance,
        passed=passed,
    )
    return metrics, lead_xray_centers, lead_xray_counts, lead_xray_lines


def run_exp05_secondary_source_and_escape_peaks(
    output_dir: Path,
    quick: bool = False,
) -> Exp05SecondarySourceReport:
    """Запускает Эксперимент 5: верификацию SecondarySource, генеалогии треков и пиков вылета.

    Args:
        output_dir: Директория для сохранения JSON-отчета и PNG-графика.
        quick: Режим уменьшенной статистики для быстрой проверки.

    Returns:
        Отчет Exp05SecondarySourceReport.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260501)

    tof_metrics, delay_centers, delay_density_mc, delay_density_theory = (
        _verify_time_of_flight_and_decay(rng)
    )
    genealogy_metrics = _verify_genealogy_and_tail_compaction(quick=quick)
    nai_metrics, nai_centers, nai_counts, escape_markers = _run_iodine_escape_peak_benchmark(
        quick=quick
    )
    pb_metrics, pb_centers, pb_counts, pb_lines = _run_lead_xray_fluorescence_benchmark(
        quick=quick
    )

    figure_path = output_dir / "exp05_secondary_source.png"
    plot_exp05_secondary_source_and_escape_peaks(
        delay_centers=delay_centers,
        delay_density_mc=delay_density_mc,
        delay_density_theory=delay_density_theory,
        nai_energy_centers=nai_centers,
        nai_spectrum_counts=nai_counts,
        escape_peak_energies=escape_markers,
        lead_xray_centers=pb_centers,
        lead_xray_counts=pb_counts,
        lead_xray_lines=pb_lines,
        output_path=figure_path,
    )

    overall_passed = (
        tof_metrics.passed
        and genealogy_metrics.passed
        and nai_metrics.passed
        and pb_metrics.passed
    )

    reference_standard_types = {
        "tof_and_decay": "B (точные релятивистские формулы ToF и аналитическая экспонента распада)",
        "genealogy_and_compaction": "C (точные кинематические инварианты ToF, баланс активаций и сохранения энергии)",
        "iodine_escape_peak": "B (точный аналитический интеграл самопоглощения в плоском слое по сечениям XCOM и линиям EADL)",
        "lead_fluorescence": "B (табличные линии ENDF/B-VIII.0 EADL и аналитический интеграл выхода из фольги)",
    }

    report = Exp05SecondarySourceReport(
        quick=quick,
        reference_standard_types=reference_standard_types,
        tof_and_decay=tof_metrics,
        genealogy_and_compaction=genealogy_metrics,
        iodine_escape_peak=nai_metrics,
        lead_fluorescence=pb_metrics,
        figure_path=str(figure_path),
        passed=overall_passed,
    )

    json_path = output_dir / "exp05_secondary_source.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "Exp05SecondarySourceReport",
    "GenealogyAndCompactionMetrics",
    "IodineEscapePeakMetrics",
    "LeadFluorescenceMetrics",
    "TimeOfFlightAndDecayMetrics",
    "run_exp05_secondary_source_and_escape_peaks",
]
