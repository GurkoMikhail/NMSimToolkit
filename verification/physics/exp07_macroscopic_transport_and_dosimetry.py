"""Эксперимент 7: Верификация макроскопического транспорта, альбедо и 3D-дозиметрии.

Проверяет:
1. Факторы накопления поглощенной энергии B(mu * r) для точечного изотропного источника 0.500 МэВ
   в центре большого кубического водного фантома (1800x1800x1800 мм^3) со сферическими детекторными
   слоями по радиусу r = ||x|| на оптических глубинах mu * r in [0.5, 4.0] MFP, с аналитическим
   расчетом кермы первых столкновений из первых принципов (квадратура Гаусса–Лежандра по сечениям
   Клейна–Нишины с формфактором некогерентного рассеяния S(x, Z)), явным выделением поправки на
   конечную толщину сферического слоя Delta(mu * r) = +-0.20, расчетом статистической ошибки SE
   и сверкой со стандартом ANSI/ANS-6.4.3-1991.
2. Полное и однократное числовое (A_N) и энергетическое (A_E) альбедо обратного рассеяния от пластин
   Water, Pyrex и Lead при нормальном падении пучка 140.5 кэВ в связанном фотон-атомном транспорте
   (с K/L-флуоресценцией), аналитическое альбедо однократного рассеяния (Single-Scatter Analytical
   Albedo), аналитическую оценку выхода K-флуоресценции свинца назад, физическую иерархию и коридоры
   полного альбедо A_N и A_E и положение комптоновского пика рассеяния на 180 градусов (~90.7 кэВ).
3. 100% паритет 3D дозовой карты DoseMapHandler на повернутой и смещенной сетке DoseGridNode
   с независимым матричным преобразованием мировых координат через элементарные матрицы вращения
   Эйлера R_z(alpha) R_y(beta) R_x(gamma), сохранение полной поглощенной энергии внутри фантома
   и корректность пересчета энерговыделения (МэВ) в поглощенную дозу (Гр = Дж/кг).
4. Строгий закон сохранения полной энергии E_initial == E_deposited + E_escaped + E_cutoff
   во всех трех макроскопических симуляциях.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.data.dose_map_handler import DoseMapHandler
from core.geometry.geometries import Box
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.volumes import Volume
from core.other.typing_definitions import Charge, Float, ID, Index, ParticleSpecies
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import ChunkType, RNGContext, SimulationDataBuffer
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import (
    _compute_log_energy_and_bin,
    _get_macroscopic_cross_sections,
    _locate_energy_interval,
)
from core.scene.dose_grid_node import DoseGridNode
from core.transport.propagator import ParticlePropagator
import settings.database_setting as database_setting
from verification.common.scattering_and_stats import (
    compute_klein_nishina_bound_pdf,
    compute_rayleigh_thomson_pdf,
)
from verification.common.transport_and_telemetry import (
    EnergyBalanceAudit,
    run_coupled_transport_loop,
)
from verification.physics.plotting import plot_exp07_macroscopic_transport_and_dosimetry

ELECTRON_REST_MASS_ENERGY: float = 0.51099895 * units.MeV


@dataclass(frozen=True)
class BuildupFactorMetrics:
    """Метрики верификации факторов накопления поглощенной энергии B(mu * r) в воде по ANSI/ANS-6.4.3-1991.

    Attributes:
        source_energy: Энергия точечного изотропного источника фотонов [МэВ].
        water_linear_attenuation: Линейный коэффициент ослабления воды при энергии источника [1/мм].
        mean_compton_recoil_energy: Аналитическая средняя энергия электрона отдачи в воде при E_0 [МэВ].
        phantom_half_width: Полуширина кубического водного фантома [мм].
        num_histories: Число запущенных первичных историй.
        ansi_reference_standard: Идентификатор эталонного стандарта факторов накопления.
        optical_depth_grid: Сетка оптических глубин mu * r.
        buildup_mc: Измеренные значения фактора накопления B(mu * r) относительно аналитической кермы.
        buildup_mc_se: Статистическая стандартная ошибка SE оценки B(mu * r) по дисперсии между историями.
        buildup_ansi_reference: Канонические точечные значения B(mu * r) по стандарту ANSI/ANS-6.4.3-1991.
        buildup_shell_averaged_reference: Эталонные значения B(mu * r), усредненные по толщине слоя +-0.20 MFP.
        shell_attenuation_sinh_factor: Аналитический множитель sinh(Delta) / Delta конечной толщины слоя.
        is_strictly_above_unity: Все ли значения B(mu * r) строго больше 1.0.
        is_monotonic_increasing: Монотонно ли возрастает B(mu * r) с оптической глубиной.
        max_rel_diff_ansi: Максимальное относительное отклонение от точечного эталона ANSI/ANS-6.4.3-1991.
        max_rel_diff_shell_corrected: Максимальное относительное отклонение от усредненного по слою эталона.
        energy_balance: Результат сквозного аудита баланса энергии в симуляции водного фантома.
        passed: Признак успешного прохождения критериев (< 6% отклонение от ANSI/ANS-6.4.3, монотонность, B > 1).
    """

    source_energy: float
    water_linear_attenuation: float
    mean_compton_recoil_energy: float
    phantom_half_width: float
    num_histories: int
    ansi_reference_standard: str
    optical_depth_grid: list[float]
    buildup_mc: list[float]
    buildup_mc_se: list[float]
    buildup_ansi_reference: list[float]
    buildup_shell_averaged_reference: list[float]
    shell_attenuation_sinh_factor: float
    is_strictly_above_unity: bool
    is_monotonic_increasing: bool
    max_rel_diff_ansi: float
    max_rel_diff_shell_corrected: float
    energy_balance: EnergyBalanceAudit
    passed: bool


@dataclass(frozen=True)
class BackscatterAlbedoMetrics:
    """Метрики верификации полного и однократного альбедо обратного рассеяния и комптоновского пика 180 градусов.

    Attributes:
        beam_energy: Начальная энергия узкого пучка фотонов [МэВ].
        num_histories_per_material: Число историй на каждую пластину-отражатель.
        materials: Список названий материалов ('Water', 'Pyrex', 'Lead').
        number_albedo: Полное числовое альбедо A_N по материалам (включая вторичную флуоресценцию).
        energy_albedo: Полное энергетическое альбедо A_E по материалам (включая вторичную флуоресценцию).
        single_scatter_number_albedo_mc: Числовое альбедо однократного рассеяния первичных фотонов A_N^(1) в МК.
        single_scatter_number_albedo_theory: Аналитическое числовое альбедо однократного рассеяния A_N^(1).
        single_scatter_albedo_z_scores: Биномиальные Z-score отклонения A_N^(1, MC) от аналитического решения.
        max_single_albedo_z_score: Максимальный Z-score однократного альбедо среди всех трех материалов.
        fluorescence_escaped_fractions: Доля вылетевших назад вторичных фотонов флуоресценции на 1 историю.
        lead_first_order_fluorescence_albedo_theory: Аналитическая оценка однократного выхода K-флуоресценции Pb назад.
        hierarchy_valid: Выполняется ли физическая иерархия A_N(Water) > A_N(Pyrex) > A_N(Lead).
        total_albedo_invariants_valid: Выполняются ли физические неравенства и коридоры для полных A_N и A_E.
        theoretical_backscatter_peak: Теоретическая энергия комптоновского рассеяния на 180 град [МэВ].
        measured_backscatter_peak: Положение максимума спектра однократно отраженных фотонов из воды [МэВ].
        peak_abs_error: Абсолютное отклонение положения пика обратного рассеяния [МэВ].
        energy_balances: Аудит сохранения энергии по каждому материалу.
        passed: Признак успешного выполнения критериев.
    """

    beam_energy: float
    num_histories_per_material: int
    materials: list[str]
    number_albedo: list[float]
    energy_albedo: list[float]
    single_scatter_number_albedo_mc: list[float]
    single_scatter_number_albedo_theory: list[float]
    single_scatter_albedo_z_scores: list[float]
    max_single_albedo_z_score: float
    fluorescence_escaped_fractions: list[float]
    lead_first_order_fluorescence_albedo_theory: float
    hierarchy_valid: bool
    total_albedo_invariants_valid: bool
    theoretical_backscatter_peak: float
    measured_backscatter_peak: float
    peak_abs_error: float
    energy_balances: list[EnergyBalanceAudit]
    passed: bool


@dataclass(frozen=True)
class DoseMapParityMetrics:
    """Метрики 100% паритета 3D дозовой карты DoseMapHandler на повернутой сетке и конвертации в поглощенную дозу.

    Attributes:
        num_histories: Число историй в симуляции водного фантома.
        grid_shape: Разрешение 3D сетки DoseGridNode (Nx, Ny, Nz).
        voxel_step: Шаг вокселя сетки дозы [мм].
        euler_rotation_angles: Углы Эйлера (alpha, beta, gamma) поворота фантома и сетки [рад].
        translation_offset: Вектор сдвига центра фантома и сетки [мм].
        total_energy_handler: Суммарная поглощенная энергия в 3D сетке DoseMapHandler [МэВ].
        total_energy_manual: Суммарная поглощенная энергия при независимой аналитической агрегации [МэВ].
        total_energy_rel_diff: Относительная невязка полной поглощенной энергии между обработчиком и эталоном.
        deposited_energy_conservation_rel_diff: Относительная невязка между суммой в сетке и полным E_deposited.
        max_layer_rel_diff: Максимальная относительная невязка среди всех непустых слоев Z.
        max_voxel_abs_diff: Максимальная повоксельная абсолютная невязка во всей 3D сетке [МэВ].
        voxel_mass: Масса одного вокселя жидкой воды [кг].
        mean_phantom_absorbed_dose: Средняя поглощенная доза по водному фантому [Гр = Дж/кг].
        max_voxel_absorbed_dose: Максимальная поглощенная доза в вокселе [Гр = Дж/кг].
        dose_conversion_rel_diff: Относительная невязка пересчета энерговыделения в поглощенную дозу (Гр).
        energy_balance: Аудит закона сохранения энергии в симуляции дозовой карты.
        passed: Признак выполнения критериев паритета, сохранения энергии и конвертации в Гр.
    """

    num_histories: int
    grid_shape: list[int]
    voxel_step: float
    euler_rotation_angles: list[float]
    translation_offset: list[float]
    total_energy_handler: float
    total_energy_manual: float
    total_energy_rel_diff: float
    deposited_energy_conservation_rel_diff: float
    max_layer_rel_diff: float
    max_voxel_abs_diff: float
    voxel_mass: float
    mean_phantom_absorbed_dose: float
    max_voxel_absorbed_dose: float
    dose_conversion_rel_diff: float
    energy_balance: EnergyBalanceAudit
    passed: bool


@dataclass(frozen=True)
class Exp07MacroscopicDosimetryReport:
    """Корневой отчет верификации макроскопического транспорта, альбедо и дозиметрии (Эксперимент 7).

    Attributes:
        experiment_id: Идентификатор эксперимента ('exp07_macroscopic_transport_and_dosimetry').
        title: Название эксперимента.
        quick_mode: Признак быстрого режима прогона.
        reference_standard_types: Классификация типов используемых эталонов по каждой проверке.
        buildup_metrics: Результаты проверки факторов накопления B(mu * r) в воде.
        albedo_metrics: Результаты проверки альбедо обратного рассеяния и пика 180 град.
        dose_parity_metrics: Результаты проверки 100% паритета DoseMapHandler и конвертации дозы.
        passed: Итоговый статус верификации.
    """

    experiment_id: str
    title: str
    quick_mode: bool
    reference_standard_types: dict[str, str]
    buildup_metrics: BuildupFactorMetrics
    albedo_metrics: BackscatterAlbedoMetrics
    dose_parity_metrics: DoseMapParityMetrics
    passed: bool


def _extract_macroscopic_lacs_from_buffer(
    mat_id: Index,
    energy: Float,
    physics_buffer: PhysicsBuffer,
) -> tuple[float, float, float, float]:
    """Извлекает макроскопические сечения (pe, compton, coherent, total) напрямую из MaterialBank."""
    out_lacs = np.zeros(3, dtype=Float)
    log_e, log_bin = _compute_log_energy_and_bin(energy, physics_buffer.material_bank)
    _get_macroscopic_cross_sections(
        energy,
        log_e,
        log_bin,
        mat_id,
        physics_buffer.material_bank,
        out_lacs,
    )
    mu_pe = float(out_lacs[0])
    mu_comp = float(out_lacs[1])
    mu_coh = float(out_lacs[2])
    mu_tot = mu_pe + mu_comp + mu_coh
    return mu_pe, mu_comp, mu_coh, mu_tot


def _extract_element_weights_from_buffer(
    mat_id: Index,
    process_col_idx: Index,
    energy: Float,
    physics_buffer: PhysicsBuffer,
) -> list[tuple[Charge, float]]:
    """Извлекает интерполированные парциальные веса элементов из ElementCSR в точности как _sample_Z_scalar."""
    start = int(physics_buffer.element_csr.element_offsets[mat_id])
    end = int(physics_buffer.element_csr.element_offsets[mat_id + 1])
    count = end - start
    if count <= 0:
        return []
    if count == 1:
        return [(Charge(physics_buffer.element_csr.element_Z[start]), 1.0)]

    row_idx, fraction = _locate_energy_interval(energy, mat_id, physics_buffer.material_bank)
    row = int(row_idx)
    frac = float(fraction)
    partial_fractions = physics_buffer.element_csr.element_partial_fractions
    num_rows = int(partial_fractions.shape[0])
    has_next_row = frac > 0.0 and (row + 1) < num_rows

    result: list[tuple[Charge, float]] = []
    cum_prob = 0.0
    for elem_offset in range(count - 1):
        weight_low = float(partial_fractions[row, process_col_idx, elem_offset])
        if has_next_row:
            weight_high = float(partial_fractions[row + 1, process_col_idx, elem_offset])
            weight = weight_low + frac * (weight_high - weight_low)
        else:
            weight = weight_low
        cum_prob += weight
        result.append((Charge(physics_buffer.element_csr.element_Z[start + elem_offset]), weight))
    last_weight = max(0.0, 1.0 - cum_prob)
    result.append((Charge(physics_buffer.element_csr.element_Z[end - 1]), last_weight))
    return result


def _compute_compound_normalized_angular_pdf(
    query_cos: NDArray[np.float64],
    photon_energy: Float,
    element_weights: list[tuple[Charge, float]],
    is_compton: bool,
) -> NDArray[np.float64]:
    """Вычисляет нормированную на [-1, 1] угловую плотность p(cos(theta)) для составного материала."""
    compound_pdf = np.zeros_like(query_cos, dtype=np.float64)
    energy_val = float(photon_energy)

    for atomic_z, elem_weight in element_weights:
        if elem_weight <= 0.0:
            continue
        if is_compton:
            elem_pdf = compute_klein_nishina_bound_pdf(
                cos_grid=query_cos,
                energy=energy_val,
                atomic_number=int(atomic_z),
            )
        else:
            elem_pdf = compute_rayleigh_thomson_pdf(
                cos_grid=query_cos,
                energy=energy_val,
                atomic_number=int(atomic_z),
            )
        compound_pdf += elem_weight * elem_pdf

    return compound_pdf


def _compute_mean_compton_recoil_energy(
    mat_id: Index,
    photon_energy: Float,
    physics_buffer: PhysicsBuffer,
) -> float:
    """Вычисляет среднюю энергию электрона отдачи <Delta E_comp(E_0)> 64-точечной квадратурой Гаусса–Лежандра."""
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    compton_elements = _extract_element_weights_from_buffer(
        mat_id=mat_id,
        process_col_idx=Index(1),
        energy=photon_energy,
        physics_buffer=physics_buffer,
    )
    pdf_compton = _compute_compound_normalized_angular_pdf(
        query_cos=gl_nodes,
        photon_energy=photon_energy,
        element_weights=compton_elements,
        is_compton=True,
    )
    energy_val = float(photon_energy)
    reduced_energy = energy_val / ELECTRON_REST_MASS_ENERGY
    recoil_energies = energy_val * (
        1.0 - 1.0 / (1.0 + reduced_energy * (1.0 - gl_nodes))
    )
    return float(np.sum(gl_weights * recoil_energies * pdf_compton))


def _compute_single_scatter_analytical_albedo(
    mat_id: Index,
    beam_energy: Float,
    slab_thickness: float,
    physics_buffer: PhysicsBuffer,
) -> float:
    """Вычисляет аналитическое числовое альбедо однократного рассеяния (Single-Scatter Analytical Albedo) A_N^(1)."""
    _, mu_comp_0, mu_coh_0, mu_tot_0 = _extract_macroscopic_lacs_from_buffer(
        mat_id=mat_id,
        energy=beam_energy,
        physics_buffer=physics_buffer,
    )
    if mu_tot_0 <= 0.0:
        return 0.0

    # 64-точечная квадратура Гаусса–Лежандра на задней полусфере cos(theta) in [-1, 0)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    cos_back = 0.5 * (gl_nodes - 1.0)
    weights_back = 0.5 * gl_weights
    u_cos = -cos_back

    compton_elements = _extract_element_weights_from_buffer(
        mat_id=mat_id,
        process_col_idx=Index(1),
        energy=beam_energy,
        physics_buffer=physics_buffer,
    )
    coherent_elements = _extract_element_weights_from_buffer(
        mat_id=mat_id,
        process_col_idx=Index(2),
        energy=beam_energy,
        physics_buffer=physics_buffer,
    )

    pdf_comp_back = _compute_compound_normalized_angular_pdf(
        query_cos=cos_back,
        photon_energy=beam_energy,
        element_weights=compton_elements,
        is_compton=True,
    )
    pdf_coh_back = _compute_compound_normalized_angular_pdf(
        query_cos=cos_back,
        photon_energy=beam_energy,
        element_weights=coherent_elements,
        is_compton=False,
    )

    energy_0 = float(beam_energy)
    reduced_energy = energy_0 / ELECTRON_REST_MASS_ENERGY

    comp_prob = mu_comp_0 / mu_tot_0
    coh_prob = mu_coh_0 / mu_tot_0

    integrand = np.zeros_like(cos_back, dtype=np.float64)
    for pt_idx in range(cos_back.shape[0]):
        cos_val = float(cos_back[pt_idx])
        u_val = float(u_cos[pt_idx])

        # Комптоновская ветвь с изменением энергии фотона E'(theta)
        scattered_energy = Float(energy_0 / (1.0 + reduced_energy * (1.0 - cos_val)))
        _, _, _, mu_tot_scat = _extract_macroscopic_lacs_from_buffer(
            mat_id=mat_id,
            energy=scattered_energy,
            physics_buffer=physics_buffer,
        )
        mu_ratio_comp = mu_tot_scat / mu_tot_0
        escape_comp = (u_val / (u_val + mu_ratio_comp)) * (
            1.0 - math.exp(-mu_tot_0 * slab_thickness * (1.0 + mu_ratio_comp / u_val))
        )

        # Когерентная ветвь без изменения энергии (mu_tot(E') == mu_tot(E_0))
        escape_coh = (u_val / (u_val + 1.0)) * (
            1.0 - math.exp(-mu_tot_0 * slab_thickness * (1.0 + 1.0 / u_val))
        )

        integrand[pt_idx] = (
            comp_prob * float(pdf_comp_back[pt_idx]) * escape_comp
            + coh_prob * float(pdf_coh_back[pt_idx]) * escape_coh
        )

    return float(np.sum(weights_back * integrand))


def _compute_first_order_fluorescence_albedo_theory(
    mat_id: Index,
    atomic_z: int,
    beam_energy: Float,
    slab_thickness: float,
    physics_buffer: PhysicsBuffer,
) -> float:
    """Вычисляет аналитическую оценку однократного выхода K-флуоресценции назад из пластины."""
    mu_pe_0, _, _, mu_tot_0 = _extract_macroscopic_lacs_from_buffer(
        mat_id=mat_id,
        energy=beam_energy,
        physics_buffer=physics_buffer,
    )
    if mu_tot_0 <= 0.0 or mu_pe_0 <= 0.0:
        return 0.0

    relax_bank = physics_buffer.relaxation_bank
    shell_start = int(relax_bank.shell_offsets[atomic_z])
    shell_end = int(relax_bank.shell_offsets[atomic_z + 1])
    if shell_end <= shell_start:
        return 0.0

    # Находим K-оболочку (subshell_designator == 1) и ее долю ионизации при E_0
    k_shell_idx = -1
    total_open_jump_weight = 0.0
    k_shell_jump_weight = 0.0
    prev_cum_weight = 0.0
    for s_idx in range(shell_start, shell_end):
        cum_weight = float(relax_bank.jump_cumulative_weight[s_idx])
        shell_weight = max(0.0, cum_weight - prev_cum_weight)
        prev_cum_weight = cum_weight
        if float(relax_bank.binding_energy[s_idx]) <= float(beam_energy):
            total_open_jump_weight += shell_weight
            if int(relax_bank.subshell_designator[s_idx]) == 1:
                k_shell_idx = s_idx
                k_shell_jump_weight = shell_weight

    if k_shell_idx < 0 or total_open_jump_weight <= 0.0:
        return 0.0

    k_shell_prob = k_shell_jump_weight / total_open_jump_weight
    fluor_yield_k = float(relax_bank.fluorescence_yield[k_shell_idx])

    line_start = int(relax_bank.line_offsets[k_shell_idx])
    line_end = int(relax_bank.line_offsets[k_shell_idx + 1])
    num_lines = line_end - line_start
    if num_lines <= 0:
        return 0.0

    # Восстанавливаем вероятности линий K-серии из таблиц Воза
    reconstructed = np.array(
        relax_bank.line_prob[line_start:line_end], dtype=np.float64
    )
    for local_i in range(num_lines):
        alias_target = int(relax_bank.line_alias[line_start + local_i])
        reconstructed[alias_target] += 1.0 - float(
            relax_bank.line_prob[line_start + local_i]
        )
    line_probs = reconstructed / float(num_lines)

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(32)
    u_nodes = 0.5 * (gl_nodes + 1.0)
    u_weights = 0.5 * gl_weights

    escape_sum = 0.0
    for local_i in range(num_lines):
        line_ene = Float(relax_bank.line_energy[line_start + local_i])
        _, _, _, mu_line = _extract_macroscopic_lacs_from_buffer(
            mat_id=mat_id,
            energy=line_ene,
            physics_buffer=physics_buffer,
        )
        mu_ratio = mu_line / mu_tot_0
        integrand = (u_nodes / (u_nodes + mu_ratio)) * (
            1.0 - np.exp(-mu_tot_0 * slab_thickness * (1.0 + mu_ratio / u_nodes))
        )
        backward_escape_prob = 0.5 * float(np.sum(u_weights * integrand))
        escape_sum += float(line_probs[local_i]) * backward_escape_prob

    return (mu_pe_0 / mu_tot_0) * k_shell_prob * fluor_yield_k * escape_sum


# Канонические эталонные значения фактора накопления поглощенной энергии B(mu * r) в воде
# при E_0 = 0.500 МэВ (точечный изотропный источник в бесконечной гомогенной водной среде):
# - Источник: стандарт ANSI/ANS-6.4.3-1991 ("Gamma-Ray Attenuation Coefficients and Buildup
#   Factors for Engineering Materials"), Table 6.4.3 (Water, Energy Absorption Buildup Factor).
# - Узлы mu * r = 1.0, 2.0, 3.0, 4.0 MFP (2.46, 4.93, 8.35, 12.80) взяты напрямую из таблицы
#   стандарта ANSI/ANS-6.4.3-1991; дробные узлы mu * r = 0.5, 1.5, 2.5, 3.5 MFP (1.60, 3.57,
#   6.53, 10.45) вычислены по канонической параметризации геометрической прогрессии Харимы
#   (G-P fitting parameters, Harima et al., Nucl. Sci. Eng. 94, 24-35, 1986), входящей в стандарт.
ANSI_ANS_643_WATER_OPTICAL_DEPTHS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ANSI_ANS_643_WATER_EABF_REFERENCE: tuple[float, ...] = (
    1.60,
    2.46,
    3.57,
    4.93,
    6.53,
    8.35,
    10.45,
    12.80,
)


def _compute_shell_averaged_ansi_reference(
    optical_depth_centers: NDArray[np.float64],
    point_reference: NDArray[np.float64],
    half_shell_optical_depth: float,
) -> tuple[NDArray[np.float64], float]:
    """Вычисляет аналитическую поправку на конечную толщину сферического слоя Delta(mu * r).

    В сферическом слое [x_k - Delta, x_k + Delta] нерассеянное энерговыделение пропорционально
    int_{x_k - Delta}^{x_k + Delta} exp(-x) dx = 2 * Delta * exp(-x_k) * (sinh(Delta) / Delta),
    а полное энерговыделение пропорционально int_{x_k - Delta}^{x_k + Delta} B(x) exp(-x) dx.
    """
    sinh_factor = math.sinh(half_shell_optical_depth) / half_shell_optical_depth
    # Аппроксимируем гладкую кривую B(x) полиномом 3-й степени по точным узлам ANSI/ANS-6.4.3-1991
    poly_coeffs = np.polyfit(optical_depth_centers, point_reference, deg=3)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(16)

    shell_averaged = np.empty_like(point_reference, dtype=np.float64)
    denom = 2.0 * math.sinh(half_shell_optical_depth)
    for idx, center_x in enumerate(optical_depth_centers):
        local_offsets = half_shell_optical_depth * gl_nodes
        query_x = float(center_x) + local_offsets
        b_vals = np.polyval(poly_coeffs, query_x)
        weights_exp = np.exp(-local_offsets)
        numerator = half_shell_optical_depth * float(
            np.sum(gl_weights * b_vals * weights_exp)
        )
        shell_averaged[idx] = numerator / denom

    return shell_averaged, sinh_factor


def _sample_isotropic_directions(
    num_particles: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Генерирует изотропные единичные векторы направлений на сфере 4*pi."""
    cos_theta = rng.uniform(-1.0, 1.0, size=num_particles)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
    phi_angle = rng.uniform(0.0, 2.0 * math.pi, size=num_particles)
    dir_x = sin_theta * np.cos(phi_angle)
    dir_y = sin_theta * np.sin(phi_angle)
    dir_z = cos_theta
    return dir_x, dir_y, dir_z


def _run_water_buildup_factor_audit(
    quick: bool,
) -> tuple[BuildupFactorMetrics, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Вычисляет факторы накопления поглощенной энергии B(mu * r) в воде и сверяет с ANSI/ANS-6.4.3-1991."""
    num_histories = 120_000 if quick else 350_000
    source_energy = Float(0.500 * units.MeV)
    # Кубический водный фантом с полушириной 900 мм (~8.7 MFP в воде при 0.5 МэВ)
    # обеспечивает > 4.5 MFP отражателя за сферическим слоем mu * r = 4.0.
    phantom_half_width = 900.0

    vacuum_mat = database_setting.material_database["Vacuum"]
    water_mat = database_setting.material_database["Water, Liquid"]

    world = Volume(
        geometry=Box(Float(2400.0), Float(2400.0), Float(2400.0)),
        material=vacuum_mat,
        name="World_Buildup",
    )
    water_cube = Volume(
        geometry=Box(
            Float(2.0 * phantom_half_width),
            Float(2.0 * phantom_half_width),
            Float(2.0 * phantom_half_width),
        ),
        material=water_mat,
        name="WaterPhantom_Buildup",
    )
    world.add_child(water_cube)

    rng = np.random.default_rng(70101)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    propagator = ParticlePropagator(rng=rng)

    geometry_buffer = GeometryCompiler().compile_scene(world)
    physics_buffer = PhysicsCompiler().compile_scene(world, propagator.processes)

    water_mat_id = Index(water_mat.ID)
    mu_pe, mu_comp, _, mu_water = _extract_macroscopic_lacs_from_buffer(
        mat_id=water_mat_id,
        energy=source_energy,
        physics_buffer=physics_buffer,
    )
    mean_compton_recoil = _compute_mean_compton_recoil_energy(
        mat_id=water_mat_id,
        photon_energy=source_energy,
        physics_buffer=physics_buffer,
    )

    optical_depth_centers = np.array(ANSI_ANS_643_WATER_OPTICAL_DEPTHS, dtype=np.float64)
    half_shell_optical_depth = 0.20
    shell_r_min = (optical_depth_centers - half_shell_optical_depth) / mu_water
    shell_r_max = (optical_depth_centers + half_shell_optical_depth) / mu_water
    num_shells = len(optical_depth_centers)

    # Аналитическое энерговыделение нерассеянных (первых) столкновений в сферических слоях из первых принципов
    mean_edep_per_first_collision = (
        mu_pe * float(source_energy) + mu_comp * mean_compton_recoil
    ) / mu_water
    uncollided_shell_edep_theory = (
        float(num_histories)
        * mean_edep_per_first_collision
        * (np.exp(-mu_water * shell_r_min) - np.exp(-mu_water * shell_r_max))
    )

    total_shell_edep = np.zeros(num_shells, dtype=np.float64)
    total_shell_edep_sq = np.zeros(num_shells, dtype=np.float64)

    batch_size = min(40_000, num_histories)
    bank = ParticleBank.allocate(batch_size * 2)
    data_buffer = SimulationDataBuffer.allocate_for(
        capacity=batch_size * 4,
        enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
    )

    min_energy = Float(1.0 * units.keV)
    batch_history_shell_edep = np.zeros((batch_size, num_shells), dtype=np.float64)
    parent_by_pid = np.full(batch_size * 8, -1, dtype=np.int64)
    current_batch_count = 0

    def _init_buildup_batch(
        target_bank: ParticleBank,
        cur_batch: int,
        _processed: int,
    ) -> float:
        nonlocal current_batch_count
        if current_batch_count > 0:
            total_shell_edep_sq[:] += np.sum(
                batch_history_shell_edep[:current_batch_count, :] ** 2, axis=0
            )
        current_batch_count = cur_batch
        dir_x, dir_y, dir_z = _sample_isotropic_directions(cur_batch, rng)

        target_bank.state.is_active[:cur_batch] = True
        target_bank.state.position.x[:cur_batch] = Float(0.0)
        target_bank.state.position.y[:cur_batch] = Float(0.0)
        target_bank.state.position.z[:cur_batch] = Float(0.0)

        target_bank.state.direction.x[:cur_batch] = dir_x
        target_bank.state.direction.y[:cur_batch] = dir_y
        target_bank.state.direction.z[:cur_batch] = dir_z

        target_bank.state.energy[:cur_batch] = source_energy
        target_bank.state.distance_traveled[:cur_batch] = Float(0.0)
        target_bank.state.species[:cur_batch] = Index(ParticleSpecies.PHOTON)

        target_bank.navigation_state.current_volume[:cur_batch] = Index(-1)
        target_bank.navigation_state.boundary_distance[:cur_batch] = Float(0.0)

        particle_ids = np.arange(cur_batch, dtype=ID)
        target_bank.initial_state.ID[:cur_batch] = particle_ids
        target_bank.initial_state.parent_ID[:cur_batch] = ID(-1)
        target_bank.initial_state.emission_time[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_energy[:cur_batch] = source_energy
        target_bank.initial_state.emission_position.x[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_position.y[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_position.z[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_direction.x[:cur_batch] = dir_x
        target_bank.initial_state.emission_direction.y[:cur_batch] = dir_y
        target_bank.initial_state.emission_direction.z[:cur_batch] = dir_z

        batch_history_shell_edep[:cur_batch, :] = 0.0
        parent_by_pid[:] = -1
        return float(cur_batch) * float(source_energy)

    def _on_buildup_interactions(
        step_buffer: SimulationDataBuffer,
        step_bank: ParticleBank,
    ) -> None:
        num_events = step_buffer.interactions.cursor_value
        if num_events <= 0:
            return
        active_slots = step_bank.active_indices
        if active_slots.size > 0:
            slot_ids = step_bank.initial_state.ID[active_slots].astype(np.int64)
            slot_parents = step_bank.initial_state.parent_ID[active_slots].astype(np.int64)
            parent_by_pid[slot_ids] = slot_parents

        inter_buf = step_buffer.interactions
        ev_x = inter_buf.position.x[:num_events]
        ev_y = inter_buf.position.y[:num_events]
        ev_z = inter_buf.position.z[:num_events]
        ev_edep = inter_buf.energy_deposit[:num_events].astype(np.float64)
        ev_pids = inter_buf.particle_ID[:num_events].astype(np.int64)
        root_ids = ev_pids.copy()
        for ev_idx in range(root_ids.size):
            curr_pid = int(root_ids[ev_idx])
            while curr_pid >= 0 and parent_by_pid[curr_pid] >= 0:
                curr_pid = int(parent_by_pid[curr_pid])
            root_ids[ev_idx] = curr_pid

        radii = np.sqrt(ev_x * ev_x + ev_y * ev_y + ev_z * ev_z)
        for shell_idx in range(num_shells):
            in_shell = (radii >= shell_r_min[shell_idx]) & (radii < shell_r_max[shell_idx])
            if np.any(in_shell):
                shell_weights = ev_edep[in_shell]
                shell_roots = root_ids[in_shell]
                total_shell_edep[shell_idx] += float(np.sum(shell_weights))
                np.add.at(
                    batch_history_shell_edep[:current_batch_count, shell_idx],
                    shell_roots,
                    shell_weights,
                )

    def _buildup_escape_mask(
        step_bank: ParticleBank,
        active_indices: NDArray[Index],
    ) -> NDArray[np.bool_]:
        act_x = np.abs(step_bank.state.position.x[active_indices])
        act_y = np.abs(step_bank.state.position.y[active_indices])
        act_z = np.abs(step_bank.state.position.z[active_indices])
        outside_cube = (
            (act_x > phantom_half_width)
            | (act_y > phantom_half_width)
            | (act_z > phantom_half_width)
        )
        return outside_cube

    energy_balance = run_coupled_transport_loop(
        num_histories=num_histories,
        batch_size=batch_size,
        bank=bank,
        data_buffer=data_buffer,
        geometry_buffer=geometry_buffer,
        physics_buffer=physics_buffer,
        propagator=propagator,
        rng_ctx=rng_ctx,
        min_energy=min_energy,
        initialize_primary_batch=_init_buildup_batch,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(1.0 * units.keV),
        electron_production_cut=Float(np.inf),
        on_interactions_step=_on_buildup_interactions,
        evaluate_custom_escape_mask=_buildup_escape_mask,
    )
    if current_batch_count > 0:
        total_shell_edep_sq[:] += np.sum(
            batch_history_shell_edep[:current_batch_count, :] ** 2, axis=0
        )

    buildup_mc = total_shell_edep / uncollided_shell_edep_theory

    # Статистическая стандартная ошибка SE по дисперсии энерговыделения между первичными историями
    hist_float = float(num_histories)
    sample_var_per_history = np.maximum(
        (total_shell_edep_sq - (total_shell_edep**2) / hist_float) / max(hist_float - 1.0, 1.0),
        0.0,
    )
    buildup_mc_se = (
        math.sqrt(hist_float) * np.sqrt(sample_var_per_history) / uncollided_shell_edep_theory
    )

    buildup_ansi_reference = np.array(ANSI_ANS_643_WATER_EABF_REFERENCE, dtype=np.float64)
    buildup_shell_ref, sinh_factor = _compute_shell_averaged_ansi_reference(
        optical_depth_centers=optical_depth_centers,
        point_reference=buildup_ansi_reference,
        half_shell_optical_depth=half_shell_optical_depth,
    )

    rel_diffs = np.abs(buildup_mc - buildup_ansi_reference) / buildup_ansi_reference
    max_rel_diff = float(np.max(rel_diffs))

    rel_diffs_shell = np.abs(buildup_mc - buildup_shell_ref) / buildup_shell_ref
    max_rel_diff_shell = float(np.max(rel_diffs_shell))

    strictly_above_one = bool(np.all(buildup_mc > 1.0))
    monotonic = bool(np.all(np.diff(buildup_mc) > 0.0))

    passed = (
        strictly_above_one
        and monotonic
        and (max_rel_diff < 0.06)
        and (max_rel_diff_shell < 0.06)
        and energy_balance.passed
    )

    metrics = BuildupFactorMetrics(
        source_energy=float(source_energy),
        water_linear_attenuation=mu_water,
        mean_compton_recoil_energy=mean_compton_recoil,
        phantom_half_width=phantom_half_width,
        num_histories=num_histories,
        ansi_reference_standard="ANSI/ANS-6.4.3-1991",
        optical_depth_grid=[float(val) for val in optical_depth_centers],
        buildup_mc=[float(val) for val in buildup_mc],
        buildup_mc_se=[float(val) for val in buildup_mc_se],
        buildup_ansi_reference=[float(val) for val in buildup_ansi_reference],
        buildup_shell_averaged_reference=[float(val) for val in buildup_shell_ref],
        shell_attenuation_sinh_factor=sinh_factor,
        is_strictly_above_unity=strictly_above_one,
        is_monotonic_increasing=monotonic,
        max_rel_diff_ansi=max_rel_diff,
        max_rel_diff_shell_corrected=max_rel_diff_shell,
        energy_balance=energy_balance,
        passed=passed,
    )
    return metrics, optical_depth_centers, buildup_mc, buildup_ansi_reference


def _run_single_slab_albedo_simulation(
    material_obj: Any,
    num_histories: int,
    beam_energy: Float,
    seed: int,
    record_single_scatter_spectrum: bool = False,
) -> tuple[float, float, float, float, float, float, NDArray[np.float64], EnergyBalanceAudit]:
    """Рассчитывает полное и однократное альбедо полубесконечной пластины в связанном транспорте."""
    vacuum_mat = database_setting.material_database["Vacuum"]
    world = Volume(
        geometry=Box(Float(1000.0), Float(1000.0), Float(1000.0)),
        material=vacuum_mat,
        name=f"World_Albedo_{material_obj.name}",
    )
    slab_thickness = 300.0
    slab_vol = Volume(
        geometry=Box(Float(800.0), Float(800.0), Float(slab_thickness)),
        material=material_obj,
        name="SemiInfiniteSlab",
    )
    slab_vol.translate(z=Float(0.5 * slab_thickness))
    world.add_child(slab_vol)

    rng = np.random.default_rng(seed)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    propagator = ParticlePropagator(rng=rng)

    geometry_buffer = GeometryCompiler().compile_scene(world)
    physics_buffer = PhysicsCompiler().compile_scene(world, propagator.processes)

    single_albedo_theory = _compute_single_scatter_analytical_albedo(
        mat_id=Index(material_obj.ID),
        beam_energy=beam_energy,
        slab_thickness=slab_thickness,
        physics_buffer=physics_buffer,
    )
    lead_fluor_albedo_theory = (
        _compute_first_order_fluorescence_albedo_theory(
            mat_id=Index(material_obj.ID),
            atomic_z=82,
            beam_energy=beam_energy,
            slab_thickness=slab_thickness,
            physics_buffer=physics_buffer,
        )
        if material_obj.name == "Pb"
        else 0.0
    )

    batch_size = min(40_000, num_histories)
    bank = ParticleBank.allocate(batch_size * 2)
    data_buffer = SimulationDataBuffer.allocate_for(
        capacity=batch_size * 4,
        enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
    )

    min_energy = Float(1.0 * units.keV)
    backscattered_count = 0
    backscattered_energy_sum = 0.0
    single_scatter_count = 0
    fluorescence_escaped_count = 0
    single_scatter_energies: list[NDArray[np.float64]] = []
    collision_counts = np.zeros(batch_size * 4, dtype=np.int32)

    def _init_albedo_batch(
        target_bank: ParticleBank,
        cur_batch: int,
        _processed: int,
    ) -> float:
        target_bank.state.is_active[:cur_batch] = True

        target_bank.state.position.x[:cur_batch] = Float(0.0)
        target_bank.state.position.y[:cur_batch] = Float(0.0)
        target_bank.state.position.z[:cur_batch] = Float(-0.5)

        target_bank.state.direction.x[:cur_batch] = Float(0.0)
        target_bank.state.direction.y[:cur_batch] = Float(0.0)
        target_bank.state.direction.z[:cur_batch] = Float(1.0)

        target_bank.state.energy[:cur_batch] = beam_energy
        target_bank.state.distance_traveled[:cur_batch] = Float(0.0)
        target_bank.state.species[:cur_batch] = Index(ParticleSpecies.PHOTON)

        target_bank.navigation_state.current_volume[:cur_batch] = Index(-1)
        target_bank.navigation_state.boundary_distance[:cur_batch] = Float(0.0)

        particle_ids = np.arange(cur_batch, dtype=ID)
        target_bank.initial_state.ID[:cur_batch] = particle_ids
        target_bank.initial_state.parent_ID[:cur_batch] = ID(-1)
        target_bank.initial_state.emission_time[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_energy[:cur_batch] = beam_energy
        target_bank.initial_state.emission_position.x[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_position.y[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_position.z[:cur_batch] = Float(-0.5)
        target_bank.initial_state.emission_direction.x[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_direction.y[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_direction.z[:cur_batch] = Float(1.0)

        collision_counts[:] = 0
        return float(cur_batch) * float(beam_energy)

    def _on_albedo_interactions(
        step_buffer: SimulationDataBuffer,
        _step_bank: ParticleBank,
    ) -> None:
        num_events = step_buffer.interactions.cursor_value
        if num_events > 0:
            ev_pids = step_buffer.interactions.particle_ID[:num_events].astype(np.int64)
            np.add.at(collision_counts, ev_pids, 1)

    def _albedo_escape_mask(
        step_bank: ParticleBank,
        active_indices: NDArray[Index],
    ) -> NDArray[np.bool_]:
        act_z = step_bank.state.position.z[active_indices]
        act_dz = step_bank.state.direction.z[active_indices]
        reflected = (act_z < 0.0) & (act_dz < 0.0)
        transmitted = (act_z > slab_thickness) & (act_dz > 0.0)
        return reflected | transmitted

    def _on_albedo_stopping(
        step_bank: ParticleBank,
        active_indices: NDArray[Index],
        escaped_mask: NDArray[np.bool_],
        _below_cutoff: NDArray[np.bool_],
    ) -> None:
        nonlocal backscattered_count, backscattered_energy_sum
        nonlocal single_scatter_count, fluorescence_escaped_count
        act_z = step_bank.state.position.z[active_indices]
        act_dz = step_bank.state.direction.z[active_indices]
        reflected_mask = escaped_mask & (act_z < 0.0) & (act_dz < 0.0)
        if not np.any(reflected_mask):
            return

        refl_indices = active_indices[reflected_mask]
        refl_energies = step_bank.state.energy[refl_indices]
        refl_pids = step_bank.initial_state.ID[refl_indices].astype(np.int64)
        refl_parents = step_bank.initial_state.parent_ID[refl_indices].astype(np.int64)

        backscattered_count += int(refl_indices.size)
        backscattered_energy_sum += float(np.sum(refl_energies))

        is_secondary = refl_parents >= 0
        fluorescence_escaped_count += int(np.sum(is_secondary))

        # В однократное рассеяние отбираем только первичные частицы (parent_ID == -1) с 1 столкновением
        single_primary_mask = (~is_secondary) & (collision_counts[refl_pids] == 1)
        if np.any(single_primary_mask):
            single_scatter_count += int(np.sum(single_primary_mask))
            if record_single_scatter_spectrum:
                single_scatter_energies.append(refl_energies[single_primary_mask].copy())

    energy_balance = run_coupled_transport_loop(
        num_histories=num_histories,
        batch_size=batch_size,
        bank=bank,
        data_buffer=data_buffer,
        geometry_buffer=geometry_buffer,
        physics_buffer=physics_buffer,
        propagator=propagator,
        rng_ctx=rng_ctx,
        min_energy=min_energy,
        initialize_primary_batch=_init_albedo_batch,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(1.0 * units.keV),
        electron_production_cut=Float(np.inf),
        on_interactions_step=_on_albedo_interactions,
        evaluate_custom_escape_mask=_albedo_escape_mask,
        on_particles_stopping=_on_albedo_stopping,
    )

    number_albedo = float(backscattered_count) / float(num_histories)
    energy_albedo = backscattered_energy_sum / (float(num_histories) * float(beam_energy))
    single_albedo_mc = float(single_scatter_count) / float(num_histories)
    fluor_fraction = float(fluorescence_escaped_count) / float(num_histories)
    collected_energies = (
        np.concatenate(single_scatter_energies)
        if single_scatter_energies
        else np.empty(0, dtype=np.float64)
    )
    return (
        number_albedo,
        energy_albedo,
        single_albedo_mc,
        single_albedo_theory,
        fluor_fraction,
        lead_fluor_albedo_theory,
        collected_energies,
        energy_balance,
    )


def _run_backscatter_albedo_audit(
    quick: bool,
) -> tuple[BackscatterAlbedoMetrics, NDArray[np.float64], NDArray[np.float64], float]:
    """Проверяет полное и однократное альбедо для Water, Pyrex, Lead и комптоновский пик 180 градусов."""
    num_histories = 60_000 if quick else 200_000
    beam_energy = Float(140.5 * units.keV)

    material_specs = [
        ("Water", database_setting.material_database["Water, Liquid"]),
        ("Pyrex", database_setting.material_database["Glass, Borosilicate (Pyrex)"]),
        ("Lead", database_setting.material_database["Pb"]),
    ]

    albedo_labels: list[str] = []
    number_albedos: list[float] = []
    energy_albedos: list[float] = []
    single_albedos_mc: list[float] = []
    single_albedos_theory: list[float] = []
    single_z_scores: list[float] = []
    fluor_fractions: list[float] = []
    energy_balances: list[EnergyBalanceAudit] = []
    lead_fluor_theory = 0.0
    water_single_scatter = np.empty(0, dtype=np.float64)

    for mat_idx, (short_label, mat_obj) in enumerate(material_specs):
        (
            num_alb,
            ene_alb,
            single_mc,
            single_th,
            fluor_frac,
            pb_fl_th,
            single_energies,
            mat_balance,
        ) = _run_single_slab_albedo_simulation(
            material_obj=mat_obj,
            num_histories=num_histories,
            beam_energy=beam_energy,
            seed=70201 + mat_idx * 19,
            record_single_scatter_spectrum=(short_label == "Water"),
        )
        std_err_single = math.sqrt(single_th * (1.0 - single_th) / float(num_histories))
        z_single = abs(single_mc - single_th) / max(std_err_single, 1e-15)

        albedo_labels.append(short_label)
        number_albedos.append(num_alb)
        energy_albedos.append(ene_alb)
        single_albedos_mc.append(single_mc)
        single_albedos_theory.append(single_th)
        single_z_scores.append(z_single)
        fluor_fractions.append(fluor_frac)
        energy_balances.append(mat_balance)
        if short_label == "Lead":
            lead_fluor_theory = pb_fl_th
        if short_label == "Water":
            water_single_scatter = single_energies

    max_single_z = max(single_z_scores)

    # Теоретическая энергия комптоновского рассеяния на 180 градусов
    theory_peak_energy = float(beam_energy) / (
        1.0 + 2.0 * float(beam_energy) / ELECTRON_REST_MASS_ENERGY
    )

    spec_edges = np.linspace(0.080, 0.140, 61, dtype=np.float64)
    spec_centers = 0.5 * (spec_edges[:-1] + spec_edges[1:])
    spec_counts, _ = np.histogram(water_single_scatter, bins=spec_edges)
    spec_counts_float = spec_counts.astype(np.float64)

    peak_bin_idx = int(np.argmax(spec_counts_float))
    measured_peak_energy = float(spec_centers[peak_bin_idx])
    peak_abs_err = abs(measured_peak_energy - theory_peak_energy)

    hierarchy_ok = (
        number_albedos[0] > number_albedos[1] > number_albedos[2]
        and energy_albedos[0] > energy_albedos[1] > energy_albedos[2]
    )

    # Проверка физических инвариантов и эталонных коридоров для полных A_N и A_E:
    # 1) Для каждого материала 0 < A_E < A_N < 1 и A_N > A_N^(1)
    # 2) В свинце при 140.5 кэВ (> E_K = 88.005 кэВ) доминирует вклад K/L-флуоресценции:
    #    полное A_N(Lead) строго превышает сумму однократного рассеяния и однократной нерассеянной
    #    K-флуоресценции за счет вклада L-оболочек и многократно рассеянных K-фотонов.
    lead_total_first_order = single_albedos_theory[2] + lead_fluor_theory
    total_albedo_invariants_ok = (
        all(
            0.0 < ene_val < num_val < 1.0 and num_val > single_val
            for num_val, ene_val, single_val in zip(
                number_albedos, energy_albedos, single_albedos_mc
            )
        )
        and (0.35 <= number_albedos[0] <= 0.46)
        and (0.20 <= energy_albedos[0] <= 0.28)
        and (0.25 <= number_albedos[1] <= 0.35)
        and (0.15 <= energy_albedos[1] <= 0.23)
        and (0.09 <= number_albedos[2] <= 0.15)
        and (0.05 <= energy_albedos[2] <= 0.09)
        and (number_albedos[2] > lead_total_first_order)
        and (abs(number_albedos[2] - lead_total_first_order) < 0.025)
    )

    peak_ok = peak_abs_err <= (3.0 * units.keV)
    single_albedo_ok = max_single_z < 3.5
    balances_ok = all(bal_item.passed for bal_item in energy_balances)
    passed = (
        hierarchy_ok
        and total_albedo_invariants_ok
        and peak_ok
        and single_albedo_ok
        and balances_ok
    )

    metrics = BackscatterAlbedoMetrics(
        beam_energy=float(beam_energy),
        num_histories_per_material=num_histories,
        materials=albedo_labels,
        number_albedo=number_albedos,
        energy_albedo=energy_albedos,
        single_scatter_number_albedo_mc=single_albedos_mc,
        single_scatter_number_albedo_theory=single_albedos_theory,
        single_scatter_albedo_z_scores=single_z_scores,
        max_single_albedo_z_score=max_single_z,
        fluorescence_escaped_fractions=fluor_fractions,
        lead_first_order_fluorescence_albedo_theory=lead_fluor_theory,
        hierarchy_valid=hierarchy_ok,
        total_albedo_invariants_valid=total_albedo_invariants_ok,
        theoretical_backscatter_peak=theory_peak_energy,
        measured_backscatter_peak=measured_peak_energy,
        peak_abs_error=peak_abs_err,
        energy_balances=energy_balances,
        passed=passed,
    )
    return metrics, spec_centers, spec_counts_float, theory_peak_energy


def _build_independent_euler_rotation_matrix(
    alpha_angle: float,
    beta_angle: float,
    gamma_angle: float,
) -> NDArray[np.float64]:
    """Строит матрицу поворота R_Euler = R_z(alpha) @ R_y(beta) @ R_x(gamma) независимо из элементарных матриц."""
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


def _run_dose_map_handler_parity_audit(
    quick: bool,
) -> tuple[DoseMapParityMetrics, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Проверяет 100% паритет 3D дозовой карты DoseMapHandler на повернутой сетке и конвертацию в Гр."""
    num_histories = 50_000 if quick else 150_000
    beam_energy = Float(140.5 * units.keV)

    vacuum_mat = database_setting.material_database["Vacuum"]
    water_mat = database_setting.material_database["Water, Liquid"]

    world = Volume(
        geometry=Box(Float(500.0), Float(500.0), Float(500.0)),
        material=vacuum_mat,
        name="World_DoseParity",
    )
    phantom_size = 160.0
    euler_alpha = 0.18 * float(units.rad)
    euler_beta = -0.14 * float(units.rad)
    euler_gamma = 0.22 * float(units.rad)
    shift_x = 5.0 * float(units.mm)
    shift_y = -3.0 * float(units.mm)
    shift_z = 80.0 * float(units.mm)

    water_phantom = Volume(
        geometry=Box(Float(phantom_size), Float(phantom_size), Float(phantom_size)),
        material=water_mat,
        name="WaterPhantom_DoseParity",
    )
    water_phantom.rotate(
        alpha=Float(euler_alpha),
        beta=Float(euler_beta),
        gamma=Float(euler_gamma),
    )
    water_phantom.translate(x=Float(shift_x), y=Float(shift_y), z=Float(shift_z))
    world.add_child(water_phantom)

    voxel_step = 8.0
    dose_grid = DoseGridNode(
        name="VerificationDoseGrid",
        size=(Float(phantom_size), Float(phantom_size), Float(phantom_size)),
        dose_voxel_size=Float(voxel_step),
        dtype="float64",
    )
    dose_grid.rotate(
        alpha=Float(euler_alpha),
        beta=Float(euler_beta),
        gamma=Float(euler_gamma),
    )
    dose_grid.translate(x=Float(shift_x), y=Float(shift_y), z=Float(shift_z))
    world.add_child(dose_grid)

    dose_handler = DoseMapHandler(
        grid_nodes=[dose_grid],
        shm_name=None,
        create_shm=False,
    )

    rng = np.random.default_rng(70301)
    rng_ctx = RNGContext.from_numpy_rng(rng)
    propagator = ParticlePropagator(rng=rng)

    geometry_buffer = GeometryCompiler().compile_scene(world)
    physics_buffer = PhysicsCompiler().compile_scene(world, propagator.processes)

    # Независимое матричное построение локального преобразования без использования dose_grid.global_matrix и origin
    nx_bins = int(round(phantom_size / voxel_step))
    ny_bins = int(round(phantom_size / voxel_step))
    nz_bins = int(round(phantom_size / voxel_step))
    manual_grid = np.zeros((nx_bins, ny_bins, nz_bins), dtype=np.float64)

    independent_euler_rot = _build_independent_euler_rotation_matrix(
        euler_alpha, euler_beta, euler_gamma
    )
    independent_translation = np.array([shift_x, shift_y, shift_z], dtype=np.float64)
    independent_local_origin = np.array(
        [-0.5 * phantom_size, -0.5 * phantom_size, -0.5 * phantom_size],
        dtype=np.float64,
    )

    batch_size = min(25_000, num_histories)
    bank = ParticleBank.allocate(batch_size * 2)
    data_buffer = SimulationDataBuffer.allocate_for(
        capacity=batch_size * 4,
        enabled=frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
    )

    min_energy = Float(1.0 * units.keV)

    def _init_dose_batch(
        target_bank: ParticleBank,
        cur_batch: int,
        _processed: int,
    ) -> float:
        jitter_x = rng.uniform(-25.0, 25.0, size=cur_batch) + shift_x
        jitter_y = rng.uniform(-25.0, 25.0, size=cur_batch) + shift_y

        target_bank.state.is_active[:cur_batch] = True

        target_bank.state.position.x[:cur_batch] = jitter_x
        target_bank.state.position.y[:cur_batch] = jitter_y
        target_bank.state.position.z[:cur_batch] = Float(-50.0)

        target_bank.state.direction.x[:cur_batch] = Float(0.0)
        target_bank.state.direction.y[:cur_batch] = Float(0.0)
        target_bank.state.direction.z[:cur_batch] = Float(1.0)

        target_bank.state.energy[:cur_batch] = beam_energy
        target_bank.state.distance_traveled[:cur_batch] = Float(0.0)
        target_bank.state.species[:cur_batch] = Index(ParticleSpecies.PHOTON)

        target_bank.navigation_state.current_volume[:cur_batch] = Index(-1)
        target_bank.navigation_state.boundary_distance[:cur_batch] = Float(0.0)

        particle_ids = np.arange(cur_batch, dtype=ID)
        target_bank.initial_state.ID[:cur_batch] = particle_ids
        target_bank.initial_state.parent_ID[:cur_batch] = ID(-1)
        target_bank.initial_state.emission_time[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_energy[:cur_batch] = beam_energy
        target_bank.initial_state.emission_position.x[:cur_batch] = jitter_x
        target_bank.initial_state.emission_position.y[:cur_batch] = jitter_y
        target_bank.initial_state.emission_position.z[:cur_batch] = Float(-50.0)
        target_bank.initial_state.emission_direction.x[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_direction.y[:cur_batch] = Float(0.0)
        target_bank.initial_state.emission_direction.z[:cur_batch] = Float(1.0)

        return float(cur_batch) * float(beam_energy)

    def _on_dose_interactions(
        step_buffer: SimulationDataBuffer,
        _step_bank: ParticleBank,
    ) -> None:
        raw_chunk = step_buffer.interactions.flush()
        dose_handler.process_chunk({"type": ChunkType.INTERACTIONS, "data": raw_chunk})

        edep_vals = raw_chunk["energy_deposit"].astype(np.float64)
        pos_mask = edep_vals > 0.0
        if not np.any(pos_mask):
            return

        world_coords = np.column_stack(
            (
                raw_chunk["pos_x"][pos_mask].astype(np.float64),
                raw_chunk["pos_y"][pos_mask].astype(np.float64),
                raw_chunk["pos_z"][pos_mask].astype(np.float64),
            )
        )
        # r_local = R_Euler^T (r_world - t) = (r_world - t) @ R_Euler
        local_coords = (world_coords - independent_translation[None, :]) @ independent_euler_rot
        rel_coords = (local_coords - independent_local_origin[None, :]) / voxel_step

        idx_x = np.floor(rel_coords[:, 0]).astype(np.int64)
        idx_y = np.floor(rel_coords[:, 1]).astype(np.int64)
        idx_z = np.floor(rel_coords[:, 2]).astype(np.int64)

        inside_mask = (
            (idx_x >= 0)
            & (idx_x < nx_bins)
            & (idx_y >= 0)
            & (idx_y < ny_bins)
            & (idx_z >= 0)
            & (idx_z < nz_bins)
        )
        if np.any(inside_mask):
            np.add.at(
                manual_grid,
                (idx_x[inside_mask], idx_y[inside_mask], idx_z[inside_mask]),
                edep_vals[pos_mask][inside_mask],
            )

    def _dose_escape_mask(
        step_bank: ParticleBank,
        active_indices: NDArray[Index],
    ) -> NDArray[np.bool_]:
        act_z = step_bank.state.position.z[active_indices]
        return (act_z < -100.0) | (act_z > 240.0)

    energy_balance = run_coupled_transport_loop(
        num_histories=num_histories,
        batch_size=batch_size,
        bank=bank,
        data_buffer=data_buffer,
        geometry_buffer=geometry_buffer,
        physics_buffer=physics_buffer,
        propagator=propagator,
        rng_ctx=rng_ctx,
        min_energy=min_energy,
        initialize_primary_batch=_init_dose_batch,
        enable_atomic_relaxation=True,
        photon_production_cut=Float(1.0 * units.keV),
        electron_production_cut=Float(np.inf),
        on_interactions_step=_on_dose_interactions,
        evaluate_custom_escape_mask=_dose_escape_mask,
    )

    handler_snapshot = dose_handler.get_dose_snapshot("VerificationDoseGrid")
    dose_handler.close()
    if handler_snapshot is None:
        raise RuntimeError("Не удалось получить снимок 3D сетки дозы из DoseMapHandler.")

    handler_grid = np.asarray(handler_snapshot, dtype=np.float64)
    total_handler = float(np.sum(handler_grid))
    total_manual = float(np.sum(manual_grid))
    total_rel_diff = abs(total_handler - total_manual) / max(total_manual, 1e-15)

    # Проверка сохранения полной выделенной энергии внутри повернутого водного фантома
    dep_conservation_rel_diff = abs(
        total_handler - energy_balance.total_deposited_energy
    ) / max(energy_balance.total_deposited_energy, 1e-15)

    profile_handler = np.sum(handler_grid, axis=(0, 1))
    profile_manual = np.sum(manual_grid, axis=(0, 1))

    non_empty_layers = profile_manual > 0.0
    if np.any(non_empty_layers):
        layer_rel_diffs = (
            np.abs(profile_handler[non_empty_layers] - profile_manual[non_empty_layers])
            / profile_manual[non_empty_layers]
        )
        max_layer_rel_diff = float(np.max(layer_rel_diffs))
    else:
        max_layer_rel_diff = 1.0

    max_voxel_abs_diff = float(np.max(np.abs(handler_grid - manual_grid)))
    depth_centers = (np.arange(nz_bins, dtype=np.float64) + 0.5) * voxel_step

    # Конвертация энерговыделения (МэВ) в поглощенную дозу (Гр = Дж/кг) с учетом объема вокселя и плотности воды
    voxel_volume = voxel_step * voxel_step * voxel_step
    water_density = float(water_mat.density)
    voxel_mass_val = (water_density * voxel_volume) / float(units.kg)
    energy_to_joule_hepunits = float(units.MeV) / float(units.joule)
    energy_to_joule_codata = 1.602176634e-13

    absorbed_dose_grid_hepunits = (handler_grid * energy_to_joule_hepunits) / voxel_mass_val
    absorbed_dose_grid_codata = (manual_grid * energy_to_joule_codata) / voxel_mass_val

    mean_phantom_dose = float(np.mean(absorbed_dose_grid_hepunits))
    max_voxel_dose = float(np.max(absorbed_dose_grid_hepunits))
    dose_conversion_rel_diff = float(
        np.max(np.abs(absorbed_dose_grid_hepunits - absorbed_dose_grid_codata))
        / max(max_voxel_dose, 1e-30)
    )

    passed = (
        (total_manual > 0.0)
        and (total_rel_diff < 1e-12)
        and (dep_conservation_rel_diff < 1e-12)
        and (max_layer_rel_diff < 1e-12)
        and (max_voxel_abs_diff < 1e-11)
        and (dose_conversion_rel_diff < 1e-12)
        and (max_voxel_dose > mean_phantom_dose > 0.0)
        and energy_balance.passed
    )

    metrics = DoseMapParityMetrics(
        num_histories=num_histories,
        grid_shape=[int(nx_bins), int(ny_bins), int(nz_bins)],
        voxel_step=voxel_step,
        euler_rotation_angles=[euler_alpha, euler_beta, euler_gamma],
        translation_offset=[shift_x, shift_y, shift_z],
        total_energy_handler=total_handler,
        total_energy_manual=total_manual,
        total_energy_rel_diff=total_rel_diff,
        deposited_energy_conservation_rel_diff=dep_conservation_rel_diff,
        max_layer_rel_diff=max_layer_rel_diff,
        max_voxel_abs_diff=max_voxel_abs_diff,
        voxel_mass=voxel_mass_val,
        mean_phantom_absorbed_dose=mean_phantom_dose,
        max_voxel_absorbed_dose=max_voxel_dose,
        dose_conversion_rel_diff=dose_conversion_rel_diff,
        energy_balance=energy_balance,
        passed=passed,
    )
    return metrics, depth_centers, profile_handler, profile_manual


def run_exp07_macroscopic_transport_and_dosimetry(
    output_dir: Path,
    quick: bool = False,
) -> Exp07MacroscopicDosimetryReport:
    """Запускает Эксперимент 7 (макроскопический транспорт, альбедо и дозиметрия) и сохраняет отчет.

    Args:
        output_dir: Директория для сохранения JSON-отчета и PNG-графика.
        quick: Если True, использует уменьшенный размер выборки для быстрой проверки.

    Returns:
        Структурированный отчет Exp07MacroscopicDosimetryReport.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    buildup_metrics, mfp_grid, buildup_mc, buildup_ansi_reference = (
        _run_water_buildup_factor_audit(quick=quick)
    )
    albedo_metrics, bs_energy_centers, bs_spectrum, bs_peak_energy = _run_backscatter_albedo_audit(
        quick=quick
    )
    dose_metrics, dose_depths, profile_handler, profile_manual = (
        _run_dose_map_handler_parity_audit(quick=quick)
    )

    png_path = output_dir / "exp07_macroscopic_dosimetry.png"
    plot_exp07_macroscopic_transport_and_dosimetry(
        mfp_grid,
        buildup_mc,
        buildup_ansi_reference,
        albedo_materials=albedo_metrics.materials,
        number_albedo_mc=albedo_metrics.number_albedo,
        energy_albedo_mc=albedo_metrics.energy_albedo,
        backscatter_energy_centers=bs_energy_centers,
        backscatter_spectrum=bs_spectrum,
        backscatter_peak_energy=bs_peak_energy,
        dose_depth_centers=dose_depths,
        dose_profile_handler=profile_handler,
        dose_profile_interactions=profile_manual,
        output_path=png_path,
    )

    overall_passed = (
        buildup_metrics.passed
        and albedo_metrics.passed
        and dose_metrics.passed
    )

    report = Exp07MacroscopicDosimetryReport(
        experiment_id="exp07_macroscopic_transport_and_dosimetry",
        title="Эксперимент 7: Макроскопический транспорт, факторы накопления, альбедо и 3D-дозиметрия",
        quick_mode=quick,
        reference_standard_types={
            "water_energy_absorption_buildup": "external_standard_ANSI_ANS_6_4_3_1991_and_GP_formula",
            "single_scatter_backscatter_albedo": "analytical_first_principles_single_scatter_quadrature",
            "lead_k_fluorescence_albedo": "analytical_first_principles_isotropic_slab_escape_integral",
            "compton_180_backscatter_peak": "analytical_exact_relativistic_compton_kinematics",
            "rotated_dose_grid_parity": "independent_analytical_euler_matrix_and_SI_gray_conversion",
        },
        buildup_metrics=buildup_metrics,
        albedo_metrics=albedo_metrics,
        dose_parity_metrics=dose_metrics,
        passed=overall_passed,
    )

    json_path = output_dir / "exp07_macroscopic_dosimetry.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "BuildupFactorMetrics",
    "BackscatterAlbedoMetrics",
    "DoseMapParityMetrics",
    "Exp07MacroscopicDosimetryReport",
    "run_exp07_macroscopic_transport_and_dosimetry",
]
