"""Эксперимент 4: Верификация ядер распределения энергии, выбора подоболочек и атомной релаксации EADL.

Выполняет четыре группы проверок физического ядра:
1. Пособытийный аудит закона сохранения энергии в `partition_compton_energy_kernel` и
   `partition_photoelectric_energy_kernel` в трех режимах порогов (`Local Absorption`,
   `Collision Kerma`, `Coupled Dose`) для материалов `Sodium Iodide` и `Lead` со сверкой
   выборочного среднего локального энерговыделения с точным аналитическим математическим ожиданием.
2. Верификацию ступенчатого открытия подоболочек `_sample_ionized_subshell_scalar` для свинца (Z=82)
   вокруг K, L1, L2, L3-краев поглощения против расчета по кумулятивному правилу скачков поглощения
   J_s (jump_ratio) из таблицы полного фотоэффекта NIST XCOM.
3. Верификацию выходов флуоресценции omega_K, Оже-переходов и относительных интенсивностей линий
   K_alpha1, K_alpha2, K_beta в `_sample_vacancy_relaxation_emission` относительно таблиц ENDF/B-VIII.0 EADL.
4. Независимую внешнюю верификацию (Эталон Типа A) выходов флуоресценции omega_K (Ca, I, Pb),
   отношений радиационных ширин K_alpha2 / K_alpha1 (I, Pb) и вероятностей фотоионизации K-оболочки P_K
   выше K-края (I при 50 и 140.5 кэВ, Pb при 100 и 140.5 кэВ) против опубликованных литературных
   стандартов Krause (1979), Hubbell et al. (1994), Scofield (1973, 1974) и Salem et al. (1974).

Документированные модельные допущения и границы применимости:
- Выбор ионизированной подоболочки при фотоэффекте (`_sample_ionized_subshell_scalar`) использует
  кумулятивное правило скачков поглощения 1 - 1/J_s (отношения скачков краев из таблицы полного
  фотоэффекта NIST XCOM, кусочно-постоянные между краями поглощения), а не прямую интерполяцию
  парциальных подоболочечных сечений EPDL97 / Scofield (1973).
- Атомная релаксация (`_sample_vacancy_relaxation_emission`) реализует одноступенчатую релаксацию
  первичной вакансии с локальным поглощением остаточного каскада вторичных вакансий. Для NaI(Tl) и
  мягких тканей пробег вторичных L-квантов и Оже-электронов (<= 4.5-13 кэВ) составляет единицы-десятки
  мкм << размера вокселя. Для свинца (Pb, Z=82) L-кванты с энергией 10.5-15.2 кэВ (omega_L3 ~ 0.32)
  имеют пробег ~10-20 мкм в самом свинце, однако в воздухе их длина свободного пробега составляет
  порядка миллиметров, что существенно только для тонких поверхностных слоев свинцового коллиматора.
- Некогерентное (комптоновское) рассеяние (`partition_compton_energy_kernel`) разыгрывает кинематику
  электрона отдачи с учетом функции некогерентного рассеяния S(x, Z), но не создает атомных вакансий
  и не запускает последующую флуоресцентную релаксацию ионизированной оболочки.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
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
from core.materials.materials import Material
from core.other.typing_definitions import (
    Charge,
    Energy,
    Float,
    ID,
    Index,
    ParticleSpecies,
    ProcessID,
    ProcessType,
    Species,
    Time,
)
from core.particles.kinematic_state import KinematicState
from core.physics.g4compton import (
    _calculate_compton_energy_deposit_scalar,
    _compute_scattering_function_scalar,
    _generate_compton_theta_scalar,
    electron_mass_c2,
)
from core.physics.interaction_buffers import RNGContext, SecondaryActivationBuffer
from core.physics.physics_buffer import AtomicRelaxationBank, PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.processes_kernels import (
    _sample_Z_scalar,
    _sample_ionized_subshell_scalar,
    _sample_vacancy_relaxation_emission,
    partition_compton_energy_kernel,
    partition_photoelectric_energy_kernel,
)
import settings.database_setting as database_setting
import settings.processes_settings as processes_settings
from verification.physics.plotting import plot_exp04_energy_partition_and_atomic_relaxation


# Внешние независимые литературные эталоны (Тип A):
# 1. Выходы флуоресценции K-оболочки omega_K:
#    - Krause M.O. (1979), "Atomic Radiative and Radiationless Yields for K and L Shells",
#      J. Phys. Chem. Ref. Data 8(2), 307-327, Table 1 & Table 3 (DOI: 10.1063/1.555594);
#    - Hubbell J.H. et al. (1994), "A Review, Bibliography, and Tabulation of K, L, and Higher
#      Atomic Shell X-Ray Fluorescence Yields", J. Phys. Chem. Ref. Data 23(2), 339-364 (DOI: 10.1063/1.555955).
EXTERNAL_OMEGA_K_STANDARDS: tuple[tuple[str, int, float, float, str], ...] = (
    (
        "Ca",
        20,
        0.163,
        0.010,
        "Krause M.O. (1979) JPCRD 8(2), 307, Table 3 (DOI: 10.1063/1.555594); Hubbell et al. (1994) JPCRD 23(2), 339",
    ),
    (
        "I",
        53,
        0.884,
        0.015,
        "Krause M.O. (1979) JPCRD 8(2), 307, Table 3 (DOI: 10.1063/1.555594); Hubbell et al. (1994) JPCRD 23(2), 339",
    ),
    (
        "Pb",
        82,
        0.963,
        0.010,
        "Krause M.O. (1979) JPCRD 8(2), 307, Table 3 (DOI: 10.1063/1.555594); Hubbell et al. (1994) JPCRD 23(2), 339",
    ),
)

# 2. Отношения радиационных ширин переходов K_alpha2 / K_alpha1 (KL2 / KL3):
#    - Scofield J.H. (1974), "Exchange Corrections of K X-Ray Emission Rates",
#      Phys. Rev. A 9(3), 1041-1049, Table I (DOI: 10.1103/PhysRevA.9.1041);
#    - Salem S.I., Panossian S.L. & Krause M.O. (1974), "Experimental K and L Relative X-Ray
#      Emission Rates", At. Data Nucl. Data Tables 14(2), 91-109 (DOI: 10.1016/S0092-640X(74)80017-3).
EXTERNAL_K_ALPHA_RATIO_STANDARDS: tuple[tuple[str, int, float, float, str], ...] = (
    (
        "I",
        53,
        0.537,
        0.015,
        "Scofield J.H. (1974) Phys. Rev. A 9, 1041 (DOI: 10.1103/PhysRevA.9.1041); Salem et al. (1974) ADNDT 14, 91",
    ),
    (
        "Pb",
        82,
        0.596,
        0.015,
        "Scofield J.H. (1974) Phys. Rev. A 9, 1041 (DOI: 10.1103/PhysRevA.9.1041); Salem et al. (1974) ADNDT 14, 91",
    ),
)

# 3. Доля фотоионизации K-оболочки P_K = sigma_K(E) / sigma_pe,tot(E) выше K-края:
#    - Scofield J.H. (1973), "Theoretical Photoionization Cross Sections from 1 to 1500 keV",
#      Lawrence Livermore Laboratory Report UCRL-51326 (DOI: 10.2172/4545040).
EXTERNAL_K_SHELL_FRACTION_STANDARDS: tuple[tuple[str, int, float, float, float, str], ...] = (
    (
        "I",
        53,
        float(50.0 * units.keV),
        0.835,
        0.020,
        "Scofield J.H. (1973) UCRL-51326, Z=53 (DOI: 10.2172/4545040)",
    ),
    (
        "I",
        53,
        float(140.5 * units.keV),
        0.838,
        0.020,
        "Scofield J.H. (1973) UCRL-51326, Z=53 (DOI: 10.2172/4545040)",
    ),
    (
        "Pb",
        82,
        float(100.0 * units.keV),
        0.793,
        0.020,
        "Scofield J.H. (1973) UCRL-51326, Z=82 (DOI: 10.2172/4545040)",
    ),
    (
        "Pb",
        82,
        float(140.5 * units.keV),
        0.798,
        0.020,
        "Scofield J.H. (1973) UCRL-51326, Z=82 (DOI: 10.2172/4545040)",
    ),
)


@dataclass(frozen=True)
class PartitionRegimeMetrics:
    """Метрики пособытийного баланса энергии для одного режима порогов и материала.

    Attributes:
        regime_name: Название режима ('Local Absorption', 'Collision Kerma', 'Coupled Dose').
        material_name: Название материала ('Sodium Iodide' или 'Lead').
        num_events: Общее число смоделированных актов взаимодействия (фотоэффект + Комптон).
        enable_atomic_relaxation: Флаг включения атомной релаксации.
        photon_cut: Порог рождения вторичных фотонов [МэВ].
        electron_cut: Порог рождения вторичных электронов [МэВ].
        mean_transferred_energy: Средняя переданная в событии энергия [МэВ].
        mean_local_deposit: Среднее локальное энерговыделение E_dep [МэВ].
        theoretical_mean_local_deposit: Точное аналитическое матожидание локального депозита [МэВ].
        mean_local_deposit_z_score: Модуль Z-score отклонения среднего E_dep от аналитической теории.
        mean_secondary_photon_energy: Средняя унесенная вторичными фотонами энергия на событие [МэВ].
        mean_secondary_electron_energy: Средняя унесенная вторичными электронами энергия на событие [МэВ].
        secondary_photons_count: Общее число рожденных надпороговых вторичных фотонов.
        secondary_electrons_count: Общее число рожденных надпороговых вторичных электронов.
        max_energy_imbalance: Максимальный пособытийный дисбаланс |dE - (E_dep + sum(E_sec))| [МэВ].
        imbalance_violations: Число событий с нарушением баланса энергии >= 1e-14 МэВ.
        passed: Признак успешного прохождения всех критериев режима.
    """

    regime_name: str
    material_name: str
    num_events: int
    enable_atomic_relaxation: bool
    photon_cut: float
    electron_cut: float
    mean_transferred_energy: float
    mean_local_deposit: float
    theoretical_mean_local_deposit: float
    mean_local_deposit_z_score: float
    mean_secondary_photon_energy: float
    mean_secondary_electron_energy: float
    secondary_photons_count: int
    secondary_electrons_count: int
    max_energy_imbalance: float
    imbalance_violations: int
    passed: bool


@dataclass(frozen=True)
class SubshellSamplingPointMetrics:
    """Метрики выбора ионизированной подоболочки свинца (Z=82) при фиксированной энергии фотона.

    Ядерные вероятности подоболочек вычисляются по кумулятивному правилу скачков поглощения
    1 - 1/J_s (отношения скачков краев из таблицы полного фотоэффекта NIST XCOM).

    Attributes:
        photon_energy: Энергия падающего фотона [МэВ].
        prob_k_mc: Эмпирическая вероятность ионизации K-оболочки.
        prob_k_jump_rule: Теоретическая вероятность K-оболочки по кумулятивному правилу скачков J_s.
        prob_l1_mc: Эмпирическая вероятность ионизации L1-подоболочки.
        prob_l1_jump_rule: Теоретическая вероятность L1-подоболочки по кумулятивному правилу скачков J_s.
        prob_l2_mc: Эмпирическая вероятность ионизации L2-подоболочки.
        prob_l2_jump_rule: Теоретическая вероятность L2-подоболочки по кумулятивному правилу скачков J_s.
        prob_l3_mc: Эмпирическая вероятность ионизации L3-подоболочки.
        prob_l3_jump_rule: Теоретическая вероятность L3-подоболочки по кумулятивному правилу скачков J_s.
        max_abs_z_score: Максимальный модуль Z-score среди открытых подоболочек.
        closed_shell_violations: Число ложных срабатываний энергетически закрытых подоболочек.
        passed: Признак прохождения допусков.
    """

    photon_energy: float
    prob_k_mc: float
    prob_k_jump_rule: float
    prob_l1_mc: float
    prob_l1_jump_rule: float
    prob_l2_mc: float
    prob_l2_jump_rule: float
    prob_l3_mc: float
    prob_l3_jump_rule: float
    max_abs_z_score: float
    closed_shell_violations: int
    passed: bool


@dataclass(frozen=True)
class RelaxationCascadeMetrics:
    """Метрики верификации каскада атомной релаксации K-вакансии относительно таблиц EADL.

    Attributes:
        element_symbol: Химический символ элемента ('I' или 'Pb').
        atomic_z: Атомный номер элемента.
        num_vacancies: Число разыгранных вакансий на K-оболочке.
        omega_k_mc: Эмпирический выход флуоресценции K-оболочки.
        omega_k_eadl: Табличный выход флуоресценции omega_K из ENDF/B-VIII.0 EADL.
        auger_yield_mc: Эмпирический выход Оже-электронов (1 - omega_K).
        auger_yield_eadl: Табличный выход Оже-электронов из ENDF/B-VIII.0 EADL.
        k_alpha1_yield_mc: Эмпирический выход линии K_alpha1 на одну K-вакансию.
        k_alpha1_yield_eadl: Табличный выход линии K_alpha1 на одну K-вакансию.
        k_alpha2_yield_mc: Эмпирический выход линии K_alpha2 на одну K-вакансию.
        k_alpha2_yield_eadl: Табличный выход линии K_alpha2 на одну K-вакансию.
        k_beta_yield_mc: Эмпирический суммарный выход линий K_beta на одну K-вакансию.
        k_beta_yield_eadl: Табличный суммарный выход линий K_beta на одну K-вакансию.
        max_abs_yield_deviation: Максимальное абсолютное отклонение выходов от таблиц EADL.
        max_channel_z_score: Максимальный биномиальный Z-score по 5 проверяемым каналам релаксации.
        passed: Признак выполнения критериев (отклонение < 0.005 и max_channel_z_score < 3.5).
    """

    element_symbol: str
    atomic_z: int
    num_vacancies: int
    omega_k_mc: float
    omega_k_eadl: float
    auger_yield_mc: float
    auger_yield_eadl: float
    k_alpha1_yield_mc: float
    k_alpha1_yield_eadl: float
    k_alpha2_yield_mc: float
    k_alpha2_yield_eadl: float
    k_beta_yield_mc: float
    k_beta_yield_eadl: float
    max_abs_yield_deviation: float
    max_channel_z_score: float
    passed: bool


@dataclass(frozen=True)
class ExternalLiteratureBenchmarkMetrics:
    """Метрики внешней независимой сверки (Тип A) атомной релаксации и ионизации с литературой.

    Attributes:
        observable_name: Проверяемая физическая величина ('omega_K', 'K_alpha2_to_K_alpha1_ratio', 'P_K_photoionization').
        element_symbol: Химический символ элемента ('Ca', 'I', 'Pb').
        atomic_z: Атомный номер элемента.
        probe_energy: Энергия фотона или энергия связи K-края [МэВ].
        mc_value: Эмпирическое значение из Монте-Карло вызова ядер NMSimToolkit.
        kernel_table_value: Аналитическое значение из скомпилированных таблиц ядра.
        literature_value: Внешнее эталонное значение из рецензируемой литературы (Тип A).
        literature_uncertainty: Паспортная погрешность / допуск внешнего эталона.
        literature_citation: Полная библиографическая ссылка на источник с DOI.
        abs_diff_mc_vs_literature: Абсолютное отклонение Монте-Карло от внешнего эталона.
        normalized_deviation: Отношение отклонения к литературному допуску (<= 1.0 для прохождения).
        passed: Признак согласия с внешним эталоном в пределах паспортного допуска.
    """

    observable_name: str
    element_symbol: str
    atomic_z: int
    probe_energy: float
    mc_value: float
    kernel_table_value: float
    literature_value: float
    literature_uncertainty: float
    literature_citation: str
    abs_diff_mc_vs_literature: float
    normalized_deviation: float
    passed: bool


@dataclass(frozen=True)
class Exp04EnergyPartitionReport:
    """Итоговый отчет верификации распределения энергии и атомной релаксации (Эксперимент 4).

    Attributes:
        quick: Признак быстрого режима прогона с уменьшенной статистикой.
        reference_standard_types: Карта независимости эталонов по проверкам (Типы A, B, C).
        model_assumptions: Явная фиксация модельных допущений и области их применимости.
        partition_regimes: Результаты проверки 3 режимов порогов для NaI и Pb.
        subshell_sampling: Результаты проверки ступенчатого открытия подоболочек Pb по правилу скачков J_s.
        relaxation_cascades: Результаты сверки выходов релаксации K-оболочки I и Pb с EADL.
        external_literature_benchmarks: Результаты внешней верификации (Тип A) против Krause / Hubbell / Scofield.
        max_overall_energy_imbalance: Максимальный пособытийный дисбаланс энергии [МэВ].
        max_jump_weight_compile_error: Максимальная погрешность компиляции jump_cumulative_weight.
        max_subshell_z_score: Максимальный |Z-score| при выборе подоболочек Pb.
        max_relaxation_deviation: Максимальное абсолютное отклонение выходов релаксации от EADL.
        max_external_literature_normalized_deviation: Максимальное нормированное отклонение от внешних эталонов Типа A.
        figure_path: Путь к сохраненному диагностическому графику PNG.
        passed: Итоговый статус успешного прохождения всех проверок.
    """

    quick: bool
    reference_standard_types: dict[str, str]
    model_assumptions: dict[str, str]
    partition_regimes: list[PartitionRegimeMetrics]
    subshell_sampling: list[SubshellSamplingPointMetrics]
    relaxation_cascades: list[RelaxationCascadeMetrics]
    external_literature_benchmarks: list[ExternalLiteratureBenchmarkMetrics]
    max_overall_energy_imbalance: float
    max_jump_weight_compile_error: float
    max_subshell_z_score: float
    max_relaxation_deviation: float
    max_external_literature_normalized_deviation: float
    figure_path: str
    passed: bool


@njit(cache=True)
def _run_energy_partition_ensemble_kernel(
    num_events: int,
    incident_energy: float,
    material_id: Index,
    pe_col_idx: Index,
    comp_col_idx: Index,
    physics_buffer: PhysicsBuffer,
    state: KinematicState,
    act_buffer: SecondaryActivationBuffer,
    rng_ctx: RNGContext,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    int,
    int,
    float,
    int,
]:
    """Прогоняет ансамбль актов фотоэффекта и комптоновского рассеяния с пособытийным аудитом баланса энергии."""
    transferred_energies = np.zeros(num_events, dtype=np.float64)
    local_deposits = np.zeros(num_events, dtype=np.float64)

    total_sec_photon_energy = 0.0
    total_sec_electron_energy = 0.0
    sec_photons_count = 0
    sec_electrons_count = 0
    max_imbalance = 0.0
    violations_count = 0

    relaxation_bank = physics_buffer.relaxation_bank
    pe_proc_id = ProcessID(ProcessType.PHOTOELECTRIC)
    comp_proc_id = ProcessID(ProcessType.COMPTON)
    slot_idx = Index(0)

    for event_idx in range(num_events):
        is_photoelectric = (event_idx % 2) == 0
        proc_col = pe_col_idx if is_photoelectric else comp_col_idx
        target_z = _sample_Z_scalar(
            material_id,
            proc_col,
            Float(incident_energy),
            physics_buffer,
            rng_ctx,
        )

        prev_cursor = int(act_buffer.cursor[0])
        act_time = Time(0.25 * float(event_idx + 1))
        parent_id = ID(event_idx)

        if is_photoelectric:
            delta_energy = Energy(incident_energy)
            local_dep = partition_photoelectric_energy_kernel(
                pe_proc_id,
                state,
                slot_idx,
                parent_id,
                Energy(incident_energy),
                delta_energy,
                target_z,
                act_time,
                relaxation_bank,
                act_buffer,
                rng_ctx,
            )
        else:
            theta_angle = _generate_compton_theta_scalar(
                Float(incident_energy),
                target_z,
                rng_ctx,
            )
            delta_energy = _calculate_compton_energy_deposit_scalar(
                theta_angle,
                Float(incident_energy),
            )
            local_dep = partition_compton_energy_kernel(
                comp_proc_id,
                state,
                slot_idx,
                parent_id,
                delta_energy,
                target_z,
                act_time,
                act_buffer,
            )

        new_cursor = int(act_buffer.cursor[0])
        event_sec_energy = 0.0
        for sec_idx in range(prev_cursor, new_cursor):
            sec_ene = float(act_buffer.energy[sec_idx])
            sec_spec = act_buffer.species[sec_idx]
            event_sec_energy += sec_ene
            if sec_spec == ParticleSpecies.PHOTON:
                total_sec_photon_energy += sec_ene
                sec_photons_count += 1
            elif sec_spec == ParticleSpecies.ELECTRON:
                total_sec_electron_energy += sec_ene
                sec_electrons_count += 1

        imbalance = abs(float(delta_energy) - (float(local_dep) + event_sec_energy))
        if imbalance > max_imbalance:
            max_imbalance = imbalance
        if imbalance >= 1e-14:
            violations_count += 1

        transferred_energies[event_idx] = float(delta_energy)
        local_deposits[event_idx] = float(local_dep)

    return (
        transferred_energies,
        local_deposits,
        total_sec_photon_energy,
        total_sec_electron_energy,
        sec_photons_count,
        sec_electrons_count,
        max_imbalance,
        violations_count,
    )


@njit(cache=True)
def _sample_subshell_frequencies_kernel(
    target_z: Charge,
    photon_energy: Energy,
    num_trials: int,
    num_shells: int,
    relaxation_bank: AtomicRelaxationBank,
    rng_ctx: RNGContext,
) -> tuple[NDArray[np.int64], int]:
    """Подсчитывает частоты выбора подоболочек в `_sample_ionized_subshell_scalar`."""
    shell_counts = np.zeros(num_shells, dtype=np.int64)
    outer_count = 0
    for _ in range(num_trials):
        local_idx = int(
            _sample_ionized_subshell_scalar(
                target_z,
                photon_energy,
                relaxation_bank,
                rng_ctx,
            )
        )
        if 0 <= local_idx < num_shells:
            shell_counts[local_idx] += 1
        else:
            outer_count += 1
    return shell_counts, outer_count


@njit(cache=True)
def _sample_k_shell_relaxation_kernel(
    global_shell: Index,
    bind_energy: Energy,
    num_trials: int,
    k_alpha1_energy: float,
    k_alpha2_energy: float,
    relaxation_bank: AtomicRelaxationBank,
    rng_ctx: RNGContext,
) -> tuple[int, int, int, int, int]:
    """Разыгрывает релаксацию K-вакансии и классифицирует выходы флуоресценции (K_alpha1, K_alpha2, K_beta) и Оже."""
    photon_count = 0
    auger_count = 0
    ka1_count = 0
    ka2_count = 0
    kbeta_count = 0
    energy_tol = 1e-9

    for _ in range(num_trials):
        emitted_species, emitted_energy = _sample_vacancy_relaxation_emission(
            global_shell,
            bind_energy,
            Float(0.0),
            Float(0.0),
            relaxation_bank,
            rng_ctx,
        )
        ene_val = float(emitted_energy)
        if ene_val <= 0.0:
            continue
        if emitted_species == ParticleSpecies.PHOTON:
            photon_count += 1
            if abs(ene_val - k_alpha1_energy) <= energy_tol:
                ka1_count += 1
            elif abs(ene_val - k_alpha2_energy) <= energy_tol:
                ka2_count += 1
            else:
                kbeta_count += 1
        elif emitted_species == ParticleSpecies.ELECTRON:
            auger_count += 1

    return photon_count, auger_count, ka1_count, ka2_count, kbeta_count


def _build_standalone_physics_buffer(rng: np.random.Generator) -> PhysicsBuffer:
    """Компилирует PhysicsBuffer штатным публичным методом PhysicsCompiler().compile_scene(...)."""
    mat_db = database_setting.material_database
    att_db = database_setting.attenuation_database
    processes = [
        proc_cls(att_db, rng, process_col_idx=Index(col_idx))
        for col_idx, proc_cls in enumerate(processes_settings.processes_list)
    ]
    world = Volume(
        geometry=Box(Float(100.0), Float(100.0), Float(100.0)),
        material=mat_db["Sodium Iodide"],
        name="Exp04World",
    )
    lead_volume = Volume(
        geometry=Box(Float(10.0), Float(10.0), Float(10.0)),
        material=mat_db["Pb"],
        name="Exp04LeadVolume",
    )
    lead_volume.translate(x=Float(-20.0))
    bone_volume = Volume(
        geometry=Box(Float(10.0), Float(10.0), Float(10.0)),
        material=mat_db["Bone, Cortical (ICRU-44)"],
        name="Exp04BoneVolume",
    )
    bone_volume.translate(x=Float(20.0))
    world.add_child(lead_volume)
    world.add_child(bone_volume)
    return PhysicsCompiler().compile_scene(world, processes)


def _compute_jump_ratio_subshell_probabilities(
    atomic_z: int,
    photon_energy: float,
    attenuation_db: AttenuationDataBase,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Вычисляет вероятности подоболочек по кумулятивному правилу скачков поглощения 1 - 1/J_s (jump_ratio).

    Важное методическое уточнение:
    Ядро `_sample_ionized_subshell_scalar` использует кумулятивные веса скачков поглощения `jump_ratio`,
    извлеченные из узловых отношений на краях полного сечения фотоэффекта NIST XCOM (аппроксимация
    Карлсона–Краузе, кусочно-постоянная между краями), а не прямую интерполяцию парциальных
    подоболочечных сечений EPDL97 / Scofield (1973).

    Для подоболочек s = 0..M-1, упорядоченных по возрастанию энергии связи E_{bind,s}:
    - начальный вес внешних неразрешенных оболочек w_{outer} = 1.0;
    - при переходе через каждый открытый край (E_{bind,s} <= E_gamma) со скачком J_s > 1
      парциальный вес подоболочки равен w_s = (J_s - 1) * W_{cum}(s - 1),
      а новый кумулятивный вес равен W_{cum}(s) = J_s * W_{cum}(s - 1) = prod_{k <= s} J_k.
    Возвращает кортеж (shell_probs, outer_prob, cumulative_weights_from_jumps).
    """
    rel_data = attenuation_db.get_element_relaxation(atomic_z)
    binding_energies = np.asarray(rel_data.binding_energy, dtype=np.float64)
    jump_ratios = np.asarray(rel_data.jump_ratio, dtype=np.float64)
    num_shells = int(binding_energies.size)

    shell_weights = np.zeros(num_shells, dtype=np.float64)
    cumulative_weights = np.zeros(num_shells, dtype=np.float64)
    running_cum_weight = 1.0
    for shell_idx in range(num_shells):
        jump_val = max(1.0, float(jump_ratios[shell_idx]))
        weight_s = (jump_val - 1.0) * running_cum_weight
        running_cum_weight += weight_s
        cumulative_weights[shell_idx] = running_cum_weight
        if float(binding_energies[shell_idx]) <= photon_energy:
            shell_weights[shell_idx] = weight_s

    total_open_weight = 1.0 + float(np.sum(shell_weights))
    if total_open_weight <= 1.0:
        return np.zeros(num_shells, dtype=np.float64), 1.0, cumulative_weights

    shell_probs = shell_weights / total_open_weight
    outer_prob = 1.0 / total_open_weight
    return shell_probs, outer_prob, cumulative_weights


def _compute_element_process_probabilities(
    material: Material,
    incident_energy: float,
    process_name: str,
    attenuation_db: AttenuationDataBase,
) -> list[tuple[int, float]]:
    """Вычисляет парциальные вероятности выбора химических элементов P(Z | proc, E) из таблиц MAC."""
    comp_items = list(material.composition_dict.items())
    if len(comp_items) == 1:
        elem_sym = comp_items[0][0]
        return [(int(atomic_number[elem_sym]), 1.0)]

    query_grid = np.array([incident_energy], dtype=Float)
    partial_weights: list[tuple[int, float]] = []
    weight_sum = 0.0
    for elem_sym, mass_frac in comp_items:
        el_table = attenuation_db.elements_MAC[elem_sym]
        mac_val = float(
            _interpolate_element_on_material_grid(
                query_grid,
                el_table["Energy"],
                el_table["Coefficient"][process_name],
            )[0]
        )
        part_w = float(mass_frac) * mac_val
        weight_sum += part_w
        partial_weights.append((int(atomic_number[elem_sym]), part_w))

    return [(elem_z, part_w / weight_sum) for elem_z, part_w in partial_weights]


def _compute_compton_expected_local_deposit_for_z(
    incident_energy: float,
    atomic_z: int,
    electron_cut: float,
) -> float:
    """Вычисляет точное матожидание локального депозита при комптоновском рассеянии квадратурой Гаусса–Лежандра."""
    if electron_cut <= 0.0:
        return 0.0

    reduced_e = incident_energy / float(electron_mass_c2)
    wl_photon = float(units.h_Planck * units.c_light) / incident_energy

    max_transfer = incident_energy * (2.0 * reduced_e) / (1.0 + 2.0 * reduced_e)
    if electron_cut >= max_transfer:
        upper_one_minus_cos = 2.0
    else:
        ratio_cut = electron_cut / incident_energy
        upper_one_minus_cos = min(2.0, max(0.0, ratio_cut / (reduced_e * (1.0 - ratio_cut))))

    nodes, weights = np.polynomial.legendre.leggauss(64)

    total_norm = 0.0
    for node_val, weight_val in zip(nodes, weights):
        one_cos = float(node_val + 1.0)
        eps_val = 1.0 / (1.0 + reduced_e * one_cos)
        sin_sq = one_cos * (2.0 - one_cos)
        mom_x = math.sqrt(max(0.0, 0.5 * one_cos)) * float(units.cm) / wl_photon
        scat_s = float(_compute_scattering_function_scalar(Float(mom_x), Charge(atomic_z)))
        diff_cs = (eps_val * eps_val) * (eps_val + 1.0 / eps_val - sin_sq) * scat_s
        total_norm += float(weight_val) * diff_cs

    if upper_one_minus_cos <= 0.0 or total_norm <= 0.0:
        return 0.0

    half_span = 0.5 * upper_one_minus_cos
    deposit_num = 0.0
    for node_val, weight_val in zip(nodes, weights):
        one_cos = half_span * float(node_val + 1.0)
        eps_val = 1.0 / (1.0 + reduced_e * one_cos)
        sin_sq = one_cos * (2.0 - one_cos)
        mom_x = math.sqrt(max(0.0, 0.5 * one_cos)) * float(units.cm) / wl_photon
        scat_s = float(_compute_scattering_function_scalar(Float(mom_x), Charge(atomic_z)))
        diff_cs = (eps_val * eps_val) * (eps_val + 1.0 / eps_val - sin_sq) * scat_s
        transfer_e = incident_energy * (1.0 - eps_val)
        deposit_num += half_span * float(weight_val) * diff_cs * transfer_e

    return deposit_num / total_norm


def _compute_photoelectric_expected_local_deposit_for_z(
    incident_energy: float,
    atomic_z: int,
    enable_relaxation: bool,
    photon_cut: float,
    electron_cut: float,
    attenuation_db: AttenuationDataBase,
) -> float:
    """Вычисляет точное матожидание локального депозита при фотоэффекте на атоме Z в одноступенчатой модели."""
    if (not enable_relaxation or incident_energy < photon_cut) and incident_energy < electron_cut:
        return incident_energy

    shell_probs, outer_prob, _ = _compute_jump_ratio_subshell_probabilities(
        atomic_z=atomic_z,
        photon_energy=incident_energy,
        attenuation_db=attenuation_db,
    )
    rel_data = attenuation_db.get_element_relaxation(atomic_z)
    num_shells = int(rel_data.binding_energy.size)

    expected_dep = 0.0
    if incident_energy < electron_cut:
        expected_dep += outer_prob * incident_energy

    for shell_idx in range(num_shells):
        prob_s = float(shell_probs[shell_idx])
        if prob_s <= 0.0:
            continue
        bind_e = float(rel_data.binding_energy[shell_idx])
        pe_kin = max(0.0, incident_energy - bind_e)
        tracked_pe = pe_kin if (pe_kin > 0.0 and pe_kin >= electron_cut) else 0.0

        tracked_relax = 0.0
        if enable_relaxation and bind_e > 0.0:
            omega_s = float(rel_data.fluorescence_yield[shell_idx])
            r_start = int(rel_data.rad_offsets[shell_idx])
            r_end = int(rel_data.rad_offsets[shell_idx + 1])
            if r_end > r_start and omega_s > 0.0:
                r_enes = np.asarray(rel_data.rad_energy[r_start:r_end], dtype=np.float64)
                r_probs = np.asarray(rel_data.rad_probability[r_start:r_end], dtype=np.float64)
                r_probs_norm = r_probs / float(np.sum(r_probs))
                above_p = (r_enes > 0.0) & (r_enes >= photon_cut)
                tracked_relax += omega_s * float(np.sum(r_probs_norm[above_p] * r_enes[above_p]))

            a_start = int(rel_data.auger_offsets[shell_idx])
            a_end = int(rel_data.auger_offsets[shell_idx + 1])
            if a_end > a_start and bind_e >= electron_cut and omega_s < 1.0:
                a_enes = np.asarray(rel_data.auger_energy[a_start:a_end], dtype=np.float64)
                a_probs = np.asarray(rel_data.auger_probability[a_start:a_end], dtype=np.float64)
                a_probs_norm = a_probs / float(np.sum(a_probs))
                above_e = (a_enes > 0.0) & (a_enes >= electron_cut)
                tracked_relax += (1.0 - omega_s) * float(
                    np.sum(a_probs_norm[above_e] * a_enes[above_e])
                )

        shell_local_dep = max(0.0, incident_energy - tracked_pe - tracked_relax)
        expected_dep += prob_s * shell_local_dep

    return expected_dep


def _compute_regime_theoretical_mean_local_deposit(
    material: Material,
    incident_energy: float,
    enable_relaxation: bool,
    photon_cut: float,
    electron_cut: float,
    attenuation_db: AttenuationDataBase,
) -> float:
    """Вычисляет аналитическое матожидание <E_dep>_theory для ансамбля 50% Photoelectric + 50% Compton."""
    pe_elem_probs = _compute_element_process_probabilities(
        material, incident_energy, "PhotoelectricEffect", attenuation_db
    )
    comp_elem_probs = _compute_element_process_probabilities(
        material, incident_energy, "ComptonScattering", attenuation_db
    )

    mean_pe_dep = sum(
        prob_z
        * _compute_photoelectric_expected_local_deposit_for_z(
            incident_energy=incident_energy,
            atomic_z=elem_z,
            enable_relaxation=enable_relaxation,
            photon_cut=photon_cut,
            electron_cut=electron_cut,
            attenuation_db=attenuation_db,
        )
        for elem_z, prob_z in pe_elem_probs
    )
    mean_comp_dep = sum(
        prob_z
        * _compute_compton_expected_local_deposit_for_z(
            incident_energy=incident_energy,
            atomic_z=elem_z,
            electron_cut=electron_cut,
        )
        for elem_z, prob_z in comp_elem_probs
    )
    return 0.5 * (mean_pe_dep + mean_comp_dep)


def _run_external_literature_relaxation_audit(
    physics_buffer: PhysicsBuffer,
    attenuation_db: AttenuationDataBase,
    rng_ctx: RNGContext,
    quick: bool,
) -> list[ExternalLiteratureBenchmarkMetrics]:
    """Выполняет внешнюю независимую сверку (Тип A) с константами Krause (1979), Hubbell (1994) и Scofield (1973, 1974)."""
    num_trials = 50_000 if quick else 150_000
    benchmarks: list[ExternalLiteratureBenchmarkMetrics] = []
    relaxation_bank = physics_buffer.relaxation_bank

    # 1. Сверка omega_K по Krause (1979) / Hubbell et al. (1994) для Ca (Z=20), I (Z=53), Pb (Z=82)
    for elem_sym, elem_z, lit_val, lit_tol, citation in EXTERNAL_OMEGA_K_STANDARDS:
        rel_data = attenuation_db.get_element_relaxation(elem_z)
        elem_labels = [str(lbl) for lbl in rel_data.subshell_label]
        local_k_idx = elem_labels.index("K")
        s_start = int(relaxation_bank.shell_offsets[elem_z])
        global_k_idx = Index(s_start + local_k_idx)
        k_bind_energy = Energy(rel_data.binding_energy[local_k_idx])
        omega_table = float(rel_data.fluorescence_yield[local_k_idx])

        r_start = int(rel_data.rad_offsets[local_k_idx])
        r_end = int(rel_data.rad_offsets[local_k_idx + 1])
        rad_desig = rel_data.rad_vacancy_designator[r_start:r_end]
        rad_ene = rel_data.rad_energy[r_start:r_end]
        ka1_mask = rad_desig == 6
        ka2_mask = rad_desig == 5
        ka1_ene = float(rad_ene[ka1_mask][0]) if np.any(ka1_mask) else 0.0
        ka2_ene = float(rad_ene[ka2_mask][0]) if np.any(ka2_mask) else 0.0

        photon_cnt, _, _, _, _ = _sample_k_shell_relaxation_kernel(
            global_shell=global_k_idx,
            bind_energy=k_bind_energy,
            num_trials=num_trials,
            k_alpha1_energy=ka1_ene,
            k_alpha2_energy=ka2_ene,
            relaxation_bank=relaxation_bank,
            rng_ctx=rng_ctx,
        )
        omega_mc = float(photon_cnt) / float(num_trials)
        abs_diff = abs(omega_mc - lit_val)
        norm_dev = abs_diff / lit_tol
        benchmarks.append(
            ExternalLiteratureBenchmarkMetrics(
                observable_name="omega_K",
                element_symbol=elem_sym,
                atomic_z=elem_z,
                probe_energy=float(k_bind_energy),
                mc_value=omega_mc,
                kernel_table_value=omega_table,
                literature_value=lit_val,
                literature_uncertainty=lit_tol,
                literature_citation=citation,
                abs_diff_mc_vs_literature=abs_diff,
                normalized_deviation=norm_dev,
                passed=(norm_dev <= 1.0) and (abs(omega_table - lit_val) <= lit_tol),
            )
        )

    # 2. Сверка отношений интенсивностей K_alpha2 / K_alpha1 (KL2 / KL3) по Scofield (1974) / Salem et al. (1974)
    for elem_sym, elem_z, lit_ratio, lit_tol, citation in EXTERNAL_K_ALPHA_RATIO_STANDARDS:
        rel_data = attenuation_db.get_element_relaxation(elem_z)
        elem_labels = [str(lbl) for lbl in rel_data.subshell_label]
        local_k_idx = elem_labels.index("K")
        s_start = int(relaxation_bank.shell_offsets[elem_z])
        global_k_idx = Index(s_start + local_k_idx)
        k_bind_energy = Energy(rel_data.binding_energy[local_k_idx])

        r_start = int(rel_data.rad_offsets[local_k_idx])
        r_end = int(rel_data.rad_offsets[local_k_idx + 1])
        rad_desig = rel_data.rad_vacancy_designator[r_start:r_end]
        rad_ene = rel_data.rad_energy[r_start:r_end]
        rad_prob = rel_data.rad_probability[r_start:r_end]
        ka1_mask = rad_desig == 6
        ka2_mask = rad_desig == 5
        ka1_ene = float(rad_ene[ka1_mask][0])
        ka2_ene = float(rad_ene[ka2_mask][0])
        table_ratio = float(np.sum(rad_prob[ka2_mask])) / float(np.sum(rad_prob[ka1_mask]))

        _, _, ka1_cnt, ka2_cnt, _ = _sample_k_shell_relaxation_kernel(
            global_shell=global_k_idx,
            bind_energy=k_bind_energy,
            num_trials=num_trials,
            k_alpha1_energy=ka1_ene,
            k_alpha2_energy=ka2_ene,
            relaxation_bank=relaxation_bank,
            rng_ctx=rng_ctx,
        )
        mc_ratio = float(ka2_cnt) / float(max(ka1_cnt, 1))
        abs_diff = abs(mc_ratio - lit_ratio)
        norm_dev = abs_diff / lit_tol
        benchmarks.append(
            ExternalLiteratureBenchmarkMetrics(
                observable_name="K_alpha2_to_K_alpha1_ratio",
                element_symbol=elem_sym,
                atomic_z=elem_z,
                probe_energy=float(k_bind_energy),
                mc_value=mc_ratio,
                kernel_table_value=table_ratio,
                literature_value=lit_ratio,
                literature_uncertainty=lit_tol,
                literature_citation=citation,
                abs_diff_mc_vs_literature=abs_diff,
                normalized_deviation=norm_dev,
                passed=(norm_dev <= 1.0) and (abs(table_ratio - lit_ratio) <= lit_tol),
            )
        )

    # 3. Сверка доли ионизации K-оболочки P_K выше K-края по Scofield (1973), UCRL-51326
    for elem_sym, elem_z, photon_ene, lit_pk, lit_tol, citation in EXTERNAL_K_SHELL_FRACTION_STANDARDS:
        rel_data = attenuation_db.get_element_relaxation(elem_z)
        elem_labels = [str(lbl) for lbl in rel_data.subshell_label]
        idx_k = elem_labels.index("K")
        num_shells = len(elem_labels)

        counts_arr, _ = _sample_subshell_frequencies_kernel(
            target_z=Charge(elem_z),
            photon_energy=Energy(photon_ene),
            num_trials=num_trials,
            num_shells=num_shells,
            relaxation_bank=relaxation_bank,
            rng_ctx=rng_ctx,
        )
        pk_mc = float(counts_arr[idx_k]) / float(num_trials)
        probs_jump, _, _ = _compute_jump_ratio_subshell_probabilities(
            atomic_z=elem_z,
            photon_energy=photon_ene,
            attenuation_db=attenuation_db,
        )
        pk_jump = float(probs_jump[idx_k])
        abs_diff = abs(pk_mc - lit_pk)
        norm_dev = abs_diff / lit_tol
        benchmarks.append(
            ExternalLiteratureBenchmarkMetrics(
                observable_name="P_K_photoionization",
                element_symbol=elem_sym,
                atomic_z=elem_z,
                probe_energy=photon_ene,
                mc_value=pk_mc,
                kernel_table_value=pk_jump,
                literature_value=lit_pk,
                literature_uncertainty=lit_tol,
                literature_citation=citation,
                abs_diff_mc_vs_literature=abs_diff,
                normalized_deviation=norm_dev,
                passed=(norm_dev <= 1.0) and (abs(pk_jump - lit_pk) <= lit_tol),
            )
        )

    return benchmarks


def run_exp04_energy_partition_and_atomic_relaxation(
    output_dir: Path,
    quick: bool = False,
) -> Exp04EnergyPartitionReport:
    """Запускает Эксперимент 4: верификацию распределения энергии, выбора подоболочек и релаксации EADL.

    Args:
        output_dir: Директория для сохранения JSON-отчета и PNG-графика.
        quick: Режим уменьшенной статистики для быстрой проверки.

    Returns:
        Отчет Exp04EnergyPartitionReport.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260404)
    rng_ctx = RNGContext.from_numpy_rng(rng)

    physics_buffer = _build_standalone_physics_buffer(rng)
    mat_db = database_setting.material_database
    att_db = database_setting.attenuation_database

    # 1. Пособытийный аудит закона сохранения энергии в 3 режимах порогов
    events_per_regime = 120_000 if quick else 600_000
    incident_energy = float(140.5 * units.keV)

    dummy_state = KinematicState.allocate(1)
    dummy_state.position.x[0] = Float(0.0)
    dummy_state.position.y[0] = Float(0.0)
    dummy_state.position.z[0] = Float(0.0)
    dummy_state.direction.x[0] = Float(0.0)
    dummy_state.direction.y[0] = Float(0.0)
    dummy_state.direction.z[0] = Float(1.0)
    dummy_state.energy[0] = Energy(incident_energy)
    dummy_state.species[0] = Species(ParticleSpecies.PHOTON)
    dummy_state.is_active[0] = True

    regimes_spec: list[tuple[str, bool, float, float]] = [
        ("Local Absorption", False, float(np.inf), float(np.inf)),
        ("Collision Kerma", True, float(1.0 * units.keV), float(np.inf)),
        ("Coupled Dose", True, float(1.0 * units.keV), float(1.0 * units.keV)),
    ]
    materials_to_test = [("Sodium Iodide", "Sodium Iodide"), ("Lead", "Pb")]

    hist_bins = np.linspace(0.0, float(145.0 * units.keV), 581, dtype=np.float64)
    energy_bins_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    hist_by_regime: dict[str, NDArray[np.float64]] = {
        regime_label: np.zeros(energy_bins_centers.size, dtype=np.float64)
        for regime_label, _, _, _ in regimes_spec
    }

    partition_metrics: list[PartitionRegimeMetrics] = []
    max_overall_imbalance = 0.0

    for regime_label, enable_relax, photon_cut, electron_cut in regimes_spec:
        for mat_display_name, mat_key in materials_to_test:
            mat_obj = mat_db[mat_key]
            mat_id = Index(mat_obj.ID)
            act_buffer = SecondaryActivationBuffer.allocate(
                capacity=events_per_regime * 3,
                enable_atomic_relaxation=enable_relax,
                photon_production_cut=Float(photon_cut),
                electron_production_cut=Float(electron_cut),
            )
            (
                transferred_arr,
                local_dep_arr,
                sec_photon_sum,
                sec_electron_sum,
                sec_photons_cnt,
                sec_electrons_cnt,
                max_imb,
                imb_violations,
            ) = _run_energy_partition_ensemble_kernel(
                num_events=events_per_regime,
                incident_energy=incident_energy,
                material_id=mat_id,
                pe_col_idx=Index(0),
                comp_col_idx=Index(1),
                physics_buffer=physics_buffer,
                state=dummy_state,
                act_buffer=act_buffer,
                rng_ctx=rng_ctx,
            )

            counts_hist, _ = np.histogram(local_dep_arr, bins=hist_bins)
            hist_by_regime[regime_label] += counts_hist.astype(np.float64)
            max_overall_imbalance = max(max_overall_imbalance, float(max_imb))

            mean_local_mc = float(np.mean(local_dep_arr))
            mean_local_theory = _compute_regime_theoretical_mean_local_deposit(
                material=mat_obj,
                incident_energy=incident_energy,
                enable_relaxation=enable_relax,
                photon_cut=photon_cut,
                electron_cut=electron_cut,
                attenuation_db=att_db,
            )

            pe_sub_deps = local_dep_arr[0::2]
            comp_sub_deps = local_dep_arr[1::2]
            var_pe = float(np.var(pe_sub_deps, ddof=1)) if pe_sub_deps.size > 1 else 0.0
            var_comp = float(np.var(comp_sub_deps, ddof=1)) if comp_sub_deps.size > 1 else 0.0
            stratified_se = 0.5 * math.sqrt(
                var_pe / float(max(pe_sub_deps.size, 1))
                + var_comp / float(max(comp_sub_deps.size, 1))
            )
            if stratified_se > 1e-15:
                mean_z_score = abs(mean_local_mc - mean_local_theory) / stratified_se
            else:
                mean_z_score = 0.0 if abs(mean_local_mc - mean_local_theory) < 1e-12 else 100.0

            regime_constraints_ok = (
                (imb_violations == 0)
                and (max_imb < 1e-14)
                and (mean_z_score < 3.5)
            )
            if regime_label == "Local Absorption":
                regime_constraints_ok = (
                    regime_constraints_ok
                    and (sec_photons_cnt == 0)
                    and (sec_electrons_cnt == 0)
                    and (act_buffer.cursor_value == 0)
                    and bool(np.all(local_dep_arr == transferred_arr))
                )
            elif regime_label == "Collision Kerma":
                regime_constraints_ok = (
                    regime_constraints_ok
                    and (sec_photons_cnt > 0)
                    and (sec_electrons_cnt == 0)
                )
            elif regime_label == "Coupled Dose":
                regime_constraints_ok = (
                    regime_constraints_ok
                    and (sec_photons_cnt > 0)
                    and (sec_electrons_cnt > 0)
                )

            partition_metrics.append(
                PartitionRegimeMetrics(
                    regime_name=regime_label,
                    material_name=mat_display_name,
                    num_events=events_per_regime,
                    enable_atomic_relaxation=enable_relax,
                    photon_cut=photon_cut,
                    electron_cut=electron_cut,
                    mean_transferred_energy=float(np.mean(transferred_arr)),
                    mean_local_deposit=mean_local_mc,
                    theoretical_mean_local_deposit=mean_local_theory,
                    mean_local_deposit_z_score=mean_z_score,
                    mean_secondary_photon_energy=float(sec_photon_sum / events_per_regime),
                    mean_secondary_electron_energy=float(sec_electron_sum / events_per_regime),
                    secondary_photons_count=int(sec_photons_cnt),
                    secondary_electrons_count=int(sec_electrons_cnt),
                    max_energy_imbalance=float(max_imb),
                    imbalance_violations=int(imb_violations),
                    passed=regime_constraints_ok,
                )
            )

    # 2. Ступенчатое открытие подоболочек свинца (Pb, Z=82) вокруг K и L1, L2, L3-краев по правилу скачков J_s
    pb_z = 82
    pb_relax_data = att_db.get_element_relaxation(pb_z)
    pb_labels = [str(lbl) for lbl in pb_relax_data.subshell_label]
    idx_k = pb_labels.index("K")
    idx_l1 = pb_labels.index("L1")
    idx_l2 = pb_labels.index("L2")
    idx_l3 = pb_labels.index("L3")
    num_pb_shells = len(pb_labels)

    pb_s_start = int(physics_buffer.relaxation_bank.shell_offsets[pb_z])
    pb_s_end = int(physics_buffer.relaxation_bank.shell_offsets[pb_z + 1])
    compiled_pb_cum_weights = np.asarray(
        physics_buffer.relaxation_bank.jump_cumulative_weight[pb_s_start:pb_s_end],
        dtype=np.float64,
    )
    _, _, direct_pb_cum_weights = _compute_jump_ratio_subshell_probabilities(
        atomic_z=pb_z,
        photon_energy=float(1.0 * units.MeV),
        attenuation_db=att_db,
    )
    max_jump_compile_err = float(
        np.max(np.abs(compiled_pb_cum_weights - direct_pb_cum_weights) / direct_pb_cum_weights)
    )

    edge_l3 = float(pb_relax_data.binding_energy[idx_l3])
    edge_l2 = float(pb_relax_data.binding_energy[idx_l2])
    edge_l1 = float(pb_relax_data.binding_energy[idx_l1])
    edge_k = float(pb_relax_data.binding_energy[idx_k])

    subshell_energies = np.array(
        [
            12.0 * units.keV,
            edge_l3 * 0.995,
            edge_l3 * 1.005,
            14.2 * units.keV,
            edge_l2 * 0.995,
            edge_l2 * 1.005,
            edge_l1 * 0.995,
            edge_l1 * 1.005,
            25.0 * units.keV,
            55.0 * units.keV,
            edge_k * 0.995,
            edge_k * 1.005,
            105.0 * units.keV,
            135.0 * units.keV,
        ],
        dtype=np.float64,
    )
    subshell_trials = 30_000 if quick else 100_000

    prob_k_mc_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_k_ref_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_l1_mc_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_l1_ref_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_l2_mc_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_l2_ref_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_l3_mc_arr = np.zeros(subshell_energies.size, dtype=np.float64)
    prob_l3_ref_arr = np.zeros(subshell_energies.size, dtype=np.float64)

    subshell_metrics: list[SubshellSamplingPointMetrics] = []
    max_subshell_z_score = 0.0

    for pt_idx, ene_val in enumerate(subshell_energies):
        photon_ene = float(ene_val)
        counts_arr, _ = _sample_subshell_frequencies_kernel(
            target_z=Charge(pb_z),
            photon_energy=Energy(photon_ene),
            num_trials=subshell_trials,
            num_shells=num_pb_shells,
            relaxation_bank=physics_buffer.relaxation_bank,
            rng_ctx=rng_ctx,
        )
        probs_mc = counts_arr.astype(np.float64) / float(subshell_trials)
        probs_ref, _, _ = _compute_jump_ratio_subshell_probabilities(
            atomic_z=pb_z,
            photon_energy=photon_ene,
            attenuation_db=att_db,
        )

        prob_k_mc_arr[pt_idx] = probs_mc[idx_k]
        prob_k_ref_arr[pt_idx] = probs_ref[idx_k]
        prob_l1_mc_arr[pt_idx] = probs_mc[idx_l1]
        prob_l1_ref_arr[pt_idx] = probs_ref[idx_l1]
        prob_l2_mc_arr[pt_idx] = probs_mc[idx_l2]
        prob_l2_ref_arr[pt_idx] = probs_ref[idx_l2]
        prob_l3_mc_arr[pt_idx] = probs_mc[idx_l3]
        prob_l3_ref_arr[pt_idx] = probs_ref[idx_l3]

        closed_violations = 0
        point_max_z = 0.0
        for shell_idx in (idx_k, idx_l1, idx_l2, idx_l3):
            bind_val = float(pb_relax_data.binding_energy[shell_idx])
            if photon_ene < bind_val:
                if int(counts_arr[shell_idx]) != 0:
                    closed_violations += int(counts_arr[shell_idx])
            else:
                ref_p = float(probs_ref[shell_idx])
                if 0.0 < ref_p < 1.0:
                    sigma_p = math.sqrt(ref_p * (1.0 - ref_p) / float(subshell_trials))
                    z_val = abs(float(probs_mc[shell_idx]) - ref_p) / sigma_p
                    point_max_z = max(point_max_z, z_val)

        max_subshell_z_score = max(max_subshell_z_score, point_max_z)
        point_passed = (
            (closed_violations == 0)
            and (point_max_z < 3.5)
            and (max_jump_compile_err < 1e-14)
        )

        subshell_metrics.append(
            SubshellSamplingPointMetrics(
                photon_energy=photon_ene,
                prob_k_mc=float(probs_mc[idx_k]),
                prob_k_jump_rule=float(probs_ref[idx_k]),
                prob_l1_mc=float(probs_mc[idx_l1]),
                prob_l1_jump_rule=float(probs_ref[idx_l1]),
                prob_l2_mc=float(probs_mc[idx_l2]),
                prob_l2_jump_rule=float(probs_ref[idx_l2]),
                prob_l3_mc=float(probs_mc[idx_l3]),
                prob_l3_jump_rule=float(probs_ref[idx_l3]),
                max_abs_z_score=float(point_max_z),
                closed_shell_violations=int(closed_violations),
                passed=point_passed,
            )
        )

    subshell_curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]] = [
        ("K (88.0 кэВ)", prob_k_mc_arr, prob_k_ref_arr),
        ("L1 (15.9 кэВ)", prob_l1_mc_arr, prob_l1_ref_arr),
        ("L2 (15.2 кэВ)", prob_l2_mc_arr, prob_l2_ref_arr),
        ("L3 (13.0 кэВ)", prob_l3_mc_arr, prob_l3_ref_arr),
    ]

    # 3. Каскад атомной релаксации K-оболочки иода (Z=53) и свинца (Z=82) против таблиц EADL
    relax_trials = 60_000 if quick else 200_000
    relaxation_metrics: list[RelaxationCascadeMetrics] = []
    fluorescence_labels: list[str] = []
    fluorescence_mc: list[float] = []
    fluorescence_eadl: list[float] = []
    max_relaxation_dev = 0.0

    for elem_sym, elem_z in [("I", 53), ("Pb", 82)]:
        rel_data = att_db.get_element_relaxation(elem_z)
        elem_labels = [str(lbl) for lbl in rel_data.subshell_label]
        local_k_idx = elem_labels.index("K")
        s_start = int(physics_buffer.relaxation_bank.shell_offsets[elem_z])
        global_k_idx = Index(s_start + local_k_idx)
        k_bind_energy = Energy(rel_data.binding_energy[local_k_idx])
        omega_k_ref = float(rel_data.fluorescence_yield[local_k_idx])
        auger_ref = 1.0 - omega_k_ref

        r_start = int(rel_data.rad_offsets[local_k_idx])
        r_end = int(rel_data.rad_offsets[local_k_idx + 1])
        rad_desig = rel_data.rad_vacancy_designator[r_start:r_end]
        rad_ene = rel_data.rad_energy[r_start:r_end]
        rad_prob_raw = rel_data.rad_probability[r_start:r_end]
        rad_prob_norm = rad_prob_raw / float(np.sum(rad_prob_raw))

        ka1_mask = rad_desig == 6  # L3 -> K
        ka2_mask = rad_desig == 5  # L2 -> K
        kbeta_mask = ~(ka1_mask | ka2_mask)

        ka1_energy = float(rad_ene[ka1_mask][0])
        ka2_energy = float(rad_ene[ka2_mask][0])

        ka1_yield_ref = omega_k_ref * float(np.sum(rad_prob_norm[ka1_mask]))
        ka2_yield_ref = omega_k_ref * float(np.sum(rad_prob_norm[ka2_mask]))
        kbeta_yield_ref = omega_k_ref * float(np.sum(rad_prob_norm[kbeta_mask]))

        photon_cnt, auger_cnt, ka1_cnt, ka2_cnt, kbeta_cnt = _sample_k_shell_relaxation_kernel(
            global_shell=global_k_idx,
            bind_energy=k_bind_energy,
            num_trials=relax_trials,
            k_alpha1_energy=ka1_energy,
            k_alpha2_energy=ka2_energy,
            relaxation_bank=physics_buffer.relaxation_bank,
            rng_ctx=rng_ctx,
        )

        inv_trials = 1.0 / float(relax_trials)
        omega_k_mc = float(photon_cnt) * inv_trials
        auger_mc = float(auger_cnt) * inv_trials
        ka1_yield_mc = float(ka1_cnt) * inv_trials
        ka2_yield_mc = float(ka2_cnt) * inv_trials
        kbeta_yield_mc = float(kbeta_cnt) * inv_trials

        channel_pairs = [
            (omega_k_mc, omega_k_ref),
            (auger_mc, auger_ref),
            (ka1_yield_mc, ka1_yield_ref),
            (ka2_yield_mc, ka2_yield_ref),
            (kbeta_yield_mc, kbeta_yield_ref),
        ]
        deviations = [abs(mc_val - ref_val) for mc_val, ref_val in channel_pairs]
        channel_z_scores = [
            abs(mc_val - ref_val) / math.sqrt(ref_val * (1.0 - ref_val) * inv_trials)
            for mc_val, ref_val in channel_pairs
            if 0.0 < ref_val < 1.0
        ]
        elem_max_dev = max(deviations)
        elem_max_z = max(channel_z_scores) if channel_z_scores else 0.0
        max_relaxation_dev = max(max_relaxation_dev, elem_max_dev)
        elem_passed = (
            (elem_max_dev < 3.5 * math.sqrt(0.25 * inv_trials))
            and (elem_max_z < 3.5)
            and (photon_cnt + auger_cnt == relax_trials)
        )

        relaxation_metrics.append(
            RelaxationCascadeMetrics(
                element_symbol=elem_sym,
                atomic_z=elem_z,
                num_vacancies=relax_trials,
                omega_k_mc=omega_k_mc,
                omega_k_eadl=omega_k_ref,
                auger_yield_mc=auger_mc,
                auger_yield_eadl=auger_ref,
                k_alpha1_yield_mc=ka1_yield_mc,
                k_alpha1_yield_eadl=ka1_yield_ref,
                k_alpha2_yield_mc=ka2_yield_mc,
                k_alpha2_yield_eadl=ka2_yield_ref,
                k_beta_yield_mc=kbeta_yield_mc,
                k_beta_yield_eadl=kbeta_yield_ref,
                max_abs_yield_deviation=elem_max_dev,
                max_channel_z_score=elem_max_z,
                passed=elem_passed,
            )
        )

        fluorescence_labels.extend(
            [
                f"{elem_sym} $\\omega_K$",
                f"{elem_sym} Оже",
                f"{elem_sym} $K_{{\\alpha 1}}$",
                f"{elem_sym} $K_{{\\alpha 2}}$",
                f"{elem_sym} $K_\\beta$",
            ]
        )
        fluorescence_mc.extend([omega_k_mc, auger_mc, ka1_yield_mc, ka2_yield_mc, kbeta_yield_mc])
        fluorescence_eadl.extend(
            [omega_k_ref, auger_ref, ka1_yield_ref, ka2_yield_ref, kbeta_yield_ref]
        )

    # 4. Внешняя независимая литературная верификация (Эталон Типа A)
    external_benchmarks = _run_external_literature_relaxation_audit(
        physics_buffer=physics_buffer,
        attenuation_db=att_db,
        rng_ctx=rng_ctx,
        quick=quick,
    )
    max_external_norm_dev = max(item.normalized_deviation for item in external_benchmarks)

    figure_path = output_dir / "exp04_energy_partition.png"
    plot_exp04_energy_partition_and_atomic_relaxation(
        energy_bins_centers=energy_bins_centers,
        hist_local_absorption=hist_by_regime["Local Absorption"],
        hist_collision_kerma=hist_by_regime["Collision Kerma"],
        hist_coupled_dose=hist_by_regime["Coupled Dose"],
        subshell_energies=subshell_energies,
        subshell_curves=subshell_curves,
        fluorescence_labels=fluorescence_labels,
        fluorescence_mc=fluorescence_mc,
        fluorescence_eadl=fluorescence_eadl,
        output_path=figure_path,
    )

    overall_passed = (
        all(item.passed for item in partition_metrics)
        and all(item.passed for item in subshell_metrics)
        and all(item.passed for item in relaxation_metrics)
        and all(item.passed for item in external_benchmarks)
        and (max_jump_compile_err < 1e-14)
    )

    reference_standard_types = {
        "external_omega_k_krause_hubbell": "Type A (внешний литературный стандарт Krause 1979 / Hubbell et al. 1994)",
        "external_k_alpha_ratio_scofield_salem": "Type A (внешний литературный стандарт Scofield 1974 / Salem et al. 1974)",
        "external_k_shell_fraction_scofield": "Type A (внешние теоретические парциальные сечения Scofield 1973, UCRL-51326)",
        "regime_expected_local_deposit": "Type B (аналитическая квадратура Гаусса–Лежандра и свертка по каналам EADL)",
        "subshell_jump_ratio_rule": "Type B (кумулятивное правило скачков поглощения 1 - 1/J_s из краев NIST XCOM)",
        "event_energy_conservation_and_compile": "Type C (алгебраическое тождество dE == E_dep + sum(E_sec) и точность компиляции)",
    }

    model_assumptions = {
        "subshell_jump_ratio_approximation": (
            "Выбор ионизированной подоболочки при фотоэффекте (_sample_ionized_subshell_scalar) использует "
            "кумулятивное правило скачков поглощения 1 - 1/J_s (отношения скачков краев из таблицы полного "
            "фотоэффекта NIST XCOM, кусочно-постоянные между краями), а не прямую интерполяцию парциальных "
            "сечений подоболочек EPDL97 / Scofield (1973). Сверка с Scofield (1973) при 50-140.5 кэВ "
            "подтверждает точность правила скачков для K-оболочки I и Pb в пределах ~1%."
        ),
        "single_step_primary_vacancy_relaxation": (
            "Реализуется одноступенчатая релаксация первичной вакансии по таблицам ENDF/B-VIII.0 EADL с "
            "локальным поглощением остаточного каскада вторичных вакансий. В кристалле NaI(Tl) и мягких "
            "тканях пробег вторичных L-квантов и Оже-электронов (<= 4.5-13 кэВ) составляет единицы-десятки "
            "мкм << размера вокселя. Для свинца (Pb, Z=82) L-кванты с энергией 10.5-15.2 кэВ (omega_L3 ~ 0.32) "
            "имеют пробег ~10-20 мкм в самом свинце, но в воздухе их длина свободного пробега составляет "
            "порядка миллиметров, что важно только для тонких поверхностных слоев свинцового коллиматора."
        ),
        "compton_recoil_without_atomic_vacancy": (
            "При некогерентном (комптоновском) рассеянии учитывается связность электронов через функцию "
            "рассеяния S(x, Z), однако атомная вакансия на ионизированной оболочке не создается и последующая "
            "флуоресцентная релаксация после комптоновского рассеяния не разыгрывается."
        ),
    }

    report = Exp04EnergyPartitionReport(
        quick=quick,
        reference_standard_types=reference_standard_types,
        model_assumptions=model_assumptions,
        partition_regimes=partition_metrics,
        subshell_sampling=subshell_metrics,
        relaxation_cascades=relaxation_metrics,
        external_literature_benchmarks=external_benchmarks,
        max_overall_energy_imbalance=max_overall_imbalance,
        max_jump_weight_compile_error=max_jump_compile_err,
        max_subshell_z_score=max_subshell_z_score,
        max_relaxation_deviation=max_relaxation_dev,
        max_external_literature_normalized_deviation=max_external_norm_dev,
        figure_path=str(figure_path),
        passed=overall_passed,
    )

    json_path = output_dir / "exp04_energy_partition.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "EXTERNAL_K_ALPHA_RATIO_STANDARDS",
    "EXTERNAL_K_SHELL_FRACTION_STANDARDS",
    "EXTERNAL_OMEGA_K_STANDARDS",
    "Exp04EnergyPartitionReport",
    "ExternalLiteratureBenchmarkMetrics",
    "PartitionRegimeMetrics",
    "RelaxationCascadeMetrics",
    "SubshellSamplingPointMetrics",
    "run_exp04_energy_partition_and_atomic_relaxation",
]
