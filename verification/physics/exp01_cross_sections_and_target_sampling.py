"""Эксперимент 1: Верификация микрофизики таблиц сечений NIST XCOM, O(1) лог-индексации и выборки мишени Z.

Проверяет фундаментальные инварианты подсистемы материалов и сечений:
1. Сверку полных массовых коэффициентов ослабления mu/rho в MaterialBank с независимыми внешними
   табличными якорями стандарта NIST XCOM (Тип A, Berger et al., NIST SRD 8) для 5 референсных сред
   (Water, Dry Air, Cortical Bone, Sodium Iodide, Lead) на сетке энергий от 15 кэВ до 1 МэВ.
2. Побитовый и численный паритет двухуровневого O(1) логарифмического индекса MaterialBank
   (_compute_log_energy_and_bin + _locate_row_with_linear + _get_macroscopic_cross_sections)
   относительно эталонного бинарного поиска fast_interp_scalar (np.searchsorted(..., side='right')).
3. Строгую право-непрерывность на дублированных узлах K/L/M-краев поглощения (I, W, Pb)
   и сверку скачков фотопоглощения J_s = mu_pe(E_edge+) / mu_pe(E_edge-) с внешними табличными
   коэффициентами скачка McMaster et al. (UCRL-50174) / Scofield / NIST (Тип A) и базой релаксации.
4. Правило аддитивности Брэгга для многоэлементных сред (Water, NaI, Bone, Pyrex) как в узлах
   объединенной сетки (max_nodal_rel_error ~ 1e-16), так и в межузловых средних геометрических точках
   sqrt(E_i * E_{i+1}) (max_internodal_rel_error < 2.5e-3), а также нормировку ElementCSR.
5. Стохастическую несмещенность выбора атомного номера мишени _sample_Z_scalar по 3 процессам
   (фотоэффект, комптоновское и когерентное рассеяние) в широком энергетическом диапазоне (включая
   околокраевые энергии 15, 30, 33.0, 33.4, 88.5, 140.5, 511 кэВ) с контролем групповой ошибки
   I рода процедурой Холма–Бонферрони и масштабируемым от объема выборки N статистическим допуском.

Документированные модельные ограничения:
- Процесс рождения электрон-позитронных пар в кулоновском поле ядра и электронов (PairProduction,
  пороги 2*m_e*c^2 = 1.022 МэВ и 4*m_e*c^2 = 2.044 МэВ) в текущей редакции ядра представлен
  классом-заглушкой и не включен в таблицу макроскопических сечений MaterialBank; поэтому выше
  1.022 МэВ полный линейный коэффициент ослабления учитывает только три фотонных процесса ЯМ-диапазона
  (фотоэффект, некогерентное и когерентное рассеяние).
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
from core.materials.atomic_properties import atomic_number as element_atomic_numbers_map
from core.materials.attenuation_functions import fast_interp_scalar
from core.materials.material_bank import MaterialBank
from core.materials.materials import Material
from core.other.typing_definitions import Charge, Float, Index
from core.physics.interaction_buffers import RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import (
    _compute_log_energy_and_bin,
    _get_macroscopic_cross_sections,
    _locate_row_with_linear,
)
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
    Process,
)
from core.physics.processes_kernels import _sample_Z_scalar
from core.scene.nodes import CompositeNode
import settings.database_setting as database_setting
from verification.common.scattering_and_stats import (
    ReferenceStandardType,
    compute_chi2_pvalue,
    evaluate_holm_bonferroni,
)
from verification.physics.plotting import plot_exp01_cross_sections_and_target_sampling

PROCESS_PHOTOELECTRIC: int = 0
PROCESS_COMPTON: int = 1
PROCESS_COHERENT: int = 2

PROCESS_LABELS: dict[int, str] = {
    PROCESS_PHOTOELECTRIC: "PhotoelectricEffect",
    PROCESS_COMPTON: "ComptonScattering",
    PROCESS_COHERENT: "CoherentScattering",
}

PROCESS_SHORT_LABELS: dict[int, str] = {
    PROCESS_PHOTOELECTRIC: "PE",
    PROCESS_COMPTON: "Comp",
    PROCESS_COHERENT: "Coh",
}

MATERIAL_CATALOG: dict[str, str] = {
    "Water": "Water, Liquid",
    "Bone": "Bone, Cortical (ICRU-44)",
    "NaI": "Sodium Iodide",
    "Lead": "Pb",
    "Tungsten": "W",
    "Pyrex": "Glass, Borosilicate (Pyrex)",
    "Air": "Air, Dry (near sea level)",
}

MASS_ATTENUATION_UNIT: float = float(units.cm2 / units.g)

# Независимые табличные якоря полных массовых коэффициентов ослабления mu/rho (с когерентным рассеянием)
# из стандарта NIST XCOM (Berger M.J. et al., NIST Standard Reference Database 8 (XGAM), NBSIR 87-3597,
# DOI: 10.18434/T48G6X) в диапазоне энергий от 15 кэВ до 1 МэВ (до порога парного рождения 1.022 МэВ).
NIST_XCOM_EXTERNAL_MAC_REFERENCE: dict[str, tuple[tuple[float, float], ...]] = {
    "Water": (
        (float(15.0 * units.keV), 1.673 * MASS_ATTENUATION_UNIT),
        (float(30.0 * units.keV), 0.3756 * MASS_ATTENUATION_UNIT),
        (float(80.0 * units.keV), 0.1837 * MASS_ATTENUATION_UNIT),
        (float(140.5 * units.keV), 0.1536 * MASS_ATTENUATION_UNIT),
        (float(511.0 * units.keV), 0.09596 * MASS_ATTENUATION_UNIT),
        (float(1.0 * units.MeV), 0.07072 * MASS_ATTENUATION_UNIT),
    ),
    "Air": (
        (float(15.0 * units.keV), 1.614 * MASS_ATTENUATION_UNIT),
        (float(30.0 * units.keV), 0.3538 * MASS_ATTENUATION_UNIT),
        (float(80.0 * units.keV), 0.1662 * MASS_ATTENUATION_UNIT),
        (float(140.5 * units.keV), 0.1383 * MASS_ATTENUATION_UNIT),
        (float(511.0 * units.keV), 0.08631 * MASS_ATTENUATION_UNIT),
        (float(1.0 * units.MeV), 0.06358 * MASS_ATTENUATION_UNIT),
    ),
    "Bone": (
        (float(15.0 * units.keV), 9.032 * MASS_ATTENUATION_UNIT),
        (float(30.0 * units.keV), 1.331 * MASS_ATTENUATION_UNIT),
        (float(80.0 * units.keV), 0.2229 * MASS_ATTENUATION_UNIT),
        (float(140.5 * units.keV), 0.1526 * MASS_ATTENUATION_UNIT),
        (float(511.0 * units.keV), 0.08937 * MASS_ATTENUATION_UNIT),
        (float(1.0 * units.MeV), 0.06565 * MASS_ATTENUATION_UNIT),
    ),
    "NaI": (
        (float(15.0 * units.keV), 47.38 * MASS_ATTENUATION_UNIT),
        (float(30.0 * units.keV), 7.358 * MASS_ATTENUATION_UNIT),
        (float(80.0 * units.keV), 2.999 * MASS_ATTENUATION_UNIT),
        (float(140.5 * units.keV), 0.7135 * MASS_ATTENUATION_UNIT),
        (float(511.0 * units.keV), 0.09326 * MASS_ATTENUATION_UNIT),
        (float(1.0 * units.MeV), 0.05881 * MASS_ATTENUATION_UNIT),
    ),
    "Lead": (
        (float(15.0 * units.keV), 111.6 * MASS_ATTENUATION_UNIT),
        (float(30.0 * units.keV), 30.32 * MASS_ATTENUATION_UNIT),
        (float(80.0 * units.keV), 2.419 * MASS_ATTENUATION_UNIT),
        (float(140.5 * units.keV), 2.370 * MASS_ATTENUATION_UNIT),
        (float(511.0 * units.keV), 0.1562 * MASS_ATTENUATION_UNIT),
        (float(1.0 * units.MeV), 0.07102 * MASS_ATTENUATION_UNIT),
    ),
}

# Внешние табличные коэффициенты скачков фотопоглощения J_s на K/L/M-краях
# Источники: McMaster W.H. et al., "Compilation of X-Ray Cross Sections", UCRL-50174 Sec. II Rev. 1 (1969);
# Scofield J.H., UCRL-51326 (1973); NIST Standard Reference Database 8 (XCOM).
EXTERNAL_EDGE_JUMP_REFERENCE: dict[tuple[str, str], float] = {
    ("I", "K"): 6.04,
    ("I", "L1"): 1.154,
    ("I", "L2"): 1.349,
    ("I", "L3"): 2.953,
    ("W", "K"): 5.10,
    ("W", "L1"): 1.156,
    ("W", "L2"): 1.378,
    ("W", "L3"): 2.615,
    ("W", "M1"): 1.043,
    ("Pb", "K"): 4.73,
    ("Pb", "L1"): 1.156,
    ("Pb", "L2"): 1.390,
    ("Pb", "L3"): 2.507,
    ("Pb", "M1"): 1.043,
}

DOCUMENTED_MODEL_LIMITATIONS: tuple[str, ...] = (
    "Процесс рождения электрон-позитронных пар (PairProduction, порог 1.022 МэВ) представлен "
    "классом-заглушкой и не включен в сумму макроскопических сечений MaterialBank; при E > 1.022 МэВ "
    "полный коэффициент ослабления учитывает только фотоэффект, комптоновское и когерентное рассеяние.",
    "В многоэлементных средах парциальные доли выбора мишени ElementCSR линейно интерполируются "
    "по ln(E) между узлами объединенной сетки, что дает методическое расхождение <= 3.1e-4 относительно "
    "точной кусочно-степенной (log-log) интерполяции парциальных сечений Брэгга.",
)


@dataclass(frozen=True)
class NistXcomReferenceMetric:
    """Метрика сверки массового коэффициента ослабления mu/rho в ядре со стандартом NIST XCOM (Тип A)."""

    material_label: str
    material_name: str
    photon_energy: float
    mac_kernel: float
    mac_nist_xcom: float
    rel_error: float
    passed: bool


@dataclass(frozen=True)
class BinningParityMetric:
    """Метрики сверки O(1) логарифмического биннинга MaterialBank с эталонным searchsorted."""

    material_label: str
    material_name: str
    num_grid_points: int
    max_rel_error_pe: float
    max_rel_error_compton: float
    max_rel_error_coherent: float
    max_rel_error_total: float
    row_mismatches: int
    passed: bool


@dataclass(frozen=True)
class EdgeJumpMetric:
    """Метрики право-непрерывности и величины скачка сечения фотоэффекта на крае поглощения."""

    element_symbol: str
    edge_label: str
    edge_energy: float
    lac_below_delta: float
    lac_at_edge: float
    lac_above_delta: float
    lac_node_left: float
    lac_node_right: float
    right_continuity_rel_error: float
    delta_jump_ratio: float
    node_jump_ratio: float
    reference_jump_ratio: float
    delta_vs_node_jump_rel_error: float
    jump_ratio_rel_error: float
    relaxation_db_jump_ratio: float
    relaxation_db_consistency_rel_error: float
    passed: bool


@dataclass(frozen=True)
class BraggAdditivityMetric:
    """Метрики проверки правила аддитивности Брэгга (в узлах и между узлами) и нормировки ElementCSR."""

    material_label: str
    material_name: str
    num_elements: int
    num_grid_rows: int
    num_internodal_points: int
    max_rel_diff_lac: float
    max_nodal_rel_error: float
    max_internodal_rel_error: float
    max_rel_diff_partial_reconstruction: float
    max_abs_fraction_sum_error: float
    passed: bool


@dataclass(frozen=True)
class TargetSamplingMetric:
    """Метрики стохастической верификации выбора элемента-мишени _sample_Z_scalar."""

    material_label: str
    process_name: str
    process_col_idx: int
    photon_energy: float
    num_samples: int
    element_symbols: list[str]
    element_atomic_numbers: list[int]
    theoretical_probabilities: list[float]
    bragg_exact_probabilities: list[float]
    empirical_probabilities: list[float]
    dominant_element_symbol: str
    dominant_theoretical_prob: float
    dominant_empirical_prob: float
    max_interpolation_scheme_diff: float
    max_abs_probability_diff: float
    allowed_probability_diff: float
    max_abs_z_score: float
    chi2_stat: float
    chi2_ndf: int
    chi2_pvalue: float
    holm_threshold: float
    passed: bool


@dataclass(frozen=True)
class Exp01CrossSectionsReport:
    """Корневой отчет верификации сечений, O(1) лог-индексации и выборки элемента-мишени (Эксперимент 1)."""

    passed: bool
    quick: bool
    num_binning_points: int
    num_mc_samples: int
    max_nist_xcom_rel_error: float
    max_binning_rel_error: float
    total_binning_row_mismatches: int
    max_edge_continuity_rel_error: float
    max_edge_delta_vs_node_rel_error: float
    max_external_jump_ratio_rel_error: float
    lead_k_jump_ratio: float
    iodine_k_jump_ratio: float
    tungsten_k_jump_ratio: float
    max_bragg_rel_error: float
    max_bragg_internodal_rel_error: float
    max_interpolation_scheme_diff: float
    max_sampling_z_score: float
    max_sampling_abs_diff: float
    sampling_holm_bonferroni_passed: bool
    min_sampling_chi2_pvalue: float
    reference_standard_types: dict[str, str]
    model_limitations: list[str]
    nist_xcom_metrics: list[NistXcomReferenceMetric]
    binning_metrics: list[BinningParityMetric]
    edge_metrics: list[EdgeJumpMetric]
    bragg_metrics: list[BraggAdditivityMetric]
    target_sampling_metrics: list[TargetSamplingMetric]
    json_path: str
    plot_path: str


@njit(cache=True)
def _evaluate_o1_binning_vs_searchsorted_kernel(
    query_energies: NDArray[Float],
    material_id: Index,
    material_bank: MaterialBank,
    ref_grid_pe: NDArray[Float],
    ref_lac_pe: NDArray[Float],
    ref_grid_comp: NDArray[Float],
    ref_lac_comp: NDArray[Float],
    ref_grid_coh: NDArray[Float],
    ref_lac_coh: NDArray[Float],
) -> tuple[NDArray[Float], NDArray[Float], NDArray[Float], int]:
    """Сравнивает O(1) лог-индексацию MaterialBank с эталонным fast_interp_scalar на сетке энергий."""
    num_points = query_energies.shape[0]
    rel_err_pe = np.zeros(num_points, dtype=Float)
    rel_err_comp = np.zeros(num_points, dtype=Float)
    rel_err_coh = np.zeros(num_points, dtype=Float)
    lacs_o1 = np.empty(3, dtype=Float)

    pointer = material_bank.mat_pointers[material_id]
    start_idx = Index(pointer["start_idx"])
    length = Index(pointer["length"])
    mat_energy_grid = material_bank.physics_energy_grid[start_idx : start_idx + length]

    row_mismatches = 0

    for pt_idx in range(num_points):
        energy_val = Float(query_energies[pt_idx])
        log_e, log_bin = _compute_log_energy_and_bin(energy_val, material_bank)
        row_o1, _, _, _ = _locate_row_with_linear(
            energy_val, log_e, log_bin, material_id, material_bank
        )
        _get_macroscopic_cross_sections(
            energy_val, log_e, log_bin, material_id, material_bank, lacs_o1
        )

        # Эталонный индекс строки через searchsorted(..., side='right')
        if energy_val < mat_energy_grid[0]:
            expected_row = start_idx
        elif energy_val >= mat_energy_grid[length - 1]:
            expected_row = Index(start_idx + length - 1)
        else:
            local_pos = int(np.searchsorted(mat_energy_grid, energy_val, side="right")) - 1
            if local_pos < 0:
                local_pos = 0
            elif local_pos >= length - 1:
                local_pos = int(length - 2)
            expected_row = Index(start_idx + local_pos)

        if row_o1 != expected_row:
            row_mismatches += 1

        ref_pe = fast_interp_scalar(energy_val, ref_grid_pe, ref_lac_pe)
        ref_comp = fast_interp_scalar(energy_val, ref_grid_comp, ref_lac_comp)
        ref_coh = fast_interp_scalar(energy_val, ref_grid_coh, ref_lac_coh)

        if ref_pe > 0.0:
            rel_err_pe[pt_idx] = abs(lacs_o1[0] - ref_pe) / ref_pe
        else:
            rel_err_pe[pt_idx] = abs(lacs_o1[0] - ref_pe)

        if ref_comp > 0.0:
            rel_err_comp[pt_idx] = abs(lacs_o1[1] - ref_comp) / ref_comp
        else:
            rel_err_comp[pt_idx] = abs(lacs_o1[1] - ref_comp)

        if ref_coh > 0.0:
            rel_err_coh[pt_idx] = abs(lacs_o1[2] - ref_coh) / ref_coh
        else:
            rel_err_coh[pt_idx] = abs(lacs_o1[2] - ref_coh)

    return rel_err_pe, rel_err_comp, rel_err_coh, row_mismatches


@njit(cache=True)
def _query_cross_sections_batch_kernel(
    query_energies: NDArray[Float],
    material_id: Index,
    material_bank: MaterialBank,
) -> NDArray[Float]:
    """Вычисляет матрицу макроскопических сечений (N, 3) через _get_macroscopic_cross_sections."""
    num_points = query_energies.shape[0]
    out_matrix = np.zeros((num_points, 3), dtype=Float)
    row_slice = np.empty(3, dtype=Float)
    for pt_idx in range(num_points):
        energy_val = Float(query_energies[pt_idx])
        log_e, log_bin = _compute_log_energy_and_bin(energy_val, material_bank)
        _get_macroscopic_cross_sections(
            energy_val, log_e, log_bin, material_id, material_bank, row_slice
        )
        out_matrix[pt_idx, 0] = row_slice[0]
        out_matrix[pt_idx, 1] = row_slice[1]
        out_matrix[pt_idx, 2] = row_slice[2]
    return out_matrix


@njit(cache=True)
def _sample_z_batch_kernel(
    material_id: Index,
    process_col_idx: Index,
    photon_energy: Float,
    num_samples: int,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
) -> NDArray[Charge]:
    """Генерирует массив из num_samples выборок атомного номера мишени Z ядром _sample_Z_scalar."""
    sampled_z = np.empty(num_samples, dtype=Charge)
    for sample_idx in range(num_samples):
        sampled_z[sample_idx] = _sample_Z_scalar(
            material_id,
            process_col_idx,
            photon_energy,
            physics_buffer,
            rng_ctx,
        )
    return sampled_z


def _build_verification_physics_buffer(
    materials_list: list[Material],
    processes_list: list[Process],
) -> PhysicsBuffer:
    """Компилирует PhysicsBuffer через публичный продакшен-конвейер PhysicsCompiler.compile_scene."""
    scene_root = CompositeNode(name="Exp01VerificationRoot")
    sample_box_size = Float(10.0 * units.mm)
    for mat_idx, material_item in enumerate(materials_list):
        sample_volume = Volume(
            geometry=Box(sample_box_size, sample_box_size, sample_box_size),
            material=material_item,
            name=f"SampleVolume_{mat_idx}_{material_item.ID}",
        )
        scene_root.add_child(sample_volume)

    compiler = PhysicsCompiler()
    return compiler.compile_scene(scene_root, processes_list)


def _run_nist_xcom_reference_audit(
    physics_buffer: PhysicsBuffer,
) -> list[NistXcomReferenceMetric]:
    """Сверяет массовые коэффициенты ослабления mu/rho из MaterialBank с таблицами NIST XCOM (Тип A)."""
    material_bank = physics_buffer.material_bank
    metrics: list[NistXcomReferenceMetric] = []

    for mat_label, reference_points in NIST_XCOM_EXTERNAL_MAC_REFERENCE.items():
        mat_db_name = MATERIAL_CATALOG[mat_label]
        material = database_setting.material_database[mat_db_name]
        mat_id = Index(material.ID)
        density_val = float(material.density)
        if density_val <= 0.0:
            raise ValueError(f"Плотность материала '{mat_db_name}' должна быть положительной.")

        query_energies = np.array([pt[0] for pt in reference_points], dtype=Float)
        lacs_matrix = _query_cross_sections_batch_kernel(query_energies, mat_id, material_bank)

        for pt_idx, (energy_val, ref_mac) in enumerate(reference_points):
            total_lac = float(
                lacs_matrix[pt_idx, PROCESS_PHOTOELECTRIC]
                + lacs_matrix[pt_idx, PROCESS_COMPTON]
                + lacs_matrix[pt_idx, PROCESS_COHERENT]
            )
            kernel_mac = total_lac / density_val
            rel_err = abs(kernel_mac - ref_mac) / ref_mac
            # Допуск 0.5% учитывает 4-значное округление справочных таблиц NIST XCOM
            item_passed = rel_err < 5.0e-3
            metrics.append(
                NistXcomReferenceMetric(
                    material_label=mat_label,
                    material_name=mat_db_name,
                    photon_energy=energy_val,
                    mac_kernel=kernel_mac,
                    mac_nist_xcom=ref_mac,
                    rel_error=rel_err,
                    passed=item_passed,
                )
            )

    return metrics


def _run_binning_parity_audit(
    physics_buffer: PhysicsBuffer,
    processes_list: list[Process],
    num_grid_points: int,
) -> tuple[list[BinningParityMetric], NDArray[np.float64], NDArray[np.float64]]:
    """Выполняет сверку O(1) логарифмического биннинга с эталонным бинарным поиском searchsorted."""
    material_bank = physics_buffer.material_bank
    min_energy = float(1.0 * units.keV)
    max_energy = float(20.0 * units.MeV)
    grid_energies = np.geomspace(min_energy, max_energy, num_grid_points, dtype=Float)
    max_rel_diff_across_materials = np.zeros(num_grid_points, dtype=np.float64)

    pe_proc = processes_list[PROCESS_PHOTOELECTRIC]
    comp_proc = processes_list[PROCESS_COMPTON]
    coh_proc = processes_list[PROCESS_COHERENT]

    metrics: list[BinningParityMetric] = []
    for mat_label, mat_db_name in MATERIAL_CATALOG.items():
        material = database_setting.material_database[mat_db_name]
        mat_id = Index(material.ID)

        ref_grid_pe, ref_lac_pe = pe_proc.attenuation_function[material]
        ref_grid_comp, ref_lac_comp = comp_proc.attenuation_function[material]
        ref_grid_coh, ref_lac_coh = coh_proc.attenuation_function[material]

        rel_pe, rel_comp, rel_coh, row_mismatches = _evaluate_o1_binning_vs_searchsorted_kernel(
            query_energies=grid_energies,
            material_id=mat_id,
            material_bank=material_bank,
            ref_grid_pe=ref_grid_pe,
            ref_lac_pe=ref_lac_pe,
            ref_grid_comp=ref_grid_comp,
            ref_lac_comp=ref_lac_comp,
            ref_grid_coh=ref_grid_coh,
            ref_lac_coh=ref_lac_coh,
        )

        pointwise_max = np.maximum(np.maximum(rel_pe, rel_comp), rel_coh)
        max_rel_diff_across_materials = np.maximum(max_rel_diff_across_materials, pointwise_max)

        max_pe = float(np.max(rel_pe))
        max_comp = float(np.max(rel_comp))
        max_coh = float(np.max(rel_coh))
        max_tot = float(np.max(pointwise_max))
        item_passed = (row_mismatches == 0) and (max_tot < 1e-14)

        metrics.append(
            BinningParityMetric(
                material_label=mat_label,
                material_name=mat_db_name,
                num_grid_points=num_grid_points,
                max_rel_error_pe=max_pe,
                max_rel_error_compton=max_comp,
                max_rel_error_coherent=max_coh,
                max_rel_error_total=max_tot,
                row_mismatches=int(row_mismatches),
                passed=item_passed,
            )
        )

    return metrics, np.asarray(grid_energies, dtype=np.float64), max_rel_diff_across_materials


def _run_edge_jump_audit(
    physics_buffer: PhysicsBuffer,
) -> list[EdgeJumpMetric]:
    """Проверяет право-непрерывность и скачки фотоэффекта J_s против внешних таблиц McMaster/Scofield/NIST."""
    material_bank = physics_buffer.material_bank
    attenuation_db = database_setting.attenuation_database
    edge_delta = float(1.0e-7 * units.MeV)

    target_edges: list[tuple[str, tuple[str, ...]]] = [
        ("I", ("K", "L1", "L2", "L3")),
        ("W", ("K", "L1", "L2", "L3", "M1")),
        ("Pb", ("K", "L1", "L2", "L3", "M1")),
    ]

    edge_metrics: list[EdgeJumpMetric] = []
    for elem_sym, requested_labels in target_edges:
        material = database_setting.material_database[elem_sym]
        mat_id = Index(material.ID)
        pointer = material_bank.mat_pointers[mat_id]
        row_start = int(pointer["start_idx"])
        row_length = int(pointer["length"])
        mat_grid = material_bank.physics_energy_grid[row_start : row_start + row_length]
        mat_pe_lacs = material_bank.physics_lac_table[row_start : row_start + row_length, PROCESS_PHOTOELECTRIC]

        edge_energies, edge_labels = attenuation_db.get_element_edges(elem_sym)
        relaxation_data = attenuation_db.get_element_relaxation(elem_sym)

        for req_label in requested_labels:
            label_matches = np.flatnonzero(edge_labels == req_label)
            if label_matches.size == 0:
                raise ValueError(
                    f"Обязательный край поглощения '{req_label}' отсутствует в таблице Edges элемента '{elem_sym}'."
                )
            edge_idx = int(label_matches[0])
            edge_energy = float(edge_energies[edge_idx])

            node_indices = np.flatnonzero(np.isclose(mat_grid, edge_energy, rtol=1e-12, atol=0.0))
            if node_indices.size < 2:
                raise ValueError(
                    f"В сетке MaterialBank элемента '{elem_sym}' отсутствует дублированная пара узлов "
                    f"для края '{req_label}' при E={edge_energy}."
                )
            left_node_idx = int(node_indices[0])
            right_node_idx = int(node_indices[-1])
            lac_node_left = float(mat_pe_lacs[left_node_idx])
            lac_node_right = float(mat_pe_lacs[right_node_idx])

            rel_shell_matches = np.flatnonzero(relaxation_data.subshell_label == req_label)
            if rel_shell_matches.size == 0:
                raise ValueError(
                    f"Подоболочка '{req_label}' отсутствует в структуре ElementRelaxationData элемента '{elem_sym}'."
                )
            relaxation_db_jump = float(relaxation_data.jump_ratio[int(rel_shell_matches[0])])

            if (elem_sym, req_label) not in EXTERNAL_EDGE_JUMP_REFERENCE:
                raise ValueError(
                    f"Отсутствует внешний эталон McMaster/NIST для скачка края ({elem_sym}, {req_label})."
                )
            external_ref_jump = EXTERNAL_EDGE_JUMP_REFERENCE[(elem_sym, req_label)]

            probe_energies = np.array(
                [edge_energy - edge_delta, edge_energy, edge_energy + edge_delta],
                dtype=Float,
            )
            probe_lacs = _query_cross_sections_batch_kernel(probe_energies, mat_id, material_bank)
            lac_below_delta = float(probe_lacs[0, PROCESS_PHOTOELECTRIC])
            lac_at_edge = float(probe_lacs[1, PROCESS_PHOTOELECTRIC])
            lac_above_delta = float(probe_lacs[2, PROCESS_PHOTOELECTRIC])

            right_continuity_rel_error = abs(lac_at_edge - lac_node_right) / max(lac_node_right, 1e-30)
            node_jump_ratio = lac_node_right / max(lac_node_left, 1e-30)
            delta_jump_ratio = lac_above_delta / max(lac_below_delta, 1e-30)
            delta_vs_node_jump_rel_error = abs(delta_jump_ratio - node_jump_ratio) / max(node_jump_ratio, 1e-30)

            jump_ratio_rel_error = abs(node_jump_ratio - external_ref_jump) / max(external_ref_jump, 1e-30)
            relaxation_db_consistency_rel_error = (
                abs(node_jump_ratio - relaxation_db_jump) / max(relaxation_db_jump, 1e-30)
            )

            delta_continuity_rel = abs(lac_at_edge - lac_above_delta) / max(lac_at_edge, 1e-30)
            passed = (
                (right_continuity_rel_error < 1e-14)
                and (delta_continuity_rel < 1e-3)
                and (delta_vs_node_jump_rel_error < 5e-4)
                and (jump_ratio_rel_error < 1.5e-2)
                and (relaxation_db_consistency_rel_error < 1e-12)
            )

            edge_metrics.append(
                EdgeJumpMetric(
                    element_symbol=elem_sym,
                    edge_label=req_label,
                    edge_energy=edge_energy,
                    lac_below_delta=lac_below_delta,
                    lac_at_edge=lac_at_edge,
                    lac_above_delta=lac_above_delta,
                    lac_node_left=lac_node_left,
                    lac_node_right=lac_node_right,
                    right_continuity_rel_error=right_continuity_rel_error,
                    delta_jump_ratio=delta_jump_ratio,
                    node_jump_ratio=node_jump_ratio,
                    reference_jump_ratio=external_ref_jump,
                    delta_vs_node_jump_rel_error=delta_vs_node_jump_rel_error,
                    jump_ratio_rel_error=jump_ratio_rel_error,
                    relaxation_db_jump_ratio=relaxation_db_jump,
                    relaxation_db_consistency_rel_error=relaxation_db_consistency_rel_error,
                    passed=passed,
                )
            )

    return edge_metrics


def _reference_loglog_interpolate_element_mac(
    material_energy: NDArray[Float],
    element_energy: NDArray[Float],
    element_coeff: NDArray[Float],
) -> NDArray[np.float64]:
    """Независимая эталонная кусочно-степенная (log-log) интерполяция MAC элемента на сетку материала.

    Корректно разрешает односторонние пределы E_edge^- и E_edge^+ на дублированных узлах
    краев поглощения без вызова функций ядра.
    """
    mat_size = int(material_energy.shape[0])
    el_size = int(element_energy.shape[0])
    interpolated = np.zeros(mat_size, dtype=np.float64)

    for node_idx in range(mat_size):
        query_e = float(material_energy[node_idx])
        is_left_limit_in_mat = (node_idx + 1 < mat_size) and (
            float(material_energy[node_idx + 1]) == query_e
        )

        exact_matches = np.flatnonzero(element_energy == material_energy[node_idx])
        if exact_matches.size >= 2:
            chosen_el_idx = int(exact_matches[0]) if is_left_limit_in_mat else int(exact_matches[-1])
            interpolated[node_idx] = float(element_coeff[chosen_el_idx])
        elif exact_matches.size == 1:
            interpolated[node_idx] = float(element_coeff[int(exact_matches[0])])
        else:
            upper_idx = int(np.searchsorted(element_energy, query_e, side="right"))
            if upper_idx <= 0:
                interpolated[node_idx] = float(element_coeff[0])
            elif upper_idx >= el_size:
                interpolated[node_idx] = float(element_coeff[el_size - 1])
            else:
                energy_low = float(element_energy[upper_idx - 1])
                energy_high = float(element_energy[upper_idx])
                mac_low = float(element_coeff[upper_idx - 1])
                mac_high = float(element_coeff[upper_idx])
                power_exponent = math.log(mac_high / mac_low) / math.log(energy_high / energy_low)
                interpolated[node_idx] = mac_low * math.pow(query_e / energy_low, power_exponent)

    return interpolated


def _run_bragg_additivity_audit(
    physics_buffer: PhysicsBuffer,
) -> list[BraggAdditivityMetric]:
    """Проверяет правило аддитивности Брэгга в узлах сетки и между узлами (sqrt(E_i * E_{i+1}))."""
    material_bank = physics_buffer.material_bank
    element_csr = physics_buffer.element_csr
    attenuation_db = database_setting.attenuation_database
    compound_labels = ("Water", "NaI", "Bone", "Pyrex")
    process_names_tuple = ("PhotoelectricEffect", "ComptonScattering", "CoherentScattering")

    bragg_metrics: list[BraggAdditivityMetric] = []
    for mat_label in compound_labels:
        mat_db_name = MATERIAL_CATALOG[mat_label]
        material = database_setting.material_database[mat_db_name]
        mat_id = int(material.ID)

        pointer = material_bank.mat_pointers[mat_id]
        row_start = int(pointer["start_idx"])
        row_length = int(pointer["length"])
        mat_grid = material_bank.physics_energy_grid[row_start : row_start + row_length]
        mat_lacs = material_bank.physics_lac_table[row_start : row_start + row_length, :]

        comp_items = list(material.composition_dict.items())
        num_elements = len(comp_items)

        fractions_block = element_csr.element_partial_fractions[
            row_start : row_start + row_length, :, :num_elements
        ]
        fraction_sums = np.sum(fractions_block, axis=2)
        max_abs_fraction_sum_err = float(np.max(np.abs(fraction_sums - 1.0)))

        # Сетка межузловых средних геометрических точек sqrt(E_i * E_{i+1}) на невырожденных интервалах
        internodal_list: list[float] = []
        for idx in range(row_length - 1):
            e_left = float(mat_grid[idx])
            e_right = float(mat_grid[idx + 1])
            if e_right > e_left * (1.0 + 1.0e-12):
                internodal_list.append(math.sqrt(e_left * e_right))
        internodal_energies = np.asarray(internodal_list, dtype=Float)
        internodal_kernel_lacs = _query_cross_sections_batch_kernel(
            internodal_energies, Index(mat_id), material_bank
        )

        max_nodal_rel_err = 0.0
        max_internodal_rel_err = 0.0
        max_rel_diff_partial = 0.0

        for proc_idx, proc_name in enumerate(process_names_tuple):
            direct_partial_lacs = np.zeros((row_length, num_elements), dtype=np.float64)
            internodal_partial_lacs = np.zeros(
                (internodal_energies.shape[0], num_elements), dtype=np.float64
            )
            for elem_offset, (elem_sym, mass_weight) in enumerate(comp_items):
                el_table = attenuation_db.elements_MAC[elem_sym]
                interp_mac = _reference_loglog_interpolate_element_mac(
                    mat_grid,
                    el_table["Energy"],
                    el_table["Coefficient"][proc_name],
                )
                direct_partial_lacs[:, elem_offset] = (
                    float(material.density) * float(mass_weight) * interp_mac
                )

                internodal_mac = _reference_loglog_interpolate_element_mac(
                    internodal_energies,
                    el_table["Energy"],
                    el_table["Coefficient"][proc_name],
                )
                internodal_partial_lacs[:, elem_offset] = (
                    float(material.density) * float(mass_weight) * internodal_mac
                )

            direct_total_lac = np.sum(direct_partial_lacs, axis=1)
            table_total_lac = np.asarray(mat_lacs[:, proc_idx], dtype=np.float64)

            rel_diff_total = np.abs(table_total_lac - direct_total_lac) / np.maximum(direct_total_lac, 1e-30)
            max_nodal_rel_err = max(max_nodal_rel_err, float(np.max(rel_diff_total)))

            internodal_direct_total = np.sum(internodal_partial_lacs, axis=1)
            internodal_table_interp = np.asarray(internodal_kernel_lacs[:, proc_idx], dtype=np.float64)
            internodal_rel_diff = np.abs(internodal_table_interp - internodal_direct_total) / np.maximum(
                internodal_direct_total, 1e-30
            )
            max_internodal_rel_err = max(max_internodal_rel_err, float(np.max(internodal_rel_diff)))

            reconstructed_partial = (
                np.asarray(fractions_block[:, proc_idx, :], dtype=np.float64)
                * table_total_lac[:, np.newaxis]
            )
            rel_diff_part = np.abs(reconstructed_partial - direct_partial_lacs) / np.maximum(
                table_total_lac[:, np.newaxis], 1e-30
            )
            max_rel_diff_partial = max(max_rel_diff_partial, float(np.max(rel_diff_part)))

        passed = (
            (max_nodal_rel_err < 1e-12)
            and (max_internodal_rel_err < 2.5e-3)
            and (max_rel_diff_partial < 1e-12)
            and (max_abs_fraction_sum_err < 1e-12)
        )
        bragg_metrics.append(
            BraggAdditivityMetric(
                material_label=mat_label,
                material_name=mat_db_name,
                num_elements=num_elements,
                num_grid_rows=row_length,
                num_internodal_points=int(internodal_energies.shape[0]),
                max_rel_diff_lac=max_nodal_rel_err,
                max_nodal_rel_error=max_nodal_rel_err,
                max_internodal_rel_error=max_internodal_rel_err,
                max_rel_diff_partial_reconstruction=max_rel_diff_partial,
                max_abs_fraction_sum_error=max_abs_fraction_sum_err,
                passed=passed,
            )
        )

    return bragg_metrics


def _compute_reference_element_probabilities(
    material: Material,
    process_col_idx: int,
    process_name: str,
    photon_energy: float,
    physics_buffer: PhysicsBuffer,
) -> tuple[list[str], list[int], list[float], list[float]]:
    """Вычисляет два эталона вероятностей выбора элемента P(Z_i | p, E):

    1. kernel_exact_probabilities: линейная интерполяция узловых парциальных долей ElementCSR
       по логарифмической доле энергии t = (ln E - ln E_k) / (ln E_{k+1} - ln E_k).
    2. bragg_exact_probabilities: точное отношение кусочно-степенных (log-log) парциальных сечений Брэгга.
    """
    attenuation_db = database_setting.attenuation_database
    comp_items = list(material.composition_dict.items())
    num_elements = len(comp_items)
    symbols: list[str] = []
    atomic_numbers_list: list[int] = []
    bragg_weights: list[float] = []

    query_e = Float(photon_energy)
    for elem_sym, mass_weight in comp_items:
        el_table = attenuation_db.elements_MAC[elem_sym]
        el_energy = np.ascontiguousarray(el_table["Energy"], dtype=Float)
        el_coeff = np.ascontiguousarray(el_table["Coefficient"][process_name], dtype=Float)
        mac_val = float(fast_interp_scalar(query_e, el_energy, el_coeff))
        symbols.append(elem_sym)
        atomic_numbers_list.append(int(element_atomic_numbers_map[elem_sym]))
        bragg_weights.append(float(mass_weight) * mac_val)

    total_bragg_weight = sum(bragg_weights)
    bragg_exact_probabilities = [val / total_bragg_weight for val in bragg_weights]

    mat_id = int(material.ID)
    pointer = physics_buffer.material_bank.mat_pointers[mat_id]
    row_start = int(pointer["start_idx"])
    row_length = int(pointer["length"])
    mat_grid = physics_buffer.material_bank.physics_energy_grid[row_start : row_start + row_length]

    local_pos = int(np.searchsorted(mat_grid, photon_energy, side="right")) - 1
    if local_pos < 0:
        local_pos = 0
        log_fraction = 0.0
    elif local_pos >= row_length - 1:
        local_pos = row_length - 1
        log_fraction = 0.0
    else:
        energy_low = float(mat_grid[local_pos])
        energy_high = float(mat_grid[local_pos + 1])
        if energy_high > energy_low and photon_energy > energy_low:
            log_fraction = (math.log(photon_energy) - math.log(energy_low)) / (
                math.log(energy_high) - math.log(energy_low)
            )
        else:
            log_fraction = 0.0

    abs_row = row_start + local_pos
    fractions_table = physics_buffer.element_csr.element_partial_fractions
    kernel_exact_probabilities: list[float] = []
    if num_elements == 1:
        kernel_exact_probabilities = [1.0]
    else:
        accum_prob = 0.0
        for elem_offset in range(num_elements - 1):
            weight_low = float(fractions_table[abs_row, process_col_idx, elem_offset])
            if log_fraction > 0.0 and (abs_row + 1) < fractions_table.shape[0]:
                weight_high = float(fractions_table[abs_row + 1, process_col_idx, elem_offset])
                interp_weight = (1.0 - log_fraction) * weight_low + log_fraction * weight_high
            else:
                interp_weight = weight_low
            kernel_exact_probabilities.append(interp_weight)
            accum_prob += interp_weight
        kernel_exact_probabilities.append(max(0.0, 1.0 - accum_prob))

    return symbols, atomic_numbers_list, kernel_exact_probabilities, bragg_exact_probabilities


def _compute_multinomial_sampling_chi2(
    observed_counts: list[int],
    expected_probs: list[float],
    num_samples: int,
    min_expected: float = 5.0,
) -> tuple[float, int, float]:
    """Вычисляет статистику хи-квадрат Пирсона и p-значение для мультиномиальной выборки Z."""
    samples_float = float(num_samples)
    pooled_obs: list[float] = []
    pooled_exp: list[float] = []
    tail_obs = 0.0
    tail_exp = 0.0

    for obs_val, prob_val in zip(observed_counts, expected_probs):
        exp_val = samples_float * prob_val
        if exp_val >= min_expected:
            pooled_obs.append(float(obs_val))
            pooled_exp.append(exp_val)
        else:
            tail_obs += float(obs_val)
            tail_exp += exp_val

    if tail_exp > 0.0:
        if tail_exp >= min_expected or len(pooled_exp) == 1:
            pooled_obs.append(tail_obs)
            pooled_exp.append(max(tail_exp, 1.0e-12))
        elif len(pooled_exp) > 1:
            smallest_idx = int(np.argmin(pooled_exp))
            pooled_obs[smallest_idx] += tail_obs
            pooled_exp[smallest_idx] += tail_exp

    if len(pooled_exp) <= 1:
        return 0.0, 1, 1.0

    ndf = len(pooled_exp) - 1
    chi2_stat = sum(
        ((obs_item - exp_item) ** 2) / max(exp_item, 1.0)
        for obs_item, exp_item in zip(pooled_obs, pooled_exp)
    )
    p_value = compute_chi2_pvalue(float(chi2_stat), ndf)
    return float(chi2_stat), ndf, p_value


def _run_target_sampling_audit(
    physics_buffer: PhysicsBuffer,
    num_samples: int,
) -> tuple[list[TargetSamplingMetric], list[str], list[float], list[float], bool, float]:
    """Проводит Монте-Карло аудит _sample_Z_scalar для Water, NaI, Bone по 3 процессам и 7 энергиям."""
    rng = np.random.default_rng(np.random.SeedSequence(18))
    rng_ctx = RNGContext.from_numpy_rng(rng)

    target_materials = ("Water", "NaI", "Bone")
    # Включены низкая энергия (15 кэВ), околокраевые энергии вокруг K-края иода 33.17 кэВ
    # (33.0 кэВ и 33.4 кэВ), энергия выше K-края свинца (88.5 кэВ) и референсные линии ЯМ (30, 140.5, 511 кэВ)
    test_energies = (
        float(15.0 * units.keV),
        float(30.0 * units.keV),
        float(33.0 * units.keV),
        float(33.4 * units.keV),
        float(88.5 * units.keV),
        float(140.5 * units.keV),
        float(511.0 * units.keV),
    )

    raw_records: list[dict[str, object]] = []
    all_pvalues: list[float] = []
    plot_labels: list[str] = []
    plot_empirical: list[float] = []
    plot_theoretical: list[float] = []

    for mat_label in target_materials:
        mat_db_name = MATERIAL_CATALOG[mat_label]
        material = database_setting.material_database[mat_db_name]
        mat_id = Index(material.ID)

        for proc_col_idx in (PROCESS_PHOTOELECTRIC, PROCESS_COMPTON, PROCESS_COHERENT):
            proc_name = PROCESS_LABELS[proc_col_idx]
            proc_short = PROCESS_SHORT_LABELS[proc_col_idx]

            for photon_energy in test_energies:
                symbols, z_list, kernel_probs, bragg_probs = _compute_reference_element_probabilities(
                    material=material,
                    process_col_idx=proc_col_idx,
                    process_name=proc_name,
                    photon_energy=photon_energy,
                    physics_buffer=physics_buffer,
                )

                sampled_z = _sample_z_batch_kernel(
                    material_id=mat_id,
                    process_col_idx=Index(proc_col_idx),
                    photon_energy=Float(photon_energy),
                    num_samples=num_samples,
                    physics_buffer=physics_buffer,
                    rng_ctx=rng_ctx,
                )

                obs_counts: list[int] = []
                emp_probs: list[float] = []
                abs_diffs: list[float] = []
                scheme_diffs: list[float] = []
                z_scores: list[float] = []
                samples_float = float(num_samples)

                for elem_z, kernel_p, bragg_p in zip(z_list, kernel_probs, bragg_probs):
                    count_val = int(np.count_nonzero(sampled_z == elem_z))
                    obs_counts.append(count_val)
                    emp_p = float(count_val) / samples_float
                    emp_probs.append(emp_p)

                    diff_val = abs(emp_p - kernel_p)
                    abs_diffs.append(diff_val)
                    scheme_diffs.append(abs(kernel_p - bragg_p))

                    # Стандартная ошибка мультиномиальной доли с нижней границей 1 отсчет для редких событий (N*p < 5)
                    binom_var_counts = samples_float * kernel_p * (1.0 - kernel_p)
                    std_err = math.sqrt(max(binom_var_counts, 1.0)) / samples_float
                    z_scores.append(diff_val / std_err)

                chi2_stat, chi2_ndf, chi2_pval = _compute_multinomial_sampling_chi2(
                    observed_counts=obs_counts,
                    expected_probs=kernel_probs,
                    num_samples=num_samples,
                )
                all_pvalues.append(chi2_pval)

                dominant_idx = int(np.argmax(kernel_probs))
                dom_sym = symbols[dominant_idx]
                dom_theo = kernel_probs[dominant_idx]
                dom_emp = emp_probs[dominant_idx]
                max_scheme_diff = max(scheme_diffs)
                max_diff = max(abs_diffs)
                max_z = max(z_scores)

                # Масштабируемый от объема выборки N порог для максимального отклонения доли элемента (4.5 * SE(N))
                max_elem_se = max(
                    math.sqrt(max(samples_float * kp * (1.0 - kp), 1.0)) / samples_float
                    for kp in kernel_probs
                )
                allowed_prob_diff = max(4.5 * max_elem_se, 3.0 / samples_float)

                raw_records.append(
                    {
                        "material_label": mat_label,
                        "process_name": proc_name,
                        "process_col_idx": proc_col_idx,
                        "photon_energy": photon_energy,
                        "num_samples": num_samples,
                        "element_symbols": symbols,
                        "element_atomic_numbers": z_list,
                        "theoretical_probabilities": kernel_probs,
                        "bragg_exact_probabilities": bragg_probs,
                        "empirical_probabilities": emp_probs,
                        "dominant_element_symbol": dom_sym,
                        "dominant_theoretical_prob": dom_theo,
                        "dominant_empirical_prob": dom_emp,
                        "max_interpolation_scheme_diff": max_scheme_diff,
                        "max_abs_probability_diff": max_diff,
                        "allowed_probability_diff": allowed_prob_diff,
                        "max_abs_z_score": max_z,
                        "chi2_stat": chi2_stat,
                        "chi2_ndf": chi2_ndf,
                        "chi2_pvalue": chi2_pval,
                    }
                )

                if abs(photon_energy - float(140.5 * units.keV)) < 1e-6:
                    plot_labels.append(f"{mat_label}/{proc_short} ({dom_sym})")
                    plot_theoretical.append(dom_theo)
                    plot_empirical.append(dom_emp)

    holm_all_passed, min_pvalue, holm_thresholds = evaluate_holm_bonferroni(all_pvalues, alpha=0.001)

    sampling_metrics: list[TargetSamplingMetric] = []
    for rec, holm_thresh in zip(raw_records, holm_thresholds):
        max_z_val = float(rec["max_abs_z_score"])  # type: ignore[arg-type]
        max_diff_val = float(rec["max_abs_probability_diff"])  # type: ignore[arg-type]
        allowed_diff_val = float(rec["allowed_probability_diff"])  # type: ignore[arg-type]
        max_scheme_val = float(rec["max_interpolation_scheme_diff"])  # type: ignore[arg-type]
        pval = float(rec["chi2_pvalue"])  # type: ignore[arg-type]

        item_passed = (
            (pval >= holm_thresh)
            and (max_z_val < 4.2)
            and (max_diff_val <= allowed_diff_val)
            and (max_scheme_val < 1.5e-3)
        )
        sampling_metrics.append(
            TargetSamplingMetric(
                material_label=str(rec["material_label"]),
                process_name=str(rec["process_name"]),
                process_col_idx=int(rec["process_col_idx"]),  # type: ignore[arg-type]
                photon_energy=float(rec["photon_energy"]),  # type: ignore[arg-type]
                num_samples=int(rec["num_samples"]),  # type: ignore[arg-type]
                element_symbols=list(rec["element_symbols"]),  # type: ignore[arg-type]
                element_atomic_numbers=list(rec["element_atomic_numbers"]),  # type: ignore[arg-type]
                theoretical_probabilities=list(rec["theoretical_probabilities"]),  # type: ignore[arg-type]
                bragg_exact_probabilities=list(rec["bragg_exact_probabilities"]),  # type: ignore[arg-type]
                empirical_probabilities=list(rec["empirical_probabilities"]),  # type: ignore[arg-type]
                dominant_element_symbol=str(rec["dominant_element_symbol"]),
                dominant_theoretical_prob=float(rec["dominant_theoretical_prob"]),  # type: ignore[arg-type]
                dominant_empirical_prob=float(rec["dominant_empirical_prob"]),  # type: ignore[arg-type]
                max_interpolation_scheme_diff=max_scheme_val,
                max_abs_probability_diff=max_diff_val,
                allowed_probability_diff=allowed_diff_val,
                max_abs_z_score=max_z_val,
                chi2_stat=float(rec["chi2_stat"]),  # type: ignore[arg-type]
                chi2_ndf=int(rec["chi2_ndf"]),  # type: ignore[arg-type]
                chi2_pvalue=pval,
                holm_threshold=holm_thresh,
                passed=item_passed,
            )
        )

    return (
        sampling_metrics,
        plot_labels,
        plot_empirical,
        plot_theoretical,
        holm_all_passed,
        min_pvalue,
    )


def _compute_global_max_interpolation_scheme_diff(
    physics_buffer: PhysicsBuffer,
) -> float:
    """Вычисляет максимальное расхождение между линейной по ln(E) интерполяцией долей ElementCSR и log-log Брэггом."""
    material_bank = physics_buffer.material_bank
    max_scheme_diff = 0.0
    for mat_label in ("Water", "NaI", "Bone", "Pyrex"):
        mat_db_name = MATERIAL_CATALOG[mat_label]
        material = database_setting.material_database[mat_db_name]
        pointer = material_bank.mat_pointers[int(material.ID)]
        row_start = int(pointer["start_idx"])
        row_length = int(pointer["length"])
        mat_grid = material_bank.physics_energy_grid[row_start : row_start + row_length]

        for row_idx in range(row_length - 1):
            e_low = float(mat_grid[row_idx])
            e_high = float(mat_grid[row_idx + 1])
            if e_high <= e_low * (1.0 + 1.0e-12):
                continue
            mid_energy = math.sqrt(e_low * e_high)
            for proc_col_idx, proc_name in PROCESS_LABELS.items():
                _, _, kernel_probs, bragg_probs = _compute_reference_element_probabilities(
                    material=material,
                    process_col_idx=proc_col_idx,
                    process_name=proc_name,
                    photon_energy=mid_energy,
                    physics_buffer=physics_buffer,
                )
                for kp, bp in zip(kernel_probs, bragg_probs):
                    max_scheme_diff = max(max_scheme_diff, abs(kp - bp))

    return max_scheme_diff


def run_exp01_cross_sections_and_target_sampling(
    output_dir: Path,
    quick: bool = False,
) -> Exp01CrossSectionsReport:
    """Запускает Эксперимент 1 (микрофизика сечений, O(1) лог-индексация и выбор элемента-мишени Z).

    Args:
        output_dir: Директория для сохранения JSON-отчета и PNG-графика.
        quick: Режим экспресс-проверки с уменьшенным размером сетки и выборки Монте-Карло.

    Returns:
        Структурированный отчет Exp01CrossSectionsReport.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_binning_points = 10_000 if quick else 50_000
    num_mc_samples = 50_000 if quick else 200_000

    attenuation_db = database_setting.attenuation_database
    processes_list: list[Process] = [
        PhotoelectricEffect(attenuation_database=attenuation_db, process_col_idx=Index(PROCESS_PHOTOELECTRIC)),
        ComptonScattering(attenuation_database=attenuation_db, process_col_idx=Index(PROCESS_COMPTON)),
        CoherentScattering(attenuation_database=attenuation_db, process_col_idx=Index(PROCESS_COHERENT)),
    ]

    required_material_names = list(MATERIAL_CATALOG.values()) + ["I", "W", "Pb"]
    materials_list = [database_setting.material_database[name] for name in dict.fromkeys(required_material_names)]
    physics_buffer = _build_verification_physics_buffer(materials_list, processes_list)

    # 0. Сверка с внешними таблицами стандарта NIST XCOM (Тип A)
    nist_xcom_metrics = _run_nist_xcom_reference_audit(physics_buffer=physics_buffer)

    # 1. Паритет O(1) логарифмического биннинга
    binning_metrics, grid_energies, rel_diff_binning = _run_binning_parity_audit(
        physics_buffer=physics_buffer,
        processes_list=processes_list,
        num_grid_points=num_binning_points,
    )

    # 2. Право-непрерывность и скачки на K/L/M-краях поглощения против McMaster/Scofield/NIST
    edge_metrics = _run_edge_jump_audit(physics_buffer=physics_buffer)

    # 3. Правило аддитивности Брэгга (в узлах и между узлами) и нормировка ElementCSR
    bragg_metrics = _run_bragg_additivity_audit(physics_buffer=physics_buffer)

    # 4. Стохастический выбор элемента-мишени _sample_Z_scalar
    (
        sampling_metrics,
        target_labels,
        empirical_probs,
        theoretical_probs,
        sampling_holm_passed,
        min_sampling_pvalue,
    ) = _run_target_sampling_audit(
        physics_buffer=physics_buffer,
        num_samples=num_mc_samples,
    )
    global_max_scheme_diff = _compute_global_max_interpolation_scheme_diff(physics_buffer=physics_buffer)

    # Подготовка кривых ослабления свинца для Панели 1
    lead_material = database_setting.material_database["Pb"]
    lead_pointer = physics_buffer.material_bank.mat_pointers[lead_material.ID]
    lead_start = int(lead_pointer["start_idx"])
    lead_len = int(lead_pointer["length"])
    lead_energies = np.asarray(
        physics_buffer.material_bank.physics_energy_grid[lead_start : lead_start + lead_len],
        dtype=np.float64,
    )
    lead_lacs = np.asarray(
        physics_buffer.material_bank.physics_lac_table[lead_start : lead_start + lead_len, :],
        dtype=np.float64,
    )
    lead_pe = lead_lacs[:, PROCESS_PHOTOELECTRIC]
    lead_comp = lead_lacs[:, PROCESS_COMPTON]
    lead_coh = lead_lacs[:, PROCESS_COHERENT]
    lead_tot = lead_pe + lead_comp + lead_coh

    pb_edge_energies, pb_edge_labels = attenuation_db.get_element_edges("Pb")
    lead_edges_for_plot: list[tuple[str, float]] = []
    for edge_label_str in ("K", "L1", "L3", "M1"):
        match_pos = np.flatnonzero(pb_edge_labels == edge_label_str)
        if match_pos.size == 0:
            raise ValueError(f"Край '{edge_label_str}' отсутствует в таблице краев свинца Pb.")
        edge_e_val = float(pb_edge_energies[int(match_pos[0])])
        lead_edges_for_plot.append((f"{edge_label_str} ({edge_e_val * 1000.0:.1f} кэВ)", edge_e_val))

    plot_path = output_dir / "exp01_cross_sections.png"
    plot_exp01_cross_sections_and_target_sampling(
        energies=lead_energies,
        lead_pe=lead_pe,
        lead_comp=lead_comp,
        lead_coh=lead_coh,
        lead_tot=lead_tot,
        lead_edges=lead_edges_for_plot,
        grid_energies=grid_energies,
        rel_diff_binning=rel_diff_binning,
        target_labels=target_labels,
        empirical_probs=empirical_probs,
        theoretical_probs=theoretical_probs,
        output_path=plot_path,
    )

    max_nist_xcom_rel_error = max(item.rel_error for item in nist_xcom_metrics)
    max_binning_rel_error = max(item.max_rel_error_total for item in binning_metrics)
    total_row_mismatches = sum(item.row_mismatches for item in binning_metrics)
    max_edge_continuity_rel_error = max(item.right_continuity_rel_error for item in edge_metrics)
    max_edge_delta_vs_node_rel_error = max(item.delta_vs_node_jump_rel_error for item in edge_metrics)
    max_external_jump_ratio_rel_error = max(item.jump_ratio_rel_error for item in edge_metrics)
    max_bragg_rel_error = max(
        max(item.max_nodal_rel_error, item.max_rel_diff_partial_reconstruction, item.max_abs_fraction_sum_error)
        for item in bragg_metrics
    )
    max_bragg_internodal_rel_error = max(item.max_internodal_rel_error for item in bragg_metrics)
    max_interpolation_scheme_diff = max(
        global_max_scheme_diff,
        max(item.max_interpolation_scheme_diff for item in sampling_metrics),
    )
    max_sampling_z_score = max(item.max_abs_z_score for item in sampling_metrics)
    max_sampling_abs_diff = max(item.max_abs_probability_diff for item in sampling_metrics)

    lead_k_jump = next(
        item.delta_jump_ratio for item in edge_metrics if item.element_symbol == "Pb" and item.edge_label == "K"
    )
    iodine_k_jump = next(
        item.delta_jump_ratio for item in edge_metrics if item.element_symbol == "I" and item.edge_label == "K"
    )
    tungsten_k_jump = next(
        item.delta_jump_ratio for item in edge_metrics if item.element_symbol == "W" and item.edge_label == "K"
    )

    overall_passed = (
        all(item.passed for item in nist_xcom_metrics)
        and all(item.passed for item in binning_metrics)
        and all(item.passed for item in edge_metrics)
        and all(item.passed for item in bragg_metrics)
        and all(item.passed for item in sampling_metrics)
        and sampling_holm_passed
        and (max_interpolation_scheme_diff < 1.5e-3)
    )

    reference_standard_types: dict[str, str] = {
        "nist_xcom_mac_anchor": ReferenceStandardType.EXTERNAL_STANDARD.value,
        "edge_jump_mcmaster_anchor": ReferenceStandardType.EXTERNAL_STANDARD.value,
        "o1_log_binning_vs_searchsorted": ReferenceStandardType.SELF_CONSISTENCY.value,
        "edge_right_continuity_and_relaxation_db": ReferenceStandardType.SELF_CONSISTENCY.value,
        "bragg_additivity_nodal_and_internodal": ReferenceStandardType.KERNEL_ANALYTIC.value,
        "target_z_sampling_multinomial": ReferenceStandardType.KERNEL_ANALYTIC.value,
    }

    json_path = output_dir / "exp01_cross_sections.json"
    report = Exp01CrossSectionsReport(
        passed=overall_passed,
        quick=quick,
        num_binning_points=num_binning_points,
        num_mc_samples=num_mc_samples,
        max_nist_xcom_rel_error=max_nist_xcom_rel_error,
        max_binning_rel_error=max_binning_rel_error,
        total_binning_row_mismatches=total_row_mismatches,
        max_edge_continuity_rel_error=max_edge_continuity_rel_error,
        max_edge_delta_vs_node_rel_error=max_edge_delta_vs_node_rel_error,
        max_external_jump_ratio_rel_error=max_external_jump_ratio_rel_error,
        lead_k_jump_ratio=lead_k_jump,
        iodine_k_jump_ratio=iodine_k_jump,
        tungsten_k_jump_ratio=tungsten_k_jump,
        max_bragg_rel_error=max_bragg_rel_error,
        max_bragg_internodal_rel_error=max_bragg_internodal_rel_error,
        max_interpolation_scheme_diff=max_interpolation_scheme_diff,
        max_sampling_z_score=max_sampling_z_score,
        max_sampling_abs_diff=max_sampling_abs_diff,
        sampling_holm_bonferroni_passed=sampling_holm_passed,
        min_sampling_chi2_pvalue=min_sampling_pvalue,
        reference_standard_types=reference_standard_types,
        model_limitations=list(DOCUMENTED_MODEL_LIMITATIONS),
        nist_xcom_metrics=nist_xcom_metrics,
        binning_metrics=binning_metrics,
        edge_metrics=edge_metrics,
        bragg_metrics=bragg_metrics,
        target_sampling_metrics=sampling_metrics,
        json_path=str(json_path),
        plot_path=str(plot_path),
    )

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "NistXcomReferenceMetric",
    "BinningParityMetric",
    "EdgeJumpMetric",
    "BraggAdditivityMetric",
    "TargetSamplingMetric",
    "Exp01CrossSectionsReport",
    "run_exp01_cross_sections_and_target_sampling",
]
