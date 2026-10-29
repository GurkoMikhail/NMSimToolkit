"""Эксперимент 2: Верификация индикатрис некогерентного (Комптон с S(x, Z)) и когерентного (Рэлей с F(x, Z)) рассеяния и 3D-кинематики.

Проверяет пять микрофизических и кинематических инвариантов:
1. Граничные свойства и табличную точность функции некогерентного рассеяния _compute_scattering_function_scalar
   (S(0, Z) = 0, S(x -> inf, Z) = Z) против первичного стандарта Hubbell et al. (1975) / Erratum (1977)
   и точной квантовомеханической волновой функции 1s водорода, а также соответствие выборки
   _generate_compton_theta_scalar дифференциальному сечению Клейна–Нишины * S(x, Z)
   при E in {15, 30, 140.5, 511} кэВ для C (Z=6), I (Z=53) и Pb (Z=82).
2. Граничные свойства и табличную точность атомного форм-фактора когерентного рассеяния F(x, Z)
   (F(0, Z) = Z, F(x -> inf, Z) -> 0) против релятивистских таблиц Хартри–Фока Hubbell & Overbo (1979) /
   International Tables for Crystallography Vol. C (Cromer & Mann 1968, RHF) для H (Z=1), C (Z=6),
   O (Z=8), I (Z=53) и Pb (Z=82).
3. Тождественное совпадение энергии отдачи электрона _calculate_compton_energy_deposit_scalar
   с аналитической релятивистской кинематикой Комптона и положением комптоновского края (cos(theta) = -1).
4. Соответствие выборки _generate_coherent_theta_scalar дифференциальному сечению Томсона * F^2(x, Z)
   при E in {15, 30, 80, 140.5} кэВ для O (Water, Z=8), I (Z=53) и Pb (Z=82) на адаптивной степенной
   сетке бинов по переменной (1 - cos(theta)), разрешающей узкий пик рассеяния вперед, с контролем
   множественных проверок процедурой Холма–Бонферрони и z-критерием для <cos(theta)>.
5. Равномерность азимутального угла _generate_scattering_phi_scalar на [-pi, pi), сохранение нормы
   ||d_1|| = 1, полярного скалярного произведения d_0 . d_1 = cos(theta), точное восстановление
   азимутального угла phi в локальном ортонормированном базисе вокруг d_0 (включая полюса d_{0z} = +-1)
   с точностью < 1e-12 рад, а также стремление к изотропии (||<d_k>|| -> 0) после цепочки многократных рассеяний.
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

from core.other.typing_definitions import Charge, Float
from core.particles.kinematic_state import KinematicState
from core.particles.particles_kernels import rotate_kernel
from core.physics.g4coherent import _generate_coherent_theta_scalar
from core.physics.g4compton import (
    _calculate_compton_energy_deposit_scalar,
    _compute_scattering_function_scalar,
    _generate_compton_theta_scalar,
    electron_mass_c2,
)
from core.physics.interaction_buffers import RNGContext
from core.physics.processes_kernels import _generate_scattering_phi_scalar
from verification.common.scattering_and_stats import (
    DEFAULT_F_FACTOR_TABLE,
    ReferenceStandardType,
    compute_chi2_pvalue,
    compute_klein_nishina_bound_pdf,
    compute_pooled_chi2_statistics,
    compute_rayleigh_thomson_pdf,
    evaluate_form_factor_f,
    evaluate_holm_bonferroni,
    evaluate_hubbell_s_function,
)
from verification.physics.plotting import plot_exp02_scattering_indicatrices_and_kinematics


# Внешние эталонные значения функции некогерентного рассеяния S(x, Z) (Тип A).
# Первичный библиографический источник:
#   1) Hubbell J.H., Veigele Wm.J., Briggs E.A., Brown R.T., Cromer D.T., Howerton R.J.,
#      "Atomic Form Factors, Incoherent Scattering Functions, and Photon Scattering Cross Sections",
#      J. Phys. Chem. Ref. Data, Vol. 4, No. 3, pp. 471–538 (1975), DOI: 10.1063/1.555523, Table I & Eq. (18).
#   2) Hubbell J.H. et al., Erratum, J. Phys. Chem. Ref. Data, Vol. 6, No. 2, pp. 615–616 (1977)
#      (включая исправленную точку S(x = 0.04 A^-1 = 4.0e6 см^-1, Z=6) = 0.2015).
#   3) Аналитическая кусочно-полиномиальная параметризация Geant4 G4ComptonScattering (Cullen et al., EPDL97).
# Переданный импульс задан в конвенции Geant4: x = sin(theta/2) / lambda [1/см] = 10^8 * x_[1/Angstrom].
HUBBELL_1975_REFERENCE_TABLE: dict[int, tuple[tuple[float, float], ...]] = {
    1: (
        (1.0e6, 0.00441),
        (5.0e6, 0.10332),
        (1.0e7, 0.34257),
        (2.0e7, 0.76885),
        (5.0e7, 0.99502),
        (1.0e8, 0.99995),
    ),
    6: (
        (1.0e6, 0.0130),
        (4.0e6, 0.2015),
        (5.0e6, 0.3081),
        (1.0e7, 1.035),
        (2.0e7, 2.614),
        (5.0e7, 4.488),
        (1.0e8, 5.393),
    ),
    8: (
        (1.0e6, 0.0110),
        (5.0e6, 0.2713),
        (1.0e7, 0.9753),
        (2.0e7, 2.806),
        (5.0e7, 5.764),
        (1.0e8, 7.041),
    ),
    53: (
        (1.0e6, 0.0540),
        (5.0e6, 1.245),
        (1.0e7, 3.860),
        (2.0e7, 9.233),
        (5.0e7, 20.15),
        (1.0e8, 30.88),
    ),
    82: (
        (1.0e6, 0.0720),
        (5.0e6, 1.567),
        (1.0e7, 4.595),
        (2.0e7, 10.61),
        (5.0e7, 23.94),
        (1.0e8, 40.38),
    ),
}

# Внешние независимые эталонные значения релятивистского атомного форм-фактора F(x, Z) (Тип A).
# Первичные библиографические источники:
#   1) Hubbell J.H., Overbo I., "Relativistic Atomic Form Factors and Photon Coherent Scattering Cross Sections",
#      J. Phys. Chem. Ref. Data, Vol. 8, No. 1, pp. 69–105 (1979), DOI: 10.1063/1.555593, Table I.
#   2) Cromer D.T., Mann J.B., "X-ray scattering factors computed from numerical Hartree-Fock wave functions",
#      Acta Crystallogr. A24, pp. 321–324 (1968); International Tables for Crystallography, Vol. C, Table 6.1.1.4.
#   3) Для водорода (Z=1): точная квантовомеханическая формула 1s-орбитали F(x, 1) = (1 + (2*pi*a_0*x)^2)^(-2).
# Переданный импульс задан в конвенции Geant4: x = sin(theta/2) / lambda [1/см] = 10^8 * x_[1/Angstrom].
FORM_FACTOR_EXTERNAL_REFERENCE_TABLE: dict[int, tuple[tuple[float, float], ...]] = {
    1: (
        (0.0, 1.0000),
        (5.0e6, 0.9469),
        (1.0e7, 0.8108),
        (2.0e7, 0.4808),
        (5.0e7, 0.0706),
        (1.0e8, 0.0069),
    ),
    6: (
        (0.0, 6.0000),
        (5.0e6, 5.7492),
        (1.0e7, 5.1076),
        (2.0e7, 3.5596),
        (5.0e7, 1.6858),
        (1.0e8, 1.1152),
    ),
    8: (
        (0.0, 8.0000),
        (5.0e6, 7.7978),
        (1.0e7, 7.2455),
        (2.0e7, 5.6226),
        (5.0e7, 2.3372),
        (1.0e8, 1.3769),
    ),
    53: (
        (0.0, 53.0000),
        (5.0e6, 51.9108),
        (1.0e7, 49.1409),
        (2.0e7, 42.3415),
        (5.0e7, 28.1417),
        (1.0e8, 17.3037),
    ),
    82: (
        (0.0, 82.0000),
        (5.0e6, 80.7602),
        (1.0e7, 77.6598),
        (2.0e7, 69.4819),
        (5.0e7, 48.9352),
        (1.0e8, 30.2283),
    ),
}


@dataclass(frozen=True)
class ScatteringFunctionBoundaryMetric:
    """Метрики граничных свойств и табличной точности некогерентной функции рассеяния S(x, Z)."""

    element_symbol: str
    atomic_z: int
    value_at_zero: float
    value_at_infinity: float
    rel_error_at_infinity: float
    max_poly_parity_rel_error: float
    max_hubbell_table_rel_error: float
    is_monotone_increasing: bool
    passed: bool


@dataclass(frozen=True)
class FormFactorBoundaryMetric:
    """Метрики граничных свойств и внешней табличной точности атомного форм-фактора F(x, Z) (Тип A)."""

    element_symbol: str
    atomic_z: int
    value_at_zero: float
    rel_error_at_zero: float
    value_at_high_momentum: float
    max_external_table_rel_error: float
    is_monotone_decreasing: bool
    passed: bool


@dataclass(frozen=True)
class ComptonIndicatrixMetric:
    """Метрики верификации угловой индикатрисы комптоновского рассеяния _generate_compton_theta_scalar."""

    element_symbol: str
    atomic_z: int
    photon_energy: float
    num_samples: int
    mean_cos_theta_mc: float
    mean_cos_theta_theory: float
    mean_cos_z_score: float
    reduced_chi2: float
    chi2_ndf: int
    chi2_pvalue: float
    holm_threshold: float
    density_l1_error: float
    passed: bool


@dataclass(frozen=True)
class ComptonKinematicsMetric:
    """Метрики точности расчета комптоновской отдачи _calculate_compton_energy_deposit_scalar."""

    photon_energy: float
    compton_edge_mc: float
    compton_edge_theory: float
    edge_rel_error: float
    max_grid_rel_error: float
    passed: bool


@dataclass(frozen=True)
class CoherentIndicatrixMetric:
    """Метрики верификации угловой индикатрисы когерентного (рэлеевского) рассеяния _generate_coherent_theta_scalar."""

    element_symbol: str
    atomic_z: int
    photon_energy: float
    num_samples: int
    binning_scheme: str
    mean_cos_theta_mc: float
    mean_cos_theta_theory: float
    mean_cos_z_score: float
    reduced_chi2: float
    chi2_ndf: int
    chi2_pvalue: float
    holm_threshold: float
    density_l1_error: float
    passed: bool


@dataclass(frozen=True)
class RotationAndAzimuthMetric:
    """Метрики верификации равномерности угла phi, восстановления азимута и 3D-поворота _rotate_particle."""

    num_phi_samples: int
    phi_in_native_range: bool
    phi_reduced_chi2: float
    phi_chi2_pvalue: float
    num_rotation_vectors: int
    num_singular_vectors: int
    max_norm_error: float
    max_dot_product_error: float
    max_azimuth_reconstruction_error: float
    multi_scatter_chain_steps: int
    multi_scatter_mean_direction_norm: float
    multi_scatter_isotropy_threshold: float
    passed: bool


@dataclass(frozen=True)
class Exp02ScatteringIndicatricesReport:
    """Корневой отчет верификации индикатрис рассеяния и 3D-кинематики (Эксперимент 2)."""

    passed: bool
    quick: bool
    num_mc_samples: int
    max_hubbell_table_rel_error: float
    max_form_factor_table_rel_error: float
    max_compton_chi2: float
    min_compton_chi2_pvalue: float
    compton_holm_bonferroni_passed: bool
    max_compton_l1_error: float
    max_compton_mean_cos_z_score: float
    max_kinematics_rel_error: float
    max_coherent_chi2: float
    min_coherent_chi2_pvalue: float
    coherent_holm_bonferroni_passed: bool
    max_coherent_mean_cos_z_score: float
    lead_coherent_forward_mean_cos: float
    phi_in_native_range: bool
    phi_reduced_chi2: float
    phi_chi2_pvalue: float
    max_rotation_norm_error: float
    max_rotation_dot_error: float
    max_rotation_azimuth_error: float
    multi_scatter_mean_direction_norm: float
    reference_standard_types: dict[str, str]
    boundary_metrics: list[ScatteringFunctionBoundaryMetric]
    form_factor_metrics: list[FormFactorBoundaryMetric]
    compton_metrics: list[ComptonIndicatrixMetric]
    kinematics_metrics: list[ComptonKinematicsMetric]
    coherent_metrics: list[CoherentIndicatrixMetric]
    rotation_metric: RotationAndAzimuthMetric
    json_path: str
    plot_path: str


@njit(cache=True)
def _sample_compton_cos_theta_batch(
    photon_energy: Float,
    atomic_z: Charge,
    num_samples: int,
    rng_ctx: RNGContext,
) -> NDArray[Float]:
    """Генерирует выборку cos(theta) для некогерентного рассеяния ядром _generate_compton_theta_scalar."""
    cos_samples = np.empty(num_samples, dtype=Float)
    for sample_idx in range(num_samples):
        theta_val = _generate_compton_theta_scalar(photon_energy, atomic_z, rng_ctx)
        cos_samples[sample_idx] = Float(math.cos(theta_val))
    return cos_samples


@njit(cache=True)
def _sample_coherent_cos_theta_batch(
    photon_energy: Float,
    atomic_z: Charge,
    num_samples: int,
    rng_ctx: RNGContext,
) -> NDArray[Float]:
    """Генерирует выборку cos(theta) для когерентного рассеяния ядром _generate_coherent_theta_scalar."""
    cos_samples = np.empty(num_samples, dtype=Float)
    for sample_idx in range(num_samples):
        theta_val = _generate_coherent_theta_scalar(photon_energy, atomic_z, rng_ctx)
        cos_samples[sample_idx] = Float(math.cos(theta_val))
    return cos_samples


@njit(cache=True)
def _evaluate_compton_edep_batch(
    polar_angles: NDArray[Float],
    photon_energy: Float,
) -> NDArray[Float]:
    """Вычисляет переданную электрону энергию отдачи через ядро _calculate_compton_energy_deposit_scalar."""
    num_pts = polar_angles.shape[0]
    edep_arr = np.empty(num_pts, dtype=Float)
    for idx in range(num_pts):
        edep_arr[idx] = _calculate_compton_energy_deposit_scalar(polar_angles[idx], photon_energy)
    return edep_arr


@njit(cache=True)
def _sample_phi_batch(
    num_samples: int,
    rng_ctx: RNGContext,
) -> NDArray[Float]:
    """Генерирует массив азимутальных углов рассеяния ядром _generate_scattering_phi_scalar."""
    phi_arr = np.empty(num_samples, dtype=Float)
    for idx in range(num_samples):
        phi_arr[idx] = _generate_scattering_phi_scalar(rng_ctx)
    return phi_arr


def _build_power_law_coherent_bin_edges(
    num_bins: int,
    power_exponent: float = 3.0,
) -> NDArray[np.float64]:
    """Строит адаптивную степенную сетку бинов по переменной (1 - cos(theta)) на [-1, 1].

    При power_exponent > 1 сетка сгущается вблизи cos(theta) = 1, обеспечивая высокое
    разрешение узкого пика когерентного рассеяния вперед.
    """
    if num_bins < 4 or power_exponent <= 0.0:
        raise ValueError("Некорректные параметры степенной сетки бинов когерентного рассеяния.")
    unit_grid = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    reversed_u = 1.0 - unit_grid
    edges = 1.0 - 2.0 * np.power(reversed_u, power_exponent)
    edges[0] = -1.0
    edges[-1] = 1.0
    return edges


def _compute_bin_probabilities_gauss_legendre(
    bin_edges: NDArray[np.float64],
    photon_energy: float,
    atomic_z: int,
    is_compton: bool,
    quad_order: int = 24,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Интегрирует теоретическую индикатрису на каждом бине [-1, 1] квадратурой Гаусса–Лежандра.

    Использует общие аналитические функции compute_klein_nishina_bound_pdf и
    compute_rayleigh_thomson_pdf из verification.common.scattering_and_stats.
    """
    nodes_ref, weights_ref = np.polynomial.legendre.leggauss(quad_order)
    num_bins = bin_edges.shape[0] - 1
    half_widths = 0.5 * (bin_edges[1:] - bin_edges[:-1])
    midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    quad_points_2d = midpoints[:, np.newaxis] + half_widths[:, np.newaxis] * nodes_ref[np.newaxis, :]
    flat_points = np.ascontiguousarray(quad_points_2d.ravel(), dtype=np.float64)

    if is_compton:
        flat_vals = compute_klein_nishina_bound_pdf(flat_points, photon_energy, atomic_z)
    else:
        flat_vals = compute_rayleigh_thomson_pdf(
            flat_points, photon_energy, atomic_z, DEFAULT_F_FACTOR_TABLE
        )

    vals_2d = np.asarray(flat_vals, dtype=np.float64).reshape((num_bins, quad_order))
    bin_integrals = half_widths * np.sum(vals_2d * weights_ref[np.newaxis, :], axis=1)
    total_integral = float(np.sum(bin_integrals))
    bin_probs = bin_integrals / total_integral
    bin_densities = bin_probs / (bin_edges[1:] - bin_edges[:-1])

    weighted_cos_2d = quad_points_2d * vals_2d
    cos_integrals = half_widths * np.sum(weighted_cos_2d * weights_ref[np.newaxis, :], axis=1)
    theoretical_mean_cos = float(np.sum(cos_integrals)) / total_integral

    weighted_cos_sq_2d = (quad_points_2d * quad_points_2d) * vals_2d
    cos_sq_integrals = half_widths * np.sum(weighted_cos_sq_2d * weights_ref[np.newaxis, :], axis=1)
    theoretical_mean_cos_sq = float(np.sum(cos_sq_integrals)) / total_integral
    theoretical_var_cos = max(0.0, theoretical_mean_cos_sq - theoretical_mean_cos * theoretical_mean_cos)
    theoretical_std_cos = math.sqrt(theoretical_var_cos)

    return bin_probs, bin_densities, theoretical_mean_cos, theoretical_std_cos


def _run_scattering_function_boundary_audit() -> list[ScatteringFunctionBoundaryMetric]:
    """Проверяет граничные условия S(0, Z) = 0, S(x -> inf, Z) = Z, монотонность и точки Hubbell (1975)."""
    test_elements: list[tuple[str, int]] = [("H", 1), ("C", 6), ("O", 8), ("I", 53), ("Pb", 82)]
    x_grid = np.geomspace(1.0e-4, 2.0e8, 250, dtype=np.float64)

    metrics: list[ScatteringFunctionBoundaryMetric] = []
    for sym_str, z_val in test_elements:
        charge_z = Charge(z_val)
        val_zero = float(_compute_scattering_function_scalar(Float(0.0), charge_z))
        val_inf = float(_compute_scattering_function_scalar(Float(1.0e12), charge_z))
        rel_err_inf = abs(val_inf - float(z_val)) / float(z_val)

        grid_vals = np.array(
            [float(_compute_scattering_function_scalar(Float(x_pt), charge_z)) for x_pt in x_grid],
            dtype=np.float64,
        )
        poly_ref_vals = evaluate_hubbell_s_function(x_grid, z_val)
        poly_rel_diffs = np.abs(grid_vals - poly_ref_vals) / np.maximum(poly_ref_vals, 1e-30)
        max_poly_rel_err = float(np.max(poly_rel_diffs))

        hubbell_points = HUBBELL_1975_REFERENCE_TABLE[z_val]
        hubbell_rel_errors: list[float] = []
        for momentum_transfer_val, s_hubbell in hubbell_points:
            s_kernel = float(_compute_scattering_function_scalar(Float(momentum_transfer_val), charge_z))
            hubbell_rel_errors.append(abs(s_kernel - s_hubbell) / s_hubbell)
        max_hubbell_rel_err = max(hubbell_rel_errors)

        diffs = np.diff(grid_vals)
        # Для Z=1 полином 4-й степени вблизи насыщения S -> 1 имеет осцилляцию аппроксимации ~5e-3
        monotone_tol = -6.0e-3 if z_val == 1 else -1.0e-12
        upper_bound_tol = 6.0e-3 if z_val == 1 else 1.0e-12
        is_monotone = bool(
            np.all(diffs >= monotone_tol)
            and np.all(grid_vals >= 0.0)
            and np.all(grid_vals <= float(z_val) + upper_bound_tol)
        )
        # Допуск 3.5% отражает погрешность кусочно-полиномиальной аппроксимации Geant4 относительно таблиц Hubbell-1975
        passed = (
            (abs(val_zero) < 1e-15)
            and (rel_err_inf < 1e-14)
            and (max_poly_rel_err < 1e-11)
            and (max_hubbell_rel_err < 3.5e-2)
            and is_monotone
        )

        metrics.append(
            ScatteringFunctionBoundaryMetric(
                element_symbol=sym_str,
                atomic_z=z_val,
                value_at_zero=val_zero,
                value_at_infinity=val_inf,
                rel_error_at_infinity=rel_err_inf,
                max_poly_parity_rel_error=max_poly_rel_err,
                max_hubbell_table_rel_error=max_hubbell_rel_err,
                is_monotone_increasing=is_monotone,
                passed=passed,
            )
        )
    return metrics


def _run_form_factor_boundary_audit() -> list[FormFactorBoundaryMetric]:
    """Проверяет атомный форм-фактор F(x, Z) против внешних релятивистских таблиц Хартри–Фока (Тип A)."""
    test_elements: list[tuple[str, int]] = [("H", 1), ("C", 6), ("O", 8), ("I", 53), ("Pb", 82)]
    x_grid = np.concatenate(
        [np.array([0.0], dtype=np.float64), np.geomspace(1.0e5, 1.0e9, 200, dtype=np.float64)]
    )

    metrics: list[FormFactorBoundaryMetric] = []
    for sym_str, z_val in test_elements:
        f_grid = evaluate_form_factor_f(x_grid, z_val, DEFAULT_F_FACTOR_TABLE)
        val_zero = float(f_grid[0])
        rel_err_zero = abs(val_zero - float(z_val)) / float(z_val)
        val_high = float(evaluate_form_factor_f(np.array([1.0e11], dtype=np.float64), z_val)[0])

        diffs = np.diff(f_grid)
        is_monotone_decreasing = bool(np.all(diffs <= 1.0e-12) and np.all(f_grid >= 0.0))

        ref_points = FORM_FACTOR_EXTERNAL_REFERENCE_TABLE[z_val]
        query_x = np.array([pt[0] for pt in ref_points], dtype=np.float64)
        ref_f = np.array([pt[1] for pt in ref_points], dtype=np.float64)
        kernel_f = evaluate_form_factor_f(query_x, z_val, DEFAULT_F_FACTOR_TABLE)
        rel_errors = np.abs(kernel_f - ref_f) / np.maximum(ref_f, 1.0e-30)
        max_table_rel_err = float(np.max(rel_errors))

        # Допуск 3.0% между 3-членной рациональной аппроксимацией Geant4 и 4-гауссовой RHF таблицей Cromer-Mann/Hubbell-1979
        passed = (
            (rel_err_zero < 1.0e-5)
            and (val_high / float(z_val) < 1.0e-4)
            and is_monotone_decreasing
            and (max_table_rel_err < 3.0e-2)
        )
        metrics.append(
            FormFactorBoundaryMetric(
                element_symbol=sym_str,
                atomic_z=z_val,
                value_at_zero=val_zero,
                rel_error_at_zero=rel_err_zero,
                value_at_high_momentum=val_high,
                max_external_table_rel_error=max_table_rel_err,
                is_monotone_decreasing=is_monotone_decreasing,
                passed=passed,
            )
        )
    return metrics


def _run_compton_indicatrix_audit(
    num_samples: int,
    bin_edges: NDArray[np.float64],
    rng_ctx: RNGContext,
) -> tuple[
    list[ComptonIndicatrixMetric],
    list[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    bool,
    float,
]:
    """Верифицирует индикатрису некогерентного рассеяния _generate_compton_theta_scalar."""
    test_elements: list[tuple[str, int]] = [("C", 6), ("I", 53), ("Pb", 82)]
    test_energies = (
        float(15.0 * units.keV),
        float(30.0 * units.keV),
        float(140.5 * units.keV),
        float(511.0 * units.keV),
    )
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    num_bins = int(bin_widths.shape[0])
    sqrt_n = math.sqrt(float(num_samples))
    allowed_l1 = 6.0 * math.sqrt(float(num_bins) / float(num_samples)) + 0.005

    raw_records: list[dict[str, object]] = []
    all_pvalues: list[float] = []
    plot_curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]] = []

    for sym_str, z_val in test_elements:
        for photon_energy in test_energies:
            cos_samples = _sample_compton_cos_theta_batch(
                photon_energy=Float(photon_energy),
                atomic_z=Charge(z_val),
                num_samples=num_samples,
                rng_ctx=rng_ctx,
            )
            counts, _ = np.histogram(cos_samples, bins=bin_edges)
            obs_counts = counts.astype(np.float64)
            emp_probs = obs_counts / float(num_samples)
            emp_density = emp_probs / bin_widths

            theo_probs, theo_density, theo_mean_cos, theo_std_cos = _compute_bin_probabilities_gauss_legendre(
                bin_edges=bin_edges,
                photon_energy=photon_energy,
                atomic_z=z_val,
                is_compton=True,
            )
            exp_counts = theo_probs * float(num_samples)
            _, chi2_ndf, red_chi2, chi2_pval = compute_pooled_chi2_statistics(obs_counts, exp_counts)
            all_pvalues.append(chi2_pval)

            l1_error = 0.5 * float(np.sum(np.abs(emp_probs - theo_probs)))
            emp_mean_cos = float(np.mean(cos_samples))
            mean_cos_se = theo_std_cos / sqrt_n
            mean_cos_z = abs(emp_mean_cos - theo_mean_cos) / max(mean_cos_se, 1e-30)

            raw_records.append(
                {
                    "element_symbol": sym_str,
                    "atomic_z": z_val,
                    "photon_energy": photon_energy,
                    "num_samples": num_samples,
                    "mean_cos_theta_mc": emp_mean_cos,
                    "mean_cos_theta_theory": theo_mean_cos,
                    "mean_cos_z_score": mean_cos_z,
                    "reduced_chi2": red_chi2,
                    "chi2_ndf": chi2_ndf,
                    "chi2_pvalue": chi2_pval,
                    "density_l1_error": l1_error,
                }
            )

            is_plot_candidate = (
                (sym_str == "C" and abs(photon_energy - float(30.0 * units.keV)) < 1e-6)
                or (
                    sym_str == "Pb"
                    and (
                        abs(photon_energy - float(30.0 * units.keV)) < 1e-6
                        or abs(photon_energy - float(140.5 * units.keV)) < 1e-6
                        or abs(photon_energy - float(511.0 * units.keV)) < 1e-6
                    )
                )
            )
            if is_plot_candidate:
                energy_display = photon_energy * 1000.0
                label_text = f"{sym_str} (Z={z_val}), {energy_display:.1f} кэВ"
                plot_curves.append((label_text, emp_density, theo_density))

    holm_passed, min_pval, holm_thresholds = evaluate_holm_bonferroni(all_pvalues, alpha=0.001)

    metrics: list[ComptonIndicatrixMetric] = []
    for rec, holm_thresh in zip(raw_records, holm_thresholds):
        pval = float(rec["chi2_pvalue"])  # type: ignore[arg-type]
        l1_val = float(rec["density_l1_error"])  # type: ignore[arg-type]
        z_val_score = float(rec["mean_cos_z_score"])  # type: ignore[arg-type]
        item_passed = (pval >= holm_thresh) and (l1_val < allowed_l1) and (z_val_score < 3.8)
        metrics.append(
            ComptonIndicatrixMetric(
                element_symbol=str(rec["element_symbol"]),
                atomic_z=int(rec["atomic_z"]),  # type: ignore[arg-type]
                photon_energy=float(rec["photon_energy"]),  # type: ignore[arg-type]
                num_samples=int(rec["num_samples"]),  # type: ignore[arg-type]
                mean_cos_theta_mc=float(rec["mean_cos_theta_mc"]),  # type: ignore[arg-type]
                mean_cos_theta_theory=float(rec["mean_cos_theta_theory"]),  # type: ignore[arg-type]
                mean_cos_z_score=z_val_score,
                reduced_chi2=float(rec["reduced_chi2"]),  # type: ignore[arg-type]
                chi2_ndf=int(rec["chi2_ndf"]),  # type: ignore[arg-type]
                chi2_pvalue=pval,
                holm_threshold=holm_thresh,
                density_l1_error=l1_val,
                passed=item_passed,
            )
        )

    return metrics, plot_curves, holm_passed, min_pval


def _run_compton_kinematics_audit() -> tuple[
    list[ComptonKinematicsMetric],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Проверяет точность _calculate_compton_energy_deposit_scalar и энергию комптоновского края."""
    polar_angles = np.linspace(0.0, math.pi, 201, dtype=Float)
    cos_grid = np.cos(polar_angles).astype(np.float64)
    one_minus_cos = 1.0 - cos_grid

    test_energies = (
        float(15.0 * units.keV),
        float(30.0 * units.keV),
        float(140.5 * units.keV),
        float(364.5 * units.keV),
        float(511.0 * units.keV),
    )

    metrics: list[ComptonKinematicsMetric] = []
    plot_edep_mc = np.zeros_like(cos_grid)
    plot_edep_theory = np.zeros_like(cos_grid)
    plot_edge_energy = 0.0

    for photon_energy in test_energies:
        edep_mc = np.asarray(
            _evaluate_compton_edep_batch(polar_angles, Float(photon_energy)),
            dtype=np.float64,
        )
        alpha_ratio = photon_energy / float(electron_mass_c2)
        k_term = alpha_ratio * one_minus_cos
        edep_theory = photon_energy * k_term / (1.0 + k_term)

        nonzero_mask = edep_theory > 0.0
        rel_errors = np.abs(edep_mc[nonzero_mask] - edep_theory[nonzero_mask]) / edep_theory[nonzero_mask]
        max_rel_err = float(np.max(rel_errors))

        edge_mc = float(_calculate_compton_energy_deposit_scalar(Float(math.pi), Float(photon_energy)))
        edge_formula = photon_energy * (2.0 * alpha_ratio) / (1.0 + 2.0 * alpha_ratio)
        edge_rel_err = abs(edge_mc - edge_formula) / edge_formula

        passed = (max_rel_err < 5.0e-15) and (edge_rel_err < 5.0e-15) and (abs(float(edep_mc[0])) < 1e-16)
        metrics.append(
            ComptonKinematicsMetric(
                photon_energy=photon_energy,
                compton_edge_mc=edge_mc,
                compton_edge_theory=edge_formula,
                edge_rel_error=edge_rel_err,
                max_grid_rel_error=max_rel_err,
                passed=passed,
            )
        )

        if abs(photon_energy - float(140.5 * units.keV)) < 1e-6:
            sort_idx = np.argsort(cos_grid)
            plot_edep_mc = edep_mc[sort_idx]
            plot_edep_theory = edep_theory[sort_idx]
            plot_edge_energy = edge_formula

    sorted_cos = np.sort(cos_grid)
    return metrics, sorted_cos, plot_edep_mc, plot_edep_theory, plot_edge_energy


def _run_coherent_indicatrix_audit(
    num_samples: int,
    plot_bin_edges: NDArray[np.float64],
    rng_ctx: RNGContext,
) -> tuple[
    list[CoherentIndicatrixMetric],
    list[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    bool,
    float,
]:
    """Верифицирует индикатрису когерентного рассеяния на адаптивных степенных бинах по (1 - cos(theta))."""
    test_elements: list[tuple[str, int]] = [("O (Water)", 8), ("I", 53), ("Pb", 82)]
    test_energies = (
        float(15.0 * units.keV),
        float(30.0 * units.keV),
        float(80.0 * units.keV),
        float(140.5 * units.keV),
    )
    num_adaptive_bins = 40
    adaptive_edges = _build_power_law_coherent_bin_edges(num_adaptive_bins, power_exponent=3.0)
    plot_bin_widths = plot_bin_edges[1:] - plot_bin_edges[:-1]
    sqrt_n = math.sqrt(float(num_samples))
    allowed_l1 = 6.0 * math.sqrt(float(num_adaptive_bins) / float(num_samples)) + 0.005

    raw_records: list[dict[str, object]] = []
    all_pvalues: list[float] = []
    plot_curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]] = []

    for sym_str, z_val in test_elements:
        for photon_energy in test_energies:
            cos_samples = _sample_coherent_cos_theta_batch(
                photon_energy=Float(photon_energy),
                atomic_z=Charge(z_val),
                num_samples=num_samples,
                rng_ctx=rng_ctx,
            )

            # Статистическая проверка на адаптивной сетке бинов по (1 - cos(theta))
            counts_adapt, _ = np.histogram(cos_samples, bins=adaptive_edges)
            obs_counts_adapt = counts_adapt.astype(np.float64)
            emp_probs_adapt = obs_counts_adapt / float(num_samples)

            theo_probs_adapt, _, theo_mean_cos, theo_std_cos = _compute_bin_probabilities_gauss_legendre(
                bin_edges=adaptive_edges,
                photon_energy=photon_energy,
                atomic_z=z_val,
                is_compton=False,
                quad_order=32,
            )
            exp_counts_adapt = theo_probs_adapt * float(num_samples)
            _, chi2_ndf, red_chi2, chi2_pval = compute_pooled_chi2_statistics(
                obs_counts_adapt, exp_counts_adapt
            )
            all_pvalues.append(chi2_pval)

            l1_error = 0.5 * float(np.sum(np.abs(emp_probs_adapt - theo_probs_adapt)))
            emp_mean_cos = float(np.mean(cos_samples))
            mean_cos_se = theo_std_cos / sqrt_n
            mean_cos_z = abs(emp_mean_cos - theo_mean_cos) / max(mean_cos_se, 1e-30)

            raw_records.append(
                {
                    "element_symbol": sym_str,
                    "atomic_z": z_val,
                    "photon_energy": photon_energy,
                    "num_samples": num_samples,
                    "binning_scheme": "power_law_one_minus_cos_exp3",
                    "mean_cos_theta_mc": emp_mean_cos,
                    "mean_cos_theta_theory": theo_mean_cos,
                    "mean_cos_z_score": mean_cos_z,
                    "reduced_chi2": red_chi2,
                    "chi2_ndf": chi2_ndf,
                    "chi2_pvalue": chi2_pval,
                    "density_l1_error": l1_error,
                }
            )

            is_plot_candidate = (
                (z_val == 8 and abs(photon_energy - float(30.0 * units.keV)) < 1e-6)
                or (
                    z_val == 82
                    and (
                        abs(photon_energy - float(30.0 * units.keV)) < 1e-6
                        or abs(photon_energy - float(80.0 * units.keV)) < 1e-6
                        or abs(photon_energy - float(140.5 * units.keV)) < 1e-6
                    )
                )
            )
            if is_plot_candidate:
                counts_plot, _ = np.histogram(cos_samples, bins=plot_bin_edges)
                emp_density_plot = (counts_plot.astype(np.float64) / float(num_samples)) / plot_bin_widths
                _, theo_density_plot, _, _ = _compute_bin_probabilities_gauss_legendre(
                    bin_edges=plot_bin_edges,
                    photon_energy=photon_energy,
                    atomic_z=z_val,
                    is_compton=False,
                    quad_order=32,
                )
                energy_display = photon_energy * 1000.0
                label_text = f"{sym_str}, {energy_display:.1f} кэВ"
                plot_curves.append((label_text, emp_density_plot, theo_density_plot))

    holm_passed, min_pval, holm_thresholds = evaluate_holm_bonferroni(all_pvalues, alpha=0.001)

    metrics: list[CoherentIndicatrixMetric] = []
    for rec, holm_thresh in zip(raw_records, holm_thresholds):
        pval = float(rec["chi2_pvalue"])  # type: ignore[arg-type]
        l1_val = float(rec["density_l1_error"])  # type: ignore[arg-type]
        z_score_val = float(rec["mean_cos_z_score"])  # type: ignore[arg-type]
        theo_mean = float(rec["mean_cos_theta_theory"])  # type: ignore[arg-type]
        z_elem = int(rec["atomic_z"])  # type: ignore[arg-type]
        e_val = float(rec["photon_energy"])  # type: ignore[arg-type]

        # Для Pb при 140.5 кэВ проверяем отсутствие статистически значимого смещения от теоретического среднего
        forward_theory_consistent = True
        if z_elem == 82 and abs(e_val - float(140.5 * units.keV)) < 1e-6:
            forward_theory_consistent = (theo_mean > 0.85) and (z_score_val < 3.5)

        item_passed = (
            (pval >= holm_thresh)
            and (l1_val < allowed_l1)
            and (z_score_val < 3.8)
            and forward_theory_consistent
        )
        metrics.append(
            CoherentIndicatrixMetric(
                element_symbol=str(rec["element_symbol"]),
                atomic_z=z_elem,
                photon_energy=e_val,
                num_samples=int(rec["num_samples"]),  # type: ignore[arg-type]
                binning_scheme=str(rec["binning_scheme"]),
                mean_cos_theta_mc=float(rec["mean_cos_theta_mc"]),  # type: ignore[arg-type]
                mean_cos_theta_theory=theo_mean,
                mean_cos_z_score=z_score_val,
                reduced_chi2=float(rec["reduced_chi2"]),  # type: ignore[arg-type]
                chi2_ndf=int(rec["chi2_ndf"]),  # type: ignore[arg-type]
                chi2_pvalue=pval,
                holm_threshold=holm_thresh,
                density_l1_error=l1_val,
                passed=item_passed,
            )
        )

    return metrics, plot_curves, holm_passed, min_pval


def _run_rotation_and_phi_audit(
    num_phi_samples: int,
    rng: np.random.Generator,
    rng_ctx: RNGContext,
) -> tuple[RotationAndAzimuthMetric, NDArray[np.float64], NDArray[np.float64]]:
    """Проверяет равномерность phi, восстановление азимута в 3D-повороте rotate_kernel и изотропизацию."""
    raw_phi = np.asarray(_sample_phi_batch(num_phi_samples, rng_ctx), dtype=np.float64)
    phi_in_native_range = bool(np.all((raw_phi >= -math.pi) & (raw_phi < math.pi)))

    num_phi_bins = 60
    native_phi_edges = np.linspace(-math.pi, math.pi, num_phi_bins + 1, dtype=np.float64)
    native_phi_counts, _ = np.histogram(raw_phi, bins=native_phi_edges)
    obs_phi = native_phi_counts.astype(np.float64)
    exp_phi = np.full(num_phi_bins, float(num_phi_samples) / float(num_phi_bins), dtype=np.float64)
    phi_chi2_stat = float(np.sum(((obs_phi - exp_phi) ** 2) / exp_phi))
    phi_ndf = num_phi_bins - 1
    phi_reduced_chi2 = phi_chi2_stat / float(phi_ndf)
    phi_pvalue = compute_chi2_pvalue(phi_chi2_stat, phi_ndf)

    plot_phi_edges = np.linspace(0.0, 2.0 * math.pi, num_phi_bins + 1, dtype=np.float64)
    phi_centers = 0.5 * (plot_phi_edges[:-1] + plot_phi_edges[1:])
    phi_widths = plot_phi_edges[1:] - plot_phi_edges[:-1]
    shifted_phi = raw_phi + math.pi
    plot_phi_counts, _ = np.histogram(shifted_phi, bins=plot_phi_edges)
    phi_density_mc = (plot_phi_counts.astype(np.float64) / float(num_phi_samples)) / phi_widths

    # 1. Аудит однократного 3D-поворота rotate_kernel на 10^5 единичных векторах (включая полюса |d_{0,z}| = 1)
    num_vectors = 100_000
    num_singular = 10_000
    num_regular = num_vectors - num_singular

    reg_cos_z = rng.uniform(-1.0, 1.0, size=num_regular)
    reg_azim = rng.uniform(0.0, 2.0 * math.pi, size=num_regular)
    reg_sin_z = np.sqrt(np.maximum(0.0, 1.0 - reg_cos_z * reg_cos_z))

    sing_signs = np.where(rng.uniform(0.0, 1.0, size=num_singular) < 0.5, 1.0, -1.0)
    sing_abs_z = rng.uniform(0.9999900001, 1.0, size=num_singular)
    sing_abs_z[0] = 1.0
    sing_abs_z[1] = 1.0
    sing_signs[0] = 1.0
    sing_signs[1] = -1.0
    sing_cos_z = sing_signs * sing_abs_z
    sing_azim = rng.uniform(0.0, 2.0 * math.pi, size=num_singular)
    sing_sin_z = np.sqrt(np.maximum(0.0, 1.0 - sing_cos_z * sing_cos_z))

    init_x = np.concatenate([reg_sin_z * np.cos(reg_azim), sing_sin_z * np.cos(sing_azim)])
    init_y = np.concatenate([reg_sin_z * np.sin(reg_azim), sing_sin_z * np.sin(sing_azim)])
    init_z = np.concatenate([reg_cos_z, sing_cos_z])

    init_norms = np.sqrt(init_x * init_x + init_y * init_y + init_z * init_z)
    init_x /= init_norms
    init_y /= init_norms
    init_z /= init_norms

    state = KinematicState.allocate(num_vectors)
    state.direction.x[:] = init_x.astype(Float)
    state.direction.y[:] = init_y.astype(Float)
    state.direction.z[:] = init_z.astype(Float)

    # Полярные углы выбираются внутри (1e-3, pi - 1e-3), где азимутальный угол phi однозначно определен
    scatter_thetas = rng.uniform(1.0e-3, math.pi - 1.0e-3, size=num_vectors).astype(Float)
    scatter_phis = rng.uniform(-math.pi, math.pi, size=num_vectors).astype(Float)
    target_indices = np.arange(num_vectors, dtype=np.int64)

    rotate_kernel(state, target_indices, scatter_thetas, scatter_phis)

    rot_x = np.asarray(state.direction.x, dtype=np.float64)
    rot_y = np.asarray(state.direction.y, dtype=np.float64)
    rot_z = np.asarray(state.direction.z, dtype=np.float64)

    rot_norms = np.sqrt(rot_x * rot_x + rot_y * rot_y + rot_z * rot_z)
    dot_products = init_x * rot_x + init_y * rot_y + init_z * rot_z
    expected_cos = np.cos(scatter_thetas.astype(np.float64))

    max_norm_err = float(np.max(np.abs(rot_norms - 1.0)))
    max_dot_err = float(np.max(np.abs(dot_products - expected_cos)))

    # Восстановление азимутального угла phi в локальном ортонормированном базисе (e_1, e_2, d_0)
    sign_z = np.where(init_z < 0.0, -1.0, 1.0)
    denom_z = 1.0 + np.abs(init_z)
    basis_e1_x = 1.0 - (init_x * init_x) / denom_z
    basis_e1_y = -(init_x * init_y) / denom_z
    basis_e1_z = -sign_z * init_x

    basis_e2_x = -(init_x * init_y) / denom_z
    basis_e2_y = 1.0 - (init_y * init_y) / denom_z
    basis_e2_z = -sign_z * init_y

    proj_e1 = rot_x * basis_e1_x + rot_y * basis_e1_y + rot_z * basis_e1_z
    proj_e2 = rot_x * basis_e2_x + rot_y * basis_e2_y + rot_z * basis_e2_z
    reconstructed_phis = np.arctan2(proj_e2, proj_e1)
    input_phis = scatter_phis.astype(np.float64)
    azimuth_diffs = np.abs(
        np.arctan2(np.sin(reconstructed_phis - input_phis), np.cos(reconstructed_phis - input_phis))
    )
    max_azimuth_err = float(np.max(azimuth_diffs))

    # 2. Проверка стремления к изотропии (||<d_k>|| -> 0) после цепочки многократных рассеяний
    chain_particles = 25_000
    chain_steps = 25
    chain_state = KinematicState.allocate(chain_particles)
    chain_state.direction.x[:] = 0.0
    chain_state.direction.y[:] = 0.0
    chain_state.direction.z[:] = 1.0
    chain_indices = np.arange(chain_particles, dtype=np.int64)
    fixed_step_thetas = np.full(chain_particles, math.acos(0.5), dtype=Float)

    for _ in range(chain_steps):
        step_phis = rng.uniform(-math.pi, math.pi, size=chain_particles).astype(Float)
        rotate_kernel(chain_state, chain_indices, fixed_step_thetas, step_phis)

    mean_dir_x = float(np.mean(chain_state.direction.x))
    mean_dir_y = float(np.mean(chain_state.direction.y))
    mean_dir_z = float(np.mean(chain_state.direction.z))
    mean_direction_norm = math.sqrt(
        mean_dir_x * mean_dir_x + mean_dir_y * mean_dir_y + mean_dir_z * mean_dir_z
    )
    # При изотропном распределении единичных векторов в R^3 среднеквадратичная норма среднего равна 1/sqrt(N)
    isotropy_threshold = 4.5 / math.sqrt(float(chain_particles))

    passed = (
        phi_in_native_range
        and (phi_pvalue >= 1.0e-4)
        and (max_norm_err < 1e-14)
        and (max_dot_err < 1e-14)
        and (max_azimuth_err < 1e-12)
        and (mean_direction_norm < isotropy_threshold)
    )

    metric = RotationAndAzimuthMetric(
        num_phi_samples=num_phi_samples,
        phi_in_native_range=phi_in_native_range,
        phi_reduced_chi2=phi_reduced_chi2,
        phi_chi2_pvalue=phi_pvalue,
        num_rotation_vectors=num_vectors,
        num_singular_vectors=num_singular,
        max_norm_error=max_norm_err,
        max_dot_product_error=max_dot_err,
        max_azimuth_reconstruction_error=max_azimuth_err,
        multi_scatter_chain_steps=chain_steps,
        multi_scatter_mean_direction_norm=mean_direction_norm,
        multi_scatter_isotropy_threshold=isotropy_threshold,
        passed=passed,
    )
    return metric, phi_centers, phi_density_mc


def run_exp02_scattering_indicatrices_and_kinematics(
    output_dir: Path,
    quick: bool = False,
) -> Exp02ScatteringIndicatricesReport:
    """Запускает Эксперимент 2 (индикатрисы Комптона S(x, Z), Рэлея F(x, Z), кинематика отдачи и 3D-поворот).

    Args:
        output_dir: Директория для сохранения JSON-отчета и PNG-графика.
        quick: Режим экспресс-проверки с уменьшенным объемом выборки Монте-Карло.

    Returns:
        Структурированный отчет Exp02ScatteringIndicatricesReport.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_mc_samples = 120_000 if quick else 500_000

    rng = np.random.default_rng(np.random.SeedSequence(20261105))
    rng_ctx = RNGContext.from_numpy_rng(rng)

    num_cos_bins = 40
    cos_bin_edges = np.linspace(-1.0, 1.0, num_cos_bins + 1, dtype=np.float64)
    cos_grid = 0.5 * (cos_bin_edges[:-1] + cos_bin_edges[1:])

    # 1. Граничные свойства S(x, Z), F(x, Z) и индикатриса некогерентного рассеяния
    boundary_metrics = _run_scattering_function_boundary_audit()
    form_factor_metrics = _run_form_factor_boundary_audit()
    compton_metrics, compton_curves, compton_holm_passed, min_compton_pval = _run_compton_indicatrix_audit(
        num_samples=num_mc_samples,
        bin_edges=cos_bin_edges,
        rng_ctx=rng_ctx,
    )

    # 2. Кинематика Комптона и комптоновский край
    kinematics_metrics, kinematics_cos, kinematics_edep_mc, kinematics_edep_theory, compton_edge_energy = (
        _run_compton_kinematics_audit()
    )

    # 3. Индикатриса когерентного (рэлеевского) рассеяния на адаптивных степенных бинах по (1 - cos(theta))
    coherent_metrics, coherent_curves, coherent_holm_passed, min_coherent_pval = (
        _run_coherent_indicatrix_audit(
            num_samples=num_mc_samples,
            plot_bin_edges=cos_bin_edges,
            rng_ctx=rng_ctx,
        )
    )

    # 4. Равномерность азимутального угла phi, восстановление азимута и 3D-поворот rotate_kernel
    rotation_metric, phi_centers, phi_density_mc = _run_rotation_and_phi_audit(
        num_phi_samples=num_mc_samples,
        rng=rng,
        rng_ctx=rng_ctx,
    )

    thomson_curve = 0.375 * (1.0 + cos_grid * cos_grid)
    plot_path = output_dir / "exp02_scattering_indicatrices.png"
    plot_exp02_scattering_indicatrices_and_kinematics(
        cos_grid=cos_grid,
        compton_curves=compton_curves,
        kinematics_cos=kinematics_cos,
        kinematics_edep_mc=kinematics_edep_mc,
        kinematics_edep_theory=kinematics_edep_theory,
        compton_edge_energy=compton_edge_energy,
        coherent_curves=coherent_curves,
        thomson_curve=thomson_curve,
        phi_centers=phi_centers,
        phi_density_mc=phi_density_mc,
        output_path=plot_path,
    )

    max_hubbell_rel_err = max(item.max_hubbell_table_rel_error for item in boundary_metrics)
    max_form_factor_rel_err = max(item.max_external_table_rel_error for item in form_factor_metrics)
    max_compton_chi2 = max(item.reduced_chi2 for item in compton_metrics)
    max_compton_l1 = max(item.density_l1_error for item in compton_metrics)
    max_compton_mean_cos_z = max(item.mean_cos_z_score for item in compton_metrics)
    max_kinematics_rel_err = max(
        max(item.max_grid_rel_error, item.edge_rel_error) for item in kinematics_metrics
    )
    max_coherent_chi2 = max(item.reduced_chi2 for item in coherent_metrics)
    max_coherent_mean_cos_z = max(item.mean_cos_z_score for item in coherent_metrics)
    lead_140_mean_cos = next(
        item.mean_cos_theta_mc
        for item in coherent_metrics
        if item.atomic_z == 82 and abs(item.photon_energy - float(140.5 * units.keV)) < 1e-6
    )

    overall_passed = (
        all(item.passed for item in boundary_metrics)
        and all(item.passed for item in form_factor_metrics)
        and all(item.passed for item in compton_metrics)
        and compton_holm_passed
        and all(item.passed for item in kinematics_metrics)
        and all(item.passed for item in coherent_metrics)
        and coherent_holm_passed
        and rotation_metric.passed
    )

    reference_standard_types: dict[str, str] = {
        "hubbell_1975_s_function_anchor": ReferenceStandardType.EXTERNAL_STANDARD.value,
        "hubbell_1979_cromer_mann_f_factor_anchor": ReferenceStandardType.EXTERNAL_STANDARD.value,
        "klein_nishina_bound_indicatrix_sampling": ReferenceStandardType.KERNEL_ANALYTIC.value,
        "rayleigh_thomson_indicatrix_sampling": ReferenceStandardType.KERNEL_ANALYTIC.value,
        "compton_recoil_kinematics_identity": ReferenceStandardType.SELF_CONSISTENCY.value,
        "rotation_3d_azimuth_and_isotropy": ReferenceStandardType.SELF_CONSISTENCY.value,
    }

    json_path = output_dir / "exp02_scattering_indicatrices.json"
    report = Exp02ScatteringIndicatricesReport(
        passed=overall_passed,
        quick=quick,
        num_mc_samples=num_mc_samples,
        max_hubbell_table_rel_error=max_hubbell_rel_err,
        max_form_factor_table_rel_error=max_form_factor_rel_err,
        max_compton_chi2=max_compton_chi2,
        min_compton_chi2_pvalue=min_compton_pval,
        compton_holm_bonferroni_passed=compton_holm_passed,
        max_compton_l1_error=max_compton_l1,
        max_compton_mean_cos_z_score=max_compton_mean_cos_z,
        max_kinematics_rel_error=max_kinematics_rel_err,
        max_coherent_chi2=max_coherent_chi2,
        min_coherent_chi2_pvalue=min_coherent_pval,
        coherent_holm_bonferroni_passed=coherent_holm_passed,
        max_coherent_mean_cos_z_score=max_coherent_mean_cos_z,
        lead_coherent_forward_mean_cos=lead_140_mean_cos,
        phi_in_native_range=rotation_metric.phi_in_native_range,
        phi_reduced_chi2=rotation_metric.phi_reduced_chi2,
        phi_chi2_pvalue=rotation_metric.phi_chi2_pvalue,
        max_rotation_norm_error=rotation_metric.max_norm_error,
        max_rotation_dot_error=rotation_metric.max_dot_product_error,
        max_rotation_azimuth_error=rotation_metric.max_azimuth_reconstruction_error,
        multi_scatter_mean_direction_norm=rotation_metric.multi_scatter_mean_direction_norm,
        reference_standard_types=reference_standard_types,
        boundary_metrics=boundary_metrics,
        form_factor_metrics=form_factor_metrics,
        compton_metrics=compton_metrics,
        kinematics_metrics=kinematics_metrics,
        coherent_metrics=coherent_metrics,
        rotation_metric=rotation_metric,
        json_path=str(json_path),
        plot_path=str(plot_path),
    )

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(report), json_file, indent=2, ensure_ascii=False)

    return report


__all__ = [
    "ScatteringFunctionBoundaryMetric",
    "FormFactorBoundaryMetric",
    "ComptonIndicatrixMetric",
    "ComptonKinematicsMetric",
    "CoherentIndicatrixMetric",
    "RotationAndAzimuthMetric",
    "Exp02ScatteringIndicatricesReport",
    "run_exp02_scattering_indicatrices_and_kinematics",
]
