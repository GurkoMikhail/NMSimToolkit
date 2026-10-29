"""Общие статистические критерии и аналитические модели индикатрис рассеяния для верификации физики.

Содержит:
1. Классификацию типов эталонов верификации ReferenceStandardType (A/B/C/D).
2. Вычисление p-значений распределения хи-квадрат через регуляризованную верхнюю неполную
   гамма-функцию Q(k/2, chi2/2) (разложение в ряд и цепная дробь Ленца) и поправку
   Холма–Бонферрони для множественных проверок гипотез.
3. Аналитические модели функции некогерентного рассеяния S(x, Z), атомного форм-фактора F(x, Z)
   и угловых плотностей Клейна–Нишины * S(x, Z) и Томсона * F^2(x, Z).

Все физические величины выражены строго в базовых единицах hepunits (МэВ, мм, нс, рад).
"""
from __future__ import annotations

from enum import Enum
import math
from typing import Sequence

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.physics.g4coherent import f_factor
from core.physics.g4coherent_arrays import PP0, PP1, PP2, PP3, PP4, PP5, PP6, PP7, PP8
from core.physics.g4compton import electron_mass_c2
from core.physics.scat_func_fit_param_array import scat_func_fit_param


class ReferenceStandardType(str, Enum):
    """Категория независимости эталона в тестах физической верификации."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    EXTERNAL_STANDARD = "A"
    KERNEL_ANALYTIC = "B"
    SELF_CONSISTENCY = "C"
    CROSS_IMPLEMENTATION = "D"


DEFAULT_F_FACTOR_TABLE: NDArray[np.float64] = np.vstack(
    [PP0, PP1, PP2, PP3, PP4, PP5, PP6, PP7, PP8]
).astype(np.float64)


def compute_chi2_pvalue(chi2_stat: float, ndf: int) -> float:
    """Вычисляет p-значение распределения хи-квадрат P(Chi2 >= chi2_stat | ndf) = Q(ndf/2, chi2_stat/2).

    Использует разложение в степенной ряд для нижней неполной гамма-функции при x < a + 1
    и модифицированный алгоритм Ленца для цепной дроби Лежандра при x >= a + 1
    с относительной точностью ~1e-14 без внешних зависимостей вне math.
    """
    if ndf <= 0:
        raise ValueError(f"Число степеней свободы ndf должно быть строго положительным, получено: {ndf}.")
    if not math.isfinite(chi2_stat) or chi2_stat < 0.0:
        raise ValueError(
            f"Статистика хи-квадрат должна быть конечной и неотрицательной, получено: {chi2_stat}."
        )
    if chi2_stat == 0.0:
        return 1.0

    half_ndf = 0.5 * float(ndf)
    half_chi2 = 0.5 * float(chi2_stat)
    log_prefactor = -half_chi2 + half_ndf * math.log(half_chi2) - math.lgamma(half_ndf)

    if half_chi2 < half_ndf + 1.0:
        if log_prefactor < -700.0:
            return 1.0
        ap_val = half_ndf
        term = 1.0 / half_ndf
        series_sum = term
        for _ in range(1000):
            ap_val += 1.0
            term *= half_chi2 / ap_val
            series_sum += term
            if abs(term) < abs(series_sum) * 1.0e-15:
                break
        lower_reg = series_sum * math.exp(log_prefactor)
        return max(0.0, min(1.0, 1.0 - lower_reg))

    if log_prefactor < -700.0:
        return 0.0

    tiny_guard = 1.0e-300
    b_coeff = half_chi2 + 1.0 - half_ndf
    c_val = 1.0 / tiny_guard
    d_val = 1.0 / (b_coeff if abs(b_coeff) >= tiny_guard else tiny_guard)
    cf_value = d_val

    for step_idx in range(1, 1000):
        an_coeff = -float(step_idx) * (float(step_idx) - half_ndf)
        b_coeff += 2.0
        d_val = an_coeff * d_val + b_coeff
        if abs(d_val) < tiny_guard:
            d_val = tiny_guard
        c_val = b_coeff + an_coeff / c_val
        if abs(c_val) < tiny_guard:
            c_val = tiny_guard
        d_val = 1.0 / d_val
        delta = d_val * c_val
        cf_value *= delta
        if abs(delta - 1.0) < 1.0e-15:
            break

    upper_reg = math.exp(log_prefactor) * cf_value
    return max(0.0, min(1.0, upper_reg))


def evaluate_holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.001,
) -> tuple[bool, float, list[float]]:
    """Выполняет нисходящую процедуру Холма–Бонферрони для контроля групповой ошибки I рода (FWER).

    Args:
        p_values: Последовательность исходных p-значений проверяемых гипотез.
        alpha: Семейственный уровень значимости FWER (по умолчанию 0.001).

    Returns:
        Кортеж (all_passed, min_raw_pvalue, holm_thresholds), где holm_thresholds содержит
        индивидуальные пороги alpha / (M - rank) в порядке исходного списка p_values.
    """
    num_tests = len(p_values)
    if num_tests == 0:
        raise ValueError("Список p_values для процедуры Холма–Бонферрони не может быть пустым.")
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"Уровень значимости alpha должен лежать в (0, 1), получено: {alpha}.")

    indexed_pvalues: list[tuple[float, int]] = []
    for idx, raw_p in enumerate(p_values):
        p_float = float(raw_p)
        if not math.isfinite(p_float) or p_float < 0.0 or p_float > 1.0:
            raise ValueError(f"Некорректное p-значение p_values[{idx}] = {p_float} вне диапазона [0, 1].")
        indexed_pvalues.append((p_float, idx))

    indexed_pvalues.sort(key=lambda item: item[0])
    holm_thresholds = [0.0] * num_tests
    all_passed = True

    for rank_idx, (p_val, orig_idx) in enumerate(indexed_pvalues):
        threshold = alpha / float(num_tests - rank_idx)
        holm_thresholds[orig_idx] = threshold
        if p_val < threshold:
            all_passed = False

    min_raw_pvalue = float(indexed_pvalues[0][0])
    return all_passed, min_raw_pvalue, holm_thresholds


def compute_pooled_chi2_statistics(
    observed_counts: NDArray[np.float64],
    expected_counts: NDArray[np.float64],
    min_expected: float = 20.0,
) -> tuple[float, int, float, float]:
    """Вычисляет статистику хи-квадрат Пирсона с объединением малозаселенных бинов (E_k >= min_expected).

    Returns:
        Кортеж (chi2_stat, ndf, reduced_chi2, p_value).
    """
    if observed_counts.shape != expected_counts.shape or observed_counts.ndim != 1:
        raise ValueError("Массивы observed_counts и expected_counts должны быть одномерными одной длины.")
    if min_expected <= 0.0:
        raise ValueError("Порог объединения бинов min_expected должен быть положительным.")

    pooled_obs: list[float] = []
    pooled_exp: list[float] = []
    accum_obs = 0.0
    accum_exp = 0.0

    for obs_val, exp_val in zip(observed_counts, expected_counts):
        accum_obs += float(obs_val)
        accum_exp += float(exp_val)
        if accum_exp >= min_expected:
            pooled_obs.append(accum_obs)
            pooled_exp.append(accum_exp)
            accum_obs = 0.0
            accum_exp = 0.0

    if accum_exp > 0.0:
        if len(pooled_exp) > 0:
            pooled_obs[-1] += accum_obs
            pooled_exp[-1] += accum_exp
        else:
            pooled_obs.append(accum_obs)
            pooled_exp.append(accum_exp)

    num_effective_bins = len(pooled_exp)
    if num_effective_bins <= 1:
        raise ValueError(
            f"После объединения бинов осталось {num_effective_bins} бинов (требуется минимум 2 для критерия хи-квадрат)."
        )

    ndf = num_effective_bins - 1
    chi2_stat = sum(
        ((obs_item - exp_item) ** 2) / exp_item
        for obs_item, exp_item in zip(pooled_obs, pooled_exp)
    )
    reduced_chi2 = float(chi2_stat) / float(ndf)
    p_value = compute_chi2_pvalue(float(chi2_stat), ndf)
    return float(chi2_stat), ndf, reduced_chi2, p_value


def _extract_f_factor_parameters(
    atomic_number: int,
    f_factor_table: NDArray[np.float64],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Извлекает 9 параметров аппроксимации форм-фактора (PP0..PP8) для заданного заряда ядра Z."""
    if f_factor_table.ndim != 2:
        raise ValueError(f"Таблица форм-факторов должна быть двумерной, получено ndim={f_factor_table.ndim}.")
    if f_factor_table.shape[0] == 9 and 1 <= atomic_number < f_factor_table.shape[1]:
        col = f_factor_table[:, atomic_number]
    elif f_factor_table.shape[1] == 9 and 1 <= atomic_number < f_factor_table.shape[0]:
        col = f_factor_table[atomic_number, :]
    else:
        raise ValueError(
            f"Некорректный индекс Z={atomic_number} для таблицы форм-факторов формы {f_factor_table.shape}."
        )
    return (
        float(col[0]),
        float(col[1]),
        float(col[2]),
        float(col[3]),
        float(col[4]),
        float(col[5]),
        float(col[6]),
        float(col[7]),
        float(col[8]),
    )


def evaluate_hubbell_s_function(
    momentum_transfer: NDArray[np.float64],
    atomic_number: int,
) -> NDArray[np.float64]:
    """Вычисляет функцию некогерентного рассеяния S(x, Z) по параметризации Hubbell/Geant4.

    Аргумент momentum_transfer задается в конвенции Geant4 x = sin(theta/2) / lambda [1/см]
    (то есть x = 10^8 * x_[1/Angstrom]).
    """
    if atomic_number < 1 or atomic_number >= scat_func_fit_param.shape[0]:
        raise ValueError(f"Атомный номер Z={atomic_number} вне диапазона таблицы scat_func_fit_param.")
    x_arr = np.asarray(momentum_transfer, dtype=np.float64)
    if np.any(x_arr < 0.0):
        raise ValueError("Переданный импульс momentum_transfer должен быть неотрицательным.")

    row = scat_func_fit_param[atomic_number]
    lg_split_low = float(row[1])
    lg_split_high = float(row[2])
    x_cutoff = float(row[3])

    flat_x = x_arr.ravel()
    flat_out = np.empty_like(flat_x)

    for idx in range(flat_x.shape[0]):
        x_val = float(flat_x[idx])
        if x_val <= 0.0:
            flat_out[idx] = 0.0
        elif x_val > x_cutoff:
            flat_out[idx] = float(atomic_number)
        else:
            lg_x = math.log10(max(x_val, 1.0e-300))
            if lg_x < lg_split_low:
                lg_s = float(row[4]) + lg_x * float(row[5])
            elif lg_x < lg_split_high:
                lg_s = float(row[6]) + lg_x * (
                    float(row[7])
                    + lg_x * (float(row[8]) + lg_x * (float(row[9]) + lg_x * float(row[10])))
                )
            else:
                lg_s = float(row[11]) + lg_x * (
                    float(row[12])
                    + lg_x * (float(row[13]) + lg_x * (float(row[14]) + lg_x * float(row[15])))
                )
            flat_out[idx] = math.pow(10.0, lg_s)

    return flat_out.reshape(x_arr.shape)


def evaluate_form_factor_f(
    momentum_transfer: NDArray[np.float64],
    atomic_number: int,
    f_factor_table: NDArray[np.float64] = DEFAULT_F_FACTOR_TABLE,
) -> NDArray[np.float64]:
    """Вычисляет атомный форм-фактор когерентного рассеяния F(x, Z) по 3-членной аппроксимации Geant4.

    Аргумент momentum_transfer задается в конвенции Geant4 x = sin(theta/2) / lambda [1/см]
    (то есть x = 10^8 * x_[1/Angstrom]). При x = 0 выполняется F(0, Z) = Z.
    """
    x_arr = np.asarray(momentum_transfer, dtype=np.float64)
    if np.any(x_arr < 0.0):
        raise ValueError("Переданный импульс momentum_transfer должен быть неотрицательным.")

    amp_0, amp_1, amp_2, beta_0, beta_1, beta_2, pow_0, pow_1, pow_2 = _extract_f_factor_parameters(
        atomic_number, f_factor_table
    )
    x_sq = x_arr * x_arr
    term_0 = amp_0 * np.power(1.0 + beta_0 * x_sq, -pow_0)
    term_1 = amp_1 * np.power(1.0 + beta_1 * x_sq, -pow_1)
    term_2 = amp_2 * np.power(1.0 + beta_2 * x_sq, -pow_2)
    f_squared = np.maximum(0.0, term_0 + term_1 + term_2)
    return np.sqrt(f_squared)


def _evaluate_klein_nishina_bound_weights(
    cos_grid: NDArray[np.float64],
    energy: float,
    atomic_number: int,
) -> NDArray[np.float64]:
    """Вычисляет ненормированную угловую плотность Клейна–Нишины * S(x, Z) на сетке cos(theta)."""
    if energy <= 0.0:
        raise ValueError(f"Энергия фотона должна быть положительной, получено: {energy}.")
    cos_arr = np.asarray(cos_grid, dtype=np.float64)
    one_minus_cos = np.clip(1.0 - cos_arr, 0.0, 2.0)
    sin_sq = np.maximum(0.0, one_minus_cos * (2.0 - one_minus_cos))

    reduced_energy = float(energy) / float(electron_mass_c2)
    epsilon = 1.0 / (1.0 + reduced_energy * one_minus_cos)
    kn_factor = (epsilon * epsilon) * (epsilon + (1.0 / epsilon) - sin_sq)

    wavelength_photon = float(units.h_Planck * units.c_light) / float(energy)
    momentum_transfer = np.sqrt(0.5 * one_minus_cos) * (float(units.cm) / wavelength_photon)
    scat_func = evaluate_hubbell_s_function(momentum_transfer, atomic_number)
    return kn_factor * scat_func


def compute_klein_nishina_bound_pdf(
    cos_grid: NDArray[np.float64],
    energy: float,
    atomic_number: int,
) -> NDArray[np.float64]:
    """Вычисляет нормированную на [-1, 1] угловую плотность вероятности p(cos(theta)) для Комптона с S(x, Z)."""
    weights = _evaluate_klein_nishina_bound_weights(cos_grid, energy, atomic_number)
    quad_nodes, quad_weights = np.polynomial.legendre.leggauss(64)
    quad_vals = _evaluate_klein_nishina_bound_weights(quad_nodes, energy, atomic_number)
    norm_integral = float(np.sum(quad_vals * quad_weights))
    if norm_integral <= 0.0:
        raise ValueError("Нормировочный интеграл индикатрисы Клейна–Нишины неположителен.")
    return weights / norm_integral


def _evaluate_rayleigh_thomson_weights(
    cos_grid: NDArray[np.float64],
    energy: float,
    atomic_number: int,
    f_factor_table: NDArray[np.float64] = DEFAULT_F_FACTOR_TABLE,
) -> NDArray[np.float64]:
    """Вычисляет ненормированную угловую плотность Томсона * F^2(x, Z) на сетке cos(theta)."""
    if energy <= 0.0:
        raise ValueError(f"Энергия фотона должна быть положительной, получено: {energy}.")
    cos_arr = np.asarray(cos_grid, dtype=np.float64)
    one_minus_cos = np.clip(1.0 - cos_arr, 0.0, 2.0)
    xx_val = float(f_factor) * float(energy) * float(energy)
    momentum_transfer = np.sqrt(xx_val * one_minus_cos)
    form_factor = evaluate_form_factor_f(momentum_transfer, atomic_number, f_factor_table)
    thomson_factor = 1.0 + cos_arr * cos_arr
    return thomson_factor * (form_factor * form_factor)


def compute_rayleigh_thomson_pdf(
    cos_grid: NDArray[np.float64],
    energy: float,
    atomic_number: int,
    f_factor_table: NDArray[np.float64] = DEFAULT_F_FACTOR_TABLE,
) -> NDArray[np.float64]:
    """Вычисляет нормированную на [-1, 1] угловую плотность вероятности p(cos(theta)) для Рэлея с F^2(x, Z)."""
    weights = _evaluate_rayleigh_thomson_weights(cos_grid, energy, atomic_number, f_factor_table)
    quad_nodes, quad_weights = np.polynomial.legendre.leggauss(64)
    quad_vals = _evaluate_rayleigh_thomson_weights(quad_nodes, energy, atomic_number, f_factor_table)
    norm_integral = float(np.sum(quad_vals * quad_weights))
    if norm_integral <= 0.0:
        raise ValueError("Нормировочный интеграл индикатрисы Рэлея неположителен.")
    return weights / norm_integral


__all__ = [
    "DEFAULT_F_FACTOR_TABLE",
    "ReferenceStandardType",
    "compute_chi2_pvalue",
    "compute_klein_nishina_bound_pdf",
    "compute_pooled_chi2_statistics",
    "compute_rayleigh_thomson_pdf",
    "evaluate_form_factor_f",
    "evaluate_holm_bonferroni",
    "evaluate_hubbell_s_function",
]
