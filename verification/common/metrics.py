"""Статистические и физические метрики для верификации моделей коллиматоров."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

from verification.common.history_classifier import (
    ClassifiedExitPhotons,
    PhotonComponent,
)


@dataclass(frozen=True)
class ComponentSensitivity:
    """Абсолютная чувствительность и статистика для одной компоненты PSF.

    Attributes:
        count: Число зарегистрированных фотонов данной компоненты в энергетическом окне.
        sensitivity: Доля прошедших фотонов в пересчете на полный телесный угол 4*pi.
        sensitivity_error: Стандартная пуассоновская ошибка чувствительности.
        fraction_of_total: Доля данной компоненты в общем числе прошедших фотонов [0..1].
    """

    count: int
    sensitivity: float
    sensitivity_error: float
    fraction_of_total: float


@dataclass(frozen=True)
class CollimatorSensitivityBreakdown:
    """Полная покомпонентная разбивка чувствительности коллиматора.

    Attributes:
        emitted_count: Число сгенерированных фотонов в симуляции.
        solid_angle_fraction: Доля телесного угла 4*pi, в которую испускались фотоны.
        total: Суммарная чувствительность по всем компонентам.
        geometric: Геометрическая компонента (прохождение через один канал).
        septal_penetration: Септальное проникновение (прямой пролет сквозь перегородки).
        collimator_scatter: Рассеяние на свинце коллиматора.
    """

    emitted_count: int
    solid_angle_fraction: float
    total: ComponentSensitivity
    geometric: ComponentSensitivity
    septal_penetration: ComponentSensitivity
    collimator_scatter: ComponentSensitivity


@dataclass(frozen=True)
class ProfileResolutionMetrics:
    """Метрики пространственного разрешения одномерного профиля PSF.

    Attributes:
        fwhm_x: Полная ширина на половине высоты вдоль оси X [мм].
        fwtm_x: Полная ширина на десятой части высоты вдоль оси X [мм].
        fwhm_y: Полная ширина на половине высоты вдоль оси Y [мм].
        fwtm_y: Полная ширина на десятой части высоты вдоль оси Y [мм].
        fwhm_mean: Среднее арифметическое FWHM по осям X и Y [мм].
        fwtm_mean: Среднее арифметическое FWTM по осям X и Y [мм].
        radial_r50: Радиус круга, содержащего 50% энергии PSF [мм].
        radial_r90: Радиус круга, содержащего 90% энергии PSF [мм].
    """

    fwhm_x: float
    fwtm_x: float
    fwhm_y: float
    fwtm_y: float
    fwhm_mean: float
    fwtm_mean: float
    radial_r50: float
    radial_r90: float


@dataclass(frozen=True)
class StatisticalComparisonResult:
    """Результаты статистического сравнения двух распределений (Parametric vs Direct).

    Attributes:
        ks_statistic_radial: Статистика Колмогорова-Смирнова для радиальных координат R.
        ks_pvalue_radial: P-значение двухвыборочного теста Колмогорова-Смирнова по R.
        reduced_chi2_radial: Приведенный хи-квадрат (chi2 / ndf) радиальных гистограмм.
        ndf_radial: Число степеней свободы радиальной гистограммы.
        max_abs_z_score_2d: Максимальное абсолютное значение Z-score на 2D-карте PSF.
        fraction_within_3sigma_2d: Доля активных бинов 2D-карты с |Z-score| <= 3.0.
        relative_sensitivity_diff: Относительная разница полной чувствительности.
        sensitivity_z_score: Z-score разницы полной чувствительности с учетом ошибок Пуассона.
        geometric_z_score: Z-score разницы геометрической компоненты чувствительности.
        septal_z_score: Z-score разницы компоненты септального проникновения.
        scatter_z_score: Z-score разницы компоненты рассеяния в коллиматоре.
    """

    ks_statistic_radial: float
    ks_pvalue_radial: float
    reduced_chi2_radial: float
    ndf_radial: int
    max_abs_z_score_2d: float
    fraction_within_3sigma_2d: float
    relative_sensitivity_diff: float
    sensitivity_z_score: float
    geometric_z_score: float
    septal_z_score: float
    scatter_z_score: float


@dataclass(frozen=True)
class AzimuthalStarMetrics:
    """Метрики азимутальной анизотропии («эффекта звезды» 6 лучей) в хвостах PSF.

    Attributes:
        bin_centers_angle: Центры угловых бинов [0, 2*pi) [рад].
        angular_counts: Распределение отсчетов по азимутальному углу в выбранном кольце.
        anisotropy_contrast: Контраст 6-лучевой модуляции (I_max - I_min) / (I_max + I_min).
        hexagonal_harmonic_ratio: Амплитуда 6-й гармоники Фурье относительно нулевой гармоники.
        dominant_lobe_angles: Углы шести главных максимумов [рад].
    """

    bin_centers_angle: NDArray[np.float64]
    angular_counts: NDArray[np.float64]
    anisotropy_contrast: float
    hexagonal_harmonic_ratio: float
    dominant_lobe_angles: NDArray[np.float64]


def filter_energy_window(
    photons: ClassifiedExitPhotons,
    nominal_energy: float,
    energy_window_half_width_fraction: float = 0.10,
) -> NDArray[np.bool_]:
    """Возвращает булеву маску фотонов, попавших в энергетическое окно фотопика.

    Args:
        photons: Классифицированные фотоны на выходе из коллиматора.
        nominal_energy: Номинальная энергия фотопика [МэВ].
        energy_window_half_width_fraction: Полуширина энергетического окна (например, 0.10 для +-10%).

    Returns:
        Булев массив длины photons.total_count.
    """
    lower_bound = nominal_energy * (1.0 - energy_window_half_width_fraction)
    upper_bound = nominal_energy * (1.0 + energy_window_half_width_fraction)
    return np.asarray(
        (photons.final_energy >= lower_bound) & (photons.final_energy <= upper_bound),
        dtype=np.bool_,
    )


def compute_sensitivity_breakdown(
    photons: ClassifiedExitPhotons,
    emitted_count: int,
    solid_angle_fraction: float,
    energy_mask: NDArray[np.bool_] | None = None,
) -> CollimatorSensitivityBreakdown:
    """Вычисляет абсолютную чувствительность и доли трех физических компонент PSF.

    Если симуляция проводилась с конусным источником (solid_angle_fraction < 1.0),
    число зарегистрированных фотонов умножается на solid_angle_fraction для приведения
    к честной абсолютной чувствительности в телесном угле 4*pi.

    Args:
        photons: Классифицированные фотоны на выходе из коллиматора.
        emitted_count: Общее число запущенных фотонов источника.
        solid_angle_fraction: Отношение телесного угла генерации к 4*pi.
        energy_mask: Опциональная булева маска энергетического окна.

    Returns:
        Структура CollimatorSensitivityBreakdown.
    """
    if emitted_count <= 0:
        raise ValueError(f"emitted_count должен быть положительным, получено {emitted_count}")
    if solid_angle_fraction <= 0.0 or solid_angle_fraction > 1.0:
        raise ValueError(
            f"solid_angle_fraction должен лежать в (0, 1], получено {solid_angle_fraction}"
        )

    component_array = photons.component if energy_mask is None else photons.component[energy_mask]
    total_count = int(component_array.shape[0])
    geom_count = int(np.sum(component_array == int(PhotonComponent.GEOMETRIC)))
    septal_count = int(np.sum(component_array == int(PhotonComponent.SEPTAL_PENETRATION)))
    scatter_count = int(np.sum(component_array == int(PhotonComponent.COLLIMATOR_SCATTER)))

    norm_factor = solid_angle_fraction / float(emitted_count)
    denom_total = max(total_count, 1)

    def _build_comp(count_val: int) -> ComponentSensitivity:
        sens_val = float(count_val) * norm_factor
        sens_err = math.sqrt(float(max(count_val, 1))) * norm_factor if count_val > 0 else 0.0
        frac_val = float(count_val) / float(denom_total) if total_count > 0 else 0.0
        return ComponentSensitivity(
            count=count_val,
            sensitivity=sens_val,
            sensitivity_error=sens_err,
            fraction_of_total=frac_val,
        )

    return CollimatorSensitivityBreakdown(
        emitted_count=emitted_count,
        solid_angle_fraction=solid_angle_fraction,
        total=_build_comp(total_count),
        geometric=_build_comp(geom_count),
        septal_penetration=_build_comp(septal_count),
        collimator_scatter=_build_comp(scatter_count),
    )


def compute_width_at_fraction(
    bin_centers: NDArray[np.float64],
    profile_values: NDArray[np.float64],
    fraction: float,
) -> float:
    """Вычисляет ширину одномерного профиля на заданной доле от максимума (например, 0.5 или 0.1).

    Использует субпиксельную линейную интерполяцию между соседними бинами вокруг
    полувысоты (или десятой части высоты) слева и справа от пика.

    Args:
        bin_centers: Центры координатных бинов (монотонно возрастающий массив) [мм].
        profile_values: Значения интенсивности в бинах.
        fraction: Доля от максимума (0.5 для FWHM, 0.1 для FWTM).

    Returns:
        Полная ширина профиля на уровне fraction * max_value [мм].
    """
    if profile_values.size < 3:
        return 0.0

    # Трехточечный биномиальный фильтр [0.25, 0.5, 0.25] подавляет пуассоновский шум
    # отдельных бинов и сохраняет инвариантными линейные склоны треугольного профиля Ангера
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    smoothed = np.convolve(profile_values, kernel, mode="same")
    max_val = float(np.max(smoothed))
    if max_val <= 0.0:
        return 0.0

    peak_idx = int(np.argmax(smoothed))
    threshold = max_val * fraction

    # Поиск левого пересечения от пика к левому краю
    left_pos = float(bin_centers[0])
    for idx in range(peak_idx, 0, -1):
        val_high = float(smoothed[idx])
        val_low = float(smoothed[idx - 1])
        if val_low <= threshold <= val_high:
            pos_high = float(bin_centers[idx])
            pos_low = float(bin_centers[idx - 1])
            if val_high > val_low:
                interp_weight = (threshold - val_low) / (val_high - val_low)
                left_pos = pos_low + interp_weight * (pos_high - pos_low)
            else:
                left_pos = 0.5 * (pos_low + pos_high)
            break

    # Поиск правого пересечения от пика к правому краю
    right_pos = float(bin_centers[-1])
    for idx in range(peak_idx, smoothed.size - 1):
        val_high = float(smoothed[idx])
        val_low = float(smoothed[idx + 1])
        if val_low <= threshold <= val_high:
            pos_high = float(bin_centers[idx])
            pos_low = float(bin_centers[idx + 1])
            if val_high > val_low:
                interp_weight = (threshold - val_low) / (val_high - val_low)
                right_pos = pos_low + interp_weight * (pos_high - pos_low)
            else:
                right_pos = 0.5 * (pos_low + pos_high)
            break

    return max(0.0, right_pos - left_pos)


def compute_psf_resolution_metrics(
    coord_x: NDArray[np.float64],
    coord_y: NDArray[np.float64],
    spatial_extent: float = 30.0,
    num_bins: int = 121,
    slice_half_width: float | None = None,
) -> ProfileResolutionMetrics:
    """Вычисляет FWHM, FWTM по осям X, Y и радиальные квантили R50, R90 для пятна рассеяния.

    Для снижения статистического шума одномерные профили вдоль осей X и Y могут
    интегрироваться как маргинальные проекции в полосе |Y| <= slice_half_width
    (или по всей апертуре пика, если slice_half_width is None).

    Args:
        coord_x: Координаты X точек вылета фотонов [мм].
        coord_y: Координаты Y точек вылета фотонов [мм].
        spatial_extent: Полуширина области гистограммирования [-spatial_extent, +spatial_extent] [мм].
        num_bins: Число бинов одномерного профиля.
        slice_half_width: Полуширина ортогональной полосы интегрирования [мм].

    Returns:
        Структура ProfileResolutionMetrics.
    """
    if coord_x.size == 0:
        return ProfileResolutionMetrics(
            fwhm_x=0.0,
            fwtm_x=0.0,
            fwhm_y=0.0,
            fwtm_y=0.0,
            fwhm_mean=0.0,
            fwtm_mean=0.0,
            radial_r50=0.0,
            radial_r90=0.0,
        )

    bin_edges = np.linspace(-spatial_extent, spatial_extent, num_bins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if slice_half_width is not None:
        mask_x = np.abs(coord_y) <= slice_half_width
        mask_y = np.abs(coord_x) <= slice_half_width
        sample_x = coord_x[mask_x]
        sample_y = coord_y[mask_y]
    else:
        sample_x = coord_x
        sample_y = coord_y

    hist_x, _ = np.histogram(sample_x, bins=bin_edges)
    hist_y, _ = np.histogram(sample_y, bins=bin_edges)

    profile_x = hist_x.astype(np.float64)
    profile_y = hist_y.astype(np.float64)

    fwhm_x = compute_width_at_fraction(bin_centers, profile_x, 0.5)
    fwtm_x = compute_width_at_fraction(bin_centers, profile_x, 0.1)
    fwhm_y = compute_width_at_fraction(bin_centers, profile_y, 0.5)
    fwtm_y = compute_width_at_fraction(bin_centers, profile_y, 0.1)

    radial_distances = np.hypot(coord_x, coord_y)
    radial_r50 = float(np.percentile(radial_distances, 50.0))
    radial_r90 = float(np.percentile(radial_distances, 90.0))

    return ProfileResolutionMetrics(
        fwhm_x=fwhm_x,
        fwtm_x=fwtm_x,
        fwhm_y=fwhm_y,
        fwtm_y=fwtm_y,
        fwhm_mean=0.5 * (fwhm_x + fwhm_y),
        fwtm_mean=0.5 * (fwtm_x + fwtm_y),
        radial_r50=radial_r50,
        radial_r90=radial_r90,
    )


def _compute_two_sample_ks(
    sample_first: NDArray[np.float64],
    sample_second: NDArray[np.float64],
) -> tuple[float, float]:
    """Вычисляет двухвыборочную статистику Колмогорова-Смирнова D и асимптотическое p-значение."""
    count_first = sample_first.size
    count_second = sample_second.size
    if count_first == 0 or count_second == 0:
        return 0.0, 1.0

    sorted_first = np.sort(sample_first)
    sorted_second = np.sort(sample_second)
    combined_grid = np.concatenate([sorted_first, sorted_second])

    cdf_first = np.searchsorted(sorted_first, combined_grid, side="right") / float(count_first)
    cdf_second = np.searchsorted(sorted_second, combined_grid, side="right") / float(count_second)
    ks_stat = float(np.max(np.abs(cdf_first - cdf_second)))

    effective_n = (float(count_first) * float(count_second)) / float(count_first + count_second)
    sqrt_ne = math.sqrt(effective_n)
    lambda_val = (sqrt_ne + 0.12 + 0.11 / sqrt_ne) * ks_stat

    if lambda_val <= 1e-6:
        return ks_stat, 1.0

    # Ряд Колмогорова для дополнительной функции распределения
    p_val = 0.0
    for term_idx in range(1, 101):
        sign = 1.0 if (term_idx % 2 == 1) else -1.0
        exponent = -2.0 * (term_idx * lambda_val) ** 2
        if exponent < -500.0:
            break
        p_val += 2.0 * sign * math.exp(exponent)

    p_val = max(0.0, min(1.0, p_val))
    return ks_stat, p_val


def compute_2d_poisson_z_score_map(
    hist_reference: NDArray[np.float64],
    hist_candidate: NDArray[np.float64],
    min_combined_counts: float = 10.0,
    scale_reference: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Строит 2D-карту нормированных пуассоновских невязок (Z-score) между двумя гистограммами.

    Для каждого бина с суммарным числом отсчетов >= min_combined_counts вычисляется:
        Z = (N_cand - scale_ref * N_ref) / sqrt(N_cand + scale_ref^2 * N_ref).

    Args:
        hist_reference: 2D-гистограмма эталонного метода (Parametric).
        hist_candidate: 2D-гистограмма тестируемого метода (Direct).
        min_combined_counts: Минимальное суммарное число отсчетов в бине для включения в статистику.
        scale_reference: Коэффициент масштабирования отсчетов эталонной гистограммы к кандидату.

    Returns:
        Кортеж (z_score_map, active_bins_mask).
    """
    scaled_reference = hist_reference * scale_reference
    combined_variance = (scale_reference ** 2) * hist_reference + hist_candidate
    active_mask = (hist_reference + hist_candidate) >= min_combined_counts
    z_score_map = np.zeros_like(hist_reference, dtype=np.float64)
    z_score_map[active_mask] = (
        hist_candidate[active_mask] - scaled_reference[active_mask]
    ) / np.sqrt(combined_variance[active_mask])
    return z_score_map, active_mask


def compare_collimator_outputs(
    photons_reference: ClassifiedExitPhotons,
    photons_candidate: ClassifiedExitPhotons,
    emitted_reference: int,
    emitted_candidate: int,
    solid_angle_fraction: float,
    spatial_extent: float = 40.0,
    num_bins_2d: int = 80,
    num_bins_radial: int = 60,
    energy_mask_reference: NDArray[np.bool_] | None = None,
    energy_mask_candidate: NDArray[np.bool_] | None = None,
) -> StatisticalComparisonResult:
    """Выполняет полное статистическое сравнение результатов Parametric и Direct коллиматоров.

    Args:
        photons_reference: Выходные фотоны эталонного коллиматора (Parametric).
        photons_candidate: Выходные фотоны проверяемого коллиматора (Direct).
        emitted_reference: Число историй эталонного прогона.
        emitted_candidate: Число историй проверяемого прогона.
        solid_angle_fraction: Доля телесного угла генерации.
        spatial_extent: Полуширина 2D/радиальной области анализа [мм].
        num_bins_2d: Число бинов вдоль каждой оси для 2D-карты Z-score.
        num_bins_radial: Число бинов для радиального критерия хи-квадрат.
        energy_mask_reference: Опциональная маска фотопика для эталона.
        energy_mask_candidate: Опциональная маска фотопика для кандидата.

    Returns:
        Структура StatisticalComparisonResult.
    """
    ref_x = (
        photons_reference.relative_x
        if energy_mask_reference is None
        else photons_reference.relative_x[energy_mask_reference]
    )
    ref_y = (
        photons_reference.relative_y
        if energy_mask_reference is None
        else photons_reference.relative_y[energy_mask_reference]
    )
    cand_x = (
        photons_candidate.relative_x
        if energy_mask_candidate is None
        else photons_candidate.relative_x[energy_mask_candidate]
    )
    cand_y = (
        photons_candidate.relative_y
        if energy_mask_candidate is None
        else photons_candidate.relative_y[energy_mask_candidate]
    )

    ref_radial = np.hypot(ref_x, ref_y)
    cand_radial = np.hypot(cand_x, cand_y)

    ks_stat, ks_pval = _compute_two_sample_ks(ref_radial, cand_radial)

    # Радиальная гистограмма и приведенный хи-квадрат с учетом возможного различия числа историй
    radial_edges = np.linspace(0.0, spatial_extent, num_bins_radial + 1, dtype=np.float64)
    hist_radial_ref, _ = np.histogram(ref_radial, bins=radial_edges)
    hist_radial_cand, _ = np.histogram(cand_radial, bins=radial_edges)

    scale_ref_to_cand = float(emitted_candidate) / float(max(emitted_reference, 1))
    scaled_radial_ref = hist_radial_ref.astype(np.float64) * scale_ref_to_cand
    radial_cand_float = hist_radial_cand.astype(np.float64)
    variance_radial = hist_radial_ref.astype(np.float64) * (scale_ref_to_cand**2) + radial_cand_float

    valid_radial_bins = (hist_radial_ref + hist_radial_cand) >= 10
    ndf_val = int(np.sum(valid_radial_bins))
    if ndf_val > 0:
        chi2_sum = float(
            np.sum(
                ((radial_cand_float[valid_radial_bins] - scaled_radial_ref[valid_radial_bins]) ** 2)
                / variance_radial[valid_radial_bins]
            )
        )
        reduced_chi2 = chi2_sum / float(ndf_val)
    else:
        reduced_chi2 = 0.0

    # 2D гистограмма и Z-score
    edges_2d = np.linspace(-spatial_extent, spatial_extent, num_bins_2d + 1, dtype=np.float64)
    hist2d_ref, _, _ = np.histogram2d(ref_x, ref_y, bins=[edges_2d, edges_2d])
    hist2d_cand, _, _ = np.histogram2d(cand_x, cand_y, bins=[edges_2d, edges_2d])

    z_map, active_mask = compute_2d_poisson_z_score_map(
        hist2d_ref,
        hist2d_cand,
        min_combined_counts=10.0,
        scale_reference=scale_ref_to_cand,
    )
    if np.any(active_mask):
        active_abs_z = np.abs(z_map[active_mask])
        max_abs_z = float(np.max(active_abs_z))
        frac_within_3sigma = float(np.mean(active_abs_z <= 3.0))
    else:
        max_abs_z = 0.0
        frac_within_3sigma = 1.0

    sens_ref = compute_sensitivity_breakdown(
        photons_reference, emitted_reference, solid_angle_fraction, energy_mask_reference
    )
    sens_cand = compute_sensitivity_breakdown(
        photons_candidate, emitted_candidate, solid_angle_fraction, energy_mask_candidate
    )

    def _comp_z(comp_ref: ComponentSensitivity, comp_cand: ComponentSensitivity) -> float:
        err_comb = math.hypot(comp_ref.sensitivity_error, comp_cand.sensitivity_error)
        return (comp_cand.sensitivity - comp_ref.sensitivity) / err_comb if err_comb > 0.0 else 0.0

    ref_sens = sens_ref.total.sensitivity
    cand_sens = sens_cand.total.sensitivity
    rel_diff = (cand_sens - ref_sens) / ref_sens if ref_sens > 0.0 else 0.0
    sens_z = _comp_z(sens_ref.total, sens_cand.total)
    geom_z = _comp_z(sens_ref.geometric, sens_cand.geometric)
    septal_z = _comp_z(sens_ref.septal_penetration, sens_cand.septal_penetration)
    scatter_z = _comp_z(sens_ref.collimator_scatter, sens_cand.collimator_scatter)

    return StatisticalComparisonResult(
        ks_statistic_radial=ks_stat,
        ks_pvalue_radial=ks_pval,
        reduced_chi2_radial=reduced_chi2,
        ndf_radial=ndf_val,
        max_abs_z_score_2d=max_abs_z,
        fraction_within_3sigma_2d=frac_within_3sigma,
        relative_sensitivity_diff=rel_diff,
        sensitivity_z_score=sens_z,
        geometric_z_score=geom_z,
        septal_z_score=septal_z,
        scatter_z_score=scatter_z,
    )


def compute_azimuthal_star_metrics(
    coord_x: NDArray[np.float64],
    coord_y: NDArray[np.float64],
    inner_radius: float,
    outer_radius: float,
    num_angle_bins: int = 72,
) -> AzimuthalStarMetrics:
    """Анализирует 6-лучевую гексагональную анизотропию («эффект звезды») в кольцевом слое PSF.

    В гексагональной решетке тонкие перегородки между соседними отверстиями ориентированы
    по 6 кристаллографическим направлениям с шагом 60 градусов (pi/3 радиан). При высокой
    энергии гамма-квантов септальный пробой формирует характерную 6-лучевую звезду, что
    проявляется в доминировании 6-й гармоники Фурье в азимутальном распределении I(phi).

    Args:
        coord_x: Координаты X вылетевших фотонов [мм].
        coord_y: Координаты Y вылетевших фотонов [мм].
        inner_radius: Внутренний радиус анализируемого кольца хвостов PSF [мм].
        outer_radius: Внешний радиус анализируемого кольца хвостов PSF [мм].
        num_angle_bins: Число бинов по углу в диапазоне [0, 2*pi).

    Returns:
        Структура AzimuthalStarMetrics.
    """
    radial_dist = np.hypot(coord_x, coord_y)
    ring_mask = (radial_dist >= inner_radius) & (radial_dist <= outer_radius)

    ring_x = coord_x[ring_mask]
    ring_y = coord_y[ring_mask]

    angle_edges = np.linspace(0.0, 2.0 * math.pi, num_angle_bins + 1, dtype=np.float64)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    if ring_x.size == 0:
        return AzimuthalStarMetrics(
            bin_centers_angle=angle_centers,
            angular_counts=np.zeros(num_angle_bins, dtype=np.float64),
            anisotropy_contrast=0.0,
            hexagonal_harmonic_ratio=0.0,
            dominant_lobe_angles=np.linspace(0.0, 5.0 * math.pi / 3.0, 6, dtype=np.float64),
        )

    angles = np.mod(np.arctan2(ring_y, ring_x), 2.0 * math.pi)
    counts, _ = np.histogram(angles, bins=angle_edges)
    counts_f = counts.astype(np.float64)

    # Сглаживание по 60-градусной периодичности для устойчивой оценки контраста лучей
    bins_per_period = max(1, num_angle_bins // 6)
    folded_profile = np.zeros(bins_per_period, dtype=np.float64)
    for period_idx in range(6):
        start_bin = period_idx * bins_per_period
        end_bin = start_bin + bins_per_period
        if end_bin <= num_angle_bins:
            folded_profile += counts_f[start_bin:end_bin]

    max_folded = float(np.max(folded_profile))
    min_folded = float(np.min(folded_profile))
    contrast = (
        (max_folded - min_folded) / (max_folded + min_folded)
        if (max_folded + min_folded) > 0.0
        else 0.0
    )

    # Отношение амплитуды 6-й гармоники Фурье к среднему уровню (0-й гармонике)
    total_ring_counts = float(ring_x.size)
    cos_6phi = float(np.sum(np.cos(6.0 * angles)))
    sin_6phi = float(np.sum(np.sin(6.0 * angles)))
    harmonic_6_amp = 2.0 * math.hypot(cos_6phi, sin_6phi) / total_ring_counts

    phase_6 = math.atan2(sin_6phi, cos_6phi)
    base_lobe_angle = (phase_6 / 6.0) % (math.pi / 3.0)
    lobe_angles = np.array(
        [base_lobe_angle + idx * (math.pi / 3.0) for idx in range(6)],
        dtype=np.float64,
    )

    return AzimuthalStarMetrics(
        bin_centers_angle=angle_centers,
        angular_counts=counts_f,
        anisotropy_contrast=contrast,
        hexagonal_harmonic_ratio=harmonic_6_amp,
        dominant_lobe_angles=lobe_angles,
    )


__all__ = [
    "ComponentSensitivity",
    "CollimatorSensitivityBreakdown",
    "ProfileResolutionMetrics",
    "StatisticalComparisonResult",
    "AzimuthalStarMetrics",
    "filter_energy_window",
    "compute_sensitivity_breakdown",
    "compute_width_at_fraction",
    "compute_psf_resolution_metrics",
    "compute_2d_poisson_z_score_map",
    "compare_collimator_outputs",
    "compute_azimuthal_star_metrics",
]

