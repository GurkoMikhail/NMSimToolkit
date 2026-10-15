"""Генерация диагностических графиков для отчетов верификации коллиматоров (Headless Agg)."""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib
import numpy as np
from numpy.typing import NDArray

from verification.common.history_classifier import (
    ClassifiedExitPhotons,
    PhotonComponent,
)
from verification.common.metrics import compute_2d_poisson_z_score_map

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_optical_depth_comparison(
    polar_angles: NDArray[np.float64],
    lead_path_direct: NDArray[np.float64],
    lead_path_reference: NDArray[np.float64],
    transmission_direct: NDArray[np.float64],
    transmission_reference: NDArray[np.float64],
    collimator_code: str,
    nuclide_name: str,
    output_path: Path,
) -> None:
    """Строит график сверки аналитической длины пути в свинце и пропускания (Эксперимент 1).

    Args:
        polar_angles: Полярные углы наклона луча относительно оси Z [рад].
        lead_path_direct: Аналитическая длина пути в свинце из ядра Direct [мм].
        lead_path_reference: Эталонная длина пути в свинце из численного Ray-Marching [мм].
        transmission_direct: Коэффициент пропускания exp(-mu * L_Pb) для Direct.
        transmission_reference: Коэффициент пропускания exp(-mu * L_Pb) для эталона.
        collimator_code: Код коллиматора GE.
        nuclide_name: Название радионуклида.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axis_angles = polar_angles * (180.0 / math.pi)

    ax_path = axes[0]
    ax_path.plot(
        axis_angles,
        lead_path_reference,
        label="Reference (Fine Ray-Marching)",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax_path.plot(
        axis_angles,
        lead_path_direct,
        label="Direct (_hex_prism_intersect)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
    )
    ax_path.set_title(f"{collimator_code}: Длина пути в свинце $L_{{Pb}}(\\theta)$")
    ax_path.set_xlabel("Полярный угол $\\theta$, град.")
    ax_path.set_ylabel("Суммарный путь в свинце, мм")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend()

    ax_trans = axes[1]
    ax_trans.semilogy(
        axis_angles,
        np.maximum(transmission_reference, 1e-12),
        label="Reference $T(\\theta)$",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax_trans.semilogy(
        axis_angles,
        np.maximum(transmission_direct, 1e-12),
        label="Direct $T(\\theta)$",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
    )
    ax_trans.set_title(f"{collimator_code} ({nuclide_name}): Пропускание $\\exp(-\\mu L_{{Pb}})$")
    ax_trans.set_xlabel("Полярный угол $\\theta$, град.")
    ax_trans.set_ylabel("Коэффициент пропускания")
    ax_trans.grid(True, which="both", alpha=0.3)
    ax_trans.legend()

    ax_diff = axes[2]
    abs_diff = np.abs(lead_path_direct - lead_path_reference)
    ax_diff.semilogy(
        axis_angles,
        np.maximum(abs_diff, 1e-15),
        color="#d62728",
        linewidth=1.6,
    )
    ax_diff.set_title("Абсолютная невязка $|L_{Direct} - L_{Ref}|$")
    ax_diff.set_xlabel("Полярный угол $\\theta$, град.")
    ax_diff.set_ylabel("Погрешность пути в свинце, мм")
    ax_diff.grid(True, which="both", alpha=0.3)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_psf_comparison_panel(
    photons_parametric: ClassifiedExitPhotons,
    photons_direct: ClassifiedExitPhotons,
    mask_parametric: NDArray[np.bool_],
    mask_direct: NDArray[np.bool_],
    collimator_code: str,
    nuclide_name: str,
    spatial_extent: float,
    output_path: Path,
) -> None:
    """Строит 4-панельное сравнение 2D/1D PSF между Parametric и Direct (Эксперимент 2).

    Args:
        photons_parametric: Выходные фотоны эталонного коллиматора (Parametric).
        photons_direct: Выходные фотоны проверяемого коллиматора (Direct).
        mask_parametric: Булева маска энергетического окна для Parametric.
        mask_direct: Булева маска энергетического окна для Direct.
        collimator_code: Код модели коллиматора GE.
        nuclide_name: Имя радионуклида.
        spatial_extent: Полуширина отображаемой области [-spatial_extent, +spatial_extent] [мм].
        output_path: Путь для сохранения PNG-изображения.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), constrained_layout=True)

    num_bins_2d = 80
    edges_2d = np.linspace(-spatial_extent, spatial_extent, num_bins_2d + 1, dtype=np.float64)

    param_x = photons_parametric.relative_x[mask_parametric]
    param_y = photons_parametric.relative_y[mask_parametric]
    param_comp = photons_parametric.component[mask_parametric]

    direct_x = photons_direct.relative_x[mask_direct]
    direct_y = photons_direct.relative_y[mask_direct]
    direct_comp = photons_direct.component[mask_direct]

    hist2d_param, _, _ = np.histogram2d(param_x, param_y, bins=[edges_2d, edges_2d])
    hist2d_direct, _, _ = np.histogram2d(direct_x, direct_y, bins=[edges_2d, edges_2d])

    vmax_val = max(float(np.max(hist2d_param)), float(np.max(hist2d_direct)), 10.0)
    extent_list = [-spatial_extent, spatial_extent, -spatial_extent, spatial_extent]

    # Панель A: 2D PSF Parametric
    ax_param_2d = axes[0, 0]
    img_param = ax_param_2d.imshow(
        np.maximum(hist2d_param.T, 0.5),
        origin="lower",
        extent=extent_list,
        cmap="inferno",
        norm=LogNorm(vmin=0.8, vmax=vmax_val),
    )
    ax_param_2d.set_title(f"{collimator_code} ({nuclide_name}): 2D PSF — Parametric (Woodcock)")
    ax_param_2d.set_xlabel("X, мм")
    ax_param_2d.set_ylabel("Y, мм")
    figure.colorbar(img_param, ax=ax_param_2d, label="Отсчеты на бин")

    # Панель B: 2D PSF Direct
    ax_direct_2d = axes[0, 1]
    img_direct = ax_direct_2d.imshow(
        np.maximum(hist2d_direct.T, 0.5),
        origin="lower",
        extent=extent_list,
        cmap="inferno",
        norm=LogNorm(vmin=0.8, vmax=vmax_val),
    )
    ax_direct_2d.set_title(f"{collimator_code} ({nuclide_name}): 2D PSF — Direct (RayCasting)")
    ax_direct_2d.set_xlabel("X, мм")
    ax_direct_2d.set_ylabel("Y, мм")
    figure.colorbar(img_direct, ax=ax_direct_2d, label="Отсчеты на бин")

    # Панель C: Карта нормированных пуассоновских невязок Z-score
    ax_zmap = axes[1, 0]
    z_map, _ = compute_2d_poisson_z_score_map(hist2d_param, hist2d_direct, min_combined_counts=6.0)
    img_z = ax_zmap.imshow(
        z_map.T,
        origin="lower",
        extent=extent_list,
        cmap="coolwarm",
        vmin=-3.5,
        vmax=3.5,
    )
    ax_zmap.set_title("2D карта невязок Пуассона $Z = (N_{Direct} - N_{Param}) / \\sqrt{N_{Direct} + N_{Param}}$")
    ax_zmap.set_xlabel("X, мм")
    ax_zmap.set_ylabel("Y, мм")
    figure.colorbar(img_z, ax=ax_zmap, label="Z-score ($\\sigma$)")

    # Панель D: Покомпонентные 1D-профили вдоль оси X (включая геометрию, септу и рассеяние)
    ax_prof = axes[1, 1]
    num_bins_1d = 81
    edges_1d = np.linspace(-spatial_extent, spatial_extent, num_bins_1d + 1, dtype=np.float64)
    centers_1d = 0.5 * (edges_1d[:-1] + edges_1d[1:])

    component_specs = [
        ("Total PSF", None, "#111111"),
        ("Geometric", int(PhotonComponent.GEOMETRIC), "#1f77b4"),
        ("Septal Penetration", int(PhotonComponent.SEPTAL_PENETRATION), "#d62728"),
        ("Collimator Scatter", int(PhotonComponent.COLLIMATOR_SCATTER), "#2ca02c"),
    ]

    for comp_label, comp_code, color_hex in component_specs:
        if comp_code is None:
            sub_param_x = param_x
            sub_direct_x = direct_x
        else:
            sub_param_x = param_x[param_comp == comp_code]
            sub_direct_x = direct_x[direct_comp == comp_code]

        hist_p, _ = np.histogram(sub_param_x, bins=edges_1d)
        hist_d, _ = np.histogram(sub_direct_x, bins=edges_1d)

        if np.sum(hist_p) == 0 and np.sum(hist_d) == 0:
            continue

        ax_prof.semilogy(
            centers_1d,
            np.maximum(hist_p.astype(np.float64), 0.5),
            color=color_hex,
            linewidth=1.9,
            label=f"{comp_label} (Param)",
        )
        ax_prof.semilogy(
            centers_1d,
            np.maximum(hist_d.astype(np.float64), 0.5),
            color=color_hex,
            linestyle="none",
            marker="o",
            markersize=3.8,
            alpha=0.8,
            label=f"{comp_label} (Direct)",
        )

    ax_prof.set_title("Покомпонентные 1D-профили PSF (проекция на ось X)")
    ax_prof.set_xlabel("Координата X, мм")
    ax_prof.set_ylabel("Отсчеты на бин (лог. шкала)")
    ax_prof.grid(True, which="both", alpha=0.3)
    ax_prof.legend(fontsize=8.5, ncol=2)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_anger_distance_curves(
    distances: NDArray[np.float64],
    fwhm_parametric: NDArray[np.float64],
    fwhm_direct: NDArray[np.float64],
    fwtm_parametric: NDArray[np.float64],
    fwtm_direct: NDArray[np.float64],
    fwhm_anger_theory: NDArray[np.float64],
    sens_geom_parametric: NDArray[np.float64],
    sens_geom_direct: NDArray[np.float64],
    sens_total_parametric: NDArray[np.float64],
    sens_total_direct: NDArray[np.float64],
    sens_anger_theory: float,
    collimator_code: str,
    nuclide_name: str,
    output_path: Path,
) -> None:
    """Строит графики зависимости разрешения FWHM/FWTM и чувствительности от расстояния Z (Эксперимент 3).

    Args:
        distances: Расстояния от точечного источника до входного торца коллиматора [мм].
        fwhm_parametric: Измеренные значения FWHM для Parametric [мм].
        fwhm_direct: Измеренные значения FWHM для Direct [мм].
        fwtm_parametric: Измеренные значения FWTM для Parametric [мм].
        fwtm_direct: Измеренные значения FWTM для Direct [мм].
        fwhm_anger_theory: Теоретические значения FWHM по формуле Ангера [мм].
        sens_geom_parametric: Геометрическая чувствительность Parametric.
        sens_geom_direct: Геометрическая чувствительность Direct.
        sens_total_parametric: Полная чувствительность Parametric.
        sens_total_direct: Полная чувствительность Direct.
        sens_anger_theory: Теоретическая геометрическая чувствительность Ангера.
        collimator_code: Код модели коллиматора GE.
        nuclide_name: Имя радионуклида.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    ax_res = axes[0]
    ax_res.plot(
        distances,
        fwhm_anger_theory,
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.0,
        label="Теория Ангера FWHM ($L_{eff}$)",
    )
    ax_res.plot(
        distances,
        fwhm_parametric,
        color="#1f77b4",
        marker="s",
        linewidth=1.8,
        label="FWHM — Parametric",
    )
    ax_res.plot(
        distances,
        fwhm_direct,
        color="#ff7f0e",
        marker="o",
        linestyle="--",
        linewidth=1.8,
        label="FWHM — Direct",
    )
    ax_res.plot(
        distances,
        fwtm_parametric,
        color="#9467bd",
        marker="^",
        linewidth=1.5,
        label="FWTM — Parametric",
    )
    ax_res.plot(
        distances,
        fwtm_direct,
        color="#d62728",
        marker="v",
        linestyle="--",
        linewidth=1.5,
        label="FWTM — Direct",
    )
    ax_res.set_title(f"{collimator_code} ({nuclide_name}): Пространственное разрешение от расстояния $Z$")
    ax_res.set_xlabel("Расстояние источник–коллиматор $Z$, мм")
    ax_res.set_ylabel("Ширина профиля (FWHM / FWTM), мм")
    ax_res.grid(True, alpha=0.3)
    ax_res.legend()

    ax_sens = axes[1]
    display_scale = 1e4
    ax_sens.axhline(
        sens_anger_theory * display_scale,
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.0,
        label="Инвариант Ангера $S_{geom}$",
    )
    ax_sens.plot(
        distances,
        sens_geom_parametric * display_scale,
        color="#1f77b4",
        marker="s",
        linewidth=1.8,
        label="Geom. Sens — Parametric",
    )
    ax_sens.plot(
        distances,
        sens_geom_direct * display_scale,
        color="#ff7f0e",
        marker="o",
        linestyle="--",
        linewidth=1.8,
        label="Geom. Sens — Direct",
    )
    ax_sens.plot(
        distances,
        sens_total_parametric * display_scale,
        color="#17becf",
        marker="D",
        linewidth=1.5,
        label="Total Sens — Parametric",
    )
    ax_sens.plot(
        distances,
        sens_total_direct * display_scale,
        color="#e377c2",
        marker="x",
        linestyle="--",
        linewidth=1.5,
        label="Total Sens — Direct",
    )
    ax_sens.set_title(f"{collimator_code} ({nuclide_name}): Инвариантность чувствительности от $Z$")
    ax_sens.set_xlabel("Расстояние источник–коллиматор $Z$, мм")
    ax_sens.set_ylabel("Чувствительность ($\\times 10^{-4}$ отсчетов/фотон)")
    ax_sens.set_ylim(bottom=0.0, top=max(float(np.max(sens_total_parametric)), sens_anger_theory) * display_scale * 1.35)
    ax_sens.grid(True, alpha=0.3)
    ax_sens.legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_septal_star_analysis(
    photons_parametric: ClassifiedExitPhotons,
    photons_direct: ClassifiedExitPhotons,
    azimuth_angles: NDArray[np.float64],
    azimuth_counts_parametric: NDArray[np.float64],
    azimuth_counts_direct: NDArray[np.float64],
    collimator_code: str,
    nuclide_name: str,
    spatial_extent: float,
    output_path: Path,
) -> None:
    """Строит визуализацию 6-лучевой звезды септального пробоя и углового профиля (Эксперимент 4).

    Args:
        photons_parametric: Классифицированные фотоны Parametric.
        photons_direct: Классифицированные фотоны Direct.
        azimuth_angles: Центры угловых бинов [рад].
        azimuth_counts_parametric: Угловой профиль интенсивности в кольце хвостов для Parametric.
        azimuth_counts_direct: Угловой профиль интенсивности в кольце хвостов для Direct.
        collimator_code: Код коллиматора GE.
        nuclide_name: Имя радионуклида.
        spatial_extent: Полуширина 2D карты [мм].
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)

    num_bins = 100
    edges = np.linspace(-spatial_extent, spatial_extent, num_bins + 1, dtype=np.float64)
    extent_list = [-spatial_extent, spatial_extent, -spatial_extent, spatial_extent]

    hist_p, _, _ = np.histogram2d(
        photons_parametric.relative_x, photons_parametric.relative_y, bins=[edges, edges]
    )
    hist_d, _, _ = np.histogram2d(
        photons_direct.relative_x, photons_direct.relative_y, bins=[edges, edges]
    )
    vmax_val = max(float(np.max(hist_p)), float(np.max(hist_d)), 10.0)

    ax_p = axes[0]
    im_p = ax_p.imshow(
        np.maximum(hist_p.T, 0.5),
        origin="lower",
        extent=extent_list,
        cmap="magma",
        norm=LogNorm(vmin=0.8, vmax=vmax_val),
    )
    ax_p.set_title(f"{collimator_code} ({nuclide_name}): 2D PSF — Parametric")
    ax_p.set_xlabel("X, мм")
    ax_p.set_ylabel("Y, мм")
    figure.colorbar(im_p, ax=ax_p)

    ax_d = axes[1]
    im_d = ax_d.imshow(
        np.maximum(hist_d.T, 0.5),
        origin="lower",
        extent=extent_list,
        cmap="magma",
        norm=LogNorm(vmin=0.8, vmax=vmax_val),
    )
    ax_d.set_title(f"{collimator_code} ({nuclide_name}): 2D PSF — Direct")
    ax_d.set_xlabel("X, мм")
    ax_d.set_ylabel("Y, мм")
    figure.colorbar(im_d, ax=ax_d)

    ax_ang = axes[2]
    display_azimuth = azimuth_angles * (180.0 / math.pi)
    ax_ang.plot(
        display_azimuth,
        azimuth_counts_parametric,
        color="#1f77b4",
        linewidth=2.0,
        label="Parametric (Woodcock)",
    )
    ax_ang.plot(
        display_azimuth,
        azimuth_counts_direct,
        color="#ff7f0e",
        linestyle="--",
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="Direct (RayCasting)",
    )
    ax_ang.set_title("Азимутальная анизотропия хвостов $I(\\varphi)$ (6 лучей септы)")
    ax_ang.set_xlabel("Азимутальный угол $\\varphi$, град.")
    ax_ang.set_ylabel("Отсчеты в кольцевом слое")
    ax_ang.set_xlim(0.0, 360.0)
    ax_ang.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax_ang.grid(True, alpha=0.3)
    ax_ang.legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_speed_benchmark(
    scenario_labels: list[str],
    throughput_parametric: list[float],
    throughput_direct: list[float],
    speedup_factors: list[float],
    output_path: Path,
) -> None:
    """Строит столбчатые диаграммы производительности и ускорения Direct vs Parametric (Эксперимент 5).

    Args:
        scenario_labels: Подписи конфигураций (например, 'LEHR / Tc-99m').
        throughput_parametric: Скорость счета Parametric [тыс. фотонов / с].
        throughput_direct: Скорость счета Direct [тыс. фотонов / с].
        speedup_factors: Отношение скорости Direct к Parametric.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)

    indices = np.arange(len(scenario_labels), dtype=np.float64)
    bar_width = 0.36

    ax_tp = axes[0]
    ax_tp.bar(
        indices - 0.5 * bar_width,
        throughput_parametric,
        width=bar_width,
        color="#1f77b4",
        label="Parametric (Woodcock Delta-Tracking)",
    )
    ax_tp.bar(
        indices + 0.5 * bar_width,
        throughput_direct,
        width=bar_width,
        color="#ff7f0e",
        label="Direct (Analytical Hex RayCasting)",
    )
    ax_tp.set_title("Пропускная способность транспорта фотонов")
    ax_tp.set_ylabel("Скорость счета, тыс. историй / с")
    ax_tp.set_xticks(indices)
    ax_tp.set_xticklabels(scenario_labels, rotation=25, ha="right")
    ax_tp.grid(True, axis="y", alpha=0.3)
    ax_tp.legend()

    ax_sp = axes[1]
    bars = ax_sp.bar(indices, speedup_factors, width=0.5, color="#2ca02c")
    ax_sp.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2, label="Паритет (1.0x)")
    for bar_obj, factor_val in zip(bars, speedup_factors):
        height = bar_obj.get_height()
        ax_sp.annotate(
            f"{factor_val:.2f}x",
            xy=(bar_obj.get_x() + bar_obj.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax_sp.set_title("Коэффициент ускорения Direct относительно Parametric")
    ax_sp.set_ylabel("Ускорение (Speedup), раз")
    ax_sp.set_xticks(indices)
    ax_sp.set_xticklabels(scenario_labels, rotation=25, ha="right")
    ax_sp.set_ylim(0.0, max(max(speedup_factors, default=1.0) * 1.22, 1.3))
    ax_sp.grid(True, axis="y", alpha=0.3)
    ax_sp.legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


__all__ = [
    "plot_optical_depth_comparison",
    "plot_psf_comparison_panel",
    "plot_anger_distance_curves",
    "plot_septal_star_analysis",
    "plot_speed_benchmark",
]

