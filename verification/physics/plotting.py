"""Генерация диагностических графиков для пакета верификации физического ядра (Headless Agg)."""
from __future__ import annotations

import io
import math
from pathlib import Path
import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(figure: matplotlib.figure.Figure, output_path: Path, dpi: int = 160) -> None:
    """Сохраняет фигуру Matplotlib в PNG через буфер памяти для исключения блокировок дескриптора."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    plt.close(figure)
    if output_path.exists():
        output_path.unlink()
    output_path.write_bytes(buffer.getvalue())


def plot_exp01_cross_sections_and_target_sampling(
    energies: NDArray[np.float64],
    lead_pe: NDArray[np.float64],
    lead_comp: NDArray[np.float64],
    lead_coh: NDArray[np.float64],
    lead_tot: NDArray[np.float64],
    lead_edges: list[tuple[str, float]],
    grid_energies: NDArray[np.float64],
    rel_diff_binning: NDArray[np.float64],
    target_labels: list[str],
    empirical_probs: list[float],
    theoretical_probs: list[float],
    output_path: Path,
) -> None:
    """Строит 3-панельный отчет верификации сечений, O(1) биннинга и выборки Z (Эксперимент 1).

    Args:
        energies: Сетка энергий для кривых ослабления свинца [МэВ].
        lead_pe: Линейный коэффициент фотоэффекта свинца [1/мм].
        lead_comp: Линейный коэффициент некогерентного рассеяния свинца [1/мм].
        lead_coh: Линейный коэффициент когерентного рассеяния свинца [1/мм].
        lead_tot: Полный линейный коэффициент ослабления свинца [1/мм].
        lead_edges: Список пар (название края, энергия [МэВ]) для Pb.
        grid_energies: Плотная сетка энергий сверки O(1) биннинга [МэВ].
        rel_diff_binning: Относительная невязка между O(1) биннингом и searchsorted.
        target_labels: Подписи сценариев выборки Z (материал / процесс / элемент).
        empirical_probs: Эмпирические доли выбора элемента из Монте-Карло.
        theoretical_probs: Теоретические доли выбора элемента по парциальным сечениям.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    display_energies = energies * 1000.0
    ax_cs = axes[0]
    ax_cs.loglog(display_energies, np.maximum(lead_tot, 1e-8), color="#111111", linewidth=2.2, label="$\\mu_{tot}$ (Pb)")
    ax_cs.loglog(display_energies, np.maximum(lead_pe, 1e-8), color="#d62728", linestyle="--", linewidth=1.7, label="$\\mu_{pe}$")
    ax_cs.loglog(display_energies, np.maximum(lead_comp, 1e-8), color="#1f77b4", linestyle="-.", linewidth=1.7, label="$\\mu_{comp}$")
    ax_cs.loglog(display_energies, np.maximum(lead_coh, 1e-8), color="#2ca02c", linestyle=":", linewidth=1.7, label="$\\mu_{coh}$")
    for edge_label, edge_energy in lead_edges:
        ax_cs.axvline(edge_energy * 1000.0, color="#9467bd", linestyle=":", alpha=0.7, linewidth=1.2)
        ax_cs.text(
            edge_energy * 1000.0 * 1.03,
            float(np.max(lead_tot)) * 0.35,
            edge_label,
            rotation=90,
            fontsize=8.0,
            color="#9467bd",
        )
    ax_cs.set_title("Линейные коэффициенты ослабления Pb и $K/L/M$-края")
    ax_cs.set_xlabel("Энергия фотона $E$, кэВ")
    ax_cs.set_ylabel("Линейный коэффициент $\\mu$, мм$^{-1}$")
    ax_cs.grid(True, which="both", alpha=0.3)
    ax_cs.legend(fontsize=8.5)

    ax_err = axes[1]
    ax_err.loglog(
        grid_energies * 1000.0,
        np.maximum(rel_diff_binning, 1e-17),
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.85,
    )
    ax_err.axhline(1e-14, color="#d62728", linestyle="--", linewidth=1.4, label="Порог $10^{-14}$")
    ax_err.set_title("Паритет $\\mathcal{O}(1)$ лог-индексации и `searchsorted`")
    ax_err.set_xlabel("Энергия фотона $E$, кэВ")
    ax_err.set_ylabel("Относительная невязка $|\\mu_{\\mathcal{O}(1)} - \\mu_{ref}| / \\mu_{ref}$")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend()

    ax_z = axes[2]
    indices = np.arange(len(target_labels), dtype=np.float64)
    bar_width = 0.36
    ax_z.bar(indices - 0.5 * bar_width, theoretical_probs, width=bar_width, color="#2ca02c", label="Теория $w_i \\sigma_{i,p} / \\Sigma_p$")
    ax_z.bar(indices + 0.5 * bar_width, empirical_probs, width=bar_width, color="#ff7f0e", label="Монте-Карло `_sample_Z_scalar`")
    ax_z.set_title("Стохастический выбор элемента-мишени $P(Z \\mid \\text{proc}, E)$")
    ax_z.set_ylabel("Вероятность выбора элемента")
    ax_z.set_xticks(indices)
    ax_z.set_xticklabels(target_labels, rotation=28, ha="right", fontsize=8.5)
    ax_z.set_ylim(0.0, 1.12)
    ax_z.grid(True, axis="y", alpha=0.3)
    ax_z.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)


def plot_exp02_scattering_indicatrices_and_kinematics(
    cos_grid: NDArray[np.float64],
    compton_curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    kinematics_cos: NDArray[np.float64],
    kinematics_edep_mc: NDArray[np.float64],
    kinematics_edep_theory: NDArray[np.float64],
    compton_edge_energy: float,
    coherent_curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    thomson_curve: NDArray[np.float64],
    phi_centers: NDArray[np.float64],
    phi_density_mc: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Строит 4-панельный отчет верификации индикатрис Комптона, Рэлея и кинематики (Эксперимент 2).

    Args:
        cos_grid: Сетка значений cos(theta) на [-1, 1].
        compton_curves: Список кортежей (подпись, эмпирическая плотность, теоретическая плотность) для Комптона.
        kinematics_cos: Значения cos(theta) для проверки кинематики отдачи.
        kinematics_edep_mc: Переданная энергия электрону отдачи из ядра [МэВ].
        kinematics_edep_theory: Теоретическая энергия электрона отдачи [МэВ].
        compton_edge_energy: Энергия комптоновского края при theta = pi [МэВ].
        coherent_curves: Список кортежей (подпись, эмпирическая плотность, теоретическая плотность) для Рэлея.
        thomson_curve: Изотропная кривая Томсона 3/8 * (1 + cos^2(theta)).
        phi_centers: Центры бинов азимутального угла phi [рад].
        phi_density_mc: Эмпирическая плотность распределения угла phi.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 10.5), constrained_layout=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax_comp = axes[0, 0]
    for idx, (label_text, density_mc, density_ref) in enumerate(compton_curves):
        color_hex = colors[idx % len(colors)]
        ax_comp.plot(cos_grid, density_ref, color=color_hex, linewidth=2.0, label=f"{label_text} (Теория)")
        ax_comp.plot(
            cos_grid[::2],
            density_mc[::2],
            color=color_hex,
            linestyle="none",
            marker="o",
            markersize=3.8,
            alpha=0.8,
            label=f"{label_text} (MC)",
        )
    ax_comp.set_title("Индикатриса некогерентного рассеяния Клейна–Нишины $\\times S(x, Z)$")
    ax_comp.set_xlabel("$\\cos\\theta$")
    ax_comp.set_ylabel("Плотность вероятности $p(\\cos\\theta)$")
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend(fontsize=8.0, ncol=2)

    ax_kin = axes[0, 1]
    ax_kin.plot(
        kinematics_cos,
        kinematics_edep_theory * 1000.0,
        color="#1f77b4",
        linewidth=2.2,
        label="Точная кинематика Комптона",
    )
    ax_kin.plot(
        kinematics_cos[::4],
        kinematics_edep_mc[::4] * 1000.0,
        color="#ff7f0e",
        linestyle="none",
        marker="s",
        markersize=4.0,
        label="Ядро `_calculate_compton_energy_deposit_scalar`",
    )
    ax_kin.axhline(
        compton_edge_energy * 1000.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Край Комптона ({compton_edge_energy * 1000.0:.2f} кэВ)",
    )
    ax_kin.set_title("Кинематика энергии электрона отдачи $E_{dep}(\\cos\\theta)$")
    ax_kin.set_xlabel("$\\cos\\theta$")
    ax_kin.set_ylabel("Энергия отдачи $E_{dep}$, кэВ")
    ax_kin.grid(True, alpha=0.3)
    ax_kin.legend(fontsize=8.5)

    ax_coh = axes[1, 0]
    ax_coh.plot(
        cos_grid,
        thomson_curve,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.8,
        label="Чистый Томсон $\\propto (1 + \\cos^2\\theta)$",
    )
    for idx, (label_text, density_mc, density_ref) in enumerate(coherent_curves):
        color_hex = colors[idx % len(colors)]
        ax_coh.semilogy(cos_grid, np.maximum(density_ref, 1e-4), color=color_hex, linewidth=2.0, label=f"{label_text} (Теория)")
        ax_coh.semilogy(
            cos_grid[::2],
            np.maximum(density_mc[::2], 1e-4),
            color=color_hex,
            linestyle="none",
            marker="^",
            markersize=3.8,
            alpha=0.8,
            label=f"{label_text} (MC)",
        )
    ax_coh.set_title("Индикатриса Рэлея: Томсон $\\times F^2(x, Z)$")
    ax_coh.set_xlabel("$\\cos\\theta$")
    ax_coh.set_ylabel("Плотность $p(\\cos\\theta)$ (лог. шкала)")
    ax_coh.grid(True, which="both", alpha=0.3)
    ax_coh.legend(fontsize=8.0, ncol=2)

    ax_phi = axes[1, 1]
    uniform_level = 1.0 / (2.0 * math.pi)
    ax_phi.plot(phi_centers, phi_density_mc, color="#1f77b4", marker="o", markersize=3.5, linewidth=1.6, label="MC `_generate_scattering_phi_scalar`")
    ax_phi.axhline(uniform_level, color="#2ca02c", linestyle="--", linewidth=2.0, label="Равномерная плотность $1/(2\\pi)$")
    ax_phi.set_title("Равномерность азимутального угла $\\varphi \\in [-\\pi, \\pi)$")
    ax_phi.set_xlabel("Азимутальный угол $\\varphi$, рад")
    ax_phi.set_ylabel("Плотность вероятности $p(\\varphi)$")
    ax_phi.set_ylim(uniform_level * 0.85, uniform_level * 1.15)
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)


def plot_exp03_woodcock_and_delta_tracking(
    depths_attenuation: NDArray[np.float64],
    transmission_mc: NDArray[np.float64],
    transmission_theory: NDArray[np.float64],
    depth_centers_delta: NDArray[np.float64],
    delta_profiles: list[tuple[str, NDArray[np.float64]]],
    delta_theory_pdf: NDArray[np.float64],
    layer_labels: list[str],
    edep_direct_slabs: list[float],
    edep_woodcock_voxel: list[float],
    edep_woodcock_parametric: list[float],
    output_path: Path,
) -> None:
    """Строит 3-панельный отчет верификации дельта-трекинга и трекинга Вудкока (Эксперимент 3).

    Args:
        depths_attenuation: Глубины для проверки закона Бугера–Ламберта–Бера [мм].
        transmission_mc: Доля нерассеянных фотонов в Монте-Карло.
        transmission_theory: Аналитическое пропускание exp(-mu * z).
        depth_centers_delta: Центры глубинных бинов в ступенчатом барьере [мм].
        delta_profiles: Список пар (подпись коэффициента мажоранты kappa, плотность коллизий MC).
        delta_theory_pdf: Аналитическая кусочно-экспоненциальная плотность первой коллизии.
        layer_labels: Подписи 5 слоев гетерогенного фантома.
        edep_direct_slabs: Поглощенная энергия по слоям для Direct Slabs [МэВ].
        edep_woodcock_voxel: Поглощенная энергия по слоям для WoodcockVoxelVolume [МэВ].
        edep_woodcock_parametric: Поглощенная энергия по слоям для WoodcockParametricVolume [МэВ].
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    ax_beer = axes[0]
    ax_beer.semilogy(depths_attenuation, np.maximum(transmission_theory, 1e-8), color="#111111", linewidth=2.2, label="Теория $\\exp(-\\mu z)$")
    ax_beer.semilogy(
        depths_attenuation,
        np.maximum(transmission_mc, 1e-8),
        color="#ff7f0e",
        linestyle="none",
        marker="o",
        markersize=5.0,
        label="MC `_generate_free_path`",
    )
    ax_beer.set_title("Закон Бугера–Ламберта–Бера в гомогенной среде")
    ax_beer.set_xlabel("Глубина проникновения $z$, мм")
    ax_beer.set_ylabel("Коэффициент пропускания $I(z)/I_0$")
    ax_beer.grid(True, which="both", alpha=0.3)
    ax_beer.legend()

    ax_delta = axes[1]
    ax_delta.plot(depth_centers_delta, delta_theory_pdf, color="#111111", linewidth=2.4, label="Точная аналитика $f(z)$")
    markers = ["o", "s", "^", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (kappa_label, pdf_mc) in enumerate(delta_profiles):
        ax_delta.plot(
            depth_centers_delta[::2],
            pdf_mc[::2],
            color=colors[idx % len(colors)],
            linestyle="none",
            marker=markers[idx % len(markers)],
            markersize=4.0,
            alpha=0.82,
            label=kappa_label,
        )
    ax_delta.set_title("Инвариантность дельта-трекинга к мажоранте $\\kappa = \\mu_{maj}/\\mu_{real}$")
    ax_delta.set_xlabel("Координата первой реальной коллизии $z$, мм")
    ax_delta.set_ylabel("Плотность вероятности $f(z)$, мм$^{-1}$")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.legend(fontsize=8.5)

    ax_triple = axes[2]
    indices = np.arange(len(layer_labels), dtype=np.float64)
    bar_width = 0.26
    ax_triple.bar(indices - bar_width, edep_direct_slabs, width=bar_width, color="#1f77b4", label="Direct Slabs (`Volume`)")
    ax_triple.bar(indices, edep_woodcock_voxel, width=bar_width, color="#ff7f0e", label="Woodcock Voxel")
    ax_triple.bar(indices + bar_width, edep_woodcock_parametric, width=bar_width, color="#2ca02c", label="Woodcock Parametric (`CFunc`)")
    ax_triple.set_title("Тройная эквивалентность в 5-слойном фантоме")
    ax_triple.set_ylabel("Суммарное энерговыделение в слое, МэВ")
    ax_triple.set_xticks(indices)
    ax_triple.set_xticklabels(layer_labels, rotation=20, ha="right")
    ax_triple.grid(True, axis="y", alpha=0.3)
    ax_triple.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)


def plot_exp04_energy_partition_and_atomic_relaxation(
    energy_bins_centers: NDArray[np.float64],
    hist_local_absorption: NDArray[np.float64],
    hist_collision_kerma: NDArray[np.float64],
    hist_coupled_dose: NDArray[np.float64],
    subshell_energies: NDArray[np.float64],
    subshell_curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    fluorescence_labels: list[str],
    fluorescence_mc: list[float],
    fluorescence_eadl: list[float],
    output_path: Path,
) -> None:
    """Строит 3-панельный отчет распределения энергии и атомной релаксации EADL (Эксперимент 4).

    Args:
        energy_bins_centers: Центры бинов локального энерговыделения E_dep [МэВ].
        hist_local_absorption: Распределение E_dep в режиме Local Absorption.
        hist_collision_kerma: Распределение E_dep в режиме Collision Kerma.
        hist_coupled_dose: Распределение E_dep в режиме Coupled Dose.
        subshell_energies: Энергии фотонов вокруг краев поглощения Pb [МэВ].
        subshell_curves: Список кортежей (имя подоболочки, доля MC, доля EADL).
        fluorescence_labels: Подписи выходов флуоресценции и линий релаксации (I, Pb).
        fluorescence_mc: Эмпирические выходы из Монте-Карло.
        fluorescence_eadl: Табличные значения ENDF/B-VIII.0 EADL.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    ax_part = axes[0]
    display_e = energy_bins_centers * 1000.0
    ax_part.semilogy(
        display_e,
        np.maximum(hist_local_absorption, 0.5),
        color="#1f77b4",
        linewidth=1.35,
        alpha=0.92,
        drawstyle="steps-mid",
        label="Local Absorption ($T_{cut}=\\infty$)",
    )
    ax_part.semilogy(
        display_e,
        np.maximum(hist_collision_kerma, 0.5),
        color="#ff7f0e",
        linewidth=1.2,
        alpha=0.88,
        drawstyle="steps-mid",
        label="Collision Kerma ($T_{e}=\\infty, T_{\\gamma}=0$)",
    )
    ax_part.semilogy(
        display_e,
        np.maximum(hist_coupled_dose, 0.5),
        color="#2ca02c",
        linewidth=1.15,
        alpha=0.88,
        drawstyle="steps-mid",
        label="Coupled Dose ($T_{e}=0, T_{\\gamma}=0$)",
    )
    spectral_landmarks: list[tuple[float, str, str, float]] = [
        (13.04, "Pb $L$-остаток", "#2ca02c", 1.5e5),
        (49.82, "Край Комптона", "#444444", 7.5e3),
        (54.70, "Pb $K_\\beta$ вылет", "#d62728", 3.2e4),
        (66.60, "Pb $K_{\\alpha 1,2}$ вылет", "#d62728", 1.4e5),
        (108.20, "I $K_\\beta$ вылет", "#9467bd", 3.0e4),
        (111.90, "I $K_\\alpha$ вылет", "#9467bd", 1.4e5),
        (127.90, "Pb $L$ вылет", "#8c564b", 1.0e4),
        (140.50, "Пик $E_0$", "#1f77b4", 7.0e5),
    ]
    for landmark_e, landmark_label, landmark_color, label_y in spectral_landmarks:
        ax_part.axvline(landmark_e, color=landmark_color, linestyle=":", linewidth=0.95, alpha=0.55)
        ax_part.text(
            landmark_e - 1.2,
            label_y,
            landmark_label,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=7.3,
            color=landmark_color,
        )
    ax_part.set_ylim(0.35, 3.5e7)
    ax_part.set_title("Распределение локального вклада $E_{dep}$ (ансамбль NaI + Pb)")
    ax_part.set_xlabel("Локальное энерговыделение $E_{dep}$, кэВ")
    ax_part.set_ylabel("Отсчеты на бин (лог. шкала)")
    ax_part.grid(True, which="both", alpha=0.3)
    ax_part.legend(fontsize=7.8, loc="upper left", framealpha=0.92)

    ax_sub = axes[1]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for idx, (shell_name, prob_mc, prob_ref) in enumerate(subshell_curves):
        color_hex = colors[idx % len(colors)]
        ax_sub.plot(subshell_energies * 1000.0, prob_ref, color=color_hex, linewidth=1.9, label=f"{shell_name} (EADL)")
        ax_sub.plot(
            subshell_energies * 1000.0,
            prob_mc,
            color=color_hex,
            linestyle="none",
            marker="o",
            markersize=4.0,
            label=f"{shell_name} (MC)",
        )
    ax_sub.set_title("Ступенчатое открытие подоболочек Pb ($Z=82$) на краях $E_b$")
    ax_sub.set_xlabel("Энергия фотона $E_\\gamma$, кэВ")
    ax_sub.set_ylabel("Вероятность ионизации подоболочки")
    ax_sub.set_ylim(-0.03, 1.05)
    ax_sub.grid(True, alpha=0.3)
    ax_sub.legend(fontsize=8.0, ncol=2)

    ax_eadl = axes[2]
    indices = np.arange(len(fluorescence_labels), dtype=np.float64)
    bar_width = 0.36
    ax_eadl.bar(indices - 0.5 * bar_width, fluorescence_eadl, width=bar_width, color="#1f77b4", label="Таблицы ENDF/B-VIII.0 EADL")
    ax_eadl.bar(indices + 0.5 * bar_width, fluorescence_mc, width=bar_width, color="#ff7f0e", label="MC `_sample_vacancy_relaxation`")
    ax_eadl.set_title("Выходы флуоресценции $\\omega_K$ и Оже-релаксации")
    ax_eadl.set_ylabel("Выход на одну вакансию")
    ax_eadl.set_xticks(indices)
    ax_eadl.set_xticklabels(fluorescence_labels, rotation=25, ha="right", fontsize=8.5)
    ax_eadl.set_ylim(0.0, 1.1)
    ax_eadl.grid(True, axis="y", alpha=0.3)
    ax_eadl.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)


def plot_exp05_secondary_source_and_escape_peaks(
    delay_centers: NDArray[np.float64],
    delay_density_mc: NDArray[np.float64],
    delay_density_theory: NDArray[np.float64],
    nai_energy_centers: NDArray[np.float64],
    nai_spectrum_counts: NDArray[np.float64],
    escape_peak_energies: list[tuple[str, float]],
    lead_xray_centers: NDArray[np.float64],
    lead_xray_counts: NDArray[np.float64],
    lead_xray_lines: list[tuple[str, float]],
    output_path: Path,
) -> None:
    """Строит 3-панельный отчет верификации SecondarySource, пика вылета иода и рентгена Pb (Эксперимент 5).

    Args:
        delay_centers: Центры бинов времени задержки распада изомера [нс].
        delay_density_mc: Эмпирическая плотность времени задержки из Монте-Карло [1/нс].
        delay_density_theory: Теоретическая экспонента lambda * exp(-lambda * dt) [1/нс].
        nai_energy_centers: Центры энергетических бинов спектра кристалла NaI(Tl) [МэВ].
        nai_spectrum_counts: Спектр суммарного энерговыделения на историю в кристалле NaI(Tl).
        escape_peak_energies: Список пар (название пика, энергия [МэВ]) для NaI(Tl).
        lead_xray_centers: Центры энергетических бинов спектра вылетевших фотонов из Pb [МэВ].
        lead_xray_counts: Спектр характеристического рентгеновского излучения свинца.
        lead_xray_lines: Список пар (название линии, энергия [МэВ]) для Pb K-shell.
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    ax_time = axes[0]
    ax_time.semilogy(delay_centers, np.maximum(delay_density_theory, 1e-10), color="#111111", linewidth=2.2, label="Теория $\\lambda \\exp(-\\lambda \\Delta t)$")
    ax_time.semilogy(
        delay_centers,
        np.maximum(delay_density_mc, 1e-10),
        color="#ff7f0e",
        linestyle="none",
        marker="o",
        markersize=4.5,
        label="MC `_sample_secondary_emission_time`",
    )
    ax_time.set_title("Кинетика распада метастабильного уровня ($T_{1/2} > 0$)")
    ax_time.set_xlabel("Задержка эмиссии $\\Delta t$, нс")
    ax_time.set_ylabel("Плотность вероятности $p(\\Delta t)$, нс$^{-1}$")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend(fontsize=8.5)

    ax_nai = axes[1]
    ax_nai.semilogy(
        nai_energy_centers * 1000.0,
        np.maximum(nai_spectrum_counts, 0.5),
        color="#1f77b4",
        linewidth=1.8,
        label="Спектр $E_{dep}$ в NaI(Tl)",
    )
    for peak_label, peak_energy in escape_peak_energies:
        ax_nai.axvline(peak_energy * 1000.0, color="#d62728", linestyle="--", linewidth=1.3, alpha=0.8)
        ax_nai.text(
            peak_energy * 1000.0 - 2.5,
            float(np.max(nai_spectrum_counts)) * 0.25,
            peak_label,
            rotation=90,
            fontsize=8.0,
            color="#d62728",
        )
    ax_nai.set_title("Пик вылета иода (Iodine K-escape) в кристалле NaI(Tl)")
    ax_nai.set_xlabel("Суммарное энерговыделение в кристалле, кэВ")
    ax_nai.set_ylabel("Отсчеты на бин (лог. шкала)")
    ax_nai.grid(True, which="both", alpha=0.3)
    ax_nai.legend(fontsize=8.5)

    ax_pb = axes[2]
    ax_pb.plot(
        lead_xray_centers * 1000.0,
        lead_xray_counts,
        color="#2ca02c",
        linewidth=1.8,
        label="Вылетевшие вторичные фотоны Pb",
    )
    for line_label, line_energy in lead_xray_lines:
        ax_pb.axvline(line_energy * 1000.0, color="#9467bd", linestyle=":", linewidth=1.4)
        ax_pb.text(
            line_energy * 1000.0 + 0.4,
            max(float(np.max(lead_xray_counts)) * 0.55, 1.0),
            line_label,
            rotation=90,
            fontsize=8.0,
            color="#9467bd",
        )
    ax_pb.set_title("Характеристический рентген свинца ($K_{\\alpha}, K_{\\beta}$)")
    ax_pb.set_xlabel("Энергия вторичного фотона, кэВ")
    ax_pb.set_ylabel("Отсчеты на бин")
    ax_pb.grid(True, alpha=0.3)
    ax_pb.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)


def plot_exp06_primary_sources_and_poisson_stream(
    cont_energy_centers: NDArray[np.float64],
    cont_density_mc: NDArray[np.float64],
    cont_density_theory: NDArray[np.float64],
    time_centers: NDArray[np.float64],
    rate_mc: NDArray[np.float64],
    rate_theory: NDArray[np.float64],
    batch_labels: list[str],
    max_position_diffs: list[float],
    max_energy_diffs: list[float],
    output_path: Path,
) -> None:
    """Строит 3-панельный отчет верификации первичных источников и потока Льюиса–Шедлера (Эксперимент 6).

    Args:
        cont_energy_centers: Центры энергетических бинов непрерывного спектра [МэВ].
        cont_density_mc: Эмпирическая плотность энергии из `_sample_channel_energy` [1/МэВ].
        cont_density_theory: Аналитическая кусочно-линейная плотность трапеций [1/МэВ].
        time_centers: Центры временных бинов нестационарного потока [нс].
        rate_mc: Эмпирическая скорость счета dN/dt из `inject_poisson_sources_kernel` [1/нс].
        rate_theory: Теоретическая интенсивность распада A(t) [1/нс].
        batch_labels: Подписи конфигураций сверки batch_capacity и Stateful Resume.
        max_position_diffs: Максимальные абсолютные невязки координат фотонов [мм].
        max_energy_diffs: Максимальные абсолютные невязки энергий фотонов [МэВ].
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    ax_spec = axes[0]
    ax_spec.plot(
        cont_energy_centers * 1000.0,
        cont_density_theory * 1e-3,
        color="#111111",
        linewidth=2.2,
        label="Кусочно-линейная плотность (Теория)",
    )
    ax_spec.plot(
        cont_energy_centers * 1000.0,
        cont_density_mc * 1e-3,
        color="#ff7f0e",
        linestyle="none",
        marker="o",
        markersize=4.2,
        label="MC `_sample_channel_energy`",
    )
    ax_spec.set_title("Обращение трапеций `ContinuousSpectrum` + таблицы Воза")
    ax_spec.set_xlabel("Энергия эмиссии $E$, кэВ")
    ax_spec.set_ylabel("Плотность вероятности $p(E)$, кэВ$^{-1}$")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=8.5)

    ax_pois = axes[1]
    ax_pois.plot(time_centers, rate_theory, color="#1f77b4", linewidth=2.2, label="Аналитическая интенсивность $\\Lambda(t)$")
    ax_pois.plot(
        time_centers,
        rate_mc,
        color="#d62728",
        linestyle="none",
        marker="s",
        markersize=4.0,
        label="Метод Льюиса–Шедлера (MC)",
    )
    ax_pois.set_title("Нестационарный пуассоновский поток $dN/dt$")
    ax_pois.set_xlabel("Время симуляции $t$, нс")
    ax_pois.set_ylabel("Скорость эмиссии $dN/dt$, нс$^{-1}$")
    ax_pois.grid(True, alpha=0.3)
    ax_pois.legend(fontsize=8.5)

    ax_inv = axes[2]
    indices = np.arange(len(batch_labels), dtype=np.float64)
    bar_width = 0.35
    pos_vals = [max(val, 1e-16) for val in max_position_diffs]
    ene_vals = [max(val, 1e-16) for val in max_energy_diffs]
    ax_inv.bar(indices - 0.5 * bar_width, pos_vals, width=bar_width, color="#1f77b4", label="$\\max |\\Delta \\mathbf{r}|$, мм")
    ax_inv.bar(indices + 0.5 * bar_width, ene_vals, width=bar_width, color="#2ca02c", label="$\\max |\\Delta E|$, МэВ")
    ax_inv.set_yscale("log")
    ax_inv.set_ylim(1e-17, 1e-10)
    ax_inv.set_title("Побитовая инвариантность `batch_capacity` и `Stateful Resume`")
    ax_inv.set_ylabel("Максимальная невязка с эталоном (лог. шкала)")
    ax_inv.set_xticks(indices)
    ax_inv.set_xticklabels(batch_labels, rotation=20, ha="right")
    ax_inv.grid(True, which="both", axis="y", alpha=0.3)
    ax_inv.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)


def plot_exp07_macroscopic_transport_and_dosimetry(
    mfp_grid: NDArray[np.float64],
    buildup_mc: NDArray[np.float64],
    buildup_berger: NDArray[np.float64],
    albedo_materials: list[str],
    number_albedo_mc: list[float],
    energy_albedo_mc: list[float],
    backscatter_energy_centers: NDArray[np.float64],
    backscatter_spectrum: NDArray[np.float64],
    backscatter_peak_energy: float,
    dose_depth_centers: NDArray[np.float64],
    dose_profile_handler: NDArray[np.float64],
    dose_profile_interactions: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Строит 3-панельный отчет макроскопического транспорта, альбедо и дозиметрии (Эксперимент 7).

    Args:
        mfp_grid: Оптическая толщина mu * r в длинах свободного пробега [MFP].
        buildup_mc: Измеренный фактор накопления поглощенной энергии B(mu * r) в воде.
        buildup_berger: Эталонный фактор накопления по стандарту ANSI/ANS-6.4.3-1991.
        albedo_materials: Названия материалов отражателей ('Water', 'Pyrex', 'Lead').
        number_albedo_mc: Числовое альбедо обратного рассеяния A_N.
        energy_albedo_mc: Энергетическое альбедо обратного рассеяния A_E.
        backscatter_energy_centers: Центры бинов спектра обратно рассеянных фотонов [МэВ].
        backscatter_spectrum: Спектр обратно рассеянных фотонов (пик 180 град).
        backscatter_peak_energy: Теоретическая энергия пика обратного рассеяния [МэВ].
        dose_depth_centers: Центры вокселей вдоль оси Z в DoseGridNode [мм].
        dose_profile_handler: Профиль энергии по слоям Z из DoseMapHandler [МэВ].
        dose_profile_interactions: Профиль энергии по слоям Z напрямую из InteractionBuffer [МэВ].
        output_path: Путь сохранения PNG-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    ax_buildup = axes[0]
    ax_buildup.plot(mfp_grid, buildup_berger, color="#111111", linewidth=2.2, label="Стандарт ANSI/ANS-6.4.3 $B_{en}(\\mu r)$")
    ax_buildup.plot(
        mfp_grid,
        buildup_mc,
        color="#1f77b4",
        marker="o",
        linestyle="--",
        linewidth=1.8,
        label="MC (Водный фантом)",
    )
    ax_buildup.set_title("Фактор накопления поглощенной энергии $B(\\mu r)$ в воде")
    ax_buildup.set_xlabel("Оптическая глубина $\\mu r$, MFP")
    ax_buildup.set_ylabel("Фактор накопления $B(\\mu r)$")
    ax_buildup.grid(True, alpha=0.3)
    ax_buildup.legend(fontsize=8.5)

    ax_alb = axes[1]
    ax_alb.plot(
        backscatter_energy_centers * 1000.0,
        backscatter_spectrum,
        color="#ff7f0e",
        linewidth=2.0,
        label="Спектр обратного рассеяния (Water)",
    )
    ax_alb.axvline(
        backscatter_peak_energy * 1000.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.6,
        label=f"Пик $180^\\circ$ ({backscatter_peak_energy * 1000.0:.1f} кэВ)",
    )
    albedo_summary = ", ".join(
        f"{mat}: $A_N$={an:.3f}" for mat, an in zip(albedo_materials, number_albedo_mc)
    )
    ax_alb.set_title(f"Обратное рассеяние ({albedo_summary})")
    ax_alb.set_xlabel("Энергия отраженного фотона, кэВ")
    ax_alb.set_ylabel("Отсчеты на бин")
    ax_alb.grid(True, alpha=0.3)
    ax_alb.legend(fontsize=8.5)

    ax_dose = axes[2]
    ax_dose.plot(
        dose_depth_centers,
        dose_profile_interactions,
        color="#111111",
        linewidth=2.2,
        label="Прямая агрегация `InteractionBuffer`",
    )
    ax_dose.plot(
        dose_depth_centers,
        dose_profile_handler,
        color="#2ca02c",
        linestyle="none",
        marker="s",
        markersize=4.5,
        label="3D сетка `DoseMapHandler`",
    )
    ax_dose.set_title("100% паритет 3D дозовой карты `DoseGridNode` и взаимодействий")
    ax_dose.set_xlabel("Глубина в фантоме $Z$, мм")
    ax_dose.set_ylabel("Поглощенная энергия в слое, МэВ")
    ax_dose.grid(True, alpha=0.3)
    ax_dose.legend(fontsize=8.5)

    _save_figure(figure, output_path, dpi=160)
