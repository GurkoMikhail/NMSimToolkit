"""Эксперимент 4: Стресс-тест септального пробоя («эффект 6-лучевой звезды») при высоких энергиях.

Исследует прохождение гамма-квантов высокой энергии I-131 (364.5 кэВ) через:
1. Неадекватный по толщине перегородки низкоэнергетический коллиматор GE LEHR (septa = 0.20 мм),
   где доминирует многократный пролет сквозь десятки соседних перегородок и формируется
   6-лучевая гексагональная звезда (Star Artifact);
2. Штатный высокоэнергетический коллиматор GE HEGP (septa = 1.80 мм), подавляющий септальный пробой.

Сверяет азимутальное распределение интенсивности I(phi) в кольце хвостов PSF, амплитуду
6-й гармоники Фурье и доли компонент SEPTAL_PENETRATION и COLLIMATOR_SCATTER между
ParametricParallelCollimator и DirectParallelCollimator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np

from verification.common.history_classifier import PhotonComponent
from verification.common.metrics import (
    CollimatorSensitivityBreakdown,
    compute_azimuthal_star_metrics,
    compute_sensitivity_breakdown,
    filter_energy_window,
)
from verification.common.plotting import plot_septal_star_analysis
from verification.common.presets import GE_COLLIMATOR_PRESETS, RADIONUCLIDE_PRESETS
from verification.common.scene_builders import run_collimator_simulation


@dataclass(frozen=True)
class SeptalStarCaseResult:
    """Результаты стресс-теста септального пробоя для одного коллиматора GE с I-131.

    Attributes:
        collimator_code: Код коллиматора ('LEHR' или 'HEGP').
        nuclide_name: Название радионуклида ('I-131').
        sensitivity_parametric: Покомпонентная чувствительность Parametric.
        sensitivity_direct: Покомпонентная чувствительность Direct.
        anisotropy_contrast_parametric: Контраст 6-лучевой модуляции в кольце хвостов (Parametric).
        anisotropy_contrast_direct: Контраст 6-лучевой модуляции в кольце хвостов (Direct).
        harmonic6_ratio_parametric: Относительная амплитуда 6-й гармоники Фурье (Parametric).
        harmonic6_ratio_direct: Относительная амплитуда 6-й гармоники Фурье (Direct).
        azimuthal_profile_correlation: Коэффициент корреляции Пирсона профилей I(phi) двух моделей.
        passed: Признак согласия двух моделей в режиме экстремального септального пробоя.
    """

    collimator_code: str
    nuclide_name: str
    sensitivity_parametric: CollimatorSensitivityBreakdown
    sensitivity_direct: CollimatorSensitivityBreakdown
    anisotropy_contrast_parametric: float
    anisotropy_contrast_direct: float
    harmonic6_ratio_parametric: float
    harmonic6_ratio_direct: float
    azimuthal_profile_correlation: float
    passed: bool


def run_single_star_stress_case(
    collimator_code: str,
    output_dir: Path,
    num_particles: int = 2_500_000,
    source_distance: float = 100.0,
    max_polar_angle: float = 0.35,
    base_seed: int = 404000,
) -> SeptalStarCaseResult:
    """Запускает парный стресс-тест септального пробоя I-131 для выбранного коллиматора.

    Args:
        collimator_code: Код коллиматора ('LEHR' или 'HEGP').
        output_dir: Папка для сохранения графиков.
        num_particles: Число фотонов в конусе.
        source_distance: Расстояние от источника до входного торца [мм].
        max_polar_angle: Полуугол конуса генерации [рад].
        base_seed: Базовый сид ГСЧ.

    Returns:
        Структура SeptalStarCaseResult.
    """
    preset = GE_COLLIMATOR_PRESETS[collimator_code]
    nuclide = RADIONUCLIDE_PRESETS["I131"]

    # В стресс-тесте звезды источник ставится точно над центром центрального отверстия
    # (cell_averaging=False), чтобы лучи гексагональной решетки не размывались фазовым джиттером
    out_param = run_collimator_simulation(
        preset=preset,
        implementation_kind="parametric",
        source_distance=source_distance,
        energy=nuclide.primary_energy,
        particles_number=num_particles,
        max_polar_angle=max_polar_angle,
        cell_averaging=False,
        seed=base_seed,
    )
    out_direct = run_collimator_simulation(
        preset=preset,
        implementation_kind="direct",
        source_distance=source_distance,
        energy=nuclide.primary_energy,
        particles_number=num_particles,
        max_polar_angle=max_polar_angle,
        cell_averaging=False,
        seed=base_seed + 500,
    )

    phot_p = out_param.photons
    phot_d = out_direct.photons

    mask_p = filter_energy_window(phot_p, float(nuclide.primary_energy), 0.10)
    mask_d = filter_energy_window(phot_d, float(nuclide.primary_energy), 0.10)

    sens_p = compute_sensitivity_breakdown(
        phot_p, out_param.particles_emitted, float(phot_p.solid_angle_fraction), mask_p
    )
    sens_d = compute_sensitivity_breakdown(
        phot_d, out_direct.particles_emitted, float(phot_d.solid_angle_fraction), mask_d
    )

    # Кольцо анализа звезды выбирается по негеометрическим фотонам (септальный пробой + рассеяние)
    # в диапазоне радиусов [6.0, 42.0] мм, чтобы широкий геометрический пик HEGP не смешивался с лучами септы
    inner_ring_radius = 6.0
    outer_ring_radius = 42.0
    non_geom_mask_p = mask_p & (phot_p.component != int(PhotonComponent.GEOMETRIC))
    non_geom_mask_d = mask_d & (phot_d.component != int(PhotonComponent.GEOMETRIC))
    star_p = compute_azimuthal_star_metrics(
        phot_p.relative_x[non_geom_mask_p],
        phot_p.relative_y[non_geom_mask_p],
        inner_radius=inner_ring_radius,
        outer_radius=outer_ring_radius,
        num_angle_bins=72,
    )
    star_d = compute_azimuthal_star_metrics(
        phot_d.relative_x[non_geom_mask_d],
        phot_d.relative_y[non_geom_mask_d],
        inner_radius=inner_ring_radius,
        outer_radius=outer_ring_radius,
        num_angle_bins=72,
    )

    std_p = float(np.std(star_p.angular_counts))
    std_d = float(np.std(star_d.angular_counts))
    if std_p > 0.0 and std_d > 0.0:
        corr_matrix = np.corrcoef(star_p.angular_counts, star_d.angular_counts)
        profile_corr = float(corr_matrix[0, 1])
    else:
        profile_corr = 1.0

    plot_septal_star_analysis(
        photons_parametric=phot_p,
        photons_direct=phot_d,
        azimuth_angles=star_p.bin_centers_angle,
        azimuth_counts_parametric=star_p.angular_counts,
        azimuth_counts_direct=star_d.angular_counts,
        collimator_code=preset.code,
        nuclide_name=nuclide.title,
        spatial_extent=45.0,
        output_path=output_dir / f"exp04_septal_star_{preset.code.lower()}_i131.png",
    )

    # Проверяем согласие долей септального пробоя и рассеяния между Parametric и Direct
    septal_diff = abs(
        sens_d.septal_penetration.fraction_of_total - sens_p.septal_penetration.fraction_of_total
    )
    scatter_diff = abs(
        sens_d.collimator_scatter.fraction_of_total - sens_p.collimator_scatter.fraction_of_total
    )
    harmonic_diff = abs(star_d.hexagonal_harmonic_ratio - star_p.hexagonal_harmonic_ratio)

    passed = (septal_diff <= 0.015) and (scatter_diff <= 0.015) and (harmonic_diff <= 0.05)

    return SeptalStarCaseResult(
        collimator_code=preset.code,
        nuclide_name=nuclide.title,
        sensitivity_parametric=sens_p,
        sensitivity_direct=sens_d,
        anisotropy_contrast_parametric=star_p.anisotropy_contrast,
        anisotropy_contrast_direct=star_d.anisotropy_contrast,
        harmonic6_ratio_parametric=star_p.hexagonal_harmonic_ratio,
        harmonic6_ratio_direct=star_d.hexagonal_harmonic_ratio,
        azimuthal_profile_correlation=profile_corr,
        passed=passed,
    )


def run_exp04_septal_star_stress(
    output_dir: Path,
    num_particles: int = 2_500_000,
) -> list[SeptalStarCaseResult]:
    """Запускает Эксперимент 4 для пары коллиматоров LEHR и HEGP на линии I-131 (364.5 кэВ).

    Args:
        output_dir: Папка для сохранения графиков, JSON и Markdown-отчета.
        num_particles: Число фотонов в конусе на каждый прогон.

    Returns:
        Список SeptalStarCaseResult для LEHR и HEGP.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_single_star_stress_case(
            collimator_code="LEHR",
            output_dir=output_dir,
            num_particles=num_particles,
            base_seed=404100,
        ),
        run_single_star_stress_case(
            collimator_code="HEGP",
            output_dir=output_dir,
            num_particles=num_particles,
            base_seed=404500,
        ),
    ]

    json_path = output_dir / "exp04_septal_star_stress.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(item) for item in results], json_file, indent=2, ensure_ascii=False)

    md_lines = [
        "# Эксперимент 4: Стресс-тест септального пробоя и эффекта 6-лучевой звезды (I-131, 364.5 кэВ)",
        "",
        "| Коллиматор GE | Метод | Доля `Geometric`, % | Доля `Septal Penetration`, % | Доля `Collimator Scatter`, % | Контраст 6 лучей | Амплитуда 6-й гармоники | Корреляция $I(\\varphi)$ | Статус |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in results:
        sens_param = item.sensitivity_parametric
        sens_direct = item.sensitivity_direct
        status_str = "PASS" if item.passed else "FAIL"
        md_lines.append(
            f"| **{item.collimator_code}** | `Parametric` | "
            f"{sens_param.geometric.fraction_of_total * 100.0:.2f}% | "
            f"{sens_param.septal_penetration.fraction_of_total * 100.0:.2f}% | "
            f"{sens_param.collimator_scatter.fraction_of_total * 100.0:.2f}% | "
            f"{item.anisotropy_contrast_parametric:.3f} | "
            f"{item.harmonic6_ratio_parametric:.3f} | "
            f"{item.azimuthal_profile_correlation:.4f} | **{status_str}** |"
        )
        md_lines.append(
            f"| **{item.collimator_code}** | `Direct` | "
            f"{sens_direct.geometric.fraction_of_total * 100.0:.2f}% | "
            f"{sens_direct.septal_penetration.fraction_of_total * 100.0:.2f}% | "
            f"{sens_direct.collimator_scatter.fraction_of_total * 100.0:.2f}% | "
            f"{item.anisotropy_contrast_direct:.3f} | "
            f"{item.harmonic6_ratio_direct:.3f} | "
            f"{item.azimuthal_profile_correlation:.4f} | **{status_str}** |"
        )

    md_path = output_dir / "exp04_septal_star_stress.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return results


__all__ = [
    "SeptalStarCaseResult",
    "run_single_star_stress_case",
    "run_exp04_septal_star_stress",
]

