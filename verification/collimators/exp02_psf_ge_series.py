"""Эксперимент 2: Покомпонентная верификация PSF на серии коллиматоров GE (LEHR, LEUHR, MEGP, HEGP).

Для каждой модели коллиматора GE выполняется парная Монте-Карло симуляция точечного источника
на расстоянии Z = 100 мм для двух реализаций:
- ParametricParallelCollimator (дельта-трекинг Вудкока с @cfunc);
- DirectParallelCollimator (аналитический RayCasting по PeriodicHexPrism).

Вылетевшие из выходного торца фотоны разделяются строго на 3 физические компоненты:
1. GEOMETRIC — пролет через одиночный гексагональный канал без взаимодействий;
2. SEPTAL_PENETRATION — прямой пролет сквозь свинцовые перегородки (септы) без рассеяния;
3. COLLIMATOR_SCATTER — фотоны, испытавшие комптоновское или когерентное рассеяние в свинце.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from verification.common.metrics import (
    CollimatorSensitivityBreakdown,
    ProfileResolutionMetrics,
    StatisticalComparisonResult,
    compare_collimator_outputs,
    compute_psf_resolution_metrics,
    compute_sensitivity_breakdown,
    filter_energy_window,
)
from verification.common.plotting import plot_psf_comparison_panel
from verification.common.presets import (
    GE_COLLIMATOR_PRESETS,
    RADIONUCLIDE_PRESETS,
    compute_anger_geometric_fwhm,
    compute_anger_geometric_sensitivity,
)
from verification.common.scene_builders import run_collimator_simulation


@dataclass(frozen=True)
class CollimatorPsfExperimentResult:
    """Полный результат верификации PSF для одной пары (коллиматор GE, радионуклид).

    Attributes:
        collimator_code: Код коллиматора ('LEHR', 'LEUHR', 'MEGP', 'HEGP').
        nuclide_name: Обозначение радионуклида.
        photon_energy: Энергия гамма-линии [МэВ].
        source_distance: Расстояние от источника до входного торца коллиматора [мм].
        num_particles: Число сгенерированных фотонов в конусе.
        anger_theoretical_fwhm: Теоретическое геометрическое разрешение по Ангеру [мм].
        anger_theoretical_sensitivity: Теоретическая геометрическая чувствительность Ангера.
        sensitivity_parametric: Покомпонентная чувствительность Parametric.
        sensitivity_direct: Покомпонентная чувствительность Direct.
        resolution_parametric: Пространственное разрешение FWHM/FWTM для Parametric.
        resolution_direct: Пространственное разрешение FWHM/FWTM для Direct.
        statistical_comparison: Статистические критерии согласия (KS-тест, хи-квадрат, Z-score).
        elapsed_time_parametric: Время расчета Parametric [с].
        elapsed_time_direct: Время расчета Direct [с].
        passed: Признак статистического согласия двух моделей.
    """

    collimator_code: str
    nuclide_name: str
    photon_energy: float
    source_distance: float
    num_particles: int
    anger_theoretical_fwhm: float
    anger_theoretical_sensitivity: float
    sensitivity_parametric: CollimatorSensitivityBreakdown
    sensitivity_direct: CollimatorSensitivityBreakdown
    resolution_parametric: ProfileResolutionMetrics
    resolution_direct: ProfileResolutionMetrics
    statistical_comparison: StatisticalComparisonResult
    elapsed_time_parametric: float
    elapsed_time_direct: float
    passed: bool


DEFAULT_GE_PSF_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("LEHR", "Tc99m"),
    ("LEUHR", "Tc99m"),
    ("MEGP", "In111"),
    ("HEGP", "I131"),
)


def run_single_psf_comparison(
    collimator_code: str,
    nuclide_key: str,
    output_dir: Path,
    num_particles: int = 5_000_000,
    source_distance: float = 100.0,
    max_polar_angle: float = 0.18,
    base_seed: int = 202601,
) -> CollimatorPsfExperimentResult:
    """Выполняет парную симуляцию и статистическое сравнение PSF для одного коллиматора GE.

    Args:
        collimator_code: Код модели коллиматора ('LEHR', 'LEUHR', 'MEGP', 'HEGP').
        nuclide_key: Ключ радионуклида в RADIONUCLIDE_PRESETS.
        output_dir: Папка для сохранения графиков.
        num_particles: Число фотонов в конусе на каждую реализацию.
        source_distance: Расстояние от источника до входного торца [мм].
        max_polar_angle: Полуугол конуса генерации [рад].
        base_seed: Базовый сид ГСЧ.

    Returns:
        Структура CollimatorPsfExperimentResult.
    """
    preset = GE_COLLIMATOR_PRESETS[collimator_code]
    nuclide = RADIONUCLIDE_PRESETS[nuclide_key]

    outcome_parametric = run_collimator_simulation(
        preset=preset,
        implementation_kind="parametric",
        source_distance=source_distance,
        energy=nuclide.primary_energy,
        particles_number=num_particles,
        max_polar_angle=max_polar_angle,
        cell_averaging=True,
        seed=base_seed,
    )

    outcome_direct = run_collimator_simulation(
        preset=preset,
        implementation_kind="direct",
        source_distance=source_distance,
        energy=nuclide.primary_energy,
        particles_number=num_particles,
        max_polar_angle=max_polar_angle,
        cell_averaging=True,
        seed=base_seed + 1000,
    )

    photons_param = outcome_parametric.photons
    photons_direct = outcome_direct.photons

    mask_param = filter_energy_window(photons_param, float(nuclide.primary_energy), 0.10)
    mask_direct = filter_energy_window(photons_direct, float(nuclide.primary_energy), 0.10)

    sens_param = compute_sensitivity_breakdown(
        photons=photons_param,
        emitted_count=outcome_parametric.particles_emitted,
        solid_angle_fraction=float(photons_param.solid_angle_fraction),
        energy_mask=mask_param,
    )
    sens_direct = compute_sensitivity_breakdown(
        photons=photons_direct,
        emitted_count=outcome_direct.particles_emitted,
        solid_angle_fraction=float(photons_direct.solid_angle_fraction),
        energy_mask=mask_direct,
    )

    spatial_extent = 30.0
    res_param = compute_psf_resolution_metrics(
        coord_x=photons_param.relative_x[mask_param],
        coord_y=photons_param.relative_y[mask_param],
        spatial_extent=spatial_extent,
        num_bins=121,
    )
    res_direct = compute_psf_resolution_metrics(
        coord_x=photons_direct.relative_x[mask_direct],
        coord_y=photons_direct.relative_y[mask_direct],
        spatial_extent=spatial_extent,
        num_bins=121,
    )

    stat_comp = compare_collimator_outputs(
        photons_reference=photons_param,
        photons_candidate=photons_direct,
        emitted_reference=outcome_parametric.particles_emitted,
        emitted_candidate=outcome_direct.particles_emitted,
        solid_angle_fraction=float(photons_param.solid_angle_fraction),
        spatial_extent=spatial_extent,
        num_bins_2d=80,
        num_bins_radial=60,
        energy_mask_reference=mask_param,
        energy_mask_candidate=mask_direct,
    )

    plot_psf_comparison_panel(
        photons_parametric=photons_param,
        photons_direct=photons_direct,
        mask_parametric=mask_param,
        mask_direct=mask_direct,
        collimator_code=preset.code,
        nuclide_name=nuclide.title,
        spatial_extent=spatial_extent,
        output_path=output_dir / f"exp02_psf_{preset.code.lower()}_{nuclide_key.lower()}.png",
    )

    anger_fwhm = float(compute_anger_geometric_fwhm(preset, source_distance, nuclide.primary_energy))
    anger_sens = float(compute_anger_geometric_sensitivity(preset, nuclide.primary_energy))

    # Критерий согласия: Z-score полной и покомпонентной чувствительности <= 3.5,
    # приведенный хи-квадрат <= 2.0, доля бинов 2D-карты в пределах +-3 sigma >= 98%
    passed = (
        abs(stat_comp.sensitivity_z_score) <= 3.5
        and abs(stat_comp.geometric_z_score) <= 3.5
        and abs(stat_comp.septal_z_score) <= 3.5
        and abs(stat_comp.scatter_z_score) <= 3.5
        and stat_comp.reduced_chi2_radial <= 2.0
        and stat_comp.fraction_within_3sigma_2d >= 0.98
    )

    return CollimatorPsfExperimentResult(
        collimator_code=preset.code,
        nuclide_name=nuclide.title,
        photon_energy=float(nuclide.primary_energy),
        source_distance=source_distance,
        num_particles=num_particles,
        anger_theoretical_fwhm=anger_fwhm,
        anger_theoretical_sensitivity=anger_sens,
        sensitivity_parametric=sens_param,
        sensitivity_direct=sens_direct,
        resolution_parametric=res_param,
        resolution_direct=res_direct,
        statistical_comparison=stat_comp,
        elapsed_time_parametric=float(outcome_parametric.elapsed_time),
        elapsed_time_direct=float(outcome_direct.elapsed_time),
        passed=passed,
    )


def run_exp02_psf_ge_series(
    output_dir: Path,
    num_particles: int = 5_000_000,
    scenarios: tuple[tuple[str, str], ...] = DEFAULT_GE_PSF_SCENARIOS,
) -> list[CollimatorPsfExperimentResult]:
    """Запускает Эксперимент 2 по серии коллиматоров GE и генерирует отчеты.

    Args:
        output_dir: Папка для сохранения графиков, JSON и Markdown-отчета.
        num_particles: Число фотонов в конусе на каждый коллиматор и метод.
        scenarios: Кортеж пар (collimator_code, nuclide_key).

    Returns:
        Список результатов CollimatorPsfExperimentResult.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[CollimatorPsfExperimentResult] = []

    for idx, (collimator_code, nuclide_key) in enumerate(scenarios):
        res = run_single_psf_comparison(
            collimator_code=collimator_code,
            nuclide_key=nuclide_key,
            output_dir=output_dir,
            num_particles=num_particles,
            base_seed=202600 + idx * 100,
        )
        results.append(res)

    json_path = output_dir / "exp02_psf_ge_series.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(item) for item in results], json_file, indent=2, ensure_ascii=False)

    md_lines = [
        "# Эксперимент 2: Покомпонентная верификация PSF на серии коллиматоров GE Healthcare",
        "",
        f"Статистика: **{num_particles:,}** историй в конусе ($\\theta_{{\\max}} = 0.18\\text{{ рад}}$) на каждый прогон.",
        "",
        "## 1. Сравнение пространственного разрешения и полной чувствительности ($Z = 100\\text{ мм}$)",
        "",
        "| Коллиматор (Нуклид) | FWHM Теория Ангера, мм | FWHM `Parametric`, мм | FWHM `Direct`, мм | FWTM `Parametric`, мм | FWTM `Direct`, мм | Sens `Parametric` ($\\times 10^{-4}$) | Sens `Direct` ($\\times 10^{-4}$) | $\\chi^2/\\text{ndf}$ | KS $p$-val | Статус |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in results:
        status_str = "PASS" if item.passed else "FAIL"
        sens_param_tot = item.sensitivity_parametric.total
        sens_direct_tot = item.sensitivity_direct.total
        stat_comp = item.statistical_comparison
        md_lines.append(
            f"| **{item.collimator_code}** ({item.nuclide_name}) | "
            f"{item.anger_theoretical_fwhm:.2f} | "
            f"{item.resolution_parametric.fwhm_mean:.2f} | "
            f"{item.resolution_direct.fwhm_mean:.2f} | "
            f"{item.resolution_parametric.fwtm_mean:.2f} | "
            f"{item.resolution_direct.fwtm_mean:.2f} | "
            f"{sens_param_tot.sensitivity * 1e4:.3f} $\\pm$ {sens_param_tot.sensitivity_error * 1e4:.3f} | "
            f"{sens_direct_tot.sensitivity * 1e4:.3f} $\\pm$ {sens_direct_tot.sensitivity_error * 1e4:.3f} | "
            f"{stat_comp.reduced_chi2_radial:.2f} | {stat_comp.ks_pvalue_radial:.3f} | **{status_str}** |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Декомпозиция вылетевших фотонов по 3 физическим компонентам коллиматора",
            "",
            "| Коллиматор | Метод | Отсчетов всего | `Geometric` (отсчеты / доля) | `Septal Penetration` (отсчеты / доля) | `Collimator Scatter` (отсчеты / доля) | $Z$-score (`Geom` / `Septal` / `Scatter`) | Время счета, с |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for item in results:
        sens_param = item.sensitivity_parametric
        sens_direct = item.sensitivity_direct
        stat_comp = item.statistical_comparison
        z_str = f"`{stat_comp.geometric_z_score:+.2f}` / `{stat_comp.septal_z_score:+.2f}` / `{stat_comp.scatter_z_score:+.2f}`"
        md_lines.append(
            f"| **{item.collimator_code}** | `Parametric` | {sens_param.total.count:,} | "
            f"{sens_param.geometric.count:,} ({sens_param.geometric.fraction_of_total * 100.0:.2f}%) | "
            f"{sens_param.septal_penetration.count:,} ({sens_param.septal_penetration.fraction_of_total * 100.0:.2f}%) | "
            f"{sens_param.collimator_scatter.count:,} ({sens_param.collimator_scatter.fraction_of_total * 100.0:.2f}%) | "
            f"{z_str} | {item.elapsed_time_parametric:.2f} |"
        )
        md_lines.append(
            f"| **{item.collimator_code}** | `Direct` | {sens_direct.total.count:,} | "
            f"{sens_direct.geometric.count:,} ({sens_direct.geometric.fraction_of_total * 100.0:.2f}%) | "
            f"{sens_direct.septal_penetration.count:,} ({sens_direct.septal_penetration.fraction_of_total * 100.0:.2f}%) | "
            f"{sens_direct.collimator_scatter.count:,} ({sens_direct.collimator_scatter.fraction_of_total * 100.0:.2f}%) | "
            f"{z_str} | {item.elapsed_time_direct:.2f} |"
        )

    md_path = output_dir / "exp02_psf_ge_series.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return results


__all__ = [
    "CollimatorPsfExperimentResult",
    "DEFAULT_GE_PSF_SCENARIOS",
    "run_single_psf_comparison",
    "run_exp02_psf_ge_series",
]

