"""Эксперимент 3: Зависимость пространственного разрешения FWHM/FWTM и чувствительности от расстояния Z.

Верифицирует линейный закон деградации разрешения Хола Ангера:
    FWHM(Z) = d * (L_eff + Z) / L_eff,  где L_eff = L - 2 / mu
и инвариантность геометрической чувствительности точечного источника S(Z) = const
при изменении расстояния Z от 20 мм до 250 мм для Parametric и Direct коллиматоров.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from verification.common.metrics import (
    compute_psf_resolution_metrics,
    compute_sensitivity_breakdown,
    filter_energy_window,
)
from verification.common.plotting import plot_anger_distance_curves
from verification.common.presets import (
    GE_COLLIMATOR_PRESETS,
    RADIONUCLIDE_PRESETS,
    compute_anger_geometric_fwhm,
    compute_anger_geometric_sensitivity,
)
from verification.common.scene_builders import run_collimator_simulation


@dataclass(frozen=True)
class DistancePointRecord:
    """Результаты измерения PSF и чувствительности на одном расстоянии Z.

    Attributes:
        source_distance: Расстояние от источника до входного торца коллиматора [мм].
        anger_fwhm: Теоретическое разрешение Ангера [мм].
        fwhm_parametric: Измеренное FWHM для Parametric [мм].
        fwhm_direct: Измеренное FWHM для Direct [мм].
        fwtm_parametric: Измеренное FWTM для Parametric [мм].
        fwtm_direct: Измеренное FWTM для Direct [мм].
        geom_sensitivity_parametric: Геометрическая чувствительность Parametric.
        geom_sensitivity_direct: Геометрическая чувствительность Direct.
        total_sensitivity_parametric: Полная чувствительность Parametric.
        total_sensitivity_direct: Полная чувствительность Direct.
    """

    source_distance: float
    anger_fwhm: float
    fwhm_parametric: float
    fwhm_direct: float
    fwtm_parametric: float
    fwtm_direct: float
    geom_sensitivity_parametric: float
    geom_sensitivity_direct: float
    total_sensitivity_parametric: float
    total_sensitivity_direct: float


@dataclass(frozen=True)
class AngerCurveExperimentResult:
    """Итоговый отчет Эксперимента 3 по кривой Ангера FWHM(Z) и S(Z).

    Attributes:
        collimator_code: Код коллиматора GE.
        nuclide_name: Имя радионуклида.
        anger_theoretical_sensitivity: Теоретическая геометрическая чувствительность Ангера.
        points: Список измерений по точкам расстояния Z.
        linear_fit_r2_parametric: Коэффициент детерминации R^2 линейности FWHM(Z) для Parametric.
        linear_fit_r2_direct: Коэффициент детерминации R^2 линейности FWHM(Z) для Direct.
        max_fwhm_diff_between_models: Максимальное расхождение FWHM между Parametric и Direct [мм].
        sensitivity_cv_across_z_direct: Коэффициент вариации геометрической чувствительности по Z.
        passed: Признак успешного прохождения верификации.
    """

    collimator_code: str
    nuclide_name: str
    anger_theoretical_sensitivity: float
    points: list[DistancePointRecord]
    linear_fit_r2_parametric: float
    linear_fit_r2_direct: float
    max_fwhm_diff_between_models: float
    sensitivity_cv_across_z_direct: float
    passed: bool


DEFAULT_DISTANCES: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0, 200.0, 250.0)


def _compute_linear_r2(x_vals: list[float], y_vals: list[float]) -> float:
    """Вычисляет коэффициент детерминации R^2 линейной регрессии y(x)."""
    arr_x = np.asarray(x_vals, dtype=np.float64)
    arr_y = np.asarray(y_vals, dtype=np.float64)
    if arr_x.size < 2:
        return 1.0
    coeffs = np.polyfit(arr_x, arr_y, 1)
    fitted = np.polyval(coeffs, arr_x)
    ss_res = float(np.sum((arr_y - fitted) ** 2))
    ss_tot = float(np.sum((arr_y - np.mean(arr_y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0


def run_exp03_anger_distance_curve(
    output_dir: Path,
    collimator_code: str = "LEHR",
    nuclide_key: str = "Tc99m",
    distances: tuple[float, ...] = DEFAULT_DISTANCES,
    num_particles_per_point: int = 1_500_000,
) -> AngerCurveExperimentResult:
    """Запускает серию расчетов по расстояниям Z и сверяет кривые с теорией Ангера.

    Args:
        output_dir: Папка для сохранения графиков, JSON и Markdown-отчета.
        collimator_code: Код модели коллиматора (по умолчанию 'LEHR').
        nuclide_key: Ключ радионуклида (по умолчанию 'Tc99m').
        distances: Кортеж расстояний источник-коллиматор [мм].
        num_particles_per_point: Число фотонов в конусе на каждую точку Z.

    Returns:
        Структура AngerCurveExperimentResult.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = GE_COLLIMATOR_PRESETS[collimator_code]
    nuclide = RADIONUCLIDE_PRESETS[nuclide_key]

    # Максимальный угол пропускания канала LEHR ~ atan(1.5 / 35) = 0.043 рад.
    # Конус 0.09 рад (> 2x критического угла канала) гарантирует полный перехват
    # всех геометрических и септальных лучей при высокой плотности статистики в пике.
    cone_half_angle = 0.09
    anger_sens = float(compute_anger_geometric_sensitivity(preset, nuclide.primary_energy))

    point_records: list[DistancePointRecord] = []

    for idx, dist_val in enumerate(distances):
        seed_base = 303000 + idx * 100
        out_param = run_collimator_simulation(
            preset=preset,
            implementation_kind="parametric",
            source_distance=dist_val,
            energy=nuclide.primary_energy,
            particles_number=num_particles_per_point,
            max_polar_angle=cone_half_angle,
            cell_averaging=True,
            seed=seed_base,
        )
        out_direct = run_collimator_simulation(
            preset=preset,
            implementation_kind="direct",
            source_distance=dist_val,
            energy=nuclide.primary_energy,
            particles_number=num_particles_per_point,
            max_polar_angle=cone_half_angle,
            cell_averaging=True,
            seed=seed_base + 50,
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

        # Масштабируем окно гистограммы пропорционально ожидаемой ширине пика для сохранения суббинной точности
        anger_fwhm_val = float(compute_anger_geometric_fwhm(preset, dist_val, nuclide.primary_energy))
        extent_val = max(10.0, anger_fwhm_val * 2.2)

        res_p = compute_psf_resolution_metrics(
            phot_p.relative_x[mask_p],
            phot_p.relative_y[mask_p],
            spatial_extent=extent_val,
            num_bins=61,
        )
        res_d = compute_psf_resolution_metrics(
            phot_d.relative_x[mask_d],
            phot_d.relative_y[mask_d],
            spatial_extent=extent_val,
            num_bins=61,
        )

        point_records.append(
            DistancePointRecord(
                source_distance=dist_val,
                anger_fwhm=anger_fwhm_val,
                fwhm_parametric=res_p.fwhm_mean,
                fwhm_direct=res_d.fwhm_mean,
                fwtm_parametric=res_p.fwtm_mean,
                fwtm_direct=res_d.fwtm_mean,
                geom_sensitivity_parametric=sens_p.geometric.sensitivity,
                geom_sensitivity_direct=sens_d.geometric.sensitivity,
                total_sensitivity_parametric=sens_p.total.sensitivity,
                total_sensitivity_direct=sens_d.total.sensitivity,
            )
        )

    z_arr = np.array([rec.source_distance for rec in point_records], dtype=np.float64)
    fwhm_p_arr = np.array([rec.fwhm_parametric for rec in point_records], dtype=np.float64)
    fwhm_d_arr = np.array([rec.fwhm_direct for rec in point_records], dtype=np.float64)
    fwtm_p_arr = np.array([rec.fwtm_parametric for rec in point_records], dtype=np.float64)
    fwtm_d_arr = np.array([rec.fwtm_direct for rec in point_records], dtype=np.float64)
    fwhm_th_arr = np.array([rec.anger_fwhm for rec in point_records], dtype=np.float64)

    sens_gp_arr = np.array([rec.geom_sensitivity_parametric for rec in point_records], dtype=np.float64)
    sens_gd_arr = np.array([rec.geom_sensitivity_direct for rec in point_records], dtype=np.float64)
    sens_tp_arr = np.array([rec.total_sensitivity_parametric for rec in point_records], dtype=np.float64)
    sens_td_arr = np.array([rec.total_sensitivity_direct for rec in point_records], dtype=np.float64)

    plot_anger_distance_curves(
        distances=z_arr,
        fwhm_parametric=fwhm_p_arr,
        fwhm_direct=fwhm_d_arr,
        fwtm_parametric=fwtm_p_arr,
        fwtm_direct=fwtm_d_arr,
        fwhm_anger_theory=fwhm_th_arr,
        sens_geom_parametric=sens_gp_arr,
        sens_geom_direct=sens_gd_arr,
        sens_total_parametric=sens_tp_arr,
        sens_total_direct=sens_td_arr,
        sens_anger_theory=anger_sens,
        collimator_code=preset.code,
        nuclide_name=nuclide.title,
        output_path=output_dir / f"exp03_anger_curves_{preset.code.lower()}.png",
    )

    r2_param = _compute_linear_r2(
        [rec.source_distance for rec in point_records],
        [rec.fwhm_parametric for rec in point_records],
    )
    r2_direct = _compute_linear_r2(
        [rec.source_distance for rec in point_records],
        [rec.fwhm_direct for rec in point_records],
    )
    max_fwhm_diff = float(np.max(np.abs(fwhm_d_arr - fwhm_p_arr)))
    sens_cv = float(np.std(sens_gd_arr) / np.mean(sens_gd_arr)) if np.mean(sens_gd_arr) > 0 else 0.0

    passed = (r2_param >= 0.995) and (r2_direct >= 0.995) and (max_fwhm_diff <= 0.45) and (sens_cv <= 0.04)

    result = AngerCurveExperimentResult(
        collimator_code=preset.code,
        nuclide_name=nuclide.title,
        anger_theoretical_sensitivity=anger_sens,
        points=point_records,
        linear_fit_r2_parametric=r2_param,
        linear_fit_r2_direct=r2_direct,
        max_fwhm_diff_between_models=max_fwhm_diff,
        sensitivity_cv_across_z_direct=sens_cv,
        passed=passed,
    )

    json_path = output_dir / "exp03_anger_distance_curve.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(asdict(result), json_file, indent=2, ensure_ascii=False)

    md_lines = [
        f"# Эксперимент 3: Зависимость разрешения и чувствительности от расстояния $Z$ ({preset.code}, {nuclide.title})",
        "",
        "| Расстояние $Z$, мм | FWHM Ангера, мм | FWHM `Parametric`, мм | FWHM `Direct`, мм | FWTM `Parametric`, мм | FWTM `Direct`, мм | $S_{\\text{geom}}$ `Param` ($\\times 10^{-4}$) | $S_{\\text{geom}}$ `Direct` ($\\times 10^{-4}$) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for rec in point_records:
        md_lines.append(
            f"| **{rec.source_distance:.0f}** | {rec.anger_fwhm:.2f} | "
            f"{rec.fwhm_parametric:.2f} | {rec.fwhm_direct:.2f} | "
            f"{rec.fwtm_parametric:.2f} | {rec.fwtm_direct:.2f} | "
            f"{rec.geom_sensitivity_parametric * 1e4:.3f} | {rec.geom_sensitivity_direct * 1e4:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## Сводные показатели согласия с теорией Хола Ангера",
            f"- Линейность $\\text{{FWHM}}(Z)$ (`Parametric`): $R^2 = {r2_param:.5f}$",
            f"- Линейность $\\text{{FWHM}}(Z)$ (`Direct`): $R^2 = {r2_direct:.5f}$",
            f"- Максимальное расхождение FWHM между моделями: `{max_fwhm_diff:.3f}` мм",
            f"- Теоретическая чувствительность Ангера $S_{{\\text{{Anger}}}}$: `{anger_sens * 1e4:.3f}e-4`",
            f"- Коэффициент вариации геометрической чувствительности по расстоянию $Z$: `{sens_cv * 100.0:.2f}%`",
            f"- Статус эксперимента: **{'PASS' if passed else 'FAIL'}**",
        ]
    )

    md_path = output_dir / "exp03_anger_distance_curve.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return result


__all__ = [
    "DistancePointRecord",
    "AngerCurveExperimentResult",
    "DEFAULT_DISTANCES",
    "run_exp03_anger_distance_curve",
]

