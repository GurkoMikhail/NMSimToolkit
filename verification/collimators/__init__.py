"""Пакет верифицирующих экспериментов для параллельных коллиматоров (Parametric vs Direct)."""

from verification.collimators.exp01_ray_optical_depth import (
    CollimatorRayAuditMetrics,
    run_exp01_ray_optical_depth,
    run_single_collimator_ray_audit,
)
from verification.collimators.exp02_psf_ge_series import (
    CollimatorPsfExperimentResult,
    run_exp02_psf_ge_series,
    run_single_psf_comparison,
)
from verification.collimators.exp03_anger_distance_curve import (
    AngerCurveExperimentResult,
    DistancePointRecord,
    run_exp03_anger_distance_curve,
)
from verification.collimators.exp04_septal_star_stress import (
    SeptalStarCaseResult,
    run_exp04_septal_star_stress,
    run_single_star_stress_case,
)
from verification.collimators.exp05_speed_benchmark import (
    SpeedBenchmarkRecord,
    run_exp05_speed_benchmark,
)

__all__ = [
    "AngerCurveExperimentResult",
    "CollimatorPsfExperimentResult",
    "CollimatorRayAuditMetrics",
    "DistancePointRecord",
    "SeptalStarCaseResult",
    "SpeedBenchmarkRecord",
    "run_exp01_ray_optical_depth",
    "run_exp02_psf_ge_series",
    "run_exp03_anger_distance_curve",
    "run_exp04_septal_star_stress",
    "run_exp05_speed_benchmark",
    "run_single_collimator_ray_audit",
    "run_single_psf_comparison",
    "run_single_star_stress_case",
]
