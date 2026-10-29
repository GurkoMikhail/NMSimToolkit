"""Набор интеграционных pytest-тестов физической верификации (EXP-01..EXP-07) и многосидовой устойчивости."""
from __future__ import annotations

import json
import math
from pathlib import Path

import hepunits as units
import numpy as np
import pytest

from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
)
from core.source.source_compiler import SourceCompiler
from core.source.sources import Source
import settings.database_setting as database_setting
from verification.common.scattering_and_stats import evaluate_holm_bonferroni
from verification.physics.exp01_cross_sections_and_target_sampling import (
    MATERIAL_CATALOG,
    PROCESS_COHERENT,
    PROCESS_COMPTON,
    PROCESS_LABELS,
    PROCESS_PHOTOELECTRIC,
    _build_verification_physics_buffer,
    _compute_multinomial_sampling_chi2,
    _compute_reference_element_probabilities,
    _sample_z_batch_kernel,
    run_exp01_cross_sections_and_target_sampling,
)
from verification.physics.exp02_scattering_indicatrices_and_kinematics import (
    _run_coherent_indicatrix_audit,
    _run_compton_indicatrix_audit,
    run_exp02_scattering_indicatrices_and_kinematics,
)
from verification.physics.exp03_woodcock_and_delta_tracking import (
    run_exp03_woodcock_and_delta_tracking,
)
from verification.physics.exp04_energy_partition_and_atomic_relaxation import (
    run_exp04_energy_partition_and_atomic_relaxation,
)
from verification.physics.exp05_secondary_source_and_escape_peaks import (
    run_exp05_secondary_source_and_escape_peaks,
)
from verification.physics.exp06_primary_sources_and_poisson_stream import (
    _run_poisson_ensemble_kernel,
    run_exp06_primary_sources_and_poisson_stream,
)
from verification.physics.exp07_macroscopic_transport_and_dosimetry import (
    run_exp07_macroscopic_transport_and_dosimetry,
)
from verification.run_verification import (
    _collect_run_provenance,
    _save_dataclass_report,
    _write_physics_master_summary,
)

MULTISEED_TUPLE: tuple[int, int, int] = (2026_101, 2026_202, 2026_303)


@pytest.mark.verification
def test_quick_physics_verification_experiments_pass(tmp_path: Path) -> None:
    """Проверяет успешное прохождение всех 7 экспериментов верификации физики в быстром режиме (quick=True)."""
    exp01_report = run_exp01_cross_sections_and_target_sampling(output_dir=tmp_path, quick=True)
    assert exp01_report.passed is True
    _save_dataclass_report(exp01_report, tmp_path / "exp01_cross_sections.json", quick=True, elapsed_seconds=0.1)

    exp02_report = run_exp02_scattering_indicatrices_and_kinematics(output_dir=tmp_path, quick=True)
    assert exp02_report.passed is True
    _save_dataclass_report(exp02_report, tmp_path / "exp02_scattering_indicatrices.json", quick=True, elapsed_seconds=0.1)

    exp03_report = run_exp03_woodcock_and_delta_tracking(output_dir=tmp_path, quick=True)
    assert exp03_report.passed is True
    _save_dataclass_report(exp03_report, tmp_path / "exp03_woodcock_tracking.json", quick=True, elapsed_seconds=0.1)

    exp04_report = run_exp04_energy_partition_and_atomic_relaxation(output_dir=tmp_path, quick=True)
    assert exp04_report.passed is True
    _save_dataclass_report(exp04_report, tmp_path / "exp04_energy_partition.json", quick=True, elapsed_seconds=0.1)

    exp05_report = run_exp05_secondary_source_and_escape_peaks(output_dir=tmp_path, quick=True)
    assert exp05_report.passed is True
    _save_dataclass_report(exp05_report, tmp_path / "exp05_secondary_source.json", quick=True, elapsed_seconds=0.1)

    exp06_report = run_exp06_primary_sources_and_poisson_stream(output_dir=tmp_path, quick=True)
    assert exp06_report.passed is True
    _save_dataclass_report(exp06_report, tmp_path / "exp06_primary_sources.json", quick=True, elapsed_seconds=0.1)

    exp07_report = run_exp07_macroscopic_transport_and_dosimetry(output_dir=tmp_path, quick=True)
    assert exp07_report.passed is True
    _save_dataclass_report(exp07_report, tmp_path / "exp07_macroscopic_dosimetry.json", quick=True, elapsed_seconds=0.1)

    provenance = _collect_run_provenance(quick=True, elapsed_seconds=0.7)
    assert "git_sha" in provenance and len(str(provenance["git_sha"])) >= 7
    assert "data_hashes_sha256" in provenance and len(provenance["data_hashes_sha256"]) >= 2

    summary_path = _write_physics_master_summary(
        output_dir=tmp_path,
        exp01_report=exp01_report,
        exp02_report=exp02_report,
        exp03_report=exp03_report,
        exp04_report=exp04_report,
        exp05_report=exp05_report,
        exp06_report=exp06_report,
        exp07_report=exp07_report,
        total_wall_time=0.7,
        quick=True,
        run_provenance=provenance,
    )
    assert summary_path.is_file()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Карта независимости эталонов" in summary_text
    assert "Физическая модель против реальности" in summary_text

    saved_json = json.loads((tmp_path / "exp01_cross_sections.json").read_text(encoding="utf-8"))
    assert "provenance" in saved_json
    assert saved_json["provenance"]["quick_mode"] is True


@pytest.mark.verification
@pytest.mark.parametrize("seed_value", MULTISEED_TUPLE)
def test_multiseed_target_z_sampling_holm_bonferroni(seed_value: int) -> None:
    """Проверяет многосидовую статистическую устойчивость выборки Z (_sample_Z_scalar) с поправкой Холма–Бонферрони."""
    materials_list = [
        database_setting.material_database[mat_name] for mat_name in MATERIAL_CATALOG.values()
    ]
    processes_list = [
        PhotoelectricEffect(),
        ComptonScattering(),
        CoherentScattering(),
    ]
    physics_buffer = _build_verification_physics_buffer(materials_list, processes_list)

    rng = np.random.default_rng(np.random.SeedSequence(seed_value))
    rng_ctx = RNGContext.from_numpy_rng(rng)

    num_samples = 40_000
    samples_float = float(num_samples)
    test_energies = (
        float(15.0 * units.keV),
        float(30.0 * units.keV),
        float(33.0 * units.keV),
        float(33.4 * units.keV),
        float(88.5 * units.keV),
        float(140.5 * units.keV),
        float(511.0 * units.keV),
    )

    all_pvalues: list[float] = []
    for mat_label in ("Water", "NaI", "Bone"):
        material = database_setting.material_database[MATERIAL_CATALOG[mat_label]]
        mat_id = Index(material.ID)

        for proc_col_idx in (PROCESS_PHOTOELECTRIC, PROCESS_COMPTON, PROCESS_COHERENT):
            proc_name = PROCESS_LABELS[proc_col_idx]
            for photon_energy in test_energies:
                _, z_list, kernel_probs, _ = _compute_reference_element_probabilities(
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

                obs_counts = [int(np.count_nonzero(sampled_z == elem_z)) for elem_z in z_list]
                emp_probs = [float(cnt_val) / samples_float for cnt_val in obs_counts]
                max_diff = max(abs(emp_p - kern_p) for emp_p, kern_p in zip(emp_probs, kernel_probs))
                max_elem_se = max(
                    math.sqrt(max(samples_float * kern_p * (1.0 - kern_p), 1.0)) / samples_float
                    for kern_p in kernel_probs
                )
                allowed_diff = max(4.5 * max_elem_se, 3.0 / samples_float)
                assert max_diff < allowed_diff

                _, _, chi2_pval = _compute_multinomial_sampling_chi2(
                    observed_counts=obs_counts,
                    expected_probs=kernel_probs,
                    num_samples=num_samples,
                )
                all_pvalues.append(chi2_pval)

    holm_passed, min_pvalue, _ = evaluate_holm_bonferroni(all_pvalues, alpha=0.001)
    assert holm_passed is True, f"Нарушен критерий Холма–Бонферрони для сида {seed_value}: min_p={min_pvalue:.6e}"


@pytest.mark.verification
@pytest.mark.parametrize("seed_value", MULTISEED_TUPLE)
def test_multiseed_compton_and_coherent_indicatrices_holm_bonferroni(seed_value: int) -> None:
    """Проверяет многосидовую устойчивость индикатрис Комптона и Рэлея с поправкой Холма–Бонферрони."""
    rng = np.random.default_rng(np.random.SeedSequence(seed_value))
    rng_ctx = RNGContext.from_numpy_rng(rng)

    num_samples = 60_000
    cos_bin_edges = np.linspace(-1.0, 1.0, 41, dtype=np.float64)

    compton_metrics, _, compton_holm_passed, min_comp_p = _run_compton_indicatrix_audit(
        num_samples=num_samples,
        bin_edges=cos_bin_edges,
        rng_ctx=rng_ctx,
    )
    assert compton_holm_passed is True, f"Комптон не прошел Холма–Бонферрони на сиде {seed_value}: min_p={min_comp_p}"
    assert all(metric_item.passed for metric_item in compton_metrics)

    coherent_metrics, _, coherent_holm_passed, min_coh_p = _run_coherent_indicatrix_audit(
        num_samples=num_samples,
        plot_bin_edges=cos_bin_edges,
        rng_ctx=rng_ctx,
    )
    assert coherent_holm_passed is True, f"Рэлей не прошел Холма–Бонферрони на сиде {seed_value}: min_p={min_coh_p}"
    assert all(metric_item.passed for metric_item in coherent_metrics)


@pytest.mark.verification
@pytest.mark.parametrize("seed_value", MULTISEED_TUPLE)
def test_multiseed_poisson_fano_dispersion_stability(seed_value: int) -> None:
    """Проверяет многосидовую статистическую устойчивость фактора Фано нестационарного потока Льюиса–Шедлера."""
    window_start = Float(0.0 * units.ns)
    window_stop = Float(60.0 * units.ns)

    first_activity = Float(18.0 / units.ns)
    first_half_life = Float(12.0 * units.ns)
    first_yield = Float(1.0)
    first_energy = Float(0.1405 * units.MeV)

    second_activity = Float(6.0 / units.ns)
    second_half_life = Float(50.0 * units.ns)
    second_yield = Float(0.85)
    second_energy = Float(0.3645 * units.MeV)

    source_fast = Source.from_point(
        activity=first_activity,
        energy=first_energy,
        half_life=first_half_life,
        branching_ratio=first_yield,
        name="FastDecaySeedTest",
    )
    source_slow = Source.from_point(
        activity=second_activity,
        energy=second_energy,
        half_life=second_half_life,
        branching_ratio=second_yield,
        name="SlowDecaySeedTest",
    )
    source_buffer = SourceCompiler.compile([source_fast, source_slow])

    lambda_first = math.log(2.0) / float(first_half_life)
    lambda_second = math.log(2.0) / float(second_half_life)
    expected_first = (float(first_activity) * float(first_yield) / lambda_first) * (
        1.0 - math.exp(-lambda_first * float(window_stop))
    )
    expected_second = (float(second_activity) * float(second_yield) / lambda_second) * (
        1.0 - math.exp(-lambda_second * float(window_stop))
    )
    expected_total = expected_first + expected_second

    num_realizations = 160
    bank = ParticleBank.allocate(2048)
    rng = np.random.default_rng(np.random.SeedSequence(seed_value))
    rng_ctx = RNGContext.from_numpy_rng(rng)

    counts_per_run, _, _, _, non_monotonic_violations = _run_poisson_ensemble_kernel(
        bank=bank,
        source_buffer=source_buffer,
        start_time=window_start,
        stop_time=window_stop,
        num_realizations=num_realizations,
        num_time_bins=24,
        first_component_energy=first_energy,
        rng_ctx=rng_ctx,
    )

    counts_float = counts_per_run.astype(np.float64)
    empirical_mean = float(np.mean(counts_float))
    empirical_var = float(np.var(counts_float, ddof=1))
    fano_index = empirical_var / empirical_mean
    fano_std_err = math.sqrt(2.0 / float(num_realizations - 1))
    fano_z_score = abs(fano_index - 1.0) / fano_std_err

    mean_std_err = math.sqrt(expected_total / float(num_realizations))
    mean_z_score = abs(empirical_mean - expected_total) / mean_std_err

    assert non_monotonic_violations == 0
    assert mean_z_score < 3.5
    assert fano_z_score <= 3.5
