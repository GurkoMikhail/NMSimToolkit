"""Пакет комплексной верификации физического ядра NMSimToolkit."""
from verification.physics.exp01_cross_sections_and_target_sampling import (
    Exp01CrossSectionsReport,
    run_exp01_cross_sections_and_target_sampling,
)
from verification.physics.exp02_scattering_indicatrices_and_kinematics import (
    Exp02ScatteringIndicatricesReport,
    run_exp02_scattering_indicatrices_and_kinematics,
)
from verification.physics.exp03_woodcock_and_delta_tracking import (
    Exp03WoodcockTrackingReport,
    run_exp03_woodcock_and_delta_tracking,
)
from verification.physics.exp04_energy_partition_and_atomic_relaxation import (
    Exp04EnergyPartitionReport,
    run_exp04_energy_partition_and_atomic_relaxation,
)
from verification.physics.exp05_secondary_source_and_escape_peaks import (
    Exp05SecondarySourceReport,
    run_exp05_secondary_source_and_escape_peaks,
)
from verification.physics.exp06_primary_sources_and_poisson_stream import (
    Exp06PrimarySourcesReport,
    run_exp06_primary_sources_and_poisson_stream,
)
from verification.physics.exp07_macroscopic_transport_and_dosimetry import (
    Exp07MacroscopicDosimetryReport,
    run_exp07_macroscopic_transport_and_dosimetry,
)

__all__ = [
    "Exp01CrossSectionsReport",
    "Exp02ScatteringIndicatricesReport",
    "Exp03WoodcockTrackingReport",
    "Exp04EnergyPartitionReport",
    "Exp05SecondarySourceReport",
    "Exp06PrimarySourcesReport",
    "Exp07MacroscopicDosimetryReport",
    "run_exp01_cross_sections_and_target_sampling",
    "run_exp02_scattering_indicatrices_and_kinematics",
    "run_exp03_woodcock_and_delta_tracking",
    "run_exp04_energy_partition_and_atomic_relaxation",
    "run_exp05_secondary_source_and_escape_peaks",
    "run_exp06_primary_sources_and_poisson_stream",
    "run_exp07_macroscopic_transport_and_dosimetry",
]
