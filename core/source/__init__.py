"""
Подпакет источников излучения, компилятора источников и Numba-ядер инжекции первичных и вторичных частиц.
"""

from core.source.alias_tables import build_alias_tables
from core.source.source_buffer import (
    SPATIAL_ANALYTIC,
    SPATIAL_POINT,
    SPATIAL_VOXEL,
    SPECTRUM_CONTINUOUS_TRAPEZOID,
    SPECTRUM_DISCRETE,
    SourceBuffer,
)
from core.source.sources import (
    AngularDistribution,
    ConeAngular,
    ContinuousSpectrum,
    DiscreteSpectrum,
    EnergySpectrum,
    IsotropicAngular,
    ParticleSource,
    ParticleSpecies,
    PointDistribution,
    RadiationChannel,
    Source,
    SpatialDistribution,
    SpatialType,
    SpectralLine,
    SpectrumType,
    VoxelDistribution,
)
from core.source.source_compiler import SourceCompiler
from core.source.primary_source import PrimarySource
from core.source.secondary_source import SecondarySource

__all__ = [
    "ParticleSpecies",
    "SpatialType",
    "SpectrumType",
    "SpatialDistribution",
    "PointDistribution",
    "VoxelDistribution",
    "SpectralLine",
    "EnergySpectrum",
    "DiscreteSpectrum",
    "ContinuousSpectrum",
    "AngularDistribution",
    "IsotropicAngular",
    "ConeAngular",
    "RadiationChannel",
    "Source",
    "ParticleSource",
    "PrimarySource",
    "SecondarySource",
    "SPATIAL_POINT",
    "SPATIAL_VOXEL",
    "SPATIAL_ANALYTIC",
    "SPECTRUM_DISCRETE",
    "SPECTRUM_CONTINUOUS_TRAPEZOID",
    "SourceBuffer",
    "SourceCompiler",
    "build_alias_tables",
]
