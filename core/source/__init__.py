"""
Подпакет источников излучения, компилятора источников и Numba-ядер инжекции первичных и вторичных частиц.
"""

from core.source.alias_tables import build_alias_tables
from core.source.source_buffer import (
    SourceBuffer,
    SpatialType,
    SpectrumType,
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
    SpectralLine,
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
    "SourceBuffer",
    "SourceCompiler",
    "build_alias_tables",
]
