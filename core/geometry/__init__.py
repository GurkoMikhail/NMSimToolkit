"""
Подпакет пространственной геометрии, физических объемов и детекторных систем.
"""

from core.geometry.direct_collimators import (
    CollimatorHoleShape,
    DirectParallelCollimator,
)
from core.geometry.flattened_scene import FlattenedScene
from core.geometry.geometries import Box, Geometry, PeriodicHexPrism
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.navigation_state import NavigationState
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.pet_scanners import PetScanner
from core.geometry.spect_kinematics import (
    compute_orbit_matrix,
    compute_spect_poses,
    extract_orbit_parameters,
)
from core.geometry.volumes import Volume, VolumeArray
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.woodcock_volumes import WoodcockParametricVolume, WoodcockVolume

__all__ = [
    "Geometry",
    "Box",
    "PeriodicHexPrism",
    "Volume",
    "VolumeArray",
    "WoodcockVolume",
    "WoodcockParametricVolume",
    "WoodcockVoxelVolume",
    "PetScanner",
    "ParametricParallelCollimator",
    "CollimatorHoleShape",
    "DirectParallelCollimator",
    "FlattenedScene",
    "NavigationState",
    "GeometryCompiler",
    "compute_spect_poses",
    "compute_orbit_matrix",
    "extract_orbit_parameters",
]

