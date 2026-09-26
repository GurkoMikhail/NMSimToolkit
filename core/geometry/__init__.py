"""
Подпакет пространственной геометрии, физических объемов и детекторных систем.
"""

from core.geometry.geometries import Geometry, Box
from core.geometry.volumes import Volume, VolumeArray
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.pet_scanners import PetScanner
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
    ParametricParallelSquareCollimator,
)
from core.geometry.navigation_state import NavigationState
from core.geometry.geometry_compiler import GeometryCompiler

__all__ = [
    'Geometry',
    'Box',
    'Volume',
    'VolumeArray',
    'WoodcockVoxelVolume',
    'GammaCamera',
    'PetScanner',
    'ParametricParallelCollimator',
    'ParametricParallelSquareCollimator',
    'NavigationState',
    'GeometryCompiler',
]
