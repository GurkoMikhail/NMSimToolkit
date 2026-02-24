import numpy as np
from typing import Tuple, Any, Union
from numpy.typing import NDArray
from core.geometry.volumes import TransformableVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Vector3D, Float

class WoodcockVolume(TransformableVolume): ...

class WoodcockParameticVolume(WoodcockVolume):
    def _parametric_function(self, position: Vector3D) -> Tuple[NDArray[np.bool_], Any]: ...
    def get_material_by_position(self, position: Vector3D, local: bool = False, as_parent: bool = True) -> MaterialArray: ...
