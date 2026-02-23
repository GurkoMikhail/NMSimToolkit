import numpy as np
from typing import Tuple, Any, Union, Generic
from numpy.typing import NDArray
from core.geometry.volumes import TransformableVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Vector3D, Precision

class WoodcockVolume(TransformableVolume[Precision]): ...

class WoodcockParameticVolume(WoodcockVolume[Precision]):
    def _parametric_function(self, position: Vector3D[Precision]) -> Tuple[NDArray[np.bool_], Any]: ...
    def get_material_by_position(self, position: Vector3D[Precision], local: bool = False, as_parent: bool = True) -> MaterialArray: ...
