from core.geometry.volumes import TransformableVolume
from typing import Tuple, Any
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Vector3D, Precision


class WoodcockVolume(TransformableVolume):
    """
    Базовый класс Woodcock объёма
    """


class WoodcockParameticVolume(WoodcockVolume):
    """
    Класс параметричекого Woodcock объёма
    """

    def _parametric_function(self, position: Vector3D) -> Tuple[Any, Any]: # type: ignore
        return [], None

    def get_material_by_position(self, position: Vector3D, local: bool = False, as_parent: bool = True) -> MaterialArray: # type: ignore
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        material = super().get_material_by_position(position, True, as_parent)
        inside = (material != 0).nonzero()[0]
        indices, new_material = self._parametric_function(position[inside])
        material[inside[indices]] = new_material
        return material
