from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from core.geometry.volumes import TransformableVolume
from core.materials.materials import Material, MaterialArray
from numba import cfunc, types

from core.other.typing_definitions import Float, Vector3D


class WoodcockVolume(TransformableVolume):
    """
    Базовый класс Woodcock объёма
    """


class WoodcockParameticVolume(WoodcockVolume):
    """
    Класс параметрического Woodcock объёма
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cfunc = None

    @property
    def cfunc_address(self) -> int:
        if self._cfunc is None:
            self._cfunc = self._compile_cfunc()
        return self._cfunc.address

    def _compile_cfunc(self):
        """
        Должен быть переопределен в наследниках для генерации @cfunc,
        которая принимает x, y, z и возвращает ID материала (int64).
        По умолчанию возвращает ID мажорантного материала (self.material).
        """
        material_id = self.material.ID
        @cfunc(types.int64(types.float64, types.float64, types.float64), cache=True)
        def default_parametric_func(x, y, z):
            return material_id
        return default_parametric_func

    @property
    def material_list(self) -> list[Material]:
        """
        Переопределите в наследниках, если используются другие материалы.
        По умолчанию возвращает только мажорантный материал.
        """
        return super().material_list

    def _parametric_function(self, position: Vector3D) -> Tuple[Union[NDArray[np.bool_], list], Any]:
        return [], None

    def get_material_by_position(self, position: Vector3D, local: bool = False, as_parent: bool = True) -> MaterialArray:
        if not local:
            position = self.convert_to_local_position(position, as_parent)
        material = super().get_material_by_position(position, True, as_parent)
        inside = (material != 0).nonzero()[0]
        indices, new_material = self._parametric_function(position[inside])
        material[inside[indices]] = new_material
        return material
