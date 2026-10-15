"""
Модуль базовых и параметрических объемов с трекингом Вудкока (Woodcock tracking).
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
from numba import cfunc
from numpy.typing import NDArray

from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index, NumbaFloat, NumbaIndex, Vector3D


class WoodcockVolume(Volume):
    """
    Базовый класс объема с навигацией методом Вудкока.
    """


class WoodcockParametricVolume(WoodcockVolume):
    """
    Класс параметрического объема с навигацией методом Вудкока.
    Позволяет задавать пространственное распределение материалов аналитической или сеточной функцией @cfunc.
    """

    _cfunc: Optional[Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cfunc = None

    @property
    def material_cfunc(self) -> Callable[[Float, Float, Float], Index]:
        """Указатель ctypes на скомпилированную функцию @cfunc определения материала по локальным координатам."""
        if self._cfunc is None:
            self._cfunc = self._compile_cfunc()
        cfunc_ptr: Callable[[Float, Float, Float], Index] = self._cfunc.ctypes
        return cfunc_ptr

    def invalidate_geometry(self) -> None:
        """Инвалидирует геометрический буфер и сбрасывает кэш скомпилированной функции материала."""
        super().invalidate_geometry()
        self._cfunc = None

    def _compile_cfunc(self) -> Any:
        """
        Переопределяется в наследниках для генерации Numba @cfunc,
        принимающей локальные координаты (pos_x, pos_y, pos_z) и возвращающей ID материала (int64).
        По умолчанию возвращает ID мажорантного материала (self.material).
        """
        material_id = Index(self.material.ID)

        @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
        def default_parametric_func(pos_x: Float, pos_y: Float, pos_z: Float) -> Index:
            return material_id

        return default_parametric_func

    @property
    def material_list(self) -> list[Material]:
        """
        Переопределяется в наследниках, если используются дополнительные материалы.
        По умолчанию возвращает список материалов из базового класса Volume.
        """
        return super().material_list

    def _parametric_function(self, position: Vector3D) -> Tuple[Union[NDArray[np.bool_], list[Any]], Any]:
        """Векторизованная параметрическая функция определения материала по координатам."""
        return [], None


__all__ = [
    "WoodcockVolume",
    "WoodcockParametricVolume",
]

