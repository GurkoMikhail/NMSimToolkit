"""
Модуль параметрических коллиматоров расчетного ядра на основе RayCasting-а (Woodcock Tracking).
Предоставляет универсальный класс ParametricParallelCollimator с поддержкой различных форм каналов
(гексагональные, квадратные) через компилируемые Numba-кернелы.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple
import numpy as np
from numba import cfunc, njit
from numpy.typing import NDArray

import settings.database_setting as settings
from core.geometry.direct_collimators import CollimatorHoleShape
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.woodcock_volumes import WoodcockParametricVolume
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index, Length, NumbaFloat, NumbaIndex, Vector3D
from core.scene.nodes import CompositeNode


class ParametricParallelCollimator(WoodcockParametricVolume):
    """
    Универсальный класс параметрического коллиматора с параллельными каналами.
    Поддерживает гексагональные и квадратные формы каналов с динамическим выбором Numba-кернела.

    Все линейные размеры (size, hole_diameter, septa) задаются в базовых единицах (мм).
    """

    _hole_diameter: Float
    _septa: Float
    _hole_shape: CollimatorHoleShape
    _compiled_material_ids: Optional[Tuple[Index, Index]]

    def __init__(
        self,
        size: Vector3D | Sequence[Length | float],
        hole_diameter: Float | float,
        septa: Float | float,
        material: Optional[Material] = None,
        hole_shape: CollimatorHoleShape = CollimatorHoleShape.HEXAGONAL,
        name: Optional[str] = None,
    ) -> None:
        material_lead: Material = settings.material_database["Pb"] if material is None else material
        size_arr = np.asarray(size, dtype=Float)
        if size_arr.shape != (3,):
            raise ValueError(f"Размер коллиматора должен содержать 3 координаты, получено shape={size_arr.shape}")
        if any(dim <= 0.0 for dim in size_arr):
            raise ValueError(f"Габаритные размеры коллиматора должны быть строго положительными (> 0), получено {size}")
        if Float(hole_diameter) <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {hole_diameter}")
        if Float(septa) <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {septa}")
        if not isinstance(hole_shape, CollimatorHoleShape):
            raise TypeError(f"hole_shape должен быть строго CollimatorHoleShape, получено {type(hole_shape).__name__}")

        self._compiled_material_ids = None
        super().__init__(
            geometry=Box(Float(size_arr[0]), Float(size_arr[1]), Float(size_arr[2])),
            material=material_lead,
            name=name,
        )
        self._hole_diameter = Float(hole_diameter)
        self._septa = Float(septa)
        self._hole_shape = hole_shape

        if self._hole_shape == CollimatorHoleShape.ROUND:
            raise NotImplementedError(
                "Форма каналов 'round' еще не реализована в параметрическом коллиматоре."
            )

        self._compute_constants()

    @property
    def hole_diameter(self) -> Float:
        """Диаметр отверстий коллиматора (мм)."""
        return self._hole_diameter

    @hole_diameter.setter
    def hole_diameter(self, value: Float | float) -> None:
        numeric_value = Float(value)
        if numeric_value <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {value}")
        self._hole_diameter = numeric_value
        self._compute_constants()
        self._cfunc = None

    @property
    def hole_width(self) -> Float:
        """Ширина отверстий коллиматора (мм). Согласована с hole_diameter."""
        return self._hole_diameter

    @hole_width.setter
    def hole_width(self, value: Float | float) -> None:
        self.hole_diameter = value

    @property
    def septa(self) -> Float:
        """Толщина перегородок (септ) между отверстиями (мм)."""
        return self._septa

    @septa.setter
    def septa(self, value: Float | float) -> None:
        numeric_value = Float(value)
        if numeric_value <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {value}")
        self._septa = numeric_value
        self._compute_constants()
        self._cfunc = None

    @property
    def hole_shape(self) -> CollimatorHoleShape:
        """Форма поперечного сечения каналов коллиматора."""
        return self._hole_shape

    @hole_shape.setter
    def hole_shape(self, value: CollimatorHoleShape) -> None:
        if not isinstance(value, CollimatorHoleShape):
            raise TypeError(
                f"hole_shape должен быть строго CollimatorHoleShape, получено: {type(value).__name__}"
            )

        if value == CollimatorHoleShape.ROUND:
            raise NotImplementedError(
                "Форма каналов 'round' еще не реализована в параметрическом коллиматоре."
            )

        self._hole_shape = value
        self._compute_constants()
        self._cfunc = None

    @property
    def parent(self) -> Optional[CompositeNode]:
        """Родительский составной узел в иерархии сцены."""
        return self._parent

    @parent.setter
    def parent(self, value: Optional[CompositeNode]) -> None:
        self._parent = value
        self._cfunc = None

    @property
    def material_cfunc(self) -> Callable[[Float, Float, Float], Index]:
        """Указатель ctypes на скомпилированную функцию @cfunc с проверкой актуальности материала каналов."""
        current_ids = (Index(self.material.ID), Index(self._resolve_hole_material().ID))
        if self._cfunc is not None and self._compiled_material_ids != current_ids:
            self._cfunc = None
        return super().material_cfunc

    def _resolve_hole_material(self) -> Material:
        """
        Динамически определяет материал каналов, поднимаясь по иерархии родительских объемов.
        При отсутствии родителя возвращает Vacuum из базы данных материалов.
        """
        current_ancestor = self.parent
        while current_ancestor is not None:
            if isinstance(current_ancestor, Volume) and current_ancestor.material is not None:
                return current_ancestor.material
            current_ancestor = current_ancestor.parent
        if "Vacuum" in settings.material_database:
            vacuum_mat: Material = settings.material_database["Vacuum"]
            return vacuum_mat
        return Material(name="Vacuum")

    @property
    def material_list(self) -> list[Material]:
        """Список используемых материалов (материал корпуса и материал каналов)."""
        return [self.material, self._resolve_hole_material()]

    def _compute_constants(self) -> None:
        """Пересчитывает геометрические константы периодической решетки каналов."""
        if self._hole_shape == CollimatorHoleShape.HEXAGONAL:
            x_period = self._hole_diameter + self._septa
            y_period = np.sqrt(3.0) * x_period
            self._period = np.stack((x_period, y_period))
            self._hex_slope = np.sqrt(3.0) / 4.0
            hex_diameter = self._hole_diameter * 2.0 / np.sqrt(3.0)
            self._corner = self._period / 2.0
            self._hex_bound = self._hex_slope * hex_diameter
            self._hex_bound_half = self._hex_bound / 2.0
        elif self._hole_shape == CollimatorHoleShape.SQUARE:
            self._square_period = Float(self._hole_diameter + self._septa)
            self._square_half_period = Float(0.5 * self._square_period)
            self._square_half_hole = Float(0.5 * self._hole_diameter)
        elif self._hole_shape == CollimatorHoleShape.ROUND:
            raise NotImplementedError(
                "Форма каналов 'round' еще не реализована в параметрическом коллиматоре."
            )

    def _compile_cfunc(self) -> Any:
        """Компилирует специализированную Numba-функцию трассировки материала по координатам."""
        material_lead_identifier = Index(self.material.ID)
        hole_material_identifier = Index(self._resolve_hole_material().ID)
        self._compiled_material_ids = (material_lead_identifier, hole_material_identifier)

        if self._hole_shape == CollimatorHoleShape.HEXAGONAL:
            period_x = Float(self._period[0])
            period_y = Float(self._period[1])
            corner_x = Float(self._corner[0])
            corner_y = Float(self._corner[1])
            bound_val = Float(self._hex_bound)
            slope_val = Float(self._hex_slope)
            bound_half = Float(self._hex_bound_half)

            @njit(inline="always", cache=True)
            def is_in_hex_cell(delta_x: Float, delta_y: Float) -> bool:
                return bool((delta_x <= bound_val) and (slope_val * delta_y + delta_x / 4.0 <= bound_half))

            @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
            def parametric_func_hex(coordinate_x: Float, coordinate_y: Float, coordinate_z: Float) -> Index:
                mod_x = coordinate_x % period_x
                mod_y = coordinate_y % period_y
                offset_x1 = abs(mod_x - corner_x)
                offset_y1 = abs(mod_y - corner_y)

                if is_in_hex_cell(offset_x1, offset_y1):
                    return hole_material_identifier

                offset_x2 = abs(offset_x1 - corner_x)
                offset_y2 = abs(offset_y1 - corner_y)

                if is_in_hex_cell(offset_x2, offset_y2):
                    return hole_material_identifier

                return material_lead_identifier

            return parametric_func_hex

        elif self._hole_shape == CollimatorHoleShape.SQUARE:
            period_val = Float(self._square_period)
            half_period_val = Float(self._square_half_period)
            half_hole_val = Float(self._square_half_hole)

            @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
            def parametric_func_square(coordinate_x: Float, coordinate_y: Float, coordinate_z: Float) -> Index:
                x_mod = ((coordinate_x + half_period_val) % period_val) - half_period_val
                y_mod = ((coordinate_y + half_period_val) % period_val) - half_period_val
                if abs(x_mod) < half_hole_val and abs(y_mod) < half_hole_val:
                    return hole_material_identifier
                return material_lead_identifier

            return parametric_func_square

        raise NotImplementedError(
            f"Форма каналов '{self._hole_shape.value}' еще не реализована для Numba-компиляции."
        )

    def _parametric_function(self, position: Vector3D) -> Tuple[NDArray[np.bool_], Material]:
        """Векторизованное определение попадания массива точек в каналы коллиматора."""
        effective_hole_material = self._resolve_hole_material()

        if self._hole_shape == CollimatorHoleShape.HEXAGONAL:
            def is_in_hex_vectorized(delta_x: NDArray[Float], delta_y: NDArray[Float]) -> NDArray[np.bool_]:
                res: NDArray[np.bool_] = (delta_x <= self._hex_bound) * (
                    self._hex_slope * delta_y + delta_x / 4.0 <= self._hex_bound_half
                )
                return res

            position_2d = np.mod(position[:, :2], self._period)
            position_2d = np.abs(position_2d - self._corner)
            collimated = is_in_hex_vectorized(position_2d[:, 0], position_2d[:, 1])

            position_remaining = np.abs(position_2d[~collimated] - self._corner)
            collimated[~collimated] = is_in_hex_vectorized(position_remaining[:, 0], position_remaining[:, 1])
            return collimated, effective_hole_material

        elif self._hole_shape == CollimatorHoleShape.SQUARE:
            coordinates_xy = position[:, :2]
            cell_offsets = np.mod(coordinates_xy + self._square_half_period, self._square_period) - self._square_half_period
            is_in_square_hole: NDArray[np.bool_] = (
                (np.abs(cell_offsets[:, 0]) < self._square_half_hole)
                & (np.abs(cell_offsets[:, 1]) < self._square_half_hole)
            )
            return is_in_square_hole, effective_hole_material

        raise NotImplementedError(
            f"Форма каналов '{self._hole_shape.value}' еще не реализована в параметрической функции."
        )


__all__ = [
    "ParametricParallelCollimator",
]


