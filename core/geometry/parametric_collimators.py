"""
Модуль параметрических коллиматоров расчетного ядра на основе RayCasting-а (Woodcock Tracking).
Предоставляет универсальный класс ParametricParallelCollimator с поддержкой различных форм каналов
(гексагональные, квадратные) через компилируемые Numba-кернелы.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numba import cfunc, njit

import settings.database_setting as settings
from core.geometry.direct_collimators import CollimatorHoleShape
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.woodcock_volumes import WoodcockParametricVolume
from core.materials.materials import Material
from core.other.typing_definitions import Float, NumbaFloat, NumbaIndex, Vector3D


class ParametricParallelCollimator(WoodcockParametricVolume):
    """
    Универсальный класс параметрического коллиматора с параллельными каналами.
    Поддерживает гексагональные и квадратные формы каналов с динамическим выбором Numba-кернела.

    [origin = (x, y, z)] = units.mm
    [size = (dx, dy, dz)] = units.mm
    [hole_diameter] = units.mm
    [septa] = units.mm
    """

    _hole_diameter: Float
    _septa: Float
    _hole_shape: CollimatorHoleShape

    def __init__(
        self,
        size: Union[np.ndarray, list, tuple],
        hole_diameter: Float,
        septa: Float,
        material: Optional[Material] = None,
        hole_shape: Union[CollimatorHoleShape, str] = CollimatorHoleShape.HEXAGONAL,
        name: Optional[str] = None,
    ) -> None:
        material_lead = settings.material_database["Pb"] if material is None else material
        super().__init__(
            geometry=Box(*size),
            material=material_lead,
            name=name,
        )
        self._hole_diameter = Float(hole_diameter)
        self._septa = Float(septa)

        if isinstance(hole_shape, str):
            try:
                self._hole_shape = CollimatorHoleShape(hole_shape.lower())
            except ValueError:
                valid_shapes = [shape_item.value for shape_item in CollimatorHoleShape]
                raise ValueError(
                    f"Неизвестная форма канала '{hole_shape}'. Допустимые формы: {valid_shapes}"
                )
        elif isinstance(hole_shape, CollimatorHoleShape):
            self._hole_shape = hole_shape
        else:
            raise TypeError(
                f"hole_shape должен быть CollimatorHoleShape или str, получено: {type(hole_shape)}"
            )

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
    def hole_diameter(self, value: Float) -> None:
        numeric_value = Float(value)
        if numeric_value <= 0.0:
            raise ValueError(f"Диаметр отверстия должен быть строго положительным (> 0), получено {value}")
        self._hole_diameter = numeric_value
        self._compute_constants()
        self.invalidate_geometry()

    @property
    def hole_width(self) -> Float:
        """Ширина отверстий коллиматора (мм). Согласована с hole_diameter."""
        return self._hole_diameter

    @hole_width.setter
    def hole_width(self, value: Float) -> None:
        self.hole_diameter = value

    @property
    def septa(self) -> Float:
        """Толщина перегородок (септ) между отверстиями (мм)."""
        return self._septa

    @septa.setter
    def septa(self, value: Float) -> None:
        numeric_value = Float(value)
        if numeric_value <= 0.0:
            raise ValueError(f"Толщина септы должна быть строго положительной (> 0), получено {value}")
        self._septa = numeric_value
        self._compute_constants()
        self.invalidate_geometry()

    @property
    def hole_shape(self) -> CollimatorHoleShape:
        """Форма поперечного сечения каналов коллиматора."""
        return self._hole_shape

    @hole_shape.setter
    def hole_shape(self, value: Union[CollimatorHoleShape, str]) -> None:
        if isinstance(value, str):
            try:
                parsed_shape = CollimatorHoleShape(value.lower())
            except ValueError:
                valid_shapes = [shape_item.value for shape_item in CollimatorHoleShape]
                raise ValueError(
                    f"Неизвестная форма канала '{value}'. Допустимые формы: {valid_shapes}"
                )
        elif isinstance(value, CollimatorHoleShape):
            parsed_shape = value
        else:
            raise TypeError(
                f"hole_shape должен быть CollimatorHoleShape или str, получено: {type(value)}"
            )

        if parsed_shape == CollimatorHoleShape.ROUND:
            raise NotImplementedError(
                "Форма каналов 'round' еще не реализована в параметрическом коллиматоре."
            )

        self._hole_shape = parsed_shape
        self._compute_constants()
        self.invalidate_geometry()

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
            return settings.material_database["Vacuum"]
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
            self._square_period = self._hole_diameter + self._septa
            self._square_half_period = 0.5 * self._square_period
            self._square_half_hole = 0.5 * self._hole_diameter
        elif self._hole_shape == CollimatorHoleShape.ROUND:
            raise NotImplementedError(
                "Форма каналов 'round' еще не реализована в параметрическом коллиматоре."
            )

    def _compile_cfunc(self):
        """Компилирует специализированную Numba-функцию трассировки материала по координатам."""
        material_lead_identifier = self.material.ID
        hole_material_identifier = self._resolve_hole_material().ID

        if self._hole_shape == CollimatorHoleShape.HEXAGONAL:
            period_x = Float(self._period[0])
            period_y = Float(self._period[1])
            corner_x = Float(self._corner[0])
            corner_y = Float(self._corner[1])
            bound_val = Float(self._hex_bound)
            slope_val = Float(self._hex_slope)
            bound_half = Float(self._hex_bound_half)

            @njit(inline="always", cache=True)
            def is_in_hex_cell(delta_x, delta_y):
                return (delta_x <= bound_val) and (slope_val * delta_y + delta_x / 4.0 <= bound_half)

            @cfunc(NumbaIndex(NumbaFloat, NumbaFloat, NumbaFloat), cache=True)
            def parametric_func_hex(coordinate_x, coordinate_y, coordinate_z):
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
            def parametric_func_square(coordinate_x, coordinate_y, coordinate_z):
                offset_ux = (coordinate_x % period_val) - half_period_val
                offset_uy = (coordinate_y % period_val) - half_period_val
                if abs(offset_ux) <= half_hole_val and abs(offset_uy) <= half_hole_val:
                    return hole_material_identifier
                return material_lead_identifier

            return parametric_func_square

        raise NotImplementedError(
            f"Форма каналов '{self._hole_shape.value}' еще не реализована для Numba-компиляции."
        )

    def _parametric_function(self, position: Vector3D) -> Tuple[np.ndarray, Material]:
        """Векторизованное определение попадания массива точек в каналы коллиматора."""
        effective_hole_material = self._resolve_hole_material()

        if self._hole_shape == CollimatorHoleShape.HEXAGONAL:
            def is_in_hex_vectorized(delta_x, delta_y):
                return (delta_x <= self._hex_bound) * (self._hex_slope * delta_y + delta_x / 4.0 <= self._hex_bound_half)

            position_2d = np.mod(position[:, :2], self._period)
            position_2d = np.abs(position_2d - self._corner)
            collimated = is_in_hex_vectorized(position_2d[:, 0], position_2d[:, 1])

            position_remaining = np.abs(position_2d[~collimated] - self._corner)
            collimated[~collimated] = is_in_hex_vectorized(position_remaining[:, 0], position_remaining[:, 1])
            return collimated, effective_hole_material

        elif self._hole_shape == CollimatorHoleShape.SQUARE:
            coordinates_xy = position[:, :2]
            cell_offsets = np.mod(coordinates_xy, self._square_period) - self._square_half_period
            is_in_square_hole = (np.abs(cell_offsets[:, 0]) <= self._square_half_hole) & (np.abs(cell_offsets[:, 1]) <= self._square_half_hole)
            return is_in_square_hole, effective_hole_material

        raise NotImplementedError(
            f"Форма каналов '{self._hole_shape.value}' еще не реализована в параметрической функции."
        )

__all__ = [
    "ParametricParallelCollimator",
]

