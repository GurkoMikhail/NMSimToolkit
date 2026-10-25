"""
Модуль функций линейного коэффициента ослабления (LAC) материалов с быстрой интерполяцией Numba.
"""

from typing import Any, Dict, Tuple, Union, overload

import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float


@njit(cache=True, inline='always')
def fast_interp_scalar(query_point: Float, grid_x: NDArray[Float], grid_y: NDArray[Float]) -> Float:
    """
    Кусочная log-log интерполяция скалярного значения энергии с право-непрерывным
    переходом на парах узлов краев поглощения.
    """
    length = len(grid_x)
    if length == 0:
        return Float(0.0)
    if query_point < grid_x[0]:
        return Float(grid_y[0])
    if query_point >= grid_x[length - 1]:
        return Float(grid_y[length - 1])

    idx = int(np.searchsorted(grid_x, query_point, side='right'))
    if idx <= 0:
        return Float(grid_y[0])
    if idx >= length:
        return Float(grid_y[length - 1])

    e0 = grid_x[idx - 1]
    e1 = grid_x[idx]
    v0 = grid_y[idx - 1]
    v1 = grid_y[idx]

    if query_point <= e0 or e1 <= e0:
        return Float(v0)

    if v0 > 0.0 and v1 > 0.0 and e0 > 0.0:
        log_fraction = (np.log(query_point) - np.log(e0)) / (np.log(e1) - np.log(e0))
        log_v0 = np.log(v0)
        log_v1 = np.log(v1)
        return Float(np.exp(log_v0 + log_fraction * (log_v1 - log_v0)))

    lin_fraction = (query_point - e0) / (e1 - e0)
    return Float(v0 + lin_fraction * (v1 - v0))


@njit(cache=True)
def fast_interp(query_points: NDArray[Float], grid_x: NDArray[Float], grid_y: NDArray[Float]) -> NDArray[Float]:
    """Быстрая кусочная log-log интерполяция массивов с учетом краев поглощения с помощью Numba JIT."""
    count = len(query_points)
    result = np.empty(count, dtype=Float)
    for i in range(count):
        result[i] = fast_interp_scalar(query_points[i], grid_x, grid_y)
    return result


class AttenuationFunction(Dict[Material, Tuple[NDArray[Float], NDArray[Float]]]):
    """
    Таблица и функция интерполяции линейных коэффициентов ослабления (LAC) для заданного физического процесса.
    """

    name: str

    def __init__(self, process: Any, attenuation_database: Dict[Material, Any], kind: str = 'log-log') -> None:
        super().__init__()
        if kind != 'log-log':
            raise ValueError(
                f"AttenuationFunction поддерживает только кусочную log-log интерполяцию ('log-log'), получено '{kind}'."
            )

        energy_range = np.asarray(process.energy_range, dtype=Float)
        if energy_range.shape != (2,) or energy_range[0] <= 0.0 or energy_range[0] >= energy_range[1]:
            raise ValueError(
                f"Некорректный энергетический диапазон процесса '{process.name}': {energy_range}."
            )

        self.name = f"{self.__class__.__name__}Of{process.name}"
        for material, attenuation_data in attenuation_database.items():
            energy = np.copy(attenuation_data['Energy'])
            attenuation_coefficient = attenuation_data['Coefficient'][process.name] * material.density
            lower_limit = int(np.searchsorted(energy, energy_range[0], side='left'))
            upper_limit = int(np.searchsorted(energy, energy_range[1], side='right'))
            energy = energy[lower_limit:upper_limit]
            attenuation_coefficient = attenuation_coefficient[lower_limit:upper_limit]

            if energy.size < 2:
                raise ValueError(
                    f"После обрезки по диапазону {energy_range} сетка материала '{material.name}' "
                    f"содержит менее 2 узлов ({energy.size})."
                )

            self.update({material: (np.ascontiguousarray(energy), np.ascontiguousarray(attenuation_coefficient))})

    @overload
    def __call__(self, material: Material, energy: Union[float, Float]) -> Float: ...

    @overload
    def __call__(self, material: Union[Material, MaterialArray], energy: NDArray[Float]) -> NDArray[Float]: ...

    def __call__(
        self,
        material: Union[Material, MaterialArray],
        energy: Union[float, Float, NDArray[Float]],
    ) -> Union[Float, NDArray[Float]]:
        """Вычисляет линейный коэффициент ослабления (LAC) для заданного материала и энергии."""
        if isinstance(material, Material):
            grid_x, grid_y = self[material]
            if isinstance(energy, np.ndarray):
                return fast_interp(np.asarray(energy, dtype=Float), grid_x, grid_y)
            return Float(fast_interp_scalar(Float(energy), grid_x, grid_y))

        if isinstance(material, MaterialArray):
            energy_arr = np.asarray(energy, dtype=Float)
            linear_coefficient = np.zeros_like(energy_arr, dtype=Float)
            for mat, indices in material.inverse_indices.items():
                grid_x, grid_y = self[mat]
                linear_coefficient[indices] = fast_interp(
                    np.asarray(energy_arr[indices], dtype=Float), grid_x, grid_y
                )
            return linear_coefficient

        raise TypeError("Неверный тип аргумента material: ожидается Material или MaterialArray.")


__all__ = [
    "fast_interp_scalar",
    "fast_interp",
    "AttenuationFunction",
]