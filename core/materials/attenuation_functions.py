"""
Модуль функций линейного коэффициента ослабления (LAC) материалов с быстрой интерполяцией Numba.
"""

from typing import Any, Dict, Tuple, Union, overload

import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float


@njit(cache=True)
def fast_interp(query_points: NDArray[Float], grid_x: NDArray[Float], grid_y: NDArray[Float]) -> NDArray[Float]:
    """Быстрая линейная интерполяция массивов с помощью Numba JIT."""
    return np.interp(query_points, grid_x, grid_y)


class AttenuationFunction(Dict[Material, Tuple[NDArray[Float], NDArray[Float]]]):
    """
    Таблица и функция интерполяции линейных коэффициентов ослабления (LAC) для заданного физического процесса.
    """

    name: str

    def __init__(self, process: Any, attenuation_database: Dict[Material, Any], kind: str = 'linear') -> None:
        super().__init__()
        if kind != 'linear':
            raise ValueError(
                f"AttenuationFunction поддерживает только линейную интерполяцию ('linear'), получено '{kind}'."
            )

        self.name = f"{self.__class__.__name__}Of{process.name}"
        for material, attenuation_data in attenuation_database.items():
            energy = np.copy(attenuation_data['Energy'])
            attenuation_coefficient = attenuation_data['Coefficient'][process.name] * material.density
            lower_limit = int(np.searchsorted(energy, process.energy_range[0], side='left'))
            upper_limit = int(np.searchsorted(energy, process.energy_range[1], side='right'))
            energy = energy[lower_limit:upper_limit]
            attenuation_coefficient = attenuation_coefficient[lower_limit:upper_limit]

            # Сохраняем непрерывные массивы для быстрой линейной интерполяции
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
            return Float(np.interp(float(energy), grid_x, grid_y))

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
    "fast_interp",
    "AttenuationFunction",
]

    
    