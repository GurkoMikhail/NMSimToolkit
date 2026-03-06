from typing import Any, Callable, Dict, Union, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Float


@njit(cache=True)
def fast_interp(x_new: NDArray[Float], x: NDArray[Float], y: NDArray[Float]) -> NDArray[Float]:
    """Быстрая интерполяция с помощью numba njit"""
    return np.interp(x_new, x, y)


class AttenuationFunction(Dict[Material, Tuple[NDArray[Float], NDArray[Float]]]):
    """ Класс функции ослабления"""
        
    def __init__(self, process: Any, attenuation_database: Dict[Material, Dict[str, Any]], kind: str = 'linear') -> None:
        if kind != 'linear':
            raise ValueError(f"AttenuationFunction currently only supports 'linear' interpolation, got '{kind}'")

        self.__class__.__name__ = self.__class__.__name__ + 'Of' + process.name
        self.__class__.__qualname__ = self.__class__.__qualname__ + 'Of' + process.name 
        for material, attenuation_data in attenuation_database.items():
            energy = np.copy(attenuation_data['Energy'])
            attenuation_coefficient = attenuation_data['Coefficient'][process.name]*material.density
            lower_limit = np.searchsorted(energy, process.energy_range[0], side='left')
            upper_limit = np.searchsorted(energy, process.energy_range[1], side='right')
            energy = energy[lower_limit:upper_limit]
            attenuation_coefficient = attenuation_coefficient[lower_limit:upper_limit]

            # Сохраняем массивы для быстрой интерполяции вместо объекта scipy
            self.update({material: (np.ascontiguousarray(energy), np.ascontiguousarray(attenuation_coefficient))})
    
    def __call__(self, material: Union[Material, MaterialArray], energy: Union[Float, NDArray[Float]]) -> Union[Float, NDArray[Float]]:
        """ Получить линейный коэффициент ослабления """
        if isinstance(material, Material):
            x, y = self[material]
            if np.isscalar(energy):
                return np.interp(energy, x, y)
            else:
                return fast_interp(np.asarray(energy, dtype=Float), x, y)

        mass_coefficient = np.zeros_like(energy, dtype=Float)
        if isinstance(material, MaterialArray):
            for mat, indices in material.inverse_indices.items():
                x, y = self[mat]
                mass_coefficient[indices] = fast_interp(np.asarray(energy[indices], dtype=Float), x, y)
        else:
            raise TypeError('Неверный тип')
        return mass_coefficient
    
    