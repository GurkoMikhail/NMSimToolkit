from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List, Tuple

import numpy as np
import hepunits as units
from numpy.typing import NDArray

from core.materials.atomic_properties import atomic_number
from core.other.nonunique_array import NonuniqueArray
from core.other.typing_definitions import Float


Composition = namedtuple('Composition', ['H'])

@dataclass(eq=True, frozen=True)
class Material:
    """ Класс материала """
    name: str = 'Vacuum'
    type: str = ''
    density: Float = 0.4 * (10 ** (-29)) * units.g / units.cm3
    composition: Any = Composition(H=1.)
    ZtoA_ratio: Float = 0.
    ID: int = 0

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Material):
            return self.Zeff*self.density < other.Zeff*other.density
        return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Material):
            return self.Zeff*self.density <= other.Zeff*other.density
        return False

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Material):
            return self.Zeff*self.density > other.Zeff*other.density
        return True

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, Material):
            return self.Zeff*self.density >= other.Zeff*other.density
        return True

    @property
    @cache
    def Zeff(self) -> Float:
        Zeff = 0.
        for element, weight in self.composition_dict.items():
            Zeff += atomic_number[element] * weight
        return Zeff
    
    @property
    @cache
    def composition_dict(self) -> Dict[str, Float]:
        return self.composition._asdict()
    
    @property
    @cache
    def composition_array(self) -> NDArray[Float]:
        composition_array = np.zeros(shape=93, dtype=Float)
        for element, weight in self.composition_dict.items():
            Z = atomic_number[element]
            composition_array[Z] = weight
        return composition_array


class MaterialArray(NonuniqueArray):
    """ 
    Класс массива материалов
    """
    
    def __new__(cls, shape: Tuple[int, ...]) -> 'MaterialArray':
        obj = super().__new__(cls, shape)
        obj.element_list = [Material(), ]
        return obj
    
    @property
    def material_list(self) -> List[Material]:
        return self.element_list
    
    @property
    def Zeff(self) -> NDArray[Float]:
        Zeff = np.zeros_like(self, dtype=Float)
        for material, indices in self.inverse_indices.items():
            Zeff[indices] = material.Zeff
        return Zeff

    @property
    def density(self) -> NDArray[Float]:
        density = np.zeros_like(self, dtype=Float)
        for material, indices in self.inverse_indices.items():
            density[indices] = material.density
        return density

