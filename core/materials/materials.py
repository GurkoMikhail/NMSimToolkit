from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from core.materials.atomic_properties import atomic_number
import numpy as np
from hepunits import*


@dataclass(eq=True, frozen=True)
class Material:
    """ Класс материала """
    name: str = 'Vacuum'
    type: str = ''
    density: float = 0.
    composition: namedtuple = namedtuple('composition', [])()
    ZtoA_ratio: float = 0.
    ID: int = 0

    def __lt__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density < other.Zeff*other.density
        return False

    def __le__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density <= other.Zeff*other.density
        return False

    def __gt__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density > other.Zeff*other.density
        return True

    def __ge__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density >= other.Zeff*other.density
        return True

    @property
    @cache
    def Zeff(self):
        Zeff = 0
        for element, weight in self.composition._asdict().items():
            Zeff += atomic_number[element]*weight
        return Zeff
    
    @property
    @cache
    def composition_array(self):
        composition_array = np.zeros(shape=93, dtype=float)
        for element, weight in self.composition._asdict().items():
            Z = atomic_number[element]
            composition_array[Z] = weight
        return composition_array


class MaterialArray(np.ndarray):
    """ 
    Класс массива материалов

    [name] = mm\n
    [type] = mm\n
    [density] = MeV\n
    [composition] = ns\n
    [ZtoA_ratio] = mm\n
    """
    
    def __new__(cls, shape):
        obj = super().__new__(cls, shape, dtype=int)
        obj.material_list = np.array([Material(), ], dtype=object)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.material_list = getattr(obj, 'material_list')
        
    def __contains__(self, key):
        return key in self.material_list

    def __getitem__(self, key):
        indices = np.ndarray.__getitem__(self, key)
        return self.material_list[indices]
    
    def __setitem__(self, key, value):
        if value not in self:
            self.material_list = np.append(self.material_list, value)
        index = (self.material_list == value).nonzero()[0]
        self.view(np.ndarray)[key] = index

    @property
    def indices(self):
        return np.copy(self)
    
    @property
    def inverse_indices(self):
        inverse_dict = {}
        indices = self.indices
        for index, material in enumerate(self.material_list):
            match = (indices == index).nonzero()[0]
            inverse_dict.update({material.name: match})
        return inverse_dict

