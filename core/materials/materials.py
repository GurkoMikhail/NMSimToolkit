from collections import namedtuple
from copy import copy
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
    density: float = 0.4*(10**(-29))*g/cm3
    composition: namedtuple = namedtuple('composition', ['H'])(H=1.)
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
        for element, weight in self.composition_dict.items():
            Zeff += atomic_number[element]*weight
        return Zeff
    
    @property
    @cache
    def composition_dict(self):
        return self.composition._asdict()
    
    @property
    @cache
    def composition_array(self):
        composition_array = np.zeros(shape=93, dtype=float)
        for element, weight in self.composition_dict.items():
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
        obj.material_list = [Material(), ]
        obj.view(np.ndarray)[:] = 0
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.material_list = copy(getattr(obj, 'material_list'))
        
    def __contains__(self, key):
        return key in self.material_list

    def __setitem__(self, key, value):
        if isinstance(value, MaterialArray):
            for material, indices in value.inverse_indices.items():
                self[indices] = material
            return
        if value not in self:
            self.material_list.append(value)
        index = self.material_list.index(value)
        self.view(np.ndarray)[key] = index

    def restore(self):
        indices = np.ndarray.__getitem__(self, slice(None))
        return np.array(self.material_list, dtype=object)[indices]

    @property
    def Zeff(self):
        Zeff = np.zeros_like(self, dtype=float)
        for material, indices in self.inverse_indices.items():
            Zeff[indices] = material.Zeff
        return Zeff

    @property
    def density(self):
        density = np.zeros_like(self, dtype=float)
        for material, indices in self.inverse_indices.items():
            density[indices] = material.density
        return density

    @property
    def indices(self):
        return np.copy(self)
    
    @property
    def inverse_indices(self):
        inverse_dict = {}
        indices = self.indices
        for index, material in enumerate(self.material_list):
            match = (indices == index).nonzero()[0]
            inverse_dict.update({material: match})
        return inverse_dict

