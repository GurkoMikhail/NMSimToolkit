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
    composition_dict: namedtuple = ()
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
    def composition(self):
        composition = np.zeros(shape=93, dtype=float)
        for element, weight in self.composition_dict._asdict().items():
            Z = atomic_number[element]
            composition[Z] = weight
        return composition


class MaterialArray(np.ndarray):
    """ 
    Класс массива материалов

    [name] = mm\n
    [type] = mm\n
    [density] = MeV\n
    [composition] = ns\n
    [ZtoA_ratio] = mm\n
    """
    
    attributes = ['name', 'type', 'density', 'composition', 'composition_dict', 'Zeff', 'ZtoA_ratio', 'ID']
    
    def __new__(cls, shape):
        obj = super().__new__(cls, shape, dtype=object)
        obj[...] = Material()
        return obj        

    def __getattr__(self, name):
        if name in self.attributes:
            return np.asarray([getattr(item, name) for item in self])
        raise AttributeError()
    
    def __setattr__(self, name, value):
        if name in self.attributes:
            raise AttributeError('Unchangeable attribute')
        else:
            super().__setattr__(name, value)

