from collections import namedtuple
from itertools import count
import numpy as np
from h5py import File
from core.materials.atomic_properties import element_symbol
from core.materials.materials import Material
from hepunits import*


class MaterialDataBase(dict):
    """ Класс базы данных материалов """

    counter = count(1)

    def __init__(self, base_name = 'NIST Materials'):
        self._base_name = base_name
        material = Material()
        self.update({material.name: material})
        self._load_materials()
        
    @property
    def base_name(self):
        return self._base_name
    
    @base_name.setter
    def base_name(self, value):
        self._base_name = value
        self._load_materials()

    def _load_materials(self):
        file = File(f'tables/{self._base_name}.h5', 'r')
        for group_name, material_type_group in file.items():
            for material_name, material_group in material_type_group.items():
                type = group_name
                density = float(np.copy(material_group['Density']))
                composition_dict = {}
                if group_name == 'Elemental media':
                    Z = int(np.copy(material_group['Z']))
                    composition_dict.update({element_symbol[Z]: 1.})
                else:
                    for element, weight in material_group['Composition'].items():
                        composition_dict.update({element: float(np.copy(weight))})
                try:
                    ZtoA_ratio = float(np.copy(material_group['Z\\A']))
                except:
                    # print(f'Для {material_name} отсутствует Z\\A')
                    ZtoA_ratio = 0.5
                ID = next(self.counter)
                composition_dict = namedtuple('composition', composition_dict)(**composition_dict)
                material = Material(material_name, type, density, composition_dict, ZtoA_ratio, ID)
                self.update({material_name: material})

