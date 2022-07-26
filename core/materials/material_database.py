from collections import namedtuple
from itertools import count
import numpy as np
from h5py import File
from core.materials.atomic_properties import element_symbol
from core.materials.materials import Material, MaterialArray
from hepunits import*


class MaterialDataBase:
    """ Класс базы данных материалов """

    counter = count(1)

    def __init__(
        self,
        path_to_materials_table = 'tables/materials.hdf',
        path_to_table_of_AC = 'tables/attenuationCoefficients.hdf',
        base_name = 'NIST'
        ):
        self.path_to_materials_table = path_to_materials_table
        self.path_to_table_of_AC = path_to_table_of_AC
        self.base_name = base_name
        self.material_array = self._load_materials()

    def __getitem__(self, key):
        if isinstance(key, str):
            index = (self.material_array.name == key).nonzero()[0]
            return self.material_array[index]
        else:
            index = (self.material_array.ID == key).nonzero()[0]
            return self.material_array[index]

    def _load_materials(self):
        materials_file = File(self.path_to_materials_table, 'r')
        base_group = materials_file[self.base_name]
        materials = []
        for group_name, material_type_group in base_group.items():
            for material_name, material_group in material_type_group.items():
                type = group_name
                density = float(np.copy(material_group['Density']))*(g/cm3)
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
                materials.append(material)
        materials = np.asarray(materials, dtype=object)
        return materials.view(MaterialArray)

