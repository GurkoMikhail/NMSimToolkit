from functools import cache
from itertools import count
import numpy as np
from h5py import File
from core.materials.atomic_properties import atomic_number
from core.materials.materials import MaterialArray
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

    @cache
    def __getitem__(self, key):
        if isinstance(key, str):
            index = (self.material_array.name == key).nonzero()[0]
            return self.material_array[index]
        else:
            index = (self.material_array.ID == key).nonzero()[0]
            return self.material_array[index]

    def _load_materials(self):
        self.materialsDict = {}
        materials_file = File(self.path_to_materials_table, 'r')
        base_group = materials_file[self.base_name]
        material_names = []
        types = []
        densities = []
        compositions = []
        ZtoA_ratios = []
        IDs = []
        for group_name, material_type_group in base_group.items():
            for material_name, material_group in material_type_group.items():
                type = group_name
                density = float(np.copy(material_group['Density']))*(g/cm3)
                composition = np.zeros(93, dtype=float)
                if group_name == 'Elemental media':
                    Z = int(np.copy(material_group['Z']))
                    composition[Z] = 1.
                else:
                    for element, weight in material_group['Composition'].items():
                        Z = atomic_number[element]
                        composition[Z] = float(np.copy(weight))
                try:
                    ZtoAratio = float(np.copy(material_group['Z\\A']))
                except:
                    # print(f'Для {material_name} отсутствует Z\\A')
                    ZtoAratio = 0.5
                ID = next(self.counter)
        
                material_names.append(material_name)
                types.append(type)
                densities.append(density)
                compositions.append(composition)
                ZtoA_ratios.append(ZtoAratio)
                IDs.append(ID)
        material_names = np.asarray(material_names)
        types = np.asarray(types)
        densities = np.asarray(densities)
        compositions = np.asarray(compositions)
        ZtoA_ratios = np.asarray(ZtoA_ratios)
        IDs = np.asarray(IDs)
        return MaterialArray.create(material_names, types, densities, compositions, ZtoA_ratios, IDs)

