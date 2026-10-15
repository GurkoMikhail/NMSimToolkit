"""
Модуль загрузки и хранения базы данных материалов из HDF5-таблиц NIST.
"""

from collections import namedtuple
from itertools import count
from typing import Dict

from h5py import File
import numpy as np

from core.materials.atomic_properties import element_symbol
from core.materials.materials import Material
from core.other.typing_definitions import Float


class MaterialDataBase(Dict[str, Material]):
    """Словарь-база данных материалов, загружаемых из HDF5-файла."""

    counter = count(1)
    _base_name: str

    def __init__(self, base_name: str = 'NIST Materials') -> None:
        super().__init__()
        self._base_name = base_name
        vacuum_material = Material()
        self.update({vacuum_material.name: vacuum_material})
        self._load_materials()

    @property
    def base_name(self) -> str:
        """Имя используемого файла базы данных материалов."""
        return self._base_name

    @base_name.setter
    def base_name(self, value: str) -> None:
        self._base_name = value
        self._load_materials()

    def _load_materials(self) -> None:
        with File(f'tables/{self._base_name}.h5', 'r') as file:
            for group_name, material_type_group in file.items():
                for material_name, material_group in material_type_group.items():
                    material_type = str(group_name)
                    density = Float(np.copy(material_group['Density']))
                    composition_dict: Dict[str, Float] = {}
                    if group_name == 'Elemental media':
                        atomic_z = int(np.copy(material_group['Z']))
                        composition_dict.update({element_symbol[atomic_z]: Float(1.0)})
                    else:
                        for element, weight in material_group['Composition'].items():
                            composition_dict.update({str(element): Float(np.copy(weight))})
                    if 'Z\\A' in material_group:
                        z_to_a_ratio = Float(np.copy(material_group['Z\\A']))
                    else:
                        z_to_a_ratio = Float(0.5)
                    mat_id = next(self.counter)
                    composition_tuple = namedtuple('composition', list(composition_dict.keys()))(**composition_dict)
                    material = Material(
                        str(material_name),
                        material_type,
                        density,
                        composition_tuple,
                        z_to_a_ratio,
                        mat_id,
                    )
                    self.update({str(material_name): material})


__all__ = ["MaterialDataBase"]


