"""
Модуль базы данных массовых коэффициентов ослабления (MAC) по таблицам NIST XCOM.
"""

from typing import Any, Dict, Iterable

import h5py
import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.materials.materials import Material
from core.other.typing_definitions import Float

processes_names: Dict[str, str] = {
    'Photoelectric absorption': 'PhotoelectricEffect',
    'Incoherent scattering': 'ComptonScattering',
    'Coherent scattering': 'CoherentScattering',
}
processes_dtype = np.dtype([(name, Float) for name in processes_names.values()])
MAC_dtype = np.dtype([('Energy', Float), ('Coefficient', processes_dtype)])


class AttenuationDataBase(Dict[Material, NDArray[Any]]):
    """База данных массовых коэффициентов ослабления для материалов и химических элементов."""

    _base_name: str
    _elements_MAC: Dict[str, NDArray[Any]]

    def __init__(self, base_name: str = 'NIST XCOM Elements MAC') -> None:
        super().__init__()
        self._base_name = base_name
        self._elements_MAC = {}
        self._load_elements_MAC()

    @property
    def base_name(self) -> str:
        """Имя файла HDF5-базы массовых коэффициентов ослабления."""
        return self._base_name

    @base_name.setter
    def base_name(self, value: str) -> None:
        self._base_name = value
        self._load_elements_MAC()

    def _load_elements_MAC(self) -> None:
        with h5py.File(f'tables/{self._base_name}.h5', 'r') as file:
            for element, element_group in file.items():
                processes_dict = {str(key): np.copy(value) for key, value in element_group.items()}
                energy = processes_dict.pop('Energy')
                mapped_processes = {
                    processes_names[key]: value for key, value in processes_dict.items() if key in processes_names
                }
                mac_table: NDArray[Any] = np.ndarray(energy.size, dtype=MAC_dtype)
                mac_table['Energy'] = energy
                for process_name, value in mapped_processes.items():
                    mac_table['Coefficient'][process_name] = value
                self._elements_MAC.update({str(element): mac_table})

    def add_material(self, material: Material) -> None:
        """Добавляет единичный материал в базу данных коэффициентов ослабления."""
        self._add_material(material)

    def add_materials(self, materials: Iterable[Material]) -> None:
        """Пакетно добавляет коллекцию материалов в базу данных."""
        for mat in materials:
            self._add_material(mat)

    def _add_material(self, material: Material) -> None:
        list_of_energy = []
        list_of_shells = []
        list_of_mac = []

        process_field_names = tuple(processes_names.values())

        for element, weight in material.composition_dict.items():
            energy = np.copy(self._elements_MAC[element]['Energy'])
            mac_arr = np.copy(self._elements_MAC[element]['Coefficient'])
            for process_name in process_field_names:
                mac_arr[process_name] *= weight

            _, indices, counts = np.unique(energy, return_index=True, return_counts=True)
            indices_of_shells = indices[counts > 1]
            energy[indices_of_shells] -= 1.0 * units.eV

            list_of_energy.append(energy)
            list_of_shells.append(energy[indices_of_shells + 1])
            list_of_mac.append(mac_arr)

        array_of_energy = np.unique(np.concatenate(list_of_energy))

        array_of_mac: NDArray[Any] = np.ndarray(array_of_energy.size, dtype=MAC_dtype)
        array_of_mac['Energy'] = array_of_energy

        for process_name in process_field_names:
            array_of_mac['Coefficient'][process_name] = 0.0
            for energy, mac_arr in zip(list_of_energy, list_of_mac):
                array_of_mac['Coefficient'][process_name] += np.interp(
                    array_of_energy, energy, mac_arr[process_name]
                )

        self.update({material: array_of_mac})


__all__ = [
    "AttenuationDataBase",
    "processes_names",
    "processes_dtype",
    "MAC_dtype",
]