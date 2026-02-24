from typing import Dict, Iterable, Union

import h5py
import numpy as np
import hepunits as units

from core.materials.materials import Material
from core.other.typing_definitions import Float


class AttenuationDataBase(Dict[Material, np.ndarray]):
    """ Класс базы данных коэффициентов ослабления """
    _base_name: str
    _elements_MAC: Dict[str, np.ndarray]

    def __init__(self, base_name: str = 'NIST XCOM Elements MAC') -> None:
        self._base_name = base_name
        self._elements_MAC = {}
        self._load_elements_MAC()
    
    @property
    def base_name(self) -> str:
        return self._base_name
    
    @base_name.setter
    def base_name(self, value):
        self._base_name = value
        self._load_elements_MAC()

    def _load_elements_MAC(self) -> None:
        file = h5py.File(f'tables/{self._base_name}.h5', 'r')
        for element, element_group in file.items():
            processes_dict = {key: np.copy(value) for key, value in element_group.items()}
            energy = processes_dict.pop('Energy')
            processes_dict = {processes_names[key]: value for key, value in processes_dict.items() if key in processes_names}
            MAC = np.ndarray(energy.size, dtype=MAC_dtype)
            MAC['Energy'] = energy
            for process, value in processes_dict.items():
                MAC['Coefficient'][process] = value
            self._elements_MAC.update({element: MAC})
    
    def add_material(self, material: Union[Material, Iterable[Material]]) -> None:
        if isinstance(material, Material):
            self._add_material(material)
            return
        if isinstance(material, Iterable):
            for mat in material:
                self._add_material(mat)
            return
        raise ValueError('Неверный тип')
    
    def _add_material(self, material: Material) -> None:
        assert isinstance(material, Material), ValueError('Неверный тип')
        list_of_energy = []
        list_of_shells = []
        list_of_MAC = []
        
        for element, weight in material.composition_dict.items():
            energy = np.copy(self._elements_MAC[element]['Energy'])
            MAC = np.copy(self._elements_MAC[element]['Coefficient'])
            for process in MAC.dtype.fields:
                MAC[process] *= weight
            
            _, indices, counts = np.unique(energy, return_index=True, return_counts=True)
            indices_of_shells = indices[counts > 1]
            energy[indices_of_shells] -= 1*units.eV
            
            list_of_energy.append(energy)
            list_of_shells.append(energy[indices_of_shells + 1])
            list_of_MAC.append(MAC)
        
        array_of_energy = np.concatenate(list_of_energy)
        array_of_energy = np.unique(array_of_energy)
        
        array_of_MAC = np.ndarray(array_of_energy.size, dtype=MAC_dtype)
        array_of_MAC['Energy'] = array_of_energy
        
        for process in MAC.dtype.fields:
            array_of_MAC['Coefficient'][process] = 0
            for energy, MAC in zip(list_of_energy, list_of_MAC):
                array_of_MAC['Coefficient'][process] += np.interp(array_of_energy, energy, MAC[process])

        self.update({material: array_of_MAC})
    
    
processes_names = {
    'Photoelectric absorption': 'PhotoelectricEffect',
    'Incoherent scattering': 'ComptonScattering',
    'Coherent scattering': 'CoherentScattering'
}
processes_dtype = np.dtype([(name, Float) for name in processes_names.values()])
MAC_dtype = np.dtype([('Energy', Float), ('Coefficient', processes_dtype)])