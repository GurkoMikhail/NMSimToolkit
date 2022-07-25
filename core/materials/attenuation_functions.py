import numpy as np
from scipy.interpolate import interp1d
from core.materials.atomic_properties import atomic_number
from h5py import File
from hepunits import*


class AttenuationFunction(interp1d):
    """ Класс функции ослабления"""
    
    def __init__(self, energy, MAC, kind='linear'):
        super().__init__(energy, MAC, kind)
        
    def get_mass_coefficient(self, material, energy):
        """ Получить массовый коэффициент ослабления """
        # return (self(energy).T*material.composition).sum(axis=1)
        return np.average(self(energy).T, axis=1, weights=material.composition)
    
    def get_linear_coefficient(self, material, energy):
        """ Получить линейный коэффициент ослабления """
        return self.get_mass_coefficient(material, energy)*material.density

    @classmethod
    def construct(cls, process, material_database, kind='linear'):
        energy, MAC, shell_energy = load_elements_MAC(process, material_database)
        cls.__name__ = cls.__name__ + 'Of' + process.name
        cls.__qualname__ = cls.__qualname__ + 'Of' + process.name 
        return cls(energy, MAC, kind)


def load_elements_MAC(process, material_database):
    file_of_AC = File(material_database.path_to_table_of_AC, 'r')
    group_of_MAC = file_of_AC[f'{material_database.base_name}/Mass attenuation coefficients/Elemental media']
    process_name = processes_names[process.name]
    
    list_of_energy = [[]]*93
    list_of_shells = [[]]*93
    list_of_MAC = [[]]*93
    
    for element_name, element_group in group_of_MAC.items():
        Z = atomic_number[element_name]
        energy = np.array(element_group['Energy'])
        MAC = np.array(element_group[process_name])*(cm2/g)
        _, indices, counts = np.unique(energy, return_index=True, return_counts=True)
        indices_of_shells = indices[counts > 1]
        list_of_shells[Z] = energy[indices_of_shells]
        energy[indices_of_shells] -= 1*eV
        
        list_of_energy[Z] = energy
        list_of_MAC[Z] = MAC
        
    file_of_AC.close()
    
    array_of_energy = np.concatenate(list_of_energy)
    array_of_energy = np.unique(array_of_energy)
    
    min_side = np.searchsorted(array_of_energy, process.energy_range[0], side='left')
    max_side = np.searchsorted(array_of_energy, process.energy_range[1], side='right')
    
    array_of_energy = array_of_energy[min_side:max_side]
    
    array_of_MAC = np.zeros((93, array_of_energy.size), float)
    for energy, MAC, Z in zip(list_of_energy, list_of_MAC, range(94)):
        if Z == 0:
            continue
        array_of_MAC[Z] = np.interp(array_of_energy, energy, MAC)
    
    return array_of_energy, array_of_MAC, list_of_shells


processes_names = {
    'PhotoelectricEffect':  'Photoelectric absorption',
    'ComptonScattering':    'Incoherent scattering',
    'CoherentScattering':   'Coherent scattering'
}

