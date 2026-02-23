import numpy as np
from scipy.interpolate import interp1d
from core.materials.materials import Material, MaterialArray
from hepunits import*


class AttenuationFunction(dict):
    """ Класс функции ослабления"""
        
    def __init__(self, process, attenuation_database, kind='linear'):
        self.__class__.__name__ = self.__class__.__name__ + 'Of' + process.name
        self.__class__.__qualname__ = self.__class__.__qualname__ + 'Of' + process.name 
        for material, attenuation_data in attenuation_database.items():
            energy = np.copy(attenuation_data['Energy'])
            attenuation_coefficient = attenuation_data['Coefficient'][process.name]*material.density
            lower_limit = np.searchsorted(energy, process.energy_range[0], side='left')
            upper_limit = np.searchsorted(energy, process.energy_range[1], side='right')
            energy = energy[lower_limit:upper_limit]
            attenuation_coefficient = attenuation_coefficient[lower_limit:upper_limit]
            attenuation_function = interp1d(energy, attenuation_coefficient, kind)
            self.update({material: attenuation_function})
    
    def __call__(self, material, energy):
        """ Получить линейный коэффициент ослабления """
        mass_coefficient = np.zeros_like(energy)
        if isinstance(material, Material):
            mass_coefficient = self[material](energy)
        elif isinstance(material, MaterialArray):
            for material, indices in material.inverse_indices.items():
                mass_coefficient[indices] = self[material](energy[indices])
        else:
            raise TypeError('Неверный тип')
        return mass_coefficient
    
    