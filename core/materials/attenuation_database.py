import numpy as np
from h5py import File


class AttenuationDataBase(dict):
    """ Класс базы данных коэффициентов ослабления """
    
    def __init__(self, base_name = 'NIST XCOM Elements MAC'):
        self._base_name = base_name
        self._load_elements_MAC()
    
    @property
    def base_name(self):
        return self._base_name
    
    @base_name.setter
    def base_name(self, value):
        self._base_name = value
        self._load_elements_MAC()

    def _load_elements_MAC(self):
        file = File(f'tables/{self._base_name}.h5', 'r')
        for element, element_group in file.items():
            keys = element_group.keys()
            values = [np.copy(value) for value in element_group.values()]
            MAC = np.asarray(values, dtype=np.dtype([(key, 'f') for key in keys]))
            self.update({element: MAC})
            
            