from copy import copy
import numpy as np
    
    
class NonuniqueArray(np.ndarray):
    """ 
    Класс массива неуникальных элементов
    """
    
    def __new__(cls, shape):
        obj = super().__new__(cls, shape, dtype=int)
        obj.element_list = [None, ]
        obj.view(np.ndarray)[:] = 0
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.element_list = copy(getattr(obj, 'element_list'))
        
    def __contains__(self, key):
        return key in self.element_list

    def __setitem__(self, key, value):
        if isinstance(value, NonuniqueArray):
            for element, indices in value.inverse_indices.items():
                self[indices] = element
            return
        if value not in self:
            self.element_list.append(value)
        index = self.element_list.index(value)
        self.view(np.ndarray)[key] = index

    def restore(self):
        indices = np.ndarray.__getitem__(self, slice(None))
        return np.array(self.element_list, dtype=object)[indices]

    def type_matching(self, type):
        match = np.zeros_like(self, dtype=bool)
        indices = np.copy(self)
        for index, element in enumerate(self.element_list):
            if isinstance(element, type):
                match += indices == index
        return match

    def matching(self, value):
        index = self.element_list.index(value)
        indices = np.copy(self)
        return indices == index
    
    @property
    def inverse_indices(self):
        inverse_dict = {}
        indices = np.copy(self)
        for index, element in enumerate(self.element_list):
            match = (indices == index).nonzero()[0]
            inverse_dict.update({element: match})
        return inverse_dict


    
    