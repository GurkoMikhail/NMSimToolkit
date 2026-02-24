from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray


class NonuniqueArray(np.ndarray):
    """ 
    Класс массива неуникальных элементов
    """
    
    element_list: List[Any]

    def __new__(cls, shape: Union[int, Tuple[int, ...]]) -> 'NonuniqueArray':
        obj = super().__new__(cls, shape, dtype=int)
        obj.element_list = [None]
        obj.view(np.ndarray)[:] = 0
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.element_list = copy(getattr(obj, 'element_list', [None]))
        
    def __contains__(self, key: Any) -> bool:
        return key in self.element_list

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NonuniqueArray):
            for element, indices in value.inverse_indices.items():
                indices = np.arange(self.size, dtype=int)[key][indices]
                self[indices] = element
            return
        if value not in self:
            self.element_list.append(value)
        index = self.element_list.index(value)
        self.view(np.ndarray)[key] = index

    def restore(self) -> NDArray[np.object_]:
        indices = np.ndarray.__getitem__(self, slice(None))
        return np.array(self.element_list, dtype=object)[indices]

    def type_matching(self, target_type: Type[Any]) -> NDArray[np.bool_]:
        match = np.zeros_like(self, dtype=bool)
        indices = np.copy(self)
        for index, element in enumerate(self.element_list):
            if isinstance(element, target_type):
                match += indices == index
        return match

    def matching(self, value: Any) -> NDArray[np.bool_]:
        index = self.element_list.index(value)
        indices = np.copy(self)
        return indices == index
    
    @property
    def inverse_indices(self) -> Dict[Any, NDArray[np.int64]]:
        inverse_dict = {}
        indices = np.copy(self)
        for index, element in enumerate(self.element_list):
            match = (indices == index).nonzero()[0]
            if match.size > 0:
                inverse_dict.update({element: match})
        return inverse_dict


    
    