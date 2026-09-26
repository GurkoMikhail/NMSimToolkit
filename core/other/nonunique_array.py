from copy import copy
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ['NonuniqueArray']


class NonuniqueArray(np.ndarray):
    """
    Класс массива неуникальных элементов с индексированным доступом.

    Отображает массив целочисленных индексов на список уникальных объектов
    с эффективным кэшированием обратного соответствия через хэш-словарь O(1).
    """

    _element_list: List[Any]
    _element_to_index: Dict[Any, int]

    def __new__(cls, shape: Union[int, Tuple[int, ...]]) -> 'NonuniqueArray':
        obj = super().__new__(cls, shape, dtype=int)
        obj._element_list = [None]
        obj._element_to_index = {None: 0}
        obj.view(np.ndarray)[:] = 0
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, NonuniqueArray):
            self._element_list = copy(obj._element_list)
            self._element_to_index = copy(obj._element_to_index)
        else:
            self._element_list = [None]
            self._element_to_index = {None: 0}

    @property
    def element_list(self) -> List[Any]:
        """Список уникальных элементов массива."""
        return self._element_list

    @element_list.setter
    def element_list(self, new_list: List[Any]) -> None:
        self._element_list = list(new_list)
        self._element_to_index = {}
        for element_idx, element in enumerate(self._element_list):
            try:
                self._element_to_index[element] = element_idx
            except TypeError:
                pass

    def _get_or_add_index(self, element: Any) -> int:
        """Получает существующий индекс элемента или добавляет новый за O(1)."""
        try:
            if element in self._element_to_index:
                return self._element_to_index[element]
            element_idx = len(self._element_list)
            self._element_list.append(element)
            self._element_to_index[element] = element_idx
            return element_idx
        except TypeError:
            if element not in self._element_list:
                self._element_list.append(element)
            return self._element_list.index(element)

    def __contains__(self, key: Any) -> bool:
        try:
            if key in self._element_to_index:
                return True
        except TypeError:
            pass
        return key in self._element_list

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NonuniqueArray):
            target_view = self.view(np.ndarray)[key]
            for element, indices in value.inverse_indices.items():
                element_idx = self._get_or_add_index(element)
                target_view[indices] = element_idx
            return
        element_idx = self._get_or_add_index(value)
        self.view(np.ndarray)[key] = element_idx

    def restore(self) -> NDArray[np.object_]:
        """Восстанавливает массив объектов исходных типов по сохраненным индексам."""
        indices = np.ndarray.__getitem__(self, slice(None))
        return np.array(self._element_list, dtype=object)[indices]

    def type_matching(self, target_type: Type[Any]) -> NDArray[np.bool_]:
        """Возвращает булеву маску совпадения элементов массива с заданным типом."""
        match_mask = np.zeros(self.shape, dtype=bool)
        indices = np.copy(self)
        for element_idx, element in enumerate(self._element_list):
            if isinstance(element, target_type):
                match_mask |= (indices == element_idx)
        return match_mask

    def matching(self, value: Any) -> NDArray[np.bool_]:
        """Возвращает булеву маску совпадения элементов массива с заданным значением."""
        try:
            element_idx = self._element_to_index.get(value)
        except TypeError:
            element_idx = None
        if element_idx is None:
            element_idx = self._element_list.index(value)
        indices = np.copy(self)
        return indices == element_idx

    @property
    def inverse_indices(self) -> Dict[Any, Tuple[NDArray[np.int64], ...]]:
        """Словарь индексов вхождений каждого уникального элемента в массиве."""
        inverse_dict = {}
        indices = np.copy(self)
        for element_idx, element in enumerate(self._element_list):
            match_coords = (indices == element_idx).nonzero()
            if match_coords[0].size > 0:
                inverse_dict[element] = match_coords
        return inverse_dict


    
    