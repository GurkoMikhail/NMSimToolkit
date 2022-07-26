import numpy as np


class NonuniqueArray:
    
    def __init__(self, shape):
        self._inverse = np.zeros(shape, dtype='u8')
        self._unique = np.ndarray(0)
    
    def __getitem__(self, key):
        new_inverse = self._inverse[key]
        new_array = NonuniqueArray(new_inverse.shape)
        new_array._inverse[...] = new_inverse
        new_array._unique = self._unique
        return new_array
    
    def __setitem__(self, key, value):
        self._unique.dtype = value.dtype
        self._unique = np.union1d(self._unique, value).view(value.__class__)
        self._inverse[key] = (self._unique == value).nonzero()[0]
    
    def __len__(self):
        return len(self._inverse)
    
    def __array__(self):
        return np.asarray(self.restore())
    
    def restore(self):
        return self._unique[self._inverse]
    
    @classmethod
    def create(cls, values):
        unique, inverse = np.unique(values, return_inverse=True)
        obj = cls(values.shape)
        obj._inverse[...] = inverse
        obj._unique = unique
        return obj
    
    