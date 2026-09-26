from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


class Vector3DSoA(NamedTuple):
    """
    SoA-структура (Structure of Arrays) для представления трехмерных векторных полей частиц.
    Содержит плоские одномерные C-непрерывные массивы numpy для компонент X, Y и Z.
    """
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

    def validate(self) -> None:
        """
        Проверяет, что компоненты являются одномерными массивами одинаковой длины.
        """
        if self.x.ndim != 1 or self.y.ndim != 1 or self.z.ndim != 1:
            raise ValueError("Массивы координат Vector3DSoA должны быть одномерными.")

        length = self.x.shape[0]
        if self.y.shape[0] != length or self.z.shape[0] != length:
            raise ValueError("Массивы координат Vector3DSoA должны иметь одинаковую длину.")

    @classmethod
    def allocate(cls, capacity: int, dtype: np.dtype = Float) -> 'Vector3DSoA':
        """
        Выделяет память под SoA-буфер векторов заданной емкости.
        """
        buffer = cls(
            x=np.empty(capacity, dtype=dtype),
            y=np.empty(capacity, dtype=dtype),
            z=np.empty(capacity, dtype=dtype)
        )
        buffer.validate()
        return buffer


__all__ = [
    'Vector3DSoA',
]
