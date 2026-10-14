"""
Модуль SoA-буфера промежуточных результатов шага транспорта частиц (TransportBuffer).
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index

DEFAULT_MAX_PROCESSES: int = 16


class TransportBuffer(NamedTuple):
    """
    SoA-буфер для записи результатов выбора процесса и материала на шаге транспорта,
    а также предвыделенного рабочего массива макроскопических сечений процессов.
    """

    process_ids: NDArray[Index]
    material_ids: NDArray[Index]
    process_lacs: NDArray[Float]

    def validate(self) -> None:
        """Проверяет корректность размерностей массивов транспортного буфера."""
        if self.process_ids.ndim != 1 or self.material_ids.ndim != 1 or self.process_lacs.ndim != 1:
            raise ValueError("Все массивы в TransportBuffer должны быть одномерными.")
        if self.process_ids.shape[0] != self.material_ids.shape[0]:
            raise ValueError("Массивы process_ids и material_ids в TransportBuffer должны иметь одинаковую длину.")

    @classmethod
    def allocate(cls, capacity: int, num_processes: int = DEFAULT_MAX_PROCESSES) -> 'TransportBuffer':
        """Выделяет память под транспортный буфер заданной емкости частиц и числа процессов."""
        buffer = cls(
            process_ids=np.empty(capacity, dtype=Index),
            material_ids=np.empty(capacity, dtype=Index),
            process_lacs=np.zeros(max(num_processes, DEFAULT_MAX_PROCESSES), dtype=Float),
        )
        buffer.validate()
        return buffer


__all__ = [
    "DEFAULT_MAX_PROCESSES",
    "TransportBuffer",
]

