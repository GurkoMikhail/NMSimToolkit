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

    @classmethod
    def allocate(cls, capacity: int, num_processes: int = DEFAULT_MAX_PROCESSES) -> 'TransportBuffer':
        """Выделяет память под транспортный буфер заданной емкости частиц и числа процессов."""
        return cls(
            process_ids=np.empty(capacity, dtype=Index),
            material_ids=np.empty(capacity, dtype=Index),
            process_lacs=np.zeros(max(num_processes, DEFAULT_MAX_PROCESSES), dtype=Float),
        )


__all__ = [
    "DEFAULT_MAX_PROCESSES",
    "TransportBuffer",
]

