"""
Модуль скомпилированного банка материалов (MaterialBank) для быстрых обращений из Numba-кернелов.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index

MaterialInfoDType = np.dtype([
    ('density', Float),
    ('Z', Float),
    ('A', Float),
])

MaterialPointerDType = np.dtype([
    ('start_idx', Index),
    ('length', Index),
])


class MaterialBank(NamedTuple):
    """
    Плоские массивы свойств материалов и таблиц линейных коэффициентов ослабления (LAC),
    скомпилированные для используемых в сцене материалов.
    """

    mat_info_buffer: NDArray[np.void]
    mat_pointers: NDArray[np.void]
    physics_energy_grid: NDArray[Float]
    physics_lac_table: NDArray[Float]

    def _validate(self) -> None:
        """Проверяет целостность типов и размерностей массивов банка материалов."""
        if self.mat_info_buffer.dtype != MaterialInfoDType:
            raise ValueError("Некорректный тип данных mat_info_buffer.")
        if self.mat_pointers.dtype != MaterialPointerDType:
            raise ValueError("Некорректный тип данных mat_pointers.")
        if self.physics_energy_grid.ndim != 1:
            raise ValueError("Массив physics_energy_grid должен быть одномерным.")
        if self.physics_lac_table.ndim != 2:
            raise ValueError("Таблица physics_lac_table должна быть двумерной.")


__all__ = [
    "MaterialInfoDType",
    "MaterialPointerDType",
    "MaterialBank",
]

