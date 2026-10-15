"""
Модуль состояния геометрической навигации частиц (NavigationState) в формате Structure of Arrays (SoA).
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index


class NavigationState(NamedTuple):
    """
    SoA-структура (Structure of Arrays) для кэширования состояния геометрической навигации частиц.
    Используется при пошаговой навигации, релокации и кэшировании расстояний до границ в трекинге Вудкока.
    """

    current_volume: NDArray[Index]
    boundary_distance: NDArray[Float]

    def validate(self) -> None:
        """Проверяет, что массивы состояния навигации являются одномерными и имеют одинаковую длину."""
        arrays = (
            self.current_volume,
            self.boundary_distance,
        )
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("Массивы NavigationState должны быть одномерными.")
        length = arrays[0].shape[0]
        for arr in arrays:
            if arr.shape[0] != length:
                raise ValueError("Массивы NavigationState должны иметь одинаковую длину.")

    @property
    def capacity(self) -> int:
        """Емкость буфера состояния навигации."""
        return int(self.current_volume.shape[0])

    @classmethod
    def allocate(cls, capacity: int) -> NavigationState:
        """Выделяет память под буфер NavigationState, заполненный значениями по умолчанию."""
        buffer = cls(
            current_volume=np.full(capacity, -1, dtype=Index),
            boundary_distance=np.full(capacity, np.inf, dtype=Float),
        )
        buffer.validate()
        return buffer


__all__ = ["NavigationState"]

