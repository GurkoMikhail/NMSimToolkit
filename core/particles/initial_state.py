import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Time, Length, Float, ID
from core.other.vectors import Vector3DSoA


class InitialState(NamedTuple):
    """
    SoA-структура (Structure of Arrays) для начальных состояний частиц.

    Содержит плоские одномерные C-непрерывные массивы NumPy с начальными
    параметрами и идентификаторами сгенерированных частиц.
    """
    ID: NDArray[ID]
    has_interacted: NDArray[np.bool_]

    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    # Вектор начальной позиции излучения
    emission_position: Vector3DSoA

    # Вектор начального направления излучения
    emission_direction: Vector3DSoA

    @property
    def capacity(self) -> int:
        return self.emission_time.shape[0]

    def validate(self) -> None:
        """
        Проверяет согласованность размерностей массивов начального состояния.
        """
        self.emission_position.validate()
        self.emission_direction.validate()

        tracked_arrays = [
            self.ID,
            self.has_interacted,
            self.emission_time,
            self.emission_energy,
        ]

        # Все базовые массивы должны быть одномерными
        for target_array in tracked_arrays:
            if target_array.ndim != 1:
                raise ValueError("Все массивы в InitialState должны быть одномерными.")

        # Проверка соответствия емкости
        for target_array in tracked_arrays:
            if target_array.shape[0] != self.capacity:
                raise ValueError("Все массивы в InitialState должны иметь одинаковую длину (емкость).")

        # Проверка соответствия длины компонент векторов
        if self.emission_position.x.shape[0] != self.capacity:
            raise ValueError("Компоненты векторов в InitialState должны иметь ту же длину, что и базовые массивы.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialState':
        """
        Выделяет память под пустое состояние InitialState заданной емкости.
        """
        buffer = cls(
            ID=np.empty(capacity, dtype=ID),
            has_interacted=np.zeros(capacity, dtype=np.bool_),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3DSoA.allocate(capacity, dtype=Length),
            emission_direction=Vector3DSoA.allocate(capacity, dtype=Float)
        )
        buffer.validate()
        return buffer
