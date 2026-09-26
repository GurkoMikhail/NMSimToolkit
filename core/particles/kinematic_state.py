import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, Length, Species
from core.other.vectors import Vector3DSoA

class KinematicState(NamedTuple):
    """
    SoA-структура (Structure of Arrays) для кинематических состояний частиц.

    Содержит исключительно одномерные плоские C-непрерывные массивы NumPy для «горячих» данных,
    активно используемых физическими кернелами.
    """
    is_active: NDArray[np.bool_]
    species: NDArray[Species]

    # Вектор пространственного положения
    position: Vector3DSoA

    # Вектор направления движения
    direction: Vector3DSoA

    energy: NDArray[Energy]

    distance_traveled: NDArray[Length]

    @property
    def capacity(self) -> int:
        return self.is_active.shape[0]

    def validate(self) -> None:
        """
        Проверяет согласованность размерностей массивов кинематического состояния.
        """
        self.position.validate()
        self.direction.validate()

        tracked_arrays = [
            self.is_active,
            self.species,
            self.energy,
            self.distance_traveled,
        ]

        # Все базовые массивы должны быть одномерными
        for target_array in tracked_arrays:
            if target_array.ndim != 1:
                raise ValueError("Все массивы в KinematicState должны быть одномерными.")

        # Проверка соответствия длины емкости пула
        for target_array in tracked_arrays:
            if target_array.shape[0] != self.capacity:
                raise ValueError("Все массивы в KinematicState должны иметь одинаковую длину (емкость).")

        # Проверка соответствия длины компонент векторов
        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Компоненты векторов в KinematicState должны иметь ту же длину, что и базовые массивы.")

    @classmethod
    def allocate(cls, capacity: int) -> 'KinematicState':
        """
        Выделяет память под пустое кинематическое состояние KinematicState заданной емкости.
        """
        buffer = cls(
            is_active=np.zeros(capacity, dtype=np.bool_),
            species=np.empty(capacity, dtype=Species),
            position=Vector3DSoA.allocate(capacity, dtype=Length),
            direction=Vector3DSoA.allocate(capacity, dtype=Float),
            energy=np.empty(capacity, dtype=Energy),
            distance_traveled=np.empty(capacity, dtype=Length),
        )
        buffer.validate()
        return buffer
