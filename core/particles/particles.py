"""
Модуль фасада пула частиц (ParticleBank), объединяющего кинематическое, начальное и навигационное состояния.
"""

from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from core.geometry.navigation_state import NavigationState
from core.other.typing_definitions import Energy, Float, ID, Index, Length, Species, Time
from core.other.vectors import Vector3DSoA
from core.particles.initial_state import InitialState
from core.particles.kinematic_state import KinematicState
import core.particles.particles_kernels as kernel


class ParticleBank(NamedTuple):
    """
    Фасад для управления пулом объектов частиц на базе SoA-структуры.

    Разделяет высокоуровневое управление жизненным циклом и Numba-кернелы расчетов.
    """

    state: KinematicState
    initial_state: InitialState
    navigation_state: NavigationState
    count_array: NDArray[Index]
    capacity: int

    @classmethod
    def allocate(cls, capacity: int) -> 'ParticleBank':
        """
        Выделяет память под пул частиц заданной емкости вместе с внутренними массивами.
        """
        state = KinematicState.allocate(capacity)
        initial_state = InitialState.allocate(capacity)
        navigation_state = NavigationState.allocate(capacity)
        count_array = np.zeros(1, dtype=Index)
        return cls(
            state=state,
            initial_state=initial_state,
            navigation_state=navigation_state,
            count_array=count_array,
            capacity=capacity,
        )

    @property
    def count(self) -> int:  # type: ignore[override]
        """Общее количество инжектированных за время симуляции частиц."""
        return int(self.count_array[0])

    def inject_particles(
        self,
        species: NDArray[Species],
        position: Vector3DSoA,
        direction: Vector3DSoA,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
        distance_traveled: NDArray[Length],
        parent_id: Optional[NDArray[ID]] = None,
    ) -> NDArray[Index]:
        """
        Инжектирует новые частицы в неактивные слоты пула объектов.

        Возвращает индексы слотов, в которые были добавлены частицы.
        Автоматически сохраняет параметры излучения на основе входного состояния.
        """
        num_new = species.shape[0]

        inactive_indices = np.where(~self.state.is_active)[0]

        if num_new > inactive_indices.shape[0]:
            raise RuntimeError(
                f"Превышена доступная емкость пула частиц: попытка инжектировать {num_new} "
                f"частиц при наличии только {inactive_indices.shape[0]} свободных слотов."
            )

        target_indices = inactive_indices[:num_new]

        current_count = int(self.count_array[0])
        new_ids = np.arange(current_count, current_count + num_new, dtype=ID)
        self.count_array[0] += num_new

        self.state.is_active[target_indices] = True
        self.initial_state.ID[target_indices] = new_ids
        if parent_id is not None:
            self.initial_state.parent_ID[target_indices] = parent_id
        else:
            self.initial_state.parent_ID[target_indices] = ID(-1)
        self.state.species[target_indices] = species
        self.state.energy[target_indices] = energy
        self.initial_state.emission_time[target_indices] = emission_time
        self.initial_state.emission_energy[target_indices] = energy
        self.state.distance_traveled[target_indices] = distance_traveled

        self.state.position.x[target_indices] = position.x
        self.state.position.y[target_indices] = position.y
        self.state.position.z[target_indices] = position.z

        self.state.direction.x[target_indices] = direction.x
        self.state.direction.y[target_indices] = direction.y
        self.state.direction.z[target_indices] = direction.z

        self.initial_state.emission_position.x[target_indices] = position.x
        self.initial_state.emission_position.y[target_indices] = position.y
        self.initial_state.emission_position.z[target_indices] = position.z

        self.initial_state.emission_direction.x[target_indices] = direction.x
        self.initial_state.emission_direction.y[target_indices] = direction.y
        self.initial_state.emission_direction.z[target_indices] = direction.z

        kernel.update_navigation_state_inject_kernel(self.navigation_state, target_indices)

        return target_indices

    @property
    def active_indices(self) -> NDArray[Index]:
        """Возвращает массив индексов активных в данный момент частиц в пуле."""
        return np.nonzero(self.state.is_active)[0]

    def move(self, target_indices: NDArray[Index], distances: NDArray[Float]) -> None:
        """
        Фасад для move_kernel, перемещающий выбранные активные частицы на заданные расстояния.
        """
        kernel.move_kernel(self.state, target_indices, distances)
        kernel.update_navigation_state_move_kernel(self.navigation_state, target_indices, distances)

    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None:
        """
        Фасад для rotate_kernel, поворачивающий направление движения выбранных активных частиц.
        """
        kernel.rotate_kernel(self.state, target_indices, thetas, phis)
        kernel.update_navigation_state_rotate_kernel(self.navigation_state, target_indices)


__all__ = ["ParticleBank"]
