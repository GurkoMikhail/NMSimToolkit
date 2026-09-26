import numpy as np
from typing import NamedTuple, Any
from numpy.typing import NDArray

from core.other.typing_definitions import Time, Length
from core.other.typing_definitions import Index, ID, Energy, Float, ProcessID, Species, Charge
from core.other.vectors import Vector3DSoA


class RNGContext(NamedTuple):
    """
    Явная CFFI-обертка над указателем состояния генератора случайных чисел для передачи в Numba-кернелы.
    """
    next_double: Any
    state_addr: int

    @classmethod
    def from_numpy_rng(cls, rng: np.random.Generator) -> 'RNGContext':
        """
        Извлекает указатель CFFI next_double и адрес состояния из генератора NumPy.
        """
        return cls(
            next_double=rng.bit_generator.cffi.next_double,
            state_addr=rng.bit_generator.cffi.state_address
        )


class InteractionBuffer(NamedTuple):
    """
    SoA-буфер для промежуточной фиксации актов взаимодействия частиц на месте.

    Выделяется однократно и используется повторно для предотвращения фрагментации памяти.
    """
    process_id: NDArray[ProcessID]
    volume_id: NDArray[Index]
    material_id: NDArray[Index]
    particle_ID: NDArray[ID]
    energy_deposit: NDArray[Energy]
    scattering_theta: NDArray[Float]
    scattering_phi: NDArray[Float]
    distance_traveled: NDArray[Float]
    species: NDArray[Species]
    Z: NDArray[Charge]

    position: Vector3DSoA
    direction: Vector3DSoA

    cursor: NDArray[Index]  # Длина 1, отслеживает число записанных элементов
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        """
        Проверяет согласованность размерностей массивов буфера взаимодействий.
        """
        self.position.validate()
        self.direction.validate()

        tracked_arrays = [
            self.process_id,
            self.volume_id,
            self.material_id,
            self.particle_ID,
            self.energy_deposit,
            self.scattering_theta,
            self.scattering_phi,
            self.distance_traveled,
            self.species,
            self.Z
        ]

        # Все базовые поля должны быть одномерными
        for target_array in tracked_arrays:
            if target_array.ndim != 1:
                raise ValueError("Все массивы в InteractionBuffer должны быть одномерными.")

        # Проверка согласованности емкости
        for target_array in tracked_arrays:
            if target_array.shape[0] != self.capacity:
                raise ValueError("Все массивы в InteractionBuffer должны иметь одинаковую длину (емкость).")

        # Проверка длины компонент векторов
        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Компоненты векторов в InteractionBuffer должны иметь ту же длину, что и базовые массивы.")

        if self.cursor.shape != (1,):
            raise ValueError("Курсор должен быть одномерным массивом длины 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InteractionBuffer':
        """
        Выделяет память под InteractionBuffer заданной емкости.
        """
        buffer = cls(
            process_id=np.empty(capacity, dtype=ProcessID),
            volume_id=np.empty(capacity, dtype=Index),
            material_id=np.empty(capacity, dtype=Index),
            particle_ID=np.empty(capacity, dtype=ID),
            energy_deposit=np.empty(capacity, dtype=Energy),
            scattering_theta=np.empty(capacity, dtype=Float),
            scattering_phi=np.empty(capacity, dtype=Float),
            distance_traveled=np.empty(capacity, dtype=Float),
            species=np.empty(capacity, dtype=Species),
            Z=np.empty(capacity, dtype=Charge),
            position=Vector3DSoA.allocate(capacity, dtype=Float),
            direction=Vector3DSoA.allocate(capacity, dtype=Float),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer

    def flush_to_dict(self, clear: bool = True) -> dict:
        cursor_pos = self.cursor_value
        chunk = {
            'process_id': self.process_id[:cursor_pos].copy(),
            'volume_id': self.volume_id[:cursor_pos].copy(),
            'material_id': self.material_id[:cursor_pos].copy(),
            'particle_ID': self.particle_ID[:cursor_pos].copy(),
            'energy_deposit': self.energy_deposit[:cursor_pos].copy(),
            'scattering_theta': self.scattering_theta[:cursor_pos].copy(),
            'scattering_phi': self.scattering_phi[:cursor_pos].copy(),
            'distance_traveled': self.distance_traveled[:cursor_pos].copy(),
            'species': self.species[:cursor_pos].copy(),
            'Z': self.Z[:cursor_pos].copy(),
            'pos_x': self.position.x[:cursor_pos].copy(),
            'pos_y': self.position.y[:cursor_pos].copy(),
            'pos_z': self.position.z[:cursor_pos].copy(),
            'dir_x': self.direction.x[:cursor_pos].copy(),
            'dir_y': self.direction.y[:cursor_pos].copy(),
            'dir_z': self.direction.z[:cursor_pos].copy(),
        }
        if clear:
            self.reset_cursor()
        return chunk
    

class InitialStateBuffer(NamedTuple):
    """
    SoA-буфер для промежуточной фиксации начальных состояний частиц
    при их первом взаимодействии в объеме.
    """
    particle_ID: NDArray[ID]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    emission_position: Vector3DSoA
    emission_direction: Vector3DSoA

    cursor: NDArray[Index]
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        self.emission_position.validate()
        self.emission_direction.validate()

        tracked_arrays = [
            self.particle_ID,
            self.emission_time,
            self.emission_energy,
        ]

        for target_array in tracked_arrays:
            if target_array.ndim != 1:
                raise ValueError("Все массивы в InitialStateBuffer должны быть одномерными.")

        for target_array in tracked_arrays:
            if target_array.shape[0] != self.capacity:
                raise ValueError("Все массивы в InitialStateBuffer должны иметь одинаковую длину (емкость).")

        if self.emission_position.x.shape[0] != self.capacity:
            raise ValueError("Компоненты векторов в InitialStateBuffer должны иметь ту же длину, что и базовые массивы.")

        if self.cursor.shape != (1,):
            raise ValueError("Курсор должен быть одномерным массивом длины 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialStateBuffer':
        buffer = cls(
            particle_ID=np.empty(capacity, dtype=ID),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3DSoA.allocate(capacity, dtype=Length),
            emission_direction=Vector3DSoA.allocate(capacity, dtype=Float),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer

    def flush_to_dict(self, clear: bool = True) -> dict:
        cursor_pos = self.cursor_value
        chunk = {
            'particle_ID': self.particle_ID[:cursor_pos].copy(),
            'emission_time': self.emission_time[:cursor_pos].copy(),
            'emission_energy': self.emission_energy[:cursor_pos].copy(),
            'pos_x': self.emission_position.x[:cursor_pos].copy(),
            'pos_y': self.emission_position.y[:cursor_pos].copy(),
            'pos_z': self.emission_position.z[:cursor_pos].copy(),
            'dir_x': self.emission_direction.x[:cursor_pos].copy(),
            'dir_y': self.emission_direction.y[:cursor_pos].copy(),
            'dir_z': self.emission_direction.z[:cursor_pos].copy(),
        }
        if clear:
            self.reset_cursor()
        return chunk


class DeadParticlesBuffer(NamedTuple):
    """
    SoA-буфер для промежуточной фиксации идентификаторов завершивших трекинг частиц.
    """
    particle_ID: NDArray[ID]
    cursor: NDArray[Index]
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        if self.particle_ID.ndim != 1:
            raise ValueError("Массив particle_ID в DeadParticlesBuffer должен быть одномерным.")
        if self.particle_ID.shape[0] != self.capacity:
            raise ValueError("Массив particle_ID в DeadParticlesBuffer должен иметь длину, равную емкости.")
        if self.cursor.shape != (1,):
            raise ValueError("Курсор должен быть одномерным массивом длины 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'DeadParticlesBuffer':
        buffer = cls(
            particle_ID=np.empty(capacity, dtype=ID),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer

    def append(self, particle_ids: NDArray[ID]) -> None:
        """
        Добавляет идентификаторы завершивших трекинг частиц в буфер и сдвигает курсор.
        """
        particles_count = len(particle_ids)
        cursor_pos = self.cursor_value
        if cursor_pos + particles_count > self.capacity:
            raise ValueError("Недостаточно места в DeadParticlesBuffer.")
        self.particle_ID[cursor_pos:cursor_pos + particles_count] = particle_ids
        self.cursor[0] += particles_count

    def flush_to_array(self, clear: bool = True) -> NDArray[ID]:
        cursor_pos = self.cursor_value
        chunk = self.particle_ID[:cursor_pos].copy()
        if clear:
            self.reset_cursor()
        return chunk


class SimulationDataBuffer(NamedTuple):
    """
    Объединенный буфер логирования данных моделирования переноса частиц.
    """
    interactions: InteractionBuffer
    initial_states: InitialStateBuffer
    dead_particles: DeadParticlesBuffer

    @classmethod
    def allocate(cls, interaction_capacity: int, initial_state_capacity: int, dead_particles_capacity: int) -> 'SimulationDataBuffer':
        """
        Выделяет буферы взаимодействий, начальных состояний и завершивших трекинг частиц заданной емкости.
        """
        return cls(
            interactions=InteractionBuffer.allocate(interaction_capacity),
            initial_states=InitialStateBuffer.allocate(initial_state_capacity),
            dead_particles=DeadParticlesBuffer.allocate(dead_particles_capacity)
        )
