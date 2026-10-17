"""
Модуль SoA-буферов регистрации взаимодействий, начальных состояний, выбывших частиц и контекста ГСЧ.
"""

from enum import Enum
from typing import AbstractSet, Any, NamedTuple, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Charge, Energy, Float, ID, Index, Length, ProcessID, Species, Time
from core.other.vectors import Vector3DSoA


class ChunkType(Enum):
    """
    Типизированное перечисление категорий чанков данных моделирования.
    """
    INITIAL_STATES = "initial_states"
    INTERACTIONS = "interactions"
    DEAD_PARTICLES = "dead_particles"
    ESCAPED_PARTICLES = "escaped_particles"


TELEMETRY_ORDER: tuple[ChunkType, ...] = (
    ChunkType.INITIAL_STATES,
    ChunkType.INTERACTIONS,
    ChunkType.ESCAPED_PARTICLES,
    ChunkType.DEAD_PARTICLES,
)
"""Причинно-следственный порядок каналов телеметрии (ранг = позиция в кортеже)."""

TelemetryPayload: TypeAlias = dict[str, NDArray[Any]] | NDArray[ID]


class TelemetryBuffer(Protocol):
    """Статический контракт буфера телеметрии с курсором записи."""

    @property
    def capacity(self) -> int: ...

    @property
    def cursor_value(self) -> int: ...

    @property
    def remaining_capacity(self) -> int: ...

    @property
    def is_enabled(self) -> bool: ...

    def reset_cursor(self) -> None: ...

    def flush(self, clear: bool = True) -> TelemetryPayload: ...


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

    @property
    def is_enabled(self) -> bool:
        return self.capacity > 0

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        """
        Проверяет согласованность размерностей массивов буфера взаимодействий.
        """
        if self.capacity < 0:
            raise ValueError("Емкость InteractionBuffer должна быть неотрицательной.")

        self.position.validate()
        self.direction.validate()

        tracked_arrays: list[NDArray[Any]] = [
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

    def flush(self, clear: bool = True) -> dict[str, NDArray[Any]]:
        cursor_pos = self.cursor_value
        chunk: dict[str, NDArray[Any]] = {
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
    SoA-буфер для промежуточной фиксации начальных состояний частиц.
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

    @property
    def is_enabled(self) -> bool:
        return self.capacity > 0

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        if self.capacity < 0:
            raise ValueError("Емкость InitialStateBuffer должна быть неотрицательной.")

        self.emission_position.validate()
        self.emission_direction.validate()

        tracked_arrays: list[NDArray[Any]] = [
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

    def flush(self, clear: bool = True) -> dict[str, NDArray[Any]]:
        cursor_pos = self.cursor_value
        chunk: dict[str, NDArray[Any]] = {
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

    @property
    def is_enabled(self) -> bool:
        return self.capacity > 0

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        if self.capacity < 0:
            raise ValueError("Емкость DeadParticlesBuffer должна быть неотрицательной.")
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

    def flush(self, clear: bool = True) -> NDArray[ID]:
        cursor_pos = self.cursor_value
        chunk = self.particle_ID[:cursor_pos].copy()
        if clear:
            self.reset_cursor()
        return chunk


class EscapedParticlesBuffer(NamedTuple):
    """
    SoA-буфер для промежуточной фиксации частиц, покинувших границы геометрии сцены.
    """
    particle_ID: NDArray[ID]
    position: Vector3DSoA

    cursor: NDArray[Index]
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    @property
    def is_enabled(self) -> bool:
        return self.capacity > 0

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        if self.capacity < 0:
            raise ValueError("Емкость EscapedParticlesBuffer должна быть неотрицательной.")

        self.position.validate()

        if self.particle_ID.ndim != 1:
            raise ValueError("Все массивы в EscapedParticlesBuffer должны быть одномерными.")

        if self.particle_ID.shape[0] != self.capacity:
            raise ValueError("Все массивы в EscapedParticlesBuffer должны иметь одинаковую длину (емкость).")

        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Компоненты векторов в EscapedParticlesBuffer должны иметь ту же длину, что и базовые массивы.")

        if self.cursor.shape != (1,):
            raise ValueError("Курсор должен быть одномерным массивом длины 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'EscapedParticlesBuffer':
        buffer = cls(
            particle_ID=np.empty(capacity, dtype=ID),
            position=Vector3DSoA.allocate(capacity, dtype=Length),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity,
        )
        buffer.validate()
        return buffer

    def flush(self, clear: bool = True) -> dict[str, NDArray[Any]]:
        cursor_pos = self.cursor_value
        chunk: dict[str, NDArray[Any]] = {
            'particle_ID': self.particle_ID[:cursor_pos].copy(),
            'pos_x': self.position.x[:cursor_pos].copy(),
            'pos_y': self.position.y[:cursor_pos].copy(),
            'pos_z': self.position.z[:cursor_pos].copy(),
        }
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
    escaped_particles: EscapedParticlesBuffer

    def channel(self, chunk_type: ChunkType) -> TelemetryBuffer:
        """Возвращает буфер канала по типу чанка."""
        match chunk_type:
            case ChunkType.INITIAL_STATES:
                return self.initial_states
            case ChunkType.INTERACTIONS:
                return self.interactions
            case ChunkType.ESCAPED_PARTICLES:
                return self.escaped_particles
            case ChunkType.DEAD_PARTICLES:
                return self.dead_particles

    @classmethod
    def allocate(
        cls,
        interaction_capacity: int,
        initial_state_capacity: int,
        dead_particles_capacity: int,
        escaped_particles_capacity: int,
    ) -> 'SimulationDataBuffer':
        """
        Выделяет буферы взаимодействий, начальных состояний, завершивших трекинг
        и вылетевших за границы сцены частиц заданной емкости.
        """
        return cls(
            interactions=InteractionBuffer.allocate(interaction_capacity),
            initial_states=InitialStateBuffer.allocate(initial_state_capacity),
            dead_particles=DeadParticlesBuffer.allocate(dead_particles_capacity),
            escaped_particles=EscapedParticlesBuffer.allocate(escaped_particles_capacity),
        )

    @classmethod
    def allocate_for(
        cls,
        capacity: int,
        enabled: AbstractSet[ChunkType],
    ) -> 'SimulationDataBuffer':
        """Выделяет заданную ёмкость только подписанным каналам; у неподписанных capacity == 0."""
        return cls.allocate(
            interaction_capacity=capacity if ChunkType.INTERACTIONS in enabled else 0,
            initial_state_capacity=capacity if ChunkType.INITIAL_STATES in enabled else 0,
            dead_particles_capacity=capacity if ChunkType.DEAD_PARTICLES in enabled else 0,
            escaped_particles_capacity=capacity if ChunkType.ESCAPED_PARTICLES in enabled else 0,
        )


__all__ = [
    "TELEMETRY_ORDER",
    "ChunkType",
    "TelemetryBuffer",
    "TelemetryPayload",
    "RNGContext",
    "InteractionBuffer",
    "InitialStateBuffer",
    "DeadParticlesBuffer",
    "EscapedParticlesBuffer",
    "SimulationDataBuffer",
]




