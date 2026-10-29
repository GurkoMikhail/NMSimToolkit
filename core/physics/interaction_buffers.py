"""
Модуль SoA-буферов регистрации взаимодействий, начальных состояний, выбывших частиц и контекста ГСЧ.
"""

from enum import Enum
from typing import AbstractSet, Any, NamedTuple, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import (
    Charge,
    Energy,
    Float,
    ID,
    Index,
    Length,
    ProcessID,
    Species,
    Time,
)
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


class _RNGContextTuple(NamedTuple):
    """
    Базовый кортеж CFFI-указателей состояния генератора случайных чисел для диспетчеризации в Numba.
    """
    next_double: Any
    state_addr: int


class RNGContext(_RNGContextTuple):
    """
    Явная CFFI-обертка над указателем состояния генератора случайных чисел для передачи в Numba-кернелы.

    Удерживает сильные ссылки на исходный экземпляр ``np.random.Generator`` и его ``bit_generator``,
    предотвращая преждевременное освобождение C-структуры состояния сборщиком мусора Python
    при передаче временного генератора ``RNGContext.from_numpy_rng(np.random.default_rng(seed))``.
    """
    _rng_reference: np.random.Generator | None
    _bit_generator_reference: Any

    @classmethod
    def from_numpy_rng(cls, rng: np.random.Generator) -> 'RNGContext':
        """
        Извлекает указатель CFFI next_double и адрес состояния из генератора NumPy,
        сохраняя сильную ссылку на генератор внутри контекста.
        """
        context = cls(
            next_double=rng.bit_generator.cffi.next_double,
            state_addr=rng.bit_generator.cffi.state_address,
        )
        context._rng_reference = rng
        context._bit_generator_reference = rng.bit_generator
        return context


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

    @classmethod
    def allocate(cls, capacity: int) -> 'InteractionBuffer':
        """
        Выделяет память под InteractionBuffer заданной емкости.
        """
        if capacity < 0:
            raise ValueError("Емкость InteractionBuffer должна быть неотрицательной.")
        return cls(
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
    parent_ID: NDArray[ID]
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

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialStateBuffer':
        if capacity < 0:
            raise ValueError("Емкость InitialStateBuffer должна быть неотрицательной.")
        return cls(
            particle_ID=np.empty(capacity, dtype=ID),
            parent_ID=np.full(capacity, -1, dtype=ID),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3DSoA.allocate(capacity, dtype=Length),
            emission_direction=Vector3DSoA.allocate(capacity, dtype=Float),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )

    def flush(self, clear: bool = True) -> dict[str, NDArray[Any]]:
        cursor_pos = self.cursor_value
        chunk: dict[str, NDArray[Any]] = {
            'particle_ID': self.particle_ID[:cursor_pos].copy(),
            'parent_ID': self.parent_ID[:cursor_pos].copy(),
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

    @classmethod
    def allocate(cls, capacity: int) -> 'DeadParticlesBuffer':
        if capacity < 0:
            raise ValueError("Емкость DeadParticlesBuffer должна быть неотрицательной.")
        return cls(
            particle_ID=np.empty(capacity, dtype=ID),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )

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

    @classmethod
    def allocate(cls, capacity: int) -> 'EscapedParticlesBuffer':
        if capacity < 0:
            raise ValueError("Емкость EscapedParticlesBuffer должна быть неотрицательной.")
        return cls(
            particle_ID=np.empty(capacity, dtype=ID),
            position=Vector3DSoA.allocate(capacity, dtype=Length),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity,
        )

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


class SecondaryActivationBuffer(NamedTuple):
    """
    Предвыделенный SoA-буфер точек активации (наведенных вторичных источников).
    Использует Vector3DSoA единообразно с InteractionBuffer и InitialStateBuffer.
    Заполняется на шаге транспорта при взаимодействиях и разряжается в ParticleBank
    на шаге пополнения с пуассоновским розыгрышем времени испускания.
    """
    position: Vector3DSoA
    direction: Vector3DSoA
    activation_time: NDArray[Time]
    energy: NDArray[Energy]
    parent_ID: NDArray[ID]
    species: NDArray[Species]
    target_z: NDArray[Charge]
    shell_idx: NDArray[Index]
    process_id: NDArray[ProcessID]
    cursor: NDArray[Index]
    capacity: int
    enable_atomic_relaxation: bool = False
    photon_production_cut: Float = Float(np.inf)
    electron_production_cut: Float = Float(np.inf)

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

    @classmethod
    def allocate(
        cls,
        capacity: int,
        enable_atomic_relaxation: bool = False,
        photon_production_cut: Float = Float(np.inf),
        electron_production_cut: Float = Float(np.inf),
    ) -> 'SecondaryActivationBuffer':
        if capacity < 0:
            raise ValueError("Емкость SecondaryActivationBuffer должна быть неотрицательной.")
        return cls(
            position=Vector3DSoA.allocate(capacity, dtype=Length),
            direction=Vector3DSoA.allocate(capacity, dtype=Float),
            activation_time=np.empty(capacity, dtype=Time),
            energy=np.empty(capacity, dtype=Energy),
            parent_ID=np.empty(capacity, dtype=ID),
            species=np.empty(capacity, dtype=Species),
            target_z=np.empty(capacity, dtype=Charge),
            shell_idx=np.empty(capacity, dtype=Index),
            process_id=np.empty(capacity, dtype=ProcessID),
            cursor=np.zeros(1, dtype=Index),
            capacity=int(capacity),
            enable_atomic_relaxation=bool(enable_atomic_relaxation),
            photon_production_cut=Float(photon_production_cut),
            electron_production_cut=Float(electron_production_cut),
        )


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
    "SecondaryActivationBuffer",
    "SimulationDataBuffer",
]
