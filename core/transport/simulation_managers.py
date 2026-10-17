"""
Модуль менеджера моделирования (SimulationManager) и состояний жизненного цикла симуляции.
"""

from datetime import datetime, timedelta
from enum import Enum, auto
import logging
from queue import Full, Queue
import threading
from typing import AbstractSet, Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Union

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_compiler import GeometryCompiler
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import TELEMETRY_ORDER, ChunkType, RNGContext, SimulationDataBuffer
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.scene.nodes import CompositeNode
from core.source.source_compiler import SourceCompiler
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.transport_kernels import (
    _push_to_dead_particles_kernel,
    _push_to_escaped_particles_kernel,
    _push_uncollided_to_initial_state_kernel,
)

_logger = logging.getLogger(__name__)

DEFAULT_CONSUMED_CHUNK_TYPES: FrozenSet[ChunkType] = frozenset({
    ChunkType.INITIAL_STATES,
    ChunkType.INTERACTIONS,
    ChunkType.DEAD_PARTICLES,
})


class SimulationState(Enum):
    """
    Состояния жизненного цикла процесса моделирования.
    """

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


class SimulationManager(threading.Thread):
    """
    Вычислительный менеджер симуляции, оптимизированный под Data-Oriented Design (SoA),
    с поддержкой непрерывной инжекции частиц и эффективного уплотнения буферов данных на месте.
    """

    MAX_NEWTON_ITERATIONS: ClassVar[int] = 5
    NEWTON_TARGET_PRECISION: ClassVar[float] = 1e-3
    MIN_TIME_STEP: ClassVar[Float] = Float(1e-9)
    QUEUE_PUT_TIMEOUT: ClassVar[float] = 0.05

    active_sources: List[Source]
    scene: CompositeNode
    propagator: ParticlePropagator
    start_time: Float
    stop_time: Float
    initial_particle_id: int
    particles_number: int
    min_energy: Float
    queue: Queue[Any]
    consumed_chunk_types: FrozenSet[ChunkType]
    step_count: int
    bank: ParticleBank
    data_buffer: SimulationDataBuffer
    geometry_buffer: NDArray[Any]
    physics_buffer: PhysicsBuffer
    rng: np.random.Generator
    rng_ctx: RNGContext
    invalidators: List[Callable[[NDArray[Index]], NDArray[np.bool_]]]
    global_timer: Float
    _run_event: threading.Event
    _state: SimulationState

    def __init__(
        self,
        scene: CompositeNode,
        propagator: Optional[ParticlePropagator] = None,
        stop_time: Union[float, Float] = Float(1.0 * units.s),
        start_time: Union[float, Float] = Float(0.0),
        particles_number: Union[int, float, Float] = 10**3,
        min_energy: Union[float, Float] = Float(1.0 * units.keV),
        queue: Optional[Queue[Any]] = None,
        buffer_capacity: Optional[int] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        initial_particle_id: int = 0,
        rng_state: Optional[Dict[str, Any]] = None,
        consumed_chunk_types: Optional[AbstractSet[ChunkType]] = None,
    ) -> None:
        super().__init__()
        if name is not None:
            self.name = name
        self.scene = scene
        self.active_sources = SourceCompiler().compile_scene(scene)
        self.propagator = ParticlePropagator() if propagator is None else propagator

        self.geometry_buffer = GeometryCompiler().compile_scene(scene)
        self.physics_buffer = PhysicsCompiler().compile_scene(scene, self.propagator.processes)
        self.start_time = Float(start_time)
        self.stop_time = Float(stop_time)
        self.initial_particle_id = int(initial_particle_id)
        self.particles_number = int(particles_number)
        effective_capacity = (
            self.particles_number if buffer_capacity is None else max(int(buffer_capacity), self.particles_number)
        )
        self.min_energy = Float(min_energy)
        self.queue = Queue(maxsize=64) if queue is None else queue
        self.consumed_chunk_types = (
            DEFAULT_CONSUMED_CHUNK_TYPES
            if consumed_chunk_types is None
            else frozenset(consumed_chunk_types)
        )
        self.step_count = 0
        self.daemon = True

        self.bank = ParticleBank.allocate(self.particles_number)
        self.bank.count_array[0] = Index(self.initial_particle_id)
        self.data_buffer = SimulationDataBuffer.allocate_for(
            capacity=effective_capacity,
            enabled=self.consumed_chunk_types,
        )

        self.rng = np.random.default_rng(seed) if seed is not None else self.propagator.rng
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.propagator.rng = self.rng
        for process in self.propagator.processes:
            process.rng = self.rng
        for source_node in self.active_sources:
            source_node.rng = self.rng

        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]
        self.global_timer = Float(start_time)

        self._state = SimulationState.IDLE
        self._run_event = threading.Event()
        self._run_event.set()

    @property
    def state(self) -> SimulationState:
        """Текущее состояние процесса моделирования."""
        return self._state

    @property
    def is_stopped(self) -> bool:
        """Флаг завершения или кооперативной остановки моделирования."""
        return self._state == SimulationState.STOPPED

    @property
    def has_active_particles(self) -> bool:
        """Проверяет наличие активных частиц, находящихся в процессе переноса."""
        return bool(np.any(self.bank.state.is_active))

    @property
    def has_pending_sources(self) -> bool:
        """Проверяет наличие активных источников частиц до достижения лимита времени."""
        return bool(self.active_sources and self.global_timer < self.stop_time)

    def pause(self) -> None:
        """Приостанавливает выполнение моделирования."""
        if self._state == SimulationState.STOPPED:
            return
        self._state = SimulationState.PAUSED
        self._run_event.clear()

    def resume(self) -> None:
        """Возобновляет приостановленное моделирование."""
        if self._state == SimulationState.STOPPED:
            return
        self._state = SimulationState.RUNNING
        self._run_event.set()

    def stop(self) -> None:
        """Кооперативно останавливает процесс моделирования."""
        self._state = SimulationState.STOPPED
        self._run_event.set()

    def step_once(self) -> None:
        """Выполняет один шаг моделирования для покадрового анализа."""
        self.next_step()
        self.flush_all()

    def _put_to_queue(self, item: Any) -> None:
        """Помещает элемент в выходную очередь с периодической проверкой флага остановки."""
        while True:
            try:
                self.queue.put(item, timeout=self.QUEUE_PUT_TIMEOUT)
                return
            except Full:
                if self.is_stopped:
                    _logger.warning(
                        "%s: элемент очереди отброшен, так как очередь переполнена после остановки",
                        self.name,
                    )
                    return

    def send_data(self, data: Any) -> None:
        """Отправляет пакет телеметрии в выходную очередь симуляции."""
        self._put_to_queue(data)

    def _flush_channel(self, chunk_type: ChunkType) -> None:
        """Сбрасывает буфер одного канала телеметрии в очередь, если в нём есть записи."""
        buffer = self.data_buffer.channel(chunk_type)
        count = buffer.cursor_value
        if count == 0:
            return
        _logger.debug("%s flushing %d %s", self.name, count, chunk_type.value)
        self.send_data({"type": chunk_type, "data": buffer.flush()})

    def _flush_through(self, last: ChunkType) -> None:
        """Каскадно сбрасывает все каналы с рангом <= ранга last в причинно-следственном порядке."""
        for chunk_type in TELEMETRY_ORDER:
            self._flush_channel(chunk_type)
            if chunk_type is last:
                return

    def flush_all(self) -> None:
        """
        Выполняет полный сброс всех буферов телеметрии в строгом причинно-следственном порядке:
        initial_states → interactions → escaped_particles → dead_particles.
        """
        self._flush_through(TELEMETRY_ORDER[-1])

    def _ensure_step_capacity(self, required: int) -> None:
        """Предусловие шага: каждый включённый канал вмещает не менее required записей."""
        for chunk_type in reversed(TELEMETRY_ORDER):
            buffer = self.data_buffer.channel(chunk_type)
            if buffer.is_enabled and buffer.remaining_capacity < required:
                self._flush_through(chunk_type)
                return

    def _check_no_overflow(self) -> None:
        """Постусловие шага: проверяет, что ни один канал телеметрии не был переполнен."""
        for chunk_type in TELEMETRY_ORDER:
            buffer = self.data_buffer.channel(chunk_type)
            if buffer.cursor_value > buffer.capacity:
                raise RuntimeError(
                    f"Переполнение канала {chunk_type.value}: "
                    f"{buffer.cursor_value} > {buffer.capacity}"
                )

    def _invalidate_by_energy(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        return self.bank.state.energy[active_indices] < self.min_energy

    def _invalidate_by_volume(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        navigation_state = self.bank.navigation_state
        return (navigation_state.current_volume[active_indices] < 0) & (
            navigation_state.boundary_distance[active_indices] > 0.0
        )

    def _apply_invalidators(self, active_indices: NDArray[Index]) -> NDArray[Index]:
        dead_mask = np.zeros(len(active_indices), dtype=np.bool_)
        for invalidator in self.invalidators:
            dead_mask |= invalidator(active_indices)

        if np.any(dead_mask):
            dead_indices = active_indices[dead_mask]
            self.bank.state.is_active[dead_indices] = False
            return dead_indices
        return np.array([], dtype=Index)

    def _calculate_time_step(self, num_to_inject: int) -> Float:
        total_activity = float(sum(source_node.get_activity(self.global_timer) for source_node in self.active_sources))
        if total_activity <= 0.0:
            return Float(self.stop_time - self.global_timer)

        # Начальное приближение: линейная экстраполяция
        delta_time = Float(num_to_inject / total_activity)

        # Метод Ньютона-Рафсона
        for _ in range(self.MAX_NEWTON_ITERATIONS):
            target_time = Float(self.global_timer + delta_time)
            function_value = (
                float(
                    sum(
                        source_node.get_expected_particles(self.global_timer, target_time)
                        for source_node in self.active_sources
                    )
                )
                - num_to_inject
            )
            if abs(function_value) < self.NEWTON_TARGET_PRECISION:
                break
            derivative_value = float(
                sum(source_node.get_activity(target_time) for source_node in self.active_sources)
            )
            if derivative_value <= 0.0:
                break
            delta_time = Float(delta_time - Float(function_value / derivative_value))

        return max(delta_time, self.MIN_TIME_STEP)

    def _distribute_quotas(self, target_time: Float, num_to_inject: int) -> None:
        expected = np.array(
            [source_node.get_expected_particles(self.global_timer, target_time) for source_node in self.active_sources],
            dtype=Float,
        )
        if np.sum(expected) <= 0.0:
            return
        target_total = num_to_inject

        quotas = np.floor(expected).astype(int)
        remainders = expected - quotas

        shortfall = int(target_total - np.sum(quotas))
        if shortfall > 0:
            indices = np.argsort(remainders)[::-1]
            for shortfall_idx in range(shortfall):
                quotas[indices[shortfall_idx % len(indices)]] += 1

        for source_obj, quota_count in zip(self.active_sources, quotas):
            if int(quota_count) > 0:
                source_obj.inject(self.bank, int(quota_count), self.global_timer, target_time)

    def _replenish_bank(self) -> None:
        if not self.active_sources or self.global_timer >= self.stop_time:
            return

        num_active = int(np.count_nonzero(self.bank.state.is_active))
        num_to_inject = int(self.bank.capacity - num_active)
        if num_to_inject <= 0:
            return

        delta_time = self._calculate_time_step(num_to_inject)
        target_time = Float(self.global_timer + delta_time)

        self._distribute_quotas(target_time, num_to_inject)

        self.global_timer = target_time

    def next_step(self) -> None:
        """Выполняет один шаг пополнения банка частиц, переноса и сброса буферов."""
        self._replenish_bank()

        active_indices = self.bank.active_indices
        if active_indices.size == 0:
            return

        self._ensure_step_capacity(active_indices.size)
        self.propagator.step(
            self.bank,
            self.data_buffer,
            self.geometry_buffer,
            self.physics_buffer,
            self.rng_ctx,
            active_indices=active_indices,
        )

        dead_indices = self._apply_invalidators(active_indices)
        if dead_indices.size > 0:
            self._record_finished_histories(dead_indices)

        self._check_no_overflow()
        self.step_count += 1

    def _record_finished_histories(self, dead_indices: NDArray[Index]) -> None:
        """Регистрирует завершившие историю частицы во всех включённых каналах."""
        bank = self.bank
        buffers = self.data_buffer
        _push_uncollided_to_initial_state_kernel(
            bank.initial_state, bank.state, dead_indices, buffers.initial_states
        )
        if buffers.escaped_particles.is_enabled:
            escaped_indices = dead_indices[bank.navigation_state.current_volume[dead_indices] < 0]
            if escaped_indices.size > 0:
                _push_to_escaped_particles_kernel(
                    bank.initial_state, bank.state, escaped_indices, buffers.escaped_particles
                )
        _push_to_dead_particles_kernel(bank.initial_state, dead_indices, buffers.dead_particles)

    def run(self) -> None:
        """Запуск цикла симуляции в рабочем потоке."""
        _logger.info(
            "%s started from %s to %s",
            self.name,
            timedelta(seconds=float(self.global_timer / units.second)),
            timedelta(seconds=float(self.stop_time / units.second)),
        )
        start_timepoint = datetime.now()
        if self._state != SimulationState.STOPPED and self._state != SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

        while (self.has_pending_sources or self.has_active_particles) and not self.is_stopped:
            self._run_event.wait()
            if self.is_stopped:
                break

            self.next_step()
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(
                    "Global timer of %s at %s",
                    self.name,
                    timedelta(seconds=float(self.global_timer / units.second)),
                )

        # Финальный сброс телеметрии
        self.flush_all()
        self._put_to_queue("stop")
        self._state = SimulationState.STOPPED

        stop_timepoint = datetime.now()
        _logger.info(
            "%s finished at %s",
            self.name,
            timedelta(seconds=float(self.global_timer / units.second)),
        )
        _logger.info("The simulation of %s took %s", self.name, stop_timepoint - start_timepoint)


__all__ = [
    "DEFAULT_CONSUMED_CHUNK_TYPES",
    "SimulationState",
    "SimulationManager",
]

