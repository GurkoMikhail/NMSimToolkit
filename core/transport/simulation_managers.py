"""
Модуль менеджера моделирования (SimulationManager) и состояний жизненного цикла симуляции.
"""

from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import queue
from signal import SIGINT, signal
import threading
from typing import AbstractSet, Any, Callable, Dict, FrozenSet, List, Optional, Union

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_compiler import GeometryCompiler
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import ChunkType, RNGContext, SimulationDataBuffer
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


Queue = queue.Queue
Thread = threading.Thread


class SimulationManager(Thread):
    """
    Вычислительный менеджер симуляции, оптимизированный под Data-Oriented Design (SoA),
    с поддержкой непрерывной инжекции частиц и эффективного уплотнения буферов данных на месте.
    """

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
        self.step = 1
        self.daemon = True

        self.bank = ParticleBank.allocate(self.particles_number)
        self.bank.count_array[0] = Index(self.initial_particle_id)
        self.data_buffer = SimulationDataBuffer.allocate(
            interaction_capacity=effective_capacity if ChunkType.INTERACTIONS in self.consumed_chunk_types else 0,
            initial_state_capacity=effective_capacity if ChunkType.INITIAL_STATES in self.consumed_chunk_types else 0,
            dead_particles_capacity=effective_capacity if ChunkType.DEAD_PARTICLES in self.consumed_chunk_types else 0,
            escaped_particles_capacity=effective_capacity if ChunkType.ESCAPED_PARTICLES in self.consumed_chunk_types else 0,
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

        if threading.current_thread() is threading.main_thread():
            signal(SIGINT, self.sigint_handler)


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

    def sigint_handler(self, signum: int, frame: Any) -> None:
        """Обработчик сигнала прерывания SIGINT."""
        _logger.error(f'{self.name} interrupted at {timedelta(seconds=self.global_timer/units.second)}')
        self.stop()

    def send_data(self, data: Any) -> None:
        """Отправляет пакет телеметрии в выходную очередь симуляции."""
        self.queue.put(data)

    def flush_interactions(self) -> None:
        """
        Сбрасывает накопленный буфер взаимодействий в очередь телеметрии.
        """
        interaction_count = self.data_buffer.interactions.cursor_value
        if interaction_count == 0:
            return
        if ChunkType.INTERACTIONS not in self.consumed_chunk_types:
            self.data_buffer.interactions.reset_cursor()
            return

        _logger.debug(f'{self.name} flushing {interaction_count} interactions')

        chunk = {
            'type': ChunkType.INTERACTIONS,
            'data': self.data_buffer.interactions.flush_to_dict(clear=True)
        }
        self.send_data(chunk)

    def flush_escaped_particles(self) -> None:
        """
        Сбрасывает накопленный буфер вылетевших за пределы сцены частиц в очередь телеметрии.
        Перед отправкой гарантированно сбрасываются начальные состояния и взаимодействия.
        """
        escaped_count = self.data_buffer.escaped_particles.cursor_value
        if escaped_count == 0:
            return
        if ChunkType.ESCAPED_PARTICLES not in self.consumed_chunk_types:
            self.data_buffer.escaped_particles.reset_cursor()
            return

        self.flush_initial_states()
        self.flush_interactions()

        _logger.debug(f'{self.name} flushing {escaped_count} escaped particles')

        chunk = {
            'type': ChunkType.ESCAPED_PARTICLES,
            'data': self.data_buffer.escaped_particles.flush_to_dict(clear=True)
        }
        self.send_data(chunk)

    def flush_dead_particles(self) -> None:
        """
        Сбрасывает накопленные идентификаторы выбывших частиц в очередь телеметрии.
        Для сохранения строгой причинно-следственной связи перед отправкой выбывших частиц
        гарантированно сбрасываются все предшествующие начальные состояния, взаимодействия
        и вылетевшие частицы.
        """
        dead_count = self.data_buffer.dead_particles.cursor_value
        if dead_count == 0:
            return
        if ChunkType.DEAD_PARTICLES not in self.consumed_chunk_types:
            self.data_buffer.dead_particles.reset_cursor()
            return

        self.flush_initial_states()
        self.flush_interactions()
        self.flush_escaped_particles()

        _logger.debug(f'{self.name} flushing {dead_count} dead particles')
        chunk = {
            'type': ChunkType.DEAD_PARTICLES,
            'data': self.data_buffer.dead_particles.flush_to_array(clear=True)
        }
        self.send_data(chunk)

    def flush_initial_states(self) -> None:
        """
        Сбрасывает буфер начальных состояний частиц в очередь телеметрии.
        """
        initial_count = self.data_buffer.initial_states.cursor_value
        if initial_count == 0:
            return
        if ChunkType.INITIAL_STATES not in self.consumed_chunk_types:
            self.data_buffer.initial_states.reset_cursor()
            return

        _logger.debug(f'{self.name} flushing {initial_count} initial states')

        chunk = {
            'type': ChunkType.INITIAL_STATES,
            'data': self.data_buffer.initial_states.flush_to_dict(clear=True)
        }
        self.send_data(chunk)

    def flush_all(self) -> None:
        """
        Выполняет полный сброс всех буферов телеметрии в строгом причинно-следственном порядке:
        initial_states → interactions → escaped_particles → dead_particles.
        """
        self.flush_initial_states()
        self.flush_interactions()
        self.flush_escaped_particles()
        self.flush_dead_particles()

    def _invalidate_by_energy(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        return self.bank.state.energy[active_indices] < self.min_energy

    def _invalidate_by_volume(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        navigation_state = self.bank.navigation_state
        return (navigation_state.current_volume[active_indices] < 0) & (navigation_state.boundary_distance[active_indices] > 0.0)

    def _apply_invalidators(self, active_indices: NDArray[Index]) -> NDArray[Index]:
        dead_mask = np.zeros(len(active_indices), dtype=np.bool_)
        for invalidator in self.invalidators:
            dead_mask |= invalidator(active_indices)

        if np.any(dead_mask):
            dead_indices = active_indices[dead_mask]
            self.bank.state.is_active[dead_indices] = False
            return dead_indices
        return np.array([], dtype=Index)

    MAX_NEWTON_ITERATIONS = 5
    NEWTON_TARGET_PRECISION = 1e-3
    MIN_TIME_STEP = Float(1e-9)

    def _calculate_time_step(self, num_to_inject: int) -> Float:
        total_activity = float(sum(source_node.get_activity(self.global_timer) for source_node in self.active_sources))
        if total_activity <= 0.0:
            return Float(self.stop_time - self.global_timer)

        # Начальное приближение: линейная экстраполяция
        delta_time = Float(num_to_inject / total_activity)

        def objective_func(current_delta_time: Float) -> float:
            return (
                float(
                    sum(
                        source_node.get_activity(self.global_timer) * source_node._get_effective_dt(current_delta_time)
                        for source_node in self.active_sources
                    )
                )
                - num_to_inject
            )

        def derivative_func(current_delta_time: Float) -> float:
            return float(
                sum(source_node.get_activity(self.global_timer + current_delta_time) for source_node in self.active_sources)
            )

        # Метод Ньютона-Рафсона
        for _ in range(self.MAX_NEWTON_ITERATIONS):
            function_value = objective_func(delta_time)
            derivative_value = derivative_func(delta_time)
            if derivative_value == 0.0:
                break
            delta_time = Float(delta_time - Float(function_value / derivative_value))
            if abs(function_value) < self.NEWTON_TARGET_PRECISION:
                break

        delta_time = max(delta_time, self.MIN_TIME_STEP)
        return delta_time

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
        num_active = int(np.count_nonzero(self.bank.state.is_active))
        num_to_inject = int(self.bank.capacity - num_active)

        if num_to_inject <= 0 or not self.active_sources or self.global_timer >= self.stop_time:
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

        if len(active_indices) > self.data_buffer.initial_states.remaining_capacity:
            self.flush_initial_states()

        if len(active_indices) > self.data_buffer.interactions.remaining_capacity:
            self.flush_interactions()

        self.propagator.step(
            self.bank,
            self.data_buffer,
            self.geometry_buffer,
            self.physics_buffer,
            self.rng_ctx,
        )

        dead_indices = self._apply_invalidators(active_indices)

        if dead_indices.size > 0:
            self._collect_escaped_particles(dead_indices)
            self._handle_dead_particles(dead_indices)

        self.step += 1

    def _push_indices_with_flush(
        self,
        indices: NDArray[Index],
        buffer: Any,
        push_fn: Callable[[NDArray[Index]], None],
        flush_fn: Callable[[], None],
        flush_on_full: bool = False,
    ) -> None:
        """
        Записывает массив индексов слотов банка в целевой кольцевой буфер порциями,
        автоматически выполняя сброс буфера при нехватке свободного места.
        """
        total_count = int(indices.shape[0])
        if total_count == 0 or buffer.capacity <= 0:
            return

        buffer_capacity = buffer.capacity
        if total_count > buffer.remaining_capacity:
            flush_fn()

        if total_count <= buffer.remaining_capacity:
            push_fn(indices)
            if flush_on_full and buffer.remaining_capacity == 0:
                flush_fn()
        else:
            for offset in range(0, total_count, buffer_capacity):
                chunk_indices = indices[offset:offset + buffer_capacity]
                if int(chunk_indices.shape[0]) > buffer.remaining_capacity:
                    flush_fn()
                push_fn(chunk_indices)
                if flush_on_full and buffer.remaining_capacity == 0:
                    flush_fn()

    def _collect_escaped_particles(self, dead_indices: NDArray[Index]) -> None:
        """
        Накопление вылетевших за границы геометрии сцены частиц в предвыделенном кольцевом буфере
        EscapedParticlesBuffer со сбросом в очередь телеметрии по заполнению буфера.
        При активной подписке на INITIAL_STATES начальные состояния вылетевших без рассеяния
        частиц предварительно записываются в InitialStateBuffer.
        """
        if self.queue is None or self.data_buffer.escaped_particles.capacity <= 0:
            return

        escaped_mask = self.bank.navigation_state.current_volume[dead_indices] < 0
        if not np.any(escaped_mask):
            return

        escaped_indices = dead_indices[escaped_mask]

        self._push_indices_with_flush(
            indices=escaped_indices,
            buffer=self.data_buffer.initial_states,
            push_fn=lambda chunk: _push_uncollided_to_initial_state_kernel(
                self.bank.initial_state, self.bank.state, chunk, self.data_buffer.initial_states
            ),
            flush_fn=self.flush_initial_states,
        )

        self._push_indices_with_flush(
            indices=escaped_indices,
            buffer=self.data_buffer.escaped_particles,
            push_fn=lambda chunk: _push_to_escaped_particles_kernel(
                self.bank.initial_state, self.bank.state, chunk, self.data_buffer.escaped_particles
            ),
            flush_fn=self.flush_escaped_particles,
            flush_on_full=True,
        )

    def _handle_dead_particles(self, dead_indices: NDArray[Index]) -> None:
        """
        Запись идентификаторов поглощенных и выбывших частиц в кольцевой буфер завершенных историй.
        """
        self._push_indices_with_flush(
            indices=dead_indices,
            buffer=self.data_buffer.dead_particles,
            push_fn=lambda chunk: _push_to_dead_particles_kernel(
                self.bank.initial_state, chunk, self.data_buffer.dead_particles
            ),
            flush_fn=self.flush_dead_particles,
        )

    def run(self) -> None:
        """Запуск цикла симуляции в рабочем потоке."""
        self._run()

    def _run(self) -> None:
        _logger.info(f'{self.name} started from {timedelta(seconds=self.global_timer/units.second)} to {timedelta(seconds=self.stop_time/units.second)}')
        start_timepoint = datetime.now()
        if self._state != SimulationState.STOPPED and self._state != SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

        while (self.has_active_particles or self.has_pending_sources) and not self.is_stopped:
            self._run_event.wait()
            if self.is_stopped:
                break

            self.next_step()
            _logger.debug(f'Global timer of {self.name} at {timedelta(seconds=self.global_timer/units.second)}')

        # Финальный сброс телеметрии
        self.flush_all()
        self.queue.put('stop')
        self._state = SimulationState.STOPPED

        stop_timepoint = datetime.now()
        _logger.info(f'{self.name} finished at {timedelta(seconds=self.global_timer/units.second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')


__all__ = [
    "DEFAULT_CONSUMED_CHUNK_TYPES",
    "SimulationState",
    "SimulationManager",
]

