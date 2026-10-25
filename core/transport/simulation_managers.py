"""
Модуль менеджера моделирования (SimulationManager) и состояний жизненного цикла симуляции.
"""

from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import math
from queue import Full, Queue
import threading
from typing import AbstractSet, Any, Callable, ClassVar, FrozenSet, List, Mapping, Optional, Union

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
from core.source.source_buffer import SourceBuffer
from core.source.source_compiler import SourceCompiler
from core.source.source_kernels import inject_poisson_sources_kernel
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
    с поддержкой пуассоновского потока распадов (прореживание Льюиса–Шедлера),
    единого генератора случайных чисел и точной обрезки по stop_time.
    """

    QUEUE_PUT_TIMEOUT: ClassVar[float] = 0.05

    active_sources: List[Source]
    source_buffer: SourceBuffer
    scene: CompositeNode
    propagator: ParticlePropagator
    start_time: Float
    stop_time: Float
    t_ref: Float
    t_last: Float
    pending_arrival_time: Float
    initial_particle_id: int
    particles_number: int
    batch_capacity: int
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
    _stop_by_time: bool
    _emission_finished: bool
    _run_event: threading.Event
    _state: SimulationState

    def __init__(
        self,
        scene: CompositeNode,
        propagator: Optional[ParticlePropagator] = None,
        stop_time: Optional[Union[float, Float]] = Float(1.0 * units.s),
        start_time: Union[float, Float] = Float(0.0),
        particles_number: Union[int, float, Float] = 10**3,
        min_energy: Union[float, Float] = Float(1.0 * units.keV),
        queue: Optional[Queue[Any]] = None,
        buffer_capacity: Optional[int] = None,
        name: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        batch_capacity: Optional[int] = None,
        consumed_chunk_types: Optional[AbstractSet[ChunkType]] = None,
        bank: Optional[ParticleBank] = None,
        resume_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if name is not None:
            self.name = name
        self.scene = scene
        source_compiler = SourceCompiler()
        self.active_sources = source_compiler.compile_scene(scene)
        self.source_buffer = source_compiler.compile(self.active_sources)
        self.propagator = ParticlePropagator() if propagator is None else propagator

        self.geometry_buffer = GeometryCompiler().compile_scene(scene)
        self.physics_buffer = PhysicsCompiler().compile_scene(scene, self.propagator.processes)
        self.start_time = Float(start_time)
        if stop_time is None:
            self.stop_time = Float(np.inf)
            self._stop_by_time = False
        else:
            self.stop_time = Float(stop_time)
            self._stop_by_time = bool(math.isfinite(float(self.stop_time)))

        self.t_ref = self.start_time
        self.t_last = self.start_time
        self.pending_arrival_time = Float(np.nan)
        self.initial_particle_id = 0
        self.particles_number = int(particles_number)
        self.batch_capacity = self.particles_number if batch_capacity is None else int(batch_capacity)
        if self.batch_capacity <= 0:
            raise ValueError(
                f"Емкость пула частиц (batch_capacity) должна быть строго положительной (> 0), получено {self.batch_capacity}."
            )
        effective_capacity = (
            self.batch_capacity
            if buffer_capacity is None
            else max(int(buffer_capacity), self.batch_capacity)
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

        self.bank = ParticleBank.allocate(self.batch_capacity) if bank is None else bank
        if self.bank.capacity != self.batch_capacity:
            raise ValueError(
                f"Несовпадение batch_capacity ({self.batch_capacity}) и емкости ParticleBank ({self.bank.capacity})."
            )
        self.bank.count_array[0] = Index(self.initial_particle_id)
        self.data_buffer = SimulationDataBuffer.allocate_for(
            capacity=effective_capacity,
            enabled=self.consumed_chunk_types,
        )

        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(int(seed))
        else:
            self.rng = self.propagator.rng

        self.propagator.rng = self.rng
        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]
        self.global_timer = Float(self.start_time)

        if not self.active_sources:
            self._emission_finished = True
        elif self._stop_by_time:
            self._emission_finished = bool(self.start_time >= self.stop_time)
        else:
            self._emission_finished = bool(self.particles_number <= 0)

        self._state = SimulationState.IDLE
        self._run_event = threading.Event()
        self._run_event.set()
        if resume_state is not None:
            self.restore_checkpoint(resume_state)

    def restore_checkpoint(self, resume_state: Mapping[str, Any]) -> None:
        """Восстанавливает состояние ГСЧ, счетчика частиц и потока эмиссии из чекпоинта."""
        self.initial_particle_id = int(resume_state.get("next_particle_id", resume_state.get("initial_particle_id", 0)))
        self.bank.count_array[0] = Index(self.initial_particle_id)
        self.t_ref = Float(self.start_time if resume_state.get("t_ref") is None else resume_state["t_ref"])
        self.t_last = Float(self.start_time if resume_state.get("t_last") is None else resume_state["t_last"])
        if resume_state.get("pending_arrival_time") is not None:
            self.pending_arrival_time = Float(resume_state["pending_arrival_time"])
        if resume_state.get("rng_state") is not None:
            self.rng.bit_generator.state = resume_state["rng_state"]
            self.rng_ctx = RNGContext.from_numpy_rng(self.rng)

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
        """Проверяет наличие активных источников частиц до достижения лимита времени или числа частиц."""
        if not self.active_sources or self._emission_finished:
            return False
        if self._stop_by_time:
            return bool(self.global_timer < self.stop_time)
        emitted_so_far = int(self.bank.count_array[0]) - self.initial_particle_id
        return emitted_so_far < self.particles_number

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

    def _replenish_bank(self) -> None:
        if not self.has_pending_sources:
            return

        if self._stop_by_time:
            _, _, self.t_last, self.pending_arrival_time = inject_poisson_sources_kernel(
                self.bank,
                self.source_buffer,
                self.stop_time,
                self.t_ref,
                self.t_last,
                self.pending_arrival_time,
                self.rng_ctx,
            )
            if not math.isnan(self.pending_arrival_time) and self.pending_arrival_time >= self.stop_time:
                self.t_last = self.stop_time
                self.global_timer = self.stop_time
                self._emission_finished = True
            else:
                self.global_timer = self.t_last
        else:
            emitted_so_far = int(self.bank.count_array[0]) - self.initial_particle_id
            remaining_particles = self.particles_number - emitted_so_far
            if remaining_particles <= 0:
                self._emission_finished = True
                return
            _, _, self.t_last, self.pending_arrival_time = inject_poisson_sources_kernel(
                self.bank,
                self.source_buffer,
                Float(np.inf),
                self.t_ref,
                self.t_last,
                self.pending_arrival_time,
                self.rng_ctx,
                max_particles=Index(remaining_particles),
            )
            self.global_timer = self.t_last
            if (
                int(self.bank.count_array[0]) - self.initial_particle_id >= self.particles_number
                or math.isinf(float(self.pending_arrival_time))
            ):
                self._emission_finished = True

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
        stop_label = (
            str(timedelta(seconds=float(self.stop_time / units.second)))
            if self._stop_by_time
            else f"{self.particles_number} particles"
        )
        _logger.info(
            "%s started from %s to %s",
            self.name,
            timedelta(seconds=float(self.global_timer / units.second)),
            stop_label,
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

