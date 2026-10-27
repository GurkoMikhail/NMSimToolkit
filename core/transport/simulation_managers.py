"""
Модуль менеджера моделирования (SimulationManager) и состояний жизненного цикла симуляции.
"""

from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import math
from queue import Full, Queue
import threading
from typing import AbstractSet, Any, Callable, ClassVar, FrozenSet, List, Mapping, Optional, Tuple, Union

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_compiler import GeometryCompiler
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import (
    TELEMETRY_ORDER,
    ChunkType,
    RNGContext,
    SecondaryActivationBuffer,
    SimulationDataBuffer,
)
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.scene.nodes import CompositeNode
from core.source.primary_source import PrimarySource
from core.source.secondary_source import SecondarySource
from core.source.source_buffer import SourceBuffer
from core.source.source_compiler import SourceCompiler
from core.source.sources import ParticleSource, Source
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
    с поддержкой двухфазного пополнения банка (наведенные вторичные источники + пуассоновский
    поток первичных распадов с прореживанием Льюиса–Шедлера), единого генератора случайных
    чисел и точной обрезки по stop_time.
    """

    QUEUE_PUT_TIMEOUT: ClassVar[float] = 0.05

    active_sources: List[Source]
    source_buffer: SourceBuffer
    scene: CompositeNode
    propagator: ParticlePropagator
    start_time: Float
    stop_time: Float
    initial_particle_id: int
    particles_number: int
    batch_capacity: int
    min_energy: Float
    enable_atomic_relaxation: bool
    photon_production_cut: Float
    electron_production_cut: Float
    queue: Queue[Any]
    consumed_chunk_types: FrozenSet[ChunkType]
    step_count: int
    bank: ParticleBank
    data_buffer: SimulationDataBuffer
    primary_source: PrimarySource
    secondary_source: SecondarySource
    sources: Tuple[ParticleSource, ...]
    geometry_buffer: NDArray[Any]
    physics_buffer: PhysicsBuffer
    rng: np.random.Generator
    rng_ctx: RNGContext
    invalidators: List[Callable[[NDArray[Index]], NDArray[np.bool_]]]
    _stop_by_time: bool
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
        enable_atomic_relaxation: bool = False,
        photon_production_cut: Union[float, Float] = Float(1.0 * units.keV),
        electron_production_cut: Union[float, Float] = Float(np.inf),
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
        self.enable_atomic_relaxation = bool(enable_atomic_relaxation)
        self.photon_production_cut = Float(photon_production_cut)
        self.electron_production_cut = Float(electron_production_cut)
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
        self.primary_source = PrimarySource(
            source_buffer=self.source_buffer,
            start_time=self.start_time,
            stop_time=self.stop_time,
            max_particles=-1 if self._stop_by_time else self.particles_number,
        )
        self.secondary_source = SecondarySource(
            activation_buffer=SecondaryActivationBuffer.allocate(
                capacity=2 * self.batch_capacity,
                enable_atomic_relaxation=self.enable_atomic_relaxation,
                photon_production_cut=self.photon_production_cut,
                electron_production_cut=self.electron_production_cut,
            ),
            relaxation_bank=self.physics_buffer.relaxation_bank,
            stop_time=self.stop_time,
        )
        self.sources = (self.secondary_source, self.primary_source)

        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(int(seed))
        else:
            self.rng = self.propagator.rng

        self.propagator.rng = self.rng
        self.rng_ctx = RNGContext.from_numpy_rng(self.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]

        self._state = SimulationState.IDLE
        self._run_event = threading.Event()
        self._run_event.set()
        if resume_state is not None:
            self.restore_checkpoint(resume_state)

    def restore_checkpoint(self, resume_state: Mapping[str, Any]) -> None:
        """Восстанавливает состояние ГСЧ, счетчиков частиц, потока эмиссии и вторичных активаций из чекпоинта."""
        self.initial_particle_id = int(resume_state.get("next_particle_id", resume_state.get("initial_particle_id", 0)))
        self.bank.count_array[0] = Index(self.initial_particle_id)
        self.primary_source.restore_state(resume_state)
        if resume_state.get("rng_state") is not None:
            self.rng.bit_generator.state = resume_state["rng_state"]
            self.rng_ctx = RNGContext.from_numpy_rng(self.rng)
        if "secondary_activations" in resume_state:
            self.secondary_source.restore_state(resume_state.get("secondary_activations"))

    @property
    def t_ref(self) -> Float:
        """Опорное время для мажоранты интенсивности первичного пуассоновского потока."""
        return self.primary_source.t_ref

    @t_ref.setter
    def t_ref(self, value: Float) -> None:
        self.primary_source.t_ref = Float(value)

    @property
    def t_last(self) -> Float:
        """Текущее время последнего принятого/проверенного кандидата первичного потока."""
        return self.primary_source.t_last

    @t_last.setter
    def t_last(self, value: Float) -> None:
        self.primary_source.t_last = Float(value)

    @property
    def pending_arrival_time(self) -> Float:
        """Отложенный кандидат времени вылета первичной частицы на границе батча или stop_time."""
        return self.primary_source.pending_arrival_time

    @pending_arrival_time.setter
    def pending_arrival_time(self, value: Float) -> None:
        self.primary_source.pending_arrival_time = Float(value)

    @property
    def emitted_primaries(self) -> int:
        """Количество сгенерированных первичных частиц."""
        return self.primary_source.emitted_particles

    @emitted_primaries.setter
    def emitted_primaries(self, value: int) -> None:
        self.primary_source.emitted_particles = int(value)

    @property
    def global_timer(self) -> Float:
        """Текущее глобальное время симуляции."""
        return self.primary_source.t_last

    @global_timer.setter
    def global_timer(self, value: Float) -> None:
        self.primary_source.t_last = Float(value)

    @property
    def activation_buffer(self) -> SecondaryActivationBuffer:
        """Текущий предвыделенный буфер наведенных вторичных активаций."""
        return self.secondary_source.activation_buffer

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
        """Проверяет наличие ожидающих эмиссии частиц хотя бы в одном рантайм-источнике."""
        return any(source.has_pending for source in self.sources)

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
        """
        Последовательно опрашивает рантайм-источники ParticleSource (SecondarySource -> PrimarySource)
        для заполнения свободных слотов ParticleBank с передачей эстафетного курсора slot_cursor.
        """
        slot_cursor = Index(0)
        for source in self.sources:
            if source.has_pending:
                _, slot_cursor = source.replenish_bank(
                    self.bank,
                    self.rng_ctx,
                    start_slot_cursor=slot_cursor,
                )

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
            activation_buffer=self.activation_buffer,
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
