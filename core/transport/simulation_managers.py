from enum import Enum, auto
import logging
import queue
import threading as mt
from datetime import datetime, timedelta
from signal import SIGINT, signal
from typing import Callable, List, Optional, Union, Protocol

import numpy as np
import hepunits as units
from numpy.typing import NDArray


class PauseEventProtocol(Protocol):
    """
    Строгий контракт интерфейса события синхронизации (совместим с threading.Event,
    multiprocessing.synchronize.Event, mp.Manager().Event() и FocusedPauseProxy).
    """

    def is_set(self) -> bool:
        ...

    def set(self) -> None:
        ...

    def clear(self) -> None:
        ...

    def wait(self, timeout: Optional[float] = None) -> bool:
        ...


class SimulationState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()

from core.geometry.volumes import Volume
from core.geometry.geometry_compiler import GeometryCompiler
from core.physics.physics_compiler import PhysicsCompiler
from core.scene.nodes import CompositeNode
from core.source.source_compiler import SourceCompiler
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import SimulationDataBuffer, RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

Queue = queue.Queue
Thread = mt.Thread


class SimulationManager(Thread):
    """
    Вычислительный менеджер симуляции, оптимизированный под Data-Oriented Design (SoA),
    с поддержкой непрерывной инжекции частиц и эффективного уплотнения буферов данных на месте.
    """
    active_sources: List[Source]
    scene: CompositeNode
    propagator: ParticlePropagator
    stop_time: Float
    particles_number: int
    min_energy: Float
    queue: Queue
    bank: ParticleBank
    data_buffer: SimulationDataBuffer
    geometry_buffer: NDArray
    physics_buffer: PhysicsBuffer
    rng_ctx: RNGContext
    invalidators: List[Callable[[NDArray[Index]], NDArray[np.bool_]]]
    global_timer: Float
    pause_event: PauseEventProtocol

    def __init__(
        self,
        scene: CompositeNode,
        propagator: Optional[ParticlePropagator] = None,
        stop_time: Float = 1*units.s,
        start_time: Float = Float(0.0),
        particles_number: Union[int, Float] = 10**3,
        min_energy: Float = 1*units.keV,
        queue: Optional[Queue] = None,
        buffer_capacity: Optional[int] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        pause_event: Optional[PauseEventProtocol] = None,
    ) -> None:
        super().__init__()
        if name is not None:
            self.name = name
        self.scene = scene
        self.active_sources = SourceCompiler().compile_scene(scene)
        self.propagator = ParticlePropagator() if propagator is None else propagator

        self.geometry_buffer = GeometryCompiler().compile_scene(scene)
        self.physics_buffer = PhysicsCompiler().compile_scene(scene, self.propagator.processes)
        self.stop_time = stop_time
        self.particles_number = int(particles_number)
        effective_capacity = self.particles_number if buffer_capacity is None else max(int(buffer_capacity), self.particles_number)
        self.min_energy = min_energy
        self.queue = Queue(maxsize=64) if queue is None else queue
        self.step = 1
        self.daemon = True

        self.bank = ParticleBank.allocate(self.particles_number)
        self.data_buffer = SimulationDataBuffer.allocate(effective_capacity, effective_capacity, effective_capacity)
        if seed is not None:
            self.propagator.rng = np.random.default_rng(seed)
            for src in self.active_sources:
                src.rng = self.propagator.rng
        self.rng_ctx = RNGContext.from_numpy_rng(self.propagator.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]
        self.global_timer = Float(start_time)

        self._state: SimulationState = SimulationState.IDLE
        self._stop_event = mt.Event()
        if pause_event is not None:
            self.pause_event = pause_event
        else:
            self.pause_event = mt.Event()
            self.pause_event.set()

        try:
            signal(SIGINT, self.sigint_handler)
        except (ValueError, AttributeError):
            pass

    @property
    def state(self) -> SimulationState:
        return self._state

    def pause(self) -> None:
        """Приостанавливает выполнение моделирования."""
        self.pause_event.clear()
        self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Возобновляет приостановленное моделирование."""
        self._state = SimulationState.RUNNING
        self.pause_event.set()

    def stop(self) -> None:
        """Кооперативно останавливает процесс моделирования."""
        self._stop_event.set()
        self.pause_event.set()
        self._state = SimulationState.STOPPED

    def step_once(self) -> None:
        """Выполняет один шаг моделирования для покадрового анализа."""
        self.next_step()
        self.flush_all()

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {timedelta(seconds=self.global_timer/units.second)}')
        self.stop()

    def send_data(self, data):
        # We need to copy or view the interaction data up to cursor
        # Actually in production we should extract recarray from SoA InteractionBuffer
        # but for now we just pass a copy or slice.
        self.queue.put(data)

    def flush_interactions(self) -> None:
        """
        Сбрасывает накопленный буфер взаимодействий в очередь телеметрии.
        """
        interaction_count = self.data_buffer.interactions.cursor_value
        if interaction_count == 0:
            return

        _logger.debug(f'{self.name} flushing {interaction_count} interactions')

        chunk = {
            'type': 'interactions',
            'data': self.data_buffer.interactions.flush_to_dict(clear=True)
        }
        self.send_data(chunk)

    def flush_dead_particles(self) -> None:
        """
        Сбрасывает накопленные идентификаторы выбывших частиц в очередь телеметрии.
        Для сохранения строгой причинно-следственной связи перед отправкой выбывших частиц
        гарантированно сбрасываются все предшествующие начальные состояния и взаимодействия.
        """
        dead_count = self.data_buffer.dead_particles.cursor_value
        if dead_count == 0:
            return

        self.flush_initial_states()
        self.flush_interactions()

        _logger.debug(f'{self.name} flushing {dead_count} dead particles')
        chunk = {
            'type': 'dead_particles',
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

        _logger.debug(f'{self.name} flushing {initial_count} initial states')

        chunk = {
            'type': 'initial_states',
            'data': self.data_buffer.initial_states.flush_to_dict(clear=True)
        }
        self.send_data(chunk)

    def flush_all(self) -> None:
        """
        Выполняет полный сброс всех буферов телеметрии в строгом причинно-следственном порядке:
        initial_states → interactions → dead_particles.
        """
        self.flush_initial_states()
        self.flush_interactions()
        self.flush_dead_particles()

    def _invalidate_by_energy(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        return self.bank.state.energy[active_indices] < self.min_energy

    def _invalidate_by_volume(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        nav = self.bank.navigation_state
        return (nav.current_volume[active_indices] < 0) & (nav.boundary_distance[active_indices] > 0.0)

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
        total_act = sum(src.get_activity(self.global_timer) for src in self.active_sources)
        if total_act <= 0:
            return Float(self.stop_time - self.global_timer)

        # Initial guess: linear approximation
        dt = Float(num_to_inject / total_act)

        def objective_func(current_dt: Float) -> float:
            return sum(src.get_activity(self.global_timer) * src._get_effective_dt(current_dt) for src in self.active_sources) - num_to_inject

        def derivative_func(current_dt: Float) -> float:
            return sum(src.get_activity(self.global_timer + current_dt) for src in self.active_sources)

        # Newton-Raphson
        for _ in range(self.MAX_NEWTON_ITERATIONS):
            func_val = objective_func(dt)
            deriv_val = derivative_func(dt)
            if deriv_val == 0:
                break
            dt = dt - Float(func_val / deriv_val)
            if abs(func_val) < self.NEWTON_TARGET_PRECISION:
                break

        dt = max(dt, self.MIN_TIME_STEP)

        # Clamp dt to not exceed simulation stop_time
        return min(dt, Float(self.stop_time - self.global_timer))

    def _distribute_quotas(self, target_time: Float, num_to_inject: int):
        expected = np.array([src.get_expected_particles(self.global_timer, target_time) for src in self.active_sources])
        target_total = min(num_to_inject, int(np.round(np.sum(expected))))

        quotas = np.floor(expected).astype(int)
        remainders = expected - quotas

        shortfall = target_total - np.sum(quotas)
        if shortfall > 0:
            indices = np.argsort(remainders)[::-1]
            for i in range(int(shortfall)):
                quotas[indices[i % len(indices)]] += 1

        for source_obj, quota_count in zip(self.active_sources, quotas):
            if quota_count > 0:
                source_obj.inject(self.bank, quota_count, self.global_timer, target_time)

    def _replenish_bank(self):
        num_active = np.count_nonzero(self.bank.state.is_active)
        num_to_inject = self.bank.capacity - num_active

        if num_to_inject <= 0 or not self.active_sources or self.global_timer >= self.stop_time:
            return

        dt = self._calculate_time_step(num_to_inject)
        target_time = self.global_timer + dt

        self._distribute_quotas(target_time, num_to_inject)

        self.global_timer = target_time

    def next_step(self):
        self._replenish_bank()

        active_indices = self.bank.active_indices

        if active_indices.size == 0:
            return

        # Pre-flight Check: Ensure buffer has enough space for a worst-case scenario
        if len(active_indices) > self.data_buffer.initial_states.remaining_capacity:
            self.flush_initial_states()

        if len(active_indices) > self.data_buffer.interactions.remaining_capacity:
            self.flush_interactions()

        # Step physics and kinematics
        self.propagator.step(
            self.bank,
            self.data_buffer,
            self.geometry_buffer,
            self.physics_buffer,
            self.rng_ctx
        )

        # Invalidation
        dead_indices = self._apply_invalidators(active_indices)

        if dead_indices.size > 0:
            self._collect_escaped_particles(dead_indices)
            self._handle_dead_particles(dead_indices)

        self.step += 1

    def _collect_escaped_particles(self, dead_indices: np.ndarray) -> None:
        """
        Регистрация вылетевших частиц за границы геометрии сцены и отправка в очередь телеметрии.
        """
        if self.queue is None:
            return

        escaped_mask = (self.bank.navigation_state.current_volume[dead_indices] < 0)
        if not np.any(escaped_mask):
            return

        esc_idx = dead_indices[escaped_mask]
        birth_x = self.bank.initial_state.emission_position.x[esc_idx]
        birth_y = self.bank.initial_state.emission_position.y[esc_idx]
        birth_z = self.bank.initial_state.emission_position.z[esc_idx]
        esc_x = self.bank.state.position.x[esc_idx]
        esc_y = self.bank.state.position.y[esc_idx]
        esc_z = self.bank.state.position.z[esc_idx]
        esc_pids = self.bank.initial_state.ID[esc_idx]
        has_interacted = self.bank.initial_state.has_interacted[esc_idx]

        esc_chunk = {
            'type': 'escaped_particles',
            'data': {
                'birth_x': np.array(birth_x, copy=True),
                'birth_y': np.array(birth_y, copy=True),
                'birth_z': np.array(birth_z, copy=True),
                'pos_x': np.array(esc_x, copy=True),
                'pos_y': np.array(esc_y, copy=True),
                'pos_z': np.array(esc_z, copy=True),
                'particle_id': np.array(esc_pids, copy=True),
                'has_interacted': np.array(has_interacted, copy=True),
            }
        }
        self.send_data(esc_chunk)

    def _handle_dead_particles(self, dead_indices: np.ndarray) -> None:
        """
        Запись идентификаторов поглощенных и выбывших частиц в кольцевой буфер завершенных историй.
        """
        dead_ids = self.bank.initial_state.ID[dead_indices]
        buffer_capacity = self.data_buffer.dead_particles.capacity
        if len(dead_ids) > self.data_buffer.dead_particles.remaining_capacity:
            self.flush_dead_particles()

        if len(dead_ids) <= self.data_buffer.dead_particles.remaining_capacity:
            self.data_buffer.dead_particles.append(dead_ids)
        else:
            for idx in range(0, len(dead_ids), buffer_capacity):
                chunk_slice = dead_ids[idx:idx + buffer_capacity]
                if len(chunk_slice) > self.data_buffer.dead_particles.remaining_capacity:
                    self.flush_dead_particles()
                self.data_buffer.dead_particles.append(chunk_slice)

    def run(self) -> None:
        """Запуск цикла симуляции в рабочем потоке."""
        self._run()

    def _run(self):
        _logger.warning(f'{self.name} started from {timedelta(seconds=self.global_timer/units.second)} to {timedelta(seconds=self.stop_time/units.second)}')
        start_timepoint = datetime.now()
        self._state = SimulationState.RUNNING
        self._stop_event.clear()

        while (np.count_nonzero(self.bank.state.is_active) > 0 or (self.active_sources and self.global_timer < self.stop_time)) and not self._stop_event.is_set():
            if not self.pause_event.is_set():
                self._state = SimulationState.PAUSED
                while not self.pause_event.is_set() and not self._stop_event.is_set():
                    self.pause_event.wait(timeout=0.05)
                if self._stop_event.is_set():
                    break
                self._state = SimulationState.RUNNING

            self.next_step()
            _logger.debug(f'Global timer of {self.name} at {timedelta(seconds=self.global_timer/units.second)}')

        # Final flush
        self.flush_all()
        self.queue.put('stop')
        self._state = SimulationState.STOPPED

        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {timedelta(seconds=self.global_timer/units.second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')
