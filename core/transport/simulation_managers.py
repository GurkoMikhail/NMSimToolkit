import logging
import queue
import threading as mt
from cProfile import runctx
from datetime import datetime
from signal import SIGINT, signal
from typing import Callable, List, Optional, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

from core.geometry.volumes import Volume
from core.scene.nodes import CompositeNode
from core.source.source_compiler import SourceCompiler
from core.other.typing_definitions import Float, Index
from core.other.utils import datetime_from_seconds
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
    DOD-optimized Simulation Manager with Continuous Injection
    and in-place Stream Compaction handling.
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

    def __init__(
        self,
        scene: CompositeNode,
        propagator: Optional[ParticlePropagator] = None,
        stop_time: Float = 1*units.s,
        particles_number: Union[int, Float] = 10**3,
        queue: Optional[Queue] = None,
        buffer_capacity: int = 100000
    ) -> None:
        super().__init__()
        self.scene = scene
        self.active_sources = SourceCompiler().compile_scene(scene)
        self.propagator = ParticlePropagator() if propagator is None else propagator

        from core.geometry.geometry_compiler import GeometryCompiler
        from core.physics.physics_compiler import PhysicsCompiler

        self.geometry_buffer = GeometryCompiler().compile_scene(scene)
        self.physics_buffer = PhysicsCompiler().compile_scene(scene, self.propagator.processes)
        self.stop_time = stop_time
        self.particles_number = int(particles_number)
        self.min_energy = 1*units.keV
        self.queue = Queue(maxsize=1) if queue is None else queue
        self.step = 1
        self.profile = False
        self.daemon = True

        self.bank = ParticleBank.allocate(self.particles_number)
        self.data_buffer = SimulationDataBuffer.allocate(buffer_capacity, buffer_capacity, buffer_capacity)
        self.rng_ctx = RNGContext.from_numpy_rng(self.propagator.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]

        signal(SIGINT, self.sigint_handler)

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {datetime_from_seconds((self.active_sources[0].timer if self.active_sources else 0.0)/units.second)}')
        self.stop_time = 0

    def send_data(self, data):
        # We need to copy or view the interaction data up to cursor
        # Actually in production we should extract recarray from SoA InteractionBuffer
        # but for now we just pass a copy or slice.
        self.queue.put(data)

    def flush_interactions(self) -> None:
        """
        Flushes only the interaction buffer to the queue if it's full.
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
        Flushes accumulated dead particle IDs to the queue.
        """
        dead_count = self.data_buffer.dead_particles.cursor_value
        if dead_count == 0:
            return

        _logger.debug(f'{self.name} flushing {dead_count} dead particles')
        chunk = {
            'type': 'dead_particles',
            'data': self.data_buffer.dead_particles.flush_to_array(clear=True)
        }
        self.send_data(chunk)

    def flush_initial_states(self) -> None:
        """
        Flushes only the initial states buffer to the queue if it's full.
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

    def next_step(self):
        active_indices = self.bank.active_indices

        if active_indices.size == 0:
            return

        # Pre-flight Check: Ensure buffer has enough space for a worst-case scenario
        if len(active_indices) > self.data_buffer.interactions.remaining_capacity:
            self.flush_interactions()

        if len(active_indices) > self.data_buffer.initial_states.remaining_capacity:
            self.flush_initial_states()

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
            if len(dead_indices) > self.data_buffer.dead_particles.remaining_capacity:
                self.flush_interactions()
                self.flush_initial_states()
                self.flush_dead_particles()

            dead_ids = self.bank.initial_state.ID[dead_indices]
            self.data_buffer.dead_particles.append(dead_ids)

        # Continuous Replenishment
        if self.active_sources and self.active_sources[0].timer <= self.stop_time:
            num_active = np.count_nonzero(self.bank.state.is_active)
            num_to_inject = self.bank.capacity - num_active

            if num_to_inject > 0:
                total_activity = sum(src.activity for src in self.active_sources)
                if total_activity > 0:
                    for src in self.active_sources:
                        n = int(num_to_inject * (src.activity / total_activity))
                        if n > 0:
                            src.inject(self.bank, n)

        self.step += 1

    def run(self):
        if self.profile:
            self.run_profile()
        else:
            self._run()

    def run_profile(self):
        runctx('self._run()', globals(), locals(), f'stats/{self.name}.txt')

    def _run(self):
        _logger.warning(f'{self.name} started from {datetime_from_seconds((self.active_sources[0].timer if self.active_sources else 0.0)/units.second)} to {datetime_from_seconds(self.stop_time/units.second)}')
        start_timepoint = datetime.now()

        # Initial injection
        total_activity = sum(src.activity for src in self.active_sources)
        if total_activity > 0:
            for src in self.active_sources:
                n = int(self.particles_number * (src.activity / total_activity))
                if n > 0:
                    src.inject(self.bank, n)

        while np.count_nonzero(self.bank.state.is_active) > 0 or (self.active_sources and self.active_sources[0].timer <= self.stop_time):
            self.next_step()
            _logger.debug(f'Source timer of {self.name} at {datetime_from_seconds((self.active_sources[0].timer if self.active_sources else 0.0)/units.second)}')

        # Final flush
        self.flush_interactions()
        self.flush_initial_states()
        self.flush_dead_particles()
        self.queue.put('stop')

        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {datetime_from_seconds((self.active_sources[0].timer if self.active_sources else 0.0)/units.second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')
