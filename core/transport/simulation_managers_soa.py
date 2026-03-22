import logging
import queue
import threading as mt
from cProfile import runctx
from datetime import datetime
from signal import SIGINT, signal
from typing import Any, Callable, List, Optional, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

from core.geometry.volumes import Volume
from core.other.typing_definitions import Float, Index
from core.other.utils import datetime_from_seconds
from core.particles.particles_soa import ParticleBank
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.source.sources_soa import SourceSoA
from core.transport.propagator_soa import ParticlePropagator

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

Queue = queue.Queue
Thread = mt.Thread

class SimulationManagerSOA(Thread):
    """
    DOD-optimized Simulation Manager with Continuous Injection
    and in-place Stream Compaction handling.
    """
    source: SourceSoA
    simulation_volume: Volume
    propagator: ParticlePropagator
    stop_time: Float
    particles_number: int
    min_energy: Float
    queue: Queue
    bank: ParticleBank
    interaction_buffer: InteractionBuffer
    geometry_buffer: NDArray
    physics_buffer: PhysicsBuffer
    rng_ctx: RNGContext
    invalidators: List[Callable[[NDArray[Index]], NDArray[np.bool_]]]

    def __init__(
        self,
        source: SourceSoA,
        simulation_volume: Volume,
        geometry_buffer: NDArray,
        physics_buffer: PhysicsBuffer,
        propagator: Optional[ParticlePropagator] = None,
        stop_time: Float = 1*units.s,
        particles_number: Union[int, Float] = 10**3,
        queue: Optional[Queue] = None,
        buffer_capacity: int = 100000
    ) -> None:
        super().__init__()
        self.source = source
        self.simulation_volume = simulation_volume
        self.geometry_buffer = geometry_buffer
        self.physics_buffer = physics_buffer
        self.propagator = ParticlePropagator() if propagator is None else propagator
        self.stop_time = stop_time
        self.particles_number = int(particles_number)
        self.min_energy = 1*units.keV
        self.queue = Queue(maxsize=1) if queue is None else queue
        self.step = 1
        self.profile = False
        self.daemon = True

        self.bank = ParticleBank.allocate(self.particles_number)
        self.interaction_buffer = InteractionBuffer.allocate(buffer_capacity)
        self.rng_ctx = RNGContext.from_numpy_rng(self.propagator.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]

        signal(SIGINT, self.sigint_handler)

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {datetime_from_seconds(self.source.timer/units.second)}')
        self.stop_time = 0

    def send_data(self, data):
        # We need to copy or view the interaction data up to cursor
        # Actually in production we should extract recarray from SoA InteractionBuffer
        # but for now we just pass a copy or slice.
        self.queue.put(data)

    def flush_buffer(self) -> None:
        """
        Flushes the interaction buffer to the queue if it's full.
        To avoid complex SoA-to-AoS conversion here, we can just trigger a queue put.
        """
        count = self.interaction_buffer.cursor[0]
        if count == 0:
            return

        _logger.debug(f'{self.name} flushing {count} events')
        # Emulate flushing for compatibility with downstream
        # Ideally, we convert InteractionBuffer slice to InteractionArray here
        self.send_data("FLUSH_SIGNAL")
        self.interaction_buffer.cursor[0] = 0


    def _invalidate_by_energy(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        return self.bank.state.energy[active_indices] < self.min_energy

    def _invalidate_by_volume(self, active_indices: NDArray[Index]) -> NDArray[np.bool_]:
        return self.bank.navigation_state.current_volume[active_indices] == -1

    def _apply_invalidators(self, active_indices: NDArray[Index]) -> None:
        dead_mask = np.zeros(len(active_indices), dtype=np.bool_)
        for invalidator in self.invalidators:
            dead_mask |= invalidator(active_indices)

        if np.any(dead_mask):
            dead_indices = active_indices[dead_mask]
            self.bank.state.is_active[dead_indices] = False
            self.bank.state.energy[dead_indices] = 0.0

    def next_step(self):
        active_indices = self.bank.active_indices

        if active_indices.size == 0:
            return

        # Pre-flight Check: Ensure buffer has enough space for a worst-case scenario
        if self.interaction_buffer.cursor[0] + len(active_indices) > self.interaction_buffer.capacity:
            self.flush_buffer()

        # Step physics and kinematics
        self.propagator.step(
            self.bank,
            self.interaction_buffer,
            self.geometry_buffer,
            self.physics_buffer,
            self.rng_ctx
        )

        # Invalidation
        self._apply_invalidators(active_indices)

        # Continuous Replenishment
        if self.source.timer <= self.stop_time:
            # We recalculate active_indices explicitly to see how many slots are free
            num_active = np.count_nonzero(self.bank.state.is_active)
            num_to_inject = self.bank.capacity - num_active

            if num_to_inject > 0:
                self.source.inject(self.bank, num_to_inject)

        self.step += 1

    def run(self):
        if self.profile:
            self.run_profile()
        else:
            self._run()

    def run_profile(self):
        runctx('self._run()', globals(), locals(), f'stats/{self.name}.txt')

    def _run(self):
        _logger.warning(f'{self.name} started from {datetime_from_seconds(self.source.timer/units.second)} to {datetime_from_seconds(self.stop_time/units.second)}')
        start_timepoint = datetime.now()

        # Initial injection
        self.source.inject(self.bank, self.particles_number)

        while np.count_nonzero(self.bank.state.is_active) > 0 or self.source.timer <= self.stop_time:
            self.next_step()
            _logger.debug(f'Source timer of {self.name} at {datetime_from_seconds(self.source.timer/units.second)}')

        # Final flush
        self.flush_buffer()
        self.queue.put('stop')

        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {datetime_from_seconds(self.source.timer/units.second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')
