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
from core.physics.interaction_soa import SimulationDataBuffer, RNGContext
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
    data_buffer: SimulationDataBuffer
    geometry_buffer: NDArray
    physics_buffer: PhysicsBuffer
    rng_ctx: RNGContext
    invalidators: List[Callable[[NDArray[Index]], NDArray[np.bool_]]]
    dead_particles_buffer: List[int]

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
        self.data_buffer = SimulationDataBuffer.allocate(buffer_capacity, buffer_capacity)
        self.rng_ctx = RNGContext.from_numpy_rng(self.propagator.rng)
        self.invalidators = [self._invalidate_by_energy, self._invalidate_by_volume]
        self.dead_particles_buffer = []

        signal(SIGINT, self.sigint_handler)

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {datetime_from_seconds(self.source.timer/units.second)}')
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
        interaction_count = self.data_buffer.interactions.cursor[0]
        if interaction_count == 0:
            return

        _logger.debug(f'{self.name} flushing {interaction_count} interactions')

        chunk = {
            'type': 'interactions',
            'data': {
                'process_id': self.data_buffer.interactions.process_id[:interaction_count].copy(),
                'volume_id': self.data_buffer.interactions.volume_id[:interaction_count].copy(),
                'material_id': self.data_buffer.interactions.material_id[:interaction_count].copy(),
                'particle_ID': self.data_buffer.interactions.particle_ID[:interaction_count].copy(),
                'energy_deposit': self.data_buffer.interactions.energy_deposit[:interaction_count].copy(),
                'scattering_theta': self.data_buffer.interactions.scattering_theta[:interaction_count].copy(),
                'scattering_phi': self.data_buffer.interactions.scattering_phi[:interaction_count].copy(),
                'distance_traveled': self.data_buffer.interactions.distance_traveled[:interaction_count].copy(),
                'species': self.data_buffer.interactions.species[:interaction_count].copy(),
                'pos_x': self.data_buffer.interactions.position.x[:interaction_count].copy(),
                'pos_y': self.data_buffer.interactions.position.y[:interaction_count].copy(),
                'pos_z': self.data_buffer.interactions.position.z[:interaction_count].copy(),
                'dir_x': self.data_buffer.interactions.direction.x[:interaction_count].copy(),
                'dir_y': self.data_buffer.interactions.direction.y[:interaction_count].copy(),
                'dir_z': self.data_buffer.interactions.direction.z[:interaction_count].copy(),
            }
        }

        self.send_data(chunk)
        self.data_buffer.interactions.cursor[0] = 0


    def flush_dead_particles(self) -> None:
        """
        Flushes accumulated dead particle IDs to the queue.
        """
        if not self.dead_particles_buffer:
            return

        _logger.debug(f'{self.name} flushing {len(self.dead_particles_buffer)} dead particles')
        chunk = {
            'type': 'dead_particles',
            'data': np.array(self.dead_particles_buffer, dtype=np.int64)
        }
        self.send_data(chunk)
        self.dead_particles_buffer.clear()

    def flush_initial_states(self) -> None:
        """
        Flushes only the initial states buffer to the queue if it's full.
        """
        initial_count = self.data_buffer.initial_states.cursor[0]
        if initial_count == 0:
            return

        _logger.debug(f'{self.name} flushing {initial_count} initial states')

        chunk = {
            'type': 'initial_states',
            'data': {
                'particle_ID': self.data_buffer.initial_states.particle_ID[:initial_count].copy(),
                'emission_time': self.data_buffer.initial_states.emission_time[:initial_count].copy(),
                'emission_energy': self.data_buffer.initial_states.emission_energy[:initial_count].copy(),
                'pos_x': self.data_buffer.initial_states.emission_position.x[:initial_count].copy(),
                'pos_y': self.data_buffer.initial_states.emission_position.y[:initial_count].copy(),
                'pos_z': self.data_buffer.initial_states.emission_position.z[:initial_count].copy(),
                'dir_x': self.data_buffer.initial_states.emission_direction.x[:initial_count].copy(),
                'dir_y': self.data_buffer.initial_states.emission_direction.y[:initial_count].copy(),
                'dir_z': self.data_buffer.initial_states.emission_direction.z[:initial_count].copy(),
            }
        }

        self.send_data(chunk)
        self.data_buffer.initial_states.cursor[0] = 0


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

            # Extract Global IDs of dead particles
            dead_ids = self.bank.initial_state.ID[dead_indices]
            self.dead_particles_buffer.extend(dead_ids.tolist())

            if len(self.dead_particles_buffer) > 10000:
                # Flush states to prevent DataManager KeyError out-of-order execution
                self.flush_interactions()
                self.flush_initial_states()
                self.flush_dead_particles()

    def next_step(self):
        active_indices = self.bank.active_indices

        if active_indices.size == 0:
            return

        # Pre-flight Check: Ensure buffer has enough space for a worst-case scenario
        if self.data_buffer.interactions.cursor[0] + len(active_indices) > self.data_buffer.interactions.capacity:
            self.flush_interactions()

        if self.data_buffer.initial_states.cursor[0] + len(active_indices) > self.data_buffer.initial_states.capacity:
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
        self.flush_interactions()
        self.flush_initial_states()
        self.flush_dead_particles()
        self.queue.put('stop')

        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {datetime_from_seconds(self.source.timer/units.second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')
