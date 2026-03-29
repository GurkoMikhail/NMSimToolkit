from core.geometry.geometry_kernels import cast_path_kernel
import numpy as np
from typing import List, Optional, Any
from numpy.typing import NDArray

import settings.processes_settings as processes_settings
import settings.database_setting as database_setting
from core.physics.processes import Process
from core.particles.particles_soa import ParticleBank
from core.physics.interaction_soa import InteractionBuffer, RNGContext, InitialStateBuffer, SimulationDataBuffer
from core.physics.physics_buffer import PhysicsBuffer
from core.other.typing_definitions import Index
from core.transport.transport_kernels import make_transport_kernel, _push_to_initial_state_kernel
from core.transport.transport_buffer import TransportBuffer

class ParticlePropagator:
    """ Python class for managing particle propagation per-step (DOD style). """
    processes: List[Process]
    process_ids: NDArray[Index]
    rng: np.random.Generator

    _flags_buffer: NDArray[Index]

    def __init__(self, processes_list: Optional[List[type]] = None, attenuation_database: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> None:
        processes_list = processes_settings.processes_list if processes_list is None else processes_list
        attenuation_database = database_setting.attenuation_database if attenuation_database is None else attenuation_database
        self.rng = np.random.default_rng() if rng is None else rng

        self.processes = [process(attenuation_database, rng) for process in processes_list]
        self.process_ids = np.array([p.process_id for p in self.processes], dtype=Index)


        self._transport_kernel = make_transport_kernel(self.process_ids)
        self.transport_buffer = TransportBuffer.allocate(0)

    def step(self, bank: ParticleBank, data_buffer: SimulationDataBuffer, geometry_buffer: NDArray, physics_buffer: PhysicsBuffer, rng_ctx: RNGContext) -> None:
        """
        Executes a single physics step for all active particles.
        """
        active_indices = bank.active_indices
        if active_indices.size == 0:
            return

        # Ensure flags buffer capacity
        if self.transport_buffer.process_ids.size < bank.capacity:
            self.transport_buffer = TransportBuffer.allocate(bank.capacity)

        # 1. Raycast for invalidated particles
        cast_path_kernel(
            bank.state.position,
            bank.state.direction,
            active_indices,
            geometry_buffer,
            bank.navigation_state
        )

        # 2. Delta Tracking
        self._transport_kernel(
            bank.state,
            bank.navigation_state,
            active_indices,
            physics_buffer,
            self.transport_buffer,
            geometry_buffer,
            rng_ctx
        )

        # 3. Push to initial state buffer for interacting particles
        interacting_mask = self.transport_buffer.process_ids[active_indices] != -1
        # Also filter out those that have already interacted
        interacting_mask &= ~bank.initial_state.has_interacted[active_indices]

        interacting_indices = active_indices[interacting_mask]

        if interacting_indices.size > 0:
            _push_to_initial_state_kernel(bank.initial_state, interacting_indices, data_buffer.initial_states)

        # 4. Stream Compaction & Dispatch to Process kernels
        for process in self.processes:
            # Find which active particles underwent this process
            mask = self.transport_buffer.process_ids[active_indices] == process.process_id
            target_indices = active_indices[mask]

            if target_indices.size > 0:
                process.apply(bank, target_indices, data_buffer.interactions, physics_buffer, self.transport_buffer.material_ids, rng_ctx)
