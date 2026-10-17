"""
Модуль пошагового пропагатора частиц (ParticlePropagator), оркеструющего геометрический
рейкастинг, дельта-трекинг ВудКока и диспетчеризацию физических процессов.
"""

from typing import Any, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_kernels import cast_path_kernel
from core.other.typing_definitions import Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext, SimulationDataBuffer
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.processes import Process
from core.transport.transport_buffer import TransportBuffer
from core.transport.transport_kernels import (
    _push_first_interacting_to_initial_state_kernel,
    make_transport_kernel,
)
import settings.database_setting as database_setting
import settings.processes_settings as processes_settings


class ParticlePropagator:
    """Оркестратор пошагового переноса частиц и применения физических процессов в стиле DOD/SoA."""

    processes: List[Process]
    process_ids: NDArray[Index]
    rng: np.random.Generator
    transport_buffer: TransportBuffer
    _transport_kernel: Any

    def __init__(
        self,
        processes_list: Optional[Sequence[type[Process]]] = None,
        attenuation_database: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        resolved_processes: Sequence[type[Process]] = (
            processes_settings.processes_list if processes_list is None else processes_list
        )
        resolved_database = (
            database_setting.attenuation_database if attenuation_database is None else attenuation_database
        )
        self.rng = np.random.default_rng() if rng is None else rng

        self.processes = [process_cls(resolved_database, self.rng) for process_cls in resolved_processes]
        self.process_ids = np.array(
            [proc.process_id if proc.process_id is not None else -1 for proc in self.processes],
            dtype=Index,
        )

        self._transport_kernel = make_transport_kernel(self.process_ids)
        self.transport_buffer = TransportBuffer.allocate(0, len(self.processes))

    def step(
        self,
        bank: ParticleBank,
        data_buffer: SimulationDataBuffer,
        geometry_buffer: NDArray[Any],
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        """
        Выполняет один полный шаг переноса и взаимодействий для всех активных частиц в банке.
        """
        active_indices = bank.active_indices
        if active_indices.size == 0:
            return

        if self.transport_buffer.process_ids.size < bank.capacity:
            self.transport_buffer = TransportBuffer.allocate(bank.capacity, len(self.processes))

        # 1. Геометрический рейкастинг для частиц с невалидным состоянием навигации
        cast_path_kernel(
            bank.state.position,
            bank.state.direction,
            active_indices,
            geometry_buffer,
            bank.navigation_state,
        )

        # 2. Дельта-трекинг ВудКока
        self._transport_kernel(
            bank.state,
            bank.navigation_state,
            active_indices,
            physics_buffer,
            self.transport_buffer,
            geometry_buffer,
            rng_ctx,
        )

        # 3. Запись начального состояния для частиц, провзаимодействовавших впервые
        _push_first_interacting_to_initial_state_kernel(
            bank.initial_state,
            bank.state,
            active_indices,
            self.transport_buffer.process_ids,
            data_buffer.initial_states,
        )

        # 4. Уплотнение потока (Stream Compaction) и диспетчеризация по ядрам процессов
        for process in self.processes:
            mask = self.transport_buffer.process_ids[active_indices] == process.process_id
            target_indices = active_indices[mask]

            if target_indices.size > 0:
                process.apply(
                    bank,
                    target_indices,
                    data_buffer.interactions,
                    physics_buffer,
                    self.transport_buffer.material_ids,
                    rng_ctx,
                )


__all__ = ["ParticlePropagator"]

