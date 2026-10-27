"""
Модуль наведенного вторичного источника излучения (SecondarySource),
разряжающего точки активации SecondaryActivationBuffer в ParticleBank.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Charge, Energy, Float, Index, Length, Species, Time
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext, SecondaryActivationBuffer
from core.physics.physics_buffer import AtomicRelaxationBank
from core.source.source_kernels import inject_secondary_sources_kernel
from core.source.sources import ParticleSource


class SecondarySource(ParticleSource):
    """
    Наведенный вторичный источник излучения, агрегирующий точки активации среды
    в SecondaryActivationBuffer и разряжающий их в ParticleBank с пуассоновским
    розыгрышем времени испускания на основе постоянных распада AtomicRelaxationBank.
    """

    activation_buffer: SecondaryActivationBuffer
    relaxation_bank: AtomicRelaxationBank
    stop_time: Float

    def __init__(
        self,
        activation_buffer: SecondaryActivationBuffer,
        relaxation_bank: AtomicRelaxationBank,
        stop_time: Float = Float(np.inf),
    ) -> None:
        self.activation_buffer = activation_buffer
        self.relaxation_bank = relaxation_bank
        self.stop_time = Float(stop_time)

    @property
    def has_pending(self) -> bool:
        """True, если в SecondaryActivationBuffer есть невыгруженные точки активации."""
        return self.activation_buffer.cursor_value > 0

    def replenish_bank(
        self,
        bank: ParticleBank,
        rng_ctx: RNGContext,
        start_slot_cursor: Index = Index(0),
        max_slots: Index = Index(-1),
    ) -> Tuple[Index, Index]:
        """
        Разряжает накопленные точки активации SecondaryActivationBuffer в свободные слоты
        ParticleBank (не более max_slots слотов при max_slots >= 0).
        Возвращает кортеж (injected_count, next_slot_cursor).
        """
        if not self.has_pending or max_slots == Index(0):
            return Index(0), start_slot_cursor
        return inject_secondary_sources_kernel(
            bank=bank,
            act_buffer=self.activation_buffer,
            relaxation_bank=self.relaxation_bank,
            stop_time=self.stop_time,
            rng_ctx=rng_ctx,
            max_particles=max_slots,
            start_slot_cursor=start_slot_cursor,
        )

    def export_state(self) -> Optional[dict[str, NDArray[Any]]]:
        """Экспортирует необработанные записи активации для чекпоинта Stateful Resume."""
        count = self.activation_buffer.cursor_value
        if count <= 0:
            return None
        buf = self.activation_buffer
        return {
            "pos_x": buf.position.x[:count].copy(),
            "pos_y": buf.position.y[:count].copy(),
            "pos_z": buf.position.z[:count].copy(),
            "dir_x": buf.direction.x[:count].copy(),
            "dir_y": buf.direction.y[:count].copy(),
            "dir_z": buf.direction.z[:count].copy(),
            "activation_time": buf.activation_time[:count].copy(),
            "energy": buf.energy[:count].copy(),
            "parent_ID": buf.parent_ID[:count].copy(),
            "species": buf.species[:count].copy(),
            "target_z": buf.target_z[:count].copy(),
            "shell_idx": buf.shell_idx[:count].copy(),
            "process_id": buf.process_id[:count].copy(),
        }

    def restore_state(self, state: Optional[Mapping[str, Any]]) -> None:
        """Восстанавливает необработанные записи активации из чекпоинта Stateful Resume."""
        if not state:
            self.activation_buffer.reset_cursor()
            return
        energy_arr = np.asarray(state["energy"], dtype=Energy)
        count = int(energy_arr.shape[0])
        if count <= 0:
            self.activation_buffer.reset_cursor()
            return
        if count > self.activation_buffer.capacity:
            old_buf = self.activation_buffer
            self.activation_buffer = SecondaryActivationBuffer.allocate(
                capacity=max(count, old_buf.capacity * 2),
                enable_atomic_relaxation=old_buf.enable_atomic_relaxation,
                photon_production_cut=old_buf.photon_production_cut,
                electron_production_cut=old_buf.electron_production_cut,
            )
        buf = self.activation_buffer
        buf.position.x[:count] = np.asarray(state["pos_x"], dtype=Length)
        buf.position.y[:count] = np.asarray(state["pos_y"], dtype=Length)
        buf.position.z[:count] = np.asarray(state["pos_z"], dtype=Length)
        buf.direction.x[:count] = np.asarray(state["dir_x"], dtype=Float)
        buf.direction.y[:count] = np.asarray(state["dir_y"], dtype=Float)
        buf.direction.z[:count] = np.asarray(state["dir_z"], dtype=Float)
        buf.activation_time[:count] = np.asarray(state["activation_time"], dtype=Time)
        buf.energy[:count] = energy_arr
        buf.parent_ID[:count] = np.asarray(state["parent_ID"], dtype=buf.parent_ID.dtype)
        buf.species[:count] = np.asarray(state["species"], dtype=Species)
        buf.target_z[:count] = np.asarray(state["target_z"], dtype=Charge)
        buf.shell_idx[:count] = np.asarray(state["shell_idx"], dtype=Index)
        buf.process_id[:count] = np.asarray(state["process_id"], dtype=buf.process_id.dtype)
        buf.cursor[0] = Index(count)


__all__ = ["SecondarySource"]
