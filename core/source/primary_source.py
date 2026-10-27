"""
Модуль рантайм-источника первичного излучения (PrimarySource), управляющего
нестационарным пуассоновским потоком эмиссии с прореживанием Льюиса–Шедлера.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext
from core.source.source_buffer import SourceBuffer
from core.source.source_kernels import inject_poisson_sources_kernel
from core.source.sources import ParticleSource


class PrimarySource(ParticleSource):
    """
    Рантайм-источник первичного излучения, управляющий нестационарным пуассоновским потоком
    эмиссии по методу прореживания Льюиса–Шедлера из скомпилированного SourceBuffer.
    """

    source_buffer: SourceBuffer
    start_time: Float
    stop_time: Float
    max_particles: int
    t_ref: Float
    t_last: Float
    pending_arrival_time: Float
    emitted_particles: int

    def __init__(
        self,
        source_buffer: SourceBuffer,
        start_time: Float = Float(0.0),
        stop_time: Float = Float(np.inf),
        max_particles: int = -1,
    ) -> None:
        self.source_buffer = source_buffer
        self.start_time = Float(start_time)
        self.stop_time = Float(stop_time)
        self.max_particles = int(max_particles)
        self.t_ref = self.start_time
        self.t_last = self.start_time
        self.pending_arrival_time = Float(np.nan)
        self.emitted_particles = 0

    @property
    def has_pending(self) -> bool:
        """True, если в SourceBuffer есть активные каналы и не достигнут предел времени или квоты частиц."""
        if self.source_buffer.num_channels == 0:
            return False
        if 0 <= self.max_particles <= self.emitted_particles:
            return False
        if self.t_last >= self.stop_time or self.pending_arrival_time >= self.stop_time:
            return False
        return True

    def replenish_bank(
        self,
        bank: ParticleBank,
        rng_ctx: RNGContext,
        start_slot_cursor: Index = Index(0),
        max_slots: Index = Index(-1),
    ) -> Tuple[Index, Index]:
        """
        Инжектирует первичные частицы из пуассоновского потока в свободные слоты ParticleBank
        с соблюдением временного окна stop_time, квоты историй max_particles и лимита слотов max_slots.
        """
        if not self.has_pending or max_slots == Index(0):
            return Index(0), start_slot_cursor

        step_limit = Index(-1)
        if self.max_particles >= 0:
            remaining_quota = self.max_particles - self.emitted_particles
            if remaining_quota <= 0:
                return Index(0), start_slot_cursor
            step_limit = Index(remaining_quota)

        if max_slots > Index(0):
            step_limit = max_slots if step_limit < Index(0) else min(step_limit, max_slots)

        injected, next_cursor, new_t_last, new_pending = inject_poisson_sources_kernel(
            bank=bank,
            source_buffer=self.source_buffer,
            stop_time=self.stop_time,
            t_ref=self.t_ref,
            t_last=self.t_last,
            pending_arrival_time=self.pending_arrival_time,
            rng_ctx=rng_ctx,
            max_particles=step_limit,
            start_slot_cursor=start_slot_cursor,
        )
        self.emitted_particles += int(injected)
        self.t_last = Float(new_t_last)
        self.pending_arrival_time = Float(new_pending)

        if math.isfinite(float(self.stop_time)) and self.pending_arrival_time >= self.stop_time:
            self.t_last = self.stop_time

        return injected, next_cursor

    def export_state(self) -> Optional[dict[str, Any]]:
        """Экспортирует состояние пуассоновского потока для чекпоинта Stateful Resume."""
        return {
            "t_ref": float(self.t_ref),
            "t_last": float(self.t_last),
            "pending_arrival_time": float(self.pending_arrival_time),
            "emitted_primaries": int(self.emitted_particles),
        }

    def restore_state(self, state: Optional[Mapping[str, Any]]) -> None:
        """Восстанавливает состояние пуассоновского потока из чекпоинта Stateful Resume."""
        if not state:
            return
        self.emitted_particles = int(state.get("emitted_primaries", 0))
        self.t_ref = Float(self.start_time if state.get("t_ref") is None else state["t_ref"])
        self.t_last = Float(self.start_time if state.get("t_last") is None else state["t_last"])
        if state.get("pending_arrival_time") is not None:
            self.pending_arrival_time = Float(state["pending_arrival_time"])


__all__ = ["PrimarySource"]
