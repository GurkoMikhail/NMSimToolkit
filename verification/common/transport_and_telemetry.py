"""Общие утилиты сбора потоковой телеметрии, связанного транспорта со вторичными частицами
и сквозного аудита закона сохранения энергии для верификационных экспериментов.

Все физические величины выражены строго в базовых единицах hepunits:
- длина: мм (units.mm == 1.0)
- энергия: МэВ (units.MeV == 1.0)
- время: нс (units.ns == 1.0)
- плоский угол: рад (units.rad == 1.0)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from typing import AbstractSet, Any

import numpy as np
from numpy.typing import NDArray

from core.data.data_handlers import BaseDataHandler
from core.other.typing_definitions import Float, ID, Index, ProcessType
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import (
    ChunkType,
    RNGContext,
    SecondaryActivationBuffer,
    SimulationDataBuffer,
)
from core.physics.physics_buffer import PhysicsBuffer
from core.source.secondary_source import SecondarySource
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager

INITIAL_STATE_KEYS: tuple[str, ...] = (
    "particle_ID",
    "parent_ID",
    "emission_time",
    "emission_energy",
    "pos_x",
    "pos_y",
    "pos_z",
    "dir_x",
    "dir_y",
    "dir_z",
)

INTERACTION_KEYS: tuple[str, ...] = (
    "process_id",
    "volume_id",
    "material_id",
    "particle_ID",
    "energy_deposit",
    "scattering_theta",
    "scattering_phi",
    "distance_traveled",
    "species",
    "Z",
    "pos_x",
    "pos_y",
    "pos_z",
    "dir_x",
    "dir_y",
    "dir_z",
)

ESCAPED_PARTICLE_KEYS: tuple[str, ...] = (
    "particle_ID",
    "pos_x",
    "pos_y",
    "pos_z",
)

ALL_TELEMETRY_CHUNK_TYPES: frozenset[ChunkType] = frozenset(
    {
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
        ChunkType.ESCAPED_PARTICLES,
        ChunkType.DEAD_PARTICLES,
    }
)


@dataclass(frozen=True)
class EnergyBalanceAudit:
    """Результат сквозного аудита закона сохранения энергии и баланса вторичных активаций.

    Проверяет выполнение фундаментального баланса энергии:
        E_initial == E_deposited + E_escaped + E_cutoff
    и отсутствие потерь вторичных частиц между SecondaryActivationBuffer и ParticleBank.

    Attributes:
        total_initial_energy: Суммарная начальная энергия всех первичных частиц [МэВ].
        total_deposited_energy: Суммарная локально выделенная энергия во всех взаимодействиях [МэВ].
        total_escaped_energy: Суммарная кинетическая энергия частиц, покинувших расчетную область [МэВ].
        total_cutoff_energy: Суммарная остаточная энергия частиц, остановленных по порогу min_energy [МэВ].
        total_secondary_emitted_count: Общее число вторичных частиц, инжектированных в ParticleBank.
        total_activations_recorded: Общее число записей активации, созданных в SecondaryActivationBuffer.
        absolute_energy_balance_error: Абсолютная невязка полного баланса энергии [МэВ].
        relative_energy_balance_error: Относительная невязка полного баланса энергии.
        passed: Признак выполнения закона сохранения энергии и баланса вторичных активаций.
    """

    total_initial_energy: float
    total_deposited_energy: float
    total_escaped_energy: float
    total_cutoff_energy: float
    total_secondary_emitted_count: int
    total_activations_recorded: int
    absolute_energy_balance_error: float
    relative_energy_balance_error: float
    passed: bool


class RecordingTelemetryHandler(BaseDataHandler):
    """Переиспользуемый обработчик данных для перехвата и агрегации всех каналов телеметрии в память."""

    def __init__(
        self,
        consumed_chunks: AbstractSet[ChunkType] = ALL_TELEMETRY_CHUNK_TYPES,
    ) -> None:
        super().__init__(floating_point_precision="float64")
        self._consumed_chunks: frozenset[ChunkType] = frozenset(consumed_chunks)
        self._initial_state_chunks: list[dict[str, NDArray[Any]]] = []
        self._interaction_chunks: list[dict[str, NDArray[Any]]] = []
        self._escaped_particle_chunks: list[dict[str, NDArray[Any]]] = []
        self._dead_particle_chunks: list[NDArray[Any]] = []

    @property
    def consumed_chunk_types(self) -> frozenset[ChunkType]:
        """Множество типов чанков телеметрии, принимаемых обработчиком."""
        return self._consumed_chunks

    def process_chunk(self, chunk: dict[str, Any]) -> None:
        """Принимает очередной чанк телеметрии и сохраняет его в соответствующий список."""
        if "type" not in chunk or "data" not in chunk:
            raise ValueError(
                f"Некорректный формат чанка телеметрии: ожидались ключи 'type' и 'data', получено {list(chunk.keys())}."
            )
        chunk_type = chunk["type"]
        chunk_data = chunk["data"]
        if chunk_type is ChunkType.INITIAL_STATES:
            self._initial_state_chunks.append(chunk_data)
        elif chunk_type is ChunkType.INTERACTIONS:
            self._interaction_chunks.append(chunk_data)
        elif chunk_type is ChunkType.ESCAPED_PARTICLES:
            self._escaped_particle_chunks.append(chunk_data)
        elif chunk_type is ChunkType.DEAD_PARTICLES:
            self._dead_particle_chunks.append(chunk_data)
        else:
            raise ValueError(f"Неизвестный тип чанка телеметрии: {chunk_type!r}.")

    def drain_queue(self, queue: Queue[Any]) -> None:
        """Извлекает все накопленные пакеты телеметрии из очереди SimulationManager."""
        while not queue.empty():
            queue_item = queue.get_nowait()
            if queue_item is None or queue_item == "stop":
                continue
            self.process_chunk(queue_item)

    @classmethod
    def from_manager(
        cls,
        manager: SimulationManager,
        consumed_chunks: AbstractSet[ChunkType] = ALL_TELEMETRY_CHUNK_TYPES,
    ) -> RecordingTelemetryHandler:
        """Создает обработчик и немедленно выгружает в него всю очередь завершившегося SimulationManager."""
        handler = cls(consumed_chunks=consumed_chunks)
        handler.drain_queue(manager.queue)
        return handler

    @staticmethod
    def _concat_dict_chunks(
        chunks: list[dict[str, NDArray[Any]]],
        keys: tuple[str, ...],
    ) -> dict[str, NDArray[Any]]:
        if not chunks:
            return {
                field_key: (
                    np.empty(0, dtype=ID)
                    if field_key in ("particle_ID", "parent_ID")
                    else np.empty(0, dtype=np.float64)
                )
                for field_key in keys
            }
        return {
            field_key: np.concatenate([chunk_item[field_key] for chunk_item in chunks])
            for field_key in keys
            if field_key in chunks[0]
        }

    def get_initial_states(self) -> dict[str, NDArray[Any]]:
        """Возвращает объединенный словарь массивов канала initial_states."""
        return self._concat_dict_chunks(self._initial_state_chunks, INITIAL_STATE_KEYS)

    def get_interactions(self) -> dict[str, NDArray[Any]]:
        """Возвращает объединенный словарь массивов канала interactions."""
        return self._concat_dict_chunks(self._interaction_chunks, INTERACTION_KEYS)

    def get_escaped_particles(self) -> dict[str, NDArray[Any]]:
        """Возвращает объединенный словарь массивов канала escaped_particles."""
        return self._concat_dict_chunks(self._escaped_particle_chunks, ESCAPED_PARTICLE_KEYS)

    def get_dead_particles(self) -> NDArray[np.int64]:
        """Возвращает объединенный массив идентификаторов канала dead_particles."""
        if not self._dead_particle_chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(self._dead_particle_chunks).astype(np.int64, copy=False)

    def unpack_all(
        self,
    ) -> tuple[
        dict[str, NDArray[Any]],
        dict[str, NDArray[Any]],
        NDArray[np.int64],
        dict[str, NDArray[Any]],
    ]:
        """Возвращает кортеж (initial_states, interactions, dead_particles, escaped_particles)."""
        return (
            self.get_initial_states(),
            self.get_interactions(),
            self.get_dead_particles(),
            self.get_escaped_particles(),
        )

    def clear(self) -> None:
        """Очищает все накопленные чанки."""
        self._initial_state_chunks.clear()
        self._interaction_chunks.clear()
        self._escaped_particle_chunks.clear()
        self._dead_particle_chunks.clear()


def audit_manager_telemetry_energy_balance(
    initial_states: dict[str, NDArray[Any]],
    interactions: dict[str, NDArray[Any]],
    escaped_particles: dict[str, NDArray[Any]],
    total_activations_recorded: int,
    total_secondary_emitted_count: int,
    min_energy: float = 1.0e-3,
    stopped_by_time: bool = False,
    cutoff_energy_total: float = 0.0,
    relative_tolerance: float = 1e-12,
) -> tuple[EnergyBalanceAudit, float]:
    """Выполняет пособытийный и интегральный аудит сохранения энергии по телеметрии SimulationManager.

    Для каждой частицы проверяет баланс:
        E_emit(p) == E_dep(p) + sum_{c: parent(c)==p} E_emit(c) + E_exit(p) + E_cutoff(p)
    причем:
    - если частица испытала фотоэффект (ProcessType.PHOTOELECTRIC), ее конечная энергия
      строго равна 0.0, и допустимый cutoff для нее равен 0.0 (любой ненулевой остаток — дисбаланс);
    - если частица не вылетела и не испытала фотоэффект (при stopped_by_time=False),
      допустимый cutoff возможен только в пределах 0 <= residual <= min_energy + tolerance.

    Returns:
        Кортеж (EnergyBalanceAudit, max_history_energy_imbalance).
    """
    init_ids = initial_states["particle_ID"].astype(np.int64)
    parent_ids = initial_states["parent_ID"].astype(np.int64)
    emit_energies = initial_states["emission_energy"].astype(np.float64)

    if init_ids.size == 0:
        empty_audit = EnergyBalanceAudit(
            total_initial_energy=0.0,
            total_deposited_energy=0.0,
            total_escaped_energy=0.0,
            total_cutoff_energy=0.0,
            total_secondary_emitted_count=total_secondary_emitted_count,
            total_activations_recorded=total_activations_recorded,
            absolute_energy_balance_error=0.0,
            relative_energy_balance_error=0.0,
            passed=(total_activations_recorded == total_secondary_emitted_count),
        )
        return empty_audit, 0.0

    max_pid = int(np.max(init_ids))
    parent_lookup = np.full(max_pid + 1, -1, dtype=np.int64)
    parent_lookup[init_ids] = parent_ids

    emit_energy_by_pid = np.zeros(max_pid + 1, dtype=np.float64)
    emit_energy_by_pid[init_ids] = emit_energies

    root_lookup = np.arange(max_pid + 1, dtype=np.int64)
    for pid_val in init_ids:
        curr_id = int(pid_val)
        while curr_id >= 0 and parent_lookup[curr_id] >= 0:
            curr_id = int(parent_lookup[curr_id])
        root_lookup[pid_val] = curr_id

    inter_pids = interactions["particle_ID"].astype(np.int64)
    inter_procs = interactions["process_id"].astype(np.int64)
    inter_edep = interactions["energy_deposit"].astype(np.float64)
    edep_by_pid = np.bincount(inter_pids, weights=inter_edep, minlength=max_pid + 1)

    had_photoelectric = np.zeros(max_pid + 1, dtype=np.bool_)
    pe_pids = inter_pids[inter_procs == int(ProcessType.PHOTOELECTRIC)]
    if pe_pids.size > 0:
        had_photoelectric[pe_pids] = True

    sec_mask = parent_ids >= 0
    sec_parents = parent_ids[sec_mask]
    sec_emit_energies = emit_energies[sec_mask]
    children_energy_by_pid = np.bincount(
        sec_parents, weights=sec_emit_energies, minlength=max_pid + 1
    )

    escaped_ids = escaped_particles["particle_ID"].astype(np.int64)
    is_escaped = np.zeros(max_pid + 1, dtype=np.bool_)
    if escaped_ids.size > 0:
        is_escaped[escaped_ids] = True

    abs_tol = relative_tolerance * max(float(np.max(emit_energies)), 1.0)
    residual_by_pid = emit_energy_by_pid - edep_by_pid - children_energy_by_pid

    # Вылетевшая частица не может одновременно иметь фотоэффект; если у вылетевшей частицы
    # residual < -abs_tol, это нарушение сохранения энергии
    escaped_energy_by_pid = np.where(
        is_escaped & (~had_photoelectric) & (residual_by_pid >= -abs_tol),
        np.maximum(residual_by_pid, 0.0),
        0.0,
    )

    # Невылетевшая частица:
    # - после фотоэффекта обязана иметь нулевой остаток (никакого cutoff);
    # - без фотоэффекта при stopped_by_time=False допустим только подпороговый остаток <= min_energy + abs_tol;
    # - без фотоэффекта при stopped_by_time=True допустим остаток живой частицы в пределах [0, emit_energy].
    max_allowed_cutoff = (
        emit_energy_by_pid + abs_tol
        if stopped_by_time
        else np.full(max_pid + 1, float(min_energy) + abs_tol, dtype=np.float64)
    )
    can_have_cutoff = (
        (~is_escaped)
        & (~had_photoelectric)
        & (residual_by_pid >= -abs_tol)
        & (residual_by_pid <= max_allowed_cutoff)
    )
    valid_cutoff_by_pid = np.where(
        can_have_cutoff,
        np.maximum(residual_by_pid, 0.0),
        0.0,
    )

    # Почастичный дисбаланс (необъясненный остаток энергии)
    particle_imbalance = np.abs(
        emit_energy_by_pid[init_ids]
        - (
            edep_by_pid[init_ids]
            + children_energy_by_pid[init_ids]
            + escaped_energy_by_pid[init_ids]
            + valid_cutoff_by_pid[init_ids]
        )
    )
    max_particle_imbalance = float(np.max(particle_imbalance))

    primary_ids = init_ids[parent_ids == -1]
    history_edep = np.bincount(
        root_lookup[init_ids], weights=edep_by_pid[init_ids], minlength=max_pid + 1
    )
    history_escaped = np.bincount(
        root_lookup[init_ids], weights=escaped_energy_by_pid[init_ids], minlength=max_pid + 1
    )
    history_cutoff = np.bincount(
        root_lookup[init_ids],
        weights=valid_cutoff_by_pid[init_ids],
        minlength=max_pid + 1,
    )

    primary_initial = emit_energy_by_pid[primary_ids]
    primary_accounted = (
        history_edep[primary_ids] + history_escaped[primary_ids] + history_cutoff[primary_ids]
    )
    max_history_imbalance = max(
        float(np.max(np.abs(primary_initial - primary_accounted))),
        max_particle_imbalance,
    )

    total_initial = float(np.sum(primary_initial))
    total_deposited = float(np.sum(inter_edep))
    total_escaped = float(np.sum(escaped_energy_by_pid[init_ids]))
    total_cutoff = (
        float(cutoff_energy_total)
        if cutoff_energy_total > 0.0
        else float(np.sum(valid_cutoff_by_pid[init_ids]))
    )

    abs_err = abs(total_initial - (total_deposited + total_escaped + total_cutoff))
    rel_err = abs_err / max(total_initial, 1e-15)
    activations_match = (
        total_activations_recorded == total_secondary_emitted_count == int(np.sum(sec_mask))
    )
    passed = (
        (rel_err < relative_tolerance)
        and (max_history_imbalance < abs_tol)
        and activations_match
    )

    audit = EnergyBalanceAudit(
        total_initial_energy=total_initial,
        total_deposited_energy=total_deposited,
        total_escaped_energy=total_escaped,
        total_cutoff_energy=total_cutoff,
        total_secondary_emitted_count=total_secondary_emitted_count,
        total_activations_recorded=total_activations_recorded,
        absolute_energy_balance_error=abs_err,
        relative_energy_balance_error=rel_err,
        passed=passed,
    )
    return audit, max_history_imbalance


def run_coupled_transport_loop(
    num_histories: int,
    batch_size: int,
    bank: ParticleBank,
    data_buffer: SimulationDataBuffer,
    geometry_buffer: NDArray[Any],
    physics_buffer: PhysicsBuffer,
    propagator: ParticlePropagator,
    rng_ctx: RNGContext,
    min_energy: Float,
    initialize_primary_batch: Callable[[ParticleBank, int, int], float],
    enable_atomic_relaxation: bool = True,
    photon_production_cut: Float = Float(1.0e-3),
    electron_production_cut: Float = Float(np.inf),
    on_interactions_step: Callable[[SimulationDataBuffer, ParticleBank], None] | None = None,
    evaluate_custom_escape_mask: Callable[[ParticleBank, NDArray[Index]], NDArray[np.bool_]]
    | None = None,
    on_particles_stopping: Callable[
        [ParticleBank, NDArray[Index], NDArray[np.bool_], NDArray[np.bool_]], None
    ]
    | None = None,
    relative_tolerance: float = 1e-12,
) -> EnergyBalanceAudit:
    """Выполняет пакетный цикл полного связанного транспорта с инжекцией вторичных частиц и аудитом энергии.

    На каждом шаге:
    1. Разряжает накопленные точки активации из SecondaryActivationBuffer в свободные слоты
       ParticleBank через SecondarySource.replenish_bank.
    2. Выполняет шаг переноса и физических взаимодействий ParticlePropagator.step с передачей
       SecondaryActivationBuffer.
    3. Накапливает локальное энерговыделение E_deposited, энергию вылетевших частиц E_escaped
       и остаточную энергию отсеченных частиц E_cutoff (0 < E < min_energy).
    4. Проверяет строгий закон сохранения энергии:
       E_initial == E_deposited + E_escaped + E_cutoff
       и полноту разрядки всех созданных вторичных активаций.
    """
    activation_buffer = SecondaryActivationBuffer.allocate(
        capacity=max(2 * batch_size, 1024),
        enable_atomic_relaxation=enable_atomic_relaxation,
        photon_production_cut=photon_production_cut,
        electron_production_cut=electron_production_cut,
    )
    secondary_source = SecondarySource(
        activation_buffer=activation_buffer,
        relaxation_bank=physics_buffer.relaxation_bank,
        stop_time=Float(np.inf),
    )

    total_initial_energy = 0.0
    total_deposited_energy = 0.0
    total_escaped_energy = 0.0
    total_cutoff_energy = 0.0
    total_secondary_emitted_count = 0
    total_activations_recorded = 0

    processed_histories = 0
    while processed_histories < num_histories:
        cur_batch = min(batch_size, num_histories - processed_histories)
        bank.state.is_active[:] = False
        bank.count_array[0] = Index(cur_batch)

        batch_init_energy = float(initialize_primary_batch(bank, cur_batch, processed_histories))
        total_initial_energy += batch_init_energy

        while True:
            if secondary_source.has_pending:
                injected_sec, _ = secondary_source.replenish_bank(
                    bank=bank,
                    rng_ctx=rng_ctx,
                    start_slot_cursor=Index(0),
                )
                total_secondary_emitted_count += int(injected_sec)

            active_indices = bank.active_indices
            if active_indices.size == 0:
                if not secondary_source.has_pending:
                    break
                continue

            data_buffer.interactions.reset_cursor()
            data_buffer.initial_states.reset_cursor()
            act_cursor_before = activation_buffer.cursor_value

            propagator.step(
                bank=bank,
                data_buffer=data_buffer,
                geometry_buffer=geometry_buffer,
                physics_buffer=physics_buffer,
                rng_ctx=rng_ctx,
                active_indices=active_indices,
                activation_buffer=activation_buffer,
            )

            act_cursor_after = activation_buffer.cursor_value
            if act_cursor_after > activation_buffer.capacity:
                raise RuntimeError(
                    f"Переполнение SecondaryActivationBuffer: {act_cursor_after} > {activation_buffer.capacity}."
                )
            total_activations_recorded += act_cursor_after - act_cursor_before

            num_events = data_buffer.interactions.cursor_value
            if num_events > 0:
                total_deposited_energy += float(
                    np.sum(data_buffer.interactions.energy_deposit[:num_events])
                )
                if on_interactions_step is not None:
                    on_interactions_step(data_buffer, bank)

            act_energy = bank.state.energy[active_indices]
            act_vol = bank.navigation_state.current_volume[active_indices]
            act_bdist = bank.navigation_state.boundary_distance[active_indices]

            below_cutoff = act_energy < float(min_energy)
            left_world = (act_vol < 0) & (act_bdist > 0.0)
            if evaluate_custom_escape_mask is not None:
                custom_escape = evaluate_custom_escape_mask(bank, active_indices)
                escaped_mask = (~below_cutoff) & (left_world | custom_escape)
            else:
                escaped_mask = (~below_cutoff) & left_world

            if on_particles_stopping is not None:
                on_particles_stopping(bank, active_indices, escaped_mask, below_cutoff)

            if np.any(escaped_mask):
                total_escaped_energy += float(np.sum(act_energy[escaped_mask]))

            positive_cutoff_mask = below_cutoff & (act_energy > 0.0)
            if np.any(positive_cutoff_mask):
                total_cutoff_energy += float(np.sum(act_energy[positive_cutoff_mask]))

            stop_mask = below_cutoff | escaped_mask
            if np.any(stop_mask):
                bank.state.is_active[active_indices[stop_mask]] = False

        processed_histories += cur_batch

    accounted_energy = total_deposited_energy + total_escaped_energy + total_cutoff_energy
    abs_err = abs(total_initial_energy - accounted_energy)
    rel_err = abs_err / max(total_initial_energy, 1e-15)
    passed = (
        (rel_err < relative_tolerance)
        and (total_activations_recorded == total_secondary_emitted_count)
        and (activation_buffer.cursor_value == 0)
    )

    return EnergyBalanceAudit(
        total_initial_energy=total_initial_energy,
        total_deposited_energy=total_deposited_energy,
        total_escaped_energy=total_escaped_energy,
        total_cutoff_energy=total_cutoff_energy,
        total_secondary_emitted_count=total_secondary_emitted_count,
        total_activations_recorded=total_activations_recorded,
        absolute_energy_balance_error=abs_err,
        relative_energy_balance_error=rel_err,
        passed=passed,
    )


__all__ = [
    "ALL_TELEMETRY_CHUNK_TYPES",
    "ESCAPED_PARTICLE_KEYS",
    "INITIAL_STATE_KEYS",
    "INTERACTION_KEYS",
    "EnergyBalanceAudit",
    "RecordingTelemetryHandler",
    "audit_manager_telemetry_energy_balance",
    "run_coupled_transport_loop",
]
