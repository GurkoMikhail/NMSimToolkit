"""
Модуль профилирования влияния емкости банка частиц (ParticleBank capacity / batch_size)
на производительность и структуру затрат конвейера NMSimToolkit с явным разделением двух фаз
жизненного цикла банка частиц:
1. Full-Bank Sustained Phase (стационарная фаза 100%-полного банка с регулярной докачкой из источника);
2. Tail Depletion Phase (хвостовая фаза опустошения банка после исчерпания квоты источника).

Штатная выборка полного режима составляет 6,000,000 историй (в 100 раз больше базовой),
а размер батча варьируется в диапазоне [2_500, 5_000, 15_000, 30_000, 60_000, 120_000, 250_000, 500_000].
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Sequence

import numpy as np

from core.geometry.geometry_kernels import cast_path_kernel
from core.other.typing_definitions import Float
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import (
    DEFAULT_CONSUMED_CHUNK_TYPES,
    SimulationManager,
)
from core.transport.transport_buffer import TransportBuffer
from core.transport.transport_kernels import (
    _push_first_interacting_to_initial_state_kernel,
)
from settings.database_setting import attenuation_database
from settings.processes_settings import processes_list
from verification.profiling.workloads import (
    CollimatorKind,
    ProfilingWorkload,
    build_nema_tb161_workload,
    build_point_source_workload,
    warmup_propagator_instance,
)

DEFAULT_BATCH_SIZES: Sequence[int] = (
    2_500,
    5_000,
    15_000,
    30_000,
    60_000,
    120_000,
    250_000,
    500_000,
)


@dataclass(frozen=True)
class BatchSizeBenchmarkRecord:
    """
    Метрики производительности и пофазного распределения времени для заданного batch_size
    с раздельным учетом фазы полного банка (Full-Bank Phase) и фазы опустошения (Depletion Phase).
    """

    workload_id: str
    title: str
    collimator_kind: CollimatorKind
    total_particles: int
    emitted_particles: int
    batch_size: int
    steps_count: int
    replenish_steps_count: int
    avg_active_particles_per_step: float
    avg_bank_occupancy_pct: float
    total_time_sec: float
    histories_per_sec: float
    source_time_sec: float
    geometry_raycast_time_sec: float
    woodcock_time_sec: float
    processes_time_sec: float
    compaction_and_escaped_time_sec: float

    # --- Фаза A: Стационарный режим полного банка (Full-Bank Sustained Phase) ---
    full_bank_steps_count: int = 0
    full_bank_completed_histories: int = 0
    full_bank_active_tracks: int = 0
    full_bank_time_sec: float = 0.0
    full_bank_histories_per_sec: float = 0.0
    full_bank_tracks_per_sec: float = 0.0
    full_source_time_sec: float = 0.0
    full_geometry_raycast_time_sec: float = 0.0
    full_woodcock_time_sec: float = 0.0
    full_processes_time_sec: float = 0.0
    full_compaction_and_escaped_time_sec: float = 0.0

    # --- Фаза B: Хвостовая фаза опустошения банка (Tail Depletion Phase) ---
    depletion_steps_count: int = 0
    depletion_completed_histories: int = 0
    depletion_active_tracks: int = 0
    depletion_avg_occupancy_pct: float = 0.0
    depletion_time_sec: float = 0.0
    depletion_time_share_pct: float = 0.0
    depletion_tracks_per_sec: float = 0.0
    depletion_track_slowdown_factor: float = 1.0

    @property
    def phase_sum_sec(self) -> float:
        """Суммарное время всех 5 агрегированных фаз шага по всему прогону [с]."""
        return max(
            self.source_time_sec
            + self.geometry_raycast_time_sec
            + self.woodcock_time_sec
            + self.processes_time_sec
            + self.compaction_and_escaped_time_sec,
            1e-12,
        )

    @property
    def full_phase_sum_sec(self) -> float:
        """Суммарное время всех 5 агрегированных фаз в стационарном режиме полного банка [с]."""
        return max(
            self.full_source_time_sec
            + self.full_geometry_raycast_time_sec
            + self.full_woodcock_time_sec
            + self.full_processes_time_sec
            + self.full_compaction_and_escaped_time_sec,
            1e-12,
        )

    @property
    def source_pct(self) -> float:
        """Доля фазы генерации и пополнения банка частиц Source по всему прогону [%]."""
        return 100.0 * self.source_time_sec / self.phase_sum_sec

    @property
    def geometry_raycast_pct(self) -> float:
        """Доля фазы геометрического рейкастинга (cast_path_kernel) по всему прогону [%]."""
        return 100.0 * self.geometry_raycast_time_sec / self.phase_sum_sec

    @property
    def woodcock_pct(self) -> float:
        """Доля фазы дельта-трекинга Вудкока (_transport_kernel) по всему прогону [%]."""
        return 100.0 * self.woodcock_time_sec / self.phase_sum_sec

    @property
    def processes_pct(self) -> float:
        """Доля ядер физических процессов (Photoelectric, Compton, Coherent) по всему прогону [%]."""
        return 100.0 * self.processes_time_sec / self.phase_sum_sec

    @property
    def compaction_and_escaped_pct(self) -> float:
        """Доля управления индексами, компактификации, инвалидации и сбора вылетевших частиц [%]."""
        return 100.0 * self.compaction_and_escaped_time_sec / self.phase_sum_sec

    @property
    def full_source_pct(self) -> float:
        """Доля фазы Source в стационарном режиме полного банка (Full-Bank Phase) [%]."""
        return 100.0 * self.full_source_time_sec / self.full_phase_sum_sec

    @property
    def full_geometry_raycast_pct(self) -> float:
        """Доля фазы Geometry Raycast в стационарном режиме полного банка (Full-Bank Phase) [%]."""
        return 100.0 * self.full_geometry_raycast_time_sec / self.full_phase_sum_sec

    @property
    def full_woodcock_pct(self) -> float:
        """Доля фазы Woodcock в стационарном режиме полного банка (Full-Bank Phase) [%]."""
        return 100.0 * self.full_woodcock_time_sec / self.full_phase_sum_sec

    @property
    def full_processes_pct(self) -> float:
        """Доля фазы Processes в стационарном режиме полного банка (Full-Bank Phase) [%]."""
        return 100.0 * self.full_processes_time_sec / self.full_phase_sum_sec

    @property
    def full_compaction_and_escaped_pct(self) -> float:
        """Доля фазы Compaction+Escaped в стационарном режиме полного банка (Full-Bank Phase) [%]."""
        return 100.0 * self.full_compaction_and_escaped_time_sec / self.full_phase_sum_sec


def run_single_batch_size_benchmark(
    workload: ProfilingWorkload,
    batch_size: int,
    total_particles: Optional[int] = None,
    seed: int = 42,
    propagator: Optional[ParticlePropagator] = None,
) -> BatchSizeBenchmarkRecord:
    """
    Запускает steady-state прогон рабочей нагрузки при фиксированном общем числе историй
    (total_particles) и заданной емкости банка частиц (batch_size), разделяя метрики
    на фазу полного банка (Full-Bank Sustained Phase) и фазу опустошения (Tail Depletion Phase).
    """
    target_total_histories = int(
        workload.particles_number if total_particles is None else total_particles
    )
    effective_batch_size = int(batch_size)
    if effective_batch_size <= 0:
        raise ValueError(f"Размер батча должен быть положительным, получено: {batch_size}")

    workload.source_node.rng = np.random.default_rng(seed)
    if propagator is None:
        propagator_inst = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(seed),
        )
    else:
        propagator_inst = propagator

    # Гарантируем, что все замыкания пропагатора прогреты ДО старта таймеров
    warmup_propagator_instance(
        propagator=propagator_inst,
        scene=workload.root_scene,
        rng=np.random.default_rng(seed),
    )
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))

    # Емкость буфера взаимодействий соразмерна батчу (не раздуваем RAM при 6 млн историй)
    buf_cap = max(effective_batch_size * 2, 200_000)
    manager = SimulationManager(
        scene=workload.root_scene,
        propagator=propagator_inst,
        stop_time=workload.stop_time,
        particles_number=effective_batch_size,
        min_energy=workload.min_energy,
        buffer_capacity=buf_cap,
        seed=seed,
        consumed_chunk_types=DEFAULT_CONSUMED_CHUNK_TYPES,
    )

    source_ns = 0
    geom_ns = 0
    woodcock_ns = 0
    processes_ns = 0
    compaction_ns = 0

    full_source_ns = 0
    full_geom_ns = 0
    full_woodcock_ns = 0
    full_processes_ns = 0
    full_compaction_ns = 0

    depl_source_ns = 0
    depl_geom_ns = 0
    depl_woodcock_ns = 0
    depl_processes_ns = 0
    depl_compaction_ns = 0

    replenish_steps = 0
    active_particles_accum = 0
    steps_executed = 0

    full_steps = 0
    full_completed_histories = 0
    full_active_tracks = 0

    depl_steps = 0
    depl_completed_histories = 0
    depl_active_tracks = 0

    wall_start = time.perf_counter()

    while True:
        # 1. Фаза пополнения банка частиц (Source) с контролем квоты target_total_histories
        t0 = time.perf_counter_ns()
        emitted_so_far = int(manager.bank.count_array[0])
        remaining_quota = max(0, target_total_histories - emitted_so_far)
        did_inject = False
        if remaining_quota > 0 and manager.active_sources:
            num_active = int(np.count_nonzero(manager.bank.state.is_active))
            free_slots = int(manager.bank.capacity - num_active)
            num_to_inject = min(free_slots, remaining_quota)
            if num_to_inject > 0:
                delta_time = manager._calculate_time_step(num_to_inject)
                target_time = Float(manager.global_timer + delta_time)
                manager._distribute_quotas(target_time, num_to_inject)
                manager.global_timer = target_time
                replenish_steps += 1
                did_inject = True
        step_source_ns = time.perf_counter_ns() - t0

        # 2. Извлечение активных индексов и подготовка буферов
        t0 = time.perf_counter_ns()
        active_indices = manager.bank.active_indices
        active_count = int(active_indices.size)
        if active_count == 0:
            step_comp0_ns = time.perf_counter_ns() - t0
            source_ns += step_source_ns
            compaction_ns += step_comp0_ns
            if int(manager.bank.count_array[0]) >= target_total_histories or not manager.active_sources:
                break
            continue

        # Шаг относится к Full-Bank Phase, если банк заполнен на 100% (active_count == effective_batch_size)
        # либо это первый шаг инжекции при target_total_histories < effective_batch_size
        is_full_bank_step = (active_count == effective_batch_size) or (did_inject and full_steps == 0)

        steps_executed += 1
        active_particles_accum += active_count

        manager._ensure_step_capacity(active_count)
        while not manager.queue.empty():
            manager.queue.get_nowait()

        if propagator_inst.transport_buffer.process_ids.size < manager.bank.capacity:
            propagator_inst.transport_buffer = TransportBuffer.allocate(
                manager.bank.capacity, len(propagator_inst.processes)
            )
        step_comp_ns = time.perf_counter_ns() - t0

        # 3. Геометрический рейкастинг (cast_path_kernel)
        t0 = time.perf_counter_ns()
        cast_path_kernel(
            manager.bank.state.position,
            manager.bank.state.direction,
            active_indices,
            manager.geometry_buffer,
            manager.bank.navigation_state,
        )
        step_geom_ns = time.perf_counter_ns() - t0

        # 4. Дельта-трекинг Вудкока (transport_kernel)
        t0 = time.perf_counter_ns()
        propagator_inst._transport_kernel(
            manager.bank.state,
            manager.bank.navigation_state,
            active_indices,
            manager.physics_buffer,
            propagator_inst.transport_buffer,
            manager.geometry_buffer,
            manager.rng_ctx,
        )
        step_woodcock_ns = time.perf_counter_ns() - t0

        # 5. Запись начальных состояний и Stream Compaction + Физические процессы
        t0 = time.perf_counter_ns()
        _push_first_interacting_to_initial_state_kernel(
            manager.bank.initial_state,
            manager.bank.state,
            active_indices,
            propagator_inst.transport_buffer.process_ids,
            manager.data_buffer.initial_states,
        )
        step_comp_ns += time.perf_counter_ns() - t0

        step_proc_ns = 0
        for process_obj in propagator_inst.processes:
            t_comp0 = time.perf_counter_ns()
            mask = propagator_inst.transport_buffer.process_ids[active_indices] == process_obj.process_id
            target_indices = active_indices[mask]
            step_comp_ns += time.perf_counter_ns() - t_comp0

            if target_indices.size > 0:
                t_proc0 = time.perf_counter_ns()
                process_obj.apply(
                    manager.bank,
                    target_indices,
                    manager.data_buffer.interactions,
                    manager.physics_buffer,
                    propagator_inst.transport_buffer.material_ids,
                    manager.rng_ctx,
                )
                step_proc_ns += time.perf_counter_ns() - t_proc0

        # 6. Инвалидация и сбор вылетевших частиц
        t0 = time.perf_counter_ns()
        dead_indices = manager._apply_invalidators(active_indices)
        dead_count = int(dead_indices.size)
        if dead_count > 0:
            manager._record_finished_histories(dead_indices)
            while not manager.queue.empty():
                manager.queue.get_nowait()
        manager._check_no_overflow()
        manager.step_count += 1
        step_comp_ns += time.perf_counter_ns() - t0

        source_ns += step_source_ns
        geom_ns += step_geom_ns
        woodcock_ns += step_woodcock_ns
        processes_ns += step_proc_ns
        compaction_ns += step_comp_ns

        if is_full_bank_step:
            full_steps += 1
            full_completed_histories += dead_count
            full_active_tracks += active_count
            full_source_ns += step_source_ns
            full_geom_ns += step_geom_ns
            full_woodcock_ns += step_woodcock_ns
            full_processes_ns += step_proc_ns
            full_compaction_ns += step_comp_ns
        else:
            depl_steps += 1
            depl_completed_histories += dead_count
            depl_active_tracks += active_count
            depl_source_ns += step_source_ns
            depl_geom_ns += step_geom_ns
            depl_woodcock_ns += step_woodcock_ns
            depl_processes_ns += step_proc_ns
            depl_compaction_ns += step_comp_ns

    total_wall = max(time.perf_counter() - wall_start, 1e-9)
    emitted_final = int(manager.bank.count_array[0])
    avg_active = float(active_particles_accum) / max(1, steps_executed)
    avg_occ_pct = 100.0 * avg_active / float(effective_batch_size)

    full_time_sec = float(
        (full_source_ns + full_geom_ns + full_woodcock_ns + full_processes_ns + full_compaction_ns) * 1e-9
    )
    full_pure_track_sec = float(
        (full_geom_ns + full_woodcock_ns + full_processes_ns + full_compaction_ns) * 1e-9
    )
    depl_time_sec = float(
        (depl_source_ns + depl_geom_ns + depl_woodcock_ns + depl_processes_ns + depl_compaction_ns) * 1e-9
    )
    depl_pure_track_sec = float(
        (depl_geom_ns + depl_woodcock_ns + depl_processes_ns + depl_compaction_ns) * 1e-9
    )
    total_phase_sec = max(full_time_sec + depl_time_sec, 1e-9)

    full_hist_per_sec = float(full_completed_histories) / max(full_time_sec, 1e-9)
    full_tracks_per_sec = float(full_active_tracks) / max(full_pure_track_sec, 1e-9)
    depl_tracks_per_sec = (
        float(depl_active_tracks) / max(depl_pure_track_sec, 1e-9) if depl_steps > 0 else full_tracks_per_sec
    )
    depl_avg_occ_pct = (
        100.0 * (float(depl_active_tracks) / max(1, depl_steps)) / float(effective_batch_size)
        if depl_steps > 0
        else 0.0
    )
    slowdown_factor = full_tracks_per_sec / max(depl_tracks_per_sec, 1e-9) if depl_steps > 0 else 1.0

    return BatchSizeBenchmarkRecord(
        workload_id=workload.workload_id,
        title=workload.title,
        collimator_kind=workload.collimator_kind,
        total_particles=target_total_histories,
        emitted_particles=emitted_final,
        batch_size=effective_batch_size,
        steps_count=int(steps_executed),
        replenish_steps_count=int(replenish_steps),
        avg_active_particles_per_step=float(avg_active),
        avg_bank_occupancy_pct=float(avg_occ_pct),
        total_time_sec=float(total_wall),
        histories_per_sec=float(emitted_final / total_wall),
        source_time_sec=float(source_ns * 1e-9),
        geometry_raycast_time_sec=float(geom_ns * 1e-9),
        woodcock_time_sec=float(woodcock_ns * 1e-9),
        processes_time_sec=float(processes_ns * 1e-9),
        compaction_and_escaped_time_sec=float(compaction_ns * 1e-9),
        full_bank_steps_count=int(full_steps),
        full_bank_completed_histories=int(full_completed_histories),
        full_bank_active_tracks=int(full_active_tracks),
        full_bank_time_sec=full_time_sec,
        full_bank_histories_per_sec=full_hist_per_sec,
        full_bank_tracks_per_sec=full_tracks_per_sec,
        full_source_time_sec=float(full_source_ns * 1e-9),
        full_geometry_raycast_time_sec=float(full_geom_ns * 1e-9),
        full_woodcock_time_sec=float(full_woodcock_ns * 1e-9),
        full_processes_time_sec=float(full_processes_ns * 1e-9),
        full_compaction_and_escaped_time_sec=float(full_compaction_ns * 1e-9),
        depletion_steps_count=int(depl_steps),
        depletion_completed_histories=int(depl_completed_histories),
        depletion_active_tracks=int(depl_active_tracks),
        depletion_avg_occupancy_pct=float(depl_avg_occ_pct),
        depletion_time_sec=depl_time_sec,
        depletion_time_share_pct=float(100.0 * depl_time_sec / total_phase_sec),
        depletion_tracks_per_sec=float(depl_tracks_per_sec),
        depletion_track_slowdown_factor=float(slowdown_factor),
    )


def run_batch_size_sweep_suite(
    total_particles: int = 6_000_000,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    seed: int = 42,
) -> List[BatchSizeBenchmarkRecord]:
    """
    Выполняет серию замеров влияния размера батча частиц (batch_size) при фиксированном
    общем числе историй (по умолчанию 6,000,000 частиц) на сценах W1_PointSource и W2_NEMA_Tb161
    для обоих видов коллиматоров (Parametric и Direct).
    """
    workloads: List[ProfilingWorkload] = [
        build_point_source_workload("parametric", particles_number=total_particles, seed=seed),
        build_point_source_workload("direct", particles_number=total_particles, seed=seed),
        build_nema_tb161_workload("parametric", particles_number=total_particles, seed=seed),
        build_nema_tb161_workload("direct", particles_number=total_particles, seed=seed),
    ]

    shared_propagator = ParticlePropagator(
        processes_list=processes_list,
        attenuation_database=attenuation_database,
        rng=np.random.default_rng(seed),
    )
    warmup_propagator_instance(
        propagator=shared_propagator,
        scene=workloads[0].root_scene,
        rng=np.random.default_rng(seed),
    )

    records: List[BatchSizeBenchmarkRecord] = []
    total_runs = len(workloads) * len(batch_sizes)
    run_idx = 0
    for workload in workloads:
        for b_size in batch_sizes:
            run_idx += 1
            rec = run_single_batch_size_benchmark(
                workload=workload,
                batch_size=int(b_size),
                total_particles=total_particles,
                seed=seed,
                propagator=shared_propagator,
            )
            records.append(rec)
            print(
                f"  [{run_idx:02d}/{total_runs:02d}] {rec.workload_id} | batch={rec.batch_size:,} | "
                f"steps={rec.steps_count} (full={rec.full_bank_steps_count}, depl={rec.depletion_steps_count}) | "
                f"full_tput={rec.full_bank_histories_per_sec:,.0f}/s | "
                f"total_tput={rec.histories_per_sec:,.0f}/s | "
                f"time={rec.total_time_sec:.2f}s",
                flush=True,
            )

    return records


__all__ = [
    "BatchSizeBenchmarkRecord",
    "DEFAULT_BATCH_SIZES",
    "run_batch_size_sweep_suite",
    "run_single_batch_size_benchmark",
]
