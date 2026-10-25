"""
Модуль синхронного 10-фазного профилировщика шага симуляции (InstrumentedStepProfiler).

Измеряет время каждой элементарной стадии внутри итерации моделирования с наносекундным
разрешением (time.perf_counter_ns) без изменения боевого кода core/:
- Декомпозиция Source.inject (генерация позиций, направлений, энергий, времен, запись в SoA);
- Декомпозиция ParticlePropagator.step (cast_path_kernel, transport_kernel, initial_states,
  stream compaction и раздельное время каждого физического процесса: Photoelectric, Compton, Coherent);
- Декомпозиция инвалидации, сбора escaped_particles, упаковки чанков и работы DataHandler + HDF5;
- Раздельный учет стационарной фазы полного банка (Full-Bank Sustained Phase) и хвостовой фазы
  опустошения банка (Tail Depletion Phase).
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

from core.data.data_handlers import HistoryAssemblerHandler
from core.data.data_manager import DataManager
from core.geometry.geometry_kernels import cast_path_kernel
from core.other.typing_definitions import Float, Index
from core.source.source_kernels import inject_poisson_sources_kernel
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
    warmup_propagator_instance,
)


class _Win32ProcessMemoryCounters(ctypes.Structure):
    """Структура PROCESS_MEMORY_COUNTERS Win32 API, предоставляющая поле num_page_faults."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

    @property
    def num_page_faults(self) -> int:
        """Кумулятивное число page faults процесса (аналог psutil.Process().memory_info().num_page_faults)."""
        return int(self.PageFaultCount)


class _ProcessMemoryTracker:
    """Легковесный трекер памяти процесса без внешних зависимостей (совместимый по memory_info() с psutil.Process)."""

    def __init__(self) -> None:
        self._is_win32 = sys.platform == "win32"
        self._counters = _Win32ProcessMemoryCounters()
        self._counters.cb = ctypes.sizeof(_Win32ProcessMemoryCounters)
        if self._is_win32:
            self._kernel32 = ctypes.windll.kernel32
            self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            self._kernel32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_Win32ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            self._kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            self._handle = self._kernel32.GetCurrentProcess()

    def memory_info(self) -> _Win32ProcessMemoryCounters:
        """Обновляет и возвращает структуру счетчиков памяти текущего процесса."""
        if self._is_win32:
            success = self._kernel32.K32GetProcessMemoryInfo(
                self._handle,
                ctypes.byref(self._counters),
                self._counters.cb,
            )
            if not success:
                raise OSError("Не удалось получить счетчики памяти процесса через K32GetProcessMemoryInfo.")
        return self._counters


def _get_process_page_faults(proc: _ProcessMemoryTracker) -> int:
    """Возвращает кумулятивный счетчик page faults текущего процесса из ОС (Windows PageFaultCount)."""
    return proc.memory_info().num_page_faults


@dataclass(frozen=True)
class SourceSubphaseBreakdown:
    """Внутренняя декомпозиция времени и page faults инжекции первичных частиц."""

    soa_conversion_and_bank_inject: float
    soa_conversion_and_bank_inject_page_faults: int = 0

    @property
    def total(self) -> float:
        """Суммарное время фазы пополнения банка частиц [с]."""
        return self.soa_conversion_and_bank_inject

    @property
    def total_page_faults(self) -> int:
        """Суммарная дельта page faults фазы пополнения банка частиц."""
        return self.soa_conversion_and_bank_inject_page_faults


@dataclass(frozen=True)
class PhysicsProcessesBreakdown:
    """Раздельное время выполнения ядер физических процессов."""

    photoelectric_time: float
    compton_time: float
    coherent_time: float
    photoelectric_calls: int
    compton_calls: int
    coherent_calls: int
    photoelectric_events: int
    compton_events: int
    coherent_events: int

    @property
    def total(self) -> float:
        """Суммарное время всех ядер физических процессов [с]."""
        return self.photoelectric_time + self.compton_time + self.coherent_time


@dataclass(frozen=True)
class StepPhaseBreakdownResult:
    """
    Полный отчет 10-фазного пошагового профилирования для одной рабочей нагрузки
    с разделением на стационарную фазу полного банка (Full-Bank Phase) и фазу опустошения (Depletion Phase),
    а также пофазными дельтами page faults.
    """

    workload_id: str
    title: str
    collimator_kind: CollimatorKind
    particles_number: int
    total_steps: int
    total_wall_time: float
    throughput_histories_per_sec: float
    p1_scene_and_compiler_init: float
    p2_source_replenish: float
    p3_active_indices_query: float
    p4_geometry_raycasting: float
    p5_woodcock_transport: float
    p6_initial_states_push: float
    p7_process_stream_compaction: float
    p8_physics_kernels: float
    p9_invalidation_and_escaped: float
    p9_sub_escaped_only: float
    p10_data_handler_and_hdf5: float
    source_subphases: SourceSubphaseBreakdown
    physics_subphases: PhysicsProcessesBreakdown
    propagator_cold_compile_time: float = 0.0

    # Разделение на фазу полного банка (Full-Bank Phase) и фазу опустошения (Tail Depletion Phase)
    bank_capacity: int = 0
    full_bank_steps: int = 0
    depletion_steps: int = 0
    full_bank_transport_time: float = 0.0
    depletion_transport_time: float = 0.0
    full_bank_active_tracks: int = 0
    depletion_active_tracks: int = 0
    full_bank_tracks_per_sec: float = 0.0
    depletion_tracks_per_sec: float = 0.0
    depletion_track_slowdown_factor: float = 1.0
    full_bank_p2_source: float = 0.0
    full_bank_p4_geometry: float = 0.0
    full_bank_p5_woodcock: float = 0.0
    full_bank_p8_processes: float = 0.0
    full_bank_p9_compaction: float = 0.0
    depletion_p4_geometry: float = 0.0
    depletion_p5_woodcock: float = 0.0
    depletion_p8_processes: float = 0.0
    depletion_p9_compaction: float = 0.0

    # Пофазные дельты page faults (psutil.Process().memory_info().num_page_faults)
    p2_page_faults: int = 0
    p3_page_faults: int = 0
    p4_page_faults: int = 0
    p5_page_faults: int = 0
    p6_page_faults: int = 0
    p7_page_faults: int = 0
    p8_page_faults: int = 0
    p9_page_faults: int = 0
    p10_page_faults: int = 0
    full_bank_p2_page_faults: int = 0
    full_bank_p4_page_faults: int = 0
    full_bank_p5_page_faults: int = 0
    full_bank_p8_page_faults: int = 0
    full_bank_p9_page_faults: int = 0
    full_bank_p10_page_faults: int = 0
    steady_full_bank_p2_page_faults: int = 0
    steady_full_bank_p4_page_faults: int = 0
    steady_full_bank_p5_page_faults: int = 0
    steady_full_bank_p8_page_faults: int = 0
    steady_full_bank_p9_page_faults: int = 0
    steady_full_bank_p10_page_faults: int = 0

    @property
    def cold_total_wall_time(self) -> float:
        """Полное время прогона при холодном экземпляре ParticlePropagator (включая JIT замыканий) [с]."""
        return self.total_wall_time + self.propagator_cold_compile_time

    @property
    def cold_throughput_histories_per_sec(self) -> float:
        """Эффективная скорость моделирования с учетом холодной JIT-компиляции замыканий [историй/с]."""
        return float(self.particles_number) / max(self.cold_total_wall_time, 1e-9)

    @property
    def warm_vs_cold_speedup(self) -> float:
        """Коэффициент ускорения steady-state конвейера относительно холодного запуска пропагатора."""
        return self.cold_total_wall_time / max(self.total_wall_time, 1e-9)

    def phase_percentages(self) -> Dict[str, float]:
        """Возвращает процентные доли всех 10 фаз от времени цикла переноса (P2..P10)."""
        loop_time = max(
            self.p2_source_replenish
            + self.p3_active_indices_query
            + self.p4_geometry_raycasting
            + self.p5_woodcock_transport
            + self.p6_initial_states_push
            + self.p7_process_stream_compaction
            + self.p8_physics_kernels
            + self.p9_invalidation_and_escaped
            + self.p10_data_handler_and_hdf5,
            1e-12,
        )
        return {
            "P2_Source_Inject": 100.0 * self.p2_source_replenish / loop_time,
            "P3_Active_Indices": 100.0 * self.p3_active_indices_query / loop_time,
            "P4_Geometry_Raycast": 100.0 * self.p4_geometry_raycasting / loop_time,
            "P5_Woodcock_Transport": 100.0 * self.p5_woodcock_transport / loop_time,
            "P6_Initial_States": 100.0 * self.p6_initial_states_push / loop_time,
            "P7_Stream_Compaction": 100.0 * self.p7_process_stream_compaction / loop_time,
            "P8_Physics_Processes": 100.0 * self.p8_physics_kernels / loop_time,
            "P9_Invalidation_Escaped": 100.0 * self.p9_invalidation_and_escaped / loop_time,
            "P10_DataHandler_HDF5": 100.0 * self.p10_data_handler_and_hdf5 / loop_time,
        }


def run_step_phase_profiler(
    workload: ProfilingWorkload,
    buffer_capacity: int = 100_000,
    seed: int = 42,
    propagator: Optional[ParticlePropagator] = None,
    cold_compile_time_override: Optional[float] = None,
) -> StepPhaseBreakdownResult:
    """
    Запускает синхронный пофазный замер всех 10 стадий шага симуляции и подфаз Source / Physics
    в режиме Warm Steady-State с раздельным сбором метрик для фазы полного банка (Full-Bank Phase)
    и фазы опустошения банка (Tail Depletion Phase).
    """
    t_prop_init = time.perf_counter_ns()
    if propagator is None:
        propagator = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(seed),
        )
    prop_init_ns = time.perf_counter_ns() - t_prop_init

    # Предварительный прогрев замыканий пропагатора ДО включения таймеров фаз
    measured_cold_compile = warmup_propagator_instance(
        propagator=propagator,
        scene=workload.root_scene,
        rng=np.random.default_rng(seed),
    )
    if cold_compile_time_override is not None and measured_cold_compile < 0.05:
        cold_compile_sec = float(cold_compile_time_override)
    else:
        cold_compile_sec = float(measured_cold_compile)

    eff_bank_capacity = int(workload.effective_bank_capacity)
    target_total_histories = int(workload.particles_number)

    # Фаза P1: Компиляция сцены, геометрии, физики и аллокация буферов SimulationManager
    t_p1_start = time.perf_counter_ns()
    manager = SimulationManager(
        scene=workload.root_scene,
        propagator=propagator,
        stop_time=workload.stop_time,
        particles_number=eff_bank_capacity,
        min_energy=workload.min_energy,
        buffer_capacity=buffer_capacity,
        seed=seed,
        consumed_chunk_types=DEFAULT_CONSUMED_CHUNK_TYPES,
    )
    if propagator.transport_buffer.process_ids.size < manager.bank.capacity:
        propagator.transport_buffer = TransportBuffer.allocate(
            manager.bank.capacity, len(propagator.processes)
        )
    p1_compiler_ns = (
        time.perf_counter_ns() - t_p1_start + prop_init_ns + int(workload.scene_build_duration * 1e9)
    )

    proc_handle = _ProcessMemoryTracker()

    p2_ns = 0
    p3_ns = 0
    p4_ns = 0
    p5_ns = 0
    p6_ns = 0
    p7_ns = 0
    p8_ns = 0
    p9_ns = 0
    p9_escaped_ns = 0
    p10_ns = 0

    p2_pf = 0
    p3_pf = 0
    p4_pf = 0
    p5_pf = 0
    p6_pf = 0
    p7_pf = 0
    p8_pf = 0
    p9_pf = 0
    p10_pf = 0

    src_bank_ns = 0
    src_bank_pf = 0

    phys_photo_ns = 0
    phys_compton_ns = 0
    phys_coherent_ns = 0
    phys_photo_calls = 0
    phys_compton_calls = 0
    phys_coherent_calls = 0
    phys_photo_events = 0
    phys_compton_events = 0
    phys_coherent_events = 0

    full_steps = 0
    depl_steps = 0
    full_tracks = 0
    depl_tracks = 0
    full_p2_ns = 0
    full_p4_ns = 0
    full_p5_ns = 0
    full_p8_ns = 0
    full_p9_ns = 0
    depl_p4_ns = 0
    depl_p5_ns = 0
    depl_p8_ns = 0
    depl_p9_ns = 0

    full_p2_pf = 0
    full_p4_pf = 0
    full_p5_pf = 0
    full_p8_pf = 0
    full_p9_pf = 0
    full_p10_pf = 0

    steady_full_p2_pf = 0
    steady_full_p4_pf = 0
    steady_full_p5_pf = 0
    steady_full_p8_pf = 0
    steady_full_p9_pf = 0
    steady_full_p10_pf = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / f"{workload.workload_id}_step_prof.hdf5"
        handler = HistoryAssemblerHandler(
            sensitive_volumes=workload.sensitive_detectors,
            save_initial_states=True,
            compression="gzip",
        )
        data_manager = DataManager(
            filename=h5_path,
            handlers=[handler],
            queue=None,
        )
        pf_dm0 = _get_process_page_faults(proc_handle)
        t_dm_init = time.perf_counter_ns()
        data_manager.initialize_metadata()
        p10_ns += time.perf_counter_ns() - t_dm_init
        p10_pf += max(0, _get_process_page_faults(proc_handle) - pf_dm0)

        def _sync_drain_queue() -> int:
            nonlocal p10_ns, p10_pf
            pf_drain0 = _get_process_page_faults(proc_handle)
            t_drain_0 = time.perf_counter_ns()
            while not manager.queue.empty():
                raw_chunk = manager.queue.get_nowait()
                if isinstance(raw_chunk, dict):
                    frozen = DataManager._freeze_chunk(raw_chunk)
                    for h_inst in data_manager.handlers:
                        h_inst.process_chunk(frozen)
            p10_ns += time.perf_counter_ns() - t_drain_0
            delta_pf = max(0, _get_process_page_faults(proc_handle) - pf_drain0)
            p10_pf += delta_pf
            return delta_pf

        wall_start = time.perf_counter()

        while True:
            # --- P2: Пополнение банка частиц из активных источников с контролем квоты ---
            pf_p2_overhead_ns = 0
            pf_p2_start = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            emitted_so_far = int(manager.bank.count_array[0])
            remaining_quota = max(0, target_total_histories - emitted_so_far)
            did_inject = False

            if remaining_quota > 0 and manager.active_sources:
                num_active = int(np.count_nonzero(manager.bank.state.is_active))
                free_slots = int(manager.bank.capacity - num_active)
                num_to_inject = min(free_slots, remaining_quota)
            else:
                num_to_inject = 0

            if num_to_inject > 0:
                t_pf0 = time.perf_counter_ns()
                pf_s0 = _get_process_page_faults(proc_handle)
                ts0 = time.perf_counter_ns()
                pf_p2_overhead_ns += ts0 - t_pf0

                injected_cnt, _, new_t_last, new_pending = inject_poisson_sources_kernel(
                    manager.bank,
                    manager.source_buffer,
                    Float(np.inf),
                    manager.t_ref,
                    manager.t_last,
                    manager.pending_arrival_time,
                    manager.rng_ctx,
                    Index(num_to_inject),
                    Index(0),
                )
                ts1 = time.perf_counter_ns()
                pf_s1 = _get_process_page_faults(proc_handle)
                pf_p2_overhead_ns += time.perf_counter_ns() - ts1

                src_bank_ns += ts1 - ts0
                src_bank_pf += max(0, pf_s1 - pf_s0)
                manager.t_last = Float(new_t_last)
                manager.pending_arrival_time = Float(new_pending)
                manager.global_timer = manager.t_last
                if int(injected_cnt) > 0:
                    did_inject = True
            step_p2_ns = max(0, (time.perf_counter_ns() - t0) - pf_p2_overhead_ns)
            step_p2_pf = max(0, _get_process_page_faults(proc_handle) - pf_p2_start)
            p2_ns += step_p2_ns
            p2_pf += step_p2_pf

            # --- P3: Извлечение активных индексов ---
            pf_p3_0 = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            active_indices = manager.bank.active_indices
            step_p3_ns = time.perf_counter_ns() - t0
            step_p3_pf = max(0, _get_process_page_faults(proc_handle) - pf_p3_0)
            p3_ns += step_p3_ns
            p3_pf += step_p3_pf

            active_count = int(active_indices.size)
            if active_count == 0:
                if int(manager.bank.count_array[0]) >= target_total_histories or not manager.active_sources:
                    break
                continue

            is_full_bank_step = (active_count == eff_bank_capacity) or (did_inject and full_steps == 0)

            # Предварительный сброс при переполнении буферов
            pf_p9_pre0 = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            manager._ensure_step_capacity(active_count)
            step_p9_pre_ns = time.perf_counter_ns() - t0
            step_p9_pre_pf = max(0, _get_process_page_faults(proc_handle) - pf_p9_pre0)
            p9_ns += step_p9_pre_ns
            p9_pf += step_p9_pre_pf
            step_p10_pf = _sync_drain_queue()

            # --- P4: Геометрический рейкастинг (cast_path_kernel) ---
            pf_p4_0 = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            cast_path_kernel(
                manager.bank.state.position,
                manager.bank.state.direction,
                active_indices,
                manager.geometry_buffer,
                manager.bank.navigation_state,
            )
            step_p4_ns = time.perf_counter_ns() - t0
            step_p4_pf = max(0, _get_process_page_faults(proc_handle) - pf_p4_0)
            p4_ns += step_p4_ns
            p4_pf += step_p4_pf

            # --- P5: Дельта-трекинг Вудкока (transport_kernel) ---
            pf_p5_0 = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            propagator._transport_kernel(
                manager.bank.state,
                manager.bank.navigation_state,
                active_indices,
                manager.physics_buffer,
                propagator.transport_buffer,
                manager.geometry_buffer,
                manager.rng_ctx,
            )
            step_p5_ns = time.perf_counter_ns() - t0
            step_p5_pf = max(0, _get_process_page_faults(proc_handle) - pf_p5_0)
            p5_ns += step_p5_ns
            p5_pf += step_p5_pf

            # --- P6: Запись начальных состояний для частиц, провзаимодействовавших впервые ---
            pf_p6_0 = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            _push_first_interacting_to_initial_state_kernel(
                manager.bank.initial_state,
                manager.bank.state,
                active_indices,
                propagator.transport_buffer.process_ids,
                manager.data_buffer.initial_states,
            )
            step_p6_ns = time.perf_counter_ns() - t0
            step_p6_pf = max(0, _get_process_page_faults(proc_handle) - pf_p6_0)
            p6_ns += step_p6_ns
            p6_pf += step_p6_pf

            # --- P7 & P8: Stream Compaction и физические процессы ---
            step_p7_ns = 0
            step_p8_ns = 0
            step_p7_pf = 0
            step_p8_pf = 0
            for process_obj in propagator.processes:
                pf_c0 = _get_process_page_faults(proc_handle)
                t_comp0 = time.perf_counter_ns()
                mask = propagator.transport_buffer.process_ids[active_indices] == process_obj.process_id
                target_indices = active_indices[mask]
                dt_comp = time.perf_counter_ns() - t_comp0
                dpf_comp = max(0, _get_process_page_faults(proc_handle) - pf_c0)
                p7_ns += dt_comp
                p7_pf += dpf_comp
                step_p7_ns += dt_comp
                step_p7_pf += dpf_comp

                if target_indices.size > 0:
                    pf_ph0 = _get_process_page_faults(proc_handle)
                    t_phys0 = time.perf_counter_ns()
                    process_obj.apply(
                        manager.bank,
                        target_indices,
                        manager.data_buffer.interactions,
                        manager.physics_buffer,
                        propagator.transport_buffer.material_ids,
                        manager.rng_ctx,
                    )
                    dt_phys = time.perf_counter_ns() - t_phys0
                    dpf_phys = max(0, _get_process_page_faults(proc_handle) - pf_ph0)
                    p8_ns += dt_phys
                    p8_pf += dpf_phys
                    step_p8_ns += dt_phys
                    step_p8_pf += dpf_phys

                    proc_cls_name = process_obj.__class__.__name__
                    ev_count = int(target_indices.size)
                    if "Photoelectric" in proc_cls_name:
                        phys_photo_ns += dt_phys
                        phys_photo_calls += 1
                        phys_photo_events += ev_count
                    elif "Compton" in proc_cls_name:
                        phys_compton_ns += dt_phys
                        phys_compton_calls += 1
                        phys_compton_events += ev_count
                    elif "Coherent" in proc_cls_name:
                        phys_coherent_ns += dt_phys
                        phys_coherent_calls += 1
                        phys_coherent_events += ev_count

            # --- P9: Инвалидация, сбор вылетевших и обработка выбывших частиц ---
            pf_p9_post0 = _get_process_page_faults(proc_handle)
            t0 = time.perf_counter_ns()
            dead_indices = manager._apply_invalidators(active_indices)
            if dead_indices.size > 0:
                t_esc0 = time.perf_counter_ns()
                manager._record_finished_histories(dead_indices)
                p9_escaped_ns += time.perf_counter_ns() - t_esc0
            manager._check_no_overflow()
            manager.step_count += 1
            step_p9_post_ns = time.perf_counter_ns() - t0
            step_p9_post_pf = max(0, _get_process_page_faults(proc_handle) - pf_p9_post0)
            p9_ns += step_p9_post_ns
            p9_pf += step_p9_post_pf

            step_p10_pf += _sync_drain_queue()

            step_compaction_total_ns = step_p3_ns + step_p6_ns + step_p7_ns + step_p9_pre_ns + step_p9_post_ns
            step_compaction_total_pf = step_p3_pf + step_p6_pf + step_p7_pf + step_p9_pre_pf + step_p9_post_pf
            if is_full_bank_step:
                is_steady_full_step = full_steps >= 1
                full_steps += 1
                full_tracks += active_count
                full_p2_ns += step_p2_ns
                full_p4_ns += step_p4_ns
                full_p5_ns += step_p5_ns
                full_p8_ns += step_p8_ns
                full_p9_ns += step_compaction_total_ns

                full_p2_pf += step_p2_pf
                full_p4_pf += step_p4_pf
                full_p5_pf += step_p5_pf
                full_p8_pf += step_p8_pf
                full_p9_pf += step_compaction_total_pf
                full_p10_pf += step_p10_pf

                if is_steady_full_step:
                    steady_full_p2_pf += step_p2_pf
                    steady_full_p4_pf += step_p4_pf
                    steady_full_p5_pf += step_p5_pf
                    steady_full_p8_pf += step_p8_pf
                    steady_full_p9_pf += step_compaction_total_pf
                    steady_full_p10_pf += step_p10_pf
            else:
                depl_steps += 1
                depl_tracks += active_count
                depl_p4_ns += step_p4_ns
                depl_p5_ns += step_p5_ns
                depl_p8_ns += step_p8_ns
                depl_p9_ns += step_compaction_total_ns + step_p2_ns

        # Финальный сброс буферов
        pf_flush0 = _get_process_page_faults(proc_handle)
        t0 = time.perf_counter_ns()
        manager.flush_all()
        p9_ns += time.perf_counter_ns() - t0
        p9_pf += max(0, _get_process_page_faults(proc_handle) - pf_flush0)

        _sync_drain_queue()

        pf_fin0 = _get_process_page_faults(proc_handle)
        t_fin0 = time.perf_counter_ns()
        for h_inst in data_manager.handlers:
            h_inst.finalize()
        data_manager.finalize_metadata(status="completed")
        p10_ns += time.perf_counter_ns() - t_fin0
        p10_pf += max(0, _get_process_page_faults(proc_handle) - pf_fin0)

        total_wall = max(time.perf_counter() - wall_start, 1e-9)

    emitted_final = int(manager.bank.count_array[0])
    full_transport_sec = float((full_p2_ns + full_p4_ns + full_p5_ns + full_p8_ns + full_p9_ns) * 1e-9)
    full_pure_track_sec = float((full_p4_ns + full_p5_ns + full_p8_ns + full_p9_ns) * 1e-9)
    depl_transport_sec = float((depl_p4_ns + depl_p5_ns + depl_p8_ns + depl_p9_ns) * 1e-9)
    full_tps = float(full_tracks) / max(full_pure_track_sec, 1e-9)
    depl_tps = float(depl_tracks) / max(depl_transport_sec, 1e-9) if depl_steps > 0 else full_tps
    slowdown = full_tps / max(depl_tps, 1e-9) if depl_steps > 0 else 1.0

    source_breakdown = SourceSubphaseBreakdown(
        soa_conversion_and_bank_inject=float(src_bank_ns * 1e-9),
        soa_conversion_and_bank_inject_page_faults=int(src_bank_pf),
    )
    physics_breakdown = PhysicsProcessesBreakdown(
        photoelectric_time=float(phys_photo_ns * 1e-9),
        compton_time=float(phys_compton_ns * 1e-9),
        coherent_time=float(phys_coherent_ns * 1e-9),
        photoelectric_calls=phys_photo_calls,
        compton_calls=phys_compton_calls,
        coherent_calls=phys_coherent_calls,
        photoelectric_events=phys_photo_events,
        compton_events=phys_compton_events,
        coherent_events=phys_coherent_events,
    )

    return StepPhaseBreakdownResult(
        workload_id=workload.workload_id,
        title=workload.title,
        collimator_kind=workload.collimator_kind,
        particles_number=emitted_final,
        total_steps=int(manager.step_count),
        total_wall_time=float(total_wall),
        throughput_histories_per_sec=float(emitted_final / total_wall),
        p1_scene_and_compiler_init=float(p1_compiler_ns * 1e-9),
        p2_source_replenish=float(p2_ns * 1e-9),
        p3_active_indices_query=float(p3_ns * 1e-9),
        p4_geometry_raycasting=float(p4_ns * 1e-9),
        p5_woodcock_transport=float(p5_ns * 1e-9),
        p6_initial_states_push=float(p6_ns * 1e-9),
        p7_process_stream_compaction=float(p7_ns * 1e-9),
        p8_physics_kernels=float(p8_ns * 1e-9),
        p9_invalidation_and_escaped=float(p9_ns * 1e-9),
        p9_sub_escaped_only=float(p9_escaped_ns * 1e-9),
        p10_data_handler_and_hdf5=float(p10_ns * 1e-9),
        source_subphases=source_breakdown,
        physics_subphases=physics_breakdown,
        propagator_cold_compile_time=float(cold_compile_sec),
        bank_capacity=eff_bank_capacity,
        full_bank_steps=int(full_steps),
        depletion_steps=int(depl_steps),
        full_bank_transport_time=full_transport_sec,
        depletion_transport_time=depl_transport_sec,
        full_bank_active_tracks=int(full_tracks),
        depletion_active_tracks=int(depl_tracks),
        full_bank_tracks_per_sec=float(full_tps),
        depletion_tracks_per_sec=float(depl_tps),
        depletion_track_slowdown_factor=float(slowdown),
        full_bank_p2_source=float(full_p2_ns * 1e-9),
        full_bank_p4_geometry=float(full_p4_ns * 1e-9),
        full_bank_p5_woodcock=float(full_p5_ns * 1e-9),
        full_bank_p8_processes=float(full_p8_ns * 1e-9),
        full_bank_p9_compaction=float(full_p9_ns * 1e-9),
        depletion_p4_geometry=float(depl_p4_ns * 1e-9),
        depletion_p5_woodcock=float(depl_p5_ns * 1e-9),
        depletion_p8_processes=float(depl_p8_ns * 1e-9),
        depletion_p9_compaction=float(depl_p9_ns * 1e-9),
        p2_page_faults=int(p2_pf),
        p3_page_faults=int(p3_pf),
        p4_page_faults=int(p4_pf),
        p5_page_faults=int(p5_pf),
        p6_page_faults=int(p6_pf),
        p7_page_faults=int(p7_pf),
        p8_page_faults=int(p8_pf),
        p9_page_faults=int(p9_pf),
        p10_page_faults=int(p10_pf),
        full_bank_p2_page_faults=int(full_p2_pf),
        full_bank_p4_page_faults=int(full_p4_pf),
        full_bank_p5_page_faults=int(full_p5_pf),
        full_bank_p8_page_faults=int(full_p8_pf),
        full_bank_p9_page_faults=int(full_p9_pf),
        full_bank_p10_page_faults=int(full_p10_pf),
        steady_full_bank_p2_page_faults=int(steady_full_p2_pf),
        steady_full_bank_p4_page_faults=int(steady_full_p4_pf),
        steady_full_bank_p5_page_faults=int(steady_full_p5_pf),
        steady_full_bank_p8_page_faults=int(steady_full_p8_pf),
        steady_full_bank_p9_page_faults=int(steady_full_p9_pf),
        steady_full_bank_p10_page_faults=int(steady_full_p10_pf),
    )


def run_step_profiling_suite(
    workloads: List[ProfilingWorkload],
    buffer_capacity: int = 100_000,
    seed: int = 42,
) -> List[StepPhaseBreakdownResult]:
    """
    Запускает 10-фазный пошаговый профилировщик для списка эталонных сцен.
    Первичная компиляция замыканий ParticlePropagator измеряется до старта таймеров шага.
    """
    if not workloads:
        return []

    shared_propagator = ParticlePropagator(
        processes_list=processes_list,
        attenuation_database=attenuation_database,
        rng=np.random.default_rng(seed),
    )
    measured_cold_sec = warmup_propagator_instance(
        propagator=shared_propagator,
        scene=workloads[0].root_scene,
        rng=np.random.default_rng(seed),
    )

    results: List[StepPhaseBreakdownResult] = []
    for w_item in workloads:
        shared_propagator.transport_buffer = TransportBuffer.allocate(0, len(shared_propagator.processes))
        results.append(
            run_step_phase_profiler(
                workload=w_item,
                buffer_capacity=buffer_capacity,
                seed=seed,
                propagator=shared_propagator,
                cold_compile_time_override=measured_cold_sec,
            )
        )
    return results


__all__ = [
    "PhysicsProcessesBreakdown",
    "SourceSubphaseBreakdown",
    "StepPhaseBreakdownResult",
    "run_step_phase_profiler",
    "run_step_profiling_suite",
]
