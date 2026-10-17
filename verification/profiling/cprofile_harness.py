"""
Модуль профилирования всего конвейера NMSimToolkit стандартным профилировщиком CPython (cProfile / pstats).

Реализует два режима:
1. Однопоточный сквозной cProfile (End-to-End Single-Thread): транспорт + обработка чанков + запись HDF5
   в едином стеке вызовов для точного ранжирования всех функций по tottime / cumtime и доменам.
2. Штатный двухпоточный cProfile (Dual-Thread): раздельные профили потоков SimulationManager и DataManager
   для измерения времени блокировки очереди (queue.put / queue.get) и задержки дозаписи хвоста (Tail Drain Lag).
"""
from __future__ import annotations

import cProfile
from dataclasses import dataclass
from pathlib import Path
import pstats
from queue import Queue
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.data.data_handlers import BaseDataHandler, HistoryAssemblerHandler
from core.data.data_manager import DataManager
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from core.transport.transport_buffer import TransportBuffer
from settings.database_setting import attenuation_database
from settings.processes_settings import processes_list
from verification.profiling.workloads import (
    ProfilingWorkload,
    warmup_propagator_instance,
)


@dataclass(frozen=True)
class CProfileFunctionRecord:
    """Запись статистики одной функции из таблицы pstats."""

    qualified_name: str
    file_name: str
    line_number: int
    function_name: str
    call_count: int
    total_time: float
    cumulative_time: float
    percent_of_total: float


@dataclass(frozen=True)
class CProfileDomainBreakdown:
    """Агрегированное распределение собственного времени (tottime) по подсистемам конвейера."""

    total_profiled_time: float
    jit_transport_and_raycast: float
    jit_physics_processes: float
    source_generation: float
    buffer_and_escaped_management: float
    data_handler_processing: float
    hdf5_disk_io: float
    python_orchestration: float

    @property
    def jit_transport_pct(self) -> float:
        """Доля JIT-ядер геометрии и дельта-трекинга Вудкока [%]."""
        return 100.0 * self.jit_transport_and_raycast / max(self.total_profiled_time, 1e-12)

    @property
    def jit_physics_pct(self) -> float:
        """Доля JIT-ядер физических процессов (фотоэффект, Комптон, Рэлей) [%]."""
        return 100.0 * self.jit_physics_processes / max(self.total_profiled_time, 1e-12)

    @property
    def source_generation_pct(self) -> float:
        """Доля генерации и инжекции первичных частиц в Source [%]."""
        return 100.0 * self.source_generation / max(self.total_profiled_time, 1e-12)

    @property
    def buffer_and_escaped_pct(self) -> float:
        """Доля управления буферами, инвалидации и сбора вылетевших частиц [%]."""
        return 100.0 * self.buffer_and_escaped_management / max(self.total_profiled_time, 1e-12)

    @property
    def data_handler_pct(self) -> float:
        """Доля фильтрации, конкатенации и сборки историй в DataHandlers [%]."""
        return 100.0 * self.data_handler_processing / max(self.total_profiled_time, 1e-12)

    @property
    def hdf5_io_pct(self) -> float:
        """Доля дисковых операций h5py [%]."""
        return 100.0 * self.hdf5_disk_io / max(self.total_profiled_time, 1e-12)

    @property
    def orchestration_pct(self) -> float:
        """Доля прочей Python-оркестрации шага [%]."""
        return 100.0 * self.python_orchestration / max(self.total_profiled_time, 1e-12)


@dataclass(frozen=True)
class DualThreadCProfileResult:
    """Результат двухпоточного профилирования SimulationManager и DataManager."""

    workload_id: str
    collimator_kind: str
    particles_number: int
    wall_time: float
    sim_thread_time: float
    data_thread_time: float
    sim_queue_put_wait_time: float
    data_queue_get_wait_time: float
    data_active_work_time: float
    tail_drain_lag: float
    sim_prof_path: str
    data_prof_path: str
    propagator_cold_compile_time: float = 0.0


@dataclass(frozen=True)
class EndToEndCProfileResult:
    """Полный результат сквозного однопоточного профилирования cProfile + двухпоточных метрик."""

    workload_id: str
    title: str
    collimator_kind: str
    particles_number: int
    simulation_steps: int
    end_to_end_wall_time: float
    throughput_histories_per_sec: float
    domain_breakdown: CProfileDomainBreakdown
    top_by_tottime: List[CProfileFunctionRecord]
    top_by_cumtime: List[CProfileFunctionRecord]
    end_to_end_prof_path: str
    dual_thread_metrics: DualThreadCProfileResult
    propagator_cold_compile_time: float = 0.0


class _InstrumentedQueue(Queue[Any]):
    """Очередь с наносекундным замером времени ожидания на операциях put и get."""

    def __init__(self, maxsize: int = 64) -> None:
        super().__init__(maxsize=maxsize)
        self.put_wait_ns: int = 0
        self.get_wait_ns: int = 0

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        t0 = time.perf_counter_ns()
        try:
            super().put(item, block=block, timeout=timeout)
        finally:
            self.put_wait_ns += time.perf_counter_ns() - t0

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        t0 = time.perf_counter_ns()
        try:
            return super().get(block=block, timeout=timeout)
        finally:
            self.get_wait_ns += time.perf_counter_ns() - t0


class _SynchronousSimulationManager(SimulationManager):
    """
    Наследник SimulationManager для сквозного однопоточного профилирования cProfile.
    Передает каждый сформированный чанк синхронно в обработчики DataManager в том же потоке.
    """

    def __init__(
        self,
        sync_data_manager: DataManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sync_data_manager = sync_data_manager

    def send_data(self, data: Any) -> None:
        """Синхронная обработка пакета данных всеми зарегистрированными DataHandlers."""
        if isinstance(data, dict):
            frozen_chunk = DataManager._freeze_chunk(data)
            for handler_item in self._sync_data_manager.handlers:
                handler_item.process_chunk(frozen_chunk)


class _ThreadProfiledSimulationManager(SimulationManager):
    """Обертка SimulationManager с опциональным профилировщиком cProfile внутри рабочего потока."""

    def __init__(self, enable_cprofile: bool = True, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_cprofile = enable_cprofile
        self.profiler = cProfile.Profile()
        self.thread_elapsed_time: float = 0.0
        self.finish_timestamp: float = 0.0

    def run(self) -> None:
        """Запуск потока транспорта с гарантированным освобождением очереди при сбое."""
        start_ts = time.perf_counter()
        if self.enable_cprofile:
            self.profiler.enable()
        try:
            super().run()
        except BaseException:
            if self.queue is not None:
                self.queue.put("stop")
            raise
        finally:
            if self.enable_cprofile:
                self.profiler.disable()
            self.finish_timestamp = time.perf_counter()
            self.thread_elapsed_time = self.finish_timestamp - start_ts


class _ThreadProfiledDataManager(DataManager):
    """Обертка DataManager с опциональным профилировщиком cProfile внутри потока записи данных."""

    def __init__(self, enable_cprofile: bool = True, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_cprofile = enable_cprofile
        self.profiler = cProfile.Profile()
        self.thread_elapsed_time: float = 0.0
        self.finish_timestamp: float = 0.0

    def run(self) -> None:
        """Запуск потока обработки данных под контролем cProfile."""
        start_ts = time.perf_counter()
        if self.enable_cprofile:
            self.profiler.enable()
        try:
            super().run()
        finally:
            if self.enable_cprofile:
                self.profiler.disable()
            self.finish_timestamp = time.perf_counter()
            self.thread_elapsed_time = self.finish_timestamp - start_ts


def _short_file_label(raw_file: str) -> str:
    """Форматирует путь к файлу относительно корня проекта или имени пакета."""
    norm_path = raw_file.replace("\\", "/")
    for marker in ("core/", "verification/", "h5py/", "numpy/"):
        idx = norm_path.find(marker)
        if idx != -1:
            return norm_path[idx:]
    return Path(norm_path).name


def _extract_top_functions(
    profiler: cProfile.Profile,
    top_count: int = 20,
) -> Tuple[List[CProfileFunctionRecord], List[CProfileFunctionRecord], float]:
    """Извлекает топ функций по собственному (tottime) и кумулятивному (cumtime) времени из pstats."""
    stats_obj = pstats.Stats(profiler)
    raw_stats: Dict[Tuple[str, int, str], Tuple[int, int, float, float, Any]] = stats_obj.stats  # type: ignore[attr-defined]
    total_tt = sum(entry[2] for entry in raw_stats.values())
    safe_total = max(total_tt, 1e-12)

    records: List[CProfileFunctionRecord] = []
    for (file_path, line_num, func_name), (prim_calls, total_calls, tottime_val, cumtime_val, _) in raw_stats.items():
        short_file = _short_file_label(file_path)
        qual_name = f"{short_file}:{line_num}({func_name})"
        records.append(
            CProfileFunctionRecord(
                qualified_name=qual_name,
                file_name=short_file,
                line_number=int(line_num),
                function_name=func_name,
                call_count=int(total_calls),
                total_time=float(tottime_val),
                cumulative_time=float(cumtime_val),
                percent_of_total=float(100.0 * tottime_val / safe_total),
            )
        )

    by_tottime = sorted(records, key=lambda rec: rec.total_time, reverse=True)[:top_count]
    by_cumtime = sorted(records, key=lambda rec: rec.cumulative_time, reverse=True)[:top_count]
    return by_tottime, by_cumtime, float(total_tt)


def _classify_domain_breakdown(profiler: cProfile.Profile) -> CProfileDomainBreakdown:
    """
    Классифицирует собственное время (tottime) всех функций в cProfile по 7 доменам конвейера,
    используя имена файлов, функций и граф вызывающих функций (callers).
    """
    stats_obj = pstats.Stats(profiler)
    raw_stats: Dict[Tuple[str, int, str], Tuple[int, int, float, float, Dict[Tuple[str, int, str], Tuple[int, int, float, float]]]] = (
        stats_obj.stats  # type: ignore[attr-defined]
    )

    total_time = 0.0
    jit_transport = 0.0
    jit_physics = 0.0
    source_gen = 0.0
    buffer_escaped = 0.0
    data_handler = 0.0
    hdf5_io = 0.0
    orchestration = 0.0

    for (file_path, _line_num, func_name), (_cc, _nc, tottime_val, _ct, callers_dict) in raw_stats.items():
        if tottime_val <= 0.0:
            continue
        total_time += tottime_val
        norm_file = file_path.replace("\\", "/").lower()
        fn_lower = func_name.lower()

        if (
            "interaction_buffers.py" in norm_file
            or fn_lower
            in (
                "_push_to_initial_state_kernel",
                "_push_first_interacting_to_initial_state_kernel",
                "_push_uncollided_to_initial_state_kernel",
                "_push_to_dead_particles_kernel",
                "_push_to_escaped_particles_kernel",
                "_write_single_initial_state",
                "_is_uncollided",
                "_ensure_step_capacity",
                "_record_finished_histories",
                "_check_no_overflow",
                "_apply_invalidators",
                "_invalidate_by_energy",
                "_invalidate_by_volume",
                "_flush_channel",
                "_flush_through",
                "flush_all",
                "flush",
                "active_indices",
            )
        ):
            buffer_escaped += tottime_val
        elif (
            "cast_path_kernel" in fn_lower
            or "transport_kernel" in fn_lower
            or "propagator.py" in norm_file
            or "transport_kernels.py" in norm_file
            or "geometry_kernels.py" in norm_file
        ):
            jit_transport += tottime_val
        elif (
            "_photoelectric_kernel" in fn_lower
            or "_compton_kernel" in fn_lower
            or "_coherent_kernel" in fn_lower
            or "update_navigation_state_rotate_kernel" in fn_lower
            or "processes_kernels.py" in norm_file
            or "processes.py" in norm_file
        ):
            jit_physics += tottime_val
        elif (
            "h5py" in norm_file
            or "data_manager.py" in norm_file
            or "write_func" in fn_lower
            or "_write_with_retry" in fn_lower
            or "_write_initial_states" in fn_lower
            or "_write_interactions" in fn_lower
        ):
            hdf5_io += tottime_val
        elif (
            "sources.py" in norm_file
            or "particles.py" in norm_file
            or "particles_kernels.py" in norm_file
            or fn_lower in ("_replenish_bank", "_calculate_time_step", "_distribute_quotas", "inject_particles")
            or "bit_generator" in fn_lower
            or "_generator" in norm_file
        ):
            source_gen += tottime_val
        elif "data_handlers.py" in norm_file or "dose_map_handler.py" in norm_file:
            data_handler += tottime_val
        else:
            # Для универсальных C-функций NumPy (array, concatenate, isin, argsort, copyto, count_nonzero)
            # распределяем их tottime пропорционально tottime со стороны вызывающих модулей (callers)
            if callers_dict:
                caller_weights = {
                    "source": 0.0,
                    "handler": 0.0,
                    "buffer": 0.0,
                    "hdf5": 0.0,
                    "orch": 0.0,
                }
                for (c_file, _c_line, c_func), c_tuple in callers_dict.items():
                    c_norm = c_file.replace("\\", "/").lower()
                    c_fn = c_func.lower()
                    c_weight = float(c_tuple[2]) if len(c_tuple) > 2 else 1.0
                    if "sources.py" in c_norm or "particles.py" in c_norm or c_fn in ("_replenish_bank", "_distribute_quotas"):
                        caller_weights["source"] += c_weight
                    elif (
                        "data_handlers.py" in c_norm
                        or "dose_map_handler.py" in c_norm
                        or "arraysetops" in c_norm
                        or "_flush_dead_particles" in c_fn
                        or "_format_and_write" in c_fn
                        or "_process_interactions" in c_fn
                    ):
                        caller_weights["handler"] += c_weight
                    elif "h5py" in c_norm or "data_manager.py" in c_norm:
                        caller_weights["hdf5"] += c_weight
                    elif c_fn in (
                        "_ensure_step_capacity",
                        "_record_finished_histories",
                        "_check_no_overflow",
                        "_apply_invalidators",
                        "_invalidate_by_energy",
                        "_invalidate_by_volume",
                        "_flush_channel",
                        "_flush_through",
                        "flush_all",
                        "flush",
                        "active_indices",
                    ) or "interaction_buffers.py" in c_norm:
                        caller_weights["buffer"] += c_weight
                    else:
                        caller_weights["orch"] += c_weight

                weight_sum = sum(caller_weights.values())
                if weight_sum > 0.0:
                    source_gen += tottime_val * (caller_weights["source"] / weight_sum)
                    data_handler += tottime_val * (caller_weights["handler"] / weight_sum)
                    hdf5_io += tottime_val * (caller_weights["hdf5"] / weight_sum)
                    buffer_escaped += tottime_val * (caller_weights["buffer"] / weight_sum)
                    orchestration += tottime_val * (caller_weights["orch"] / weight_sum)
                else:
                    orchestration += tottime_val
            else:
                orchestration += tottime_val

    return CProfileDomainBreakdown(
        total_profiled_time=float(total_time),
        jit_transport_and_raycast=float(jit_transport),
        jit_physics_processes=float(jit_physics),
        source_generation=float(source_gen),
        buffer_and_escaped_management=float(buffer_escaped),
        data_handler_processing=float(data_handler),
        hdf5_disk_io=float(hdf5_io),
        python_orchestration=float(orchestration),
    )


def run_dual_thread_cprofile(
    workload: ProfilingWorkload,
    stats_dir: Path,
    buffer_capacity: int = 100_000,
    compression: Optional[str] = "gzip",
    propagator: Optional[ParticlePropagator] = None,
    cold_compile_time_override: Optional[float] = None,
) -> DualThreadCProfileResult:
    """
    Выполняет двухпоточное профилирование (SimulationManager + DataManager в отдельных потоках)
    в режиме Warm Steady-State с сохранением .prof дампов каждого потока в stats_dir.
    Поскольку sys.monitoring в Python 3.12+ использует единый слот PROFILER_ID на процесс,
    профили потоков транспорта и записи снимаются в два последовательных двухпоточных прогона.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)

    if propagator is None:
        propagator_inst = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(42),
        )
    else:
        propagator_inst = propagator

    # Предварительный прогрев замыканий пропагатора ДО запуска потоков и cProfile
    measured_cold = warmup_propagator_instance(
        propagator=propagator_inst,
        scene=workload.root_scene,
        rng=np.random.default_rng(42),
    )
    if cold_compile_time_override is not None and measured_cold < 0.05:
        cold_compile_sec = float(cold_compile_time_override)
    else:
        cold_compile_sec = float(measured_cold)

    # Прогон 1: cProfile активен в потоке SimulationManager
    workload.source_node.rng = np.random.default_rng(42)
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))
    queue_sim = _InstrumentedQueue(maxsize=64)

    with tempfile.TemporaryDirectory() as tmp_dir_sim:
        h5_path_sim = Path(tmp_dir_sim) / f"{workload.workload_id}_dual_sim.hdf5"
        handler_sim: BaseDataHandler = HistoryAssemblerHandler(
            sensitive_volumes=workload.sensitive_detectors,
            save_initial_states=True,
            compression=compression,
        )
        data_manager_plain = _ThreadProfiledDataManager(
            enable_cprofile=False,
            filename=h5_path_sim,
            handlers=[handler_sim],
            queue=queue_sim,
        )
        data_manager_plain.initialize_metadata()

        sim_manager = _ThreadProfiledSimulationManager(
            enable_cprofile=True,
            scene=workload.root_scene,
            propagator=propagator_inst,
            stop_time=workload.stop_time,
            particles_number=workload.particles_number,
            min_energy=workload.min_energy,
            queue=queue_sim,
            buffer_capacity=buffer_capacity,
            seed=42,
            consumed_chunk_types=data_manager_plain.consumed_chunk_types,
        )

        wall_start = time.perf_counter()
        data_manager_plain.start()
        sim_manager.start()
        sim_manager.join()
        data_manager_plain.join()
        wall_elapsed = time.perf_counter() - wall_start
        handler_sim.close()

    # Прогон 2: cProfile активен в потоке DataManager (использует уже прогретый propagator_inst)
    workload.source_node.rng = np.random.default_rng(42)
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))
    queue_data = _InstrumentedQueue(maxsize=64)

    with tempfile.TemporaryDirectory() as tmp_dir_data:
        h5_path_data = Path(tmp_dir_data) / f"{workload.workload_id}_dual_data.hdf5"
        handler_data: BaseDataHandler = HistoryAssemblerHandler(
            sensitive_volumes=workload.sensitive_detectors,
            save_initial_states=True,
            compression=compression,
        )
        data_manager = _ThreadProfiledDataManager(
            enable_cprofile=True,
            filename=h5_path_data,
            handlers=[handler_data],
            queue=queue_data,
        )
        data_manager.initialize_metadata()

        sim_manager_plain = _ThreadProfiledSimulationManager(
            enable_cprofile=False,
            scene=workload.root_scene,
            propagator=propagator_inst,
            stop_time=workload.stop_time,
            particles_number=workload.particles_number,
            min_energy=workload.min_energy,
            queue=queue_data,
            buffer_capacity=buffer_capacity,
            seed=42,
            consumed_chunk_types=data_manager.consumed_chunk_types,
        )

        data_manager.start()
        sim_manager_plain.start()
        sim_manager_plain.join()
        data_manager.join()
        handler_data.close()

    sim_prof_file = stats_dir / f"{workload.workload_id}_sim_thread.prof"
    data_prof_file = stats_dir / f"{workload.workload_id}_data_thread.prof"
    sim_manager.profiler.dump_stats(str(sim_prof_file))
    data_manager.profiler.dump_stats(str(data_prof_file))

    sim_put_wait = float(queue_sim.put_wait_ns * 1e-9)
    data_get_wait = min(float(queue_data.get_wait_ns * 1e-9), float(data_manager.thread_elapsed_time))
    data_active = max(0.0, float(data_manager.thread_elapsed_time) - data_get_wait)
    tail_lag = max(0.0, data_manager_plain.finish_timestamp - sim_manager.finish_timestamp)

    return DualThreadCProfileResult(
        workload_id=workload.workload_id,
        collimator_kind=workload.collimator_kind,
        particles_number=workload.particles_number,
        wall_time=float(wall_elapsed),
        sim_thread_time=float(sim_manager.thread_elapsed_time),
        data_thread_time=float(data_manager.thread_elapsed_time),
        sim_queue_put_wait_time=float(sim_put_wait),
        data_queue_get_wait_time=float(data_get_wait),
        data_active_work_time=float(data_active),
        tail_drain_lag=float(tail_lag),
        sim_prof_path=str(sim_prof_file),
        data_prof_path=str(data_prof_file),
        propagator_cold_compile_time=float(cold_compile_sec),
    )


def run_end_to_end_cprofile(
    workload: ProfilingWorkload,
    stats_dir: Path,
    buffer_capacity: int = 100_000,
    compression: Optional[str] = "gzip",
    top_count: int = 20,
    propagator: Optional[ParticlePropagator] = None,
    cold_compile_time_override: Optional[float] = None,
) -> EndToEndCProfileResult:
    """
    Запускает полное сквозное однопоточное профилирование cProfile и двухпоточный замер
    для выбранной нагрузки в режиме Warm Steady-State, сохраняя дампы .prof в stats_dir.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)

    if propagator is None:
        propagator_inst = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(42),
        )
    else:
        propagator_inst = propagator

    # Прогрев замыканий ParticlePropagator ДО любых замеров
    measured_cold = warmup_propagator_instance(
        propagator=propagator_inst,
        scene=workload.root_scene,
        rng=np.random.default_rng(42),
    )
    if cold_compile_time_override is not None and measured_cold < 0.05:
        cold_compile_sec = float(cold_compile_time_override)
    else:
        cold_compile_sec = float(measured_cold)

    # 1. Снимаем двухпоточные метрики на прогретом пропагаторе
    dual_metrics = run_dual_thread_cprofile(
        workload=workload,
        stats_dir=stats_dir,
        buffer_capacity=buffer_capacity,
        compression=compression,
        propagator=propagator_inst,
        cold_compile_time_override=cold_compile_sec,
    )

    # 2. Сбрасываем RNG и буфер пропагатора, затем запускаем сквозной однопоточный cProfile
    workload.source_node.rng = np.random.default_rng(42)
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))

    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / f"{workload.workload_id}_sync.hdf5"
        handler = HistoryAssemblerHandler(
            sensitive_volumes=workload.sensitive_detectors,
            save_initial_states=True,
            compression=compression,
        )
        sync_dm = DataManager(
            filename=h5_path,
            handlers=[handler],
            queue=None,
        )
        sync_dm.initialize_metadata()

        sync_sim = _SynchronousSimulationManager(
            sync_data_manager=sync_dm,
            scene=workload.root_scene,
            propagator=propagator_inst,
            stop_time=workload.stop_time,
            particles_number=workload.particles_number,
            min_energy=workload.min_energy,
            buffer_capacity=buffer_capacity,
            seed=42,
            consumed_chunk_types=sync_dm.consumed_chunk_types,
        )

        profiler = cProfile.Profile()
        start_wall = time.perf_counter()
        profiler.enable()
        try:
            while sync_sim.has_active_particles or sync_sim.has_pending_sources:
                sync_sim.next_step()
            sync_sim.flush_all()
            for handler_item in sync_dm.handlers:
                handler_item.finalize()
            sync_dm.finalize_metadata(status="completed")
        finally:
            profiler.disable()
        end_to_end_elapsed = max(time.perf_counter() - start_wall, 1e-9)

    e2e_prof_file = stats_dir / f"{workload.workload_id}_end_to_end.prof"
    profiler.dump_stats(str(e2e_prof_file))

    top_tottime, top_cumtime, _ = _extract_top_functions(profiler, top_count=top_count)
    domain_breakdown = _classify_domain_breakdown(profiler)
    throughput = float(workload.particles_number) / end_to_end_elapsed

    return EndToEndCProfileResult(
        workload_id=workload.workload_id,
        title=workload.title,
        collimator_kind=workload.collimator_kind,
        particles_number=workload.particles_number,
        simulation_steps=int(sync_sim.step_count),
        end_to_end_wall_time=float(end_to_end_elapsed),
        throughput_histories_per_sec=float(throughput),
        domain_breakdown=domain_breakdown,
        top_by_tottime=top_tottime,
        top_by_cumtime=top_cumtime,
        end_to_end_prof_path=str(e2e_prof_file),
        dual_thread_metrics=dual_metrics,
        propagator_cold_compile_time=float(cold_compile_sec),
    )


def run_cprofile_suite(
    workloads: List[ProfilingWorkload],
    stats_dir: Path,
) -> List[EndToEndCProfileResult]:
    """
    Запускает сквозное и двухпоточное профилирование cProfile для списка эталонных нагрузок
    на предварительно прогретом экземпляре ParticlePropagator.
    """
    if not workloads:
        return []

    shared_propagator = ParticlePropagator(
        processes_list=processes_list,
        attenuation_database=attenuation_database,
        rng=np.random.default_rng(42),
    )
    measured_cold_sec = warmup_propagator_instance(
        propagator=shared_propagator,
        scene=workloads[0].root_scene,
        rng=np.random.default_rng(42),
    )

    return [
        run_end_to_end_cprofile(
            w_item,
            stats_dir=stats_dir,
            propagator=shared_propagator,
            cold_compile_time_override=measured_cold_sec,
        )
        for w_item in workloads
    ]


__all__ = [
    "CProfileDomainBreakdown",
    "CProfileFunctionRecord",
    "DualThreadCProfileResult",
    "EndToEndCProfileResult",
    "run_cprofile_suite",
    "run_dual_thread_cprofile",
    "run_end_to_end_cprofile",
]
