"""
Модуль сравнительного профилирования обработчиков данных (DataHandlers) и подсистемы HDF5 I/O.

Позволяет точно измерить, какая доля времени симуляции тратится на:
1. Чистый транспорт частиц (SimulationManager без обработки чанков);
2. Формирование неиспользуемых пакетов escaped_particles (_collect_escaped_particles);
3. Вычисления в памяти внутри обработчика (NumPy-фильтрация, конкатенация, сортировка, np.add.at);
4. Дисковый ввод-вывод h5py (открытие/закрытие файла, resize датасетов, сжатие gzip).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray

from core.data.data_handlers import (
    BaseDataHandler,
    DirectStreamHandler,
    HistoryAssemblerHandler,
    SensitiveVolumeHandler,
)
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.other.typing_definitions import Index
from core.physics.interaction_buffers import ChunkType
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from core.transport.transport_buffer import TransportBuffer
from settings.database_setting import attenuation_database
from settings.processes_settings import processes_list
from verification.profiling.workloads import (
    CollimatorKind,
    ProfilingWorkload,
    build_nema_tb161_workload,
    build_point_source_workload,
    warmup_propagator_instance,
)


class NullDrainHandler(BaseDataHandler):
    """Эталонный пустой обработчик данных (No-Op), мгновенно отбрасывающий входящие чанки."""

    _DEFAULT_CONSUMED: FrozenSet[ChunkType] = frozenset({
        ChunkType.INITIAL_STATES,
        ChunkType.INTERACTIONS,
        ChunkType.DEAD_PARTICLES,
        ChunkType.ESCAPED_PARTICLES,
    })

    def __init__(self, consumed_types: Optional[FrozenSet[ChunkType]] = None) -> None:
        super().__init__()
        self.received_chunks: int = 0
        self._consumed_types: FrozenSet[ChunkType] = (
            self._DEFAULT_CONSUMED if consumed_types is None else consumed_types
        )

    @property
    def consumed_chunk_types(self) -> FrozenSet[ChunkType]:
        return self._consumed_types

    def process_chunk(self, chunk: Dict[str, Any]) -> None:
        """Увеличивает счетчик полученных пакетов без обработки массивов и записи в HDF5."""
        self.received_chunks += 1


@dataclass(frozen=True)
class DataHandlerBenchmarkRecord:
    """
    Метрики производительности одной конфигурации обработчика данных на эталонной сцене.
    """

    workload_id: str
    collimator_kind: CollimatorKind
    handler_code: str
    handler_label: str
    particles_number: int
    buffer_capacity: int
    compression: str
    collect_escaped_enabled: bool
    total_sync_time: float
    transport_only_time: float
    handler_cpu_time: float
    hdf5_io_time: float
    threaded_wall_time: float
    threaded_tail_lag: float
    overhead_vs_null_no_escaped_pct: float
    data_share_of_total_pct: float
    chunks_dispatched: int
    hdf5_write_transactions: int
    hdf5_file_size_mb: float
    propagator_cold_compile_time: float = 0.0


@dataclass(frozen=True)
class BufferCapacityScalingRecord:
    """
    Зависимость времени работы HistoryAssemblerHandler и HDF5 от размера buffer_capacity.
    """

    workload_id: str
    collimator_kind: CollimatorKind
    particles_number: int
    buffer_capacity: int
    total_sync_time: float
    transport_time: float
    handler_cpu_time: float
    hdf5_io_time: float
    chunks_dispatched: int
    hdf5_write_transactions: int


class _TimingDataManager(DataManager):
    """
    Инструментированный наследник DataManager, замеряющий чистое время дисковых транзакций h5py
    и число открытий файла HDF5 в _write_with_retry.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hdf5_io_accumulated_time: float = 0.0
        self.hdf5_transactions_count: int = 0

    def _write_with_retry(self, write_func: Callable[[h5py.File], None]) -> None:
        io_start = time.perf_counter()
        super()._write_with_retry(write_func)
        self.hdf5_io_accumulated_time += time.perf_counter() - io_start
        self.hdf5_transactions_count += 1


class _InstrumentedSyncSimManager(SimulationManager):
    """
    Однопоточный наследник SimulationManager, разделяющий время чистого транспорта
    и время обработки чанков в DataHandlers + HDF5.
    """

    def __init__(
        self,
        timing_dm: _TimingDataManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._timing_dm = timing_dm
        self.handler_total_time: float = 0.0
        self.chunks_dispatched: int = 0

    def send_data(self, data: Any) -> None:
        if isinstance(data, dict):
            self.chunks_dispatched += 1
            h_start = time.perf_counter()
            frozen_chunk = DataManager._freeze_chunk(data)
            for handler_item in self._timing_dm.handlers:
                handler_item.process_chunk(frozen_chunk)
            self.handler_total_time += time.perf_counter() - h_start


class _TimestampedSimManager(SimulationManager):
    """Двухпоточный наследник SimulationManager, фиксирующий точный момент завершения потока симуляции."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.finish_timestamp: float = 0.0

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.finish_timestamp = time.perf_counter()


class _TimestampedDataManager(DataManager):
    """Двухпоточный DataManager, фиксирующий точный момент завершения обработки всех чанков."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.finish_timestamp: float = 0.0

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.finish_timestamp = time.perf_counter()


def _create_handlers_for_mode(
    handler_code: str,
    workload: ProfilingWorkload,
    compression: Optional[str],
) -> Sequence[BaseDataHandler]:
    """Создает свежие экземпляры обработчиков данных для заданного кода режима."""
    if handler_code == "H0_NullDrain_NoEscaped":
        return [
            NullDrainHandler(
                consumed_types=frozenset({
                    ChunkType.INITIAL_STATES,
                    ChunkType.INTERACTIONS,
                    ChunkType.DEAD_PARTICLES,
                })
            )
        ]
    if handler_code == "H0_NullDrain_WithEscaped":
        return [NullDrainHandler()]
    if handler_code == "H1_SensitiveVolume":
        return [
            SensitiveVolumeHandler(
                sensitive_volumes=workload.sensitive_detectors,
                compression=compression,
            )
        ]
    if handler_code == "H2_HistoryAssembler_NoInit":
        return [
            HistoryAssemblerHandler(
                sensitive_volumes=workload.sensitive_detectors,
                save_initial_states=False,
                compression=compression,
            )
        ]
    if handler_code in ("H3_HistoryAssembler_Full_Gzip", "H4_HistoryAssembler_Full_NoComp"):
        return [
            HistoryAssemblerHandler(
                sensitive_volumes=workload.sensitive_detectors,
                save_initial_states=True,
                compression=compression,
            )
        ]
    if handler_code == "H5_DirectStream":
        return [
            DirectStreamHandler(
                compression=compression,
            )
        ]
    if handler_code == "H6_DoseMap":
        return [
            DoseMapHandler(
                grid_nodes=workload.dose_grids,
                create_shm=False,
            )
        ]
    if handler_code == "H7_Combined_History_And_Dose":
        return [
            HistoryAssemblerHandler(
                sensitive_volumes=workload.sensitive_detectors,
                save_initial_states=True,
                compression=compression,
            ),
            DoseMapHandler(
                grid_nodes=workload.dose_grids,
                create_shm=False,
            ),
        ]
    raise ValueError(f"Неизвестный код обработчика данных: {handler_code}")


HANDLER_SPECS: Tuple[Tuple[str, str, Optional[str], bool], ...] = (
    ("H0_NullDrain_NoEscaped", "Baseline: Без обработки и без escaped_particles", None, False),
    ("H0_NullDrain_WithEscaped", "NullDrain: С генерацией всех чанков включая escaped_particles", None, True),
    ("H1_SensitiveVolume", "SensitiveVolumeHandler (только детекторы, gzip)", "gzip", False),
    ("H2_HistoryAssembler_NoInit", "HistoryAssembler (без initial_states, gzip)", "gzip", False),
    ("H3_HistoryAssembler_Full_Gzip", "HistoryAssembler (полный штатный, gzip)", "gzip", False),
    ("H4_HistoryAssembler_Full_NoComp", "HistoryAssembler (полный, без сжатия)", None, False),
    ("H5_DirectStream", "DirectStreamHandler (все объемы сцены, gzip)", "gzip", False),
    ("H6_DoseMap", "DoseMapHandler (3D-сетка дозы np.add.at)", None, False),
    ("H7_Combined_History_And_Dose", "HistoryAssembler + DoseMapHandler (gzip)", "gzip", False),
)


def run_single_handler_benchmark(
    workload: ProfilingWorkload,
    handler_code: str,
    handler_label: str,
    compression: Optional[str],
    enable_escaped: bool = False,
    buffer_capacity: int = 100_000,
    baseline_transport_time: Optional[float] = None,
    seed: int = 42,
    propagator: Optional[ParticlePropagator] = None,
    cold_compile_time_override: Optional[float] = None,
) -> DataHandlerBenchmarkRecord:
    """
    Запускает синхронный (с разделением времени Transport / Handler CPU / HDF5 I/O)
    и двухпоточный замеры для одной конфигурации обработчика данных в режиме Warm Steady-State.
    """
    _ = enable_escaped
    is_null = handler_code.startswith("H0_")

    if propagator is None:
        propagator_inst = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(seed),
        )
    else:
        propagator_inst = propagator

    # Прогрев замыканий ParticlePropagator ДО включения таймеров транспорта
    measured_cold = warmup_propagator_instance(
        propagator=propagator_inst,
        scene=workload.root_scene,
        rng=np.random.default_rng(seed),
    )
    if cold_compile_time_override is not None and measured_cold < 0.05:
        cold_compile_sec = float(cold_compile_time_override)
    else:
        cold_compile_sec = float(measured_cold)

    # 1. Синхронный профилировочный прогон для точного разделения времени
    workload.source_node.rng = np.random.default_rng(seed)
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))

    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_sync_path = Path(tmp_dir) / f"{workload.workload_id}_{handler_code}_sync.hdf5"
        handlers_sync = list(_create_handlers_for_mode(handler_code, workload, compression))
        timing_dm = _TimingDataManager(
            filename=h5_sync_path,
            handlers=handlers_sync,
            queue=None,
        )
        escaped_active = bool(ChunkType.ESCAPED_PARTICLES in timing_dm.consumed_chunk_types)

        sync_manager = _InstrumentedSyncSimManager(
            timing_dm=timing_dm,
            scene=workload.root_scene,
            propagator=propagator_inst,
            stop_time=workload.stop_time,
            particles_number=workload.particles_number,
            min_energy=workload.min_energy,
            buffer_capacity=buffer_capacity,
            seed=seed,
            consumed_chunk_types=timing_dm.consumed_chunk_types,
        )

        sync_start = time.perf_counter()
        if not is_null:
            timing_dm.initialize_metadata()

        while sync_manager.has_active_particles or sync_manager.has_pending_sources:
            sync_manager.next_step()

        sync_manager.flush_all()

        fin_start = time.perf_counter()
        for handler_item in timing_dm.handlers:
            handler_item.finalize()
        if not is_null:
            timing_dm.finalize_metadata(status="completed")
        sync_manager.handler_total_time += time.perf_counter() - fin_start

        total_sync_elapsed = max(time.perf_counter() - sync_start, 1e-9)

        file_size_mb = 0.0
        if h5_sync_path.exists():
            file_size_mb = float(h5_sync_path.stat().st_size) / (1024.0 * 1024.0)

        for handler_item in handlers_sync:
            handler_item.close()

    hdf5_io_elapsed = float(timing_dm.hdf5_io_accumulated_time)
    handler_cpu_elapsed = max(0.0, float(sync_manager.handler_total_time) - hdf5_io_elapsed)
    transport_elapsed = max(0.0, total_sync_elapsed - float(sync_manager.handler_total_time))

    # 2. Штатный двухпоточный прогон (использует уже прогретый propagator_inst)
    workload.source_node.rng = np.random.default_rng(seed)
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))

    with tempfile.TemporaryDirectory() as tmp_dir_thread:
        h5_thread_path = Path(tmp_dir_thread) / f"{workload.workload_id}_{handler_code}_thread.hdf5"
        handlers_thread = list(_create_handlers_for_mode(handler_code, workload, compression))
        consumed_thread = frozenset().union(*(h.consumed_chunk_types for h in handlers_thread))
        thread_sim = _TimestampedSimManager(
            scene=workload.root_scene,
            propagator=propagator_inst,
            stop_time=workload.stop_time,
            particles_number=workload.particles_number,
            min_energy=workload.min_energy,
            buffer_capacity=buffer_capacity,
            seed=seed,
            consumed_chunk_types=consumed_thread,
        )
        thread_dm = _TimestampedDataManager(
            filename=h5_thread_path,
            handlers=handlers_thread,
            queue=thread_sim.queue,
        )
        if not is_null:
            thread_dm.initialize_metadata()

        wall_start = time.perf_counter()
        thread_dm.start()
        thread_sim.start()
        thread_sim.join()
        thread_dm.join()
        threaded_wall = max(time.perf_counter() - wall_start, 1e-9)
        tail_lag = max(0.0, thread_dm.finish_timestamp - thread_sim.finish_timestamp)

        for handler_item in handlers_thread:
            handler_item.close()

    ref_time = baseline_transport_time if baseline_transport_time is not None else total_sync_elapsed
    overhead_pct = 100.0 * (total_sync_elapsed - ref_time) / max(ref_time, 1e-9)
    data_share_pct = 100.0 * (handler_cpu_elapsed + hdf5_io_elapsed) / max(total_sync_elapsed, 1e-9)

    return DataHandlerBenchmarkRecord(
        workload_id=workload.workload_id,
        collimator_kind=workload.collimator_kind,
        handler_code=handler_code,
        handler_label=handler_label,
        particles_number=workload.particles_number,
        buffer_capacity=buffer_capacity,
        compression=compression or "none",
        collect_escaped_enabled=escaped_active,
        total_sync_time=float(total_sync_elapsed),
        transport_only_time=float(transport_elapsed),
        handler_cpu_time=float(handler_cpu_elapsed),
        hdf5_io_time=float(hdf5_io_elapsed),
        threaded_wall_time=float(threaded_wall),
        threaded_tail_lag=float(tail_lag),
        overhead_vs_null_no_escaped_pct=float(overhead_pct),
        data_share_of_total_pct=float(data_share_pct),
        chunks_dispatched=int(sync_manager.chunks_dispatched),
        hdf5_write_transactions=int(timing_dm.hdf5_transactions_count),
        hdf5_file_size_mb=float(file_size_mb),
        propagator_cold_compile_time=float(cold_compile_sec),
    )


def run_data_handlers_comparison_suite(
    particles_number: int = 100_000,
    buffer_capacity: int = 100_000,
) -> Tuple[List[DataHandlerBenchmarkRecord], List[BufferCapacityScalingRecord]]:
    """
    Выполняет полное сравнение всех 9 конфигураций DataHandlers на сценах W1 (PointSource) и W2 (NEMA 161Tb)
    для обоих видов коллиматоров (Parametric и Direct) в режиме Warm Steady-State,
    а также тест чувствительности к buffer_capacity.
    """
    all_records: List[DataHandlerBenchmarkRecord] = []

    workloads: List[ProfilingWorkload] = [
        build_point_source_workload("parametric", particles_number=particles_number),
        build_point_source_workload("direct", particles_number=particles_number),
        build_nema_tb161_workload("parametric", particles_number=particles_number),
        build_nema_tb161_workload("direct", particles_number=particles_number),
    ]

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

    for workload in workloads:
        baseline_time: Optional[float] = None
        for h_code, h_label, comp_mode, esc_flag in HANDLER_SPECS:
            rec = run_single_handler_benchmark(
                workload=workload,
                handler_code=h_code,
                handler_label=h_label,
                compression=comp_mode,
                enable_escaped=esc_flag,
                buffer_capacity=buffer_capacity,
                baseline_transport_time=baseline_time,
                propagator=shared_propagator,
                cold_compile_time_override=measured_cold_sec,
            )
            if baseline_time is None:
                baseline_time = rec.total_sync_time
            all_records.append(rec)

    # Исследование влияния buffer_capacity на HistoryAssemblerHandler (W2 NEMA Direct и Parametric)
    scaling_records: List[BufferCapacityScalingRecord] = []
    capacities = (5_000, 15_000, 50_000, 200_000)
    for coll_kind in ("parametric", "direct"):
        w_nema = build_nema_tb161_workload(coll_kind, particles_number=particles_number)
        for cap_val in capacities:
            rec = run_single_handler_benchmark(
                workload=w_nema,
                handler_code="H3_HistoryAssembler_Full_Gzip",
                handler_label=f"HistoryAssembler (buffer={cap_val})",
                compression="gzip",
                enable_escaped=True,
                buffer_capacity=cap_val,
                propagator=shared_propagator,
                cold_compile_time_override=measured_cold_sec,
            )
            scaling_records.append(
                BufferCapacityScalingRecord(
                    workload_id=w_nema.workload_id,
                    collimator_kind=coll_kind,
                    particles_number=particles_number,
                    buffer_capacity=cap_val,
                    total_sync_time=rec.total_sync_time,
                    transport_time=rec.transport_only_time,
                    handler_cpu_time=rec.handler_cpu_time,
                    hdf5_io_time=rec.hdf5_io_time,
                    chunks_dispatched=rec.chunks_dispatched,
                    hdf5_write_transactions=rec.hdf5_write_transactions,
                )
            )

    return all_records, scaling_records


__all__ = [
    "NullDrainHandler",
    "DataHandlerBenchmarkRecord",
    "BufferCapacityScalingRecord",
    "HANDLER_SPECS",
    "run_single_handler_benchmark",
    "run_data_handlers_comparison_suite",
]
