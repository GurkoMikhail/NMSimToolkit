"""
Модуль профилирования параллельного оркестратора (Orchestrator) и конкуренции за HDF5.

Исследует:
1. Масштабируемость параллельного пула процессов (pool_size = 1, 2, 4) на многошаговом
   протоколе ОФЭКТ (StepAndShoot) для обоих типов коллиматоров (Parametric и Direct).
2. Конкуренцию воркеров за межпроцессный файловый лок (multiprocessing.Manager.Lock)
   при записи в единый общий HDF5-файл по сравнению с записью в индивидуальные файлы на каждый ракурс.
3. Долю накладных расходов на инициализацию задачи в воркере (чтение RAW-фантомов,
   построение таблиц эмиссии, компиляция геометрии и физики) в общем времени выполнения.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List

import yaml

from core.config.orchestrator import Orchestrator
from verification.profiling.workloads import CollimatorKind


@dataclass(frozen=True)
class OrchestratorScalingRecord:
    """Метрики одного прогона параллельного оркестратора."""

    collimator_type: str
    pool_size: int
    tasks_count: int
    file_mode: str  # "per_task_file" или "shared_single_file"
    wall_time: float
    tasks_per_second: float
    speedup_vs_seq: float
    parallel_efficiency: float
    total_hdf5_bytes: int


def _load_base_protocol_dict(collimator_type: CollimatorKind) -> Dict[str, Any]:
    """Загружает базовый YAML-конфиг протокола NEMA 161Tb для выбранного типа коллиматора."""
    yaml_filename = (
        "tb161_nema_protocol.yaml"
        if collimator_type == "parametric"
        else "tb161_nema_protocol_direct.yaml"
    )
    yaml_path = Path(yaml_filename).resolve()
    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        loaded_data: Dict[str, Any] = yaml.safe_load(yaml_file)
    return loaded_data


def _prepare_benchmark_protocol_dict(
    base_dict: Dict[str, Any],
    output_dir: Path,
    pool_size: int,
    steps_count: int,
    time_per_step_str: str,
    particles_number: int,
    shared_file: bool,
) -> Dict[str, Any]:
    """Формирует изолированный словарь конфигурации для бенчмарка Orchestrator."""
    cfg = deepcopy(base_dict)
    cfg["pool_size"] = pool_size
    cfg["logging"] = {
        "output_dir": str(output_dir / "logs"),
        "level": "WARNING",
        "console": False,
        "per_task_files": True,
        "filename_pattern": "task_{task_id}.log",
    }
    cfg["protocol"] = {
        "type": "StepAndShoot",
        "steps": steps_count,
        "start_angle": "0.0 deg",
        "end_angle": "180.0 deg",
        "endpoint": False,
        "rotation_direction": "CW",
        "time_per_step": time_per_step_str,
        "initial_time": "0.0 s",
        "sequential_time": True,
    }
    if shared_file:
        hdf_pattern = str(output_dir / "shared_projection_data.hdf")
    else:
        hdf_pattern = str(output_dir / "proj_${gantry_angle deg}_deg.hdf")

    cfg["data_manager"] = {
        "filename": hdf_pattern,
        "buffer_capacity": 50_000,
        "floating_point_precision": "float32",
        "compression": "gzip",
        "compression_opts": 4,
        "handlers": [
            {
                "type": "HistoryAssemblerHandler",
                "sensitive_volumes": ["Detector_1", "Detector_2"],
                "save_initial_states": True,
            }
        ],
    }
    cfg["simulation_manager"] = {
        "particles_number": particles_number,
        "min_energy": "1.0 keV",
        "physics_list": "emstandard_opt4",
        "seed": 424242,
    }
    return cfg


def _sum_hdf5_bytes(directory: Path) -> int:
    """Подсчитывает суммарный объем созданных HDF5-файлов в директории."""
    total_bytes = 0
    for hdf_file in directory.glob("*.hdf"):
        if hdf_file.is_file():
            total_bytes += int(hdf_file.stat().st_size)
    return total_bytes


def run_orchestrator_scaling_benchmark(
    collimator_types: List[CollimatorKind],
    pool_sizes: List[int] | None = None,
    steps_count: int = 4,
    time_per_step_str: str = "0.0015 s",
    particles_number: int = 25_000,
) -> List[OrchestratorScalingRecord]:
    """
    Запускает серию тестов масштабирования Orchestrator по числу процессов (pool_size)
    и режиму файлового вывода (индивидуальные файлы ракурсов против единого общего HDF5).
    """
    if pool_sizes is None:
        pool_sizes = [1, 2, 4]

    records: List[OrchestratorScalingRecord] = []
    temp_root = Path(tempfile.mkdtemp(prefix="nmsim_orch_prof_"))

    try:
        for coll_type in collimator_types:
            base_dict = _load_base_protocol_dict(coll_type)
            seq_wall_time: float | None = None

            # 1. Масштабирование по числу процессов при раздельных файлах (per_task_file)
            for p_size in pool_sizes:
                run_dir = temp_root / f"{coll_type}_pool{p_size}_per_task"
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg_dict = _prepare_benchmark_protocol_dict(
                    base_dict=base_dict,
                    output_dir=run_dir,
                    pool_size=p_size,
                    steps_count=steps_count,
                    time_per_step_str=time_per_step_str,
                    particles_number=particles_number,
                    shared_file=False,
                )
                orchestrator = Orchestrator(cfg_dict)
                t0 = time.perf_counter()
                orchestrator.run()
                wall_elapsed = max(time.perf_counter() - t0, 1e-9)

                if seq_wall_time is None or p_size == 1:
                    seq_wall_time = wall_elapsed

                speedup = seq_wall_time / wall_elapsed
                efficiency = speedup / p_size
                hdf_bytes = _sum_hdf5_bytes(run_dir)

                records.append(
                    OrchestratorScalingRecord(
                        collimator_type=coll_type,
                        pool_size=p_size,
                        tasks_count=steps_count,
                        file_mode="per_task_file",
                        wall_time=wall_elapsed,
                        tasks_per_second=steps_count / wall_elapsed,
                        speedup_vs_seq=speedup,
                        parallel_efficiency=efficiency,
                        total_hdf5_bytes=hdf_bytes,
                    )
                )
                shutil.rmtree(run_dir, ignore_errors=True)

            # 2. Сравнение с записью в общий файл (shared_single_file) при максимальном pool_size
            max_pool = max(pool_sizes)
            if max_pool > 1:
                shared_dir = temp_root / f"{coll_type}_pool{max_pool}_shared"
                shared_dir.mkdir(parents=True, exist_ok=True)
                shared_cfg = _prepare_benchmark_protocol_dict(
                    base_dict=base_dict,
                    output_dir=shared_dir,
                    pool_size=max_pool,
                    steps_count=steps_count,
                    time_per_step_str=time_per_step_str,
                    particles_number=particles_number,
                    shared_file=True,
                )
                orchestrator_shared = Orchestrator(shared_cfg)
                t0 = time.perf_counter()
                orchestrator_shared.run()
                shared_wall = max(time.perf_counter() - t0, 1e-9)

                base_seq = seq_wall_time if seq_wall_time is not None else shared_wall
                speedup_sh = base_seq / shared_wall
                efficiency_sh = speedup_sh / max_pool
                hdf_bytes_sh = _sum_hdf5_bytes(shared_dir)

                records.append(
                    OrchestratorScalingRecord(
                        collimator_type=coll_type,
                        pool_size=max_pool,
                        tasks_count=steps_count,
                        file_mode="shared_single_file",
                        wall_time=shared_wall,
                        tasks_per_second=steps_count / shared_wall,
                        speedup_vs_seq=speedup_sh,
                        parallel_efficiency=efficiency_sh,
                        total_hdf5_bytes=hdf_bytes_sh,
                    )
                )
                shutil.rmtree(shared_dir, ignore_errors=True)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return records


__all__ = [
    "OrchestratorScalingRecord",
    "run_orchestrator_scaling_benchmark",
]
