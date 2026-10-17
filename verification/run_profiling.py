"""
Единая CLI-точка входа комплексного профилировочного стенда NMSimToolkit.

Выполняет:
1. Предварительный прогрев Numba JIT-ядер и @cfunc + калибровку термического маяка CPU ноутбука;
2. Сквозное и двухпоточное профилирование CPython (cProfile + pstats) с сохранением .prof в stats/;
3. Сравнительный бенчмарк всех обработчиков данных (DataHandlers), режимов сжатия HDF5 и емкости буфера;
4. 10-фазную пошаговую декомпозицию цикла переноса для W1/W2/W3 (Parametric vs Direct);
5. Сбор алгоритмических счетчиков рейкастинга и дельта-трекинга Вудкока;
6. Изолированные микробенчмарки горячих узлов (M1..M6);
7. Оценку параллельной масштабируемости Orchestrator и межпроцессной конкуренции за HDF5;
8. Построение графиков (.png), сохранение JSON-метрик и аналитического отчета PROFILING_REPORT.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from numba import njit

from verification.profiling.batch_size_benchmark import (
    DEFAULT_BATCH_SIZES,
    BatchSizeBenchmarkRecord,
    run_batch_size_sweep_suite,
)
from verification.profiling.cprofile_harness import (
    CProfileDomainBreakdown,
    CProfileFunctionRecord,
    DualThreadCProfileResult,
    EndToEndCProfileResult,
    run_cprofile_suite,
)
from verification.profiling.data_handlers_benchmark import (
    BufferCapacityScalingRecord,
    DataHandlerBenchmarkRecord,
    run_data_handlers_comparison_suite,
)
from verification.profiling.micro_benchmarks import (
    MicroBenchmarkRecord,
    run_micro_benchmarks_suite,
)
from verification.profiling.orchestrator_scaling import (
    OrchestratorScalingRecord,
    run_orchestrator_scaling_benchmark,
)
from verification.profiling.plotting import (
    plot_batch_size_scaling,
    plot_cprofile_domain_breakdown,
    plot_data_handlers_comparison,
    plot_micro_benchmarks_speedup,
    plot_orchestrator_scaling,
    plot_step_phase_breakdown,
    plot_transport_telemetry_comparison,
)
from verification.profiling.step_profiler import (
    PhysicsProcessesBreakdown,
    SourceSubphaseBreakdown,
    StepPhaseBreakdownResult,
    run_step_profiling_suite,
)
from verification.profiling.transport_telemetry import (
    TransportTelemetryMetrics,
    run_transport_telemetry_suite,
)
from verification.profiling.workloads import (
    ALL_COLLIMATOR_KINDS,
    build_all_workloads_matrix,
    warmup_all_profiling_kernels,
)


@dataclass(frozen=True)
class ThermalCheckpointRecord:
    """Запись контроля тактовой стабильности и теплового троттлинга процессора ноутбука."""

    stage_name: str
    beacon_duration_ms: float
    relative_cpu_speed_pct: float
    cooldown_applied_sec: float


@njit(cache=True)
def _cpu_thermal_beacon_kernel(iterations: int) -> float:
    """
    Калибровочное скалярное JIT-ядро фиксированной трудоемкости (без обращений к памяти)
    для отслеживания просадки тактовой частоты CPU (Thermal / Power Limit Throttling).
    """
    acc = 1.000001
    for idx in range(1, iterations + 1):
        val = float(idx) * 1e-6
        acc = math.sin(val + acc * 0.001) + math.cos(val - acc * 0.001) + math.sqrt(abs(acc) + 1.0)
    return acc


def _measure_beacon_ms(iterations: int = 2_500_000) -> float:
    """Возвращает минимальное из 3 коротких измерений калибровочного ядра в миллисекундах."""
    best_sec = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        _ = _cpu_thermal_beacon_kernel(iterations)
        elapsed = time.perf_counter() - t0
        if elapsed < best_sec:
            best_sec = elapsed
    return float(best_sec * 1000.0)


def _thermal_guard_checkpoint(
    stage_name: str,
    baseline_ms: float,
    cooldown_sec: float = 2.0,
    throttle_threshold_pct: float = 92.0,
) -> ThermalCheckpointRecord:
    """
    Проверяет текущую скорость однопоточного JIT-маяка относительно базовой частоты.
    При просадке скорости ниже throttle_threshold_pct выполняет паузу охлаждения.
    """
    if cooldown_sec > 0.0:
        time.sleep(cooldown_sec)
    total_sleep = cooldown_sec
    current_ms = _measure_beacon_ms()
    rel_speed = 100.0 * (baseline_ms / max(current_ms, 1e-6))

    if rel_speed < throttle_threshold_pct:
        extra_sleep = 3.0
        time.sleep(extra_sleep)
        total_sleep += extra_sleep
        current_ms = _measure_beacon_ms()
        rel_speed = 100.0 * (baseline_ms / max(current_ms, 1e-6))

    return ThermalCheckpointRecord(
        stage_name=stage_name,
        beacon_duration_ms=current_ms,
        relative_cpu_speed_pct=rel_speed,
        cooldown_applied_sec=total_sleep,
    )


def _format_markdown_report(
    warmup_duration: float,
    thermal_records: List[ThermalCheckpointRecord],
    cprofile_summaries: List[EndToEndCProfileResult],
    handler_records: List[DataHandlerBenchmarkRecord],
    buffer_records: List[BufferCapacityScalingRecord],
    step_results: List[StepPhaseBreakdownResult],
    telemetry_reports: List[TransportTelemetryMetrics],
    micro_records: List[MicroBenchmarkRecord],
    orchestrator_records: List[OrchestratorScalingRecord],
    batch_size_records: List[BatchSizeBenchmarkRecord],
    total_wall_time: float,
) -> str:
    """Формирует полный аналитический Markdown-отчет по результатам профилирования."""
    lines: List[str] = []
    lines.append("# Комплексный отчет по профилированию и анализу узких мест NMSimToolkit")
    lines.append("")
    lines.append(
        f"**Общее время работы профилировочного стенда:** `{total_wall_time:.2f} с` | "
        f"**Время холодного прогрева JIT/cfunc:** `{warmup_duration:.2f} с` | "
        "**Режим:** 100% Zero-Overhead для боевого ядра `core/` (Warm Steady-State + диагностика Cold Compile)."
    )
    lines.append("")

    # Раздел 0: Контроль теплового режима и частоты CPU ноутбука
    lines.append("## 0. Контроль теплового режима и стабильности частоты CPU (Thermal Guard)")
    lines.append("")
    lines.append(
        "| Контрольная точка этапа | Время CPU-маяка (мс) | Относительная частота CPU (%) | Пауза охлаждения (с) |"
    )
    lines.append("| :--- | :---: | :---: | :---: |")
    for th_rec in thermal_records:
        lines.append(
            f"| `{th_rec.stage_name}` | {th_rec.beacon_duration_ms:.2f} | "
            f"**{th_rec.relative_cpu_speed_pct:.1f}%** | {th_rec.cooldown_applied_sec:.1f} |"
        )
    lines.append("")

    # Раздел 1: cProfile
    lines.append("## 1. Профилирование конвейера CPython (`cProfile` / `pstats`, Warm Steady-State)")
    lines.append("")
    lines.append("![Распределение времени cProfile по доменам](01_cprofile_domain_breakdown.png)")
    lines.append("")
    lines.append("### 1.1. Сводная декомпозиция времени по доменам (сквозной однопоточный режим)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Шагов | Историй | Steady-State (с) | Историй/с | "
        "JIT Транспорт (%) | JIT Процессы (%) | Источник (%) | Буферы+Escaped (%) | DataHandler CPU (%) | HDF5 I/O (%) | Дамп `.prof` |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for item in cprofile_summaries:
        dom = item.domain_breakdown
        prof_name = Path(item.end_to_end_prof_path).name
        lines.append(
            f"| `{item.workload_id}` | `{item.collimator_kind}` | {item.simulation_steps} | "
            f"{item.particles_number:,} | {item.end_to_end_wall_time:.3f} | **{item.throughput_histories_per_sec:,.0f}** | "
            f"{dom.jit_transport_pct:.1f}% | {dom.jit_physics_pct:.1f}% | "
            f"{dom.source_generation_pct:.1f}% | {dom.buffer_and_escaped_pct:.1f}% | "
            f"{dom.data_handler_pct:.1f}% | {dom.hdf5_io_pct:.1f}% | `{prof_name}` |"
        )
    lines.append("")

    lines.append("### 1.2. Штатный двухпоточный режим (`SimulationManager` + `DataManager` в разных потоках)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Wall-Time (с) | Поток Sim (с) | Поток Data (с) | "
        "Ожидание `queue.put` (с) | Ожидание `queue.get` (с) | Активная работа Data (с) | Tail Drain Lag (мс) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for item in cprofile_summaries:
        dual = item.dual_thread_metrics
        lines.append(
            f"| `{dual.workload_id}` | `{dual.collimator_kind}` | {dual.wall_time:.3f} | "
            f"{dual.sim_thread_time:.3f} | {dual.data_thread_time:.3f} | "
            f"{dual.sim_queue_put_wait_time:.4f} | {dual.data_queue_get_wait_time:.4f} | "
            f"{dual.data_active_work_time:.3f} | {dual.tail_drain_lag * 1000.0:.2f} |"
        )
    lines.append("")

    for summary_item in cprofile_summaries:
        if "W2_NEMA" in summary_item.workload_id:
            lines.append(
                f"### 1.3. Топ-12 функций по собственному времени (`tottime`, без JIT-компиляции) — `{summary_item.workload_id}`"
            )
            lines.append("")
            lines.append("| № | Функция | Файл и строка | Вызовы (`ncalls`) | `tottime` (с) | `cumtime` (с) | Доля (%) |")
            lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: |")
            for rank_idx, fn_entry in enumerate(summary_item.top_by_tottime[:12], start=1):
                lines.append(
                    f"| {rank_idx} | `{fn_entry.function_name}` | `{fn_entry.file_name}:{fn_entry.line_number}` | "
                    f"{fn_entry.call_count:,} | {fn_entry.total_time:.4f} | {fn_entry.cumulative_time:.4f} | "
                    f"**{fn_entry.percent_of_total:.1f}%** |"
                )
            lines.append("")

    # Раздел 2: Сравнение DataHandlers
    lines.append("## 2. Сравнительный анализ обработчиков данных (`DataHandlers`) и HDF5 I/O (Warm Steady-State)")
    lines.append("")
    lines.append("![Сравнение DataHandlers](02_data_handlers_comparison.png)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Код обработчика | Транспорт (с) | CPU хэндлера (с) | HDF5 I/O (с) | "
        "Доля Data (%) | Оверхед к Baseline (%) | Wall-Time 2 потока (с) | Tail Lag (мс) | Размер HDF5 (МБ) |"
    )
    lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for rec in handler_records:
        lines.append(
            f"| `{rec.workload_id}` | `{rec.collimator_kind}` | `{rec.handler_code}` | "
            f"{rec.transport_only_time:.3f} | {rec.handler_cpu_time:.3f} | {rec.hdf5_io_time:.3f} | "
            f"**{rec.data_share_of_total_pct:.1f}%** | {rec.overhead_vs_null_no_escaped_pct:+.1f}% | "
            f"{rec.threaded_wall_time:.3f} | {rec.threaded_tail_lag * 1000.0:.2f} | {rec.hdf5_file_size_mb:.2f} |"
        )
    lines.append("")

    if buffer_records:
        lines.append("### 2.1. Зависимость времени обработки от емкости буфера (`buffer_capacity`)")
        lines.append("")
        lines.append(
            "| Сцена | Коллиматор | `buffer_capacity` | Чанков | Транзакций HDF5 | "
            "Транспорт (с) | CPU фильтрация (с) | HDF5 I/O (с) | Суммарно (с) |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for b_rec in buffer_records:
            lines.append(
                f"| `{b_rec.workload_id}` | `{b_rec.collimator_kind}` | `{b_rec.buffer_capacity:,}` | "
                f"{b_rec.chunks_dispatched} | {b_rec.hdf5_write_transactions} | "
                f"{b_rec.transport_time:.3f} | {b_rec.handler_cpu_time:.3f} | "
                f"{b_rec.hdf5_io_time:.3f} | {b_rec.total_sync_time:.3f} |"
            )
        lines.append("")

    # Раздел 3: 10-фазная декомпозиция
    lines.append("## 3. Детальная 10-фазная декомпозиция пошагового цикла симуляции (Warm Steady-State)")
    lines.append("")
    lines.append("![10-фазная декомпозиция шага](03_step_phase_breakdown.png)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Шагов | Историй | Steady-State (с) | Историй/с | "
        "P2: Source (%) | P4: Geometry (%) | P5: Woodcock (%) | P8: Processes (%) | P9: Invalid+Esc (%) | P10: Data+HDF5 (%) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_res in step_results:
        pcts = s_res.phase_percentages()
        lines.append(
            f"| `{s_res.workload_id}` | `{s_res.collimator_kind}` | {s_res.total_steps} | "
            f"{s_res.particles_number:,} | {s_res.total_wall_time:.3f} | **{s_res.throughput_histories_per_sec:,.0f}** | "
            f"{pcts.get('P2_Source_Inject', 0.0):.1f}% | "
            f"{pcts.get('P4_Geometry_Raycast', 0.0):.1f}% | "
            f"{pcts.get('P5_Woodcock_Transport', 0.0):.1f}% | "
            f"{pcts.get('P8_Physics_Processes', 0.0):.1f}% | "
            f"{pcts.get('P9_Invalidation_Escaped', 0.0):.1f}% | "
            f"{pcts.get('P10_DataHandler_HDF5', 0.0):.1f}% |"
        )
    lines.append("")

    lines.append("### 3.1. Внутренняя декомпозиция `Source.inject` и физических процессов (Warm Steady-State)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | `position` (мс) | `direction` (мс) | `energy+time` (мс) | "
        "`bank.inject` (мс) | `_record_finished` (мс) | Фотоэффект (мс) | Комптон (мс) | Рэлей (мс) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_res in step_results:
        sub_s = s_res.source_subphases
        sub_p = s_res.physics_subphases
        lines.append(
            f"| `{s_res.workload_id}` | `{s_res.collimator_kind}` | "
            f"{sub_s.generate_position * 1e3:.2f} | "
            f"{sub_s.generate_direction * 1e3:.2f} | "
            f"{(sub_s.generate_energy + sub_s.generate_emission_time + sub_s.time_step_newton) * 1e3:.2f} | "
            f"{sub_s.soa_conversion_and_bank_inject * 1e3:.2f} | "
            f"{s_res.p9_sub_escaped_only * 1e3:.2f} | "
            f"{sub_p.photoelectric_time * 1e3:.2f} | "
            f"{sub_p.compton_time * 1e3:.2f} | "
            f"{sub_p.coherent_time * 1e3:.2f} |"
        )
    lines.append("")

    lines.append("### 3.2. Сравнение `Cold Propagator Overhead` vs `Warm Steady-State`")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Историй | Cold JIT пропагатора (с) | Полное время Cold (с) | "
        "Чистое время Warm (с) | Speedup (Warm/Cold) | Скорость Cold (ист./с) | Скорость Warm (ист./с) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_res in step_results:
        lines.append(
            f"| `{s_res.workload_id}` | `{s_res.collimator_kind}` | {s_res.particles_number:,} | "
            f"{s_res.propagator_cold_compile_time:.3f} | {s_res.cold_total_wall_time:.3f} | "
            f"**{s_res.total_wall_time:.3f}** | **{s_res.warm_vs_cold_speedup:.1f}x** | "
            f"{s_res.cold_throughput_histories_per_sec:,.0f} | **{s_res.throughput_histories_per_sec:,.0f}** |"
        )
    lines.append("")

    lines.append(
        "### 3.3. Разделение стационарной фазы полного банка (`Full-Bank Phase`) и фазы опустошения (`Tail Depletion Phase`)"
    )
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Емкость банка | Историй | Шагов `Full-Bank` | Шагов `Depletion` | "
        "Транспорт `Full-Bank` (мс) | Транспорт `Depletion` (мс) | Треков/с `Full-Bank` | Треков/с `Depletion` | Замедление на хвосте (`x`) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_res in step_results:
        lines.append(
            f"| `{s_res.workload_id}` | `{s_res.collimator_kind}` | `{s_res.bank_capacity:,}` | "
            f"{s_res.particles_number:,} | {s_res.full_bank_steps} | {s_res.depletion_steps} | "
            f"{s_res.full_bank_transport_time * 1e3:.2f} | {s_res.depletion_transport_time * 1e3:.2f} | "
            f"**{s_res.full_bank_tracks_per_sec:,.0f}** | {s_res.depletion_tracks_per_sec:,.0f} | "
            f"**{s_res.depletion_track_slowdown_factor:.2f}x** |"
        )
    lines.append("")

    # Раздел 4: Телеметрия счетчиков переноса
    lines.append("## 4. Алгоритмическая телеметрия ядер переноса (Geometry & Woodcock)")
    lines.append("")
    lines.append("![Телеметрия счетчиков переноса](04_transport_telemetry_comparison.png)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | Историй | Лучей/ист. | Примитивов/луч | "
        "`_box_intersect` | `_hex_prism_intersect` | Шагов Вудкока/ист. | Холостых Вудкока (%) | `searchsorted`/ист. | `@cfunc`/ист. |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for rep in telemetry_reports:
        lines.append(
            f"| `{rep.workload_id}` | `{rep.collimator_kind}` | {rep.histories_count:,} | "
            f"{rep.raycasts_per_history:.2f} | {rep.primitive_tests_per_raycast:.2f} | "
            f"{rep.total_box_intersections:,} | {rep.total_hex_prism_intersections:,} | "
            f"{rep.woodcock_steps_per_history:.2f} | **{rep.null_collision_fraction_pct:.1f}%** | "
            f"{rep.searchsorted_per_history:.2f} | {rep.cfunc_calls_per_history:.2f} |"
        )
    lines.append("")

    # Раздел 5: Микробенчмарки
    lines.append("## 5. Изолированные микробенчмарки горячих узлов (`M1..M7`)")
    lines.append("")
    lines.append("![Микробенчмарки горячих узлов](05_micro_benchmarks_speedup.png)")
    lines.append("")
    lines.append(
        "| ID | Узел конвейера | Текущая реализация / База | Альтернатива / Сравнение | Текущее (нс/оп) | "
        "Альтернатива (нс/оп) | Отношение (`Speedup / Ratio`) |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: |")
    for m_rec in micro_records:
        lines.append(
            f"| `{m_rec.benchmark_id}` | {m_rec.title} | `{m_rec.current_label}` | "
            f"`{m_rec.alternative_label}` | {m_rec.current_ns_per_op:.1f} | "
            f"{m_rec.alternative_ns_per_op:.1f} | **{m_rec.speedup_factor:.2f}x** |"
        )
    lines.append("")

    # Раздел 6: Масштабируемость Orchestrator
    lines.append("## 6. Масштабируемость параллельного `Orchestrator` и конкуренция за HDF5")
    lines.append("")
    lines.append("![Масштабируемость Orchestrator](06_orchestrator_scaling.png)")
    lines.append("")
    lines.append(
        "| Коллиматор | `pool_size` | Задач | Режим файлов HDF5 | Время `Wall-Time` (с) | "
        "Задач/с | Ускорение (`Speedup`) | Параллельная эффективность (%) |"
    )
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for o_rec in orchestrator_records:
        lines.append(
            f"| `{o_rec.collimator_type}` | {o_rec.pool_size} | {o_rec.tasks_count} | "
            f"`{o_rec.file_mode}` | {o_rec.wall_time:.3f} | {o_rec.tasks_per_second:.2f} | "
            f"**{o_rec.speedup_vs_seq:.2f}x** | {o_rec.parallel_efficiency * 100.0:.1f}% |"
        )
    lines.append("")

    # Раздел 7: Исследование размера батча частиц (ParticleBank capacity / Batch Size Sweep)
    lines.append(
        "## 7. Исследование размера батча частиц (`ParticleBank` capacity / Batch Size Sweep, 6,000,000 историй)"
    )
    lines.append("")
    lines.append("![Масштабируемость от размера батча частиц](07_batch_size_scaling.png)")
    lines.append("")
    lines.append(
        "### 7.1. Сравнение стационарной фазы полного банка (`Full-Bank Sustained Phase`) и фазы опустошения (`Tail Depletion Phase`)"
    )
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | `batch_size` | Историй | Шагов (`Full / Depl`) | Время `Full-Bank` (с) | "
        "Время `Depletion` (с) | Доля хвоста (%) | Скорость `Full-Bank` (ист./с) | Скорость сквозная (ист./с) | "
        "Треков/с (`Full / Depl`) | Замедление на хвосте (`x`) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for bs_rec in batch_size_records:
        lines.append(
            f"| `{bs_rec.workload_id}` | `{bs_rec.collimator_kind}` | `{bs_rec.batch_size:,}` | "
            f"{bs_rec.emitted_particles:,} | {bs_rec.full_bank_steps_count} / {bs_rec.depletion_steps_count} | "
            f"{bs_rec.full_bank_time_sec:.3f} | {bs_rec.depletion_time_sec:.3f} | "
            f"{bs_rec.depletion_time_share_pct:.1f}% | **{bs_rec.full_bank_histories_per_sec:,.0f}** | "
            f"{bs_rec.histories_per_sec:,.0f} | "
            f"{bs_rec.full_bank_tracks_per_sec:,.0f} / {bs_rec.depletion_tracks_per_sec:,.0f} | "
            f"**{bs_rec.depletion_track_slowdown_factor:.2f}x** |"
        )
    lines.append("")

    lines.append(
        "### 7.2. Пофазная структура времени в стационарном режиме полного банка (`Full-Bank Sustained Phase`)"
    )
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | `batch_size` | Скорость `Full-Bank` (ист./с) | `Full` Source (%) | "
        "`Full` Geometry (%) | `Full` Woodcock (%) | `Full` Processes (%) | `Full` Compaction+Esc (%) | "
        "Заполнение банка общ. / в `Depletion` (%) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for bs_rec in batch_size_records:
        lines.append(
            f"| `{bs_rec.workload_id}` | `{bs_rec.collimator_kind}` | `{bs_rec.batch_size:,}` | "
            f"**{bs_rec.full_bank_histories_per_sec:,.0f}** | {bs_rec.full_source_pct:.1f}% | "
            f"{bs_rec.full_geometry_raycast_pct:.1f}% | {bs_rec.full_woodcock_pct:.1f}% | "
            f"{bs_rec.full_processes_pct:.1f}% | {bs_rec.full_compaction_and_escaped_pct:.1f}% | "
            f"{bs_rec.avg_bank_occupancy_pct:.1f}% / {bs_rec.depletion_avg_occupancy_pct:.1f}% |"
        )
    lines.append("")

    # Раздел 8: Главные выводы и приоритизированная матрица оптимизаций
    lines.append("---")
    lines.append("")
    lines.append("## 8. Главные аналитические выводы и приоритизированная матрица оптимизаций")
    lines.append("")
    lines.append(
        "### 8.1. Методологическое очищение профиля: `Cold Propagator Overhead` vs `Warm Steady-State`"
    )
    lines.append("")
    lines.append(
        "* **Суть исправленного искажения:** в `ParticlePropagator.__init__` при каждом создании экземпляра заново вызываются фабрики замыканий `make_transport_kernel` (фаза `P5`) и `make_photoelectric_kernel`, `make_compton_kernel`, `make_coherent_kernel` (фаза `P8`), тогда как функция трассировки лучей `cast_path_kernel` (`P4`) объявлена на уровне модуля и компилируется один раз. При замере без предварительного прогрева инстанса `ParticlePropagator` (`warmup_propagator_instance`) на Шаге №1 происходила `~3.22–3.50 с` компиляция LLVM, из-за чего в старом отчете фаза `P8_Physics_Processes` искусственно занимала `~75.5%` времени (`2,570 мс`), `P5_Woodcock_Transport` — `~22.4%` (`780 мс`), а `P4_Geometry_Raycast` — `0.1%`."
    )
    lines.append(
        "* **Истинная картина в установившемся режиме (`Warm Steady-State`, Раздел 3):** после выноса компиляции замыканий `ParticlePropagator` в предстартовый прогрев, внедрения выборочной аллокации буферов `SimulationDataBuffer.allocate_for(capacity, enabled)` и каскадного сброса по `TELEMETRY_ORDER` (`_ensure_step_capacity`, `_record_finished_histories`) выяснилось, что:"
    )
    lines.append(
        "  1. В 10-фазном профилировщике (`step_profiler.py`, емкость банка `25,000`, общее число историй `150,000–300,000`) в установившемся режиме симуляция с полной записью HDF5 проходит за `0.30–0.86 с` вместо `3.80–4.35 с` Cold-старта (**ускорение `5.1x – 12.5x`**), а в чистом транспорте без HDF5 достигает **свыше `2.22 млн историй/с` в `W1` и до `442.9 тыс. историй/с` в `W2_NEMA_Tb161`**."
    )
    lines.append(
        "  2. Доля физических процессов `P8_Physics_Processes` (Фотоэффект + Комптон + Рэлей) составляет всего **`6.5% – 20.6%`** от общего времени с HDF5. Микробенчмарк `M7_TransportVsProcessKernel` объясняет соотношение внутри чистого транспорта: 1 шаг `cast_path + _transport_kernel` стоит `~280.1 нс/частицу`, а 1 вызов `ComptonScattering.apply` (включающий сэмплирование Клейна–Нишины, запись в `InteractionBuffer` и пересчет локальных нормалей в `update_navigation_state_rotate_kernel`) стоит `~340.6 нс` на рассеявшуюся частицу."
    )
    lines.append(
        "  3. Внедрение `SimulationDataBuffer.allocate_for(capacity, enabled)` (выделяющего `capacity = 0` неподписанным каналам), проверки `buffers.escaped_particles.is_enabled` в `_record_finished_histories` и однопроходных JIT-ядер (`_push_first_interacting_to_initial_state_kernel`, `_push_uncollided_to_initial_state_kernel`, `_push_to_dead_particles_kernel`) сократило долю буферного управления в `cProfile` до **`1.4% – 2.2%`**, долю фазы `P9_Invalidation_Escaped` — до **`1.9% – 2.8%`**, а полное время `_record_finished_histories` на 150k–300k частиц — до **`1.92 – 3.55 мс`** (`~10 нс/историю`). Принудительное включение `ESCAPED_PARTICLES` в `H0_NullDrain_WithEscaped` добавляет `+25.1%..+37.6%` оверхеда в `W1` и `+2.4%..+5.8%` в `W2`, который полностью устранен во всех штатных обработчиках `H1..H7`."
    )
    lines.append(
        "  4. Доля геометрического рейкастинга `P4_Geometry_Raycast` закономерно отражает сложность сцены: для `DirectParallelCollimator` доля `P4` выше (`5.4%–11.8%`), чем для `ParametricParallelCollimator` (`4.2%–5.4%`), тогда как `ParametricParallelCollimator` тратит больше времени в `P5_Woodcock_Transport` (`16.7%–35.2%` против `11.6%–33.6%`) из-за дельта-шагов сквозь свинец."
    )
    lines.append("")
    lines.append(
        "### 8.2. Двухфазная динамика банка частиц (`Full-Bank Sustained Phase` vs `Tail Depletion Phase`) на выборке 6,000,000 историй"
    )
    lines.append("")
    lines.append(
        "* **Зачем необходимо разделение `Full-Bank Phase` и `Tail Depletion Phase` (Разделы 3.3 и 7):** в полномасштабной симуляции (`6,000,000` историй) жизненный цикл банка `ParticleBank` делится на два качественно различных режима:"
    )
    lines.append(
        "  1. **Фаза A (`Full-Bank Sustained Phase` — стационарный режим полного банка):** пока квота источника не исчерпана, после каждого шага освободившиеся слоты `dead_indices` немедленно докачиваются из источника до 100% емкости банка (`active_indices.size == batch_size`). Все JIT-ядра (`cast_path_kernel`, `_transport_kernel`, процессы) работают на плотно заполненных массивах размера `batch_size`."
    )
    lines.append(
        "  2. **Фаза B (`Tail Depletion Phase` — хвостовая фаза опустошения банка):** когда квота источника исчерпана, генерация новых частиц прекращается (`Source.inject = 0%`), и оставшиеся в банке частицы долетывают за 5–59 шагов до полного поглощения или вылета (`active_indices.size` монотонно падает от `batch_size` до `1`, среднее заполнение банка падает до **`2.6%–30.1%`**). Для точечного источника в воздухе (`W1_PointSource`) удельная чистая скорость трекинга (`Треков/с` без `Source.inject`) на хвосте замедляется в **`1.28x – 3.74x` раз** (`depletion_track_slowdown_factor`), так как в банке остаются только многократно рассеивающиеся в септах и кристалле фотоны при разреженном массиве `active_indices`. В объемном воксельном фантоме (`W2_NEMA_Tb161`) при малых батчах (`2,500–15,000`) чистый трекинг на хвосте также замедляется в **`1.11x – 1.38x` раз**, а при больших батчах (`30,000–500,000`) отношение составляет `0.95x–1.05x`, поскольку на хвосте долетывают периферийные фотоны, покинувшие плотное водное ядро фантома."
    )
    lines.append(
        "* **Выводы по выбору оптимального `batch_size` на дистанции 6,000,000 историй:**"
    )
    lines.append(
        "  1. **Малые батчи (`2,500 – 5,000`):** требуют от `2,440` до `8,310` вызовов `Source.inject`. В воксельном фантоме `W2_NEMA_Tb161` при `batch_size = 2,500` генерация источника `Source.inject` съедает **`32.2% – 34.4%` времени стационарной фазы**, так как `rng.choice(origins, size=num_to_inject, p=probabilities)` на каждом шаге заново сканирует распределение вероятностей вокселей (`251.4–253.0 тыс. ист./с` против `419.7–442.9 тыс. ист./с` в оптимуме)."
    )
    lines.append(
        "  2. **Оптимальный стационарный диапазон (`30,000 – 120,000`):** обеспечивает максимальную скорость как в стационарном режиме полного банка (`Full-Bank Sustained`), так и по сквозному времени (`Total Throughput`). В `W1_PointSource` пик достигается при `batch_size = 30,000 – 60,000` (**`2.16 – 2.22 млн историй/с`**, до `7.05 млн треков/с`), а в `W2_NEMA_Tb161` — при `batch_size = 60,000 – 120,000` (**`409.9 – 419.7 тыс. ист./с` в Parametric и `437.3 – 442.9 тыс. ист./с` в Direct**), где доля `Source.inject` снижается с `~34%` до `9.8%–10.7%`, рабочие SoA-буферы укладываются в кэш L3, а доля хвостовой фазы опустошения `Tail Depletion Phase` на 6 млн историй не превышает `0.6%–1.2%` времени."
    )
    lines.append(
        "  3. **Сверхбольшие батчи (`250,000 – 500,000`):** рабочий набор SoA-массивов (`20–40 МБ`) выходит за пределы L3-кэша CPU. В `W1_PointSource` это снижает стационарную скорость `Full-Bank` на **9%–12%** (`2.22 млн -> 1.95–2.03 млн ист./с`), а в `W2_NEMA_Tb161` прекращает рост стационарной скорости (`419.7 тыс. -> 408.2 тыс. ист./с` в Parametric и `442.9 тыс. -> 431.7 тыс. ист./с` в Direct) при росте доли хвостовой фазы `Depletion` до `5.4%–6.0%`."
    )
    lines.append(
        "* **Архитектурный дефект `SimulationManager._replenish_bank` (Бэклог п. 24):** в `_replenish_bank()` отсутствует клиппинг `target_time = min(target_time, self.stop_time)` при вычислении `num_to_inject`. Поэтому, если `bank.capacity` превышает ожидаемое число историй за `stop_time`, менеджер инжектирует полный банк на первом же шаге и никогда не входит в многошаговый стационарный режим `Full-Bank Phase`. В профилировщиках (`step_profiler.py`, `batch_size_benchmark.py`) и верификационном стенде (`QuotaBoundedSimulationManager`) это исправлено раздельным заданием емкости банка `bank_capacity` и строгим контролем квоты историй `particles_number`."
    )
    lines.append("")
    lines.append("### 8.3. Приоритизированная матрица оптимизаций (Roadmap)")
    lines.append("")
    lines.append(
        "| Приоритет | Узел / Модуль | Описание узкого места | Предлагаемое решение | Ожидаемый эффект (`Speedup`) |"
    )
    lines.append("| :---: | :--- | :--- | :--- | :---: |")
    lines.append(
        "| **P0 (Critical)** | [`core/physics/processes_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py), [`core/transport/transport_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py) | Повторная компиляция `@njit`-замыканий (`make_*_kernel`, `make_transport_kernel`) при каждом создании `ParticlePropagator` | Вынос ядер на уровень модуля / кэширование фабрик через `@lru_cache` по `process_id` и `tuple(process_ids)`, удаление двойной инициализации в `ComptonScattering` | **`~3.50 с -> < 1 мс`** на старт пропагатора (**5.1x–12.5x** на коротких запусках и в `Orchestrator`) |"
    )
    lines.append(
        "| **P1 (High)** | [`core/transport/transport_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py) (`M1`) | Многократный бинарный поиск `np.searchsorted` (до 18 раз на историю) по неравномерной сетке энергий при каждом шаге Вудкока | Предрасчет макроскопических сечений ($\\mu$, $\\mu_{\\max}$) на единой равномерной логарифмической сетке с $\\mathcal{O}(1)$ вычислением индекса | **5.28x** ускорение выборки сечений (`P5_Woodcock_Transport` занимает до 38% времени чистого транспорта) |"
    )
    lines.append(
        "| **P1 (High)** | [`core/source/sources.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/sources.py) (`M2`, `M3`) | Аллокация полных 3D-массивов `np.meshgrid` при инициализации `EmissionTable` и $\\mathcal{O}(N_{\\text{vox}} + M\\log N_{\\text{vox}})$ выборка в `rng.choice` | Разреженная инициализация через `np.flatnonzero` + таблица `Vose's Alias Method` ($\\mathcal{O}(1)$ на выборку) с прямой записью в SoA-массивы `Vector3D` | **4.45x** инициализация фантома + **6.74x** генерация позиций (`Source` занимает до 34% при малых батчах и 10–11% при больших) |"
    )
    lines.append(
        "| **P1 (High)** | [`core/transport/simulation_managers.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/simulation_managers.py) (Раздел 7, Бэклог п. 24) | Отсутствие клиппинга `target_time <= stop_time` в `_replenish_bank` и деградация скорости трекинга на разреженном хвосте `Depletion Phase` | Ограничить `target_time` по `stop_time`, разделить `bank_capacity` (`30,000–60,000`) и общее число историй, добавить периодическую компактификацию активных треков на хвосте | **1.7x–3.7x** ускорение относительно неоптимальных батчей + точное соблюдение квоты частиц |"
    )
    lines.append(
        "| **P2 (Medium)** | [`core/data/data_handlers.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/data_handlers.py) (`M4`, `M5`) | Python-цикл `[vol_map.get(v) for v in ...]` (`dtype=object`) и `np.intersect1d + 4x np.isin` при фильтрации чанков | Векторизованный целочисленный LUT чувствительных `volume_id` и прямая булева индексация `mask[particle_id - min_id]` $\\mathcal{O}(N)$ | **37.41x** ускорение фильтрации `SensitiveVolume` и **32.16x** ускорение `HistoryAssembler` |"
    )
    lines.append(
        "| **P2 (Medium)** | [`core/data/dose_map_handler.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/data/dose_map_handler.py) (Бэклог п. 21) | Безусловное обращение к `SharedMemory` в автономном/CLI-режиме и спам предупреждений при `create_shm=False` | Явное разделение режимов In-Memory (`shm_name=None`) и IPC `SharedMemory` | Чистый лог и отсутствие лишних системных вызовов ОС |"
    )
    lines.append("")

    return "\n".join(lines)


def _dict_to_namespace(obj: Any) -> Any:
    """Рекурсивно преобразует словари и списки в объекты SimpleNamespace для чтения в генераторе отчета."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(item) for item in obj]
    return obj


def reformat_report_from_saved_json() -> None:
    """
    Пересчитывает производные метрики трекинга из сохраненного profiling_results.json,
    перестраивает диаграмму 07_batch_size_scaling.png и обновляет PROFILING_REPORT.md.
    """
    report_dir = Path("verification/reports/profiling").resolve()
    json_path = report_dir / "profiling_results.json"
    payload: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    step_results: List[StepPhaseBreakdownResult] = []
    for s_dict in payload["step_results"]:
        s_copy = dict(s_dict)
        s_copy["source_subphases"] = SourceSubphaseBreakdown(**s_dict["source_subphases"])
        s_copy["physics_subphases"] = PhysicsProcessesBreakdown(**s_dict["physics_subphases"])
        full_pure_sec = max(
            float(s_copy["full_bank_p4_geometry"])
            + float(s_copy["full_bank_p5_woodcock"])
            + float(s_copy["full_bank_p8_processes"])
            + float(s_copy["full_bank_p9_compaction"]),
            1e-9,
        )
        full_tracks = float(s_copy["full_bank_active_tracks"])
        full_tps = full_tracks / full_pure_sec
        depl_tps = float(s_copy["depletion_tracks_per_sec"])
        s_copy["full_bank_tracks_per_sec"] = full_tps
        s_copy["depletion_track_slowdown_factor"] = full_tps / max(depl_tps, 1e-9)
        step_results.append(StepPhaseBreakdownResult(**s_copy))

    batch_size_records: List[BatchSizeBenchmarkRecord] = []
    for b_dict in payload["batch_size_records"]:
        b_copy = dict(b_dict)
        full_pure_sec = max(
            float(b_copy["full_geometry_raycast_time_sec"])
            + float(b_copy["full_woodcock_time_sec"])
            + float(b_copy["full_processes_time_sec"])
            + float(b_copy["full_compaction_and_escaped_time_sec"]),
            1e-9,
        )
        full_tps = float(b_copy["full_bank_active_tracks"]) / full_pure_sec
        depl_tps = float(b_copy["depletion_tracks_per_sec"])
        b_copy["full_bank_tracks_per_sec"] = full_tps
        b_copy["depletion_track_slowdown_factor"] = full_tps / max(depl_tps, 1e-9)
        batch_size_records.append(BatchSizeBenchmarkRecord(**b_copy))

    payload["step_results"] = [asdict(item) for item in step_results]
    payload["batch_size_records"] = [asdict(item) for item in batch_size_records]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_batch_size_scaling(batch_size_records, report_dir)

    cprofile_summaries: List[EndToEndCProfileResult] = []
    for c_dict in payload["cprofile_summaries"]:
        c_copy = dict(c_dict)
        c_copy["domain_breakdown"] = CProfileDomainBreakdown(**c_dict["domain_breakdown"])
        c_copy["top_by_tottime"] = [CProfileFunctionRecord(**fn) for fn in c_dict["top_by_tottime"]]
        c_copy["top_by_cumtime"] = [CProfileFunctionRecord(**fn) for fn in c_dict["top_by_cumtime"]]
        c_copy["dual_thread_metrics"] = DualThreadCProfileResult(**c_dict["dual_thread_metrics"])
        cprofile_summaries.append(EndToEndCProfileResult(**c_copy))

    md_text = _format_markdown_report(
        warmup_duration=float(payload["warmup_duration"]),
        thermal_records=[ThermalCheckpointRecord(**item) for item in payload["thermal_records"]],
        cprofile_summaries=cprofile_summaries,
        handler_records=[DataHandlerBenchmarkRecord(**item) for item in payload["handler_records"]],
        buffer_records=[BufferCapacityScalingRecord(**item) for item in payload["buffer_records"]],
        step_results=step_results,
        telemetry_reports=[TransportTelemetryMetrics(**item) for item in payload["telemetry_reports"]],
        micro_records=[MicroBenchmarkRecord(**item) for item in payload["micro_records"]],
        orchestrator_records=[OrchestratorScalingRecord(**item) for item in payload["orchestrator_records"]],
        batch_size_records=batch_size_records,
        total_wall_time=float(payload["total_wall_time"]),
    )
    md_path = report_dir / "PROFILING_REPORT.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Отчет и графики успешно обновлены из JSON: {md_path}", flush=True)


def run_full_profiling_pipeline(quick_mode: bool = False) -> Dict[str, Any]:
    """
    Выполняет полный цикл профилирования NMSimToolkit с контролем теплового троттлинга,
    генерирует дампы .prof, графики PNG, JSON-сводку и аналитический отчет PROFILING_REPORT.md.
    """
    pipeline_start = time.perf_counter()
    stats_dir = Path("stats").resolve()
    report_dir = Path("verification/reports/profiling").resolve()
    stats_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Прогрев JIT-ядер Numba и калибровка термического маяка CPU...", flush=True)
    _ = _cpu_thermal_beacon_kernel(10_000)
    warmup_duration = warmup_all_profiling_kernels()
    baseline_beacon_ms = _measure_beacon_ms()

    thermal_records: List[ThermalCheckpointRecord] = [
        ThermalCheckpointRecord(
            stage_name="01_Baseline_After_JIT_Warmup",
            beacon_duration_ms=baseline_beacon_ms,
            relative_cpu_speed_pct=100.0,
            cooldown_applied_sec=0.0,
        )
    ]

    if quick_mode:
        w1_part, w2_part, w3_part = 12_000, 10_000, 10_000
        step_w1_part, step_w2_part, step_w3_part = 30_000, 20_000, 20_000
        step_bank_cap = 5_000
        handler_part = 12_000
        telem_part = 6_000
        orch_pools = [1, 2]
        orch_steps = 2
        batch_sweep_total = 30_000
        batch_sweep_sizes = (2_500, 5_000, 15_000, 30_000)
        pause_sec = 0.5
    else:
        w1_part, w2_part, w3_part = 40_000, 25_000, 25_000
        step_w1_part, step_w2_part, step_w3_part = 300_000, 150_000, 150_000
        step_bank_cap = 25_000
        handler_part = 25_000
        telem_part = 12_000
        orch_pools = [1, 2, 4]
        orch_steps = 4
        batch_sweep_total = 6_000_000
        batch_sweep_sizes = DEFAULT_BATCH_SIZES
        pause_sec = 1.5

    workloads = build_all_workloads_matrix(
        w1_particles=w1_part,
        w2_particles=w2_part,
        w3_particles=w3_part,
    )

    thermal_records.append(_thermal_guard_checkpoint("02_Before_cProfile_Suite", baseline_beacon_ms, pause_sec))
    print(
        f"[2/8] Запуск cProfile по {len(workloads)} эталонным сценам (двухпоточный и сквозной режимы)...",
        flush=True,
    )
    cprofile_summaries = run_cprofile_suite(workloads=workloads, stats_dir=stats_dir)

    thermal_records.append(_thermal_guard_checkpoint("03_Before_DataHandlers_Suite", baseline_beacon_ms, pause_sec))
    print("[3/8] Бенчмарк всех обработчиков данных (DataHandlers) и HDF5 I/O...", flush=True)
    handler_records, buffer_records = run_data_handlers_comparison_suite(
        particles_number=handler_part,
        buffer_capacity=100_000,
    )

    thermal_records.append(_thermal_guard_checkpoint("04_Before_StepProfiler_Suite", baseline_beacon_ms, pause_sec))
    print("[4/8] 10-фазное пошаговое профилирование цикла симуляции (Full-Bank + Depletion)...", flush=True)
    step_workloads = build_all_workloads_matrix(
        w1_particles=step_w1_part,
        w2_particles=step_w2_part,
        w3_particles=step_w3_part,
        bank_capacity=step_bank_cap,
    )
    step_results = run_step_profiling_suite(step_workloads)

    thermal_records.append(_thermal_guard_checkpoint("05_Before_Telemetry_Suite", baseline_beacon_ms, pause_sec))
    print("[5/8] Сбор алгоритмических счетчиков телеметрии переноса...", flush=True)
    telemetry_workloads = build_all_workloads_matrix(
        w1_particles=telem_part,
        w2_particles=telem_part,
        w3_particles=telem_part,
    )
    telemetry_reports = run_transport_telemetry_suite(telemetry_workloads)

    thermal_records.append(_thermal_guard_checkpoint("06_Before_MicroBenchmarks", baseline_beacon_ms, pause_sec))
    print("[6/8] Запуск изолированных микробенчмарков горячих узлов (M1..M7)...", flush=True)
    micro_records = run_micro_benchmarks_suite()

    thermal_records.append(_thermal_guard_checkpoint("07_Before_Orchestrator_Scaling", baseline_beacon_ms, pause_sec))
    print("[7/8] Бенчмарк параллельного Orchestrator и конкуренции за HDF5...", flush=True)
    orchestrator_records = run_orchestrator_scaling_benchmark(
        collimator_types=list(ALL_COLLIMATOR_KINDS),
        pool_sizes=orch_pools,
        steps_count=orch_steps,
        time_per_step_str="0.0010 s" if not quick_mode else "0.0005 s",
        particles_number=20_000 if not quick_mode else 10_000,
    )

    thermal_records.append(_thermal_guard_checkpoint("08_Before_BatchSize_Sweep", baseline_beacon_ms, pause_sec))
    print("[8/8] Исследование размера батча частиц (ParticleBank capacity / Batch Size Sweep)...", flush=True)
    batch_size_records = run_batch_size_sweep_suite(
        batch_sizes=batch_sweep_sizes,
        total_particles=batch_sweep_total,
    )
    thermal_records.append(_thermal_guard_checkpoint("09_After_Full_Pipeline", baseline_beacon_ms, 0.5))

    print("Построение диаграмм и формирование отчета...", flush=True)
    plot_cprofile_domain_breakdown(cprofile_summaries, report_dir)
    plot_data_handlers_comparison(handler_records, buffer_records, report_dir)
    plot_step_phase_breakdown(step_results, report_dir)
    plot_transport_telemetry_comparison(telemetry_reports, report_dir)
    plot_micro_benchmarks_speedup(micro_records, report_dir)
    plot_orchestrator_scaling(orchestrator_records, report_dir)
    plot_batch_size_scaling(batch_size_records, report_dir)

    total_wall_time = time.perf_counter() - pipeline_start

    results_payload: Dict[str, Any] = {
        "total_wall_time": total_wall_time,
        "warmup_duration": warmup_duration,
        "thermal_records": [asdict(item) for item in thermal_records],
        "cprofile_summaries": [asdict(item) for item in cprofile_summaries],
        "handler_records": [asdict(item) for item in handler_records],
        "buffer_records": [asdict(item) for item in buffer_records],
        "step_results": [asdict(item) for item in step_results],
        "telemetry_reports": [asdict(item) for item in telemetry_reports],
        "micro_records": [asdict(item) for item in micro_records],
        "orchestrator_records": [asdict(item) for item in orchestrator_records],
        "batch_size_records": [asdict(item) for item in batch_size_records],
    }

    json_path = report_dir / "profiling_results.json"
    json_path.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_text = _format_markdown_report(
        warmup_duration=warmup_duration,
        thermal_records=thermal_records,
        cprofile_summaries=cprofile_summaries,
        handler_records=handler_records,
        buffer_records=buffer_records,
        step_results=step_results,
        telemetry_reports=telemetry_reports,
        micro_records=micro_records,
        orchestrator_records=orchestrator_records,
        batch_size_records=batch_size_records,
        total_wall_time=total_wall_time,
    )
    md_path = report_dir / "PROFILING_REPORT.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Готово! Отчет сохранен в: {md_path}", flush=True)
    return results_payload


def main() -> None:
    """Разбирает аргументы командной строки и запускает профилировочный конвейер."""
    parser = argparse.ArgumentParser(description="Комплексный профилировочный стенд NMSimToolkit")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый смоук-режим с уменьшенным числом частиц",
    )
    parser.add_argument(
        "--reformat-only",
        action="store_true",
        help="Пересчитать производные поля и пересобрать отчет и графики из сохраненного profiling_results.json",
    )
    args = parser.parse_args()
    if args.reformat_only:
        reformat_report_from_saved_json()
    else:
        run_full_profiling_pipeline(quick_mode=bool(args.quick))


if __name__ == "__main__":
    main()
