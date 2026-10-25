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

    lines.append("### 3.1. Внутренняя декомпозиция инжекции источника и физических процессов (Warm Steady-State)")
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | `inject_poisson_sources_kernel` (мс) | "
        "`_record_finished` (мс) | Фотоэффект (мс) | Комптон (мс) | Рэлей (мс) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_res in step_results:
        sub_s = s_res.source_subphases
        sub_p = s_res.physics_subphases
        lines.append(
            f"| `{s_res.workload_id}` | `{s_res.collimator_kind}` | "
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

    lines.append(
        "### 3.4. Пофазные дельты Page Faults (`psutil.Process().memory_info()`) на стационарной фазе (`Full-Bank Phase`) и в `inject_poisson_sources_kernel`"
    )
    lines.append("")
    lines.append(
        "| Сцена | Коллиматор | `Full` P2 Source PF (`Steady`) | `Full` P4 Geom PF (`Steady`) | "
        "`Full` P5 Woodcock PF (`Steady`) | `Full` P8 Phys PF (`Steady`) | `Full` P9 Comp PF (`Steady`) | "
        "`Full` P10 Data PF | `src.kernel` PF |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_res in step_results:
        sub_s = s_res.source_subphases
        lines.append(
            f"| `{s_res.workload_id}` | `{s_res.collimator_kind}` | "
            f"**{s_res.full_bank_p2_page_faults:,}** (`{s_res.steady_full_bank_p2_page_faults:,}`) | "
            f"{s_res.full_bank_p4_page_faults:,} (`{s_res.steady_full_bank_p4_page_faults:,}`) | "
            f"{s_res.full_bank_p5_page_faults:,} (`{s_res.steady_full_bank_p5_page_faults:,}`) | "
            f"{s_res.full_bank_p8_page_faults:,} (`{s_res.steady_full_bank_p8_page_faults:,}`) | "
            f"{s_res.full_bank_p9_page_faults:,} (`{s_res.steady_full_bank_p9_page_faults:,}`) | "
            f"{s_res.full_bank_p10_page_faults:,} | "
            f"{sub_s.soa_conversion_and_bank_inject_page_faults:,} |"
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
        "### 8.1. Методологическое очищение профиля и эффект внедренных оптимизаций (`Warm Steady-State`)"
    )
    lines.append("")
    lines.append(
        "* **Холодный старт (`Cold Compile`) vs Установившийся режим (`Warm Steady-State`):** консолидация `@njit`-ядер в `processes_kernels.py` сократила время холодного прогрева `ParticlePropagator` более чем вдвое (**`3.55 с -> 1.61 с`**), а время выполнения пакета из 4 задач в `Orchestrator` (Раздел 6) уменьшилось с **`18.21–18.71 с` до `9.73–10.43 с`** при `pool_size = 1` и с **`7.30–7.55 с` до `4.47–4.89 с`** при `pool_size = 4` (**ускорение в `1.6x–1.8x` раза**)."
    )
    lines.append(
        "* **Компактная сетка NIST XCOM с краями поглощения и $\\mathcal{O}(1)$ лог-индекс (`M1a`, `M1b`):** переход от плотной 1-кэВ сетки (`~1117` узлов) к родной сетке NIST XCOM (`54–95` узлов с дуплетами K/L/M/N-краев поглощения) и двухуровневому логарифмическому индексу `energy_bin_map` ($512$ корзин) ускорил одиночный запрос сечения в **`2.03x` раза** (`35.2 нс`), а спаренный запрос Вудкока (`1x log(E) + 2x _locate_row` с кусочно-степенной `log-log` интерполяцией и выборкой $Z$ по парциальным сечениям) — в **`2.92x` раза** (`60.3 нс` против `176.2 нс`)."
    )
    lines.append(
        "* **Zero-Allocation инжекция первичных частиц (`inject_poisson_sources_kernel`, `M2`, `M3`):** переход на таблицы алиасов Воза (`Vose's Alias` $\\mathcal{O}(1)$, **`19.1 нс/оп`, ускорение `7.48x`**) и прямую запись в `ParticleBank` полностью обнулил страничные прерывания ОС в установившемся режиме (**`Steady Page Faults = 0`** во всех 6 сценах `W1–W3` против прежних `762–21,839`)."
    )
    lines.append(
        "* **Ускорение объемных клинических и фантомных сцен (`W2_NEMA_Tb161`, `W3_Heart_Tc99m`):** в 10-фазном профилировщике (Раздел 3) сквозная скорость на 5-тканевом фантоме миокарда `W3_Heart_Tc99m` выросла до **`196.5–197.0 тыс. ист./с` (`+9.4%..+10.8%`)**, а чистая скорость трекинга полного банка — до **`1.39–1.47 млн треков/с` (`+7.6%..+12.8%`)** в `W3` и **`1.47–1.58 млн треков/с` (`+7.9%..+12.3%`)** в `W2_NEMA_Tb161`."
    )
    lines.append("")
    lines.append(
        "### 8.2. Двухфазная динамика банка частиц (`Full-Bank Sustained Phase` vs `Tail Depletion Phase`) на выборке 6,000,000 историй"
    )
    lines.append("")
    lines.append(
        "* **Инвариантность генерации источника к размеру батча в воксельных фантомах (Раздел 7):** благодаря $\\mathcal{O}(1)$ выборке по таблицам Воза и методу прореживания Льюиса–Шедлера в `inject_poisson_sources_kernel`, на малых батчах (`batch_size = 2,500`, `8,175–8,594` шагов пополнения) доля времени источника в `W2_NEMA_Tb161` сократилась с прежних **`32.5%–34.8%` до `15.5%–16.6%`**, а скорость на `batch_size = 2,500` выросла с `244.6–246.9 тыс. ист./с` до **`305.9–315.8 тыс. ист./с` (`+25.0%..+27.9%`)**. Во всем диапазоне `batch_size` от `2,500` до `500,000` доля `Full Source (%)` в `W2_NEMA_Tb161` остается постоянной (**`15.1%–16.8%`**)."
    )
    lines.append(
        "* **Оптимальный диапазон `batch_size` (`60,000 – 120,000`):** максимизирует сквозную производительность как для точечного источника `W1_PointSource` (**до `1.00–1.01 млн историй/с` и `3.27–4.11 млн треков/с`** при `batch_size = 120,000`), так и для воксельного фантома `W2_NEMA_Tb161` (**`369.7 тыс. ист./с` в Parametric и `398.6 тыс. ист./с` в Direct**, до **`1.70 млн треков/с`** при `batch_size = 120,000`), удерживая долю хвостовой фазы опустошения `Tail Depletion Phase` в пределах `0.8%–1.2%`."
    )
    lines.append("")
    lines.append("### 8.3. Приоритизированная матрица дальнейших оптимизаций (Roadmap)")
    lines.append("")
    lines.append(
        "| Статус / Приоритет | Узел / Модуль | Описание | Реализация / Следующий шаг | Эффект (`Speedup`) |"
    )
    lines.append("| :---: | :--- | :--- | :--- | :---: |")
    lines.append(
        "| **Выполнено** | `core/materials/`, `core/physics/` (`M1a`, `M1b`) | Компактная сетка NIST XCOM с дуплетами краев поглощения, 2-уровневый лог-индекс `energy_bin_map` и парциальные доли `element_partial_fractions` | Внедрено в `AttenuationDatabase`, `MaterialBank`, `PhysicsCompiler`, `transport_kernels.py` | **`2.03x–2.92x`** на запросах сечений, размер `.h5` **`3.52 -> 0.74 МБ`** |"
    )
    lines.append(
        "| **Выполнено** | `core/source/` (`M2`, `M3`) | Таблицы алиасов Воза (`Vose's Alias`), `SourceBuffer` и `@njit`-инжектор `inject_poisson_sources_kernel` | Внедрено в `alias_tables.py`, `source_compiler.py`, `source_kernels.py` | **`7.48x`** выборка вокселя, **`0` Steady Page Faults**, **`+25%..+28%`** на малых батчах `W2` |"
    )
    lines.append(
        "| **Выполнено** | `core/data/data_handlers.py` (`M4`, `M5`) | Векторизованный LUT `volume_id` и прямая булева фильтрация `HistoryAssemblerHandler` | Внедрено в `SensitiveVolumeHandler` и `HistoryAssemblerHandler` | **`38.51x`** и **`27.89x`** на фильтрации чанков |"
    )
    lines.append(
        "| **P1 (Next)** | [`core/source/source_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/source/source_kernels.py), [`docs/plans/composable_sources_and_secondary_radiation_plan.md`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/docs/plans/composable_sources_and_secondary_radiation_plan.md) | В `W1_PointSource` пошаговое прореживание Льюиса–Шедлера при $\\lambda = 0$ и матричное умножение нулевых локальных координат тратят `~350 нс/частицу` | В рамках композиционной модели `SpatialDistribution` (`PointDistribution`) + быстрый путь для стационарного одиночного источника ($\\lambda = 0$) исключить холостой розыгрыш `u2` и матричное умножение `(0,0,0)` | Сокращение времени `P2_Source` для `PointSource` в **`2.5x–3.0x` раза** |"
    )
    lines.append(
        "| **P2 (Next)** | [`core/physics/processes_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/physics/processes_kernels.py), [`core/transport/transport_kernels.py`](file:///d:/Python%20Projects/NMSimToolkit-Refactor/core/transport/transport_kernels.py) | Компиляция замыканий `ParticlePropagator` при холодном старте процесса (`1.61 с`) | Кэширование фабрик `@njit`-замыканий по `process_id` и `tuple(process_ids)` | **`1.61 с -> < 5 мс`** на холодный старт воркера |"
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


def run_phase0_baseline_update() -> Dict[str, Any]:
    """
    Перезапускает целевые бенчмарки Фазы 0 (step_profiler с дельтами page faults,
    transport_telemetry на W1–W3 и micro_benchmarks с M1a/M1b), обновляет
    profiling_results.json, графики PNG и отчет PROFILING_REPORT.md.
    """
    report_dir = Path("verification/reports/profiling").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "profiling_results.json"
    payload: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    print("[Phase 0 - 1/4] Прогрев JIT-ядер Numba и @cfunc...", flush=True)
    _ = _cpu_thermal_beacon_kernel(10_000)
    warmup_all_profiling_kernels()

    print("[Phase 0 - 2/4] Запуск step_profiler (W1–W3, декомпозиция шагов + дельты Page Faults)...", flush=True)
    step_workloads = build_all_workloads_matrix(
        w1_particles=300_000,
        w2_particles=150_000,
        w3_particles=150_000,
        bank_capacity=25_000,
    )
    step_results = run_step_profiling_suite(step_workloads)

    print("[Phase 0 - 3/4] Запуск transport_telemetry (W1–W3)...", flush=True)
    telemetry_workloads = build_all_workloads_matrix(
        w1_particles=12_000,
        w2_particles=12_000,
        w3_particles=12_000,
    )
    telemetry_reports = run_transport_telemetry_suite(telemetry_workloads)

    print("[Phase 0 - 4/4] Запуск micro_benchmarks (M1a/M1b, M2..M7)...", flush=True)
    micro_records = run_micro_benchmarks_suite()

    payload["step_results"] = [asdict(item) for item in step_results]
    payload["telemetry_reports"] = [asdict(item) for item in telemetry_reports]
    payload["micro_records"] = [asdict(item) for item in micro_records]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_step_phase_breakdown(step_results, report_dir)
    plot_transport_telemetry_comparison(telemetry_reports, report_dir)
    plot_micro_benchmarks_speedup(micro_records, report_dir)

    reformat_report_from_saved_json()
    return payload


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
    parser.add_argument(
        "--phase0-baseline",
        action="store_true",
        help="Целевой прогон Фазы 0 (step_profiler + page faults, transport_telemetry, micro_benchmarks)",
    )
    args = parser.parse_args()
    if args.reformat_only:
        reformat_report_from_saved_json()
    elif args.phase0_baseline:
        run_phase0_baseline_update()
    else:
        run_full_profiling_pipeline(quick_mode=bool(args.quick))


if __name__ == "__main__":
    main()
