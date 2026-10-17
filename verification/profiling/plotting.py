"""
Модуль генерации иллюстративных графиков профилирования конвейера NMSimToolkit.

Формирует набор PNG-диаграмм в директории verification/reports/profiling/:
1. 01_cprofile_domain_breakdown.png — распределение времени CPython (cProfile) по доменам;
2. 02_data_handlers_comparison.png — сравнение всех обработчиков данных (CPU vs HDF5 I/O vs Threaded);
3. 03_step_phase_breakdown.png — 10-фазная декомпозиция цикла переноса и пропускная способность;
4. 04_transport_telemetry_comparison.png — алгоритмические счетчики рейкастинга и Вудкока;
5. 05_micro_benchmarks_speedup.png — изолированные микробенчмарки горячих узлов (M1..M7);
6. 06_orchestrator_scaling.png — параллельное масштабирование Orchestrator и конкуренция за HDF5;
7. 07_batch_size_scaling.png — влияние размера батча частиц (ParticleBank capacity) на скорость и фазы шага.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verification.profiling.batch_size_benchmark import BatchSizeBenchmarkRecord
from verification.profiling.cprofile_harness import EndToEndCProfileResult
from verification.profiling.data_handlers_benchmark import (
    BufferCapacityScalingRecord,
    DataHandlerBenchmarkRecord,
)
from verification.profiling.micro_benchmarks import MicroBenchmarkRecord
from verification.profiling.orchestrator_scaling import OrchestratorScalingRecord
from verification.profiling.step_profiler import StepPhaseBreakdownResult
from verification.profiling.transport_telemetry import TransportTelemetryMetrics


def _apply_plot_style() -> None:
    """Настраивает единый читаемый стиль графиков с поддержкой кириллицы."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def plot_cprofile_domain_breakdown(
    summaries: List[EndToEndCProfileResult],
    output_dir: Path,
) -> Path:
    """Строит диаграмму распределения собственного времени (tottime) по доменам CPython."""
    _apply_plot_style()
    output_path = output_dir / "01_cprofile_domain_breakdown.png"

    labels = [f"{item.workload_id}" for item in summaries]
    domain_specs = [
        ("jit_transport_and_raycast", "jit_transport_pct", "JIT: Геометрия и Вудкок", "#1f77b4"),
        ("jit_physics_processes", "jit_physics_pct", "JIT: Физические процессы", "#17becf"),
        ("source_generation", "source_generation_pct", "Генерация источника (Source)", "#ff7f0e"),
        ("buffer_and_escaped_management", "buffer_and_escaped_pct", "Буферы и escaped_particles", "#2ca02c"),
        ("data_handler_processing", "data_handler_pct", "Обработчики данных (CPU)", "#d62728"),
        ("hdf5_disk_io", "hdf5_io_pct", "Файловый вывод HDF5 (h5py)", "#9467bd"),
        ("python_orchestration", "orchestration_pct", "Python-оркестрация шага", "#7f7f7f"),
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))
    x_pos = np.arange(len(summaries))
    bottom_vals = np.zeros(len(summaries), dtype=np.float64)
    bottom_pct = np.zeros(len(summaries), dtype=np.float64)

    for attr_abs, attr_pct, title_str, color_hex in domain_specs:
        abs_values = np.array(
            [_get_domain_value(item, attr_abs) for item in summaries],
            dtype=np.float64,
        )
        pct_values = np.array(
            [_get_domain_value(item, attr_pct) for item in summaries],
            dtype=np.float64,
        )
        ax_left.bar(x_pos, abs_values, bottom=bottom_vals, label=title_str, color=color_hex, width=0.55)
        ax_right.bar(x_pos, pct_values, bottom=bottom_pct, label=title_str, color=color_hex, width=0.55)
        bottom_vals += abs_values
        bottom_pct += pct_values

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax_left.set_ylabel("Суммарное собственное время tottime (с)")
    ax_left.set_title("Абсолютный вклад доменов в сквозном профиле cProfile")
    ax_left.legend(loc="upper left", fontsize=8)

    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax_right.set_ylabel("Доля времени (%)")
    ax_right.set_ylim(0, 105)
    ax_right.set_title("Относительное распределение времени по подсистемам (%)")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _get_domain_value(item: EndToEndCProfileResult, field_name: str) -> float:
    """Возвращает числовое поле или свойство из CProfileDomainBreakdown без динамического getattr."""
    dom = item.domain_breakdown
    if field_name == "jit_transport_and_raycast":
        return dom.jit_transport_and_raycast
    if field_name == "jit_physics_processes":
        return dom.jit_physics_processes
    if field_name == "source_generation":
        return dom.source_generation
    if field_name == "buffer_and_escaped_management":
        return dom.buffer_and_escaped_management
    if field_name == "data_handler_processing":
        return dom.data_handler_processing
    if field_name == "hdf5_disk_io":
        return dom.hdf5_disk_io
    if field_name == "python_orchestration":
        return dom.python_orchestration
    if field_name == "jit_transport_pct":
        return dom.jit_transport_pct
    if field_name == "jit_physics_pct":
        return dom.jit_physics_pct
    if field_name == "source_generation_pct":
        return dom.source_generation_pct
    if field_name == "buffer_and_escaped_pct":
        return dom.buffer_and_escaped_pct
    if field_name == "data_handler_pct":
        return dom.data_handler_pct
    if field_name == "hdf5_io_pct":
        return dom.hdf5_io_pct
    if field_name == "orchestration_pct":
        return dom.orchestration_pct
    return 0.0


def plot_data_handlers_comparison(
    records: List[DataHandlerBenchmarkRecord],
    buffer_records: List[BufferCapacityScalingRecord],
    output_dir: Path,
) -> Path:
    """Строит сравнительные диаграммы производительности всех обработчиков данных (DataHandlers)."""
    _apply_plot_style()
    output_path = output_dir / "02_data_handlers_comparison.png"

    nema_direct = [rec for rec in records if "W2_NEMA" in rec.workload_id and rec.collimator_kind == "direct"]
    sub_records = nema_direct if nema_direct else records[:9]

    fig = plt.figure(figsize=(14, 10))
    ax_top = plt.subplot(2, 1, 1)

    names = [rec.handler_code.replace("_", "\n") for rec in sub_records]
    x_pos = np.arange(len(sub_records))
    width = 0.25

    transport_times = np.array([rec.transport_only_time for rec in sub_records], dtype=np.float64)
    cpu_times = np.array([rec.handler_cpu_time for rec in sub_records], dtype=np.float64)
    io_times = np.array([rec.hdf5_io_time for rec in sub_records], dtype=np.float64)
    threaded_times = np.array([rec.threaded_wall_time for rec in sub_records], dtype=np.float64)

    ax_top.bar(x_pos - width, transport_times, width=width, label="Поток переноса + очередь (с)", color="#1f77b4")
    ax_top.bar(x_pos, cpu_times, width=width, label="CPU обработка в DataHandler (с)", color="#ff7f0e")
    ax_top.bar(
        x_pos,
        io_times,
        width=width,
        bottom=cpu_times,
        label="Запись на диск HDF5 I/O (с)",
        color="#d62728",
    )
    ax_top.bar(
        x_pos + width,
        threaded_times,
        width=width,
        label="Сквозное Wall-Time (2 потока Sim+Data) (с)",
        color="#2ca02c",
    )

    ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels(names, fontsize=8)
    ax_top.set_ylabel("Время выполнения (с)")
    ax_top.set_title("Декомпозиция времени переноса, CPU-фильтрации и HDF5 I/O по типам DataHandlers (W2 NEMA Direct)")
    ax_top.legend(loc="upper left", fontsize=9)

    ax_bot_left = plt.subplot(2, 2, 3)
    ax_bot_right = plt.subplot(2, 2, 4)

    shares = np.array([rec.data_share_of_total_pct for rec in sub_records], dtype=np.float64)
    bar_colors = ["#d62728" if val > 25.0 else "#2ca02c" for val in shares]
    ax_bot_left.barh(np.arange(len(sub_records)), shares, color=bar_colors)
    ax_bot_left.set_yticks(np.arange(len(sub_records)))
    ax_bot_left.set_yticklabels([rec.handler_code for rec in sub_records], fontsize=8)
    ax_bot_left.set_xlabel("Доля (DataHandler CPU + HDF5 I/O) в общем времени (%)")
    ax_bot_left.set_title("Доля времени на обработку и запись данных (%)")

    direct_buf_recs = [rec for rec in buffer_records if rec.collimator_kind == "direct"]
    if not direct_buf_recs:
        direct_buf_recs = buffer_records
    if direct_buf_recs:
        caps = [str(rec.buffer_capacity) for rec in direct_buf_recs]
        c_pos = np.arange(len(direct_buf_recs))
        h_cpu = [rec.handler_cpu_time for rec in direct_buf_recs]
        h_io = [rec.hdf5_io_time for rec in direct_buf_recs]
        ax_bot_right.bar(c_pos, h_cpu, width=0.45, label="CPU фильтрация (HistoryAssembler)", color="#ff7f0e")
        ax_bot_right.bar(c_pos, h_io, width=0.45, bottom=h_cpu, label="HDF5 I/O запись", color="#9467bd")
        ax_bot_right.set_xticks(c_pos)
        ax_bot_right.set_xticklabels(caps)
        ax_bot_right.set_xlabel("Емкость буфера взаимодействий (buffer_capacity)")
        ax_bot_right.set_ylabel("Суммарное время обработки данных (с)")
        ax_bot_right.set_title("Влияние buffer_capacity на накладные расходы HistoryAssemblerHandler")
        ax_bot_right.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_step_phase_breakdown(
    results: List[StepPhaseBreakdownResult],
    output_dir: Path,
) -> Path:
    """Строит 10-фазную декомпозицию пошагового цикла симуляции и пропускную способность."""
    _apply_plot_style()
    output_path = output_dir / "03_step_phase_breakdown.png"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    labels = [f"{res.workload_id}" for res in results]
    x_pos = np.arange(len(results))

    phase_specs = [
        ("P2_Source_Inject", "P2: Инжекция источника (Source.inject)", "#ff7f0e"),
        ("P4_Geometry_Raycast", "P4: Навигация и рейкастинг (Geometry)", "#1f77b4"),
        ("P5_Woodcock_Transport", "P5: Дельта-трекинг Вудкока (Transport)", "#2ca02c"),
        ("P8_Physics_Processes", "P8: Физические ядра (Фото/Комптон/Рэлей)", "#d62728"),
        ("P9_Invalidation_Escaped", "P9: Инвалидация и сбор escaped_particles", "#e377c2"),
        ("P10_DataHandler_HDF5", "P10: Фильтрация DataHandler + HDF5 I/O", "#9467bd"),
        ("OTHER", "P3+P6+P7: Индексы, initial_states, compaction", "#7f7f7f"),
    ]

    bottom_pct = np.zeros(len(results), dtype=np.float64)
    for key_name, phase_label, color_hex in phase_specs:
        pct_vals = np.zeros(len(results), dtype=np.float64)
        for idx, res in enumerate(results):
            pcts = res.phase_percentages()
            if key_name == "OTHER":
                pct_vals[idx] = (
                    pcts.get("P3_Active_Indices", 0.0)
                    + pcts.get("P6_Initial_States", 0.0)
                    + pcts.get("P7_Stream_Compaction", 0.0)
                )
            else:
                pct_vals[idx] = pcts.get(key_name, 0.0)
        ax_left.bar(x_pos, pct_vals, bottom=bottom_pct, label=phase_label, color=color_hex, width=0.55)
        bottom_pct += pct_vals

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax_left.set_ylabel("Доля времени шага (%)")
    ax_left.set_ylim(0, 105)
    ax_left.set_title("Пофазная структура времени цикла переноса (10 фаз)")
    ax_left.legend(loc="upper right", fontsize=8)

    throughputs = np.array([res.throughput_histories_per_sec for res in results], dtype=np.float64)
    bar_colors = ["#1f77b4" if res.collimator_kind == "parametric" else "#ff7f0e" for res in results]
    bars = ax_right.bar(x_pos, throughputs, color=bar_colors, width=0.55)
    for bar_item, val in zip(bars, throughputs):
        ax_right.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            bar_item.get_height() * 1.01,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax_right.set_ylabel("Скорость моделирования (историй / с)")
    ax_right.set_title("Сравнение пропускной способности сцен (Parametric vs Direct)")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_transport_telemetry_comparison(
    reports: List[TransportTelemetryMetrics],
    output_dir: Path,
) -> Path:
    """Строит графики алгоритмических счетчиков переноса (рейкастинг, Вудкок, сечения)."""
    _apply_plot_style()
    output_path = output_dir / "04_transport_telemetry_comparison.png"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    labels = [f"{rep.workload_id}" for rep in reports]
    x_pos = np.arange(len(reports))
    width = 0.25

    box_per_hist = np.array([rep.total_box_intersections / max(1, rep.histories_count) for rep in reports])
    hex_per_hist = np.array([rep.total_hex_prism_intersections / max(1, rep.histories_count) for rep in reports])
    search_per_hist = np.array([rep.searchsorted_per_history for rep in reports])

    ax_left.bar(x_pos - width, box_per_hist, width=width, label="Вызовы _box_intersect / ист.", color="#1f77b4")
    ax_left.bar(x_pos, hex_per_hist, width=width, label="Вызовы _hex_prism_intersect / ист.", color="#ff7f0e")
    ax_left.bar(
        x_pos + width,
        search_per_hist,
        width=width,
        label="Вызовы np.searchsorted (LAC) / ист.",
        color="#2ca02c",
    )

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax_left.set_ylabel("Среднее число вызовов на 1 историю")
    ax_left.set_title("Интенсивность вызовов геометрических и физических ядер на 1 частицу")
    ax_left.legend(loc="upper left", fontsize=8)

    real_steps = np.array([rep.total_real_interactions / max(1, rep.histories_count) for rep in reports])
    null_steps = np.array([rep.total_null_collisions / max(1, rep.histories_count) for rep in reports])

    ax_right.bar(x_pos, real_steps, width=0.5, label="Реальные взаимодействия на 1 ист.", color="#2ca02c")
    ax_right.bar(
        x_pos,
        null_steps,
        width=0.5,
        bottom=real_steps,
        label="Холостые (виртуальные) дельта-шаги Вудкока",
        color="#d62728",
    )
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax_right.set_ylabel("Среднее число итераций Вудкока на 1 историю")
    ax_right.set_title("Баланс реальных и виртуальных шагов в WoodcockVoxelVolume / Collimator")
    ax_right.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_micro_benchmarks_speedup(
    records: List[MicroBenchmarkRecord],
    output_dir: Path,
) -> Path:
    """Строит диаграмму ускорения (Speedup) изолированных микробенчмарков M1..M6."""
    _apply_plot_style()
    output_path = output_dir / "05_micro_benchmarks_speedup.png"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    short_labels = [rec.benchmark_id for rec in records]
    y_pos = np.arange(len(records))
    speedups = np.array([rec.speedup_factor for rec in records], dtype=np.float64)

    bars = ax_left.barh(y_pos, speedups, color="#1f77b4", height=0.55)
    ax_left.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    for bar_item, val in zip(bars, speedups):
        ax_left.text(
            bar_item.get_width() * 1.02,
            bar_item.get_y() + bar_item.get_height() / 2.0,
            f"{val:.2f}x",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(short_labels, fontsize=9)
    ax_left.set_xlabel("Коэффициент ускорения (Текущая / Альтернативная реализация)")
    ax_left.set_xlim(0, max(speedups) * 1.22)
    ax_left.set_title("Потенциал ускорения горячих узлов в микробенчмарках (M1..M6)")

    curr_ns = np.array([rec.current_ns_per_op for rec in records], dtype=np.float64)
    alt_ns = np.array([rec.alternative_ns_per_op for rec in records], dtype=np.float64)
    height = 0.35

    ax_right.barh(y_pos - height / 2.0, curr_ns, height=height, label="Текущая реализация", color="#d62728")
    ax_right.barh(y_pos + height / 2.0, alt_ns, height=height, label="Оптимизированный прототип", color="#2ca02c")
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(short_labels, fontsize=9)
    ax_right.set_xscale("log")
    ax_right.set_xlabel("Время на одну элементарную операцию (нс, лог. шкала)")
    ax_right.set_title("Латентность элементарных операций (нс / вызов)")
    ax_right.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_orchestrator_scaling(
    records: List[OrchestratorScalingRecord],
    output_dir: Path,
) -> Path:
    """Строит график масштабируемости параллельного пула Orchestrator и файловой конкуренции."""
    _apply_plot_style()
    output_path = output_dir / "06_orchestrator_scaling.png"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

    for coll_type, color_hex, marker_style in [
        ("parametric", "#1f77b4", "o"),
        ("direct", "#ff7f0e", "s"),
    ]:
        per_task_recs = [
            rec
            for rec in records
            if rec.collimator_type == coll_type and rec.file_mode == "per_task_file"
        ]
        if per_task_recs:
            pools = [rec.pool_size for rec in per_task_recs]
            speedups = [rec.speedup_vs_seq for rec in per_task_recs]
            ax_left.plot(
                pools,
                speedups,
                marker=marker_style,
                linewidth=2.0,
                color=color_hex,
                label=f"Коллиматор: {coll_type} (per-task HDF5)",
            )

    all_pools = sorted({rec.pool_size for rec in records})
    if all_pools:
        ax_left.plot(all_pools, all_pools, linestyle="--", color="gray", label="Линейный идеал (Nх)")
    ax_left.set_xlabel("Размер пула процессов (pool_size)")
    ax_left.set_ylabel("Коэффициент ускорения (Speedup)")
    ax_left.set_title("Масштабируемость Orchestrator по числу воркеров")
    ax_left.legend(loc="upper left", fontsize=9)

    max_pool = max(all_pools) if all_pools else 1
    max_pool_recs = [rec for rec in records if rec.pool_size == max_pool]
    labels = [f"{rec.collimator_type}\n({rec.file_mode})" for rec in max_pool_recs]
    wall_vals = [rec.wall_time for rec in max_pool_recs]
    x_pos = np.arange(len(max_pool_recs))
    colors = ["#2ca02c" if rec.file_mode == "per_task_file" else "#d62728" for rec in max_pool_recs]

    ax_right.bar(x_pos, wall_vals, color=colors, width=0.5)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(labels, fontsize=9)
    ax_right.set_ylabel("Сквозное время выполнения протокола (с)")
    ax_right.set_title(f"Конкуренция за HDF5 Lock при pool_size={max_pool}")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_batch_size_scaling(
    records: List[BatchSizeBenchmarkRecord],
    output_dir: Path,
) -> Path:
    """
    Строит 4-панельную диаграмму влияния размера батча частиц (batch_size) на скорость и фазы переноса
    с разделением стационарной фазы полного банка (Full-Bank Phase) и фазы опустошения (Depletion Phase).
    """
    _apply_plot_style()
    output_path = output_dir / "07_batch_size_scaling.png"

    fig = plt.figure(figsize=(16, 10.5))
    ax_tl = plt.subplot(2, 2, 1)
    ax_tr = plt.subplot(2, 2, 2)
    ax_bl = plt.subplot(2, 2, 3)
    ax_br = plt.subplot(2, 2, 4)

    workload_ids = sorted({rec.workload_id for rec in records})
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

    total_hist_label = f"{records[0].total_particles:,}" if records else "6,000,000"

    for idx, wid in enumerate(workload_ids):
        w_recs = sorted(
            [rec for rec in records if rec.workload_id == wid],
            key=lambda item: item.batch_size,
        )
        if not w_recs:
            continue
        b_sizes = [rec.batch_size for rec in w_recs]
        full_tps = [rec.full_bank_histories_per_sec for rec in w_recs]
        tot_tps = [rec.histories_per_sec for rec in w_recs]
        slowdown = [rec.depletion_track_slowdown_factor for rec in w_recs]
        color_hex = palette[idx % len(palette)]
        m_style = markers[idx % len(markers)]

        ax_tl.plot(
            b_sizes,
            full_tps,
            marker=m_style,
            linewidth=2.2,
            color=color_hex,
            label=f"{wid} (Full-Bank)",
        )
        ax_tl.plot(
            b_sizes,
            tot_tps,
            marker=m_style,
            linewidth=1.3,
            linestyle="--",
            alpha=0.75,
            color=color_hex,
            label=f"{wid} (С хвостом)",
        )
        ax_tr.plot(
            b_sizes,
            slowdown,
            marker=m_style,
            linewidth=2.0,
            color=color_hex,
            label=f"{wid} (Slowdown трека в Depletion)",
        )

    ax_tl.set_xscale("log")
    ax_tl.set_xlabel("Емкость банка частиц (batch_size, лог. шкала)")
    ax_tl.set_ylabel("Скорость моделирования (историй / с)")
    ax_tl.set_title(f"Скорость Full-Bank vs Сквозная ({total_hist_label} историй)")
    ax_tl.legend(loc="lower right", fontsize=7.5, ncol=2)

    ax_tr.set_xscale("log")
    ax_tr.axhline(1.0, linestyle=":", color="gray", linewidth=1.2)
    ax_tr.set_xlabel("Емкость банка частиц (batch_size, лог. шкала)")
    ax_tr.set_ylabel("Коэффициент замедления трека (Full / Depletion)")
    ax_tr.set_title("Потеря эффективности векторизации на фазе опустошения банка")
    ax_tr.legend(loc="upper left", fontsize=8)

    # Нижние графики: пофазный разбор в режиме Full-Bank Phase для W2_NEMA_Tb161_Direct
    target_wid = "W2_NEMA_Tb161_Direct"
    focus_recs = sorted(
        [rec for rec in records if rec.workload_id == target_wid],
        key=lambda item: item.batch_size,
    )
    if not focus_recs and records:
        target_wid = records[0].workload_id
        focus_recs = sorted(
            [rec for rec in records if rec.workload_id == target_wid],
            key=lambda item: item.batch_size,
        )

    if focus_recs:
        x_pos = np.arange(len(focus_recs))
        x_labels = [f"{rec.batch_size:,}" for rec in focus_recs]
        phase_defs = [
            ("source", "Source (Инжекция)", "#ff7f0e"),
            ("geom", "Geometry (Рейкастинг)", "#1f77b4"),
            ("woodcock", "Woodcock (Дельта-трекинг)", "#2ca02c"),
            ("processes", "Processes (Фото/Комптон/Рэлей)", "#d62728"),
            ("compaction", "Compaction + Escaped", "#9467bd"),
        ]

        bot_sec = np.zeros(len(focus_recs), dtype=np.float64)
        bot_pct = np.zeros(len(focus_recs), dtype=np.float64)

        for p_key, p_label, p_color in phase_defs:
            if p_key == "source":
                vals_sec = np.array([rec.full_source_time_sec for rec in focus_recs])
                vals_pct = np.array([rec.full_source_pct for rec in focus_recs])
            elif p_key == "geom":
                vals_sec = np.array([rec.full_geometry_raycast_time_sec for rec in focus_recs])
                vals_pct = np.array([rec.full_geometry_raycast_pct for rec in focus_recs])
            elif p_key == "woodcock":
                vals_sec = np.array([rec.full_woodcock_time_sec for rec in focus_recs])
                vals_pct = np.array([rec.full_woodcock_pct for rec in focus_recs])
            elif p_key == "processes":
                vals_sec = np.array([rec.full_processes_time_sec for rec in focus_recs])
                vals_pct = np.array([rec.full_processes_pct for rec in focus_recs])
            else:
                vals_sec = np.array([rec.full_compaction_and_escaped_time_sec for rec in focus_recs])
                vals_pct = np.array([rec.full_compaction_and_escaped_pct for rec in focus_recs])

            ax_bl.bar(x_pos, vals_sec, bottom=bot_sec, width=0.58, label=p_label, color=p_color)
            ax_br.bar(x_pos, vals_pct, bottom=bot_pct, width=0.58, label=p_label, color=p_color)
            bot_sec += vals_sec
            bot_pct += vals_pct

        # Наложим линию времени фазы опустошения поверх столбцов на ax_bl
        depl_secs = np.array([rec.depletion_time_sec for rec in focus_recs])
        ax_bl.plot(
            x_pos,
            depl_secs,
            marker="o",
            linestyle="--",
            linewidth=2.0,
            color="#111111",
            label="Хвост опустошения (Depletion, с)",
        )

        ax_bl.set_xticks(x_pos)
        ax_bl.set_xticklabels(x_labels, fontsize=8, rotation=20, ha="right")
        ax_bl.set_xlabel("Размер батча (batch_size)")
        ax_bl.set_ylabel("Суммарное время (с)")
        ax_bl.set_title(f"Абсолютное время фаз Full-Bank и хвоста Depletion ({target_wid})")
        ax_bl.legend(loc="upper right", fontsize=7.5)

        ax_br.set_xticks(x_pos)
        ax_br.set_xticklabels(x_labels, fontsize=8, rotation=20, ha="right")
        ax_br.set_xlabel("Размер батча (batch_size)")
        ax_br.set_ylabel("Доля времени в Full-Bank Phase (%)")
        ax_br.set_ylim(0, 105)
        ax_br.set_title(f"Структура затрат (%) в стационарном режиме Full-Bank ({target_wid})")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "plot_batch_size_scaling",
    "plot_cprofile_domain_breakdown",
    "plot_data_handlers_comparison",
    "plot_micro_benchmarks_speedup",
    "plot_orchestrator_scaling",
    "plot_step_phase_breakdown",
    "plot_transport_telemetry_comparison",
]
