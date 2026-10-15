"""Эксперимент 5: Бенчмарк производительности ParametricParallelCollimator vs DirectParallelCollimator.

Измеряет чистую скорость транспорта фотонов (после предварительного прогрева JIT-компилятора Numba)
для всех 4 коллиматоров серии GE Healthcare (LEHR, LEUHR, MEGP, HEGP) при различных энергиях
гамма-излучения (Tc-99m 140.5 кэВ, In-111 245.4 кэВ, I-131 364.5 кэВ) и режимах облучения.

Физико-вычислительная причина различия скорости:
- В ParametricParallelCollimator используется дельта-трекинг Вудкока с мажорантным сечением свинца:
  внутри вакуумных каналов частица совершает множество фиктивных шагов длины ~ 1 / mu_Pb(E)
  с вызовом @cfunc на каждом шаге (особенно медленно при низких энергиях, где mu_Pb велико).
- В DirectParallelCollimator внутри вакуумного канала (PeriodicHexPrism) сечение равно нулю,
  и частица пролетает весь канал за один аналитический шаг _hex_prism_intersect до стенки или выхода.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics

from verification.common.plotting import plot_speed_benchmark
from verification.common.presets import GE_COLLIMATOR_PRESETS, RADIONUCLIDE_PRESETS
from verification.common.scene_builders import run_collimator_simulation


@dataclass(frozen=True)
class SpeedBenchmarkRecord:
    """Запись бенчмарка скорости для одной конфигурации (коллиматор, нуклид, режим источника).

    Attributes:
        scenario_label: Краткая подпись сценария для графиков и таблиц.
        collimator_code: Код коллиматора GE.
        nuclide_name: Название радионуклида.
        photon_energy: Энергия фотонов [МэВ].
        emission_mode: Режим испускания ('Cone 14 deg' или 'Isotropic 4pi').
        num_particles: Число фотонов в одном прогоне.
        num_repeats: Число повторов замера.
        median_time_parametric: Медианное время выполнения Parametric [с].
        median_time_direct: Медианное время выполнения Direct [с].
        throughput_parametric: Пропускная способность Parametric [тыс. историй / с].
        throughput_direct: Пропускная способность Direct [тыс. историй / с].
        speedup_factor: Коэффициент ускорения Direct относительно Parametric (T_param / T_direct).
    """

    scenario_label: str
    collimator_code: str
    nuclide_name: str
    photon_energy: float
    emission_mode: str
    num_particles: int
    num_repeats: int
    median_time_parametric: float
    median_time_direct: float
    throughput_parametric: float
    throughput_direct: float
    speedup_factor: float


DEFAULT_BENCHMARK_CONFIGS: tuple[tuple[str, str, float | None, str], ...] = (
    ("LEHR", "Tc99m", 0.25, "LEHR / Tc-99m (Cone)"),
    ("LEUHR", "Tc99m", 0.25, "LEUHR / Tc-99m (Cone)"),
    ("MEGP", "In111", 0.25, "MEGP / In-111 (Cone)"),
    ("HEGP", "I131", 0.25, "HEGP / I-131 (Cone)"),
    ("LEHR", "I131", 0.25, "LEHR / I-131 (Star Cone)"),
    ("LEHR", "Tc99m", None, "LEHR / Tc-99m (4pi Iso)"),
)


def _warmup_jit_kernels() -> None:
    """Выполняет короткий прогревочный прогон обеих моделей для компиляции всех ядер Numba."""
    preset = GE_COLLIMATOR_PRESETS["LEHR"]
    nuclide = RADIONUCLIDE_PRESETS["Tc99m"]
    run_collimator_simulation(
        preset=preset,
        implementation_kind="parametric",
        source_distance=100.0,
        energy=nuclide.primary_energy,
        particles_number=5_000,
        max_polar_angle=0.25,
        seed=9991,
    )
    run_collimator_simulation(
        preset=preset,
        implementation_kind="direct",
        source_distance=100.0,
        energy=nuclide.primary_energy,
        particles_number=5_000,
        max_polar_angle=0.25,
        seed=9992,
    )


def run_exp05_speed_benchmark(
    output_dir: Path,
    num_particles: int = 1_000_000,
    num_repeats: int = 3,
    configs: tuple[tuple[str, str, float | None, str], ...] = DEFAULT_BENCHMARK_CONFIGS,
) -> list[SpeedBenchmarkRecord]:
    """Запускает серию замеров производительности Parametric vs Direct и сохраняет отчет.

    Args:
        output_dir: Папка для сохранения графиков, JSON и Markdown-отчета.
        num_particles: Число фотонов в одном замере.
        num_repeats: Число повторов для вычисления медианного времени.
        configs: Кортеж конфигураций (collimator_code, nuclide_key, max_polar_angle, label).

    Returns:
        Список записей SpeedBenchmarkRecord.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _warmup_jit_kernels()

    records: list[SpeedBenchmarkRecord] = []

    for cfg_idx, (collimator_code, nuclide_key, cone_angle, label_str) in enumerate(configs):
        preset = GE_COLLIMATOR_PRESETS[collimator_code]
        nuclide = RADIONUCLIDE_PRESETS[nuclide_key]
        effective_angle = cone_angle if cone_angle is not None else math.pi

        times_param: list[float] = []
        times_direct: list[float] = []

        for rep_idx in range(num_repeats):
            seed_val = 505000 + cfg_idx * 100 + rep_idx * 10
            out_p = run_collimator_simulation(
                preset=preset,
                implementation_kind="parametric",
                source_distance=100.0,
                energy=nuclide.primary_energy,
                particles_number=num_particles,
                max_polar_angle=effective_angle,
                cell_averaging=True,
                seed=seed_val,
            )
            times_param.append(float(out_p.elapsed_time))

            out_d = run_collimator_simulation(
                preset=preset,
                implementation_kind="direct",
                source_distance=100.0,
                energy=nuclide.primary_energy,
                particles_number=num_particles,
                max_polar_angle=effective_angle,
                cell_averaging=True,
                seed=seed_val + 1,
            )
            times_direct.append(float(out_d.elapsed_time))

        med_param = float(statistics.median(times_param))
        med_direct = float(statistics.median(times_direct))

        throughput_param = (float(num_particles) / 1000.0) / max(med_param, 1e-9)
        throughput_direct = (float(num_particles) / 1000.0) / max(med_direct, 1e-9)
        speedup = med_param / max(med_direct, 1e-9)

        mode_str = f"Cone ({cone_angle:.2f} rad)" if cone_angle is not None else "Isotropic 4pi"

        records.append(
            SpeedBenchmarkRecord(
                scenario_label=label_str,
                collimator_code=preset.code,
                nuclide_name=nuclide.title,
                photon_energy=float(nuclide.primary_energy),
                emission_mode=mode_str,
                num_particles=num_particles,
                num_repeats=num_repeats,
                median_time_parametric=med_param,
                median_time_direct=med_direct,
                throughput_parametric=throughput_param,
                throughput_direct=throughput_direct,
                speedup_factor=speedup,
            )
        )

    plot_speed_benchmark(
        scenario_labels=[rec.scenario_label for rec in records],
        throughput_parametric=[rec.throughput_parametric for rec in records],
        throughput_direct=[rec.throughput_direct for rec in records],
        speedup_factors=[rec.speedup_factor for rec in records],
        output_path=output_dir / "exp05_speed_benchmark.png",
    )

    json_path = output_dir / "exp05_speed_benchmark.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(rec) for rec in records], json_file, indent=2, ensure_ascii=False)

    md_lines = [
        "# Эксперимент 5: Бенчмарк производительности `ParametricParallelCollimator` vs `DirectParallelCollimator`",
        "",
        f"Число историй на прогон: **{num_particles:,}** (медиана по **{num_repeats}** запускам после прогрева JIT).",
        "",
        "| Конфигурация | Режим источника | Время `Parametric`, с | Время `Direct`, с | Скорость `Parametric`, тыс./с | Скорость `Direct`, тыс./с | Ускорение (`Speedup`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rec in records:
        md_lines.append(
            f"| **{rec.scenario_label}** | {rec.emission_mode} | "
            f"{rec.median_time_parametric:.3f} | {rec.median_time_direct:.3f} | "
            f"{rec.throughput_parametric:,.1f} | {rec.throughput_direct:,.1f} | "
            f"**{rec.speedup_factor:.2f}x** |"
        )

    md_path = output_dir / "exp05_speed_benchmark.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return records


__all__ = [
    "SpeedBenchmarkRecord",
    "DEFAULT_BENCHMARK_CONFIGS",
    "run_exp05_speed_benchmark",
]

