"""
Модуль изолированных микробенчмарков горячих узлов NMSimToolkit.

Оценивает чистую пропускную способность узких мест и сравнивает текущую реализацию
с алгоритмическими альтернативами из архитектурного бэклога (docs/architecture_backlog.md):
- M1: Интерполяция макроскопических сечений (np.searchsorted O(log K) vs лог-сетка O(1));
- M2: Построение таблицы эмиссии воксельного источника (np.meshgrid vs np.flatnonzero);
- M3: Генерация позиций в воксельном фантоме (rng.choice O(M log M) vs Vose's Alias O(1) @njit);
- M4: Маппинг объемов в SensitiveVolumeHandler (Python-цикл + dtype=object vs целочисленный LUT);
- M5: Фильтрация историй в HistoryAssemblerHandler (set + 4x np.isin vs булева таблица ID);
- M6: Ядра геометрии (_box_intersect vs _hex_prism_intersect);
- M7: Удельная стоимость 1 шага навигации+Вудкока (cast_path_kernel + _transport_kernel) против физических ядер (_compton_kernel).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Tuple

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.geometry_kernels import _box_intersect, _hex_prism_intersect, cast_path_kernel
from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import Float, Index, Length, Species
from core.other.vectors import Vector3DSoA
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext, SimulationDataBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import _get_macroscopic_cross_sections
from core.transport.propagator import ParticlePropagator
from core.transport.transport_buffer import TransportBuffer
from settings.processes_settings import processes_list
from verification.profiling.workloads import (
    build_nema_tb161_workload,
    warmup_propagator_instance,
)


@dataclass(frozen=True)
class MicroBenchmarkRecord:
    """Результат одного изолированного микробенчмарка (текущая реализация vs оптимизированный прототип)."""

    benchmark_id: str
    title: str
    operations_count: int
    current_label: str
    alternative_label: str
    current_time: float
    alternative_time: float
    current_ns_per_op: float
    alternative_ns_per_op: float
    speedup_factor: float


@njit(cache=True)
def _bench_current_cross_sections_loop(
    energies: NDArray[Float],
    mat_ids: NDArray[Index],
    material_bank: MaterialBank,
) -> Float:
    """Пакетный вызов текущей функции _get_macroscopic_cross_sections (np.searchsorted)."""
    out_lacs = np.zeros(3, dtype=Float)
    acc = Float(0.0)
    for idx in range(energies.shape[0]):
        _get_macroscopic_cross_sections(energies[idx], mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
    return acc


@njit(cache=True)
def _bench_o1_log_grid_cross_sections_loop(
    energies: NDArray[Float],
    mat_ids: NDArray[Index],
    uniform_lac_table: NDArray[Float],
    log_e_min: Float,
    inv_dlog_e: Float,
    grid_len: int,
) -> Float:
    """Пакетный вызов O(1) интерполяции сечений по равномерной логарифмической сетке без searchsorted."""
    acc = Float(0.0)
    max_idx = grid_len - 2
    for idx in range(energies.shape[0]):
        energy = energies[idx]
        mat_id = mat_ids[idx]
        scaled = (math.log(energy) - log_e_min) * inv_dlog_e
        bin_idx = int(scaled)
        if bin_idx < 0:
            bin_idx = 0
            frac = Float(0.0)
        elif bin_idx > max_idx:
            bin_idx = max_idx
            frac = Float(1.0)
        else:
            frac = Float(scaled - bin_idx)

        row0 = mat_id * grid_len + bin_idx
        row1 = row0 + 1
        lac0 = uniform_lac_table[row0, 0] + frac * (uniform_lac_table[row1, 0] - uniform_lac_table[row0, 0])
        lac1 = uniform_lac_table[row0, 1] + frac * (uniform_lac_table[row1, 1] - uniform_lac_table[row0, 1])
        lac2 = uniform_lac_table[row0, 2] + frac * (uniform_lac_table[row1, 2] - uniform_lac_table[row0, 2])
        acc = Float(acc + lac0 + lac1 + lac2)
    return acc


def _build_vose_alias_tables(probabilities: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Строит таблицы prob и alias для алгоритма Воза (Vose's Alias Method) за O(M)."""
    num_items = probabilities.shape[0]
    prob_table = np.zeros(num_items, dtype=np.float64)
    alias_table = np.zeros(num_items, dtype=np.int64)
    scaled_probs = probabilities * float(num_items)

    small_list: List[int] = []
    large_list: List[int] = []
    for item_idx in range(num_items):
        if scaled_probs[item_idx] < 1.0:
            small_list.append(item_idx)
        else:
            large_list.append(item_idx)

    while small_list and large_list:
        less_idx = small_list.pop()
        more_idx = large_list.pop()
        prob_table[less_idx] = scaled_probs[less_idx]
        alias_table[less_idx] = more_idx
        scaled_probs[more_idx] = (scaled_probs[more_idx] + scaled_probs[less_idx]) - 1.0
        if scaled_probs[more_idx] < 1.0:
            small_list.append(more_idx)
        else:
            large_list.append(more_idx)

    while large_list:
        prob_table[large_list.pop()] = 1.0
    while small_list:
        prob_table[small_list.pop()] = 1.0

    return prob_table, alias_table


@njit(cache=True)
def _sample_vose_alias_positions_kernel(
    voxel_origins: NDArray[np.float64],
    prob_table: NDArray[np.float64],
    alias_table: NDArray[np.int64],
    rand_u1: NDArray[np.float64],
    rand_u2: NDArray[np.float64],
    jitter_xyz: NDArray[np.float64],
    voxel_size: float,
    out_x: NDArray[np.float64],
    out_y: NDArray[np.float64],
    out_z: NDArray[np.float64],
) -> None:
    """Генерирует глобальные координаты в SoA-массивы за O(1) методом Воза без промежуточных аллокаций."""
    num_samples = rand_u1.shape[0]
    num_voxels = prob_table.shape[0]
    for sample_idx in range(num_samples):
        bin_idx = int(rand_u1[sample_idx] * num_voxels)
        if bin_idx >= num_voxels:
            bin_idx = num_voxels - 1
        chosen_voxel = bin_idx if rand_u2[sample_idx] < prob_table[bin_idx] else alias_table[bin_idx]
        out_x[sample_idx] = voxel_origins[chosen_voxel, 0] + jitter_xyz[sample_idx, 0] * voxel_size
        out_y[sample_idx] = voxel_origins[chosen_voxel, 1] + jitter_xyz[sample_idx, 1] * voxel_size
        out_z[sample_idx] = voxel_origins[chosen_voxel, 2] + jitter_xyz[sample_idx, 2] * voxel_size


@njit(cache=True)
def _bench_box_intersect_loop(
    positions: NDArray[Float],
    directions: NDArray[Float],
) -> Float:
    """Пакетный прогон _box_intersect."""
    acc = Float(0.0)
    for idx in range(positions.shape[0]):
        tmin, tmax = _box_intersect(
            positions[idx, 0],
            positions[idx, 1],
            positions[idx, 2],
            directions[idx, 0],
            directions[idx, 1],
            directions[idx, 2],
            Float(270.0),
            Float(200.0),
            Float(17.5),
        )
        if tmax > tmin:
            acc = Float(acc + tmax - tmin)
    return acc


@njit(cache=True)
def _bench_hex_prism_intersect_loop(
    positions: NDArray[Float],
    directions: NDArray[Float],
    shape_data: NDArray[Float],
) -> Float:
    """Пакетный прогон _hex_prism_intersect."""
    acc = Float(0.0)
    for idx in range(positions.shape[0]):
        tmin, tmax = _hex_prism_intersect(
            positions[idx, 0],
            positions[idx, 1],
            positions[idx, 2],
            directions[idx, 0],
            directions[idx, 1],
            directions[idx, 2],
            shape_data[0],
        )
        if tmax > tmin and tmax < 1e9:
            acc = Float(acc + tmax - tmin)
    return acc


def run_micro_benchmarks_suite() -> List[MicroBenchmarkRecord]:
    """
    Запускает все 6 изолированных микробенчмарков (M1..M6) с предварительным прогревом JIT
    и возвращает список записей MicroBenchmarkRecord.
    """
    records: List[MicroBenchmarkRecord] = []
    rng = np.random.default_rng(12345)

    workload = build_nema_tb161_workload("direct", particles_number=10_000)
    propagator = ParticlePropagator(processes_list=processes_list, rng=rng)
    physics_buffer = PhysicsCompiler().compile_scene(workload.root_scene, propagator.processes)
    material_bank = physics_buffer.material_bank

    # --- M1: Интерполяция макроскопических сечений (np.searchsorted vs O(1) лог-сетка) ---
    n_ops_m1 = 5_000_000
    test_energies = rng.uniform(0.015, 0.300, size=n_ops_m1).astype(Float)
    num_mats = len(material_bank.mat_pointers)
    test_mat_ids = rng.integers(0, max(1, num_mats), size=n_ops_m1, dtype=Index)

    grid_len = 1024
    e_min = 0.001
    e_max = 100.0
    log_e_min = Float(math.log(e_min))
    inv_dlog_e = Float((grid_len - 1) / (math.log(e_max) - math.log(e_min)))
    uniform_lac_table = np.ones((max(1, num_mats) * grid_len, 3), dtype=Float)

    # Прогрев M1
    _bench_current_cross_sections_loop(test_energies[:1000], test_mat_ids[:1000], material_bank)
    _bench_o1_log_grid_cross_sections_loop(
        test_energies[:1000], test_mat_ids[:1000], uniform_lac_table, log_e_min, inv_dlog_e, grid_len
    )

    t0 = time.perf_counter()
    _bench_current_cross_sections_loop(test_energies, test_mat_ids, material_bank)
    t_m1_curr = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    _bench_o1_log_grid_cross_sections_loop(
        test_energies, test_mat_ids, uniform_lac_table, log_e_min, inv_dlog_e, grid_len
    )
    t_m1_alt = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M1_CrossSections",
            title="Интерполяция макроскопических сечений LAC (5 млн запросов)",
            operations_count=n_ops_m1,
            current_label="np.searchsorted O(log K)",
            alternative_label="Равномерная лог-сетка O(1)",
            current_time=t_m1_curr,
            alternative_time=t_m1_alt,
            current_ns_per_op=t_m1_curr * 1e9 / n_ops_m1,
            alternative_ns_per_op=t_m1_alt * 1e9 / n_ops_m1,
            speedup_factor=t_m1_curr / t_m1_alt,
        )
    )

    # --- M2: Построение таблицы эмиссии 3D-фантома (np.meshgrid vs разреженный flatnonzero) ---
    dist_3d = workload.source_node.distribution
    voxel_sz = float(workload.source_node.voxel_size)
    size_vec = np.asarray(dist_3d.shape, dtype=Float) * voxel_sz
    n_repeats_m2 = 20

    t0 = time.perf_counter()
    for _ in range(n_repeats_m2):
        gx, gy, gz = np.meshgrid(
            np.linspace(0.0, size_vec[0], dist_3d.shape[0], endpoint=False),
            np.linspace(0.0, size_vec[1], dist_3d.shape[1], endpoint=False),
            np.linspace(0.0, size_vec[2], dist_3d.shape[2], endpoint=False),
            indexing="ij",
        )
        pos_full = np.stack((gx, gy, gz), axis=3).reshape(-1, 3) - size_vec / 2.0
        prob_flat = dist_3d.ravel()
        nz_idx = prob_flat.nonzero()[0]
        _ = pos_full[nz_idx]
    t_m2_curr = (time.perf_counter() - t0) / n_repeats_m2

    t0 = time.perf_counter()
    for _ in range(n_repeats_m2):
        prob_flat = dist_3d.ravel()
        nz_idx = np.flatnonzero(prob_flat)
        ix, iy, iz = np.unravel_index(nz_idx, dist_3d.shape)
        half_sz = size_vec / 2.0
        pos_sparse = np.column_stack(
            (
                ix * voxel_sz - half_sz[0],
                iy * voxel_sz - half_sz[1],
                iz * voxel_sz - half_sz[2],
            )
        )
    t_m2_alt = max((time.perf_counter() - t0) / n_repeats_m2, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M2_EmissionTableInit",
            title="Инициализация таблицы эмиссии 3D-фантома 128x128x92",
            operations_count=int(dist_3d.size),
            current_label="3D np.meshgrid + stack",
            alternative_label="Разреженный flatnonzero + unravel",
            current_time=t_m2_curr,
            alternative_time=t_m2_alt,
            current_ns_per_op=t_m2_curr * 1e9 / dist_3d.size,
            alternative_ns_per_op=t_m2_alt * 1e9 / dist_3d.size,
            speedup_factor=t_m2_curr / t_m2_alt,
        )
    )

    # --- M3: Генерация позиций первичных частиц в воксельном фантоме (rng.choice vs Vose's Alias) ---
    n_samples_m3 = 500_000
    voxel_origins = np.ascontiguousarray(workload.source_node.emission_table[0], dtype=np.float64)
    voxel_probs = np.ascontiguousarray(workload.source_node.emission_table[1], dtype=np.float64)
    prob_table, alias_table = _build_vose_alias_tables(voxel_probs)

    out_x = np.empty(n_samples_m3, dtype=np.float64)
    out_y = np.empty(n_samples_m3, dtype=np.float64)
    out_z = np.empty(n_samples_m3, dtype=np.float64)

    # Прогрев JIT для Воза
    u1_warm = rng.random(100)
    u2_warm = rng.random(100)
    jit_warm = rng.random((100, 3))
    _sample_vose_alias_positions_kernel(
        voxel_origins, prob_table, alias_table, u1_warm, u2_warm, jit_warm, voxel_sz, out_x[:100], out_y[:100], out_z[:100]
    )

    t0 = time.perf_counter()
    _ = workload.source_node.generate_position(n_samples_m3)
    t_m3_curr = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    u1_arr = rng.random(n_samples_m3)
    u2_arr = rng.random(n_samples_m3)
    jitter_arr = rng.random((n_samples_m3, 3))
    _sample_vose_alias_positions_kernel(
        voxel_origins, prob_table, alias_table, u1_arr, u2_arr, jitter_arr, voxel_sz, out_x, out_y, out_z
    )
    t_m3_alt = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M3_VoxelPositionSampling",
            title="Генерация 500 тыс. позиций частиц в воксельном фантоме",
            operations_count=n_samples_m3,
            current_label="rng.choice(origins, p=prob)",
            alternative_label="Vose's Alias O(1) + Direct SoA",
            current_time=t_m3_curr,
            alternative_time=t_m3_alt,
            current_ns_per_op=t_m3_curr * 1e9 / n_samples_m3,
            alternative_ns_per_op=t_m3_alt * 1e9 / n_samples_m3,
            speedup_factor=t_m3_curr / t_m3_alt,
        )
    )

    # --- M4: Маппинг объемов в SensitiveVolumeHandler (Python loop + dtype=object vs Integer LUT) ---
    n_events_m4 = 200_000
    volume_ids_sample = rng.integers(0, 12, size=n_events_m4, dtype=np.int32)
    name_map = {idx: (f"Detector_{idx}" if idx in (5, 9) else "Simulation_volume") for idx in range(12)}
    detector_names = ("Detector_5", "Detector_9")

    t0 = time.perf_counter()
    mapped_names_list = [name_map.get(int(vid)) for vid in volume_ids_sample]
    root_vol_names = np.array(mapped_names_list, dtype=object)
    for det_name in detector_names:
        _ = root_vol_names == det_name
    t_m4_curr = max(time.perf_counter() - t0, 1e-9)

    lut_group = np.zeros(16, dtype=np.int32)
    lut_group[5] = 1
    lut_group[9] = 2
    t0 = time.perf_counter()
    mapped_groups = lut_group[volume_ids_sample]
    for group_code in (1, 2):
        _ = mapped_groups == group_code
    t_m4_alt = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M4_SensitiveVolumeMapping",
            title="Маппинг идентификаторов объемов в SensitiveVolumeHandler (200 тыс. событий)",
            operations_count=n_events_m4,
            current_label="Python for-loop + dtype=object",
            alternative_label="Векторизованный целочисленный LUT",
            current_time=t_m4_curr,
            alternative_time=t_m4_alt,
            current_ns_per_op=t_m4_curr * 1e9 / n_events_m4,
            alternative_ns_per_op=t_m4_alt * 1e9 / n_events_m4,
            speedup_factor=t_m4_curr / t_m4_alt,
        )
    )

    # --- M5: Фильтрация историй в HistoryAssemblerHandler (set + 4x np.isin vs Boolean Bitmap LUT) ---
    n_total_ids = 200_000
    init_pids = np.arange(n_total_ids, dtype=np.int64)
    inter_pids = rng.integers(0, n_total_ids, size=450_000, dtype=np.int64)
    dead_ids = np.arange(0, 150_000, dtype=np.int64)
    scored_set = set(rng.integers(0, n_total_ids, size=15_000, dtype=np.int64).tolist())

    t0 = time.perf_counter()
    scored_dead = np.intersect1d(dead_ids, list(scored_set))
    _ = np.isin(init_pids, scored_dead)
    _ = ~np.isin(init_pids, dead_ids)
    _ = np.isin(inter_pids, scored_dead)
    _ = ~np.isin(inter_pids, dead_ids)
    t_m5_curr = max(time.perf_counter() - t0, 1e-9)

    scored_bitmap = np.zeros(n_total_ids + 1, dtype=np.bool_)
    for s_id in scored_set:
        scored_bitmap[s_id] = True
    dead_bitmap = np.zeros(n_total_ids + 1, dtype=np.bool_)

    t0 = time.perf_counter()
    dead_bitmap[dead_ids] = True
    scored_dead_bitmap = dead_bitmap & scored_bitmap
    _ = scored_dead_bitmap[init_pids]
    _ = ~dead_bitmap[init_pids]
    _ = scored_dead_bitmap[inter_pids]
    _ = ~dead_bitmap[inter_pids]
    t_m5_alt = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M5_HistoryAssemblerFiltering",
            title="Фильтрация историй по particle_ID в HistoryAssemblerHandler (450 тыс. взаимод.)",
            operations_count=int(inter_pids.size),
            current_label="intersect1d + 4x np.isin",
            alternative_label="Прямая булева индексация O(1)",
            current_time=t_m5_curr,
            alternative_time=t_m5_alt,
            current_ns_per_op=t_m5_curr * 1e9 / inter_pids.size,
            alternative_ns_per_op=t_m5_alt * 1e9 / inter_pids.size,
            speedup_factor=t_m5_curr / t_m5_alt,
        )
    )

    # --- M6: Геометрические примитивы (_box_intersect vs _hex_prism_intersect) ---
    n_rays_m6 = 5_000_000
    ray_pos = rng.uniform(-100.0, 100.0, size=(n_rays_m6, 3)).astype(Float)
    ray_dir = rng.normal(size=(n_rays_m6, 3)).astype(Float)
    norms = np.linalg.norm(ray_dir, axis=1, keepdims=True)
    ray_dir = (ray_dir / np.maximum(norms, 1e-12)).astype(Float)

    # Извлекаем shape_data PeriodicHexPrism из скомпилированной сцены Direct
    geom_buf = GeometryCompiler().compile_scene(workload.root_scene)
    hex_entries = geom_buf[geom_buf["shape_data"]["shape"] == 1]
    hex_shape_arr = np.ascontiguousarray(hex_entries["shape_data"][:1])

    _bench_box_intersect_loop(ray_pos[:1000], ray_dir[:1000])
    _bench_hex_prism_intersect_loop(ray_pos[:1000], ray_dir[:1000], hex_shape_arr)

    t0 = time.perf_counter()
    _bench_box_intersect_loop(ray_pos, ray_dir)
    t_m6_box = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    _bench_hex_prism_intersect_loop(ray_pos, ray_dir, hex_shape_arr)
    t_m6_hex = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M6_GeometryPrimitives",
            title="Аналитическое пересечение примитивов (5 млн лучей)",
            operations_count=n_rays_m6,
            current_label="_hex_prism_intersect (Voronoi 6-slab)",
            alternative_label="_box_intersect (AABB 3-slab)",
            current_time=t_m6_hex,
            alternative_time=t_m6_box,
            current_ns_per_op=t_m6_hex * 1e9 / n_rays_m6,
            alternative_ns_per_op=t_m6_box * 1e9 / n_rays_m6,
            speedup_factor=t_m6_hex / t_m6_box,
        )
    )

    # --- M7: Удельное время 1 шага геометрии+Вудкока против физического процесса на 1 частицу ---
    warmup_propagator_instance(propagator, workload.root_scene, rng)
    n_particles_m7 = 100_000
    m7_bank = ParticleBank.allocate(n_particles_m7)
    m7_data_buf = SimulationDataBuffer.allocate(n_particles_m7 * 2, n_particles_m7 * 2, n_particles_m7 * 2, n_particles_m7 * 2)
    m7_rng_ctx = RNGContext.from_numpy_rng(rng)
    propagator.transport_buffer = TransportBuffer.allocate(n_particles_m7, len(propagator.processes))

    pos_m7 = workload.source_node.generate_position(n_particles_m7)
    dir_m7 = workload.source_node.generate_direction(n_particles_m7)
    en_m7 = workload.source_node.generate_energy(n_particles_m7)
    m7_bank.inject_particles(
        species=np.zeros(n_particles_m7, dtype=Species),
        position=Vector3DSoA(
            x=pos_m7[:, 0].astype(Length),
            y=pos_m7[:, 1].astype(Length),
            z=pos_m7[:, 2].astype(Length),
        ),
        direction=Vector3DSoA(
            x=dir_m7[:, 0].astype(Float),
            y=dir_m7[:, 1].astype(Float),
            z=dir_m7[:, 2].astype(Float),
        ),
        energy=en_m7,
        emission_time=np.zeros(n_particles_m7, dtype=Float),
        distance_traveled=np.zeros(n_particles_m7, dtype=Length),
    )
    active_m7 = m7_bank.active_indices

    t0 = time.perf_counter()
    cast_path_kernel(
        m7_bank.state.position,
        m7_bank.state.direction,
        active_m7,
        geom_buf,
        m7_bank.navigation_state,
    )
    propagator._transport_kernel(
        m7_bank.state,
        m7_bank.navigation_state,
        active_m7,
        physics_buffer,
        propagator.transport_buffer,
        geom_buf,
        m7_rng_ctx,
    )
    t_m7_ray_and_woodcock = max(time.perf_counter() - t0, 1e-9)

    compton_proc = next(p for p in propagator.processes if "Compton" in p.__class__.__name__)
    m7_mat_ids = np.ones(n_particles_m7, dtype=Index)
    m7_bank.state.energy[:n_particles_m7] = Float(0.1405)

    t0 = time.perf_counter()
    compton_proc.apply(
        m7_bank,
        active_m7,
        m7_data_buf.interactions,
        physics_buffer,
        m7_mat_ids,
        m7_rng_ctx,
    )
    t_m7_compton_proc = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M7_TransportVsProcessKernel",
            title="1 шаг Raycast+Woodcock против 1 шага ComptonScattering (100 тыс. частиц)",
            operations_count=n_particles_m7,
            current_label="cast_path + _transport_kernel",
            alternative_label="_compton_kernel + rotate",
            current_time=t_m7_ray_and_woodcock,
            alternative_time=t_m7_compton_proc,
            current_ns_per_op=t_m7_ray_and_woodcock * 1e9 / n_particles_m7,
            alternative_ns_per_op=t_m7_compton_proc * 1e9 / n_particles_m7,
            speedup_factor=t_m7_ray_and_woodcock / t_m7_compton_proc,
        )
    )

    return records


__all__ = [
    "MicroBenchmarkRecord",
    "run_micro_benchmarks_suite",
]
