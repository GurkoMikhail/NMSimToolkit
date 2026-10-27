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
from typing import List

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.geometry.flattened_scene import FlattenedScene
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.geometry_kernels import _box_intersect, _hex_prism_intersect, cast_path_kernel
from core.materials.material_bank import MaterialBank
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext, SimulationDataBuffer
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import (
    _compute_log_energy_and_bin,
    _get_macroscopic_cross_sections,
)
from core.scene.nodes import CompositeNode
from core.source.source_buffer import SourceBuffer
from core.source.source_compiler import SourceCompiler
from core.source.sources import VoxelDistribution
from core.source.source_kernels import (
    _sample_alias,
    _sample_voxel_position,
    inject_poisson_sources_kernel,
)
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


@njit(cache=True, inline='always')
def _reference_searchsorted_cross_sections(
    energy: Float,
    material_id: Index,
    material_bank: MaterialBank,
    out_lacs_slice: NDArray[Float],
) -> None:
    """
    Эталонная реализация поиска интервала через np.searchsorted(..., side='right')
    и кусочной log-log интерполяции на реальном MaterialBank (база до Фазы 3).
    """
    mat_pointers = material_bank.mat_pointers
    grid = material_bank.physics_energy_grid
    lacs = material_bank.physics_lac_table

    pointer = mat_pointers[material_id]
    start_idx = Index(pointer['start_idx'])
    length = Index(pointer['length'])
    num_procs = min(len(out_lacs_slice), lacs.shape[1])

    if length <= 0:
        for proc_idx in range(len(out_lacs_slice)):
            out_lacs_slice[proc_idx] = 0.0
    else:
        mat_energy_grid = grid[start_idx : start_idx + length]
        if energy < mat_energy_grid[0]:
            for proc_idx in range(num_procs):
                out_lacs_slice[proc_idx] = lacs[start_idx, proc_idx]
        elif energy >= mat_energy_grid[length - 1]:
            for proc_idx in range(num_procs):
                out_lacs_slice[proc_idx] = lacs[start_idx + length - 1, proc_idx]
        else:
            idx = np.searchsorted(mat_energy_grid, energy, side='right')
            if idx <= 0:
                for proc_idx in range(num_procs):
                    out_lacs_slice[proc_idx] = lacs[start_idx, proc_idx]
            elif idx >= length:
                for proc_idx in range(num_procs):
                    out_lacs_slice[proc_idx] = lacs[start_idx + length - 1, proc_idx]
            else:
                row0 = start_idx + idx - 1
                row1 = start_idx + idx
                energy_low = mat_energy_grid[idx - 1]
                energy_high = mat_energy_grid[idx]

                if energy <= energy_low or energy_high <= energy_low:
                    for proc_idx in range(num_procs):
                        out_lacs_slice[proc_idx] = lacs[row0, proc_idx]
                else:
                    lin_fraction = (energy - energy_low) / (energy_high - energy_low)
                    log_fraction = (
                        (math.log(energy) - math.log(energy_low))
                        / (math.log(energy_high) - math.log(energy_low))
                        if energy_low > 0.0
                        else 0.0
                    )

                    for proc_idx in range(num_procs):
                        val_low = lacs[row0, proc_idx]
                        val_high = lacs[row1, proc_idx]
                        if val_low > 0.0 and val_high > 0.0 and energy_low > 0.0:
                            log_val_low = math.log(val_low)
                            log_val_high = math.log(val_high)
                            out_lacs_slice[proc_idx] = math.exp(
                                log_val_low + log_fraction * (log_val_high - log_val_low)
                            )
                        else:
                            out_lacs_slice[proc_idx] = val_low + lin_fraction * (val_high - val_low)


@njit(cache=True)
def _bench_binary_search_cross_sections_loop(
    energies: NDArray[Float],
    mat_ids: NDArray[Index],
    material_bank: MaterialBank,
) -> Float:
    """Пакетный вызов эталонного бинарного поиска np.searchsorted на реальном MaterialBank."""
    out_lacs = np.zeros(3, dtype=Float)
    acc = Float(0.0)
    for idx in range(energies.shape[0]):
        _reference_searchsorted_cross_sections(energies[idx], mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
    return acc


@njit(cache=True)
def _bench_two_level_cross_sections_loop(
    energies: NDArray[Float],
    mat_ids: NDArray[Index],
    material_bank: MaterialBank,
) -> Float:
    """
    Пакетный вызов реального двухуровневого лог-индекса MaterialBank
    (_compute_log_energy_and_bin + _get_macroscopic_cross_sections) без внутренних NRT_incref.
    """
    out_lacs = np.zeros(3, dtype=Float)
    acc = Float(0.0)
    for idx in range(energies.shape[0]):
        energy = energies[idx]
        log_e, log_bin = _compute_log_energy_and_bin(energy, material_bank)
        _get_macroscopic_cross_sections(energy, log_e, log_bin, mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
    return acc


@njit(cache=True)
def _bench_woodcock_pair_binary_search_loop(
    energies: NDArray[Float],
    maj_mat_ids: NDArray[Index],
    vox_mat_ids: NDArray[Index],
    material_bank: MaterialBank,
) -> Float:
    """Пакетный вызов пары запросов Вудкока (мажоранта + материал вокселя) через бинарный поиск np.searchsorted."""
    out_lacs = np.zeros(3, dtype=Float)
    acc = Float(0.0)
    for idx in range(energies.shape[0]):
        energy = energies[idx]
        _reference_searchsorted_cross_sections(energy, maj_mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
        _reference_searchsorted_cross_sections(energy, vox_mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
    return acc


@njit(cache=True)
def _bench_woodcock_pair_two_level_loop(
    energies: NDArray[Float],
    maj_mat_ids: NDArray[Index],
    vox_mat_ids: NDArray[Index],
    material_bank: MaterialBank,
) -> Float:
    """
    Пакетный вызов пары запросов Вудкока (мажоранта + материал вокселя) через реальный
    двухуровневый лог-индекс MaterialBank с однократным расчетом (log_e, log_bin) на шаг частицы.
    """
    out_lacs = np.zeros(3, dtype=Float)
    acc = Float(0.0)
    for idx in range(energies.shape[0]):
        energy = energies[idx]
        log_e, log_bin = _compute_log_energy_and_bin(energy, material_bank)
        _get_macroscopic_cross_sections(energy, log_e, log_bin, maj_mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
        _get_macroscopic_cross_sections(energy, log_e, log_bin, vox_mat_ids[idx], material_bank, out_lacs)
        acc = Float(acc + out_lacs[0] + out_lacs[1] + out_lacs[2])
    return acc


def _extract_scene_unique_materials(scene_root: CompositeNode) -> List[Material]:
    """Извлекает уникальные материалы из всех объемов дерева сцены."""
    flat_list = FlattenedScene(scene_root).flat_list
    all_materials: List[Material] = []
    for vol, _, _ in flat_list:
        all_materials.extend(vol.material_list)
    unique_materials: List[Material] = []
    seen_ids = set()
    for mat in all_materials:
        if mat.ID not in seen_ids:
            seen_ids.add(mat.ID)
            unique_materials.append(mat)
    return unique_materials


@njit(cache=True)
def _bench_source_buffer_voxel_positions_kernel(
    source_buffer: SourceBuffer,
    source_idx: Index,
    num_samples: int,
    rng_context: RNGContext,
    out_x: NDArray[Float],
    out_y: NDArray[Float],
    out_z: NDArray[Float],
) -> None:
    """Генерирует глобальные координаты в SoA-массивы за O(1) реальными ядрами SourceBuffer без аллокаций."""
    vox_start = source_buffer.voxel_range[source_idx, 0]
    vox_end = source_buffer.voxel_range[source_idx, 1]
    vox_count = vox_end - vox_start

    voxel_index = source_buffer.voxel_index
    voxel_prob = source_buffer.voxel_prob
    voxel_alias = source_buffer.voxel_alias

    ny = source_buffer.grid_shape[source_idx, 1]
    nz = source_buffer.grid_shape[source_idx, 2]
    voxel_sz = source_buffer.voxel_size[source_idx]
    half_x = source_buffer.half_size[source_idx, 0]
    half_y = source_buffer.half_size[source_idx, 1]
    half_z = source_buffer.half_size[source_idx, 2]
    jitter_x = source_buffer.cell_jitter[source_idx, 0]
    jitter_y = source_buffer.cell_jitter[source_idx, 1]
    local_to_global = source_buffer.local_to_global

    for sample_idx in range(num_samples):
        chosen_vox = _sample_alias(voxel_prob, voxel_alias, vox_start, vox_count, rng_context)
        flat_idx = Index(voxel_index[chosen_vox])
        pos_x, pos_y, pos_z = _sample_voxel_position(
            flat_idx,
            ny,
            nz,
            voxel_sz,
            half_x,
            half_y,
            half_z,
            local_to_global,
            source_idx,
            rng_context,
            jitter_x,
            jitter_y,
        )
        out_x[sample_idx] = pos_x
        out_y[sample_idx] = pos_y
        out_z[sample_idx] = pos_z


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

    # --- M1: Интерполяция макроскопических сечений на реальном MaterialBank (np.searchsorted vs 2-ур. лог-индекс) ---
    n_ops_m1 = 5_000_000
    test_energies = rng.uniform(0.015, 0.300, size=n_ops_m1).astype(Float)
    unique_mats = _extract_scene_unique_materials(workload.root_scene)
    valid_mat_ids = np.array([int(mat.ID) for mat in unique_mats], dtype=Index)
    test_mat_ids = rng.choice(valid_mat_ids, size=n_ops_m1).astype(Index)
    test_vox_mat_ids = rng.choice(valid_mat_ids, size=n_ops_m1).astype(Index)
    curr_avg_nodes = float(np.mean(material_bank.mat_pointers[valid_mat_ids]["length"]))

    # Прогрев JIT-ядер M1 на реальном MaterialBank
    _bench_binary_search_cross_sections_loop(test_energies[:1000], test_mat_ids[:1000], material_bank)
    _bench_two_level_cross_sections_loop(test_energies[:1000], test_mat_ids[:1000], material_bank)
    _bench_woodcock_pair_binary_search_loop(
        test_energies[:1000], test_mat_ids[:1000], test_vox_mat_ids[:1000], material_bank
    )
    _bench_woodcock_pair_two_level_loop(
        test_energies[:1000], test_mat_ids[:1000], test_vox_mat_ids[:1000], material_bank
    )

    t0 = time.perf_counter()
    _bench_binary_search_cross_sections_loop(test_energies, test_mat_ids, material_bank)
    t_m1_bin_single = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    _bench_two_level_cross_sections_loop(test_energies, test_mat_ids, material_bank)
    t_m1_log_single = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    _bench_woodcock_pair_binary_search_loop(test_energies, test_mat_ids, test_vox_mat_ids, material_bank)
    t_m1_bin_pair = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    _bench_woodcock_pair_two_level_loop(test_energies, test_mat_ids, test_vox_mat_ids, material_bank)
    t_m1_log_pair = max(time.perf_counter() - t0, 1e-9)

    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M1a_LogIndexSingle",
            title=f"Интерполяция LAC в MaterialBank (~{curr_avg_nodes:.0f} узлов, одиночный запрос, 5 млн зап.)",
            operations_count=n_ops_m1,
            current_label=f"np.searchsorted O(log K) (~{curr_avg_nodes:.0f} узлов)",
            alternative_label=f"2-ур. лог-индекс O(1) ({material_bank.lookup_num_bins} корзин)",
            current_time=t_m1_bin_single,
            alternative_time=t_m1_log_single,
            current_ns_per_op=t_m1_bin_single * 1e9 / n_ops_m1,
            alternative_ns_per_op=t_m1_log_single * 1e9 / n_ops_m1,
            speedup_factor=t_m1_bin_single / t_m1_log_single,
        )
    )
    records.append(
        MicroBenchmarkRecord(
            benchmark_id="M1b_LogIndexWoodcockPair",
            title=f"Пара запросов Вудкока (мажоранта + воксель, общий log(E), ~{curr_avg_nodes:.0f} узлов, 5 млн шагов)",
            operations_count=n_ops_m1,
            current_label="2x np.searchsorted + 2x log(E)",
            alternative_label="1x log(E) + 2x _locate_row (MaterialBank)",
            current_time=t_m1_bin_pair,
            alternative_time=t_m1_log_pair,
            current_ns_per_op=t_m1_bin_pair * 1e9 / n_ops_m1,
            alternative_ns_per_op=t_m1_log_pair * 1e9 / n_ops_m1,
            speedup_factor=t_m1_bin_pair / t_m1_log_pair,
        )
    )

    # --- M2: Построение таблицы эмиссии 3D-фантома (np.meshgrid vs разреженный flatnonzero) ---
    spatial_vox = workload.source_node.spatial
    assert isinstance(spatial_vox, VoxelDistribution)
    dist_3d = spatial_vox.distribution
    voxel_sz = float(spatial_vox.voxel_size)
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
    active_sources = SourceCompiler().compile_scene(workload.root_scene)
    source_buffer = SourceCompiler().compile(active_sources)

    vox_indices = spatial_vox.voxel_indices
    voxel_probs = np.ascontiguousarray(spatial_vox.voxel_probabilities, dtype=np.float64)
    ix_ref, iy_ref, iz_ref = np.unravel_index(vox_indices, dist_3d.shape)
    half_sz_ref = size_vec / 2.0
    voxel_origins = np.ascontiguousarray(
        np.column_stack(
            (
                ix_ref * voxel_sz - half_sz_ref[0],
                iy_ref * voxel_sz - half_sz_ref[1],
                iz_ref * voxel_sz - half_sz_ref[2],
            )
        ),
        dtype=np.float64,
    )

    out_x = np.empty(n_samples_m3, dtype=Float)
    out_y = np.empty(n_samples_m3, dtype=Float)
    out_z = np.empty(n_samples_m3, dtype=Float)
    m3_rng_ctx = RNGContext.from_numpy_rng(rng)

    # Прогрев JIT для боевого ядра Воза
    _bench_source_buffer_voxel_positions_kernel(
        source_buffer,
        Index(0),
        100,
        m3_rng_ctx,
        out_x[:100],
        out_y[:100],
        out_z[:100],
    )

    t0 = time.perf_counter()
    chosen_origins = rng.choice(voxel_origins, size=n_samples_m3, p=voxel_probs)
    local_positions = chosen_origins + rng.uniform(
        (0.0, 0.0, 0.0),
        (voxel_sz, voxel_sz, voxel_sz),
        size=(n_samples_m3, 3),
    )
    _ = workload.source_node.convert_to_global_position(local_positions.astype(Float))
    t_m3_curr = max(time.perf_counter() - t0, 1e-9)

    t0 = time.perf_counter()
    _bench_source_buffer_voxel_positions_kernel(
        source_buffer,
        Index(0),
        n_samples_m3,
        m3_rng_ctx,
        out_x,
        out_y,
        out_z,
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

    inject_poisson_sources_kernel(
        m7_bank,
        source_buffer,
        Float(np.inf),
        Float(0.0),
        Float(0.0),
        Float(np.nan),
        m7_rng_ctx,
        Index(n_particles_m7),
        Index(0),
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
