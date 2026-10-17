"""
Модуль диагностической телеметрии счетчиков переноса частиц (TransportTelemetryCollector).

Собирает точные алгоритмические счетчики на 1 симулированную историю для понимания внутренней
структуры затрат внутри Numba-ядер cast_path_kernel и transport_kernel без модификации core/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_kernels import (
    _get_intersection,
    _transform_to_local,
)
from core.geometry.navigation_state import NavigationState
from core.other.typing_definitions import Float, Index
from core.other.vectors import Vector3DSoA
from core.particles.kinematic_state import KinematicState
from core.particles.particles_kernels import _invalidate_navigation_state, _move_particle
from core.physics.interaction_buffers import RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_kernels import _get_macroscopic_cross_sections
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from core.transport.transport_buffer import TransportBuffer
from core.transport.transport_kernels import (
    BOUNDARY_PUSH_EPSILON,
    _generate_free_path,
    _get_random_double,
    _push_first_interacting_to_initial_state_kernel,
    call_cfunc_ptr,
)
from settings.database_setting import attenuation_database
from settings.processes_settings import processes_list
from verification.profiling.workloads import CollimatorKind, ProfilingWorkload


# Индексы массива счетчиков telemetry_counters (int64[10]):
# 0: raycast_traces_count (число вызовов _trace_single_ray)
# 1: box_intersections_count (число проверок _box_intersect)
# 2: hex_prism_intersections_count (число проверок _hex_prism_intersect)
# 3: woodcock_steps_count (общее число итераций цикла while free_path < boundary_distance)
# 4: null_collisions_count (число фиктивных дельта-рассеяний Вудкока)
# 5: real_interactions_count (число реальных взаимодействий)
# 6: boundary_crossings_count (число пересечений границ объемов)
# 7: cfunc_calls_count (число вызовов @cfunc воксельного фантома или параметрического коллиматора)
# 8: cross_section_lookups_count (число вызовов _get_macroscopic_cross_sections / searchsorted)


@njit
def _telemetry_cast_path_kernel(
    positions: Vector3DSoA,
    directions: Vector3DSoA,
    target_indices: NDArray[Index],
    geom_buffer: NDArray[Any],
    nav_state: NavigationState,
    counters: NDArray[np.int64],
) -> None:
    """Диагностический аналог cast_path_kernel с подсчетом числа трассировок и пересечений примитивов."""
    num_particles = target_indices.shape[0]
    pos_x = positions.x
    pos_y = positions.y
    pos_z = positions.z
    dir_x = directions.x
    dir_y = directions.y
    dir_z = directions.z
    bound_dist = nav_state.boundary_distance
    nav_curr_vol = nav_state.current_volume
    buffer_len = geom_buffer.shape[0]

    for particle_slot in range(num_particles):
        particle_idx = target_indices[particle_slot]
        if bound_dist[particle_idx] > 0.0:
            continue

        counters[0] += 1
        world_px = pos_x[particle_idx]
        world_py = pos_y[particle_idx]
        world_pz = pos_z[particle_idx]
        world_dx = dir_x[particle_idx]
        world_dy = dir_y[particle_idx]
        world_dz = dir_z[particle_idx]

        closest_dist = Float(np.inf)
        current_vol = Index(-1)
        geom_idx = 0

        while geom_idx < buffer_len:
            geom = geom_buffer[geom_idx]
            loc_px, loc_py, loc_pz, loc_dx, loc_dy, loc_dz = _transform_to_local(
                world_px,
                world_py,
                world_pz,
                world_dx,
                world_dy,
                world_dz,
                geom["transform"],
            )
            shape_id = geom["shape_data"]["shape"]
            if shape_id == 0:
                counters[1] += 1
            elif shape_id == 1:
                counters[2] += 1

            tmin, tmax = _get_intersection(
                loc_px,
                loc_py,
                loc_pz,
                loc_dx,
                loc_dy,
                loc_dz,
                geom["shape_data"],
            )

            if tmax <= 0.0 or tmax <= tmin:
                geom_idx = int(geom["miss_index"])
            elif tmin <= 0.0:
                current_vol = Index(geom["volume_index"])
                if tmax < closest_dist:
                    closest_dist = tmax
                geom_idx += 1
            else:
                if tmin < closest_dist:
                    closest_dist = tmin
                geom_idx = int(geom["miss_index"])

        bound_dist[particle_idx] = closest_dist
        nav_curr_vol[particle_idx] = current_vol


def _make_telemetry_transport_kernel(mapped_process_ids: NDArray[Index]) -> Any:
    """Создает диагностическое ядро дельта-трекинга с подсчетом шагов Вудкока, @cfunc и searchsorted."""
    num_processes = mapped_process_ids.shape[0]

    @njit(inline="always")
    def _sample_process_id(majorant_lac: Float, process_lacs: NDArray[Float], rng_ctx: RNGContext) -> Index:
        rnd = _get_random_double(rng_ctx) * majorant_lac
        p0 = 0.0
        for proc_idx in range(num_processes):
            p1 = p0 + process_lacs[proc_idx]
            if p0 <= rnd < p1:
                return Index(mapped_process_ids[proc_idx])
            p0 = p1
        return Index(-1)

    @njit
    def _telemetry_transport_kernel(
        state: KinematicState,
        nav_state: NavigationState,
        target_indices: NDArray[Index],
        physics_buffer: PhysicsBuffer,
        transport_buffer: TransportBuffer,
        geom_buffer: NDArray[Any],
        rng_ctx: RNGContext,
        counters: NDArray[np.int64],
    ) -> None:
        num_particles = target_indices.shape[0]
        process_ids = transport_buffer.process_ids
        material_ids = transport_buffer.material_ids
        process_lacs = transport_buffer.process_lacs[:num_processes]

        for particle_slot in range(num_particles):
            p_idx = target_indices[particle_slot]
            current_vol = nav_state.current_volume[p_idx]
            if current_vol < 0:
                process_ids[p_idx] = -1
                continue

            majorant_mat_id = physics_buffer.majorant_material_map[current_vol]
            cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]

            counters[8] += 1
            _get_macroscopic_cross_sections(
                state.energy[p_idx], majorant_mat_id, physics_buffer.material_bank, process_lacs
            )

            majorant_lac = Float(0.0)
            for proc_idx in range(num_processes):
                majorant_lac = Float(majorant_lac + process_lacs[proc_idx])

            free_path = _generate_free_path(majorant_lac, rng_ctx)

            while free_path < nav_state.boundary_distance[p_idx]:
                counters[3] += 1
                _move_particle(state, p_idx, free_path)
                nav_state.boundary_distance[p_idx] -= free_path

                if cfunc_addr != 0:
                    vol_transform = geom_buffer[current_vol]["transform"]
                    loc_x, loc_y, loc_z, _, _, _ = _transform_to_local(
                        state.position.x[p_idx],
                        state.position.y[p_idx],
                        state.position.z[p_idx],
                        Float(0.0),
                        Float(0.0),
                        Float(0.0),
                        vol_transform,
                    )
                    counters[7] += 1
                    mat_id: Index = call_cfunc_ptr(cfunc_addr, loc_x, loc_y, loc_z)
                    counters[8] += 1
                    _get_macroscopic_cross_sections(
                        state.energy[p_idx], mat_id, physics_buffer.material_bank, process_lacs
                    )
                else:
                    mat_id = majorant_mat_id

                selected_process = _sample_process_id(majorant_lac, process_lacs, rng_ctx)
                if selected_process != -1:
                    counters[5] += 1
                    material_ids[p_idx] = mat_id
                    process_ids[p_idx] = selected_process
                    break

                counters[4] += 1
                free_path = _generate_free_path(majorant_lac, rng_ctx)
            else:
                counters[6] += 1
                shift = Float(nav_state.boundary_distance[p_idx] + BOUNDARY_PUSH_EPSILON)
                _move_particle(state, p_idx, shift)
                _invalidate_navigation_state(nav_state, p_idx)
                process_ids[p_idx] = -1

    return _telemetry_transport_kernel


@dataclass(frozen=True)
class TransportTelemetryMetrics:
    """Абсолютные и удельные (на 1 историю) счетчики операций переноса и геометрии."""

    workload_id: str
    title: str
    collimator_kind: CollimatorKind
    histories_count: int
    total_raycast_traces: int
    total_box_intersections: int
    total_hex_prism_intersections: int
    total_woodcock_steps: int
    total_null_collisions: int
    total_real_interactions: int
    total_boundary_crossings: int
    total_cfunc_calls: int
    total_cross_section_lookups: int

    @property
    def raycasts_per_history(self) -> float:
        """Среднее число вызовов _trace_single_ray на 1 историю."""
        return float(self.total_raycast_traces) / max(self.histories_count, 1)

    @property
    def primitive_tests_per_raycast(self) -> float:
        """Среднее число проверок примитивов (Box + HexPrism) на один луч."""
        total_prims = self.total_box_intersections + self.total_hex_prism_intersections
        return float(total_prims) / max(self.total_raycast_traces, 1)

    @property
    def woodcock_steps_per_history(self) -> float:
        """Среднее число виртуальных шагов Вудкока на 1 историю."""
        return float(self.total_woodcock_steps) / max(self.histories_count, 1)

    @property
    def null_collision_fraction_pct(self) -> float:
        """Процент холостых (фиктивных) дельта-шагов Вудкока от всех шагов внутри объемов [%]."""
        return 100.0 * float(self.total_null_collisions) / max(self.total_woodcock_steps, 1)

    @property
    def cfunc_calls_per_history(self) -> float:
        """Среднее число вызовов @cfunc на 1 историю."""
        return float(self.total_cfunc_calls) / max(self.histories_count, 1)

    @property
    def searchsorted_per_history(self) -> float:
        """Среднее число бинарных поисков np.searchsorted в таблице сечений на 1 историю."""
        return float(self.total_cross_section_lookups) / max(self.histories_count, 1)


def collect_transport_telemetry(
    workload: ProfilingWorkload,
    seed: int = 42,
    propagator: Any = None,
    telemetry_kernel: Any = None,
) -> TransportTelemetryMetrics:
    """
    Выполняет диагностический прогон рабочей нагрузки и собирает точные счетчики
    трассировки лучей, шагов Вудкока, холостых рассеяний, вызовов @cfunc и searchsorted.
    """
    workload.source_node.rng = np.random.default_rng(seed)
    if propagator is None:
        propagator_inst = ParticlePropagator(
            processes_list=processes_list,
            attenuation_database=attenuation_database,
            rng=np.random.default_rng(seed),
        )
    else:
        propagator_inst = propagator
    propagator_inst.transport_buffer = TransportBuffer.allocate(0, len(propagator_inst.processes))

    manager = SimulationManager(
        scene=workload.root_scene,
        propagator=propagator_inst,
        stop_time=workload.stop_time,
        particles_number=workload.particles_number,
        min_energy=workload.min_energy,
        buffer_capacity=max(workload.particles_number * 2, 100_000),
        seed=seed,
    )

    telemetry_transport_kernel = (
        _make_telemetry_transport_kernel(propagator_inst.process_ids)
        if telemetry_kernel is None
        else telemetry_kernel
    )
    counters = np.zeros(10, dtype=np.int64)

    while manager.has_active_particles or manager.has_pending_sources:
        manager._replenish_bank()
        active_indices = manager.bank.active_indices
        if active_indices.size == 0:
            break

        if len(active_indices) > manager.data_buffer.initial_states.remaining_capacity:
            manager.data_buffer.initial_states.flush(clear=True)
        if len(active_indices) > manager.data_buffer.interactions.remaining_capacity:
            manager.data_buffer.interactions.flush(clear=True)

        if propagator_inst.transport_buffer.process_ids.size < manager.bank.capacity:
            propagator_inst.transport_buffer = TransportBuffer.allocate(
                manager.bank.capacity, len(propagator_inst.processes)
            )

        _telemetry_cast_path_kernel(
            manager.bank.state.position,
            manager.bank.state.direction,
            active_indices,
            manager.geometry_buffer,
            manager.bank.navigation_state,
            counters,
        )

        telemetry_transport_kernel(
            manager.bank.state,
            manager.bank.navigation_state,
            active_indices,
            manager.physics_buffer,
            propagator_inst.transport_buffer,
            manager.geometry_buffer,
            manager.rng_ctx,
            counters,
        )

        _push_first_interacting_to_initial_state_kernel(
            manager.bank.initial_state,
            manager.bank.state,
            active_indices,
            propagator_inst.transport_buffer.process_ids,
            manager.data_buffer.initial_states,
        )

        for process_obj in propagator_inst.processes:
            mask = propagator_inst.transport_buffer.process_ids[active_indices] == process_obj.process_id
            target_indices = active_indices[mask]
            if target_indices.size > 0:
                process_obj.apply(
                    manager.bank,
                    target_indices,
                    manager.data_buffer.interactions,
                    manager.physics_buffer,
                    propagator_inst.transport_buffer.material_ids,
                    manager.rng_ctx,
                )

        manager._apply_invalidators(active_indices)
        manager.step_count += 1

    return TransportTelemetryMetrics(
        workload_id=workload.workload_id,
        title=workload.title,
        collimator_kind=workload.collimator_kind,
        histories_count=int(workload.particles_number),
        total_raycast_traces=int(counters[0]),
        total_box_intersections=int(counters[1]),
        total_hex_prism_intersections=int(counters[2]),
        total_woodcock_steps=int(counters[3]),
        total_null_collisions=int(counters[4]),
        total_real_interactions=int(counters[5]),
        total_boundary_crossings=int(counters[6]),
        total_cfunc_calls=int(counters[7]),
        total_cross_section_lookups=int(counters[8]),
    )


def run_transport_telemetry_suite(
    workloads: List[ProfilingWorkload],
    seed: int = 42,
) -> List[TransportTelemetryMetrics]:
    """Собирает телеметрию счетчиков переноса для списка эталонных сцен."""
    if not workloads:
        return []
    shared_propagator = ParticlePropagator(
        processes_list=processes_list,
        attenuation_database=attenuation_database,
        rng=np.random.default_rng(seed),
    )
    shared_kernel = _make_telemetry_transport_kernel(shared_propagator.process_ids)
    return [
        collect_transport_telemetry(
            workload=w_item,
            seed=seed,
            propagator=shared_propagator,
            telemetry_kernel=shared_kernel,
        )
        for w_item in workloads
    ]


__all__ = [
    "TransportTelemetryMetrics",
    "collect_transport_telemetry",
    "run_transport_telemetry_suite",
]
