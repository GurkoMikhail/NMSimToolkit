"""
Модуль Numba-ядер переноса частиц методом дельта-трекинга (Woodcock tracking)
и регистрации начальных состояний взаимодействующих частиц.
"""

from typing import Any

from llvmlite import ir
from numba import njit
from numba.core import types
from numba.extending import intrinsic
import numpy as np
from numpy.typing import NDArray

from core.geometry.geometry_kernels import _transform_to_local
from core.geometry.navigation_state import NavigationState
from core.other.typing_definitions import Float, Index
from core.particles.initial_state import InitialState
from core.particles.kinematic_state import KinematicState
from core.particles.particles_kernels import _invalidate_navigation_state, _move_particle
from core.physics.interaction_buffers import (
    DeadParticlesBuffer,
    EscapedParticlesBuffer,
    InitialStateBuffer,
    RNGContext,
)
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_kernels import _get_macroscopic_cross_sections
from core.transport.transport_buffer import TransportBuffer

BOUNDARY_PUSH_EPSILON: Float = Float(1e-6)


def _call_cfunc_ptr_impl(typingctx: Any, ptr: Any, pos_x: Any, pos_y: Any, pos_z: Any) -> Any:
    """Низкоуровневый LLVM-интринсик для прямого вызова скомпилированной @cfunc по целочисленному указателю."""
    sig = types.int64(ptr, pos_x, pos_y, pos_z)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        ptr_val, x_val, y_val, z_val = args
        fnty = ir.FunctionType(ir.IntType(64), [ir.DoubleType(), ir.DoubleType(), ir.DoubleType()])
        fnptr = builder.inttoptr(ptr_val, fnty.as_pointer())
        return builder.call(fnptr, [x_val, y_val, z_val])

    return sig, codegen


call_cfunc_ptr: Any = intrinsic(_call_cfunc_ptr_impl)


@njit(cache=True, inline='always')
def _get_random_double(rng_ctx: RNGContext) -> Float:
    """Возвращает равномерно распределенное случайное число на [0, 1) из контекста CFFI ГСЧ."""
    return Float(rng_ctx.next_double(rng_ctx.state_addr))


@njit(cache=True, inline='always')
def _generate_free_path(majorant_lac: Float, rng_ctx: RNGContext) -> Float:
    """Генерирует длину свободного пробега частицы по мажорантному линейному коэффициенту ослабления."""
    if majorant_lac <= 0.0:
        return Float(np.inf)
    rand_u = max(_get_random_double(rng_ctx), 1e-300)
    return Float(-np.log(rand_u) / majorant_lac)


@njit(cache=True, inline='always')
def _is_uncollided(
    initial_state: InitialState,
    state: KinematicState,
    p_idx: Index,
) -> bool:
    """Проверяет, сохранила ли частица начальную энергию и направление (не испытывала взаимодействий)."""
    return bool(
        state.energy[p_idx] == initial_state.emission_energy[p_idx]
        and state.direction.x[p_idx] == initial_state.emission_direction.x[p_idx]
        and state.direction.y[p_idx] == initial_state.emission_direction.y[p_idx]
        and state.direction.z[p_idx] == initial_state.emission_direction.z[p_idx]
    )


@njit(cache=True, inline='always')
def _write_single_initial_state(
    initial_state: InitialState,
    p_idx: Index,
    initial_state_buffer: InitialStateBuffer,
) -> None:
    """Записывает начальное состояние одной частицы по индексу слота банка в InitialStateBuffer."""
    idx = initial_state_buffer.cursor[0]
    initial_state_buffer.cursor[0] = idx + 1
    if idx >= initial_state_buffer.capacity:
        return

    initial_state_buffer.particle_ID[idx] = initial_state.ID[p_idx]
    initial_state_buffer.emission_time[idx] = initial_state.emission_time[p_idx]
    initial_state_buffer.emission_energy[idx] = initial_state.emission_energy[p_idx]

    initial_state_buffer.emission_position.x[idx] = initial_state.emission_position.x[p_idx]
    initial_state_buffer.emission_position.y[idx] = initial_state.emission_position.y[p_idx]
    initial_state_buffer.emission_position.z[idx] = initial_state.emission_position.z[p_idx]

    initial_state_buffer.emission_direction.x[idx] = initial_state.emission_direction.x[p_idx]
    initial_state_buffer.emission_direction.y[idx] = initial_state.emission_direction.y[p_idx]
    initial_state_buffer.emission_direction.z[idx] = initial_state.emission_direction.z[p_idx]


@njit(cache=True)
def _push_to_initial_state_kernel(
    initial_state: InitialState,
    target_indices: NDArray[Index],
    initial_state_buffer: InitialStateBuffer,
) -> None:
    """Записывает начальные состояния выбранных частиц в буфер с курсором InitialStateBuffer."""
    if initial_state_buffer.capacity <= 0:
        return

    for j in range(target_indices.shape[0]):
        _write_single_initial_state(initial_state, target_indices[j], initial_state_buffer)


@njit(cache=True)
def _push_first_interacting_to_initial_state_kernel(
    initial_state: InitialState,
    state: KinematicState,
    active_indices: NDArray[Index],
    process_ids: NDArray[Index],
    initial_state_buffer: InitialStateBuffer,
) -> None:
    """
    Однопроходная безаллокационная запись в InitialStateBuffer начальных состояний активных частиц,
    которые на текущем шаге впервые вступают во взаимодействие.
    """
    if initial_state_buffer.capacity <= 0:
        return

    for j in range(active_indices.shape[0]):
        p_idx = active_indices[j]
        if process_ids[p_idx] != -1 and _is_uncollided(initial_state, state, p_idx):
            _write_single_initial_state(initial_state, p_idx, initial_state_buffer)


@njit(cache=True)
def _push_uncollided_to_initial_state_kernel(
    initial_state: InitialState,
    state: KinematicState,
    candidate_indices: NDArray[Index],
    initial_state_buffer: InitialStateBuffer,
) -> None:
    """
    Однопроходная безаллокационная запись в InitialStateBuffer начальных состояний частиц
    из массива candidate_indices, ни разу не испытавших взаимодействия.
    """
    if initial_state_buffer.capacity <= 0:
        return

    for j in range(candidate_indices.shape[0]):
        p_idx = candidate_indices[j]
        if _is_uncollided(initial_state, state, p_idx):
            _write_single_initial_state(initial_state, p_idx, initial_state_buffer)


@njit(cache=True)
def _push_to_dead_particles_kernel(
    initial_state: InitialState,
    target_indices: NDArray[Index],
    dead_buffer: DeadParticlesBuffer,
) -> None:
    """Записывает идентификаторы завершивших трекинг частиц по индексам слотов в DeadParticlesBuffer."""
    if dead_buffer.capacity <= 0:
        return

    for j in range(target_indices.shape[0]):
        idx = dead_buffer.cursor[0]
        dead_buffer.cursor[0] = idx + 1
        if idx >= dead_buffer.capacity:
            continue
        p_idx = target_indices[j]
        dead_buffer.particle_ID[idx] = initial_state.ID[p_idx]


@njit(cache=True)
def _push_to_escaped_particles_kernel(
    initial_state: InitialState,
    state: KinematicState,
    target_indices: NDArray[Index],
    escaped_buffer: EscapedParticlesBuffer,
) -> None:
    """Записывает идентификаторы и координаты вылетевших за пределы сцены частиц в EscapedParticlesBuffer."""
    if escaped_buffer.capacity <= 0:
        return

    for j in range(target_indices.shape[0]):
        idx = escaped_buffer.cursor[0]
        escaped_buffer.cursor[0] = idx + 1
        if idx >= escaped_buffer.capacity:
            continue
        p_idx = target_indices[j]
        escaped_buffer.particle_ID[idx] = initial_state.ID[p_idx]
        escaped_buffer.position.x[idx] = state.position.x[p_idx]
        escaped_buffer.position.y[idx] = state.position.y[p_idx]
        escaped_buffer.position.z[idx] = state.position.z[p_idx]


def make_transport_kernel(mapped_process_ids: NDArray[Index]) -> Any:
    """Создает скомпилированное Numba-ядро шага транспорта с запеченными кодами процессов."""
    num_processes = mapped_process_ids.shape[0]

    @njit(inline='always')
    def _sample_process_id(majorant_lac: Float, process_lacs: NDArray[Float], rng_ctx: RNGContext) -> Index:
        rnd = _get_random_double(rng_ctx) * majorant_lac
        p0 = 0.0
        for i in range(num_processes):
            p1 = p0 + process_lacs[i]
            if p0 <= rnd < p1:
                return Index(mapped_process_ids[i])
            p0 = p1
        return Index(-1)

    @njit
    def transport_kernel(
        state: KinematicState,
        nav_state: NavigationState,
        target_indices: NDArray[Index],
        physics_buffer: PhysicsBuffer,
        transport_buffer: TransportBuffer,
        geom_buffer: NDArray[Any],
        rng_ctx: RNGContext,
    ) -> None:
        """Выполняет шаг дельта-трекинга ВудКока для всех активных частиц без аллокаций памяти."""
        num_particles = target_indices.shape[0]
        process_ids = transport_buffer.process_ids
        material_ids = transport_buffer.material_ids
        process_lacs = transport_buffer.process_lacs[:num_processes]

        for j in range(num_particles):
            p_idx = target_indices[j]

            current_vol = nav_state.current_volume[p_idx]
            if current_vol < 0:
                process_ids[p_idx] = -1
                continue

            majorant_mat_id = physics_buffer.majorant_material_map[current_vol]
            cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]

            _get_macroscopic_cross_sections(
                state.energy[p_idx], majorant_mat_id, physics_buffer.material_bank, process_lacs
            )

            majorant_lac = Float(0.0)
            for i in range(num_processes):
                majorant_lac = Float(majorant_lac + process_lacs[i])

            free_path = _generate_free_path(majorant_lac, rng_ctx)

            while free_path < nav_state.boundary_distance[p_idx]:
                _move_particle(state, p_idx, free_path)
                nav_state.boundary_distance[p_idx] -= free_path

                if cfunc_addr != 0:
                    vol_transform = geom_buffer[current_vol]['transform']
                    loc_x, loc_y, loc_z, _, _, _ = _transform_to_local(
                        state.position.x[p_idx],
                        state.position.y[p_idx],
                        state.position.z[p_idx],
                        Float(0.0),
                        Float(0.0),
                        Float(0.0),
                        vol_transform,
                    )
                    mat_id: Index = call_cfunc_ptr(cfunc_addr, loc_x, loc_y, loc_z)
                    _get_macroscopic_cross_sections(
                        state.energy[p_idx], mat_id, physics_buffer.material_bank, process_lacs
                    )
                else:
                    mat_id = majorant_mat_id

                selected_process = _sample_process_id(majorant_lac, process_lacs, rng_ctx)

                if selected_process != -1:
                    material_ids[p_idx] = mat_id
                    process_ids[p_idx] = selected_process
                    break

                # Фиктивное взаимодействие (дельта-рассеяние ВудКока)
                free_path = _generate_free_path(majorant_lac, rng_ctx)

            else:
                # Достижение границы геометрического объема
                shift = Float(nav_state.boundary_distance[p_idx] + BOUNDARY_PUSH_EPSILON)
                _move_particle(state, p_idx, shift)

                _invalidate_navigation_state(nav_state, p_idx)
                process_ids[p_idx] = -1

    return transport_kernel


__all__ = [
    "BOUNDARY_PUSH_EPSILON",
    "call_cfunc_ptr",
    "_get_random_double",
    "_generate_free_path",
    "_is_uncollided",
    "_push_to_initial_state_kernel",
    "_push_first_interacting_to_initial_state_kernel",
    "_push_uncollided_to_initial_state_kernel",
    "_push_to_dead_particles_kernel",
    "_push_to_escaped_particles_kernel",
    "make_transport_kernel",
]

