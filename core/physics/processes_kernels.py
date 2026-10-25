"""
Модуль Numba-ядер физических процессов взаимодействия частиц (фотоэффект,
комптоновское и когерентное рассеяние) с O(1) лог-индексацией и записи актов взаимодействия в буфер.
"""

from typing import Any

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Charge, Energy, Float, ID, Index, ProcessID, Species
from core.particles.kinematic_state import KinematicState
from core.particles.particles_kernels import _rotate_particle
from core.physics.g4coherent import _generate_coherent_theta_scalar
from core.physics.g4compton import (
    _calculate_compton_energy_deposit_scalar,
    _generate_compton_theta_scalar,
)
from core.physics.interaction_buffers import InteractionBuffer, RNGContext
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.physics_kernels import _locate_energy_interval


@njit(cache=True, inline='always')
def _generate_scattering_phi_scalar(rng_ctx: RNGContext) -> Float:
    """Генерирует азимутальный угол рассеяния phi равномерно на интервале [-pi, pi]."""
    return Float(np.pi * (rng_ctx.next_double(rng_ctx.state_addr) * 2.0 - 1.0))


@njit(cache=True, inline='always')
def _push_to_interaction_buffer(
    inter_buffer: InteractionBuffer,
    process_id: ProcessID,
    volume_id: Index,
    material_id: Index,
    particle_ID: ID,
    energy_deposit: Energy,
    scattering_theta: Float,
    scattering_phi: Float,
    distance_traveled: Float,
    species: Species,
    Z: Charge,
    pos_x: Float,
    pos_y: Float,
    pos_z: Float,
    dir_x: Float,
    dir_y: Float,
    dir_z: Float,
) -> None:
    """Записывает акт взаимодействия частицы напрямую в буфер с курсором InteractionBuffer."""
    if inter_buffer.capacity <= 0:
        return

    idx = inter_buffer.cursor[0]
    inter_buffer.cursor[0] = idx + 1
    if idx >= inter_buffer.capacity:
        return

    inter_buffer.process_id[idx] = process_id
    inter_buffer.volume_id[idx] = volume_id
    inter_buffer.material_id[idx] = material_id
    inter_buffer.particle_ID[idx] = particle_ID
    inter_buffer.energy_deposit[idx] = energy_deposit

    inter_buffer.scattering_theta[idx] = scattering_theta
    inter_buffer.scattering_phi[idx] = scattering_phi

    inter_buffer.distance_traveled[idx] = distance_traveled
    inter_buffer.species[idx] = species
    inter_buffer.Z[idx] = Z

    inter_buffer.position.x[idx] = pos_x
    inter_buffer.position.y[idx] = pos_y
    inter_buffer.position.z[idx] = pos_z

    inter_buffer.direction.x[idx] = dir_x
    inter_buffer.direction.y[idx] = dir_y
    inter_buffer.direction.z[idx] = dir_z


@njit(cache=True, inline='always')
def _sample_Z_scalar(
    mat_id: Index,
    process_col_idx: Index,
    energy: Float,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
) -> Charge:
    """
    Скалярный выбор атомного номера Z элемента-мишени по предрассчитанным нормированным
    парциальным долям сечений процесса в ElementCSR с линейной интерполяцией между узлами сетки.
    Для одноэлементного материала возвращает Z детерминированно без обращения к генератору ГСЧ.
    """
    start = physics_buffer.element_csr.element_offsets[mat_id]
    end = physics_buffer.element_csr.element_offsets[mat_id + 1]
    count = end - start

    if count <= 0:
        return Charge(1)

    if count == 1:
        return Charge(physics_buffer.element_csr.element_Z[start])

    row, fraction = _locate_energy_interval(energy, mat_id, physics_buffer.material_bank)
    rnd = rng_ctx.next_double(rng_ctx.state_addr)
    cumulative_prob = 0.0
    partial_fractions = physics_buffer.element_csr.element_partial_fractions
    num_rows = partial_fractions.shape[0]
    has_next_row = fraction > 0.0 and (row + 1) < num_rows

    for elem_offset in range(count - 1):
        weight_low = partial_fractions[row, process_col_idx, elem_offset]
        if has_next_row:
            weight_high = partial_fractions[row + 1, process_col_idx, elem_offset]
            weight = weight_low + fraction * (weight_high - weight_low)
        else:
            weight = weight_low
        cumulative_prob += weight
        if rnd < cumulative_prob:
            return Charge(physics_buffer.element_csr.element_Z[start + elem_offset])

    return Charge(physics_buffer.element_csr.element_Z[end - 1])


@njit(cache=True)
def _process_kernel_impl(
    state: KinematicState,
    particle_ids: NDArray[ID],
    target_indices: NDArray[Index],
    volume_ids: NDArray[Index],
    material_ids: NDArray[Index],
    inter_buffer: InteractionBuffer,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
    process_id_c: ProcessID,
    process_col_idx: Index,
    kind: Index,
) -> None:
    """
    Применяет физический процесс (0: фотоэффект, 1: комптоновское рассеяние, 2: когерентное рассеяние)
    на месте к выбранным частицам и записывает события в inter_buffer.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, process_col_idx, energy, physics_buffer, rng_ctx)

        if kind == 0:
            theta = Float(0.0)
            phi = Float(0.0)
            energy_deposit = energy
            state.energy[p_idx] = 0.0
        elif kind == 1:
            theta = _generate_compton_theta_scalar(energy, atomic_z, rng_ctx)
            phi = _generate_scattering_phi_scalar(rng_ctx)
            energy_deposit = _calculate_compton_energy_deposit_scalar(theta, energy)
            state.energy[p_idx] -= energy_deposit
            _rotate_particle(state, p_idx, theta, phi)
        else:
            theta = _generate_coherent_theta_scalar(energy, atomic_z, rng_ctx)
            phi = _generate_scattering_phi_scalar(rng_ctx)
            energy_deposit = Float(0.0)
            _rotate_particle(state, p_idx, theta, phi)

        _push_to_interaction_buffer(
            inter_buffer,
            process_id_c,
            volume_ids[p_idx],
            mat_id,
            particle_ids[p_idx],
            energy_deposit,
            theta,
            phi,
            state.distance_traveled[p_idx],
            state.species[p_idx],
            atomic_z,
            state.position.x[p_idx],
            state.position.y[p_idx],
            state.position.z[p_idx],
            state.direction.x[p_idx],
            state.direction.y[p_idx],
            state.direction.z[p_idx],
        )


_KERNEL_CACHE: dict[tuple[int, int], Any] = {}


def _make_process_kernel(process_id: ProcessID, kind: int) -> Any:
    cache_key = (kind, int(process_id))
    cached_kernel = _KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    process_id_c = ProcessID(process_id)
    kind_c = Index(kind)

    def _process_kernel(
        state: KinematicState,
        particle_ids: NDArray[ID],
        target_indices: NDArray[Index],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_col_idx: Index = Index(0),
    ) -> None:
        _process_kernel_impl(
            state,
            particle_ids,
            target_indices,
            volume_ids,
            material_ids,
            inter_buffer,
            physics_buffer,
            rng_ctx,
            process_id_c,
            process_col_idx,
            kind_c,
        )

    _KERNEL_CACHE[cache_key] = _process_kernel
    return _process_kernel


def make_photoelectric_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро фотоэффекта с запеченным идентификатором процесса."""
    return _make_process_kernel(process_id, 0)


def make_compton_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро комптоновского рассеяния с запеченным идентификатором процесса."""
    return _make_process_kernel(process_id, 1)


def make_coherent_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро когерентного (рэлеевского) рассеяния с запеченным идентификатором процесса."""
    return _make_process_kernel(process_id, 2)


__all__ = [
    "_generate_scattering_phi_scalar",
    "_push_to_interaction_buffer",
    "_sample_Z_scalar",
    "make_photoelectric_kernel",
    "make_compton_kernel",
    "make_coherent_kernel",
]

