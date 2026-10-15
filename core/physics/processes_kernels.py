"""
Модуль Numba-ядер физических процессов взаимодействия частиц (фотоэффект,
комптоновское и когерентное рассеяние) и записи актов взаимодействия в буфер.
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
    """Записывает акт взаимодействия частицы напрямую в кольцевой буфер InteractionBuffer."""
    idx = inter_buffer.cursor[0] % inter_buffer.capacity

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

    inter_buffer.cursor[0] += 1


@njit(cache=True, inline='always')
def _sample_Z_scalar(mat_id: Index, physics_buffer: PhysicsBuffer, rng_ctx: RNGContext) -> Charge:
    """
    Скалярный выбор атомного номера Z элемента из состава материала по кумулятивной доле в ElementCSR.
    """
    start = physics_buffer.element_csr.element_offsets[mat_id]
    end = physics_buffer.element_csr.element_offsets[mat_id + 1]

    if end <= start:
        return Charge(1)

    if end - start == 1:
        return Charge(physics_buffer.element_csr.element_Z[start])

    rnd = rng_ctx.next_double(rng_ctx.state_addr)
    p0 = 0.0

    for i in range(start, end):
        p1 = p0 + physics_buffer.element_csr.element_fraction[i]
        if p0 <= rnd < p1:
            return Charge(physics_buffer.element_csr.element_Z[i])
        p0 = p1

    return Charge(physics_buffer.element_csr.element_Z[end - 1])


def make_photoelectric_kernel(process_id: ProcessID) -> Any:
    """
    Создает скомпилированное Numba-ядро фотоэффекта с запеченным идентификатором процесса.
    """
    process_id_c = ProcessID(process_id)

    @njit(cache=True, inline='always')
    def _photoelectric_device_func(
        p_idx: Index,
        state: KinematicState,
        particle_ids: NDArray[ID],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        energy_deposit = state.energy[p_idx]
        state.energy[p_idx] = 0.0

        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, physics_buffer, rng_ctx)

        _push_to_interaction_buffer(
            inter_buffer,
            process_id_c,
            volume_ids[p_idx],
            mat_id,
            particle_ids[p_idx],
            energy_deposit,
            Float(0.0),
            Float(0.0),
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

    @njit(cache=True)
    def _photoelectric_kernel(
        state: KinematicState,
        particle_ids: NDArray[ID],
        target_indices: NDArray[Index],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        """
        Применяет фотоэффект на месте к выбранным частицам и записывает события в inter_buffer.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]
            _photoelectric_device_func(
                p_idx, state, particle_ids, volume_ids, material_ids, inter_buffer, physics_buffer, rng_ctx
            )

    return _photoelectric_kernel


def make_compton_kernel(process_id: ProcessID) -> Any:
    """
    Создает скомпилированное Numba-ядро комптоновского рассеяния с запеченным идентификатором процесса.
    """
    process_id_c = ProcessID(process_id)

    @njit(cache=True, inline='always')
    def _compton_device_func(
        p_idx: Index,
        state: KinematicState,
        particle_ids: NDArray[ID],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        energy = state.energy[p_idx]

        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, physics_buffer, rng_ctx)

        theta = _generate_compton_theta_scalar(energy, atomic_z, rng_ctx)
        phi = _generate_scattering_phi_scalar(rng_ctx)

        energy_deposit = _calculate_compton_energy_deposit_scalar(theta, energy)

        state.energy[p_idx] -= energy_deposit

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

    @njit(cache=True)
    def _compton_kernel(
        state: KinematicState,
        particle_ids: NDArray[ID],
        target_indices: NDArray[Index],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        """
        Применяет комптоновское рассеяние на месте к выбранным частицам и записывает события в inter_buffer.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]
            _compton_device_func(
                p_idx, state, particle_ids, volume_ids, material_ids, inter_buffer, physics_buffer, rng_ctx
            )

    return _compton_kernel


def make_coherent_kernel(process_id: ProcessID) -> Any:
    """
    Создает скомпилированное Numba-ядро когерентного (рэлеевского) рассеяния с запеченным идентификатором процесса.
    """
    process_id_c = ProcessID(process_id)

    @njit(cache=True, inline='always')
    def _coherent_device_func(
        p_idx: Index,
        state: KinematicState,
        particle_ids: NDArray[ID],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, physics_buffer, rng_ctx)

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

    @njit(cache=True)
    def _coherent_kernel(
        state: KinematicState,
        particle_ids: NDArray[ID],
        target_indices: NDArray[Index],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
    ) -> None:
        """
        Применяет когерентное рассеяние на месте к выбранным частицам и записывает события в inter_buffer.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]
            _coherent_device_func(
                p_idx, state, particle_ids, volume_ids, material_ids, inter_buffer, physics_buffer, rng_ctx
            )

    return _coherent_kernel


__all__ = [
    "_generate_scattering_phi_scalar",
    "_push_to_interaction_buffer",
    "_sample_Z_scalar",
    "make_photoelectric_kernel",
    "make_compton_kernel",
    "make_coherent_kernel",
]

