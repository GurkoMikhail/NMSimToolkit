from core.physics.physics_buffer import PhysicsBuffer
import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Index, Charge, ProcessID, ID, Energy, Float
from core.particles.particles_soa import ParticleState
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.physics.g4compton_soa import _generate_compton_theta_scalar, _calculate_compton_energy_deposit_scalar
from core.physics.g4coherent_soa import _generate_coherent_theta_scalar
from core.particles.particles_soa_kernels import _rotate_particle


@njit(cache=True, inline='always')
def _generate_scattering_phi_scalar(rng_ctx: RNGContext) -> Float:
    """Generates the azimuthal scattering angle phi uniformly in [-pi, pi]."""
    return np.pi * (rng_ctx.next_double(rng_ctx.state_addr) * 2.0 - 1.0)


@njit(cache=True, inline='always')
def _push_to_interaction_buffer(
    inter_buffer: InteractionBuffer,
    process_id: ProcessID,
    particle_ID: ID,
    energy_deposit: Energy,
    scattering_theta: Float,
    scattering_phi: Float,
    pos_x: Float, pos_y: Float, pos_z: Float,
    dir_x: Float, dir_y: Float, dir_z: Float
) -> None:
    """Logs the interaction directly to the buffer using wrap-around cursor if necessary."""
    idx = inter_buffer.cursor[0] % inter_buffer.capacity

    inter_buffer.process_id[idx] = process_id
    inter_buffer.particle_ID[idx] = particle_ID
    inter_buffer.energy_deposit[idx] = energy_deposit

    inter_buffer.scattering_theta[idx] = scattering_theta
    inter_buffer.scattering_phi[idx] = scattering_phi

    inter_buffer.position.x[idx] = pos_x
    inter_buffer.position.y[idx] = pos_y
    inter_buffer.position.z[idx] = pos_z

    inter_buffer.direction.x[idx] = dir_x
    inter_buffer.direction.y[idx] = dir_y
    inter_buffer.direction.z[idx] = dir_z

    inter_buffer.cursor[0] += 1

@njit(cache=True, inline='always')
def _sample_Z_scalar(mat_id: Index, physics_buffer: PhysicsBuffer, rng_ctx: RNGContext) -> Charge:
    start = physics_buffer.element_csr.element_offsets[mat_id]
    end = physics_buffer.element_csr.element_offsets[mat_id + 1]

    if end - start == 1:
        return physics_buffer.element_csr.element_Z[start]

    rnd = rng_ctx.next_double(rng_ctx.state_addr)
    p0 = 0.0

    for i in range(start, end):
        p1 = p0 + physics_buffer.element_csr.element_fraction[i]
        if p0 <= rnd < p1:
            return physics_buffer.element_csr.element_Z[i]
        p0 = p1

    # Fallback
    return physics_buffer.element_csr.element_Z[end - 1]



def make_photoelectric_kernel(process_id: ProcessID):
    """
    Creates a photoelectric effect kernel with a baked-in process_id.
    """
    process_id_c = ProcessID(process_id)

    @njit(cache=True, inline='always')
    def _photoelectric_device_func(
        p_idx: Index,
        state: ParticleState,
        inter_buffer: InteractionBuffer
    ) -> None:
        # The entire energy of the particle is deposited
        energy_deposit = state.energy[p_idx]
        state.energy[p_idx] = 0.0

        _push_to_interaction_buffer(
            inter_buffer,
            process_id_c,
            state.ID[p_idx],
            energy_deposit,
            0.0, 0.0,
            state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx],
            state.direction.x[p_idx], state.direction.y[p_idx], state.direction.z[p_idx]
        )

    @njit(cache=True)
    def _photoelectric_kernel(
        state: ParticleState,
        target_indices: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext
    ) -> None:
        """
        Applies photoelectric effect IN-PLACE to target particles and logs to inter_buffer.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]
            _photoelectric_device_func(p_idx, state, inter_buffer)

    return _photoelectric_kernel


def make_compton_kernel(process_id: ProcessID):
    """
    Creates a Compton scattering kernel with a baked-in process_id.
    """
    process_id_c = ProcessID(process_id)

    @njit(cache=True, inline='always')
    def _compton_device_func(
        p_idx: Index,
        state: ParticleState,
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext
    ) -> None:
        energy = state.energy[p_idx]

        mat_id = material_ids[p_idx]
        Z = _sample_Z_scalar(mat_id, physics_buffer, rng_ctx)

        theta = _generate_compton_theta_scalar(energy, Z, rng_ctx)
        phi = _generate_scattering_phi_scalar(rng_ctx)

        energy_deposit = _calculate_compton_energy_deposit_scalar(theta, energy)

        # Update particle state IN-PLACE
        state.energy[p_idx] -= energy_deposit

        _rotate_particle(state, p_idx, theta, phi)

        _push_to_interaction_buffer(
            inter_buffer,
            process_id_c,
            state.ID[p_idx],
            energy_deposit,
            theta, phi,
            state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx],
            state.direction.x[p_idx], state.direction.y[p_idx], state.direction.z[p_idx]
        )

    @njit(cache=True)
    def _compton_kernel(
        state: ParticleState,
        target_indices: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext
    ) -> None:
        """
        Applies Compton scattering IN-PLACE to target particles and logs to inter_buffer.
        Requires effective Z of the material.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]
            _compton_device_func(p_idx, state, material_ids, inter_buffer, physics_buffer, rng_ctx)

    return _compton_kernel


def make_coherent_kernel(process_id: ProcessID):
    """
    Creates a Coherent (Rayleigh) scattering kernel with a baked-in process_id.
    """
    process_id_c = ProcessID(process_id)

    @njit(cache=True, inline='always')
    def _coherent_device_func(
        p_idx: Index,
        state: ParticleState,
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext
    ) -> None:
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        Z = _sample_Z_scalar(mat_id, physics_buffer, rng_ctx)

        theta = _generate_coherent_theta_scalar(energy, Z, rng_ctx)
        phi = _generate_scattering_phi_scalar(rng_ctx)

        # Coherent scattering has 0 energy deposit
        energy_deposit = 0.0

        _rotate_particle(state, p_idx, theta, phi)

        _push_to_interaction_buffer(
            inter_buffer,
            process_id_c,
            state.ID[p_idx],
            energy_deposit,
            theta, phi,
            state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx],
            state.direction.x[p_idx], state.direction.y[p_idx], state.direction.z[p_idx]
        )

    @njit(cache=True)
    def _coherent_kernel(
        state: ParticleState,
        target_indices: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext
    ) -> None:
        """
        Applies Coherent scattering IN-PLACE to target particles and logs to inter_buffer.
        Requires effective Z of the material.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]
            _coherent_device_func(p_idx, state, material_ids, inter_buffer, physics_buffer, rng_ctx)

    return _coherent_kernel
