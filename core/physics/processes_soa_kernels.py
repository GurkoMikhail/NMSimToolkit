import numpy as np
from numba import njit
from numpy.typing import NDArray
import ctypes

from core.other.typing_definitions import Index
from core.particles.particles_soa import ParticleState
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.physics.g4compton_soa import _generate_compton_theta_scalar, _calculate_compton_energy_deposit_scalar
from core.physics.g4coherent_soa import _generate_coherent_theta_scalar
from core.other.vectors_soa import _rotate_direction_scalar


def make_photoelectric_kernel(process_id: int):
    """
    Creates a photoelectric effect kernel with a baked-in process_id.
    """
    @njit(cache=True)
    def _photoelectric_kernel(
        state: ParticleState,
        target_indices: NDArray[Index],
        inter_buffer: InteractionBuffer,
        rng_ctx: RNGContext
    ) -> None:
        """
        Applies photoelectric effect IN-PLACE to target particles and logs to inter_buffer.
        """
        for j in range(len(target_indices)):
            p_idx = target_indices[j]

            # The entire energy of the particle is deposited
            energy_deposit = state.energy[p_idx]
            state.energy[p_idx] = 0.0

            # Logging IN-PLACE
            idx = inter_buffer.cursor[0]

            # Check capacity to prevent out-of-bounds (the manager should flush when needed)
            if idx >= inter_buffer.capacity:
                continue

            inter_buffer.process_id[idx] = process_id
            inter_buffer.particle_ID[idx] = state.ID[p_idx]
            inter_buffer.energy_deposit[idx] = energy_deposit

            inter_buffer.scattering_theta[idx] = 0.0
            inter_buffer.scattering_phi[idx] = 0.0

            inter_buffer.position.x[idx] = state.position.x[p_idx]
            inter_buffer.position.y[idx] = state.position.y[p_idx]
            inter_buffer.position.z[idx] = state.position.z[p_idx]

            inter_buffer.direction.x[idx] = state.direction.x[p_idx]
            inter_buffer.direction.y[idx] = state.direction.y[p_idx]
            inter_buffer.direction.z[idx] = state.direction.z[p_idx]

            inter_buffer.cursor[0] += 1

    return _photoelectric_kernel


def make_compton_kernel(process_id: int):
    """
    Creates a Compton scattering kernel with a baked-in process_id.
    """
    @njit(cache=True)
    def _compton_kernel(
        state: ParticleState,
        target_indices: NDArray[Index],
        Z: int,
        inter_buffer: InteractionBuffer,
        rng_ctx: RNGContext
    ) -> None:
        """
        Applies Compton scattering IN-PLACE to target particles and logs to inter_buffer.
        Requires effective Z of the material.
        """
        next_double = rng_ctx.next_double
        state_addr = rng_ctx.state_addr

        for j in range(len(target_indices)):
            p_idx = target_indices[j]

            energy = state.energy[p_idx]

            theta = _generate_compton_theta_scalar(energy, Z, rng_ctx)
            phi = np.pi * (next_double(state_addr) * 2.0 - 1.0)

            energy_deposit = _calculate_compton_energy_deposit_scalar(theta, energy)

            # Update particle state IN-PLACE
            state.energy[p_idx] -= energy_deposit

            dir_x = state.direction.x[p_idx]
            dir_y = state.direction.y[p_idx]
            dir_z = state.direction.z[p_idx]

            new_dir_x, new_dir_y, new_dir_z = _rotate_direction_scalar(dir_x, dir_y, dir_z, theta, phi)
            state.direction.x[p_idx] = new_dir_x
            state.direction.y[p_idx] = new_dir_y
            state.direction.z[p_idx] = new_dir_z

            # Logging IN-PLACE
            idx = inter_buffer.cursor[0]
            if idx >= inter_buffer.capacity:
                continue

            inter_buffer.process_id[idx] = process_id
            inter_buffer.particle_ID[idx] = state.ID[p_idx]
            inter_buffer.energy_deposit[idx] = energy_deposit

            inter_buffer.scattering_theta[idx] = theta
            inter_buffer.scattering_phi[idx] = phi

            inter_buffer.position.x[idx] = state.position.x[p_idx]
            inter_buffer.position.y[idx] = state.position.y[p_idx]
            inter_buffer.position.z[idx] = state.position.z[p_idx]

            inter_buffer.direction.x[idx] = state.direction.x[p_idx]
            inter_buffer.direction.y[idx] = state.direction.y[p_idx]
            inter_buffer.direction.z[idx] = state.direction.z[p_idx]

            inter_buffer.cursor[0] += 1

    return _compton_kernel


def make_coherent_kernel(process_id: int):
    """
    Creates a Coherent (Rayleigh) scattering kernel with a baked-in process_id.
    """
    @njit(cache=True)
    def _coherent_kernel(
        state: ParticleState,
        target_indices: NDArray[Index],
        Z: int,
        inter_buffer: InteractionBuffer,
        rng_ctx: RNGContext
    ) -> None:
        """
        Applies Coherent scattering IN-PLACE to target particles and logs to inter_buffer.
        Requires effective Z of the material.
        """
        next_double = rng_ctx.next_double
        state_addr = rng_ctx.state_addr

        for j in range(len(target_indices)):
            p_idx = target_indices[j]

            energy = state.energy[p_idx]

            theta = _generate_coherent_theta_scalar(energy, Z, rng_ctx)
            phi = np.pi * (next_double(state_addr) * 2.0 - 1.0)

            # Coherent scattering has 0 energy deposit
            energy_deposit = 0.0

            dir_x = state.direction.x[p_idx]
            dir_y = state.direction.y[p_idx]
            dir_z = state.direction.z[p_idx]

            new_dir_x, new_dir_y, new_dir_z = _rotate_direction_scalar(dir_x, dir_y, dir_z, theta, phi)
            state.direction.x[p_idx] = new_dir_x
            state.direction.y[p_idx] = new_dir_y
            state.direction.z[p_idx] = new_dir_z

            # Logging IN-PLACE
            idx = inter_buffer.cursor[0]
            if idx >= inter_buffer.capacity:
                continue

            inter_buffer.process_id[idx] = process_id
            inter_buffer.particle_ID[idx] = state.ID[p_idx]
            inter_buffer.energy_deposit[idx] = energy_deposit

            inter_buffer.scattering_theta[idx] = theta
            inter_buffer.scattering_phi[idx] = phi

            inter_buffer.position.x[idx] = state.position.x[p_idx]
            inter_buffer.position.y[idx] = state.position.y[p_idx]
            inter_buffer.position.z[idx] = state.position.z[p_idx]

            inter_buffer.direction.x[idx] = state.direction.x[p_idx]
            inter_buffer.direction.y[idx] = state.direction.y[p_idx]
            inter_buffer.direction.z[idx] = state.direction.z[p_idx]

            inter_buffer.cursor[0] += 1

    return _coherent_kernel
