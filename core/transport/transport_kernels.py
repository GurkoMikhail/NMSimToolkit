from core.transport.transport_buffer import TransportBuffer

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Index, Float, CFuncAddress

from core.particles.kinematic_state import KinematicState
from core.particles.initial_state import InitialState
from core.physics.interaction_soa import InitialStateBuffer
from core.particles.particles_soa_kernels import _move_particle
from core.geometry.navigation_state import NavigationState
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.interaction_soa import RNGContext
from core.physics.physics_kernels import _get_macroscopic_cross_sections

from numba.extending import intrinsic
from numba.core import types

@intrinsic
def call_cfunc_ptr(typingctx, ptr, x, y, z):
    sig = types.int64(ptr, x, y, z)
    def codegen(context, builder, signature, args):
        ptr_val, x_val, y_val, z_val = args
        # Cast integer pointer to a function pointer
        from llvmlite import ir
        fnty = ir.FunctionType(ir.IntType(64), [ir.DoubleType(), ir.DoubleType(), ir.DoubleType()])
        fnptr = builder.inttoptr(ptr_val, fnty.as_pointer())
        return builder.call(fnptr, [x_val, y_val, z_val])
    return sig, codegen


@njit(cache=True, inline='always')
def _get_random_double(rng_ctx: RNGContext) -> Float:
    return rng_ctx.next_double(rng_ctx.state_addr)

@njit(cache=True, inline='always')
def _generate_free_path(majorant_lac: Float, rng_ctx: RNGContext) -> Float:
    if majorant_lac <= 0.0:
        return np.inf
    return -np.log(_get_random_double(rng_ctx)) / majorant_lac

@njit(cache=True)
def _push_to_initial_state_kernel(
    initial_state: InitialState,
    target_indices: NDArray[Index],
    initial_state_buffer: InitialStateBuffer
) -> None:
    for j in range(target_indices.shape[0]):
        p_idx = target_indices[j]

        if not initial_state.has_interacted[p_idx]:
            initial_state.has_interacted[p_idx] = True

            idx = initial_state_buffer.cursor[0] % initial_state_buffer.capacity

            initial_state_buffer.particle_ID[idx] = initial_state.ID[p_idx]
            initial_state_buffer.emission_time[idx] = initial_state.emission_time[p_idx]
            initial_state_buffer.emission_energy[idx] = initial_state.emission_energy[p_idx]

            initial_state_buffer.emission_position.x[idx] = initial_state.emission_position.x[p_idx]
            initial_state_buffer.emission_position.y[idx] = initial_state.emission_position.y[p_idx]
            initial_state_buffer.emission_position.z[idx] = initial_state.emission_position.z[p_idx]

            initial_state_buffer.emission_direction.x[idx] = initial_state.emission_direction.x[p_idx]
            initial_state_buffer.emission_direction.y[idx] = initial_state.emission_direction.y[p_idx]
            initial_state_buffer.emission_direction.z[idx] = initial_state.emission_direction.z[p_idx]

            initial_state_buffer.cursor[0] += 1

def make_transport_kernel(mapped_process_ids: NDArray[Index]):
    num_processes = mapped_process_ids.shape[0]

    @njit(inline='always')
    def _sample_process_id(majorant_lac: Float, process_lacs: NDArray[Float], rng_ctx: RNGContext) -> Index:
        rnd = _get_random_double(rng_ctx) * majorant_lac
        p0 = 0.0
        for i in range(num_processes):
            p1 = p0 + process_lacs[i]
            if p0 <= rnd < p1:
                return mapped_process_ids[i]
            p0 = p1
        return -1

    @njit
    def transport_kernel(
        state: KinematicState,
        nav_state: NavigationState,
        target_indices: NDArray[Index],
        physics_buffer: PhysicsBuffer,
        transport_buffer: TransportBuffer,
        rng_ctx: RNGContext
    ) -> None:
        num_particles = target_indices.shape[0]
        process_ids = transport_buffer.process_ids
        material_ids = transport_buffer.material_ids

        # Pre-allocate temporary buffer outside the particle loop
        process_lacs = np.empty(num_processes, dtype=Float)

        for j in range(num_particles):
            p_idx = target_indices[j]

            current_vol = nav_state.current_volume[p_idx]
            if current_vol == -1:
                process_ids[p_idx] = -1
                continue

            majorant_mat_id = physics_buffer.majorant_material_map[current_vol]
            cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]

            # Get majorant cross-sections
            _get_macroscopic_cross_sections(state.energy[p_idx], majorant_mat_id, physics_buffer.material_bank, process_lacs)

            majorant_lac = 0.0
            for i in range(num_processes):
                majorant_lac += process_lacs[i]

            free_path = _generate_free_path(majorant_lac, rng_ctx)

            while free_path < nav_state.boundary_distance[p_idx]:
                _move_particle(state, p_idx, free_path)
                nav_state.boundary_distance[p_idx] -= free_path

                if cfunc_addr != 0:
                    mat_id = call_cfunc_ptr(cfunc_addr, state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx])
                    _get_macroscopic_cross_sections(state.energy[p_idx], mat_id, physics_buffer.material_bank, process_lacs)
                else:
                    mat_id = majorant_mat_id

                selected_process = _sample_process_id(majorant_lac, process_lacs, rng_ctx)

                if selected_process != -1:
                    material_ids[p_idx] = mat_id
                    process_ids[p_idx] = selected_process
                    break

                # Fictitious interaction (Delta scattering)
                free_path = _generate_free_path(majorant_lac, rng_ctx)

            else:
                # Reached boundary
                shift = nav_state.boundary_distance[p_idx] + 1e-6
                _move_particle(state, p_idx, shift)

                nav_state.current_volume[p_idx] = nav_state.next_volume[p_idx]
                nav_state.boundary_distance[p_idx] = 0.0
                process_ids[p_idx] = -1

    return transport_kernel
