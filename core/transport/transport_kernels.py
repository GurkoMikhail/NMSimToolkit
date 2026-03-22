from core.transport.transport_buffer import TransportBuffer

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Index, Float, CFuncAddress

from core.particles.particles_soa import ParticleState
from core.geometry.navigation_state import NavigationState
from core.physics.physics_buffer import PhysicsBuffer
from core.physics.interaction_soa import RNGContext
from core.physics.physics_kernels import _get_macroscopic_cross_sections

import ctypes
from numba.extending import intrinsic
from numba.core import cgutils
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

def make_transport_kernel(mapped_process_ids: NDArray[Index]):
    num_processes = mapped_process_ids.shape[0]

    @njit(inline='always')
    def _sample_process_id(majorant_lac: Float, out_lacs: NDArray[np.float64], rng_ctx: RNGContext) -> Index:
        rnd = _get_random_double(rng_ctx) * majorant_lac
        p0 = 0.0
        for i in range(len(mapped_process_ids)):
            p1 = p0 + out_lacs[i]
            if p0 <= rnd < p1:
                return mapped_process_ids[i]
            p0 = p1
        return -1

    @njit
    def transport_kernel(
        state: ParticleState,
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
        out_lacs = np.empty(len(mapped_process_ids), dtype=np.float64)

        for j in range(num_particles):
            p_idx = target_indices[j]

            current_vol = nav_state.current_volume[p_idx]
            if current_vol == -1:
                process_ids[p_idx] = -1
                continue

            majorant_mat_id = physics_buffer.majorant_material_map[current_vol]
            cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]

            # Get majorant cross-sections
            _get_macroscopic_cross_sections(state.energy[p_idx], majorant_mat_id, physics_buffer.material_bank, out_lacs)

            majorant_lac = 0.0
            for i in range(len(mapped_process_ids)):
                majorant_lac += out_lacs[i]

            free_path = _generate_free_path(majorant_lac, rng_ctx)

            while free_path < nav_state.boundary_distance[p_idx]:
                state.position.x[p_idx] += state.direction.x[p_idx] * free_path
                state.position.y[p_idx] += state.direction.y[p_idx] * free_path
                state.position.z[p_idx] += state.direction.z[p_idx] * free_path
                nav_state.boundary_distance[p_idx] -= free_path

                if cfunc_addr != 0:
                    mat_id = call_cfunc_ptr(cfunc_addr, state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx])
                    _get_macroscopic_cross_sections(state.energy[p_idx], mat_id, physics_buffer.material_bank, out_lacs)
                else:
                    mat_id = majorant_mat_id

                selected_process = _sample_process_id(majorant_lac, out_lacs, rng_ctx)

                if selected_process != -1:
                    material_ids[p_idx] = mat_id
                    process_ids[p_idx] = selected_process
                    break

                # Fictitious interaction (Delta scattering)
                free_path = _generate_free_path(majorant_lac, rng_ctx)

            else:
                # Reached boundary
                shift = nav_state.boundary_distance[p_idx] + 1e-6
                state.position.x[p_idx] += state.direction.x[p_idx] * shift
                state.position.y[p_idx] += state.direction.y[p_idx] * shift
                state.position.z[p_idx] += state.direction.z[p_idx] * shift

                nav_state.current_volume[p_idx] = nav_state.next_volume[p_idx]
                nav_state.boundary_distance[p_idx] = 0.0
                process_ids[p_idx] = -1

    return transport_kernel
