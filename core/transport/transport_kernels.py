
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


def make_transport_kernel(num_processes: int):
    @njit
    def transport_kernel(
        state: ParticleState,
        nav_state: NavigationState,
        target_indices: NDArray[Index],
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_ids: NDArray[Index],
        materials_buffer: NDArray[Index],
        mapped_process_ids: NDArray[Index]
    ) -> None:
        num_particles = target_indices.shape[0]

        for j in range(num_particles):
            p_idx = target_indices[j]

            current_vol = nav_state.current_volume[p_idx]
            if current_vol == -1:
                process_ids[p_idx] = -1
                continue

            majorant_mat_id = physics_buffer.majorant_material_map[current_vol]
            cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]

            out_lacs = np.empty(num_processes, dtype=np.float64)
            _get_macroscopic_cross_sections(state.energy[p_idx], majorant_mat_id, physics_buffer.material_bank, out_lacs)

            majorant_lac = 0.0
            for i in range(num_processes):
                majorant_lac += out_lacs[i]

            if majorant_lac <= 0.0:
                free_path = np.inf
            else:
                free_path = -np.log(rng_ctx.next_double(rng_ctx.state_addr)) / majorant_lac

            is_real_interaction = False

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

                rnd = rng_ctx.next_double(rng_ctx.state_addr) * majorant_lac
                p0 = 0.0

                for i in range(num_processes):
                    p1 = p0 + out_lacs[i]
                    if p0 <= rnd < p1:
                        materials_buffer[p_idx] = mat_id
                        process_ids[p_idx] = mapped_process_ids[i]
                        is_real_interaction = True
                        break
                    p0 = p1

                if is_real_interaction:
                    break

                # Fictitious interaction (Delta scattering)
                free_path = -np.log(rng_ctx.next_double(rng_ctx.state_addr)) / majorant_lac

            if is_real_interaction:
                continue

            # Reached boundary
            shift = nav_state.boundary_distance[p_idx] + 1e-6
            state.position.x[p_idx] += state.direction.x[p_idx] * shift
            state.position.y[p_idx] += state.direction.y[p_idx] * shift
            state.position.z[p_idx] += state.direction.z[p_idx] * shift

            nav_state.current_volume[p_idx] = nav_state.next_volume[p_idx]
            nav_state.boundary_distance[p_idx] = 0.0
            process_ids[p_idx] = -1

    return transport_kernel
