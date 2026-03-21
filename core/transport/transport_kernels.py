
from numba import njit, prange
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
    sig = types.int64(types.uint64, types.float64, types.float64, types.float64)
    def codegen(context, builder, signature, args):
        ptr_val, x_val, y_val, z_val = args
        # Cast integer pointer to a function pointer
        fnty = context.get_function_pointer_type(
            types.int64(types.float64, types.float64, types.float64)
        )
        fnptr = builder.inttoptr(ptr_val, fnty)
        return builder.call(fnptr, [x_val, y_val, z_val])
    return sig, codegen




def make_transport_kernel(num_processes: int):
    @njit(cache=True)
    def transport_kernel(
        state: ParticleState,
        nav_state: NavigationState,
        target_indices: NDArray[Index],
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_ids: NDArray[Index],
        mapped_process_ids: NDArray[Index]
    ) -> None:
        num_particles = target_indices.shape[0]

        for j in prange(num_particles):
            p_idx = target_indices[j]




        # Delta Tracking Loop
        while True:
            current_vol = nav_state.current_volume[p_idx]

            # Out of bounds
            if current_vol == -1:
                process_ids[p_idx] = -1
                break

            material_id = physics_buffer.majorant_material_map[current_vol]

            # Get majorant (or real for analog) LACs
            out_lacs = np.zeros(num_processes, dtype=np.float64)
            _get_macroscopic_cross_sections(state.energy[p_idx], material_id, physics_buffer.material_bank, out_lacs)

            total_lac = 0.0
            for i in range(num_processes):
                total_lac += out_lacs[i]

            # Prevent division by zero
            if total_lac <= 0.0:
                # effectively infinite free path
                free_path = np.inf
            else:
                free_path = -np.log(rng_ctx.next_double(rng_ctx.state_addr)) / total_lac

            if free_path < nav_state.boundary_distance[p_idx]:
                # 2a. Move particle by free_path
                state.position.x[p_idx] += state.direction.x[p_idx] * free_path
                state.position.y[p_idx] += state.direction.y[p_idx] * free_path
                state.position.z[p_idx] += state.direction.z[p_idx] * free_path
                nav_state.boundary_distance[p_idx] -= free_path

                # Woodcock checking
                cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]
                if cfunc_addr != 0:


                    real_material_id = call_cfunc_ptr(cfunc_addr, state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx])



                    real_lacs = np.zeros(num_processes, dtype=np.float64)
                    _get_macroscopic_cross_sections(state.energy[p_idx], real_material_id, physics_buffer.material_bank, real_lacs)

                    real_total_lac = 0.0
                    for i in range(num_processes):
                        real_total_lac += real_lacs[i]

                    prob = real_total_lac / total_lac if total_lac > 0.0 else 0.0
                    if rng_ctx.next_double(rng_ctx.state_addr) > prob:
                        # Fictitious interaction (Delta scattering)
                        continue

                    # Real interaction: update out_lacs to use for sampling
                    for i in range(num_processes):
                        out_lacs[i] = real_lacs[i]
                    total_lac = real_total_lac

                # Sample process
                rnd = rng_ctx.next_double(rng_ctx.state_addr) * total_lac
                p0 = 0.0
                selected_proc_idx = -1
                for i in range(num_processes):
                    p1 = p0 + out_lacs[i]
                    if p0 <= rnd < p1:
                        selected_proc_idx = i
                        break
                    p0 = p1

                # Fallback if precision issues
                if selected_proc_idx == -1:
                    selected_proc_idx = num_processes - 1

                process_ids[p_idx] = mapped_process_ids[selected_proc_idx]
                break

            else:
                # 2b. Move particle to boundary
                shift = nav_state.boundary_distance[p_idx] + 1e-6
                state.position.x[p_idx] += state.direction.x[p_idx] * shift
                state.position.y[p_idx] += state.direction.y[p_idx] * shift
                state.position.z[p_idx] += state.direction.z[p_idx] * shift

                nav_state.current_volume[p_idx] = nav_state.next_volume[p_idx]
                nav_state.boundary_distance[p_idx] = 0.0
                process_ids[p_idx] = -1
                break

    return transport_kernel
