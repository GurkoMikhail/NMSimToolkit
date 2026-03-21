
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
    sig = types.int64(types.uint64, types.float64, types.float64, types.float64)
    def codegen(context, builder, signature, args):
        ptr_val, x_val, y_val, z_val = args
        # Cast integer pointer to a function pointer
        from llvmlite import ir
        fnty = ir.FunctionType(ir.IntType(64), [ir.DoubleType(), ir.DoubleType(), ir.DoubleType()])
        fnptr = builder.inttoptr(ptr_val, fnty.as_pointer())
        return builder.call(fnptr, [x_val, y_val, z_val])
    return sig, codegen




@njit(inline='always')
def _woodcock_acceptance_test(
    p_idx: Index,
    state: ParticleState,
    current_vol: Index,
    total_lac: Float,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
    num_processes: int,
    out_lacs: NDArray[np.float64],
    materials_buffer: NDArray[Index]
) -> bool:
    cfunc_addr = physics_buffer.woodcock_function_pointers[current_vol]
    if cfunc_addr != 0:
        material_id = call_cfunc_ptr(cfunc_addr, state.position.x[p_idx], state.position.y[p_idx], state.position.z[p_idx])
        real_lacs = np.empty(num_processes, dtype=np.float64)
        _get_macroscopic_cross_sections(state.energy[p_idx], material_id, physics_buffer.material_bank, real_lacs)

        real_total_lac = 0.0
        for i in range(num_processes):
            real_total_lac += real_lacs[i]

        prob = real_total_lac / total_lac if total_lac > 0.0 else 0.0
        if rng_ctx.next_double(rng_ctx.state_addr) > prob:
            return False # Fictitious interaction

        for i in range(num_processes):
            out_lacs[i] = real_lacs[i]

        materials_buffer[p_idx] = material_id
        return True
    else:
        material_id = physics_buffer.majorant_material_map[current_vol]
        materials_buffer[p_idx] = material_id
        return True

@njit(inline='always')
def _sample_process(
    total_lac: Float,
    out_lacs: NDArray[np.float64],
    mapped_process_ids: NDArray[Index],
    rng_ctx: RNGContext,
    num_processes: int
) -> Index:
    # Scale RNG to avoid division
    rnd = rng_ctx.next_double(rng_ctx.state_addr) * total_lac
    p0 = 0.0
    for i in range(num_processes):
        p1 = p0 + out_lacs[i]
        if p0 <= rnd < p1:
            return mapped_process_ids[i]
        p0 = p1
    return mapped_process_ids[num_processes - 1]

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

            out_lacs = np.empty(num_processes, dtype=np.float64)
            real_lacs = np.empty(num_processes, dtype=np.float64)

            # Delta Tracking Loop
            while True:
                current_vol = nav_state.current_volume[p_idx]

                # Out of bounds
                if current_vol == -1:
                    process_ids[p_idx] = -1
                    break

                material_id = physics_buffer.majorant_material_map[current_vol]

                # Get majorant (or real for analog) LACs
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

                    # Woodcock acceptance test
                    accepted = _woodcock_acceptance_test(
                        p_idx, state, current_vol, total_lac, physics_buffer, rng_ctx, num_processes, out_lacs, materials_buffer
                    )

                    if not accepted:
                        continue

                    # Update total_lac to real_total_lac for correct process sampling
                    new_total_lac = 0.0
                    for i in range(num_processes):
                        new_total_lac += out_lacs[i]

                    # Sample process
                    process_ids[p_idx] = _sample_process(new_total_lac, out_lacs, mapped_process_ids, rng_ctx, num_processes)
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
