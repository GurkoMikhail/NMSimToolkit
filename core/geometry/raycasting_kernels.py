import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple

from core.geometry.geometry_buffer import GeometryBuffer
from core.particles.particles_soa import ParticleState
from core.other.typing_definitions import Float, Index


@njit(cache=True)
def cast_path_kernel(
    state: ParticleState,
    target_indices: NDArray[Index],
    geom_buffer: GeometryBuffer,
    distance_epsilon: Float = 1e-3
) -> Tuple[NDArray[Float], NDArray[Index]]:
    """
    Flat Raycasting Numba Kernel computing distance to intersection
    and the volume/material ID intersected for each target particle.

    Implements a Frustum Culling jump-loop leveraging GeometryBuffer.miss_index.
    """
    num_particles = target_indices.shape[0]

    # Preallocate output arrays
    out_distances = np.full(num_particles, np.inf, dtype=np.float64)
    out_volumes = np.full(num_particles, -1, dtype=np.int64)

    num_volumes = geom_buffer.volume_id.shape[0]

    for j in range(num_particles):
        p_idx = target_indices[j]

        # World Position and Direction
        wx = state.position.x[p_idx]
        wy = state.position.y[p_idx]
        wz = state.position.z[p_idx]

        wdx = state.direction.x[p_idx]
        wdy = state.direction.y[p_idx]
        wdz = state.direction.z[p_idx]

        # We track the closest intersection distance across the entire tree
        # Initialized to infinity for each ray
        min_dist = np.inf
        hit_volume = -1

        g_idx = 0
        while g_idx < num_volumes:

            # 1. Transform World -> Local
            # The matrix represents World -> Local. We compute LocalPos = (WorldPos @ R.T) + T
            # Wait, `volumes.py` logic:
            #   local_position = np.ones((N, 4))
            #   local_position[:, :3] = position
            #   np.matmul(local_position, transformation_matrix.T, out=local_position)
            # This is exactly:
            #   x' = x*R00 + y*R01 + z*R02 + 1*TX
            #   y' = x*R10 + y*R11 + z*R12 + 1*TY
            #   z' = x*R20 + y*R21 + z*R22 + 1*TZ

            r00 = geom_buffer.transform.r00[g_idx]
            r01 = geom_buffer.transform.r01[g_idx]
            r02 = geom_buffer.transform.r02[g_idx]
            tx  = geom_buffer.transform.tx[g_idx]

            r10 = geom_buffer.transform.r10[g_idx]
            r11 = geom_buffer.transform.r11[g_idx]
            r12 = geom_buffer.transform.r12[g_idx]
            ty  = geom_buffer.transform.ty[g_idx]

            r20 = geom_buffer.transform.r20[g_idx]
            r21 = geom_buffer.transform.r21[g_idx]
            r22 = geom_buffer.transform.r22[g_idx]
            tz  = geom_buffer.transform.tz[g_idx]

            # Local Position
            lx = wx * r00 + wy * r01 + wz * r02 + tx
            ly = wx * r10 + wy * r11 + wz * r12 + ty
            lz = wx * r20 + wy * r21 + wz * r22 + tz

            # Local Direction (only rotation)
            ldx = wdx * r00 + wdy * r01 + wdz * r02
            ldy = wdx * r10 + wdy * r11 + wdz * r12
            ldz = wdx * r20 + wdy * r21 + wdz * r22

            # 2. Check Intersection (Box geometry)
            # Box half-sizes
            hx = geom_buffer.sizes.x[g_idx]
            hy = geom_buffer.sizes.y[g_idx]
            hz = geom_buffer.sizes.z[g_idx]

            # check_inside logic:
            inside = (abs(lx) <= hx) and (abs(ly) <= hy) and (abs(lz) <= hz)

            # Raycasting intersection calculation
            # Handle ldx == 0.0 safely to avoid NaN (np.inf - np.inf)
            if ldx == 0.0:
                # Ray is parallel to YZ plane.
                # Does it intersect the infinite slab between -hx and +hx?
                if abs(lx) <= hx:
                    tmin_x = -np.inf
                    tmax_x = np.inf
                else:
                    tmin_x = np.inf
                    tmax_x = -np.inf
            else:
                inv_ldx = 1.0 / ldx
                norm_x = -lx * inv_ldx
                norm_sz_x = abs(hx * inv_ldx)
                tmin_x = norm_x - norm_sz_x
                tmax_x = norm_x + norm_sz_x

            if ldy == 0.0:
                if abs(ly) <= hy:
                    tmin_y = -np.inf
                    tmax_y = np.inf
                else:
                    tmin_y = np.inf
                    tmax_y = -np.inf
            else:
                inv_ldy = 1.0 / ldy
                norm_y = -ly * inv_ldy
                norm_sz_y = abs(hy * inv_ldy)
                tmin_y = norm_y - norm_sz_y
                tmax_y = norm_y + norm_sz_y

            if ldz == 0.0:
                if abs(lz) <= hz:
                    tmin_z = -np.inf
                    tmax_z = np.inf
                else:
                    tmin_z = np.inf
                    tmax_z = -np.inf
            else:
                inv_ldz = 1.0 / ldz
                norm_z = -lz * inv_ldz
                norm_sz_z = abs(hz * inv_ldz)
                tmin_z = norm_z - norm_sz_z
                tmax_z = norm_z + norm_sz_z

            tmin = max(tmin_x, max(tmin_y, tmin_z))
            tmax = min(tmax_x, min(tmax_y, tmax_z))

            dist = np.inf
            if tmax > tmin:
                dist = tmin

            if inside:
                dist = tmax

            if dist < 0.0:
                dist = np.inf

            # Handle distance epsilon similar to OOP Geometry.cast_path
            if dist != np.inf:
                dist += distance_epsilon

            # 3. Process Result & Frustum Culling
            if dist != np.inf:
                # HIT: We keep traversing children.
                # Only update the minimum distance if this child is CLOSER than the current minimum distance.
                # In the DFS structure, parent boundaries are tested before children.
                if dist < min_dist:
                    min_dist = dist
                    hit_volume = geom_buffer.volume_id[g_idx]

                # Advance to child node
                g_idx += 1
            else:
                # MISS: Ray entirely misses this volume.
                # Frustum Culling jump: skip all children!
                g_idx = geom_buffer.miss_index[g_idx]

        # Final assignment after traversing all volumes for this particle
        out_distances[j] = min_dist
        out_volumes[j] = hit_volume

    return out_distances, out_volumes
