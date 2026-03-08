import numpy as np
from numba import njit
from typing import Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.particles.particles_soa import ParticleState
from core.geometry.geometry_soa import GeometryBuffer

@njit(cache=True)
def _box_intersect(
    pos_x: float, pos_y: float, pos_z: float,
    dir_x: float, dir_y: float, dir_z: float,
    hx: float, hy: float, hz: float
) -> Tuple[float, float]:
    """
    Computes ray intersection with an AABB Box centered at origin.
    Returns tmin, tmax.
    """
    # X axis
    if dir_x != 0.0:
        inv_dir_x = 1.0 / dir_x
        tx1 = (-hx - pos_x) * inv_dir_x
        tx2 = ( hx - pos_x) * inv_dir_x
        tmin_x = min(tx1, tx2)
        tmax_x = max(tx1, tx2)
    else:
        # if ray is parallel to X plane, and outside box, tmin=inf, tmax=-inf
        if abs(pos_x) > hx:
            return np.inf, -np.inf
        else:
            tmin_x = -np.inf
            tmax_x = np.inf

    # Y axis
    if dir_y != 0.0:
        inv_dir_y = 1.0 / dir_y
        ty1 = (-hy - pos_y) * inv_dir_y
        ty2 = ( hy - pos_y) * inv_dir_y
        tmin_y = min(ty1, ty2)
        tmax_y = max(ty1, ty2)
    else:
        if abs(pos_y) > hy:
            return np.inf, -np.inf
        else:
            tmin_y = -np.inf
            tmax_y = np.inf

    # Z axis
    if dir_z != 0.0:
        inv_dir_z = 1.0 / dir_z
        tz1 = (-hz - pos_z) * inv_dir_z
        tz2 = ( hz - pos_z) * inv_dir_z
        tmin_z = min(tz1, tz2)
        tmax_z = max(tz1, tz2)
    else:
        if abs(pos_z) > hz:
            return np.inf, -np.inf
        else:
            tmin_z = -np.inf
            tmax_z = np.inf

    tmin = max(tmin_x, tmin_y, tmin_z)
    tmax = min(tmax_x, tmax_y, tmax_z)

    return tmin, tmax


@njit(cache=True)
def cast_path_kernel(
    particles: ParticleState,
    target_indices: NDArray[Index],
    geom_buffer: GeometryBuffer,
    out_distances: NDArray[Float],
    out_volume_indices: NDArray[Index]
) -> None:
    """
    Raycasting Numba kernel over an array of target active particles against a flattened GeometryBuffer.
    Applies loop unrolling for coordinate transformations and uses miss_index for Boundary Tracking / Frustum Culling.
    """
    num_particles = target_indices.shape[0]
    num_geoms = geom_buffer.geom_type.shape[0]

    for j in range(num_particles):
        p_idx = target_indices[j]

        # Original World Position and Direction
        w_pos_x = particles.position.x[p_idx]
        w_pos_y = particles.position.y[p_idx]
        w_pos_z = particles.position.z[p_idx]

        w_dir_x = particles.direction.x[p_idx]
        w_dir_y = particles.direction.y[p_idx]
        w_dir_z = particles.direction.z[p_idx]

        closest_dist = np.inf
        closest_vol = -1

        g_idx = 0
        while g_idx < num_geoms:
            # 1. Transform World -> Local (Loop Unrolling, without np.matmul)
            rot_00 = geom_buffer.transform.rot_00[g_idx]
            rot_01 = geom_buffer.transform.rot_01[g_idx]
            rot_02 = geom_buffer.transform.rot_02[g_idx]
            rot_10 = geom_buffer.transform.rot_10[g_idx]
            rot_11 = geom_buffer.transform.rot_11[g_idx]
            rot_12 = geom_buffer.transform.rot_12[g_idx]
            rot_20 = geom_buffer.transform.rot_20[g_idx]
            rot_21 = geom_buffer.transform.rot_21[g_idx]
            rot_22 = geom_buffer.transform.rot_22[g_idx]

            tr_x = geom_buffer.transform.tr_x[g_idx]
            tr_y = geom_buffer.transform.tr_y[g_idx]
            tr_z = geom_buffer.transform.tr_z[g_idx]

            # local_pos = W * R + T (assuming total_matrix is applied like W_vec * Matrix)
            # wait, previous logic said W is row vector [x, y, z, 1]
            # W @ mat.T = (mat @ W.T).T => mat is applied as M * v column vector.
            # R is 3x3, T is translation
            l_pos_x = rot_00 * w_pos_x + rot_01 * w_pos_y + rot_02 * w_pos_z + tr_x
            l_pos_y = rot_10 * w_pos_x + rot_11 * w_pos_y + rot_12 * w_pos_z + tr_y
            l_pos_z = rot_20 * w_pos_x + rot_21 * w_pos_y + rot_22 * w_pos_z + tr_z

            l_dir_x = rot_00 * w_dir_x + rot_01 * w_dir_y + rot_02 * w_dir_z
            l_dir_y = rot_10 * w_dir_x + rot_11 * w_dir_y + rot_12 * w_dir_z
            l_dir_z = rot_20 * w_dir_x + rot_21 * w_dir_y + rot_22 * w_dir_z

            # 2. Geometry Branching (Fat Node / Generic Parameters)
            geom_type = geom_buffer.geom_type[g_idx]
            param_0 = geom_buffer.param_0[g_idx]
            param_1 = geom_buffer.param_1[g_idx]
            param_2 = geom_buffer.param_2[g_idx]
            param_3 = geom_buffer.param_3[g_idx]

            if geom_type == 0:  # Box
                tmin, tmax = _box_intersect(
                    l_pos_x, l_pos_y, l_pos_z,
                    l_dir_x, l_dir_y, l_dir_z,
                    param_0, param_1, param_2
                )
            else:
                # Unsupported geometry
                tmin = np.inf
                tmax = -np.inf

            distance_epsilon = param_3

            # 3. Frustum Culling / Boundary Tracking Logic

            # Case 1: Miss
            if tmax < 0 or tmax < tmin:
                # Particle ray does not intersect this volume.
                # Jump over children.
                g_idx = geom_buffer.miss_index[g_idx]
                continue

            # Case 2: Hit from OUTSIDE
            elif tmin > 0:
                dist = tmin + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = geom_buffer.material_index[g_idx]

                # We do NOT check children, because they are inside this parent.
                # The closest boundary is the entry to this parent.
                g_idx = geom_buffer.miss_index[g_idx]
                continue

            # Case 3: Hit from INSIDE (tmin < 0 and tmax > 0)
            else:
                dist = tmax + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = geom_buffer.material_index[g_idx]

                # Particle is inside parent, moving towards exit (tmax).
                # The ray might hit children on the way to the exit.
                # Fall through to children.
                g_idx += 1

        out_distances[j] = closest_dist
        out_volume_indices[j] = closest_vol
