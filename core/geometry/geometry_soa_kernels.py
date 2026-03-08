import numpy as np
from numba import njit
from typing import Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.particles.particles_soa import ParticleState
from core.geometry.geometry_soa import GeometryBuffer
from core.other.vectors_soa import Vector3DSoA

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
    positions_x: NDArray[Float],
    positions_y: NDArray[Float],
    positions_z: NDArray[Float],
    directions_x: NDArray[Float],
    directions_y: NDArray[Float],
    directions_z: NDArray[Float],
    target_indices: NDArray[Index],

    geom_shape: NDArray[np.int32],
    param_0: NDArray[Float],
    param_1: NDArray[Float],
    param_2: NDArray[Float],
    param_3: NDArray[Float],

    rot_m00: NDArray[Float], rot_m01: NDArray[Float], rot_m02: NDArray[Float],
    rot_m10: NDArray[Float], rot_m11: NDArray[Float], rot_m12: NDArray[Float],
    rot_m20: NDArray[Float], rot_m21: NDArray[Float], rot_m22: NDArray[Float],

    tr_x: NDArray[Float], tr_y: NDArray[Float], tr_z: NDArray[Float],

    miss_index: NDArray[Index],
    volume_index: NDArray[Index],

    out_distances: NDArray[Float],
    out_volume_indices: NDArray[Index]
) -> None:
    """
    Raycasting Numba kernel over an array of target active particles against a flattened GeometryBuffer.
    Applies loop unrolling for coordinate transformations and uses miss_index for Boundary Tracking / Frustum Culling.
    """
    num_particles = target_indices.shape[0]
    num_geoms = geom_shape.shape[0]

    for j in range(num_particles):
        p_idx = target_indices[j]

        # Original World Position and Direction
        w_pos_x = positions_x[p_idx]
        w_pos_y = positions_y[p_idx]
        w_pos_z = positions_z[p_idx]

        w_dir_x = directions_x[p_idx]
        w_dir_y = directions_y[p_idx]
        w_dir_z = directions_z[p_idx]

        closest_dist = np.inf
        closest_vol = -1

        g_idx = 0
        while g_idx < num_geoms:
            # 1. Transform World -> Local (Loop Unrolling, without np.matmul)
            m00 = rot_m00[g_idx]
            m01 = rot_m01[g_idx]
            m02 = rot_m02[g_idx]
            m10 = rot_m10[g_idx]
            m11 = rot_m11[g_idx]
            m12 = rot_m12[g_idx]
            m20 = rot_m20[g_idx]
            m21 = rot_m21[g_idx]
            m22 = rot_m22[g_idx]

            t_x = tr_x[g_idx]
            t_y = tr_y[g_idx]
            t_z = tr_z[g_idx]

            # local_pos = W * R + T (assuming total_matrix is applied like W_vec * Matrix)
            # R is 3x3, T is translation
            l_pos_x = m00 * w_pos_x + m01 * w_pos_y + m02 * w_pos_z + t_x
            l_pos_y = m10 * w_pos_x + m11 * w_pos_y + m12 * w_pos_z + t_y
            l_pos_z = m20 * w_pos_x + m21 * w_pos_y + m22 * w_pos_z + t_z

            l_dir_x = m00 * w_dir_x + m01 * w_dir_y + m02 * w_dir_z
            l_dir_y = m10 * w_dir_x + m11 * w_dir_y + m12 * w_dir_z
            l_dir_z = m20 * w_dir_x + m21 * w_dir_y + m22 * w_dir_z

            # 2. Geometry Branching (Fat Node / Generic Parameters)
            geom_type = geom_shape[g_idx]
            p0 = param_0[g_idx]
            p1 = param_1[g_idx]
            p2 = param_2[g_idx]
            p3 = param_3[g_idx]

            if geom_type == 0:  # Box
                tmin, tmax = _box_intersect(
                    l_pos_x, l_pos_y, l_pos_z,
                    l_dir_x, l_dir_y, l_dir_z,
                    p0, p1, p2
                )
            else:
                # Unsupported geometry
                tmin = np.inf
                tmax = -np.inf

            distance_epsilon = p3

            # 3. Frustum Culling / Boundary Tracking Logic

            # Case 1: Miss
            if tmax < 0 or tmax < tmin:
                # Particle ray does not intersect this volume.
                # Jump over children.
                g_idx = miss_index[g_idx]
                continue

            # Case 2: Hit from OUTSIDE
            elif tmin > 0:
                dist = tmin + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = volume_index[g_idx]

                # We do NOT check children, because they are inside this parent.
                # The closest boundary is the entry to this parent.
                g_idx = miss_index[g_idx]
                continue

            # Case 3: Hit from INSIDE (tmin < 0 and tmax > 0)
            else:
                dist = tmax + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = volume_index[g_idx]

                # Particle is inside parent, moving towards exit (tmax).
                # The ray might hit children on the way to the exit.
                # Fall through to children.
                g_idx += 1

        out_distances[j] = closest_dist
        out_volume_indices[j] = closest_vol
