import numpy as np
from numba import njit
from typing import Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.particles.particles_soa import ParticleState
from core.geometry.geometry_soa import ShapeBuffer
from core.other.vectors_soa import Vector3DSoA

@njit(cache=True)
def _box_intersect(
    pos_x: Float, pos_y: Float, pos_z: Float,
    dir_x: Float, dir_y: Float, dir_z: Float,
    hx: Float, hy: Float, hz: Float
) -> Tuple[Float, Float]:
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
    positions: Vector3DSoA,
    directions: Vector3DSoA,
    target_indices: NDArray[Index],
    shape_buffer: ShapeBuffer,
    out_distances: NDArray[Float],
    out_volume_indices: NDArray[Index]
) -> None:
    """
    Raycasting Numba kernel over an array of target active particles against a flattened ShapeBuffer.
    Applies loop unrolling for coordinate transformations and uses miss_index for Boundary Tracking / Frustum Culling.
    """
    num_particles = target_indices.shape[0]
    num_geoms = shape_buffer.shape.shape[0]

    for j in range(num_particles):
        p_idx = target_indices[j]

        # Original World Position and Direction
        w_pos_x = positions.x[p_idx]
        w_pos_y = positions.y[p_idx]
        w_pos_z = positions.z[p_idx]

        w_dir_x = directions.x[p_idx]
        w_dir_y = directions.y[p_idx]
        w_dir_z = directions.z[p_idx]

        closest_dist = np.inf
        closest_vol = -1

        g_idx = 0
        while g_idx < num_geoms:
            # 1. Transform World -> Local (Loop Unrolling, without np.matmul)
            rot_00 = shape_buffer.transform.rotation.rot_00[g_idx]
            rot_01 = shape_buffer.transform.rotation.rot_01[g_idx]
            rot_02 = shape_buffer.transform.rotation.rot_02[g_idx]
            rot_10 = shape_buffer.transform.rotation.rot_10[g_idx]
            rot_11 = shape_buffer.transform.rotation.rot_11[g_idx]
            rot_12 = shape_buffer.transform.rotation.rot_12[g_idx]
            rot_20 = shape_buffer.transform.rotation.rot_20[g_idx]
            rot_21 = shape_buffer.transform.rotation.rot_21[g_idx]
            rot_22 = shape_buffer.transform.rotation.rot_22[g_idx]

            tr_x = shape_buffer.transform.translation.x[g_idx]
            tr_y = shape_buffer.transform.translation.y[g_idx]
            tr_z = shape_buffer.transform.translation.z[g_idx]

            # local_pos = W * R + T (assuming total_matrix is applied like W_vec * Matrix)
            # R is 3x3, T is translation
            l_pos_x = rot_00 * w_pos_x + rot_01 * w_pos_y + rot_02 * w_pos_z + tr_x
            l_pos_y = rot_10 * w_pos_x + rot_11 * w_pos_y + rot_12 * w_pos_z + tr_y
            l_pos_z = rot_20 * w_pos_x + rot_21 * w_pos_y + rot_22 * w_pos_z + tr_z

            l_dir_x = rot_00 * w_dir_x + rot_01 * w_dir_y + rot_02 * w_dir_z
            l_dir_y = rot_10 * w_dir_x + rot_11 * w_dir_y + rot_12 * w_dir_z
            l_dir_z = rot_20 * w_dir_x + rot_21 * w_dir_y + rot_22 * w_dir_z

            # 2. Geometry Branching (Fat Node / Generic Parameters)
            geom_type = shape_buffer.shape[g_idx]
            param_0 = shape_buffer.shape_parameters.param_0[g_idx]
            param_1 = shape_buffer.shape_parameters.param_1[g_idx]
            param_2 = shape_buffer.shape_parameters.param_2[g_idx]
            param_3 = shape_buffer.shape_parameters.param_3[g_idx]

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
                g_idx = shape_buffer.miss_index[g_idx]
                continue

            # Case 2: Hit from OUTSIDE
            elif tmin > 0:
                dist = tmin + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = shape_buffer.material_index[g_idx]

                # We do NOT check children, because they are inside this parent.
                # The closest boundary is the entry to this parent.
                g_idx = shape_buffer.miss_index[g_idx]
                continue

            # Case 3: Hit from INSIDE (tmin < 0 and tmax > 0)
            else:
                dist = tmax + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = shape_buffer.material_index[g_idx]

                # Particle is inside parent, moving towards exit (tmax).
                # The ray might hit children on the way to the exit.
                # Fall through to children.
                g_idx += 1

        out_distances[j] = closest_dist
        out_volume_indices[j] = closest_vol
