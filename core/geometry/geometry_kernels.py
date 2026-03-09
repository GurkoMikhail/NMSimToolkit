import numpy as np
from numba import njit
from typing import Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
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
    geom_buffer: NDArray,
    out_distances: NDArray[Float],
    out_volume_indices: NDArray[Index]
) -> None:
    """
    Raycasting Numba kernel over an array of target active particles against a structured GeometryBuffer array (AoS).
    Applies loop unrolling for coordinate transformations and uses miss_index for Boundary Tracking / Frustum Culling.
    """
    num_particles = target_indices.shape[0]
    num_geoms = geom_buffer.shape[0]

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
            geom = geom_buffer[g_idx]
            transform = geom['transform']
            rotation = transform['rotation']
            translation = transform['translation']

            # 1. Transform World -> Local (Loop Unrolling, without np.matmul)
            m00 = rotation['m00']
            m01 = rotation['m01']
            m02 = rotation['m02']
            m10 = rotation['m10']
            m11 = rotation['m11']
            m12 = rotation['m12']
            m20 = rotation['m20']
            m21 = rotation['m21']
            m22 = rotation['m22']

            t_x = translation['x']
            t_y = translation['y']
            t_z = translation['z']

            l_pos_x = m00 * w_pos_x + m01 * w_pos_y + m02 * w_pos_z + t_x
            l_pos_y = m10 * w_pos_x + m11 * w_pos_y + m12 * w_pos_z + t_y
            l_pos_z = m20 * w_pos_x + m21 * w_pos_y + m22 * w_pos_z + t_z

            l_dir_x = m00 * w_dir_x + m01 * w_dir_y + m02 * w_dir_z
            l_dir_y = m10 * w_dir_x + m11 * w_dir_y + m12 * w_dir_z
            l_dir_z = m20 * w_dir_x + m21 * w_dir_y + m22 * w_dir_z

            # 2. Geometry Branching (Fat Node / Generic Parameters)
            shape_data = geom['shape_data']
            geom_type = shape_data['shape']
            p0 = shape_data['param_0']
            p1 = shape_data['param_1']
            p2 = shape_data['param_2']
            p3 = shape_data['param_3']

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
                g_idx = geom['miss_index']
                continue

            # Case 2: Hit from OUTSIDE
            elif tmin > 0:
                dist = tmin + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = geom['volume_index']

                g_idx = geom['miss_index']
                continue

            # Case 3: Hit from INSIDE (tmin < 0 and tmax > 0)
            else:
                dist = tmax + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    closest_vol = geom['volume_index']

                g_idx += 1

        out_distances[j] = closest_dist
        out_volume_indices[j] = closest_vol
