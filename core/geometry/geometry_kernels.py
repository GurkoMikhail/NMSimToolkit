import numpy as np
from numba import njit
from typing import Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.other.vectors_soa import Vector3DSoA
from core.geometry.navigation_state import NavigationState

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
    nav_state: NavigationState
) -> None:
    """
    Raycasting Numba kernel over an array of target active particles against a structured GeometryBuffer array (AoS).
    Applies loop unrolling for coordinate transformations and uses miss_index for Boundary Tracking / Frustum Culling.
    Updates NavigationState directly for Woodcock Tracking (current_volume, next_volume, boundary_distance).
    """
    num_particles = target_indices.shape[0]
    num_geoms = geom_buffer.shape[0]

    for j in range(num_particles):
        p_idx = target_indices[j]

        # Ray Distance Caching
        if nav_state.boundary_distance[p_idx] > 0.0:
            continue

        # Stateful Navigation & Relocation
        curr_vol_idx = nav_state.current_volume[p_idx]
        next_vol_idx = nav_state.next_volume[p_idx]

        if next_vol_idx >= 0:
            # Bottom-Up Relocation with Coplanar Boundary Epsilon Overshoot check
            # For this simplified prototype, we assign current to next.
            curr_vol_idx = next_vol_idx

        start_idx = 0
        end_idx = num_geoms
        if curr_vol_idx >= 0:
            start_idx = curr_vol_idx
            end_idx = geom_buffer[curr_vol_idx]['miss_index']

        # Original World Position and Direction
        w_pos_x = positions.x[p_idx]
        w_pos_y = positions.y[p_idx]
        w_pos_z = positions.z[p_idx]

        w_dir_x = directions.x[p_idx]
        w_dir_y = directions.y[p_idx]
        w_dir_z = directions.z[p_idx]

        closest_dist = np.inf
        current_vol = -1
        next_vol = -1

        g_idx = start_idx
        while g_idx < end_idx:
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
            # We need to return CURRENT volume and distance to NEXT boundary.

            if tmax < 0 or tmax < tmin:
                # Miss
                g_idx = geom['miss_index']
                continue

            if tmin <= 0 and tmax > 0:
                # INSIDE the volume
                # distance to NEXT boundary is tmax
                dist = tmax + distance_epsilon

                # We update current volume
                # Since children are strictly inside parents, the last volume we find ourselves inside
                # (deepest in the tree) is the actual current volume.
                # Because we traverse parents before children, and only fall through if inside parent.
                current_vol = geom['volume_index']
                if dist < closest_dist:
                    closest_dist = dist
                    next_vol = -1 # we don't necessarily know the next volume, but the distance is minimal

                # Check children because we might be inside a child too
                g_idx += 1

            elif tmin > 0:
                # OUTSIDE the volume, but ray hits it in the future.
                # This could be the NEXT boundary if we are in some parent.
                dist = tmin + distance_epsilon
                if dist < closest_dist:
                    closest_dist = dist
                    next_vol = geom['volume_index']

                # Skip checking children, because they are further away than the entry to this parent.
                g_idx = geom['miss_index']

            else:
                # Miss
                g_idx = geom['miss_index']

        nav_state.boundary_distance[p_idx] = closest_dist
        nav_state.current_volume[p_idx] = current_vol
        nav_state.next_volume[p_idx] = next_vol
        # Ray distance caching will be decremented by the caller (propagation engine)
