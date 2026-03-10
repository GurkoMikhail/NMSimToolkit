import numpy as np
from numba import njit, prange
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
def _transform_to_local(
    w_pos_x: Float, w_pos_y: Float, w_pos_z: Float,
    w_dir_x: Float, w_dir_y: Float, w_dir_z: Float,
    transform: NDArray
) -> Tuple[Float, Float, Float, Float, Float, Float]:
    rotation = transform['rotation']
    translation = transform['translation']

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

    return l_pos_x, l_pos_y, l_pos_z, l_dir_x, l_dir_y, l_dir_z


@njit(cache=True)
def _get_intersection(
    l_pos_x: Float, l_pos_y: Float, l_pos_z: Float,
    l_dir_x: Float, l_dir_y: Float, l_dir_z: Float,
    shape_data: NDArray
) -> Tuple[Float, Float]:
    geom_type = shape_data['shape']
    p0 = shape_data['param_0']
    p1 = shape_data['param_1']
    p2 = shape_data['param_2']

    if geom_type == 0:  # Box
        return _box_intersect(
            l_pos_x, l_pos_y, l_pos_z,
            l_dir_x, l_dir_y, l_dir_z,
            p0, p1, p2
        )
    else:
        return np.inf, -np.inf


@njit(cache=True)
def _relocate_bottom_up(
    w_pos_x: Float, w_pos_y: Float, w_pos_z: Float,
    w_dir_x: Float, w_dir_y: Float, w_dir_z: Float,
    curr_vol_idx: Index,
    geom_buffer: NDArray
) -> Index:
    while curr_vol_idx >= 0:
        geom = geom_buffer[curr_vol_idx]
        l_pos_x, l_pos_y, l_pos_z, l_dir_x, l_dir_y, l_dir_z = _transform_to_local(
            w_pos_x, w_pos_y, w_pos_z,
            w_dir_x, w_dir_y, w_dir_z,
            geom['transform']
        )
        tmin, tmax = _get_intersection(
            l_pos_x, l_pos_y, l_pos_z,
            l_dir_x, l_dir_y, l_dir_z,
            geom['shape_data']
        )

        # Check if inside
        if tmin <= 0.0 and tmax > 0.0:
            return curr_vol_idx

        curr_vol_idx = geom['parent_index']

    return -1


@njit(cache=True, inline='always')
def _trace_single_ray(
    w_pos_x: Float, w_pos_y: Float, w_pos_z: Float,
    w_dir_x: Float, w_dir_y: Float, w_dir_z: Float,
    curr_vol_idx: Index,
    geom_buffer: NDArray
) -> Tuple[Float, Index, Index]:

    num_geoms = geom_buffer.shape[0]

    if curr_vol_idx >= 0:
        curr_vol_idx = _relocate_bottom_up(
            w_pos_x, w_pos_y, w_pos_z,
            w_dir_x, w_dir_y, w_dir_z,
            curr_vol_idx,
            geom_buffer
        )

    start_idx = 0
    end_idx = num_geoms
    if curr_vol_idx >= 0:
        start_idx = curr_vol_idx
        end_idx = geom_buffer[curr_vol_idx]['miss_index']

    closest_dist = np.inf
    current_vol = np.int64(-1)
    next_vol = np.int64(-1)

    g_idx = start_idx
    while g_idx < end_idx:
        geom = geom_buffer[g_idx]

        l_pos_x, l_pos_y, l_pos_z, l_dir_x, l_dir_y, l_dir_z = _transform_to_local(
            w_pos_x, w_pos_y, w_pos_z,
            w_dir_x, w_dir_y, w_dir_z,
            geom['transform']
        )
        tmin, tmax = _get_intersection(
            l_pos_x, l_pos_y, l_pos_z,
            l_dir_x, l_dir_y, l_dir_z,
            geom['shape_data']
        )

        # 3. Frustum Culling / Boundary Tracking Logic
        # We need to return CURRENT volume and distance to NEXT boundary.

        if tmax < 0 or tmax < tmin:
            # Miss
            g_idx = geom['miss_index']
            continue

        if tmin <= 0 and tmax > 0:
            # INSIDE the volume
            # distance to NEXT boundary is tmax
            dist = tmax

            # We update current volume
            # Since children are strictly inside parents, the last volume we find ourselves inside
            # (deepest in the tree) is the actual current volume.
            # Because we traverse parents before children, and only fall through if inside parent.
            current_vol = geom['volume_index']
            if dist < closest_dist:
                closest_dist = dist
                next_vol = np.int64(-1) # we don't necessarily know the next volume, but the distance is minimal

            # Check children because we might be inside a child too
            g_idx += 1

        elif tmin > 0:
            # OUTSIDE the volume, but ray hits it in the future.
            # This could be the NEXT boundary if we are in some parent.
            dist = tmin
            if dist < closest_dist:
                closest_dist = dist
                next_vol = geom['volume_index']

            # Skip checking children, because they are further away than the entry to this parent.
            g_idx = geom['miss_index']

        else:
            # Miss
            g_idx = geom['miss_index']

    return closest_dist, current_vol, next_vol


@njit(cache=True, parallel=True, fastmath=True)
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

    # Extract arrays to avoid Numba parfor data race on NamedTuple attribute access
    pos_x = positions.x
    pos_y = positions.y
    pos_z = positions.z
    dir_x = directions.x
    dir_y = directions.y
    dir_z = directions.z

    bound_dist = nav_state.boundary_distance
    nav_curr_vol = nav_state.current_volume
    nav_next_vol = nav_state.next_volume

    for j in prange(num_particles):
        p_idx = target_indices[j]

        if bound_dist[p_idx] > 0.0:
            continue

        closest_dist, current_vol, next_vol = _trace_single_ray(
            pos_x[p_idx], pos_y[p_idx], pos_z[p_idx],
            dir_x[p_idx], dir_y[p_idx], dir_z[p_idx],
            nav_curr_vol[p_idx],
            geom_buffer
        )

        bound_dist[p_idx] = closest_dist
        nav_curr_vol[p_idx] = current_vol
        nav_next_vol[p_idx] = next_vol
