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
    world_pos_x: Float, world_pos_y: Float, world_pos_z: Float,
    world_dir_x: Float, world_dir_y: Float, world_dir_z: Float,
    transform: NDArray
) -> Tuple[Float, Float, Float, Float, Float, Float]:
    """
    Transforms world coordinates (position and direction) into the local coordinate system
    of a volume using loop-unrolled matrix multiplications to maximize JIT vectorization.
    """
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

    trans_x = translation['x']
    trans_y = translation['y']
    trans_z = translation['z']

    local_pos_x = m00 * world_pos_x + m01 * world_pos_y + m02 * world_pos_z + trans_x
    local_pos_y = m10 * world_pos_x + m11 * world_pos_y + m12 * world_pos_z + trans_y
    local_pos_z = m20 * world_pos_x + m21 * world_pos_y + m22 * world_pos_z + trans_z

    local_dir_x = m00 * world_dir_x + m01 * world_dir_y + m02 * world_dir_z
    local_dir_y = m10 * world_dir_x + m11 * world_dir_y + m12 * world_dir_z
    local_dir_z = m20 * world_dir_x + m21 * world_dir_y + m22 * world_dir_z

    return local_pos_x, local_pos_y, local_pos_z, local_dir_x, local_dir_y, local_dir_z


@njit(cache=True)
def _get_intersection(
    local_pos_x: Float, local_pos_y: Float, local_pos_z: Float,
    local_dir_x: Float, local_dir_y: Float, local_dir_z: Float,
    shape_data: NDArray
) -> Tuple[Float, Float]:
    """
    Dispatcher for intersection functions based on shape_id.
    Returns (tmin, tmax) indicating distances to entry and exit.
    """
    shape_id = shape_data['shape']
    param_0 = shape_data['param_0']
    param_1 = shape_data['param_1']
    param_2 = shape_data['param_2']

    if shape_id == 0:  # Box
        return _box_intersect(
            local_pos_x, local_pos_y, local_pos_z,
            local_dir_x, local_dir_y, local_dir_z,
            param_0, param_1, param_2
        )
    else:
        # Fallback for undefined geometry
        return np.inf, -np.inf


@njit(cache=True)
def _relocate_bottom_up(
    world_pos_x: Float, world_pos_y: Float, world_pos_z: Float,
    world_dir_x: Float, world_dir_y: Float, world_dir_z: Float,
    curr_vol_idx: Index,
    geom_buffer: NDArray
) -> Index:
    """
    Performs rigorous Bottom-Up geometric validation to resolve Coplanar Boundary Epsilon Overshoot.
    Traces the parent hierarchy upward until it finds a volume that strictly contains the ray origin.
    Returns the corrected index of the actual volume containing the point.
    """
    while curr_vol_idx >= 0:
        geom = geom_buffer[curr_vol_idx]
        local_pos_x, local_pos_y, local_pos_z, local_dir_x, local_dir_y, local_dir_z = _transform_to_local(
            world_pos_x, world_pos_y, world_pos_z,
            world_dir_x, world_dir_y, world_dir_z,
            geom['transform']
        )
        tmin, tmax = _get_intersection(
            local_pos_x, local_pos_y, local_pos_z,
            local_dir_x, local_dir_y, local_dir_z,
            geom['shape_data']
        )

        # Check if inside
        if tmin <= 0.0 and tmax > 0.0:
            return curr_vol_idx

        curr_vol_idx = geom['parent_index']

    return -1


@njit(cache=True, inline='always')
def _trace_single_ray(
    world_pos_x: Float, world_pos_y: Float, world_pos_z: Float,
    world_dir_x: Float, world_dir_y: Float, world_dir_z: Float,
    curr_vol_idx: Index,
    geom_buffer: NDArray
) -> Tuple[Float, Index, Index]:
    """
    Device Function encapsulating the mathematical core of raycasting over the flattened scene graph.
    Designed to be fully inlined (Zero-Cost Abstraction) into the dispatcher.
    Returns (closest_dist, current_volume_idx, next_volume_idx).
    """

    # === [1] RELOCATION (BOTTOM-UP) ===
    # If the particle has a known volume, verify if it's still inside (due to Coplanar Boundary Epsilon Overshoot).
    if curr_vol_idx >= 0:
        curr_vol_idx = _relocate_bottom_up(
            world_pos_x, world_pos_y, world_pos_z,
            world_dir_x, world_dir_y, world_dir_z,
            curr_vol_idx,
            geom_buffer
        )

    # === [2] INITIALIZATION ===
    # If the particle exited the root volume (curr_vol_idx == -1 after relocation),
    # there is no need to search from 0 again because it's completely outside the world.
    # But if it's freshly injected (initial curr_vol_idx == -1), we must search from root.
    if curr_vol_idx >= 0:
        # Search restricted to current volume and its children
        search_start_idx = curr_vol_idx
        search_end_idx = geom_buffer[curr_vol_idx]['miss_index']
    else:
        # Full scene search (for newly injected particles)
        search_start_idx = 0
        search_end_idx = geom_buffer.shape[0]

    closest_dist = np.inf
    current_vol = -1
    next_vol = -1

    # === [3] RAY TRAVERSAL LOOP ===
    g_idx = search_start_idx
    while g_idx < search_end_idx:
        geom = geom_buffer[g_idx]

        # Transform World -> Local
        local_pos_x, local_pos_y, local_pos_z, local_dir_x, local_dir_y, local_dir_z = _transform_to_local(
            world_pos_x, world_pos_y, world_pos_z,
            world_dir_x, world_dir_y, world_dir_z,
            geom['transform']
        )

        # Calculate intersection tmin, tmax
        tmin, tmax = _get_intersection(
            local_pos_x, local_pos_y, local_pos_z,
            local_dir_x, local_dir_y, local_dir_z,
            geom['shape_data']
        )

        # --- Frustum Culling / Boundary Tracking Logic ---
        if tmax < 0 or tmax < tmin:
            # MISS: Ray completely misses this volume.
            # Jump over all its children via miss_index.
            g_idx = geom['miss_index']
            continue

        if tmin <= 0 and tmax > 0:
            # INSIDE: The particle is currently inside this volume.
            # The next boundary is its exit (tmax).
            if tmax < closest_dist:
                closest_dist = tmax
                # We don't know the exact next volume at this level, so reset it.
                next_vol = -1

            # Record as current volume. Because we traverse parent->child,
            # the deepest child we find ourselves in will overwrite this.
            current_vol = geom['volume_index']

            # Fall through to check children
            g_idx += 1

        elif tmin > 0:
            # OUTSIDE: The particle is outside, but the ray will hit it (tmin).
            # This is a candidate for the next volume boundary.
            if tmin < closest_dist:
                closest_dist = tmin
                next_vol = geom['volume_index']

            # We hit the outer boundary, but we are not inside.
            # We do NOT check children because they are further away (inside this volume).
            g_idx = geom['miss_index']

        else:
            # Edge case fallback (e.g. point exactly on surface and pointing away)
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
