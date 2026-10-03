import numpy as np
from numba import njit, prange
from typing import Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.other.vectors import Vector3DSoA
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
def _hex_prism_intersect(
    local_pos_x: Float, local_pos_y: Float, local_pos_z: Float,
    local_dir_x: Float, local_dir_y: Float, local_dir_z: Float,
    shape_data: NDArray
) -> Tuple[Float, Float]:
    """
    Вычисляет аналитическое пересечение луча с периодической гексагональной призмой
    для алгоритма Single-Pass Painter's Algorithm.

    Возвращает (time_min, time_max):
    - Если точка внутри шестиугольного канала (Air):
      time_min = 0.0, time_max = time_next (расстояние до стенки гексагона или торца Z).
    - Если точка в септе (Pb):
      time_min = time_next (расстояние до входа в канал, границы ячейки или торца Z),
      time_max = np.inf.
    """
    x_period = shape_data['param_0']
    y_period = shape_data['param_1']
    channel_half_width = shape_data['param_2']
    channel_side_limit = shape_data['param_3']
    half_height_z = shape_data['param_4']
    cell_half_width = shape_data['param_5']
    cell_side_limit = x_period

    # 1. Аналитическое пересечение с торцами призмы по оси Z
    if local_dir_z != 0.0:
        inv_dir_z = 1.0 / local_dir_z
        time_z_near = (-half_height_z - local_pos_z) * inv_dir_z
        time_z_far = (half_height_z - local_pos_z) * inv_dir_z
        time_enter_z = min(time_z_near, time_z_far)
        time_exit_z = max(time_z_near, time_z_far)
    else:
        if abs(local_pos_z) > half_height_z:
            return np.inf, -np.inf
        time_enter_z = -np.inf
        time_exit_z = np.inf

    # Если луч не попадает в интервал Z или направлен в противоположную сторону
    if time_exit_z <= 0.0 or time_exit_z <= time_enter_z:
        return np.inf, -np.inf

    # Смещение при входе снаружи по оси Z
    time_offset = 0.0
    eval_pos_x = local_pos_x
    eval_pos_y = local_pos_y
    if time_enter_z > 0.0:
        time_offset = time_enter_z
        eval_pos_x = local_pos_x + local_dir_x * time_enter_z
        eval_pos_y = local_pos_y + local_dir_y * time_enter_z

    remaining_exit_z = time_exit_z - time_offset

    # 2. Определение ближайшего центра гексагона в центрированной периодической решетке
    # Подрешетка 1: узлы (i * x_period, j * y_period)
    index_x_first = round(eval_pos_x / x_period)
    index_y_first = round(eval_pos_y / y_period)
    center_x_first = index_x_first * x_period
    center_y_first = index_y_first * y_period
    delta_x_first = eval_pos_x - center_x_first
    delta_y_first = eval_pos_y - center_y_first
    dist_sq_first = delta_x_first * delta_x_first + delta_y_first * delta_y_first

    # Подрешетка 2: узлы ((i + 0.5) * x_period, (j + 0.5) * y_period)
    shift_x = 0.5 * x_period
    shift_y = 0.5 * y_period
    index_x_second = round((eval_pos_x - shift_x) / x_period)
    index_y_second = round((eval_pos_y - shift_y) / y_period)
    center_x_second = index_x_second * x_period + shift_x
    center_y_second = index_y_second * y_period + shift_y
    delta_x_second = eval_pos_x - center_x_second
    delta_y_second = eval_pos_y - center_y_second
    dist_sq_second = delta_x_second * delta_x_second + delta_y_second * delta_y_second

    if dist_sq_first <= dist_sq_second:
        cell_pos_x = delta_x_first
        cell_pos_y = delta_y_first
    else:
        cell_pos_x = delta_x_second
        cell_pos_y = delta_y_second

    # 3. Проверка попадания расчетной точки внутрь шестиугольного канала
    sqrt_three = 1.7320508075688772
    abs_cell_x = abs(cell_pos_x)
    abs_cell_y = abs(cell_pos_y)
    is_inside_channel = (abs_cell_x <= channel_half_width) and (abs_cell_x + sqrt_three * abs_cell_y <= channel_side_limit)

    # 4. Аналитическое вычисление пересечения с 3 парами плоскостей (слэбов) канала
    # Слэб 1: вертикальные грани x = +/- channel_half_width
    pos_u1 = cell_pos_x
    dir_v1 = local_dir_x
    bound_1 = channel_half_width
    if dir_v1 != 0.0:
        inv_v1 = 1.0 / dir_v1
        time_slab_1a = (-bound_1 - pos_u1) * inv_v1
        time_slab_1b = ( bound_1 - pos_u1) * inv_v1
        time_enter_1 = min(time_slab_1a, time_slab_1b)
        time_exit_1 = max(time_slab_1a, time_slab_1b)
    else:
        if abs(pos_u1) > bound_1:
            time_enter_1 = np.inf
            time_exit_1 = -np.inf
        else:
            time_enter_1 = -np.inf
            time_exit_1 = np.inf

    # Слэб 2: наклонные грани (x + sqrt(3)*y) = +/- channel_side_limit
    pos_u2 = cell_pos_x + sqrt_three * cell_pos_y
    dir_v2 = local_dir_x + sqrt_three * local_dir_y
    bound_2 = channel_side_limit
    if dir_v2 != 0.0:
        inv_v2 = 1.0 / dir_v2
        time_slab_2a = (-bound_2 - pos_u2) * inv_v2
        time_slab_2b = ( bound_2 - pos_u2) * inv_v2
        time_enter_2 = min(time_slab_2a, time_slab_2b)
        time_exit_2 = max(time_slab_2a, time_slab_2b)
    else:
        if abs(pos_u2) > bound_2:
            time_enter_2 = np.inf
            time_exit_2 = -np.inf
        else:
            time_enter_2 = -np.inf
            time_exit_2 = np.inf

    # Слэб 3: наклонные грани (x - sqrt(3)*y) = +/- channel_side_limit
    pos_u3 = cell_pos_x - sqrt_three * cell_pos_y
    dir_v3 = local_dir_x - sqrt_three * local_dir_y
    bound_3 = channel_side_limit
    if dir_v3 != 0.0:
        inv_v3 = 1.0 / dir_v3
        time_slab_3a = (-bound_3 - pos_u3) * inv_v3
        time_slab_3b = ( bound_3 - pos_u3) * inv_v3
        time_enter_3 = min(time_slab_3a, time_slab_3b)
        time_exit_3 = max(time_slab_3a, time_slab_3b)
    else:
        if abs(pos_u3) > bound_3:
            time_enter_3 = np.inf
            time_exit_3 = -np.inf
        else:
            time_enter_3 = -np.inf
            time_exit_3 = np.inf

    time_enter_channel = max(time_enter_1, time_enter_2, time_enter_3)
    time_exit_channel = min(time_exit_1, time_exit_2, time_exit_3)

    # 5. Аналитическое вычисление расстояния до границы ячейки периодичности (ячейки Вороного)
    cell_bound_1 = cell_half_width
    cell_bound_2 = cell_side_limit
    cell_bound_3 = cell_side_limit

    if dir_v1 != 0.0:
        inv_v1 = 1.0 / dir_v1
        time_cell_1a = (-cell_bound_1 - pos_u1) * inv_v1
        time_cell_1b = ( cell_bound_1 - pos_u1) * inv_v1
        time_exit_cell_1 = max(time_cell_1a, time_cell_1b)
    else:
        time_exit_cell_1 = np.inf

    if dir_v2 != 0.0:
        inv_v2 = 1.0 / dir_v2
        time_cell_2a = (-cell_bound_2 - pos_u2) * inv_v2
        time_cell_2b = ( cell_bound_2 - pos_u2) * inv_v2
        time_exit_cell_2 = max(time_cell_2a, time_cell_2b)
    else:
        time_exit_cell_2 = np.inf

    if dir_v3 != 0.0:
        inv_v3 = 1.0 / dir_v3
        time_cell_3a = (-cell_bound_3 - pos_u3) * inv_v3
        time_cell_3b = ( cell_bound_3 - pos_u3) * inv_v3
        time_exit_cell_3 = max(time_cell_3a, time_cell_3b)
    else:
        time_exit_cell_3 = np.inf

    time_exit_cell = min(time_exit_cell_1, time_exit_cell_2, time_exit_cell_3)

    # 6. Формирование tmin и tmax в соответствии с контрактом Single-Pass Painter's Algorithm
    if is_inside_channel:
        time_next = min(time_exit_channel, remaining_exit_z)
        if time_next < 1e-9:
            time_next = 1e-9
        return time_offset, time_offset + time_next
    else:
        channel_is_hit = (time_enter_channel <= time_exit_channel) and (time_enter_channel > 1e-9)
        time_to_channel = time_enter_channel if channel_is_hit else np.inf
        time_to_cell_boundary = time_exit_cell if time_exit_cell > 1e-9 else np.inf
        time_to_z = remaining_exit_z if remaining_exit_z > 1e-9 else np.inf

        time_next = min(time_to_channel, time_to_cell_boundary, time_to_z)
        if time_next < 1e-9:
            time_next = 1e-9

        if time_offset > 0.0:
            if channel_is_hit:
                return time_offset + time_enter_channel, time_offset + min(time_exit_channel, remaining_exit_z)
            else:
                return np.inf, -np.inf

        return time_next, np.inf


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
    elif shape_id == 1:  # PeriodicHexPrism
        return _hex_prism_intersect(
            local_pos_x, local_pos_y, local_pos_z,
            local_dir_x, local_dir_y, local_dir_z,
            shape_data
        )
    else:
        # Fallback for undefined geometry
        return np.inf, -np.inf


@njit(cache=True, inline='always')
def _trace_single_ray(
    world_pos_x: Float, world_pos_y: Float, world_pos_z: Float,
    world_dir_x: Float, world_dir_y: Float, world_dir_z: Float,
    geom_buffer: NDArray
) -> Tuple[Float, Index]:
    """
    Device Function encapsulating the Single-Pass Painter's Algorithm for raycasting.
    Designed to be fully inlined (Zero-Cost Abstraction) into the dispatcher.
    Returns (closest_dist, current_volume_idx).
    """

    closest_dist = np.inf
    current_vol = -1

    g_idx = 0
    buffer_len = geom_buffer.shape[0]

    while g_idx < buffer_len:
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

        # --- Single-Pass Painter's Algorithm with Frustum Culling ---
        if tmax <= 0.0 or tmax <= tmin:
            # MISS: Ray completely misses this volume or it's behind us.
            # Jump over all its children via miss_index.
            g_idx = geom['miss_index']

        elif tmin <= 0.0:
            # INSIDE: The particle is currently inside this volume.
            current_vol = geom['volume_index']
            closest_dist = tmax
            # Check children since they have higher priority (Z-order)
            g_idx += 1

        else:
            # OUTSIDE: The particle is outside, but the ray will hit it (tmin > 0).
            if tmin < closest_dist:
                closest_dist = tmin

            # Jump over children since we are not inside this volume yet
            g_idx = geom['miss_index']

    return closest_dist, current_vol


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
    Updates NavigationState directly for Woodcock Tracking (current_volume, boundary_distance).
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

    for j in prange(num_particles):
        p_idx = target_indices[j]

        if bound_dist[p_idx] > 0.0:
            continue

        closest_dist, current_vol = _trace_single_ray(
            pos_x[p_idx], pos_y[p_idx], pos_z[p_idx],
            dir_x[p_idx], dir_y[p_idx], dir_z[p_idx],
            geom_buffer
        )

        bound_dist[p_idx] = closest_dist
        nav_curr_vol[p_idx] = current_vol
