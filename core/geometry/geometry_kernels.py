"""
Модуль низкоуровневых Numba-кернелов аналитической трассировки лучей (Raycasting)
и навигации по скомпилированному буферу геометрии (GeometryBuffer).
"""

from typing import Any, Tuple
import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.geometry.navigation_state import NavigationState
from core.other.typing_definitions import Float, Index
from core.other.vectors import Vector3DSoA

GEOMETRY_EPSILON: Float = Float(1e-9)
SQRT_THREE: Float = Float(1.7320508075688772)
_INF: Float = Float(np.inf)
_NEG_INF: Float = Float(-np.inf)
_ZERO: Float = Float(0.0)


@njit(cache=True)
def _box_intersect(
    pos_x: Float,
    pos_y: Float,
    pos_z: Float,
    dir_x: Float,
    dir_y: Float,
    dir_z: Float,
    half_x: Float,
    half_y: Float,
    half_z: Float,
) -> Tuple[Float, Float]:
    """
    Вычисляет аналитическое пересечение луча с выровненным по осям параллелепипедом (AABB Box)
    с центром в начале локальных координат.
    Возвращает кортеж (tmin, tmax).
    """
    # Ось X
    if dir_x != 0.0:
        inv_dir_x = Float(1.0 / dir_x)
        tx1 = Float((-half_x - pos_x) * inv_dir_x)
        tx2 = Float((half_x - pos_x) * inv_dir_x)
        tmin_x = min(tx1, tx2)
        tmax_x = max(tx1, tx2)
    else:
        if abs(pos_x) > half_x:
            return _INF, _NEG_INF
        tmin_x = _NEG_INF
        tmax_x = _INF

    # Ось Y
    if dir_y != 0.0:
        inv_dir_y = Float(1.0 / dir_y)
        ty1 = Float((-half_y - pos_y) * inv_dir_y)
        ty2 = Float((half_y - pos_y) * inv_dir_y)
        tmin_y = min(ty1, ty2)
        tmax_y = max(ty1, ty2)
    else:
        if abs(pos_y) > half_y:
            return _INF, _NEG_INF
        tmin_y = _NEG_INF
        tmax_y = _INF

    # Ось Z
    if dir_z != 0.0:
        inv_dir_z = Float(1.0 / dir_z)
        tz1 = Float((-half_z - pos_z) * inv_dir_z)
        tz2 = Float((half_z - pos_z) * inv_dir_z)
        tmin_z = min(tz1, tz2)
        tmax_z = max(tz1, tz2)
    else:
        if abs(pos_z) > half_z:
            return _INF, _NEG_INF
        tmin_z = _NEG_INF
        tmax_z = _INF

    tmin = max(tmin_x, tmin_y, tmin_z)
    tmax = min(tmax_x, tmax_y, tmax_z)

    return tmin, tmax


@njit(cache=True)
def _transform_to_local(
    world_pos_x: Float,
    world_pos_y: Float,
    world_pos_z: Float,
    world_dir_x: Float,
    world_dir_y: Float,
    world_dir_z: Float,
    transform: Any,
) -> Tuple[Float, Float, Float, Float, Float, Float]:
    """
    Преобразует мировые координаты (позицию и направление) в локальную систему координат объема
    с использованием развернутого матричного умножения для векторизации в Numba.
    """
    rotation = transform['rotation']
    translation = transform['translation']

    m00 = Float(rotation['m00'])
    m01 = Float(rotation['m01'])
    m02 = Float(rotation['m02'])
    m10 = Float(rotation['m10'])
    m11 = Float(rotation['m11'])
    m12 = Float(rotation['m12'])
    m20 = Float(rotation['m20'])
    m21 = Float(rotation['m21'])
    m22 = Float(rotation['m22'])

    trans_x = Float(translation['x'])
    trans_y = Float(translation['y'])
    trans_z = Float(translation['z'])

    local_pos_x = Float(m00 * world_pos_x + m01 * world_pos_y + m02 * world_pos_z + trans_x)
    local_pos_y = Float(m10 * world_pos_x + m11 * world_pos_y + m12 * world_pos_z + trans_y)
    local_pos_z = Float(m20 * world_pos_x + m21 * world_pos_y + m22 * world_pos_z + trans_z)

    local_dir_x = Float(m00 * world_dir_x + m01 * world_dir_y + m02 * world_dir_z)
    local_dir_y = Float(m10 * world_dir_x + m11 * world_dir_y + m12 * world_dir_z)
    local_dir_z = Float(m20 * world_dir_x + m21 * world_dir_y + m22 * world_dir_z)

    return local_pos_x, local_pos_y, local_pos_z, local_dir_x, local_dir_y, local_dir_z


@njit(cache=True)
def _hex_prism_intersect(
    local_pos_x: Float,
    local_pos_y: Float,
    local_pos_z: Float,
    local_dir_x: Float,
    local_dir_y: Float,
    local_dir_z: Float,
    shape_data: Any,
) -> Tuple[Float, Float]:
    """
    Вычисляет аналитическое пересечение луча с периодической гексагональной призмой
    для алгоритма Single-Pass Painter's Algorithm.

    Возвращает (time_min, time_max):
    - Если точка внутри шестиугольного канала (Air/Vacuum):
      time_min = 0.0, time_max = time_next (расстояние до стенки гексагона или торца Z).
    - Если точка в септе (Pb):
      time_min = time_next (расстояние до входа в канал, границы ячейки или торца Z),
      time_max = np.inf.
    """
    x_period = Float(shape_data['param_0'])
    y_period = Float(shape_data['param_1'])
    channel_half_width = Float(shape_data['param_2'])
    channel_side_limit = Float(shape_data['param_3'])
    half_height_z = Float(shape_data['param_4'])
    cell_half_width = Float(shape_data['param_5'])
    cell_side_limit = x_period

    # 1. Аналитическое пересечение с торцами призмы по оси Z
    if local_dir_z != 0.0:
        inv_dir_z = Float(1.0 / local_dir_z)
        time_z_near = Float((-half_height_z - local_pos_z) * inv_dir_z)
        time_z_far = Float((half_height_z - local_pos_z) * inv_dir_z)
        time_enter_z = min(time_z_near, time_z_far)
        time_exit_z = max(time_z_near, time_z_far)
    else:
        if abs(local_pos_z) > half_height_z:
            return _INF, _NEG_INF
        time_enter_z = _NEG_INF
        time_exit_z = _INF

    # Если луч не попадает в интервал Z или направлен в противоположную сторону
    if time_exit_z <= 0.0 or time_exit_z <= time_enter_z:
        return _INF, _NEG_INF

    # Смещение при входе снаружи по оси Z
    time_offset: Float = _ZERO
    eval_pos_x: Float = local_pos_x
    eval_pos_y: Float = local_pos_y
    if time_enter_z > 0.0:
        time_offset = time_enter_z
        eval_pos_x = Float(local_pos_x + local_dir_x * time_enter_z)
        eval_pos_y = Float(local_pos_y + local_dir_y * time_enter_z)

    remaining_exit_z = Float(time_exit_z - time_offset)

    # 2. Определение ближайшего центра гексагона в центрированной периодической решетке
    # Подрешетка 1: узлы (i * x_period, j * y_period)
    index_x_first = np.round(eval_pos_x / x_period)
    index_y_first = np.round(eval_pos_y / y_period)
    center_x_first = Float(index_x_first * x_period)
    center_y_first = Float(index_y_first * y_period)
    delta_x_first = Float(eval_pos_x - center_x_first)
    delta_y_first = Float(eval_pos_y - center_y_first)
    dist_sq_first = Float(delta_x_first * delta_x_first + delta_y_first * delta_y_first)

    # Подрешетка 2: узлы ((i + 0.5) * x_period, (j + 0.5) * y_period)
    shift_x = Float(0.5 * x_period)
    shift_y = Float(0.5 * y_period)
    index_x_second = np.round((eval_pos_x - shift_x) / x_period)
    index_y_second = np.round((eval_pos_y - shift_y) / y_period)
    center_x_second = Float(index_x_second * x_period + shift_x)
    center_y_second = Float(index_y_second * y_period + shift_y)
    delta_x_second = Float(eval_pos_x - center_x_second)
    delta_y_second = Float(eval_pos_y - center_y_second)
    dist_sq_second = Float(delta_x_second * delta_x_second + delta_y_second * delta_y_second)

    if dist_sq_first <= dist_sq_second:
        cell_pos_x = delta_x_first
        cell_pos_y = delta_y_first
    else:
        cell_pos_x = delta_x_second
        cell_pos_y = delta_y_second

    # 3. Проверка попадания расчетной точки внутрь шестиугольного канала
    abs_cell_x = abs(cell_pos_x)
    abs_cell_y = abs(cell_pos_y)
    is_inside_channel = (abs_cell_x <= channel_half_width) and (
        abs_cell_x + SQRT_THREE * abs_cell_y <= channel_side_limit
    )

    # 4. Аналитическое вычисление пересечения с 3 парами плоскостей (слэбов) канала
    # Слэб 1: вертикальные грани x = +/- channel_half_width
    pos_u1 = cell_pos_x
    dir_v1 = local_dir_x
    bound_1 = channel_half_width
    if dir_v1 != 0.0:
        inv_v1 = Float(1.0 / dir_v1)
        time_slab_1a = Float((-bound_1 - pos_u1) * inv_v1)
        time_slab_1b = Float((bound_1 - pos_u1) * inv_v1)
        time_enter_1 = min(time_slab_1a, time_slab_1b)
        time_exit_1 = max(time_slab_1a, time_slab_1b)
    else:
        if abs(pos_u1) > bound_1:
            time_enter_1 = _INF
            time_exit_1 = _NEG_INF
        else:
            time_enter_1 = _NEG_INF
            time_exit_1 = _INF

    # Слэб 2: наклонные грани (x + sqrt(3)*y) = +/- channel_side_limit
    pos_u2 = Float(cell_pos_x + SQRT_THREE * cell_pos_y)
    dir_v2 = Float(local_dir_x + SQRT_THREE * local_dir_y)
    bound_2 = channel_side_limit
    if dir_v2 != 0.0:
        inv_v2 = Float(1.0 / dir_v2)
        time_slab_2a = Float((-bound_2 - pos_u2) * inv_v2)
        time_slab_2b = Float((bound_2 - pos_u2) * inv_v2)
        time_enter_2 = min(time_slab_2a, time_slab_2b)
        time_exit_2 = max(time_slab_2a, time_slab_2b)
    else:
        if abs(pos_u2) > bound_2:
            time_enter_2 = _INF
            time_exit_2 = _NEG_INF
        else:
            time_enter_2 = _NEG_INF
            time_exit_2 = _INF

    # Слэб 3: наклонные грани (x - sqrt(3)*y) = +/- channel_side_limit
    pos_u3 = Float(cell_pos_x - SQRT_THREE * cell_pos_y)
    dir_v3 = Float(local_dir_x - SQRT_THREE * local_dir_y)
    bound_3 = channel_side_limit
    if dir_v3 != 0.0:
        inv_v3 = Float(1.0 / dir_v3)
        time_slab_3a = Float((-bound_3 - pos_u3) * inv_v3)
        time_slab_3b = Float((bound_3 - pos_u3) * inv_v3)
        time_enter_3 = min(time_slab_3a, time_slab_3b)
        time_exit_3 = max(time_slab_3a, time_slab_3b)
    else:
        if abs(pos_u3) > bound_3:
            time_enter_3 = _INF
            time_exit_3 = _NEG_INF
        else:
            time_enter_3 = _NEG_INF
            time_exit_3 = _INF

    time_enter_channel = max(time_enter_1, time_enter_2, time_enter_3)
    time_exit_channel = min(time_exit_1, time_exit_2, time_exit_3)

    # 5. Аналитическое вычисление расстояния до границы ячейки периодичности (ячейки Вороного)
    cell_bound_1 = cell_half_width
    cell_bound_2 = cell_side_limit
    cell_bound_3 = cell_side_limit

    if dir_v1 != 0.0:
        inv_v1 = Float(1.0 / dir_v1)
        time_cell_1a = Float((-cell_bound_1 - pos_u1) * inv_v1)
        time_cell_1b = Float((cell_bound_1 - pos_u1) * inv_v1)
        time_exit_cell_1 = max(time_cell_1a, time_cell_1b)
    else:
        time_exit_cell_1 = _INF

    if dir_v2 != 0.0:
        inv_v2 = Float(1.0 / dir_v2)
        time_cell_2a = Float((-cell_bound_2 - pos_u2) * inv_v2)
        time_cell_2b = Float((cell_bound_2 - pos_u2) * inv_v2)
        time_exit_cell_2 = max(time_cell_2a, time_cell_2b)
    else:
        time_exit_cell_2 = _INF

    if dir_v3 != 0.0:
        inv_v3 = Float(1.0 / dir_v3)
        time_cell_3a = Float((-cell_bound_3 - pos_u3) * inv_v3)
        time_cell_3b = Float((cell_bound_3 - pos_u3) * inv_v3)
        time_exit_cell_3 = max(time_cell_3a, time_cell_3b)
    else:
        time_exit_cell_3 = _INF

    time_exit_cell = min(time_exit_cell_1, time_exit_cell_2, time_exit_cell_3)

    # 6. Формирование tmin и tmax в соответствии с контрактом Single-Pass Painter's Algorithm
    if is_inside_channel:
        time_next = min(time_exit_channel, remaining_exit_z)
        if time_next < GEOMETRY_EPSILON:
            time_next = GEOMETRY_EPSILON
        return time_offset, Float(time_offset + time_next)
    else:
        channel_is_hit = (
            (time_enter_channel <= time_exit_channel)
            and (time_exit_channel > GEOMETRY_EPSILON)
            and (time_enter_channel >= -GEOMETRY_EPSILON)
        )
        time_to_channel = max(time_enter_channel, GEOMETRY_EPSILON) if channel_is_hit else _INF
        time_to_cell_boundary = time_exit_cell if time_exit_cell > GEOMETRY_EPSILON else _INF
        time_to_z = remaining_exit_z if remaining_exit_z > GEOMETRY_EPSILON else _INF

        time_next = min(time_to_channel, time_to_cell_boundary, time_to_z)
        if time_next < GEOMETRY_EPSILON:
            time_next = GEOMETRY_EPSILON

        if time_offset > 0.0:
            if channel_is_hit:
                return (
                    Float(time_offset + max(time_enter_channel, _ZERO)),
                    Float(time_offset + min(time_exit_channel, remaining_exit_z)),
                )
            elif time_next < remaining_exit_z:
                return Float(time_offset + time_next), _INF
            else:
                return _INF, _NEG_INF

        return time_next, _INF


@njit(cache=True)
def _get_intersection(
    local_pos_x: Float,
    local_pos_y: Float,
    local_pos_z: Float,
    local_dir_x: Float,
    local_dir_y: Float,
    local_dir_z: Float,
    shape_data: Any,
) -> Tuple[Float, Float]:
    """
    Диспетчер функций аналитического пересечения луча с геометрическим примитивом по коду формы shape_id.
    Возвращает кортеж (tmin, tmax) расстояний до входа и выхода из объема.
    """
    shape_id = shape_data['shape']
    param_0 = Float(shape_data['param_0'])
    param_1 = Float(shape_data['param_1'])
    param_2 = Float(shape_data['param_2'])

    if shape_id == 0:  # Box
        return _box_intersect(
            local_pos_x,
            local_pos_y,
            local_pos_z,
            local_dir_x,
            local_dir_y,
            local_dir_z,
            param_0,
            param_1,
            param_2,
        )
    elif shape_id == 1:  # PeriodicHexPrism
        return _hex_prism_intersect(
            local_pos_x,
            local_pos_y,
            local_pos_z,
            local_dir_x,
            local_dir_y,
            local_dir_z,
            shape_data,
        )
    else:
        return _INF, _NEG_INF


@njit(cache=True, inline='always')
def _trace_single_ray(
    world_pos_x: Float,
    world_pos_y: Float,
    world_pos_z: Float,
    world_dir_x: Float,
    world_dir_y: Float,
    world_dir_z: Float,
    geom_buffer: NDArray[Any],
) -> Tuple[Float, Index]:
    """
    Однопроходный алгоритм художника (Single-Pass Painter's Algorithm) с отсечением ветвей (Frustum Culling)
    для трассировки одного луча по скомпилированному буферу геометрии.
    Возвращает кортеж (closest_dist, current_volume_idx).
    """
    closest_dist: Float = _INF
    current_vol: Index = Index(-1)

    geom_idx = 0
    buffer_len = geom_buffer.shape[0]

    while geom_idx < buffer_len:
        geom = geom_buffer[geom_idx]

        # Преобразование Мировые -> Локальные координаты
        local_pos_x, local_pos_y, local_pos_z, local_dir_x, local_dir_y, local_dir_z = _transform_to_local(
            world_pos_x,
            world_pos_y,
            world_pos_z,
            world_dir_x,
            world_dir_y,
            world_dir_z,
            geom['transform'],
        )

        # Вычисление пересечения tmin, tmax
        tmin, tmax = _get_intersection(
            local_pos_x,
            local_pos_y,
            local_pos_z,
            local_dir_x,
            local_dir_y,
            local_dir_z,
            geom['shape_data'],
        )

        # --- Single-Pass Painter's Algorithm с Frustum Culling ---
        if tmax <= 0.0 or tmax <= tmin:
            # Промах: луч не пересекает объем или объем позади луча — пропускаем все поддерево через miss_index
            geom_idx = int(geom['miss_index'])

        elif tmin <= 0.0:
            # Внутри: частица находится внутри текущего объема
            current_vol = Index(geom['volume_index'])
            if tmax < closest_dist:
                closest_dist = tmax
            # Проверяем дочерние объемы (они имеют более высокий приоритет Z-order)
            geom_idx += 1

        else:
            # Снаружи: частица вне объема, но луч войдет в него на расстоянии tmin > 0
            if tmin < closest_dist:
                closest_dist = tmin

            # Пропускаем дочерние объемы, так как частица еще не вошла в родительский объем
            geom_idx = int(geom['miss_index'])

    return closest_dist, current_vol


@njit(cache=True)
def cast_path_kernel(
    positions: Vector3DSoA,
    directions: Vector3DSoA,
    target_indices: NDArray[Index],
    geom_buffer: NDArray[Any],
    nav_state: NavigationState,
) -> None:
    """
    Numba-кернел пакетной трассировки лучей для массива активных частиц по буферу GeometryBuffer (AoS).
    Обновляет состояние навигации NavigationState (current_volume, boundary_distance).
    """
    num_particles = target_indices.shape[0]

    pos_x = positions.x
    pos_y = positions.y
    pos_z = positions.z
    dir_x = directions.x
    dir_y = directions.y
    dir_z = directions.z

    bound_dist = nav_state.boundary_distance
    nav_curr_vol = nav_state.current_volume

    for j in range(num_particles):
        particle_idx = target_indices[j]

        if bound_dist[particle_idx] > 0.0:
            continue

        closest_dist, current_vol = _trace_single_ray(
            pos_x[particle_idx],
            pos_y[particle_idx],
            pos_z[particle_idx],
            dir_x[particle_idx],
            dir_y[particle_idx],
            dir_z[particle_idx],
            geom_buffer,
        )

        bound_dist[particle_idx] = closest_dist
        nav_curr_vol[particle_idx] = current_vol


__all__ = [
    "GEOMETRY_EPSILON",
    "SQRT_THREE",
    "_box_intersect",
    "_transform_to_local",
    "_hex_prism_intersect",
    "_get_intersection",
    "_trace_single_ray",
    "cast_path_kernel",
]

