"""
Модуль высокопроизводительных Numba JIT-ядер генерации и прямой инжекции
первичных частиц из SourceBuffer в ParticleBank без промежуточных аллокаций (Zero-Allocation).
"""

import math
from typing import Tuple

from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Index, Length, Species, Time
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import RNGContext
from core.source.source_buffer import SourceBuffer


@njit(cache=True, inline='always')
def _sample_alias(
    prob: NDArray[Float],
    alias: NDArray[np.int32],
    offset: Index,
    count: Index,
    rng_ctx: RNGContext,
) -> Index:
    """
    Выбирает глобальный индекс в объединенной таблице алиасов Воза за O(1).

    При count <= 1 возвращает offset без обращения к генератору случайных чисел.
    При count > 1 потребляет ровно два равномерных случайных числа из rng_ctx
    (выбор корзины и сравнение с порогом prob).
    """
    if count <= Index(1):
        return offset

    rand_u1 = rng_ctx.next_double(rng_ctx.state_addr)
    rand_u2 = rng_ctx.next_double(rng_ctx.state_addr)

    bin_local = Index(rand_u1 * Float(count))
    if bin_local >= count:
        bin_local = count - Index(1)

    table_idx = offset + bin_local
    if rand_u2 < prob[table_idx]:
        return table_idx
    return offset + Index(alias[table_idx])


@njit(cache=True, inline='always')
def _sample_voxel_position(
    flat_voxel_idx: Index,
    shape_y: Index,
    shape_z: Index,
    voxel_size: Float,
    half_size_x: Float,
    half_size_y: Float,
    half_size_z: Float,
    local_to_global: NDArray[Float],
    source_idx: Index,
    rng_ctx: RNGContext,
    cell_jitter_x: Float = Float(0.0),
    cell_jitter_y: Float = Float(0.0),
) -> Tuple[Float, Float, Float]:
    """
    Вычисляет глобальные координаты рождения частицы по плоскому индексу вокселя.

    Раскладывает flat_voxel_idx в (idx_x, idx_y, idx_z), находит левый нижний угол вокселя
    idx * voxel_size - half_size и добавляет равномерное смещение в [0.0, voxel_size)
    (строго согласовано с геометрией WoodcockVoxelVolume без сдвига на полвокселя).
    Для PointSource с voxel_size == 0.0 локальная координата равна (0, 0, 0) без
    потребления случайных чисел на воксельное смещение.
    """
    if voxel_size > 0.0:
        u_flat = np.uint32(flat_voxel_idx)
        u_ny = np.uint32(shape_y)
        u_nz = np.uint32(shape_z)
        u_ny_nz = u_ny * u_nz
        idx_x = u_flat // u_ny_nz
        rem = u_flat - idx_x * u_ny_nz
        idx_y = rem // u_nz
        idx_z = rem - idx_y * u_nz

        rand_ux = rng_ctx.next_double(rng_ctx.state_addr)
        rand_uy = rng_ctx.next_double(rng_ctx.state_addr)
        rand_uz = rng_ctx.next_double(rng_ctx.state_addr)

        local_x = Float(idx_x) * voxel_size - half_size_x + rand_ux * voxel_size
        local_y = Float(idx_y) * voxel_size - half_size_y + rand_uy * voxel_size
        local_z = Float(idx_z) * voxel_size - half_size_z + rand_uz * voxel_size
    else:
        local_x = Float(0.0)
        local_y = Float(0.0)
        local_z = Float(0.0)
        if cell_jitter_x > 0.0:
            local_x = (rng_ctx.next_double(rng_ctx.state_addr) - 0.5) * cell_jitter_x
        if cell_jitter_y > 0.0:
            local_y = (rng_ctx.next_double(rng_ctx.state_addr) - 0.5) * cell_jitter_y

    m00 = local_to_global[source_idx, 0, 0]
    m01 = local_to_global[source_idx, 0, 1]
    m02 = local_to_global[source_idx, 0, 2]
    m03 = local_to_global[source_idx, 0, 3]
    m10 = local_to_global[source_idx, 1, 0]
    m11 = local_to_global[source_idx, 1, 1]
    m12 = local_to_global[source_idx, 1, 2]
    m13 = local_to_global[source_idx, 1, 3]
    m20 = local_to_global[source_idx, 2, 0]
    m21 = local_to_global[source_idx, 2, 1]
    m22 = local_to_global[source_idx, 2, 2]
    m23 = local_to_global[source_idx, 2, 3]

    global_x = m00 * local_x + m01 * local_y + m02 * local_z + m03
    global_y = m10 * local_x + m11 * local_y + m12 * local_z + m13
    global_z = m20 * local_x + m21 * local_y + m22 * local_z + m23
    return Float(global_x), Float(global_y), Float(global_z)


@njit(cache=True, inline='always')
def _sample_isotropic_direction(rng_ctx: RNGContext) -> Tuple[Float, Float, Float]:
    """
    Генерирует изотропный единичный вектор направления равномерно на сфере 4*pi.
    Потребляет два равномерных случайных числа (для cos(theta) в [-1, 1] и phi в [0, 2*pi)).
    """
    rand_u1 = rng_ctx.next_double(rng_ctx.state_addr)
    rand_u2 = rng_ctx.next_double(rng_ctx.state_addr)

    cos_theta = 1.0 - 2.0 * rand_u1
    sin_theta_sq = 1.0 - cos_theta * cos_theta
    sin_theta = math.sqrt(sin_theta_sq) if sin_theta_sq > 0.0 else 0.0
    phi_angle = 2.0 * math.pi * rand_u2

    dir_x = sin_theta * math.cos(phi_angle)
    dir_y = sin_theta * math.sin(phi_angle)
    dir_z = cos_theta
    return Float(dir_x), Float(dir_y), Float(dir_z)


@njit(cache=True, inline='always')
def _sample_cone_direction(
    cos_theta_min: Float,
    local_to_global: NDArray[Float],
    source_idx: Index,
    rng_ctx: RNGContext,
) -> Tuple[Float, Float, Float]:
    """
    Генерирует единичный вектор направления равномерно в конусе с cos(theta) >= cos_theta_min
    вокруг локальной оси +Z с поворотом в глобальную систему координат.
    """
    if cos_theta_min <= -1.0:
        return _sample_isotropic_direction(rng_ctx)

    rand_u1 = rng_ctx.next_double(rng_ctx.state_addr)
    rand_u2 = rng_ctx.next_double(rng_ctx.state_addr)

    cos_theta = 1.0 - rand_u1 * (1.0 - cos_theta_min)
    sin_theta_sq = 1.0 - cos_theta * cos_theta
    sin_theta = math.sqrt(sin_theta_sq) if sin_theta_sq > 0.0 else 0.0
    phi_angle = 2.0 * math.pi * rand_u2

    local_dx = sin_theta * math.cos(phi_angle)
    local_dy = sin_theta * math.sin(phi_angle)
    local_dz = cos_theta

    m00 = local_to_global[source_idx, 0, 0]
    m01 = local_to_global[source_idx, 0, 1]
    m02 = local_to_global[source_idx, 0, 2]
    m10 = local_to_global[source_idx, 1, 0]
    m11 = local_to_global[source_idx, 1, 1]
    m12 = local_to_global[source_idx, 1, 2]
    m20 = local_to_global[source_idx, 2, 0]
    m21 = local_to_global[source_idx, 2, 1]
    m22 = local_to_global[source_idx, 2, 2]

    global_dx = m00 * local_dx + m01 * local_dy + m02 * local_dz
    global_dy = m10 * local_dx + m11 * local_dy + m12 * local_dz
    global_dz = m20 * local_dx + m21 * local_dy + m22 * local_dz
    return Float(global_dx), Float(global_dy), Float(global_dz)


@njit(cache=True, inline='always')
def _write_primary(
    bank: ParticleBank,
    slot: Index,
    particle_id: ID,
    species_val: Species,
    energy_val: Energy,
    emission_time_val: Time,
    pos_x: Float,
    pos_y: Float,
    pos_z: Float,
    dir_x: Float,
    dir_y: Float,
    dir_z: Float,
) -> None:
    """
    Прямая запись состояния новой первичной частицы в слот массивов KinematicState,
    InitialState и сброс навигационного состояния NavigationState.
    """
    state = bank.state
    initial_state = bank.initial_state
    nav_state = bank.navigation_state

    state.is_active[slot] = True
    state.species[slot] = species_val
    state.energy[slot] = energy_val
    state.distance_traveled[slot] = Length(0.0)

    state.position.x[slot] = pos_x
    state.position.y[slot] = pos_y
    state.position.z[slot] = pos_z

    state.direction.x[slot] = dir_x
    state.direction.y[slot] = dir_y
    state.direction.z[slot] = dir_z

    initial_state.ID[slot] = particle_id
    initial_state.emission_time[slot] = emission_time_val
    initial_state.emission_energy[slot] = energy_val

    initial_state.emission_position.x[slot] = pos_x
    initial_state.emission_position.y[slot] = pos_y
    initial_state.emission_position.z[slot] = pos_z

    initial_state.emission_direction.x[slot] = dir_x
    initial_state.emission_direction.y[slot] = dir_y
    initial_state.emission_direction.z[slot] = dir_z

    nav_state.current_volume[slot] = Index(-1)
    nav_state.boundary_distance[slot] = Float(0.0)


@njit(cache=True, inline='always')
def _emit_primary_from_source(
    bank: ParticleBank,
    source_buffer: SourceBuffer,
    source_idx: Index,
    slot: Index,
    particle_id: ID,
    emission_time_val: Time,
    rng_ctx: RNGContext,
) -> None:
    """
    Генерирует первичную частицу для выбранного источника source_idx с заданным
    временем испускания emission_time_val и записывает её напрямую в слот slot банка частиц.

    Порядок потребления случайных чисел из rng_ctx:
    1) выбор спектральной линии методом алиасов Воза (_sample_alias);
    2) выбор активного вокселя методом алиасов Воза (_sample_alias);
    3) смещение внутри вокселя / ячейки (_sample_voxel_position);
    4) изотропное или конусное направление вылета (_sample_cone_direction).
    """
    line_offset = source_buffer.line_offsets[source_idx]
    line_count = source_buffer.line_offsets[source_idx + Index(1)] - line_offset
    line_table_idx = _sample_alias(
        source_buffer.line_prob,
        source_buffer.line_alias,
        line_offset,
        line_count,
        rng_ctx,
    )
    energy_val = source_buffer.line_energy[line_table_idx]

    vox_offset = source_buffer.voxel_offsets[source_idx]
    vox_count = source_buffer.voxel_offsets[source_idx + Index(1)] - vox_offset
    vox_table_idx = _sample_alias(
        source_buffer.voxel_prob,
        source_buffer.voxel_alias,
        vox_offset,
        vox_count,
        rng_ctx,
    )
    flat_voxel_idx = Index(source_buffer.voxel_index[vox_table_idx])

    pos_x, pos_y, pos_z = _sample_voxel_position(
        flat_voxel_idx,
        source_buffer.grid_shape[source_idx, 1],
        source_buffer.grid_shape[source_idx, 2],
        source_buffer.voxel_size[source_idx],
        source_buffer.half_size[source_idx, 0],
        source_buffer.half_size[source_idx, 1],
        source_buffer.half_size[source_idx, 2],
        source_buffer.local_to_global,
        source_idx,
        rng_ctx,
        source_buffer.cell_jitter[source_idx, 0],
        source_buffer.cell_jitter[source_idx, 1],
    )

    dir_x, dir_y, dir_z = _sample_cone_direction(
        source_buffer.cos_theta_min[source_idx],
        source_buffer.local_to_global,
        source_idx,
        rng_ctx,
    )

    _write_primary(
        bank,
        slot,
        particle_id,
        Species(0),
        energy_val,
        emission_time_val,
        pos_x,
        pos_y,
        pos_z,
        dir_x,
        dir_y,
        dir_z,
    )


@njit(cache=True)
def inject_poisson_sources_kernel(
    bank: ParticleBank,
    source_buffer: SourceBuffer,
    stop_time: Float,
    t_ref: Float,
    t_last: Float,
    pending_arrival_time: Float,
    rng_ctx: RNGContext,
    max_particles: Index = Index(-1),
    start_slot_cursor: Index = Index(0),
) -> Tuple[Index, Index, Float, Float]:
    """
    Генерирует и инжектирует первичные частицы из суперпозиции источников в ParticleBank
    по нестационарному пуассоновскому потоку с прореживанием Льюиса–Шедлера (Lewis–Shedler thinning).

    Мажоранта интенсивности вычисляется в опорной точке начала первого сегмента:
        A_max = sum_s A_{0,s} * exp(-lambda_s * t_ref).

    Инварианты потока случайных чисел (для побитовой независимости от batch_capacity и досчёта):
    1. Свободный слот в ParticleBank ищется ДО извлечения или генерации очередного кандидата по времени.
    2. Если передан не-NaN pending_arrival_time, он используется как кандидат t без обращения к ГСЧ.
    3. Если t >= stop_time (при конечном stop_time), кандидат сохраняется в pending_arrival_time,
       и работа ядра завершается без розыгрыша прореживания и параметров частицы.
    4. Иначе обновляется t_last = t, разыгрывается прореживание x = u_2 * A_max с выбором источника;
       при отвержении кандидата цикл продолжается для того же найденного свободного слота.

    Возвращает кортеж (injected_count, next_slot_cursor, new_t_last, new_pending_arrival_time).
    """
    current_t_last = Float(t_last)
    current_pending = Float(pending_arrival_time)
    slot_cursor = start_slot_cursor if start_slot_cursor >= Index(0) else Index(0)

    num_sources = Index(source_buffer.initial_activity.shape[0])
    if num_sources <= Index(0) or max_particles == Index(0):
        return Index(0), slot_cursor, current_t_last, current_pending

    initial_activity = source_buffer.initial_activity
    decay_constant = source_buffer.decay_constant

    a_max = Float(0.0)
    for src_i in range(num_sources):
        act_0 = initial_activity[src_i]
        lam = decay_constant[src_i]
        if lam > 0.0 and t_ref != 0.0:
            a_max = Float(a_max + act_0 * math.exp(-lam * t_ref))
        else:
            a_max = Float(a_max + act_0)

    if a_max <= 0.0 or not math.isfinite(a_max):
        return Index(0), slot_cursor, current_t_last, Float(np.inf)

    is_active = bank.state.is_active
    count_array = bank.count_array
    capacity = Index(is_active.shape[0])
    next_id = ID(count_array[0])

    next_double = rng_ctx.next_double
    state_addr = rng_ctx.state_addr

    has_max_particles = max_particles >= Index(0)
    injected = Index(0)

    while True:
        if has_max_particles and injected >= max_particles:
            break

        while slot_cursor < capacity and is_active[slot_cursor]:
            slot_cursor += Index(1)
        if slot_cursor >= capacity:
            break

        if not math.isnan(current_pending) and current_pending >= 0.0:
            candidate_t = current_pending
            current_pending = Float(np.nan)
        else:
            rand_u1 = next_double(state_addr)
            delta_t = -math.log(1.0 - rand_u1) / a_max
            candidate_t = Float(current_t_last + delta_t)

        if candidate_t >= stop_time or not math.isfinite(candidate_t):
            current_pending = candidate_t
            break

        current_t_last = candidate_t
        rand_u2 = next_double(state_addr)
        threshold_x = Float(rand_u2 * a_max)

        accepted_source = Index(-1)
        total_act_t = Float(0.0)
        for src_i in range(num_sources):
            act_0 = initial_activity[src_i]
            lam = decay_constant[src_i]
            if lam > 0.0 and candidate_t != 0.0:
                act_t = Float(act_0 * math.exp(-lam * candidate_t))
            else:
                act_t = act_0
            total_act_t = Float(total_act_t + act_t)
            threshold_x = Float(threshold_x - act_t)
            if threshold_x < 0.0:
                accepted_source = Index(src_i)
                break

        if accepted_source < Index(0):
            if total_act_t <= 1e-30:
                current_pending = Float(np.inf)
                break
            if total_act_t * 16.0 < a_max:
                a_max = total_act_t
            continue

        _emit_primary_from_source(
            bank,
            source_buffer,
            accepted_source,
            slot_cursor,
            next_id,
            Time(candidate_t),
            rng_ctx,
        )

        next_id += ID(1)
        slot_cursor += Index(1)
        injected += Index(1)

    count_array[0] = Index(next_id)
    return injected, slot_cursor, current_t_last, current_pending


@njit(cache=True)
def inject_source_particles_kernel(
    bank: ParticleBank,
    source_buffer: SourceBuffer,
    source_idx: Index,
    num_to_inject: Index,
    time_start: Float,
    time_end: Float,
    rng_ctx: RNGContext,
    start_slot_cursor: Index = Index(0),
) -> Tuple[Index, Index]:
    """
    Пакетное Zero-Allocation ядро генерации и прямой записи первичных частиц из источника
    source_idx в свободные слоты ParticleBank со сканированием state.is_active с курсора.

    Возвращает кортеж (injected_count, next_slot_cursor).
    Поток ГСЧ продвигается строго почастично (энергия -> воксель -> позиция -> направление -> время)
    и останавливается немедленно при исчерпании свободных слотов банка без лишних вызовов ГСЧ.
    """
    if num_to_inject <= Index(0) or source_idx < Index(0) or source_idx >= Index(source_buffer.initial_activity.shape[0]):
        return Index(0), start_slot_cursor

    is_active = bank.state.is_active
    count_array = bank.count_array

    capacity = Index(is_active.shape[0])
    slot_cursor = start_slot_cursor if start_slot_cursor >= Index(0) else Index(0)
    next_id = ID(count_array[0])

    decay_const = source_buffer.decay_constant[source_idx]
    delta_time = Float(time_end - time_start)
    has_delta_time = delta_time > 0.0
    use_exp_time = decay_const > 0.0 and has_delta_time and (decay_const * delta_time >= 1e-6)
    decay_eff_dt = Float(1.0 - math.exp(-decay_const * delta_time)) if use_exp_time else Float(0.0)
    inv_decay = Float(1.0 / decay_const) if use_exp_time else Float(0.0)

    next_double = rng_ctx.next_double
    state_addr = rng_ctx.state_addr
    emission_times = bank.initial_state.emission_time

    injected = Index(0)
    while injected < num_to_inject:
        while slot_cursor < capacity and is_active[slot_cursor]:
            slot_cursor += Index(1)
        if slot_cursor >= capacity:
            break

        _emit_primary_from_source(
            bank,
            source_buffer,
            source_idx,
            slot_cursor,
            next_id,
            Time(time_start),
            rng_ctx,
        )
        if has_delta_time:
            rand_time_u = next_double(state_addr)
            if use_exp_time:
                emission_times[slot_cursor] = Time(
                    time_start - math.log(1.0 - rand_time_u * decay_eff_dt) * inv_decay
                )
            else:
                emission_times[slot_cursor] = Time(time_start + rand_time_u * delta_time)

        next_id += ID(1)
        slot_cursor += Index(1)
        injected += Index(1)

    count_array[0] = Index(next_id)
    return injected, slot_cursor


__all__ = [
    "_sample_alias",
    "_sample_voxel_position",
    "_sample_isotropic_direction",
    "_sample_cone_direction",
    "_write_primary",
    "_emit_primary_from_source",
    "inject_poisson_sources_kernel",
    "inject_source_particles_kernel",
]

