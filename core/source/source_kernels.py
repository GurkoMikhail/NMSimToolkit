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
from core.physics.interaction_buffers import RNGContext, SecondaryActivationBuffer
from core.physics.physics_buffer import AtomicRelaxationBank, MAX_RELAXATION_Z
from core.source.source_buffer import (
    SPATIAL_POINT,
    SPECTRUM_CONTINUOUS_TRAPEZOID,
    SourceBuffer,
)


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
def _sample_channel_energy(
    source_buffer: SourceBuffer,
    channel_idx: Index,
    rng_ctx: RNGContext,
) -> Energy:
    """
    Выбирает энергию испускаемой частицы из спектра канала channel_idx за O(1).

    Для дискретного спектра (SPECTRUM_DISCRETE) возвращает энергию выбранной линии Воза.
    Для непрерывного кусочно-линейного спектра (SPECTRUM_CONTINUOUS_TRAPEZOID) выбирает
    интервал трапеции [E_k, E_{k+1}] по таблице Воза и аналитически обращает линейную
    плотность вероятности на интервале за одно дополнительное равномерное число u_trap.
    """
    line_offset = source_buffer.spectrum_offsets[channel_idx]
    line_count = source_buffer.spectrum_offsets[channel_idx + Index(1)] - line_offset
    line_table_idx = _sample_alias(
        source_buffer.line_prob,
        source_buffer.line_alias,
        line_offset,
        line_count,
        rng_ctx,
    )
    e_low = source_buffer.line_energy[line_table_idx]
    if source_buffer.spectrum_type[channel_idx] == SPECTRUM_CONTINUOUS_TRAPEZOID:
        e_high = source_buffer.line_grid_high[line_table_idx]
        r_val = source_buffer.line_pdf_ratio[line_table_idx]
        u_trap = rng_ctx.next_double(rng_ctx.state_addr)
        one_minus_r = 1.0 - r_val
        disc = (1.0 - u_trap) * r_val * r_val + u_trap * one_minus_r * one_minus_r
        denom = r_val + math.sqrt(disc)
        t_frac = u_trap / denom if denom > 0.0 else 0.0
        return Energy(e_low + t_frac * (e_high - e_low))
    return Energy(e_low)


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
    При voxel_size == 0.0 и нулевом cell_jitter читает мировой сдвиг (m03, m13, m23) напрямую.
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
        if cell_jitter_x <= 0.0 and cell_jitter_y <= 0.0:
            return (
                Float(local_to_global[source_idx, 0, 3]),
                Float(local_to_global[source_idx, 1, 3]),
                Float(local_to_global[source_idx, 2, 3]),
            )
        local_x = (rng_ctx.next_double(rng_ctx.state_addr) - 0.5) * cell_jitter_x if cell_jitter_x > 0.0 else Float(0.0)
        local_y = (rng_ctx.next_double(rng_ctx.state_addr) - 0.5) * cell_jitter_y if cell_jitter_y > 0.0 else Float(0.0)
        local_z = Float(0.0)

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
    parent_id: ID = ID(-1),
) -> None:
    """
    Прямая запись состояния новой частицы в слот массивов KinematicState,
    InitialState (с заданным parent_ID, по умолчанию -1 для первичных) и сброс NavigationState.
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
    initial_state.parent_ID[slot] = parent_id
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
    channel_idx: Index,
    slot: Index,
    particle_id: ID,
    emission_time_val: Time,
    rng_ctx: RNGContext,
) -> None:
    """
    Генерирует первичную частицу для выбранного канала эмиссии channel_idx с заданным
    временем испускания emission_time_val и записывает её напрямую в слот slot банка частиц.

    Порядок потребления случайных чисел из rng_ctx:
    1) выбор энергии из дискретного или непрерывного спектра (_sample_channel_energy);
    2) для SPATIAL_VOXEL — выбор активного вокселя методом алиасов Воза (_sample_alias) и
       равномерное смещение внутри вокселя; для SPATIAL_POINT — прямое чтение (m03, m13, m23)
       при нулевом cell_jitter или смещение в [-dx/2, +dx/2] x [-dy/2, +dy/2];
    3) изотропное или конусное направление вылета (_sample_cone_direction).
    """
    energy_val = _sample_channel_energy(source_buffer, channel_idx, rng_ctx)

    if source_buffer.spatial_type[channel_idx] == SPATIAL_POINT:
        pos_x, pos_y, pos_z = _sample_voxel_position(
            Index(0),
            Index(0),
            Index(0),
            Float(0.0),
            Float(0.0),
            Float(0.0),
            Float(0.0),
            source_buffer.local_to_global,
            channel_idx,
            rng_ctx,
            source_buffer.cell_jitter[channel_idx, 0],
            source_buffer.cell_jitter[channel_idx, 1],
        )
    else:
        vox_offset = source_buffer.voxel_range[channel_idx, 0]
        vox_count = source_buffer.voxel_range[channel_idx, 1] - vox_offset
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
            source_buffer.grid_shape[channel_idx, 1],
            source_buffer.grid_shape[channel_idx, 2],
            source_buffer.voxel_size[channel_idx],
            source_buffer.half_size[channel_idx, 0],
            source_buffer.half_size[channel_idx, 1],
            source_buffer.half_size[channel_idx, 2],
            source_buffer.local_to_global,
            channel_idx,
            rng_ctx,
            Float(0.0),
            Float(0.0),
        )

    dir_x, dir_y, dir_z = _sample_cone_direction(
        source_buffer.cos_theta_min[channel_idx],
        source_buffer.local_to_global,
        channel_idx,
        rng_ctx,
    )

    _write_primary(
        bank,
        slot,
        particle_id,
        source_buffer.species[channel_idx],
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
    Генерирует и инжектирует первичные частицы из суперпозиции каналов эмиссии в ParticleBank
    по нестационарному пуассоновскому потоку с прореживанием Льюиса–Шедлера (Lewis–Shedler thinning).

    Мажоранта интенсивности вычисляется в опорной точке начала первого сегмента:
        I_max = sum_c I_{0,c} * exp(-lambda_c * t_ref).

    Инварианты потока случайных чисел (для побитовой независимости от batch_capacity и досчёта):
    1. Свободный слот в ParticleBank ищется ДО извлечения или генерации очередного кандидата по времени.
    2. Если передан не-NaN pending_arrival_time, он используется как кандидат t без обращения к ГСЧ.
    3. Если t >= stop_time (при конечном stop_time), кандидат сохраняется в pending_arrival_time,
       и работа ядра завершается без розыгрыша прореживания и параметров частицы.
    4. Иначе обновляется t_last = t, разыгрывается прореживание x = u_2 * I_max с выбором канала эмиссии;
       при отвержении кандидата цикл продолжается для того же найденного свободного слота.

    Возвращает кортеж (injected_count, next_slot_cursor, new_t_last, new_pending_arrival_time).
    """
    current_t_last = Float(t_last)
    current_pending = Float(pending_arrival_time)
    slot_cursor = start_slot_cursor if start_slot_cursor >= Index(0) else Index(0)

    num_channels = Index(source_buffer.initial_emission_rate.shape[0])
    if num_channels <= Index(0) or max_particles == Index(0):
        return Index(0), slot_cursor, current_t_last, current_pending

    initial_rate = source_buffer.initial_emission_rate
    decay_constant = source_buffer.decay_constant

    a_max = Float(0.0)
    for ch_i in range(num_channels):
        act_0 = initial_rate[ch_i]
        lam = decay_constant[ch_i]
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

        accepted_channel = Index(-1)
        total_act_t = Float(0.0)
        for ch_i in range(num_channels):
            act_0 = initial_rate[ch_i]
            lam = decay_constant[ch_i]
            if lam > 0.0 and candidate_t != 0.0:
                act_t = Float(act_0 * math.exp(-lam * candidate_t))
            else:
                act_t = act_0
            total_act_t = Float(total_act_t + act_t)
            threshold_x = Float(threshold_x - act_t)
            if threshold_x < 0.0:
                accepted_channel = Index(ch_i)
                break

        if accepted_channel < Index(0):
            if total_act_t <= 1e-30:
                current_pending = Float(np.inf)
                break
            if total_act_t * 16.0 < a_max:
                a_max = total_act_t
            continue

        _emit_primary_from_source(
            bank,
            source_buffer,
            accepted_channel,
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
def inject_secondary_sources_kernel(
    bank: ParticleBank,
    act_buffer: SecondaryActivationBuffer,
    relaxation_bank: AtomicRelaxationBank,
    stop_time: Float,
    rng_ctx: RNGContext,
    max_particles: Index = Index(-1),
    start_slot_cursor: Index = Index(0),
) -> Tuple[Index, Index]:
    """
    Разряжает буфер наведенных вторичных источников SecondaryActivationBuffer в свободные слоты
    банка частиц ParticleBank с пуассоновским розыгрышем времени испускания:
        t_emit = t_act - ln(1 - u) / lambda_sec(Z, shell_idx).

    Если в SecondaryActivationBuffer больше записей, чем свободных слотов в ParticleBank
    (или достигнут лимит max_particles >= 0), необработанный хвост сдвигается в начало буфера
    для выгрузки на следующем шаге.
    Возвращает кортеж (injected_count, next_slot_cursor).
    """
    slot_cursor = start_slot_cursor if start_slot_cursor >= Index(0) else Index(0)
    if act_buffer.capacity <= 0 or max_particles == Index(0):
        return Index(0), slot_cursor

    num_activations = act_buffer.cursor[0]
    if num_activations <= Index(0):
        return Index(0), slot_cursor

    is_active = bank.state.is_active
    count_array = bank.count_array
    capacity = Index(is_active.shape[0])
    next_id = ID(count_array[0])

    next_double = rng_ctx.next_double
    state_addr = rng_ctx.state_addr

    has_max_particles = max_particles >= Index(0)
    injected = Index(0)
    act_idx = Index(0)

    while act_idx < num_activations:
        if has_max_particles and injected >= max_particles:
            break

        while slot_cursor < capacity and is_active[slot_cursor]:
            slot_cursor += Index(1)
        if slot_cursor >= capacity:
            break

        t_act = act_buffer.activation_time[act_idx]
        target_z = Index(act_buffer.target_z[act_idx])
        local_shell = act_buffer.shell_idx[act_idx]

        decay_lambda = Float(0.0)
        if (
            local_shell >= Index(0)
            and target_z > Index(0)
            and target_z <= Index(MAX_RELAXATION_Z)
            and relaxation_bank.shell_offsets.shape[0] > target_z + 1
        ):
            s_start = relaxation_bank.shell_offsets[target_z]
            s_end = relaxation_bank.shell_offsets[target_z + 1]
            global_shell = Index(s_start + local_shell)
            if global_shell < s_end:
                decay_lambda = relaxation_bank.decay_constant[global_shell]

        if decay_lambda > 0.0 and math.isfinite(decay_lambda):
            rand_u = next_double(state_addr)
            delta_t = -math.log(1.0 - rand_u) / decay_lambda
            t_emit = Time(t_act + delta_t)
        else:
            t_emit = Time(t_act)

        if t_emit >= stop_time or not math.isfinite(t_emit):
            act_idx += Index(1)
            continue

        dir_x, dir_y, dir_z = _sample_isotropic_direction(rng_ctx)

        _write_primary(
            bank,
            slot_cursor,
            next_id,
            Species(act_buffer.species[act_idx]),
            act_buffer.energy[act_idx],
            t_emit,
            act_buffer.position.x[act_idx],
            act_buffer.position.y[act_idx],
            act_buffer.position.z[act_idx],
            dir_x,
            dir_y,
            dir_z,
            act_buffer.parent_ID[act_idx],
        )

        next_id += ID(1)
        slot_cursor += Index(1)
        injected += Index(1)
        act_idx += Index(1)

    if act_idx < num_activations:
        remaining = Index(num_activations - act_idx)
        for dst_i in range(remaining):
            src_i = act_idx + dst_i
            act_buffer.position.x[dst_i] = act_buffer.position.x[src_i]
            act_buffer.position.y[dst_i] = act_buffer.position.y[src_i]
            act_buffer.position.z[dst_i] = act_buffer.position.z[src_i]
            act_buffer.direction.x[dst_i] = act_buffer.direction.x[src_i]
            act_buffer.direction.y[dst_i] = act_buffer.direction.y[src_i]
            act_buffer.direction.z[dst_i] = act_buffer.direction.z[src_i]
            act_buffer.activation_time[dst_i] = act_buffer.activation_time[src_i]
            act_buffer.energy[dst_i] = act_buffer.energy[src_i]
            act_buffer.parent_ID[dst_i] = act_buffer.parent_ID[src_i]
            act_buffer.species[dst_i] = act_buffer.species[src_i]
            act_buffer.target_z[dst_i] = act_buffer.target_z[src_i]
            act_buffer.shell_idx[dst_i] = act_buffer.shell_idx[src_i]
            act_buffer.process_id[dst_i] = act_buffer.process_id[src_i]
        act_buffer.cursor[0] = remaining
    else:
        act_buffer.cursor[0] = Index(0)

    count_array[0] = Index(next_id)
    return injected, slot_cursor


__all__ = [
    "_sample_alias",
    "_sample_channel_energy",
    "_sample_voxel_position",
    "_sample_isotropic_direction",
    "_sample_cone_direction",
    "_write_primary",
    "_emit_primary_from_source",
    "inject_poisson_sources_kernel",
    "inject_secondary_sources_kernel",
]
