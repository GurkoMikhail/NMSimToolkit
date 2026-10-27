"""
Модуль Numba-ядер физических процессов взаимодействия частиц (фотоэффект,
комптоновское и когерентное рассеяние) с O(1) лог-индексацией, распределением
переданной энергии по вторичным каналам и записи актов взаимодействия в буферы.
"""

import math
from typing import Any

import hepunits as units
from numba import njit
import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import (
    Charge,
    Energy,
    Float,
    ID,
    Index,
    ParticleSpecies,
    ProcessID,
    Species,
    Time,
)
from core.particles.kinematic_state import KinematicState
from core.particles.particles_kernels import _rotate_particle
from core.physics.g4coherent import _generate_coherent_theta_scalar
from core.physics.g4compton import (
    _calculate_compton_energy_deposit_scalar,
    _generate_compton_theta_scalar,
    electron_mass_c2,
)
from core.physics.interaction_buffers import (
    InteractionBuffer,
    RNGContext,
    SecondaryActivationBuffer,
)
from core.physics.physics_buffer import AtomicRelaxationBank, MAX_RELAXATION_Z, PhysicsBuffer
from core.physics.physics_kernels import _locate_energy_interval
from core.source.source_kernels import _sample_alias

PHOTON_SPECIES_ID: Species = Species(ParticleSpecies.PHOTON)
ELECTRON_SPECIES_ID: Species = Species(ParticleSpecies.ELECTRON)
C_LIGHT: Float = Float(299.792458 * units.mm / units.ns)
ELECTRON_MASS: Float = electron_mass_c2


@njit(cache=True, inline='always')
def _generate_scattering_phi_scalar(rng_ctx: RNGContext) -> Float:
    """Генерирует азимутальный угол рассеяния phi равномерно на интервале [-pi, pi]."""
    return Float(np.pi * (rng_ctx.next_double(rng_ctx.state_addr) * 2.0 - 1.0))


@njit(cache=True, inline='always')
def _push_to_interaction_buffer(
    inter_buffer: InteractionBuffer,
    process_id: ProcessID,
    volume_id: Index,
    material_id: Index,
    particle_ID: ID,
    energy_deposit: Energy,
    scattering_theta: Float,
    scattering_phi: Float,
    distance_traveled: Float,
    species: Species,
    Z: Charge,
    pos_x: Float,
    pos_y: Float,
    pos_z: Float,
    dir_x: Float,
    dir_y: Float,
    dir_z: Float,
) -> None:
    """Записывает акт взаимодействия частицы напрямую в буфер с курсором InteractionBuffer."""
    if inter_buffer.capacity <= 0:
        return

    idx = inter_buffer.cursor[0]
    inter_buffer.cursor[0] = idx + 1
    if idx >= inter_buffer.capacity:
        return

    inter_buffer.process_id[idx] = process_id
    inter_buffer.volume_id[idx] = volume_id
    inter_buffer.material_id[idx] = material_id
    inter_buffer.particle_ID[idx] = particle_ID
    inter_buffer.energy_deposit[idx] = energy_deposit

    inter_buffer.scattering_theta[idx] = scattering_theta
    inter_buffer.scattering_phi[idx] = scattering_phi

    inter_buffer.distance_traveled[idx] = distance_traveled
    inter_buffer.species[idx] = species
    inter_buffer.Z[idx] = Z

    inter_buffer.position.x[idx] = pos_x
    inter_buffer.position.y[idx] = pos_y
    inter_buffer.position.z[idx] = pos_z

    inter_buffer.direction.x[idx] = dir_x
    inter_buffer.direction.y[idx] = dir_y
    inter_buffer.direction.z[idx] = dir_z


@njit(cache=True, inline='always')
def _sample_Z_scalar(
    mat_id: Index,
    process_col_idx: Index,
    energy: Float,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
) -> Charge:
    """
    Скалярный выбор атомного номера Z элемента-мишени по предрассчитанным нормированным
    парциальным долям сечений процесса в ElementCSR с линейной интерполяцией между узлами сетки.
    Для одноэлементного материала возвращает Z детерминированно без обращения к генератору ГСЧ.
    """
    start = physics_buffer.element_csr.element_offsets[mat_id]
    end = physics_buffer.element_csr.element_offsets[mat_id + 1]
    count = end - start

    if count <= 0:
        return Charge(1)

    if count == 1:
        return Charge(physics_buffer.element_csr.element_Z[start])

    row, fraction = _locate_energy_interval(energy, mat_id, physics_buffer.material_bank)
    rnd = rng_ctx.next_double(rng_ctx.state_addr)
    cumulative_prob = 0.0
    partial_fractions = physics_buffer.element_csr.element_partial_fractions
    num_rows = partial_fractions.shape[0]
    has_next_row = fraction > 0.0 and (row + 1) < num_rows

    for elem_offset in range(count - 1):
        weight_low = partial_fractions[row, process_col_idx, elem_offset]
        if has_next_row:
            weight_high = partial_fractions[row + 1, process_col_idx, elem_offset]
            weight = weight_low + fraction * (weight_high - weight_low)
        else:
            weight = weight_low
        cumulative_prob += weight
        if rnd < cumulative_prob:
            return Charge(physics_buffer.element_csr.element_Z[start + elem_offset])

    return Charge(physics_buffer.element_csr.element_Z[end - 1])


@njit(cache=True, inline='always')
def _compute_activation_time(
    emission_time: Time,
    distance_traveled: Float,
    kinetic_energy: Energy,
    species: Species,
) -> Time:
    """
    Вычисляет мировое время точки активации на основе времени вылета и пройденного пути:
    t_act = t_emit + l / v(E, m_0).
    """
    if distance_traveled <= 0.0:
        return emission_time

    if species == PHOTON_SPECIES_ID:
        velocity = C_LIGHT
    else:
        if kinetic_energy <= 0.0:
            return emission_time
        total_energy = kinetic_energy + ELECTRON_MASS
        gamma_inv = ELECTRON_MASS / total_energy
        beta_sq = max(0.0, 1.0 - gamma_inv * gamma_inv)
        velocity = Float(math.sqrt(beta_sq)) * C_LIGHT

    if velocity <= 0.0:
        return emission_time
    return Time(emission_time + distance_traveled / velocity)


@njit(cache=True, inline='always')
def _push_to_activation_buffer(
    act_buffer: SecondaryActivationBuffer,
    pos_x: Float,
    pos_y: Float,
    pos_z: Float,
    dir_x: Float,
    dir_y: Float,
    dir_z: Float,
    activation_time: Time,
    energy: Energy,
    parent_id: ID,
    species: Species,
    target_z: Charge,
    shell_idx: Index,
    process_id: ProcessID,
) -> bool:
    """Записывает точку активации вторичной частицы в предвыделенный SecondaryActivationBuffer."""
    if act_buffer.capacity <= 0:
        return False

    idx = act_buffer.cursor[0]
    if idx >= act_buffer.capacity:
        return False
    act_buffer.cursor[0] = idx + 1

    act_buffer.position.x[idx] = pos_x
    act_buffer.position.y[idx] = pos_y
    act_buffer.position.z[idx] = pos_z
    act_buffer.direction.x[idx] = dir_x
    act_buffer.direction.y[idx] = dir_y
    act_buffer.direction.z[idx] = dir_z
    act_buffer.activation_time[idx] = activation_time
    act_buffer.energy[idx] = energy
    act_buffer.parent_ID[idx] = parent_id
    act_buffer.species[idx] = species
    act_buffer.target_z[idx] = target_z
    act_buffer.shell_idx[idx] = shell_idx
    act_buffer.process_id[idx] = process_id
    return True


@njit(cache=True, inline='always')
def _sample_ionized_subshell_scalar(
    target_z: Charge,
    parent_energy: Energy,
    relaxation_bank: AtomicRelaxationBank,
    rng_ctx: RNGContext,
) -> Index:
    """
    Разыгрывает локальный индекс ионизированной подоболочки shell_idx в момент фотоэффекта
    по условию энергетической доступности E_parent >= E_bind(Z, s) и кумулятивным весам
    скачков поглощения jump_cumulative_weight.
    Возвращает локальный индекс подоболочки [0 .. num_shells - 1] или -1 (внешняя оболочка).
    """
    z_idx = Index(target_z)
    if (
        z_idx <= 0
        or z_idx > MAX_RELAXATION_Z
        or relaxation_bank.shell_offsets.shape[0] <= z_idx + 1
    ):
        return Index(-1)

    s_start = relaxation_bank.shell_offsets[z_idx]
    s_end = relaxation_bank.shell_offsets[z_idx + 1]
    if s_end <= s_start:
        return Index(-1)

    num_shells = s_end - s_start
    open_count = Index(0)
    for local_s in range(num_shells):
        if relaxation_bank.binding_energy[s_start + local_s] <= parent_energy:
            open_count = Index(local_s + 1)
        else:
            break

    if open_count <= 0:
        return Index(-1)

    total_weight = relaxation_bank.jump_cumulative_weight[s_start + open_count - 1]
    if total_weight <= 1.0:
        return Index(-1)

    target_weight = Float(rng_ctx.next_double(rng_ctx.state_addr) * total_weight)
    if target_weight < 1.0:
        return Index(-1)

    for local_s in range(open_count):
        if relaxation_bank.jump_cumulative_weight[s_start + local_s] > target_weight:
            return Index(local_s)

    return Index(open_count - 1)


@njit(cache=True, inline='always')
def partition_transferred_energy_kernel(
    kind: Index,
    process_id_c: ProcessID,
    parent_id: ID,
    parent_energy: Energy,
    delta_energy: Energy,
    target_z: Charge,
    pos_x: Float,
    pos_y: Float,
    pos_z: Float,
    dir_x: Float,
    dir_y: Float,
    dir_z: Float,
    activation_time: Time,
    relaxation_bank: AtomicRelaxationBank,
    act_buffer: SecondaryActivationBuffer,
    rng_ctx: RNGContext,
) -> Energy:
    """
    Ядро распределения переданной при взаимодействии энергии delta_energy между
    локальным энерговыделением energy_deposit и надпороговыми вторичными частицами
    в SecondaryActivationBuffer. Строго соблюдает закон сохранения энергии:
    delta_energy == local_energy_deposit + sum(E_secondary_tracked).
    """
    if delta_energy <= 0.0:
        return Float(0.0)

    enable_atomic_relaxation = act_buffer.enable_atomic_relaxation
    photon_production_cut = act_buffer.photon_production_cut
    electron_production_cut = act_buffer.electron_production_cut

    if kind == 1:
        # Комптоновское рассеяние: вся переданная энергия переходит комптоновскому электрону отдачи
        if delta_energy >= electron_production_cut:
            if _push_to_activation_buffer(
                act_buffer,
                pos_x,
                pos_y,
                pos_z,
                dir_x,
                dir_y,
                dir_z,
                activation_time,
                delta_energy,
                parent_id,
                ELECTRON_SPECIES_ID,
                target_z,
                Index(-1),
                process_id_c,
            ):
                return Float(0.0)
        return delta_energy

    if kind != 0:
        return delta_energy

    # Быстрый выход (Fast-path) для фотоэффекта без релаксации и ниже порогов отсечки:
    # 0 обращений к ГСЧ -> 100% побитовая обратная воспроизводимость.
    if (
        (not enable_atomic_relaxation or delta_energy < photon_production_cut)
        and delta_energy < electron_production_cut
    ):
        return delta_energy

    local_shell_idx = _sample_ionized_subshell_scalar(
        target_z,
        parent_energy,
        relaxation_bank,
        rng_ctx,
    )
    if local_shell_idx < 0:
        if delta_energy >= electron_production_cut:
            if _push_to_activation_buffer(
                act_buffer,
                pos_x,
                pos_y,
                pos_z,
                dir_x,
                dir_y,
                dir_z,
                activation_time,
                delta_energy,
                parent_id,
                ELECTRON_SPECIES_ID,
                target_z,
                Index(-1),
                process_id_c,
            ):
                return Float(0.0)
        return delta_energy

    s_start = relaxation_bank.shell_offsets[Index(target_z)]
    global_shell = Index(s_start + local_shell_idx)
    bind_e = relaxation_bank.binding_energy[global_shell]
    pe_energy = Float(max(0.0, delta_energy - bind_e))
    local_deposit = delta_energy

    if pe_energy > 0.0 and pe_energy >= electron_production_cut:
        if _push_to_activation_buffer(
            act_buffer,
            pos_x,
            pos_y,
            pos_z,
            dir_x,
            dir_y,
            dir_z,
            activation_time,
            pe_energy,
            parent_id,
            ELECTRON_SPECIES_ID,
            target_z,
            Index(-1),
            process_id_c,
        ):
            local_deposit = Float(max(0.0, local_deposit - pe_energy))

    if enable_atomic_relaxation and bind_e > 0.0:
        omega = relaxation_bank.fluorescence_yield[global_shell]
        l_start = relaxation_bank.line_offsets[global_shell]
        l_end = relaxation_bank.line_offsets[global_shell + 1]
        a_start = relaxation_bank.auger_offsets[global_shell]
        a_end = relaxation_bank.auger_offsets[global_shell + 1]

        u_relax = rng_ctx.next_double(rng_ctx.state_addr)
        if u_relax < omega and l_end > l_start:
            line_table_idx = _sample_alias(
                relaxation_bank.line_prob,
                relaxation_bank.line_alias,
                l_start,
                Index(l_end - l_start),
                rng_ctx,
            )
            line_e = relaxation_bank.line_energy[line_table_idx]
            if line_e > 0.0 and line_e >= photon_production_cut:
                if _push_to_activation_buffer(
                    act_buffer,
                    pos_x,
                    pos_y,
                    pos_z,
                    dir_x,
                    dir_y,
                    dir_z,
                    activation_time,
                    line_e,
                    parent_id,
                    PHOTON_SPECIES_ID,
                    target_z,
                    local_shell_idx,
                    process_id_c,
                ):
                    local_deposit = Float(max(0.0, local_deposit - line_e))
        elif a_end > a_start and bind_e >= electron_production_cut:
            auger_table_idx = _sample_alias(
                relaxation_bank.auger_prob,
                relaxation_bank.auger_alias,
                a_start,
                Index(a_end - a_start),
                rng_ctx,
            )
            auger_e = relaxation_bank.auger_energy[auger_table_idx]
            if auger_e > 0.0 and auger_e >= electron_production_cut:
                if _push_to_activation_buffer(
                    act_buffer,
                    pos_x,
                    pos_y,
                    pos_z,
                    dir_x,
                    dir_y,
                    dir_z,
                    activation_time,
                    auger_e,
                    parent_id,
                    ELECTRON_SPECIES_ID,
                    target_z,
                    local_shell_idx,
                    process_id_c,
                ):
                    local_deposit = Float(max(0.0, local_deposit - auger_e))

    return local_deposit


@njit(cache=True)
def _process_kernel_impl(
    state: KinematicState,
    particle_ids: NDArray[ID],
    emission_times: NDArray[Time],
    target_indices: NDArray[Index],
    volume_ids: NDArray[Index],
    material_ids: NDArray[Index],
    inter_buffer: InteractionBuffer,
    act_buffer: SecondaryActivationBuffer,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
    process_id_c: ProcessID,
    process_col_idx: Index,
    kind: Index,
) -> None:
    """
    Применяет физический процесс (0: фотоэффект, 1: комптоновское рассеяние, 2: когерентное рассеяние)
    на месте к выбранным частицам, распределяет переданную энергию и записывает события в буферы.
    """
    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, process_col_idx, energy, physics_buffer, rng_ctx)
        pos_x = state.position.x[p_idx]
        pos_y = state.position.y[p_idx]
        pos_z = state.position.z[p_idx]
        parent_dir_x = state.direction.x[p_idx]
        parent_dir_y = state.direction.y[p_idx]
        parent_dir_z = state.direction.z[p_idx]
        dir_x = parent_dir_x
        dir_y = parent_dir_y
        dir_z = parent_dir_z
        dist_traveled = state.distance_traveled[p_idx]
        species_val = state.species[p_idx]
        parent_id = particle_ids[p_idx]

        if kind == 0:
            theta = Float(0.0)
            phi = Float(0.0)
            delta_energy = energy
            state.energy[p_idx] = 0.0
        elif kind == 1:
            theta = _generate_compton_theta_scalar(energy, atomic_z, rng_ctx)
            phi = _generate_scattering_phi_scalar(rng_ctx)
            delta_energy = _calculate_compton_energy_deposit_scalar(theta, energy)
            state.energy[p_idx] -= delta_energy
            _rotate_particle(state, p_idx, theta, phi)
            dir_x = state.direction.x[p_idx]
            dir_y = state.direction.y[p_idx]
            dir_z = state.direction.z[p_idx]
        else:
            theta = _generate_coherent_theta_scalar(energy, atomic_z, rng_ctx)
            phi = _generate_scattering_phi_scalar(rng_ctx)
            delta_energy = Float(0.0)
            _rotate_particle(state, p_idx, theta, phi)
            dir_x = state.direction.x[p_idx]
            dir_y = state.direction.y[p_idx]
            dir_z = state.direction.z[p_idx]

        if kind == 0 or kind == 1:
            emit_t = emission_times[p_idx]
            act_time = _compute_activation_time(emit_t, dist_traveled, energy, species_val)
            energy_deposit = partition_transferred_energy_kernel(
                kind,
                process_id_c,
                parent_id,
                energy,
                delta_energy,
                atomic_z,
                pos_x,
                pos_y,
                pos_z,
                parent_dir_x,
                parent_dir_y,
                parent_dir_z,
                act_time,
                physics_buffer.relaxation_bank,
                act_buffer,
                rng_ctx,
            )
        else:
            energy_deposit = Float(0.0)

        _push_to_interaction_buffer(
            inter_buffer,
            process_id_c,
            volume_ids[p_idx],
            mat_id,
            parent_id,
            energy_deposit,
            theta,
            phi,
            dist_traveled,
            species_val,
            atomic_z,
            pos_x,
            pos_y,
            pos_z,
            dir_x,
            dir_y,
            dir_z,
        )


_KERNEL_CACHE: dict[tuple[int, int], Any] = {}


def _make_process_kernel(process_id: ProcessID, kind: int) -> Any:
    cache_key = (kind, int(process_id))
    cached_kernel = _KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    process_id_c = ProcessID(process_id)
    kind_c = Index(kind)

    def _process_kernel(
        state: KinematicState,
        particle_ids: NDArray[ID],
        emission_times: NDArray[Time],
        target_indices: NDArray[Index],
        volume_ids: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        act_buffer: SecondaryActivationBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_col_idx: Index = Index(0),
    ) -> None:
        _process_kernel_impl(
            state,
            particle_ids,
            emission_times,
            target_indices,
            volume_ids,
            material_ids,
            inter_buffer,
            act_buffer,
            physics_buffer,
            rng_ctx,
            process_id_c,
            process_col_idx,
            kind_c,
        )

    _KERNEL_CACHE[cache_key] = _process_kernel
    return _process_kernel


def make_photoelectric_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро фотоэффекта с запеченным идентификатором процесса."""
    return _make_process_kernel(process_id, 0)


def make_compton_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро комптоновского рассеяния с запеченным идентификатором процесса."""
    return _make_process_kernel(process_id, 1)


def make_coherent_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро когерентного (рэлеевского) рассеяния с запеченным идентификатором процесса."""
    return _make_process_kernel(process_id, 2)


__all__ = [
    "_compute_activation_time",
    "_generate_scattering_phi_scalar",
    "_push_to_activation_buffer",
    "_push_to_interaction_buffer",
    "_sample_Z_scalar",
    "_sample_ionized_subshell_scalar",
    "partition_transferred_energy_kernel",
    "make_photoelectric_kernel",
    "make_compton_kernel",
    "make_coherent_kernel",
]

