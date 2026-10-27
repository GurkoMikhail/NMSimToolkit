"""
Модуль Numba-ядер физических процессов взаимодействия частиц (фотоэффект,
комптоновское и когерентное рассеяние) с O(1) лог-индексацией, распределением
переданной энергии по вторичным каналам и записи актов взаимодействия в буферы.
"""

import math
from typing import Any, Tuple

from hepunits import c_light
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
    ProcessType,
    Species,
    Time,
)
from core.particles.kinematic_state import KinematicState
from core.particles.particles import ParticleBank
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


@njit(cache=True, inline='always')
def _generate_scattering_phi_scalar(rng_ctx: RNGContext) -> Float:
    """Генерирует азимутальный угол рассеяния phi равномерно на интервале [-pi, pi]."""
    return Float(np.pi * (rng_ctx.next_double(rng_ctx.state_addr) * 2.0 - 1.0))


@njit(cache=True, inline='always')
def _push_to_interaction_buffer(
    inter_buffer: InteractionBuffer,
    state: KinematicState,
    p_idx: Index,
    process_id: ProcessID,
    volume_id: Index,
    material_id: Index,
    particle_id: ID,
    energy_deposit: Energy,
    scattering_theta: Float,
    scattering_phi: Float,
    atomic_z: Charge,
) -> None:
    """Записывает акт взаимодействия частицы p_idx напрямую в буфер с курсором InteractionBuffer."""
    if inter_buffer.capacity <= 0:
        return

    idx = inter_buffer.cursor[0]
    inter_buffer.cursor[0] = idx + 1
    if idx >= inter_buffer.capacity:
        return

    inter_buffer.process_id[idx] = process_id
    inter_buffer.volume_id[idx] = volume_id
    inter_buffer.material_id[idx] = material_id
    inter_buffer.particle_ID[idx] = particle_id
    inter_buffer.energy_deposit[idx] = energy_deposit

    inter_buffer.scattering_theta[idx] = scattering_theta
    inter_buffer.scattering_phi[idx] = scattering_phi

    inter_buffer.distance_traveled[idx] = state.distance_traveled[p_idx]
    inter_buffer.species[idx] = state.species[p_idx]
    inter_buffer.Z[idx] = atomic_z

    inter_buffer.position.x[idx] = state.position.x[p_idx]
    inter_buffer.position.y[idx] = state.position.y[p_idx]
    inter_buffer.position.z[idx] = state.position.z[p_idx]

    inter_buffer.direction.x[idx] = state.direction.x[p_idx]
    inter_buffer.direction.y[idx] = state.direction.y[p_idx]
    inter_buffer.direction.z[idx] = state.direction.z[p_idx]


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

    if species == ParticleSpecies.PHOTON:
        velocity = Float(c_light)
    else:
        if kinetic_energy <= 0.0:
            return emission_time
        total_energy = kinetic_energy + electron_mass_c2
        gamma_inv = electron_mass_c2 / total_energy
        beta_sq = max(0.0, 1.0 - gamma_inv * gamma_inv)
        velocity = Float(math.sqrt(beta_sq) * c_light)

    if velocity <= 0.0:
        return emission_time
    return Time(emission_time + distance_traveled / velocity)


@njit(cache=True, inline='always')
def _push_to_activation_buffer(
    act_buffer: SecondaryActivationBuffer,
    state: KinematicState,
    p_idx: Index,
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

    act_buffer.position.x[idx] = state.position.x[p_idx]
    act_buffer.position.y[idx] = state.position.y[p_idx]
    act_buffer.position.z[idx] = state.position.z[p_idx]
    act_buffer.direction.x[idx] = state.direction.x[p_idx]
    act_buffer.direction.y[idx] = state.direction.y[p_idx]
    act_buffer.direction.z[idx] = state.direction.z[p_idx]
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
def _sample_vacancy_relaxation_emission(
    global_shell: Index,
    bind_energy: Energy,
    photon_production_cut: Float,
    electron_production_cut: Float,
    relaxation_bank: AtomicRelaxationBank,
    rng_ctx: RNGContext,
) -> Tuple[Species, Energy]:
    """
    Разыгрывает канал атомной релаксации ионизированной подоболочки global_shell
    (флуоресцентный фотон с вероятностью omega или Оже-электрон с вероятностью 1 - omega)
    по таблицам Воза и проверяет пороги рождения вторичных частиц.
    Возвращает (species, emitted_energy); если переход допороговый или безызлучательный,
    возвращает энергию 0.0.
    """
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
            return Species(ParticleSpecies.PHOTON), Energy(line_e)
    elif a_end > a_start and bind_energy >= electron_production_cut:
        auger_table_idx = _sample_alias(
            relaxation_bank.auger_prob,
            relaxation_bank.auger_alias,
            a_start,
            Index(a_end - a_start),
            rng_ctx,
        )
        auger_e = relaxation_bank.auger_energy[auger_table_idx]
        if auger_e > 0.0 and auger_e >= electron_production_cut:
            return Species(ParticleSpecies.ELECTRON), Energy(auger_e)

    return Species(ParticleSpecies.PHOTON), Energy(0.0)


@njit(cache=True, inline='always')
def partition_compton_energy_kernel(
    process_id_c: ProcessID,
    state: KinematicState,
    p_idx: Index,
    parent_id: ID,
    delta_energy: Energy,
    target_z: Charge,
    activation_time: Time,
    act_buffer: SecondaryActivationBuffer,
) -> Energy:
    """
    Ядро распределения переданной при комптоновском рассеянии энергии delta_energy между
    локальным энерговыделением и надпороговым электроном отдачи в SecondaryActivationBuffer.
    Строго соблюдает закон сохранения энергии: delta_energy == local_energy_deposit + E_secondary_tracked.
    """
    if delta_energy <= 0.0:
        return Float(0.0)

    if delta_energy >= act_buffer.electron_production_cut:
        if _push_to_activation_buffer(
            act_buffer,
            state,
            p_idx,
            activation_time,
            delta_energy,
            parent_id,
            Species(ParticleSpecies.ELECTRON),
            target_z,
            Index(-1),
            process_id_c,
        ):
            return Float(0.0)
    return delta_energy


@njit(cache=True, inline='always')
def partition_photoelectric_energy_kernel(
    process_id_c: ProcessID,
    state: KinematicState,
    p_idx: Index,
    parent_id: ID,
    parent_energy: Energy,
    delta_energy: Energy,
    target_z: Charge,
    activation_time: Time,
    relaxation_bank: AtomicRelaxationBank,
    act_buffer: SecondaryActivationBuffer,
    rng_ctx: RNGContext,
) -> Energy:
    """
    Ядро распределения поглощенной при фотоэффекте энергии delta_energy между локальным
    энерговыделением, фотоэлектроном и каналом атомной релаксации ионизированной оболочки
    (флуоресцентный фотон или Оже-электрон) в SecondaryActivationBuffer.
    Строго соблюдает закон сохранения энергии: delta_energy == local_energy_deposit + sum(E_secondary_tracked).
    """
    if delta_energy <= 0.0:
        return Float(0.0)

    enable_atomic_relaxation = act_buffer.enable_atomic_relaxation
    photon_production_cut = act_buffer.photon_production_cut
    electron_production_cut = act_buffer.electron_production_cut

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
                state,
                p_idx,
                activation_time,
                delta_energy,
                parent_id,
                Species(ParticleSpecies.ELECTRON),
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
            state,
            p_idx,
            activation_time,
            pe_energy,
            parent_id,
            Species(ParticleSpecies.ELECTRON),
            target_z,
            Index(-1),
            process_id_c,
        ):
            local_deposit = Float(max(0.0, local_deposit - pe_energy))

    if enable_atomic_relaxation and bind_e > 0.0:
        relax_species, relax_energy = _sample_vacancy_relaxation_emission(
            global_shell,
            bind_e,
            photon_production_cut,
            electron_production_cut,
            relaxation_bank,
            rng_ctx,
        )
        if relax_energy > 0.0:
            if _push_to_activation_buffer(
                act_buffer,
                state,
                p_idx,
                activation_time,
                relax_energy,
                parent_id,
                relax_species,
                target_z,
                local_shell_idx,
                process_id_c,
            ):
                local_deposit = Float(max(0.0, local_deposit - relax_energy))

    return local_deposit


@njit(cache=True)
def _photoelectric_kernel_impl(
    bank: ParticleBank,
    target_indices: NDArray[Index],
    material_ids: NDArray[Index],
    inter_buffer: InteractionBuffer,
    act_buffer: SecondaryActivationBuffer,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
    process_id_c: ProcessID,
    process_col_idx: Index,
) -> None:
    """
    Применяет фотоэлектрическое поглощение на месте к выбранным частицам в ParticleBank,
    распределяет поглощенную энергию по вторичным каналам и записывает события в буферы.
    """
    state = bank.state
    particle_ids = bank.initial_state.ID
    emission_times = bank.initial_state.emission_time
    volume_ids = bank.navigation_state.current_volume
    relaxation_bank = physics_buffer.relaxation_bank

    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, process_col_idx, energy, physics_buffer, rng_ctx)
        parent_id = particle_ids[p_idx]

        delta_energy = energy
        state.energy[p_idx] = 0.0

        act_time = _compute_activation_time(
            emission_times[p_idx],
            state.distance_traveled[p_idx],
            energy,
            state.species[p_idx],
        )
        energy_deposit = partition_photoelectric_energy_kernel(
            process_id_c,
            state,
            p_idx,
            parent_id,
            energy,
            delta_energy,
            atomic_z,
            act_time,
            relaxation_bank,
            act_buffer,
            rng_ctx,
        )

        _push_to_interaction_buffer(
            inter_buffer,
            state,
            p_idx,
            process_id_c,
            volume_ids[p_idx],
            mat_id,
            parent_id,
            energy_deposit,
            Float(0.0),
            Float(0.0),
            atomic_z,
        )


@njit(cache=True)
def _compton_kernel_impl(
    bank: ParticleBank,
    target_indices: NDArray[Index],
    material_ids: NDArray[Index],
    inter_buffer: InteractionBuffer,
    act_buffer: SecondaryActivationBuffer,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
    process_id_c: ProcessID,
    process_col_idx: Index,
) -> None:
    """
    Применяет некогерентное (комптоновское) рассеяние на месте к выбранным частицам в ParticleBank,
    распределяет переданную электрону отдачи энергию (до поворота родителя), поворачивает фотон
    и записывает события в буферы.
    """
    state = bank.state
    particle_ids = bank.initial_state.ID
    emission_times = bank.initial_state.emission_time
    volume_ids = bank.navigation_state.current_volume

    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, process_col_idx, energy, physics_buffer, rng_ctx)
        parent_id = particle_ids[p_idx]

        theta = _generate_compton_theta_scalar(energy, atomic_z, rng_ctx)
        phi = _generate_scattering_phi_scalar(rng_ctx)
        delta_energy = _calculate_compton_energy_deposit_scalar(theta, energy)
        state.energy[p_idx] -= delta_energy

        act_time = _compute_activation_time(
            emission_times[p_idx],
            state.distance_traveled[p_idx],
            energy,
            state.species[p_idx],
        )
        energy_deposit = partition_compton_energy_kernel(
            process_id_c,
            state,
            p_idx,
            parent_id,
            delta_energy,
            atomic_z,
            act_time,
            act_buffer,
        )

        _rotate_particle(state, p_idx, theta, phi)

        _push_to_interaction_buffer(
            inter_buffer,
            state,
            p_idx,
            process_id_c,
            volume_ids[p_idx],
            mat_id,
            parent_id,
            energy_deposit,
            theta,
            phi,
            atomic_z,
        )


@njit(cache=True)
def _coherent_kernel_impl(
    bank: ParticleBank,
    target_indices: NDArray[Index],
    material_ids: NDArray[Index],
    inter_buffer: InteractionBuffer,
    physics_buffer: PhysicsBuffer,
    rng_ctx: RNGContext,
    process_id_c: ProcessID,
    process_col_idx: Index,
) -> None:
    """
    Применяет когерентное (рэлеевское) рассеяние на месте к выбранным частицам в ParticleBank,
    поворачивает фотон без потери энергии и записывает события в InteractionBuffer.
    """
    state = bank.state
    particle_ids = bank.initial_state.ID
    volume_ids = bank.navigation_state.current_volume

    for j in range(len(target_indices)):
        p_idx = target_indices[j]
        energy = state.energy[p_idx]
        mat_id = material_ids[p_idx]
        atomic_z = _sample_Z_scalar(mat_id, process_col_idx, energy, physics_buffer, rng_ctx)
        parent_id = particle_ids[p_idx]

        theta = _generate_coherent_theta_scalar(energy, atomic_z, rng_ctx)
        phi = _generate_scattering_phi_scalar(rng_ctx)

        _rotate_particle(state, p_idx, theta, phi)

        _push_to_interaction_buffer(
            inter_buffer,
            state,
            p_idx,
            process_id_c,
            volume_ids[p_idx],
            mat_id,
            parent_id,
            Float(0.0),
            theta,
            phi,
            atomic_z,
        )


_KERNEL_CACHE: dict[tuple[int, int], Any] = {}


def make_photoelectric_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро фотоэффекта с запеченным идентификатором процесса."""
    cache_key = (int(ProcessType.PHOTOELECTRIC), int(process_id))
    cached_kernel = _KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    process_id_c = ProcessID(process_id)

    def _process_kernel(
        bank: ParticleBank,
        target_indices: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        act_buffer: SecondaryActivationBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_col_idx: Index = Index(0),
    ) -> None:
        _photoelectric_kernel_impl(
            bank,
            target_indices,
            material_ids,
            inter_buffer,
            act_buffer,
            physics_buffer,
            rng_ctx,
            process_id_c,
            process_col_idx,
        )

    _KERNEL_CACHE[cache_key] = _process_kernel
    return _process_kernel


def make_compton_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро комптоновского рассеяния с запеченным идентификатором процесса."""
    cache_key = (int(ProcessType.COMPTON), int(process_id))
    cached_kernel = _KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    process_id_c = ProcessID(process_id)

    def _process_kernel(
        bank: ParticleBank,
        target_indices: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        act_buffer: SecondaryActivationBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_col_idx: Index = Index(0),
    ) -> None:
        _compton_kernel_impl(
            bank,
            target_indices,
            material_ids,
            inter_buffer,
            act_buffer,
            physics_buffer,
            rng_ctx,
            process_id_c,
            process_col_idx,
        )

    _KERNEL_CACHE[cache_key] = _process_kernel
    return _process_kernel


def make_coherent_kernel(process_id: ProcessID) -> Any:
    """Создает скомпилированное Numba-ядро когерентного (рэлеевского) рассеяния с запеченным идентификатором процесса."""
    cache_key = (int(ProcessType.COHERENT), int(process_id))
    cached_kernel = _KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    process_id_c = ProcessID(process_id)

    def _process_kernel(
        bank: ParticleBank,
        target_indices: NDArray[Index],
        material_ids: NDArray[Index],
        inter_buffer: InteractionBuffer,
        act_buffer: SecondaryActivationBuffer,
        physics_buffer: PhysicsBuffer,
        rng_ctx: RNGContext,
        process_col_idx: Index = Index(0),
    ) -> None:
        _coherent_kernel_impl(
            bank,
            target_indices,
            material_ids,
            inter_buffer,
            physics_buffer,
            rng_ctx,
            process_id_c,
            process_col_idx,
        )

    _KERNEL_CACHE[cache_key] = _process_kernel
    return _process_kernel


__all__ = [
    "_compute_activation_time",
    "_generate_scattering_phi_scalar",
    "_push_to_activation_buffer",
    "_push_to_interaction_buffer",
    "_sample_Z_scalar",
    "_sample_ionized_subshell_scalar",
    "_sample_vacancy_relaxation_emission",
    "partition_compton_energy_kernel",
    "partition_photoelectric_energy_kernel",
    "_photoelectric_kernel_impl",
    "_compton_kernel_impl",
    "_coherent_kernel_impl",
    "make_photoelectric_kernel",
    "make_compton_kernel",
    "make_coherent_kernel",
]
