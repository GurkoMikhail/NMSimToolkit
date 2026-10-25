"""
Модуль компиляции физических таблиц (MaterialBank, карты мажорант ВудКока и CSR-состава элементов)
из графа сцены в плоские буферы PhysicsBuffer для исполнения в Numba.
"""

import ctypes
import math
from typing import Any, List, Optional

import numpy as np

from core.geometry.flattened_scene import FlattenedScene
from core.materials.atomic_properties import atomic_number
from core.materials.attenuation_database import _interpolate_element_on_material_grid
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.materials.materials import Material
from core.other.typing_definitions import CFuncAddress, Charge, Float, Index
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.physics_kernels import _compute_log_1d_array, _compute_log_2d_array
from core.physics.processes import Process
from core.scene.nodes import CompositeNode
from core.source.source_compiler import SourceCompiler
import settings.database_setting as settings

DEFAULT_LOOKUP_NUM_BINS: int = 512
MAX_LOOKUP_ADVANCE_LIMIT: int = 16


class PhysicsCompiler:
    """
    Компилятор физических данных сцены (MaterialBank, мажорантных материалов, указателей cfunc ВудКока
    и таблиц элементов ElementCSR) для высокопроизводительного трекинга в Numba.
    """

    def _validate_sources_energy_range(
        self,
        root_node: CompositeNode,
        processes_list: List[Process],
    ) -> None:
        """
        Проверяет предусловие: энергетические диапазоны всех активных процессов согласованы,
        а все линии спектров источников сцены лежат внутри [E_min, E_max] таблиц сечений.
        """
        if len(processes_list) == 0:
            return

        first_proc = processes_list[0]
        first_range = np.asarray(first_proc.energy_range, dtype=Float)
        if first_range.shape != (2,) or first_range[0] <= 0.0 or first_range[0] >= first_range[1]:
            raise ValueError(
                f"Некорректный энергетический диапазон процесса '{first_proc.name}': {first_range}."
            )

        for process in processes_list[1:]:
            proc_range = np.asarray(process.energy_range, dtype=Float)
            if proc_range.shape != (2,) or not np.array_equal(proc_range, first_range):
                raise ValueError(
                    f"Энергетический диапазон процесса '{process.name}' ({proc_range}) "
                    f"не совпадает с диапазоном процесса '{first_proc.name}' ({first_range})."
                )

        min_energy = float(first_range[0])
        max_energy = float(first_range[1])

        active_sources = SourceCompiler().compile_scene(root_node)
        for source_node in active_sources:
            line_energies = np.asarray(source_node.energy['energy'], dtype=Float)
            if line_energies.size > 0 and np.any((line_energies < min_energy) | (line_energies > max_energy)):
                raise ValueError(
                    f"Спектр источника '{source_node.name}' содержит линии вне допустимого диапазона "
                    f"таблиц сечений [{min_energy}, {max_energy}]: {line_energies}."
                )

    def _build_material_bank(
        self,
        materials_list: List[Material],
        processes_list: List[Process],
        lookup_num_bins: int = DEFAULT_LOOKUP_NUM_BINS,
        max_allowed_advance: int = MAX_LOOKUP_ADVANCE_LIMIT,
    ) -> MaterialBank:
        """
        Строит плоский банк материалов, таблиц линейных коэффициентов ослабления (LAC),
        предвычисленных логарифмов и двухуровневого логарифмического индекса поиска интервала.
        """
        if lookup_num_bins <= 0:
            raise ValueError(
                f"Число корзин логарифмического индекса lookup_num_bins должно быть > 0, получено {lookup_num_bins}."
            )
        if max_allowed_advance < 0:
            raise ValueError(
                f"Максимально допустимый порог шагов досмотра max_allowed_advance должен быть >= 0, "
                f"получено {max_allowed_advance}."
            )

        unique_materials = list({mat.ID: mat for mat in materials_list}.values())

        max_db_id = max((mat.ID for mat in settings.material_database.values()), default=0)
        max_scene_id = max((mat.ID for mat in unique_materials), default=0)
        capacity = max(max_db_id, max_scene_id) + 1

        mat_info_buffer = np.zeros(capacity, dtype=MaterialInfoDType)
        mat_pointers = np.zeros(capacity, dtype=MaterialPointerDType)

        all_energies = []
        all_lacs = []
        current_idx = 0
        active_material_ids: List[int] = []

        for col_idx, process in enumerate(processes_list):
            process.process_col_idx = Index(col_idx)

        for material in unique_materials:
            mat_id = material.ID
            mat_info_buffer[mat_id]['density'] = material.density
            mat_info_buffer[mat_id]['Z'] = material.Zeff
            mat_info_buffer[mat_id]['A'] = 0.0

            if len(processes_list) == 0:
                mat_pointers[mat_id]['start_idx'] = current_idx
                mat_pointers[mat_id]['length'] = 0
                mat_pointers[mat_id]['lookup_offset'] = 0
                continue

            first_proc = processes_list[0]
            has_in_first = material in first_proc.attenuation_function
            if has_in_first:
                energy_grid, _ = first_proc.attenuation_function[material]
            else:
                energy_grid = np.array([1e-3, 20.0], dtype=Float)

            length = len(energy_grid)
            mat_pointers[mat_id]['start_idx'] = current_idx
            mat_pointers[mat_id]['length'] = length
            mat_pointers[mat_id]['lookup_offset'] = len(active_material_ids) * lookup_num_bins
            current_idx += length
            if length > 0:
                active_material_ids.append(int(mat_id))

            all_energies.append(energy_grid)

            lac_matrix = np.zeros((length, len(processes_list)), dtype=Float)
            for p_idx, process in enumerate(processes_list):
                if material in process.attenuation_function:
                    if not has_in_first:
                        raise ValueError(
                            f"Материал '{material.name}' отсутствует в процессе '{first_proc.name}', "
                            f"но присутствует в процессе '{process.name}'."
                        )
                    proc_energy_grid, proc_lacs = process.attenuation_function[material]
                    if proc_energy_grid.shape != energy_grid.shape or not np.array_equal(
                        proc_energy_grid, energy_grid
                    ):
                        raise ValueError(
                            f"Энергетическая сетка процесса '{process.name}' для материала '{material.name}' "
                            f"не совпадает с сеткой процесса '{first_proc.name}'."
                        )
                    lac_matrix[:, p_idx] = proc_lacs
                elif has_in_first:
                    raise ValueError(
                        f"Материал '{material.name}' присутствует в процессе '{first_proc.name}', "
                        f"но отсутствует в процессе '{process.name}'."
                    )

            all_lacs.append(lac_matrix)

        if len(all_energies) > 0:
            physics_energy_grid = np.concatenate(all_energies).astype(Float, copy=False)
            physics_lac_table = np.concatenate(all_lacs).astype(Float, copy=False)
            if np.any(physics_energy_grid <= 0.0):
                raise ValueError("Энергетическая сетка MaterialBank должна содержать строго положительные энергии.")
            physics_log_energy_grid = _compute_log_1d_array(physics_energy_grid)
            physics_log_lac_table = _compute_log_2d_array(physics_lac_table)

            min_energy = float(np.min(physics_energy_grid))
            max_energy = float(np.max(physics_energy_grid))
            lookup_log_e_min = Float(math.log(min_energy))
            if max_energy > min_energy:
                log_e_max = Float(math.log(max_energy))
                lookup_inv_dlog_e = Float(float(lookup_num_bins) / float(log_e_max - lookup_log_e_min))
            else:
                lookup_inv_dlog_e = Float(1.0)
        else:
            physics_energy_grid = np.array([], dtype=Float)
            physics_lac_table = np.empty((0, len(processes_list)), dtype=Float)
            physics_log_energy_grid = np.array([], dtype=Float)
            physics_log_lac_table = np.empty((0, len(processes_list)), dtype=Float)
            lookup_log_e_min = Float(math.log(1e-3))
            lookup_inv_dlog_e = Float(float(lookup_num_bins) / (math.log(20.0) - math.log(1e-3)))

        num_active_materials = len(active_material_ids)
        lookup_bin_start = np.zeros(num_active_materials * lookup_num_bins, dtype=Index)
        lookup_max_advance = 0

        if num_active_materials > 0:
            bin_indices = np.arange(lookup_num_bins, dtype=Float)
            bin_left_energies = np.exp(
                float(lookup_log_e_min) + bin_indices / float(lookup_inv_dlog_e)
            ).astype(Float)

            for mat_id in active_material_ids:
                start_idx = int(mat_pointers[mat_id]['start_idx'])
                length = int(mat_pointers[mat_id]['length'])
                lookup_offset = int(mat_pointers[mat_id]['lookup_offset'])
                mat_grid = physics_energy_grid[start_idx : start_idx + length]
                mat_log_grid = physics_log_energy_grid[start_idx : start_idx + length]
                max_valid_idx = max(length - 2, 0)

                raw_starts = np.searchsorted(mat_grid, bin_left_energies, side='right') - 1
                mat_bin_start = np.clip(raw_starts, 0, max_valid_idx).astype(Index)
                mat_bin_start[0] = Index(0)

                for bin_idx in range(1, lookup_num_bins):
                    pos = int(mat_bin_start[bin_idx])
                    while pos > 0:
                        e_prev = float(np.nextafter(mat_grid[pos], -np.inf))
                        if e_prev <= 0.0:
                            break
                        scaled_prev = (math.log(e_prev) - float(lookup_log_e_min)) * float(lookup_inv_dlog_e)
                        bin_prev = 0 if scaled_prev <= 0.0 else min(int(scaled_prev), lookup_num_bins - 1)
                        if bin_prev >= bin_idx:
                            pos -= 1
                        else:
                            break
                    mat_bin_start[bin_idx] = Index(pos)

                lookup_bin_start[lookup_offset : lookup_offset + lookup_num_bins] = mat_bin_start

                if length >= 2:
                    scaled_nodes = (mat_log_grid[: length - 1] - lookup_log_e_min) * lookup_inv_dlog_e
                    node_bins = np.clip(
                        np.floor(np.maximum(scaled_nodes, 0.0)).astype(Index),
                        0,
                        lookup_num_bins - 1,
                    )
                    for bin_idx in range(lookup_num_bins):
                        start_pos = int(mat_bin_start[bin_idx])
                        end_pos = int(np.searchsorted(node_bins, bin_idx, side='right')) - 1
                        if bin_idx == lookup_num_bins - 1:
                            end_pos = max(end_pos, length - 2)
                        advance = max(0, end_pos - start_pos)
                        if advance > lookup_max_advance:
                            lookup_max_advance = advance

        if lookup_max_advance > max_allowed_advance:
            raise ValueError(
                f"Превышен допустимый порог шагов линейного досмотра в двухуровневом логарифмическом индексе "
                f"MaterialBank: lookup_max_advance={lookup_max_advance} > max_allowed_advance={max_allowed_advance} "
                f"(lookup_num_bins={lookup_num_bins})."
            )

        material_bank = MaterialBank(
            mat_info_buffer=mat_info_buffer,
            mat_pointers=mat_pointers,
            physics_energy_grid=physics_energy_grid,
            physics_lac_table=physics_lac_table,
            physics_log_energy_grid=physics_log_energy_grid,
            physics_log_lac_table=physics_log_lac_table,
            lookup_bin_start=lookup_bin_start,
            lookup_log_e_min=lookup_log_e_min,
            lookup_inv_dlog_e=lookup_inv_dlog_e,
            lookup_num_bins=int(lookup_num_bins),
            lookup_max_advance=int(lookup_max_advance),
        )
        material_bank._validate()
        return material_bank

    def _build_element_csr(
        self,
        materials_list: List[Material],
        processes_list: List[Process],
        material_bank: MaterialBank,
        capacity: Optional[int] = None,
    ) -> ElementCSR:
        """
        Строит CSR-представление элементного состава материалов и предрассчитывает
        нормированные парциальные доли элементов w_{p,i}(E_k) = mu_{p,i}(E_k) / mu_p(E_k)
        на строках финальной энергетической сетки материала в MaterialBank.
        """
        base_capacity = len(material_bank.mat_info_buffer) if capacity is None else capacity
        max_mat_id = max((mat.ID for mat in materials_list), default=-1)
        effective_capacity = max(base_capacity, max_mat_id + 1)
        counts = np.zeros(effective_capacity, dtype=Index)
        for material in materials_list:
            counts[material.ID] = len(material.composition_dict)

        element_offsets = np.zeros(effective_capacity + 1, dtype=Index)
        element_offsets[1:] = np.cumsum(counts)

        total_elements = int(element_offsets[-1])
        element_Z = np.zeros(total_elements, dtype=Charge)

        total_rows = int(material_bank.physics_lac_table.shape[0])
        num_processes = len(processes_list)
        max_elements = int(np.max(counts)) if counts.size > 0 else 0
        element_partial_fractions = np.zeros(
            (total_rows, num_processes, max_elements),
            dtype=Float,
        )

        for material in materials_list:
            mat_id = material.ID
            elem_start = int(element_offsets[mat_id])
            comp_items = list(material.composition_dict.items())
            num_mat_elements = len(comp_items)

            for elem_offset, (element_symbol_str, _) in enumerate(comp_items):
                element_Z[elem_start + elem_offset] = atomic_number[element_symbol_str]

            if num_mat_elements == 0 or num_processes == 0:
                continue

            row_start = int(material_bank.mat_pointers[mat_id]['start_idx'])
            row_length = int(material_bank.mat_pointers[mat_id]['length'])
            if row_length == 0:
                continue

            mat_energy_grid = material_bank.physics_energy_grid[row_start : row_start + row_length]
            total_weight = float(sum(weight for _, weight in comp_items))
            if any(float(weight) < 0.0 for _, weight in comp_items) or total_weight <= 0.0:
                raise ValueError(
                    f"Некорректные массовые доли элементов в материале '{material.name}': "
                    f"веса должны быть неотрицательными с положительной суммой (total_weight={total_weight})."
                )
            mass_fractions = np.array(
                [float(weight) / total_weight for _, weight in comp_items],
                dtype=Float,
            )

            for process_col_idx, process in enumerate(processes_list):
                if material in process.attenuation_function:
                    partial_lacs = np.zeros((row_length, num_mat_elements), dtype=Float)
                    elements_mac_db = process.attenuation_database.elements_MAC
                    for elem_offset, (element_symbol_str, weight) in enumerate(comp_items):
                        if element_symbol_str not in elements_mac_db:
                            raise ValueError(
                                f"Химический элемент '{element_symbol_str}' материала '{material.name}' "
                                f"отсутствует в базе сечений процесса '{process.name}'."
                            )
                        el_table = elements_mac_db[element_symbol_str]
                        el_energy = el_table['Energy']
                        el_coeff = el_table['Coefficient'][process.name]
                        interpolated_mac = _interpolate_element_on_material_grid(
                            mat_energy_grid,
                            el_energy,
                            el_coeff,
                        )
                        partial_lacs[:, elem_offset] = (
                            Float(material.density) * Float(weight) * interpolated_mac
                        )

                    total_lac = material_bank.physics_lac_table[
                        row_start : row_start + row_length, process_col_idx
                    ]
                    sum_partial_lac = np.sum(partial_lacs, axis=1)
                    if np.any(total_lac < 0.0):
                        raise ValueError(
                            f"Нарушен инвариант парциальных сечений материала '{material.name}' "
                            f"для процесса '{process.name}': обнаружены отрицательные полные сечения в MaterialBank."
                        )

                    positive_mask = total_lac > 0.0
                    zero_mask = ~positive_mask
                    if np.any(zero_mask) and np.any(sum_partial_lac[zero_mask] > 0.0):
                        raise ValueError(
                            f"Нарушен инвариант парциальных сечений материала '{material.name}' "
                            f"для процесса '{process.name}': полное сечение в MaterialBank равно 0.0 "
                            f"при ненулевой сумме парциальных вкладов элементов."
                        )

                    if np.any(positive_mask):
                        rel_error = (
                            np.abs(sum_partial_lac[positive_mask] - total_lac[positive_mask])
                            / total_lac[positive_mask]
                        )
                        max_rel_error = float(np.max(rel_error))
                        if max_rel_error > 1e-12:
                            raise ValueError(
                                f"Нарушен инвариант парциальных сечений материала '{material.name}' "
                                f"для процесса '{process.name}': максимальная относительная погрешность "
                                f"{max_rel_error:.3e} > 1e-12."
                            )

                    norm_fractions = np.zeros((row_length, num_mat_elements), dtype=Float)
                    if np.any(positive_mask):
                        norm_fractions[positive_mask, :] = (
                            partial_lacs[positive_mask, :] / total_lac[positive_mask, np.newaxis]
                        )
                        sum_norm = np.sum(norm_fractions[positive_mask, :], axis=1)
                        max_norm_error = float(np.max(np.abs(sum_norm - 1.0)))
                        if max_norm_error > 1e-12:
                            raise ValueError(
                                f"Сумма нормированных парциальных долей элементов материала '{material.name}' "
                                f"для процесса '{process.name}' отклоняется от 1.0 на {max_norm_error:.3e} > 1e-12."
                            )

                    if np.any(zero_mask):
                        norm_fractions[zero_mask, :] = mass_fractions[np.newaxis, :]

                    element_partial_fractions[
                        row_start : row_start + row_length,
                        process_col_idx,
                        :num_mat_elements,
                    ] = norm_fractions
                else:
                    element_partial_fractions[
                        row_start : row_start + row_length,
                        process_col_idx,
                        :num_mat_elements,
                    ] = mass_fractions[np.newaxis, :]

        return ElementCSR(
            element_offsets=element_offsets,
            element_Z=element_Z,
            element_partial_fractions=element_partial_fractions,
        )

    def compile_scene(
        self,
        root_node: CompositeNode,
        processes_list: List[Process],
        lookup_num_bins: int = DEFAULT_LOOKUP_NUM_BINS,
        max_allowed_advance: int = MAX_LOOKUP_ADVANCE_LIMIT,
    ) -> PhysicsBuffer:
        """
        Формирует полный буфер PhysicsBuffer по корневому узлу сцены и списку активных процессов.
        """
        self._validate_sources_energy_range(root_node, processes_list)
        flat_list = FlattenedScene(root_node).flat_list

        unique_materials = list(
            {mat.ID: mat for vol, _, _ in flat_list for mat in vol.material_list}.values()
        )

        material_bank = self._build_material_bank(
            unique_materials,
            processes_list,
            lookup_num_bins=lookup_num_bins,
            max_allowed_advance=max_allowed_advance,
        )

        capacity_mat_info = len(material_bank.mat_info_buffer)
        element_csr = self._build_element_csr(
            unique_materials,
            processes_list,
            material_bank,
            capacity=capacity_mat_info,
        )
        capacity = len(flat_list)

        majorant_material_map = np.zeros(capacity, dtype=Index)
        woodcock_function_pointers = np.zeros(capacity, dtype=CFuncAddress)

        for i, (vol, _, _) in enumerate(flat_list):
            majorant_material_map[i] = vol.majorant_material.ID
            cfunc_obj: Any = vol.material_cfunc
            if cfunc_obj is not None:
                ptr_val = ctypes.cast(cfunc_obj, ctypes.c_void_p).value
                woodcock_function_pointers[i] = ptr_val if ptr_val is not None else 0
            else:
                woodcock_function_pointers[i] = 0

        return PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers,
            element_csr=element_csr,
        )


__all__ = [
    "DEFAULT_LOOKUP_NUM_BINS",
    "MAX_LOOKUP_ADVANCE_LIMIT",
    "PhysicsCompiler",
]


