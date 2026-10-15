"""
Модуль компиляции физических таблиц (MaterialBank, карты мажорант ВудКока и CSR-состава элементов)
из графа сцены в плоские буферы PhysicsBuffer для исполнения в Numba.
"""

import ctypes
from typing import Any, List

import numpy as np

from core.geometry.flattened_scene import FlattenedScene
from core.materials.atomic_properties import atomic_number
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.materials.materials import Material
from core.other.typing_definitions import CFuncAddress, Charge, Float, Index
from core.physics.physics_buffer import ElementCSR, PhysicsBuffer
from core.physics.processes import Process
from core.scene.nodes import CompositeNode
import settings.database_setting as settings


class PhysicsCompiler:
    """
    Компилятор физических данных сцены (MaterialBank, мажорантных материалов, указателей cfunc ВудКока
    и таблиц элементов ElementCSR) для высокопроизводительного трекинга в Numba.
    """

    def _build_material_bank(self, materials_list: List[Material], processes_list: List[Process]) -> MaterialBank:
        """Строит плоский банк материалов и таблиц линейных коэффициентов ослабления (LAC)."""
        max_db_id = max((mat.ID for mat in settings.material_database.values()), default=0)
        max_scene_id = max((mat.ID for mat in materials_list), default=0)
        capacity = max(max_db_id, max_scene_id) + 1

        mat_info_buffer = np.zeros(capacity, dtype=MaterialInfoDType)
        mat_pointers = np.zeros(capacity, dtype=MaterialPointerDType)

        all_energies = []
        all_lacs = []
        current_idx = 0

        for material in materials_list:
            mat_id = material.ID
            mat_info_buffer[mat_id]['density'] = material.density
            mat_info_buffer[mat_id]['Z'] = material.Zeff
            mat_info_buffer[mat_id]['A'] = 0.0

            if len(processes_list) == 0:
                mat_pointers[mat_id]['start_idx'] = current_idx
                mat_pointers[mat_id]['length'] = 0
                continue

            first_proc = processes_list[0]
            if material in first_proc.attenuation_function:
                energy_grid, _ = first_proc.attenuation_function[material]
            else:
                energy_grid = np.array([1e-3, 100.0], dtype=Float)

            length = len(energy_grid)
            mat_pointers[mat_id]['start_idx'] = current_idx
            mat_pointers[mat_id]['length'] = length
            current_idx += length

            all_energies.append(energy_grid)

            lac_matrix = np.zeros((length, len(processes_list)), dtype=Float)
            for p_idx, process in enumerate(processes_list):
                if material in process.attenuation_function:
                    lacs = process.attenuation_function(material, energy_grid)
                    lac_matrix[:, p_idx] = lacs

            all_lacs.append(lac_matrix)

        if len(all_energies) > 0:
            physics_energy_grid = np.concatenate(all_energies)
            physics_lac_table = np.concatenate(all_lacs)
        else:
            physics_energy_grid = np.array([], dtype=Float)
            physics_lac_table = np.empty((0, len(processes_list)), dtype=Float)

        return MaterialBank(
            mat_info_buffer=mat_info_buffer,
            mat_pointers=mat_pointers,
            physics_energy_grid=physics_energy_grid,
            physics_lac_table=physics_lac_table,
        )

    def _build_element_csr(self, materials_list: List[Material], capacity: int) -> ElementCSR:
        """Строит CSR-представление элементного состава материалов для выбора атомного номера Z."""
        max_mat_id = max((mat.ID for mat in materials_list), default=-1)
        effective_capacity = max(capacity, max_mat_id + 1)
        counts = np.zeros(effective_capacity, dtype=Index)
        for material in materials_list:
            counts[material.ID] = len(material.composition_dict)

        element_offsets = np.zeros(effective_capacity + 1, dtype=Index)
        element_offsets[1:] = np.cumsum(counts)

        total_elements = int(element_offsets[-1])
        element_Z = np.zeros(total_elements, dtype=Charge)
        element_fraction = np.zeros(total_elements, dtype=Float)

        for material in materials_list:
            start_idx = int(element_offsets[material.ID])
            total_weight = float(sum(material.composition_dict.values()))

            current_idx = start_idx
            for element, weight in material.composition_dict.items():
                element_Z[current_idx] = atomic_number[element]
                element_fraction[current_idx] = float(weight) / total_weight if total_weight > 0.0 else 0.0
                current_idx += 1

        return ElementCSR(
            element_offsets=element_offsets,
            element_Z=element_Z,
            element_fraction=element_fraction,
        )

    def compile_scene(self, root_node: CompositeNode, processes_list: List[Process]) -> PhysicsBuffer:
        """
        Формирует полный буфер PhysicsBuffer по корневому узлу сцены и списку активных процессов.
        """
        flat_list = FlattenedScene(root_node).flat_list

        all_materials: List[Material] = []
        for vol, _, _ in flat_list:
            all_materials.extend(vol.material_list)
        unique_materials: List[Material] = []
        seen_ids = set()
        for mat in all_materials:
            if mat.ID not in seen_ids:
                seen_ids.add(mat.ID)
                unique_materials.append(mat)

        material_bank = self._build_material_bank(unique_materials, processes_list)

        capacity_mat_info = len(material_bank.mat_info_buffer)
        element_csr = self._build_element_csr(unique_materials, capacity_mat_info)
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


__all__ = ["PhysicsCompiler"]

