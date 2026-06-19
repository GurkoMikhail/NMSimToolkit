import numpy as np
from typing import List

from core.geometry.volumes import Volume
from core.scene.nodes import CompositeNode
from core.physics.processes import Process
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.physics.physics_buffer import PhysicsBuffer, ElementCSR
from core.materials.materials import Material
from core.materials.atomic_properties import atomic_number
from core.other.typing_definitions import Index, Float, CFuncAddress, Charge

class PhysicsCompiler:
    """
    Compiles physics data (MaterialBank, Majorant Material Map, Woodcock Pointers)
    from the scene for fast Numba execution.
    """

    def _build_material_bank(self, materials_list: List[Material], processes_list: List[Process]) -> MaterialBank:
        import settings.database_setting as settings
        capacity = len(settings.material_database) + 1

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
            try:
                energy_grid, _ = first_proc.attenuation_function[material]
            except KeyError:
                energy_grid = np.array([1e-3, 100.0], dtype=Float)

            length = len(energy_grid)
            mat_pointers[mat_id]['start_idx'] = current_idx
            mat_pointers[mat_id]['length'] = length
            current_idx += length

            all_energies.append(energy_grid)

            lac_matrix = np.zeros((length, len(processes_list)), dtype=Float)
            for p_idx, process in enumerate(processes_list):
                lacs = process.attenuation_function(material, energy_grid) # type: ignore
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
            physics_lac_table=physics_lac_table
        )

    def _build_element_csr(self, materials_list: List[Material], capacity: int) -> ElementCSR:
        counts = np.zeros(capacity, dtype=Index)
        for material in materials_list:
            counts[material.ID] = len(material.composition_dict)

        element_offsets = np.zeros(capacity + 1, dtype=Index)
        element_offsets[1:] = np.cumsum(counts)

        total_elements = element_offsets[-1]
        element_Z = np.zeros(total_elements, dtype=Charge)
        element_fraction = np.zeros(total_elements, dtype=Float)

        for material in materials_list:
            start_idx = element_offsets[material.ID]
            total_weight = sum(material.composition_dict.values())

            current_idx = start_idx
            for element, weight in material.composition_dict.items():
                element_Z[current_idx] = atomic_number[element]
                element_fraction[current_idx] = weight / total_weight
                current_idx += 1

        return ElementCSR(
            element_offsets=element_offsets,
            element_Z=element_Z,
            element_fraction=element_fraction
        )

    def compile_scene(self, root_node: 'CompositeNode', processes_list: List[Process]) -> PhysicsBuffer:
        """
        Builds the complete PhysicsBuffer from the root volume and active processes.
        """
        from core.geometry.flattened_scene import FlattenedScene
        flat_list = FlattenedScene(root_node).flat_list

        all_materials = []
        for vol, _, _ in flat_list:
            all_materials.extend(vol.material_list)
        unique_materials = []
        seen_ids = set()
        for mat in all_materials:
            if mat.ID not in seen_ids:
                seen_ids.add(mat.ID)
                unique_materials.append(mat)

        # Build dynamic material bank (Zero Memory Waste)
        material_bank = self._build_material_bank(unique_materials, processes_list)

        # Build element CSR for sampling elements
        capacity_mat_info = len(material_bank.mat_info_buffer)
        element_csr = self._build_element_csr(unique_materials, capacity_mat_info)
        capacity = len(flat_list)

        majorant_material_map = np.zeros(capacity, dtype=Index)
        woodcock_function_pointers = np.zeros(capacity, dtype=CFuncAddress)

        for i, (vol, _, _) in enumerate(flat_list):
            majorant_material_map[i] = vol.majorant_material.ID
            if vol.material_cfunc is not None:
                import ctypes
                woodcock_function_pointers[i] = ctypes.cast(vol.material_cfunc, ctypes.c_void_p).value
            else:
                woodcock_function_pointers[i] = 0

        return PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers,
            element_csr=element_csr
        )
