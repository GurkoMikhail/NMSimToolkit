import numpy as np
from typing import List
from numpy.typing import NDArray

from core.geometry.volumes import Volume
from core.physics.processes import Process
from core.materials.material_bank import MaterialBank, MaterialInfoDType, MaterialPointerDType
from core.physics.physics_buffer import PhysicsBuffer
from core.materials.materials import Material
from core.other.typing_definitions import Index, Float

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
                lacs = process.get_LAC(type('ParticleDummy', (object,), {'energy': energy_grid})(), material) # type: ignore
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

    def compile_scene(self, root_volume: Volume, processes_list: List[Process]) -> PhysicsBuffer:
        """
        Builds the complete PhysicsBuffer from the root volume and active processes.
        """
        # Ensure we have a unique list of materials used in the entire scene hierarchy
        all_materials = root_volume.material_list
        unique_materials = list(set(all_materials))

        # Build dynamic material bank (Zero Memory Waste)
        material_bank = self._build_material_bank(unique_materials, processes_list)

        # Get the flattened scene to ensure indexes match the GeometryBuffer
        flat_list = root_volume.flattened_scene.flat_list
        capacity = len(flat_list)

        majorant_material_map = np.zeros(capacity, dtype=Index)
        woodcock_function_pointers = np.zeros(capacity, dtype=Index)

        for i, (vol, _, _) in enumerate(flat_list):
            majorant_material_map[i] = vol.majorant_material.ID
            woodcock_function_pointers[i] = vol.material_cfunc_address

        return PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers
        )
