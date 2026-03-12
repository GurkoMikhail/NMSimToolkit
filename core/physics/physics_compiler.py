import numpy as np
from typing import List
from numpy.typing import NDArray

from core.geometry.volumes import Volume
from core.physics.processes import Process
from core.materials.material_bank import MaterialBank
from core.physics.physics_buffer import PhysicsBuffer
from core.other.typing_definitions import Index

class PhysicsCompiler:
    """
    Compiles physics data (MaterialBank, Majorant Material Map, Woodcock Pointers)
    from the scene for fast Numba execution.
    """

    def compile_scene(self, root_volume: Volume, processes_list: List[Process]) -> PhysicsBuffer:
        """
        Builds the complete PhysicsBuffer from the root volume and active processes.
        """
        # Ensure we have a unique list of materials used in the entire scene hierarchy
        all_materials = root_volume.material_list
        unique_materials = list(set(all_materials))

        # Build dynamic material bank (Zero Memory Waste)
        material_bank = MaterialBank.build_from_scene(unique_materials, processes_list)

        # Get the flattened scene to ensure indexes match the GeometryBuffer
        flat_list = root_volume.flattened_scene.flat_list
        capacity = len(flat_list)

        majorant_material_map = np.zeros(capacity, dtype=Index)
        woodcock_function_pointers = np.zeros(capacity, dtype=Index)

        for i, (vol, _, _) in enumerate(flat_list):
            majorant_material_map[i] = vol.majorant_material.ID
            woodcock_function_pointers[i] = vol.cfunc_address

        return PhysicsBuffer(
            material_bank=material_bank,
            majorant_material_map=majorant_material_map,
            woodcock_function_pointers=woodcock_function_pointers
        )
