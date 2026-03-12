import numpy as np
from typing import List
from numpy.typing import NDArray

from core.materials.materials import Material
from core.physics.processes import Process
from core.other.typing_definitions import Float, Index


MaterialInfoDType = np.dtype([
    ('density', Float),
    ('Z', Float),
    ('A', Float)
])

MaterialPointerDType = np.dtype([
    ('start_idx', Index),
    ('length', Index)
])


class MaterialBank:
    """
    Dynamically builds flat AoS numpy arrays of material properties and physical cross-sections (LAC).
    Follows Zero Memory Waste principles by compiling only what's used in the scene.
    """
    mat_info_buffer: NDArray[np.void]
    mat_pointers: NDArray[np.void]
    physics_energy_grid: NDArray[Float]
    physics_lac_table: NDArray[Float]

    @classmethod
    def build_from_scene(cls, materials_list: List[Material], processes_list: List[Process]) -> 'MaterialBank':
        """
        Constructs the MaterialBank from the unique list of materials and active processes.
        """
        bank = cls()

        # We need the maximum ID to size the AoS lookup arrays correctly.
        import settings.database_setting as settings
        capacity = len(settings.material_database) + 1

        bank.mat_info_buffer = np.zeros(capacity, dtype=MaterialInfoDType)
        bank.mat_pointers = np.zeros(capacity, dtype=MaterialPointerDType)

        all_energies = []
        all_lacs = []
        current_idx = 0

        for material in materials_list:
            mat_id = material.ID

            # Populate info buffer
            bank.mat_info_buffer[mat_id]['density'] = material.density
            bank.mat_info_buffer[mat_id]['Z'] = material.Zeff
            bank.mat_info_buffer[mat_id]['A'] = 0.0 # 'A' is not currently in Material dict, defaulting to 0.0 unless needed

            # Since each process has its own attenuation function, but they all share the same energy grid
            # for a specific material in attenuation_database, we can just grab the energy grid from the first process.
            # However, it's safer to query each process and see if their energy grids match, or just interpolate
            # them onto a common grid for this material.
            # In attenuation_functions.py, the energy grid is from the database.

            # Let's use the energy grid from the first active process for this material.
            # If there are no processes, we just add an empty grid.
            if len(processes_list) == 0:
                bank.mat_pointers[mat_id]['start_idx'] = current_idx
                bank.mat_pointers[mat_id]['length'] = 0
                continue

            first_proc = processes_list[0]
            # AttenuationFunction __call__ supports NDArray.
            # We can get the internal grid directly from attenuation_function dict:
            try:
                energy_grid, _ = first_proc.attenuation_function[material]
            except KeyError:
                # Material might not be in the database for this process, fallback to a default grid or skip
                energy_grid = np.array([1e-3, 100.0], dtype=Float)

            length = len(energy_grid)

            bank.mat_pointers[mat_id]['start_idx'] = current_idx
            bank.mat_pointers[mat_id]['length'] = length
            current_idx += length

            all_energies.append(energy_grid)

            lac_matrix = np.zeros((length, len(processes_list)), dtype=Float)
            for p_idx, process in enumerate(processes_list):
                # get_LAC calls attenuation_function which interpolates or returns exact.
                # Since we are querying at the exact energy_grid points of the material, it should be fast and exact.
                # However, some processes might not have the exact same grid, so we just use attenuation_function
                # to get the values at `energy_grid`.
                lacs = process.get_LAC(type('ParticleDummy', (object,), {'energy': energy_grid})(), material) # type: ignore
                lac_matrix[:, p_idx] = lacs

            all_lacs.append(lac_matrix)

        if len(all_energies) > 0:
            bank.physics_energy_grid = np.concatenate(all_energies)
            bank.physics_lac_table = np.concatenate(all_lacs)
        else:
            bank.physics_energy_grid = np.array([], dtype=Float)
            bank.physics_lac_table = np.empty((0, len(processes_list)), dtype=Float)

        return bank
