import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.materials.material_bank import MaterialBank


@njit(cache=True, inline='always')
def _get_macroscopic_cross_sections(
    energy: Float,
    material_id: Index,
    material_bank: MaterialBank,
    out_lacs_slice: NDArray[Float]
) -> None:
    """
    Device function to compute Macroscopic Cross Sections (LACs) for all processes
    of a specific material via fast inline linear interpolation.

    Instead of returning an array, it receives a pre-allocated 1D slice (out_lacs_slice)
    and updates it in-place to enforce ZERO ALLOCATIONS.
    """
    pointer = material_bank.mat_pointers[material_id]
    start_idx = pointer['start_idx']
    length = pointer['length']

    if length == 0:
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = 0.0
        return

    # Extract the material-specific grid slice
    mat_energy_grid = material_bank.physics_energy_grid[start_idx : start_idx + length]

    # Find insertion point (similar to side='left' behavior in np.interp)
    idx = np.searchsorted(mat_energy_grid, energy)

    if idx == 0:
        # Energy is below or equal to the lowest grid point
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = material_bank.physics_lac_table[start_idx, i]
    elif idx >= length:
        # Energy is above or equal to the highest grid point
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = material_bank.physics_lac_table[start_idx + length - 1, i]
    else:
        # Linear interpolation
        e0 = mat_energy_grid[idx - 1]
        e1 = mat_energy_grid[idx]

        # Branchless prevention of division-by-zero
        delta_e = e1 - e0
        fraction = (energy - e0) / delta_e if delta_e > 0.0 else 0.0

        for i in range(len(out_lacs_slice)):
            v0 = material_bank.physics_lac_table[start_idx + idx - 1, i]
            v1 = material_bank.physics_lac_table[start_idx + idx, i]
            out_lacs_slice[i] = v0 + fraction * (v1 - v0)
