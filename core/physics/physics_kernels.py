import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index


@njit(cache=True, fastmath=True, inline='always')
def _get_macroscopic_cross_sections(
    energy: Float,
    material_id: int,
    start_idx: int,
    length: int,
    energy_grid: NDArray[Float],
    lac_table: NDArray[Float],
    out_lacs_slice: NDArray[Float]
) -> None:
    """
    Device function to compute Macroscopic Cross Sections (LACs) for all processes
    of a specific material via fast inline linear interpolation.

    Instead of returning an array, it receives a pre-allocated 1D slice (out_lacs_slice)
    and updates it in-place to enforce ZERO ALLOCATIONS.

    Arguments:
    - energy: scalar, the energy of the particle
    - material_id: ID of the material
    - start_idx: offset index in the unified energy_grid for this material
    - length: length of the energy grid for this material
    - energy_grid: 1D unified array of all energy points
    - lac_table: 2D array (total_energy_points, num_processes)
    - out_lacs_slice: 1D view of size `num_processes` to be updated IN-PLACE.
    """
    if length == 0:
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = 0.0
        return

    # Extract the material-specific grid slice
    mat_energy_grid = energy_grid[start_idx : start_idx + length]

    # Find insertion point (similar to side='left' behavior in np.interp)
    idx = np.searchsorted(mat_energy_grid, energy)

    if idx == 0:
        # Energy is below or equal to the lowest grid point
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = lac_table[start_idx, i]
    elif idx >= length:
        # Energy is above or equal to the highest grid point
        for i in range(len(out_lacs_slice)):
            out_lacs_slice[i] = lac_table[start_idx + length - 1, i]
    else:
        # Linear interpolation
        e0 = mat_energy_grid[idx - 1]
        e1 = mat_energy_grid[idx]

        # Branchless prevention of division-by-zero
        delta_e = e1 - e0
        fraction = (energy - e0) / delta_e if delta_e > 0.0 else 0.0

        for i in range(len(out_lacs_slice)):
            v0 = lac_table[start_idx + idx - 1, i]
            v1 = lac_table[start_idx + idx, i]
            out_lacs_slice[i] = v0 + fraction * (v1 - v0)
