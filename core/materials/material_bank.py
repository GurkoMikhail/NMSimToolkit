import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

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


class MaterialBank(NamedTuple):
    """
    Flat AoS numpy arrays of material properties and physical cross-sections (LAC).
    Follows Zero Memory Waste principles by compiling only what's used in the scene.
    """
    mat_info_buffer: NDArray[np.void]
    mat_pointers: NDArray[np.void]
    physics_energy_grid: NDArray[Float]
    physics_lac_table: NDArray[Float]

    def _validate(self) -> None:
        """ Validates the integrity of the material bank arrays. """
        assert self.mat_info_buffer.dtype == MaterialInfoDType, "Invalid mat_info_buffer dtype"
        assert self.mat_pointers.dtype == MaterialPointerDType, "Invalid mat_pointers dtype"
        assert self.physics_energy_grid.ndim == 1, "physics_energy_grid must be 1D"
        assert self.physics_lac_table.ndim == 2, "physics_lac_table must be 2D"
