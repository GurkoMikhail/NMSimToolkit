import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float

MaterialInfoDType: np.dtype
MaterialPointerDType: np.dtype

class MaterialBank(NamedTuple):
    mat_info_buffer: NDArray[np.void]
    mat_pointers: NDArray[np.void]
    physics_energy_grid: NDArray[Float]
    physics_lac_table: NDArray[Float]

    def _validate(self) -> None: ...
