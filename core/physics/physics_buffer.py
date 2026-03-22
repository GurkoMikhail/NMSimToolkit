import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import Index, CFuncAddress

from core.other.typing_definitions import Float, Charge

class ElementCSR(NamedTuple):
    """
    CSR arrays for element sampling (Z and mass fractions).
    """
    element_offsets: NDArray[Index]
    element_Z: NDArray[Charge]
    element_fraction: NDArray[Float]

class PhysicsBuffer(NamedTuple):
    """
    Data buffer compiled by PhysicsCompiler containing:
    - Dynamic MaterialBank
    - Majorant Material Map for Woodcock Tracking
    - Woodcock Function Pointers (@cfunc addresses)
    - ElementCSR containing Z and mass fractions arrays
    """
    material_bank: MaterialBank
    majorant_material_map: NDArray[Index]
    woodcock_function_pointers: NDArray[CFuncAddress]

    element_csr: ElementCSR
