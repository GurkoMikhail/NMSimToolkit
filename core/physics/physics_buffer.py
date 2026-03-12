from typing import NamedTuple
from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import Index

class PhysicsBuffer(NamedTuple):
    """
    Data buffer compiled by PhysicsCompiler containing:
    - Dynamic MaterialBank
    - Majorant Material Map for Woodcock Tracking
    - Woodcock Function Pointers (@cfunc addresses)
    """
    material_bank: MaterialBank
    majorant_material_map: NDArray[Index]
    woodcock_function_pointers: NDArray[Index]
