from typing import NamedTuple
from numpy.typing import NDArray

from core.materials.material_bank import MaterialBank
from core.other.typing_definitions import Index, CFuncAddress, Float, Charge

class ElementCSR(NamedTuple):
    element_offsets: NDArray[Index]
    element_Z: NDArray[Charge]
    element_fraction: NDArray[Float]

class PhysicsBuffer(NamedTuple):
    material_bank: MaterialBank
    majorant_material_map: NDArray[Index]
    woodcock_function_pointers: NDArray[CFuncAddress]
    element_csr: ElementCSR
