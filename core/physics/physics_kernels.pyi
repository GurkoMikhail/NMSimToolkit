from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.materials.material_bank import MaterialBank

def _get_macroscopic_cross_sections(
    energy: Float,
    material_id: Index,
    material_bank: MaterialBank,
    out_lacs_slice: NDArray[Float]
) -> None: ...
