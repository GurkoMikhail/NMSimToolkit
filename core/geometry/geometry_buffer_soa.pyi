import numpy as np
from typing import NamedTuple, Any
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index

class GeometryBuffer(NamedTuple):
    node_type: NDArray[np.uint8]
    material_id: NDArray[np.uint32]
    miss_index: NDArray[Index]

    half_size_x: NDArray[Float]
    half_size_y: NDArray[Float]
    half_size_z: NDArray[Float]

    R_xx: NDArray[Float]
    R_xy: NDArray[Float]
    R_xz: NDArray[Float]

    R_yx: NDArray[Float]
    R_yy: NDArray[Float]
    R_yz: NDArray[Float]

    R_zx: NDArray[Float]
    R_zy: NDArray[Float]
    R_zz: NDArray[Float]

    T_x: NDArray[Float]
    T_y: NDArray[Float]
    T_z: NDArray[Float]

    def _validate(self) -> None: ...
