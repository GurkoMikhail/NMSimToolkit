from typing import TypeAlias, Final, Any
import numpy as np
from numpy.typing import NDArray

Float: TypeAlias = np.float64
Length: TypeAlias = Float
Energy: TypeAlias = Float
Time: TypeAlias = Float
Activity: TypeAlias = Float
Angle: TypeAlias = Float
Density: TypeAlias = Float

Vector3D: TypeAlias = NDArray[Float]
ID: TypeAlias = np.uint64
Species: TypeAlias = np.uint8
Index: TypeAlias = np.int64
ShapeID: TypeAlias = np.int32
CMaterialFunc: Any
CFuncAddress: TypeAlias = np.uint64

NumbaFloat: Any
NumbaIndex: Any

ProcessID: TypeAlias = np.uint8
Charge: TypeAlias = np.int8
