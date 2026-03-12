from typing import TypeAlias, Final
import numpy as np
from numpy.typing import NDArray
import numba.types as ntypes
from numba.core.types.scalars import Float as NumbaFloatBase, Integer as NumbaIntBase

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
CFuncAddress: TypeAlias = np.int64

NumbaFloat: NumbaFloatBase = ntypes.float64
NumbaIndex: NumbaIntBase = ntypes.int64
