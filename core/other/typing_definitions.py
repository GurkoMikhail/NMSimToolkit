from typing import TypeAlias, Final
import numpy as np
from numpy.typing import NDArray
from numba import from_dtype

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
from typing import Any
import ctypes

ShapeID: TypeAlias = np.int32

# CFUNCTYPE strictly signatured for material lookup functions
CMaterialFunc = ctypes.CFUNCTYPE(
    np.ctypeslib.as_ctypes_type(Index),
    np.ctypeslib.as_ctypes_type(Float),
    np.ctypeslib.as_ctypes_type(Float),
    np.ctypeslib.as_ctypes_type(Float)
)

CFuncAddress: TypeAlias = np.uint64

NumbaFloat = from_dtype(np.dtype(Float))
NumbaIndex = from_dtype(np.dtype(Index))

ProcessID: TypeAlias = np.uint8
Charge: TypeAlias = np.int8
