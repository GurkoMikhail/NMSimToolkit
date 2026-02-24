import numpy as np
from typing import TypeVar, Union, Any
from numpy.typing import NDArray

# Precision TypeVar to support both float32 and float64
Precision = TypeVar("Precision", np.float32, np.float64)

# Scalar types for scientific quantities
# In implementation, these are usually floats, but can be numpy scalars
Scalar = Union[float, np.float32, np.float64]

# Scientific quantities (aliases for clarity)
Length = Scalar
Energy = Scalar
Time = Scalar
Activity = Scalar
Angle = Scalar
Density = Scalar

# Vector types
# Vector3D[np.float64] or Vector3D[np.float32]
Vector3D = NDArray[Precision]

# Generic Array type
Array = NDArray[Precision]

# Identifiers
ID = np.uint64
