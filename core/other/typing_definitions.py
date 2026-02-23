import numpy as np
from typing import Union, Any
from numpy.typing import NDArray

# Static Precision configuration (change this to np.float32 if needed)
Precision = np.float64

# Scalar types for scientific quantities
Scalar = Union[float, Precision]

# Scientific quantities (aliases for clarity)
Length = Scalar
Energy = Scalar
Time = Scalar
Activity = Scalar
Angle = Scalar
Density = Scalar

# Vector types
Vector3D = NDArray[Precision]

# Array type
Array = NDArray[Precision]

# Identifiers
ID = np.uint64
