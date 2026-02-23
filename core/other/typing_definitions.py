import numpy as np
from typing import Any
from numpy.typing import NDArray

# Project-wide Float configuration (change this to np.float32 if needed)
Float = np.float64

# Scientific quantities (aliases using the configured Float type)
Length = Float
Energy = Float
Time = Float
Activity = Float
Angle = Float
Density = Float

# Vector types (bound to configured Float)
Vector3D = NDArray[Float]

# Array type
Array = NDArray[Float]

# Identifiers
ID = np.uint64
