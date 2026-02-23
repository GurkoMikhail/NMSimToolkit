from typing import TypeAlias, Final
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
