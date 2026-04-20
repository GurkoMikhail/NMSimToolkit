import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, Length, Species
from core.other.vectors import Vector3D

class KinematicState(NamedTuple):
    is_active: NDArray[np.bool_]
    species: NDArray[Species]
    position: Vector3D
    direction: Vector3D
    energy: NDArray[Energy]
    distance_traveled: NDArray[Length]

    @property
    def capacity(self) -> int: ...

    def validate(self) -> None: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'KinematicState': ...
