import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

from core.other.typing_definitions import Energy, Length, Species
from core.other.vectors_soa import Vector3DSoA

class KinematicState(NamedTuple):
    species: NDArray[Species]
    position: Vector3DSoA
    direction: Vector3DSoA
    energy: NDArray[Energy]
    distance_traveled: NDArray[Length]
    is_active: NDArray[np.bool_]

    @property
    def capacity(self) -> int: ...

    def validate(self) -> None: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'KinematicState': ...
