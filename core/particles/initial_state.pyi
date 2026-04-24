import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Time, ID
from core.other.vectors import Vector3D


class InitialState(NamedTuple):
    ID: NDArray[ID]
    has_interacted: NDArray[np.bool_]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: Vector3D
    emission_direction: Vector3D

    @property
    def capacity(self) -> int: ...

    def validate(self) -> None: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialState': ...
