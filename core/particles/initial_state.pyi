import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Time, Length, Float, ID
from core.other.vectors_soa import Vector3DSoA


class InitialState(NamedTuple):
    ID: NDArray[ID]
    has_interacted: NDArray[np.bool_]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: Vector3DSoA
    emission_direction: Vector3DSoA

    @property
    def capacity(self) -> int: ...

    def validate(self) -> None: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialState': ...
