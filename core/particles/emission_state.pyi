import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Time, Length, Float
from core.other.vectors_soa import Vector3DSoA


class EmissionState(NamedTuple):
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: Vector3DSoA
    emission_direction: Vector3DSoA

    @property
    def capacity(self) -> int: ...

    def validate(self) -> None: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'EmissionState': ...
