import numpy as np
from typing import NamedTuple, Optional, Union
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Vector3D, Species


class ParticleArraySoA(NamedTuple):
    species: NDArray[Species]
    position: NDArray[Float]
    direction: NDArray[Float]
    energy: NDArray[Energy]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: NDArray[Float]
    emission_direction: NDArray[Float]
    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

    @property
    def size(self) -> int: ...

    @classmethod
    def create(
        cls,
        species: NDArray[Species],
        position: NDArray[Float],
        direction: NDArray[Float],
        energy: NDArray[Energy],
        emission_time: Optional[NDArray[Time]] = None,
        emission_position: Optional[NDArray[Float]] = None,
        emission_direction: Optional[NDArray[Float]] = None,
        distance_traveled: Optional[NDArray[Length]] = None
    ) -> 'ParticleArraySoA': ...
