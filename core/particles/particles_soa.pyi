import numpy as np
from typing import NamedTuple, List
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species

class Vector3DSoA(NamedTuple):
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

class ParticleState(NamedTuple):
    is_active: NDArray[np.bool_]
    species: NDArray[Species]

    position: Vector3DSoA
    direction: Vector3DSoA

    energy: NDArray[Energy]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    emission_position: Vector3DSoA
    emission_direction: Vector3DSoA

    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

class ParticleBank:
    capacity: int
    count: int
    state: ParticleState

    def __init__(self, capacity: int) -> None: ...
    def _allocate_pool(self, capacity: int) -> ParticleState: ...
    def move(self, target_indices: NDArray[np.int64], distances: NDArray[Length]) -> None: ...
    def rotate(self, target_indices: NDArray[np.int64], thetas: NDArray[Float], phis: NDArray[Float]) -> None: ...
    def inject(
        self,
        species: NDArray[Species],
        position_x: NDArray[Float],
        position_y: NDArray[Float],
        position_z: NDArray[Float],
        direction_x: NDArray[Float],
        direction_y: NDArray[Float],
        direction_z: NDArray[Float],
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
    ) -> NDArray[np.int64]: ...
