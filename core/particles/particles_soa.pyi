import numpy as np
from typing import NamedTuple, List, Any
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species, Index
from core.particles.vector3d_soa import Vector3DSoA

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
    def _validate(self) -> None: ...

class ParticleBank:
    _capacity: int
    _count: int
    _state: ParticleState

    @property
    def capacity(self) -> int: ...
    @property
    def count(self) -> int: ...
    @property
    def state(self) -> ParticleState: ...

    def __init__(self, capacity: int) -> None: ...
    def _allocate_pool(self, capacity: int) -> ParticleState: ...
    def move(self, target_indices: NDArray[Index], distances: NDArray[Length]) -> None: ...
    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None: ...
    def inject(
        self,
        species: NDArray[Species],
        position: Vector3DSoA,
        direction: Vector3DSoA,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
    ) -> NDArray[Index]: ...
