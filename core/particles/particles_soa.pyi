import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species, Index
from core.other.vectors_soa import Vector3DSoA
from core.geometry.navigation_state import NavigationState
from core.particles.emission_state import EmissionState


class ParticleState(NamedTuple):
    species: NDArray[Species]
    position: Vector3DSoA
    direction: Vector3DSoA
    energy: NDArray[Energy]
    distance_traveled: NDArray[Length]
    ID: NDArray[ID]
    is_active: NDArray[np.bool_]

    @property
    def capacity(self) -> int: ...

    def validate(self) -> None: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'ParticleState': ...


class ParticleBank(NamedTuple):
    state: ParticleState
    emission_state: EmissionState
    navigation_state: NavigationState
    count_array: NDArray[Index]
    capacity: int

    @classmethod
    def allocate(cls, capacity: int) -> 'ParticleBank': ...

    @property
    def count(self) -> int: ...

    def inject_particles(
        self,
        species: NDArray[Species],
        position: Vector3DSoA,
        direction: Vector3DSoA,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
        distance_traveled: NDArray[Length]
    ) -> NDArray[Index]: ...

    @property
    def active_indices(self) -> NDArray[Index]: ...

    def move(self, target_indices: NDArray[Index], distances: NDArray[Float]) -> None: ...

    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None: ...
