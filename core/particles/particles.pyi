import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species, Index
from core.other.vectors import Vector3D
from core.geometry.navigation_state import NavigationState
from core.particles.initial_state import InitialState
from core.particles.kinematic_state import KinematicState


class ParticleBank(NamedTuple):
    state: KinematicState
    initial_state: InitialState
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
        position: Vector3D,
        direction: Vector3D,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
        distance_traveled: NDArray[Length]
    ) -> NDArray[Index]: ...

    @property
    def active_indices(self) -> NDArray[Index]: ...

    def move(self, target_indices: NDArray[Index], distances: NDArray[Float]) -> None: ...

    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None: ...
