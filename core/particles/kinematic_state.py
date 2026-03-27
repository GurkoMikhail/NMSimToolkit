import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, Length, Species
from core.other.vectors_soa import Vector3DSoA

class KinematicState(NamedTuple):
    """
    Structure of Arrays (SoA) database for particle kinematic states.
    Contains strictly 1D flat, C-contiguous NumPy arrays for "hot" data
    actively used by the physics kernels.
    """
    is_active: NDArray[np.bool_]
    species: NDArray[Species]

    # Position Vector
    position: Vector3DSoA

    # Direction Vector
    direction: Vector3DSoA

    energy: NDArray[Energy]

    distance_traveled: NDArray[Length]

    @property
    def capacity(self) -> int:
        return self.is_active.shape[0]

    def validate(self) -> None:
        """
        Validates that all arrays within the KinematicState have
        matching capacities and are 1-dimensional.
        """
        self.position.validate()
        self.direction.validate()

        arrays = [
            self.is_active,
            self.species,
            self.energy,
            self.distance_traveled,
        ]

        # All base fields should be 1-dimensional
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in KinematicState must be 1-dimensional.")

        # Validate lengths match the pool capacity
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in KinematicState must have the same length (capacity).")

        # Validate vector lengths against capacity
        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in KinematicState must have the same length as the base arrays.")

    @classmethod
    def allocate(cls, capacity: int) -> 'KinematicState':
        """
        Allocates an empty KinematicState with the specified capacity.
        """
        buffer = cls(
            is_active=np.zeros(capacity, dtype=np.bool_),
            species=np.empty(capacity, dtype=Species),
            position=Vector3DSoA.allocate(capacity, dtype=Length),
            direction=Vector3DSoA.allocate(capacity, dtype=Float),
            energy=np.empty(capacity, dtype=Energy),
            distance_traveled=np.empty(capacity, dtype=Length),
        )
        buffer.validate()
        return buffer
