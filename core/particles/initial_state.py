import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Time, Length, Float, ID
from core.other.vectors import Vector3D


class InitialState(NamedTuple):
    """
    Structure of Arrays (SoA) database for particle initial states.
    Contains 1D flat, C-contiguous NumPy arrays for the initial
    conditions and identity of particles.
    """
    ID: NDArray[ID]
    has_interacted: NDArray[np.bool_]

    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    # Emission Position Vector
    emission_position: Vector3D

    # Emission Direction Vector
    emission_direction: Vector3D

    @property
    def capacity(self) -> int:
        return self.emission_time.shape[0]

    def validate(self) -> None:
        """
        Validates that all arrays within the InitialState have
        matching capacities and are 1-dimensional.
        """
        self.emission_position.validate()
        self.emission_direction.validate()

        arrays = [
            self.ID,
            self.has_interacted,
            self.emission_time,
            self.emission_energy,
        ]

        # All base fields should be 1-dimensional
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in InitialState must be 1-dimensional.")

        # Validate lengths match the pool capacity
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in InitialState must have the same length (capacity).")

        # Validate vector lengths against capacity
        if self.emission_position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in InitialState must have the same length as the base arrays.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialState':
        """
        Allocates an empty InitialState with the specified capacity.
        """
        buffer = cls(
            ID=np.empty(capacity, dtype=ID),
            has_interacted=np.zeros(capacity, dtype=np.bool_),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3D.allocate(capacity, dtype=Length),
            emission_direction=Vector3D.allocate(capacity, dtype=Float)
        )
        buffer.validate()
        return buffer
