import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Time, Length, Float
from core.other.vectors_soa import Vector3DSoA


class EmissionState(NamedTuple):
    """
    Structure of Arrays (SoA) database for particle emission states.
    Contains 1D flat, C-contiguous NumPy arrays for the initial
    conditions of particles when they were generated.
    """
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    # Emission Position Vector
    emission_position: Vector3DSoA

    # Emission Direction Vector
    emission_direction: Vector3DSoA

    @property
    def capacity(self) -> int:
        return self.emission_time.shape[0]

    def validate(self) -> None:
        """
        Validates that all arrays within the EmissionState have
        matching capacities and are 1-dimensional.
        """
        self.emission_position.validate()
        self.emission_direction.validate()

        arrays = [
            self.emission_time,
            self.emission_energy,
        ]

        # All base fields should be 1-dimensional
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in EmissionState must be 1-dimensional.")

        # Validate lengths match the pool capacity
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in EmissionState must have the same length (capacity).")

        # Validate vector lengths against capacity
        if self.emission_position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in EmissionState must have the same length as the base arrays.")

    @classmethod
    def allocate(cls, capacity: int) -> 'EmissionState':
        """
        Allocates an empty EmissionState with the specified capacity.
        """
        buffer = cls(
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3DSoA.allocate(capacity, dtype=Length),
            emission_direction=Vector3DSoA.allocate(capacity, dtype=Float)
        )
        buffer.validate()
        return buffer
