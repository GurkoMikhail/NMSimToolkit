import numpy as np
from typing import NamedTuple, Optional, Union
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Vector3D, Species


class ParticleArraySoA(NamedTuple):
    """
    Structure of Arrays (SoA) Numba-compatible container for particles.
    Replaces the AoS np.ndarray structured array design to improve cache locality
    and vectorization capabilities.
    """
    species: NDArray[Species]
    position: NDArray[Float]  # Shape (N, 3)
    direction: NDArray[Float] # Shape (N, 3)
    energy: NDArray[Energy]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: NDArray[Float] # Shape (N, 3)
    emission_direction: NDArray[Float] # Shape (N, 3)
    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

    @property
    def size(self) -> int:
        return self.energy.size

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
    ) -> 'ParticleArraySoA':
        """
        Initializes the SoA container ensuring arrays are contiguous and properly dimensioned.
        """
        n = energy.size

        # Ensure C-contiguous arrays for optimal memory access in Numba
        species_arr = np.ascontiguousarray(species, dtype=Species)
        position_arr = np.ascontiguousarray(position, dtype=Float)
        direction_arr = np.ascontiguousarray(direction, dtype=Float)
        energy_arr = np.ascontiguousarray(energy, dtype=Energy)

        if emission_time is None:
            emission_time_arr = np.zeros(n, dtype=Time)
        else:
            emission_time_arr = np.ascontiguousarray(emission_time, dtype=Time)

        if emission_position is None:
            emission_position_arr = np.copy(position_arr)
        else:
            emission_position_arr = np.ascontiguousarray(emission_position, dtype=Float)

        if emission_direction is None:
            emission_direction_arr = np.copy(direction_arr)
        else:
            emission_direction_arr = np.ascontiguousarray(emission_direction, dtype=Float)

        if distance_traveled is None:
            distance_traveled_arr = np.zeros(n, dtype=Length)
        else:
            distance_traveled_arr = np.ascontiguousarray(distance_traveled, dtype=Length)

        # Generates ID array (assuming ID starts at 0 for standalone testing)
        # Note: In production, ID generation should be synced globally as in AoS
        id_arr = np.arange(0, n, dtype=ID)

        return cls(
            species=species_arr,
            position=position_arr,
            direction=direction_arr,
            energy=energy_arr,
            emission_time=emission_time_arr,
            emission_energy=np.copy(energy_arr),
            emission_position=emission_position_arr,
            emission_direction=emission_direction_arr,
            distance_traveled=distance_traveled_arr,
            ID=id_arr
        )
