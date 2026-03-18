import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Index, ID, Energy, Float, ProcessID
from core.other.vectors_soa import Vector3DSoA


from typing import Any

class RNGContext(NamedTuple):
    """
    Explicit CFFI state pointer wrapper to pass into Numba kernels for random number generation.
    """
    next_double: Any
    state_addr: int


class InteractionBuffer(NamedTuple):
    """
    SoA Ring/Flush buffer for in-place logging of particle interactions.
    Allocated once and reused to avoid array concatenation and memory fragmentation.
    """
    process_id: NDArray[ProcessID]
    particle_ID: NDArray[ID]
    energy_deposit: NDArray[Energy]
    scattering_theta: NDArray[Float]
    scattering_phi: NDArray[Float]

    position: Vector3DSoA
    direction: Vector3DSoA

    cursor: NDArray[Index]  # Length 1, tracks the number of elements written
    capacity: int

    def validate(self) -> None:
        """
        Validates that all arrays within the InteractionBuffer have
        matching capacities and are 1-dimensional.
        """
        self.position.validate()
        self.direction.validate()

        arrays = [
            self.process_id,
            self.particle_ID,
            self.energy_deposit,
            self.scattering_theta,
            self.scattering_phi
        ]

        # All base fields should be 1-dimensional
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in InteractionBuffer must be 1-dimensional.")

        # Validate lengths match the pool capacity
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in InteractionBuffer must have the same length (capacity).")

        # Validate vector lengths against capacity
        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in InteractionBuffer must have the same length as the base arrays.")

        if self.cursor.shape != (1,):
            raise ValueError("Cursor must be a 1-dimensional array of length 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InteractionBuffer':
        """
        Allocates an empty InteractionBuffer with the specified capacity.
        """
        buffer = cls(
            process_id=np.empty(capacity, dtype=ProcessID),
            particle_ID=np.empty(capacity, dtype=ID),
            energy_deposit=np.empty(capacity, dtype=Energy),
            scattering_theta=np.empty(capacity, dtype=Float),
            scattering_phi=np.empty(capacity, dtype=Float),
            position=Vector3DSoA(
                x=np.empty(capacity, dtype=Float),
                y=np.empty(capacity, dtype=Float),
                z=np.empty(capacity, dtype=Float)
            ),
            direction=Vector3DSoA(
                x=np.empty(capacity, dtype=Float),
                y=np.empty(capacity, dtype=Float),
                z=np.empty(capacity, dtype=Float)
            ),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer
