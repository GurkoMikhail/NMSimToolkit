import numpy as np
from typing import NamedTuple, Any
from numpy.typing import NDArray

from core.other.typing_definitions import Index, ID, Energy, Float, ProcessID, Time, Length
from core.other.vectors_soa import Vector3DSoA

class RNGContext(NamedTuple):
    next_double: Any
    state_addr: int

    @classmethod
    def from_numpy_rng(cls, rng: np.random.Generator) -> 'RNGContext':
        return cls(
            next_double=rng.bit_generator.cffi.next_double,
            state_addr=rng.bit_generator.cffi.state_address
        )


class InteractionBuffer(NamedTuple):
    process_id: NDArray[ProcessID]
    particle_ID: NDArray[ID]
    energy_deposit: NDArray[Energy]
    scattering_theta: NDArray[Float]
    scattering_phi: NDArray[Float]
    volume_id: NDArray[Index]
    material_id: NDArray[Index]

    position: Vector3DSoA
    direction: Vector3DSoA

    cursor: NDArray[Index]
    capacity: int

    def validate(self) -> None:
        self.position.validate()
        self.direction.validate()

        arrays = [
            self.process_id,
            self.particle_ID,
            self.energy_deposit,
            self.scattering_theta,
            self.scattering_phi,
            self.volume_id,
            self.material_id
        ]

        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in InteractionBuffer must be 1-dimensional.")
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in InteractionBuffer must have the same length (capacity).")
        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in InteractionBuffer must have the same length as the base arrays.")
        if self.cursor.shape != (1,):
            raise ValueError("Cursor must be a 1-dimensional array of length 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InteractionBuffer':
        buffer = cls(
            process_id=np.empty(capacity, dtype=ProcessID),
            particle_ID=np.empty(capacity, dtype=ID),
            energy_deposit=np.empty(capacity, dtype=Energy),
            scattering_theta=np.empty(capacity, dtype=Float),
            scattering_phi=np.empty(capacity, dtype=Float),
            volume_id=np.empty(capacity, dtype=Index),
            material_id=np.empty(capacity, dtype=Index),
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

class InitialStateBuffer(NamedTuple):
    particle_ID: NDArray[ID]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]
    emission_position: Vector3DSoA
    emission_direction: Vector3DSoA
    cursor: NDArray[Index]
    capacity: int

    def validate(self) -> None:
        self.emission_position.validate()
        self.emission_direction.validate()

        arrays = [
            self.particle_ID,
            self.emission_time,
            self.emission_energy
        ]

        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in InitialStateBuffer must be 1-dimensional.")
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in InitialStateBuffer must have the same length (capacity).")
        if self.emission_position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in InitialStateBuffer must have the same length as the base arrays.")
        if self.cursor.shape != (1,):
            raise ValueError("Cursor must be a 1-dimensional array of length 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'InitialStateBuffer':
        buffer = cls(
            particle_ID=np.empty(capacity, dtype=ID),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3DSoA.allocate(capacity, dtype=Length),
            emission_direction=Vector3DSoA.allocate(capacity, dtype=Float),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer
