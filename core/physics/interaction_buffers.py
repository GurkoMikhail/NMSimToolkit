import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Time, Length
from core.other.typing_definitions import Index, ID, Energy, Float, ProcessID, Species, Charge
from core.other.vectors import Vector3D


from typing import Any

class RNGContext(NamedTuple):
    """
    Explicit CFFI state pointer wrapper to pass into Numba kernels for random number generation.
    """
    next_double: Any
    state_addr: int

    @classmethod
    def from_numpy_rng(cls, rng: np.random.Generator) -> 'RNGContext':
        """
        Extracts the CFFI next_double pointer and state address from a NumPy generator.
        """
        return cls(
            next_double=rng.bit_generator.cffi.next_double,
            state_addr=rng.bit_generator.cffi.state_address
        )


class InteractionBuffer(NamedTuple):
    """
    SoA Ring/Flush buffer for in-place logging of particle interactions.
    Allocated once and reused to avoid array concatenation and memory fragmentation.
    """
    process_id: NDArray[ProcessID]
    volume_id: NDArray[Index]
    material_id: NDArray[Index]
    particle_ID: NDArray[ID]
    energy_deposit: NDArray[Energy]
    scattering_theta: NDArray[Float]
    scattering_phi: NDArray[Float]
    distance_traveled: NDArray[Float]
    species: NDArray[Species]
    Z: NDArray[Charge]

    position: Vector3D
    direction: Vector3D

    cursor: NDArray[Index]  # Length 1, tracks the number of elements written
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        """
        Validates that all arrays within the InteractionBuffer have
        matching capacities and are 1-dimensional.
        """
        self.position.validate()
        self.direction.validate()

        arrays = [
            self.process_id,
            self.volume_id,
            self.material_id,
            self.particle_ID,
            self.energy_deposit,
            self.scattering_theta,
            self.scattering_phi,
            self.distance_traveled,
            self.species,
            self.Z
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
            volume_id=np.empty(capacity, dtype=Index),
            material_id=np.empty(capacity, dtype=Index),
            particle_ID=np.empty(capacity, dtype=ID),
            energy_deposit=np.empty(capacity, dtype=Energy),
            scattering_theta=np.empty(capacity, dtype=Float),
            scattering_phi=np.empty(capacity, dtype=Float),
            distance_traveled=np.empty(capacity, dtype=Float),
            species=np.empty(capacity, dtype=Species),
            Z=np.empty(capacity, dtype=Charge),
            position=Vector3D.allocate(capacity, dtype=Float),
            direction=Vector3D.allocate(capacity, dtype=Float),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer

    def flush_to_dict(self, clear: bool = True) -> dict:
        c = self.cursor_value
        chunk = {
            'process_id': self.process_id[:c].copy(),
            'volume_id': self.volume_id[:c].copy(),
            'material_id': self.material_id[:c].copy(),
            'particle_ID': self.particle_ID[:c].copy(),
            'energy_deposit': self.energy_deposit[:c].copy(),
            'scattering_theta': self.scattering_theta[:c].copy(),
            'scattering_phi': self.scattering_phi[:c].copy(),
            'distance_traveled': self.distance_traveled[:c].copy(),
            'species': self.species[:c].copy(),
            'Z': self.Z[:c].copy(),
            'pos_x': self.position.x[:c].copy(),
            'pos_y': self.position.y[:c].copy(),
            'pos_z': self.position.z[:c].copy(),
            'dir_x': self.direction.x[:c].copy(),
            'dir_y': self.direction.y[:c].copy(),
            'dir_z': self.direction.z[:c].copy(),
        }
        if clear:
            self.reset_cursor()
        return chunk
    

class InitialStateBuffer(NamedTuple):
    """
    SoA Ring/Flush buffer for in-place logging of initial particle states
    upon their first interaction in the volume.
    """
    particle_ID: NDArray[ID]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    emission_position: Vector3D
    emission_direction: Vector3D

    cursor: NDArray[Index]
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        self.emission_position.validate()
        self.emission_direction.validate()

        arrays = [
            self.particle_ID,
            self.emission_time,
            self.emission_energy,
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
            emission_position=Vector3D.allocate(capacity, dtype=Length),
            emission_direction=Vector3D.allocate(capacity, dtype=Float),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer

    def flush_to_dict(self, clear: bool = True) -> dict:
        c = self.cursor_value
        chunk = {
            'particle_ID': self.particle_ID[:c].copy(),
            'emission_time': self.emission_time[:c].copy(),
            'emission_energy': self.emission_energy[:c].copy(),
            'pos_x': self.emission_position.x[:c].copy(),
            'pos_y': self.emission_position.y[:c].copy(),
            'pos_z': self.emission_position.z[:c].copy(),
            'dir_x': self.emission_direction.x[:c].copy(),
            'dir_y': self.emission_direction.y[:c].copy(),
            'dir_z': self.emission_direction.z[:c].copy(),
        }
        if clear:
            self.reset_cursor()
        return chunk


class DeadParticlesBuffer(NamedTuple):
    """
    SoA Ring/Flush buffer for in-place logging of dead particle IDs.
    """
    particle_ID: NDArray[ID]
    cursor: NDArray[Index]
    capacity: int

    @property
    def cursor_value(self) -> int:
        return int(self.cursor[0])

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.cursor_value

    def reset_cursor(self) -> None:
        self.cursor[0] = 0

    def validate(self) -> None:
        if self.particle_ID.ndim != 1:
            raise ValueError("particle_ID array in DeadParticlesBuffer must be 1-dimensional.")
        if self.particle_ID.shape[0] != self.capacity:
            raise ValueError("particle_ID array in DeadParticlesBuffer must have length equal to capacity.")
        if self.cursor.shape != (1,):
            raise ValueError("Cursor must be a 1-dimensional array of length 1.")

    @classmethod
    def allocate(cls, capacity: int) -> 'DeadParticlesBuffer':
        buffer = cls(
            particle_ID=np.empty(capacity, dtype=ID),
            cursor=np.zeros(1, dtype=Index),
            capacity=capacity
        )
        buffer.validate()
        return buffer

    def append(self, particle_ids: NDArray[ID]) -> None:
        """
        Appends dead particle IDs to the buffer and advances the cursor.
        """
        n = len(particle_ids)
        c = self.cursor_value
        if c + n > self.capacity:
            raise ValueError("Insufficient capacity in DeadParticlesBuffer.")
        self.particle_ID[c:c + n] = particle_ids
        self.cursor[0] += n

    def flush_to_array(self, clear: bool = True) -> NDArray[ID]:
        c = self.cursor_value
        chunk = self.particle_ID[:c].copy()
        if clear:
            self.reset_cursor()
        return chunk


class SimulationDataBuffer(NamedTuple):
    """
    Combined Data-Oriented logging buffer for particle transport.
    """
    interactions: InteractionBuffer
    initial_states: InitialStateBuffer
    dead_particles: DeadParticlesBuffer

    @classmethod
    def allocate(cls, interaction_capacity: int, initial_state_capacity: int, dead_particles_capacity: int) -> 'SimulationDataBuffer':
        """
        Allocates interaction, initial state, and dead particle buffers with given capacities.
        """
        return cls(
            interactions=InteractionBuffer.allocate(interaction_capacity),
            initial_states=InitialStateBuffer.allocate(initial_state_capacity),
            dead_particles=DeadParticlesBuffer.allocate(dead_particles_capacity)
        )
