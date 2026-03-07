import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species
from core.other.vectors_soa import Vector3DSoA, validate_vector3d_soa


class ParticleState(NamedTuple):
    """
    Structure of Arrays (SoA) database for particle states.
    Contains strictly 1D flat, C-contiguous NumPy arrays.
    """
    species: NDArray[Species]

    # Position Vector
    position: Vector3DSoA

    # Direction Vector
    direction: Vector3DSoA

    energy: NDArray[Energy]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    # Emission Position Vector
    emission_position: Vector3DSoA

    # Emission Direction Vector
    emission_direction: Vector3DSoA

    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

    # Object Pool lifecycle flag
    is_active: NDArray[np.bool_]


def validate_particle_state(state: ParticleState) -> None:
    """
    Validates that all arrays within the ParticleState have
    matching capacities and are 1-dimensional.
    """
    validate_vector3d_soa(state.position)
    validate_vector3d_soa(state.direction)
    validate_vector3d_soa(state.emission_position)
    validate_vector3d_soa(state.emission_direction)

    arrays = [
        state.species,
        state.energy,
        state.emission_time,
        state.emission_energy,
        state.distance_traveled,
        state.ID,
        state.is_active
    ]

    # All base fields should be 1-dimensional
    for arr in arrays:
        if arr.ndim != 1:
            raise ValueError("All arrays in ParticleState must be 1-dimensional.")

    capacity = state.is_active.shape[0]

    # Validate lengths match the pool capacity
    for arr in arrays:
        if arr.shape[0] != capacity:
            raise ValueError("All arrays in ParticleState must have the same length (capacity).")

    # Validate vector lengths against capacity
    if state.position.x.shape[0] != capacity:
        raise ValueError("Vector components in ParticleState must have the same length as the base arrays.")

class ParticleBank:
    """
    Facade for managing the object pool of SoA-based particles.
    Separates OOP lifecycle management from Numba computational kernels.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.count = 0

        # Instantiate base flat arrays
        self.state = ParticleState(
            species=np.empty(capacity, dtype=Species),
            position=Vector3DSoA(
                x=np.empty(capacity, dtype=Length),
                y=np.empty(capacity, dtype=Length),
                z=np.empty(capacity, dtype=Length)
            ),
            direction=Vector3DSoA(
                x=np.empty(capacity, dtype=Float),
                y=np.empty(capacity, dtype=Float),
                z=np.empty(capacity, dtype=Float)
            ),
            energy=np.empty(capacity, dtype=Energy),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),
            emission_position=Vector3DSoA(
                x=np.empty(capacity, dtype=Length),
                y=np.empty(capacity, dtype=Length),
                z=np.empty(capacity, dtype=Length)
            ),
            emission_direction=Vector3DSoA(
                x=np.empty(capacity, dtype=Float),
                y=np.empty(capacity, dtype=Float),
                z=np.empty(capacity, dtype=Float)
            ),
            distance_traveled=np.empty(capacity, dtype=Length),
            ID=np.empty(capacity, dtype=ID),
            is_active=np.zeros(capacity, dtype=np.bool_)
        )
        validate_particle_state(self.state)

    def inject_particles(
        self,
        species: NDArray[Species],
        position: tuple[NDArray[Float], NDArray[Float], NDArray[Float]],
        direction: tuple[NDArray[Float], NDArray[Float], NDArray[Float]],
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
        emission_position: tuple[NDArray[Float], NDArray[Float], NDArray[Float]],
        emission_direction: tuple[NDArray[Float], NDArray[Float], NDArray[Float]],
        distance_traveled: NDArray[Length]
    ) -> NDArray[np.int64]:
        """
        Injects new particles into inactive slots in the object pool.
        Returns the indices where the particles were successfully injected.
        """
        num_new = species.shape[0]

        # Find available inactive slots (we use where to get array of indices)
        inactive_indices = np.where(~self.state.is_active)[0]

        if num_new > inactive_indices.shape[0]:
            raise RuntimeError(
                f"Particle pool capacity exceeded. Tried to inject {num_new} "
                f"particles but only {inactive_indices.shape[0]} slots available."
            )

        # Select slots for injection
        target_indices = inactive_indices[:num_new]

        # Generate IDs
        new_ids = np.arange(self.count, self.count + num_new, dtype=ID)
        self.count += num_new

        # Set base arrays in-place
        self.state.is_active[target_indices] = True
        self.state.ID[target_indices] = new_ids
        self.state.species[target_indices] = species
        self.state.energy[target_indices] = energy
        self.state.emission_time[target_indices] = emission_time
        self.state.emission_energy[target_indices] = energy
        self.state.distance_traveled[target_indices] = distance_traveled

        # Set Position
        self.state.position.x[target_indices] = position[0]
        self.state.position.y[target_indices] = position[1]
        self.state.position.z[target_indices] = position[2]

        # Set Direction
        self.state.direction.x[target_indices] = direction[0]
        self.state.direction.y[target_indices] = direction[1]
        self.state.direction.z[target_indices] = direction[2]

        # Set Emission Position
        self.state.emission_position.x[target_indices] = emission_position[0]
        self.state.emission_position.y[target_indices] = emission_position[1]
        self.state.emission_position.z[target_indices] = emission_position[2]

        # Set Emission Direction
        self.state.emission_direction.x[target_indices] = emission_direction[0]
        self.state.emission_direction.y[target_indices] = emission_direction[1]
        self.state.emission_direction.z[target_indices] = emission_direction[2]

        return target_indices

    def get_active_indices(self) -> NDArray[np.int64]:
        """Returns the indices of currently active particles in the pool."""
        return np.nonzero(self.state.is_active)[0]

    def move(self, target_indices: NDArray[np.int64], distances: NDArray[Float]) -> None:
        """
        Facade for move_kernel, applying distances across target active particles.
        """
        # Imported lazily or globally later
        from core.particles.particles_soa_kernels import move_kernel
        move_kernel(self.state, target_indices, distances)

    def rotate(self, target_indices: NDArray[np.int64], thetas: NDArray[Float], phis: NDArray[Float]) -> None:
        """
        Facade for rotate_kernel, applying thetas and phis across target active particles.
        """
        from core.particles.particles_soa_kernels import rotate_kernel
        rotate_kernel(self.state, target_indices, thetas, phis)
