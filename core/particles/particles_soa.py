import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species, Index
from core.other.vectors_soa import Vector3DSoA
from core.geometry.navigation_state import NavigationState
from core.particles.emission_state import EmissionState


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

    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

    # Object Pool lifecycle flag
    is_active: NDArray[np.bool_]

    @property
    def capacity(self) -> int:
        return self.is_active.shape[0]

    def validate(self) -> None:
        """
        Validates that all arrays within the ParticleState have
        matching capacities and are 1-dimensional.
        """
        self.position.validate()
        self.direction.validate()

        arrays = [
            self.species,
            self.energy,
            self.distance_traveled,
            self.ID,
            self.is_active
        ]

        # All base fields should be 1-dimensional
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("All arrays in ParticleState must be 1-dimensional.")

        # Validate lengths match the pool capacity
        for arr in arrays:
            if arr.shape[0] != self.capacity:
                raise ValueError("All arrays in ParticleState must have the same length (capacity).")

        # Validate vector lengths against capacity
        if self.position.x.shape[0] != self.capacity:
            raise ValueError("Vector components in ParticleState must have the same length as the base arrays.")

    @classmethod
    def allocate(cls, capacity: int) -> 'ParticleState':
        """
        Allocates an empty ParticleState with the specified capacity.
        """
        buffer = cls(
            species=np.empty(capacity, dtype=Species),
            position=Vector3DSoA.allocate(capacity, dtype=Length),
            direction=Vector3DSoA.allocate(capacity, dtype=Float),
            energy=np.empty(capacity, dtype=Energy),
            distance_traveled=np.empty(capacity, dtype=Length),
            ID=np.empty(capacity, dtype=ID),
            is_active=np.zeros(capacity, dtype=np.bool_)
        )
        buffer.validate()
        return buffer


class ParticleBank(NamedTuple):
    """
    Facade for managing the object pool of SoA-based particles.
    Separates OOP lifecycle management from Numba computational kernels.
    """
    state: ParticleState
    emission_state: EmissionState
    navigation_state: NavigationState
    count_array: NDArray[Index]
    capacity: int

    @classmethod
    def allocate(cls, capacity: int) -> 'ParticleBank':
        """
        Allocates a complete ParticleBank Object Pool with its internal arrays.
        """
        state = ParticleState.allocate(capacity)
        emission_state = EmissionState.allocate(capacity)
        navigation_state = NavigationState.allocate(capacity)
        count_array = np.zeros(1, dtype=Index)
        return cls(
            state=state,
            emission_state=emission_state,
            navigation_state=navigation_state,
            count_array=count_array,
            capacity=capacity
        )

    @property
    def count(self) -> int:
        return int(self.count_array[0])

    def inject_particles(
        self,
        species: NDArray[Species],
        position: Vector3DSoA,
        direction: Vector3DSoA,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
        distance_traveled: NDArray[Length]
    ) -> NDArray[Index]:
        """
        Injects new particles into inactive slots in the object pool.
        Returns the indices where the particles were successfully injected.
        Sets emission data automatically based on input state.
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
        current_count = self.count_array[0]
        new_ids = np.arange(current_count, current_count + num_new, dtype=ID)
        self.count_array[0] += num_new

        # Set base arrays in-place
        self.state.is_active[target_indices] = True
        self.state.ID[target_indices] = new_ids
        self.state.species[target_indices] = species
        self.state.energy[target_indices] = energy
        self.emission_state.emission_time[target_indices] = emission_time
        self.emission_state.emission_energy[target_indices] = energy
        self.state.distance_traveled[target_indices] = distance_traveled

        # Set Position
        self.state.position.x[target_indices] = position.x
        self.state.position.y[target_indices] = position.y
        self.state.position.z[target_indices] = position.z

        # Set Direction
        self.state.direction.x[target_indices] = direction.x
        self.state.direction.y[target_indices] = direction.y
        self.state.direction.z[target_indices] = direction.z

        # Set Emission Position
        self.emission_state.emission_position.x[target_indices] = position.x
        self.emission_state.emission_position.y[target_indices] = position.y
        self.emission_state.emission_position.z[target_indices] = position.z

        # Set Emission Direction
        self.emission_state.emission_direction.x[target_indices] = direction.x
        self.emission_state.emission_direction.y[target_indices] = direction.y
        self.emission_state.emission_direction.z[target_indices] = direction.z

        # Invalidate navigation state for reused slots
        import core.particles.particles_soa_kernels as kernel
        kernel.update_navigation_state_inject_kernel(self.navigation_state, target_indices)

        return target_indices

    @property
    def active_indices(self) -> NDArray[Index]:
        """Returns the indices of currently active particles in the pool."""
        return np.nonzero(self.state.is_active)[0]

    def move(self, target_indices: NDArray[Index], distances: NDArray[Float]) -> None:
        """
        Facade for move_kernel, applying distances across target active particles.
        """
        import core.particles.particles_soa_kernels as kernel
        kernel.move_kernel(self.state, target_indices, distances)
        kernel.update_navigation_state_move_kernel(self.navigation_state, target_indices, distances)

    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None:
        """
        Facade for rotate_kernel, applying thetas and phis across target active particles.
        """
        import core.particles.particles_soa_kernels as kernel
        kernel.rotate_kernel(self.state, target_indices, thetas, phis)
        kernel.update_navigation_state_rotate_kernel(self.navigation_state, target_indices)
