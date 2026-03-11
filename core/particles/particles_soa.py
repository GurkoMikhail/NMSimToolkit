import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species, Index
from core.other.vectors_soa import Vector3DSoA
from core.geometry.navigation_state import NavigationState


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
        self.emission_position.validate()
        self.emission_direction.validate()

        arrays = [
            self.species,
            self.energy,
            self.emission_time,
            self.emission_energy,
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


class ParticleBank:
    """
    Facade for managing the object pool of SoA-based particles.
    Separates OOP lifecycle management from Numba computational kernels.
    """

    def __init__(self, capacity: int):
        self._count = 0

        # Instantiate base flat arrays
        self._state = ParticleState(
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
        self._state.validate()

        self._navigation_state = NavigationState(
            current_volume=np.full(capacity, -1, dtype=Index),
            next_volume=np.full(capacity, -1, dtype=Index),
            boundary_distance=np.full(capacity, np.inf, dtype=Float)
        )
        self._navigation_state.validate()

    @property
    def capacity(self) -> int:
        return self._state.capacity

    @property
    def count(self) -> int:
        return self._count

    @property
    def state(self) -> ParticleState:
        return self._state

    @property
    def navigation_state(self) -> NavigationState:
        return self._navigation_state

    def inject_particles(
        self,
        species: NDArray[Species],
        position: Vector3DSoA,
        direction: Vector3DSoA,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
        emission_position: Vector3DSoA,
        emission_direction: Vector3DSoA,
        distance_traveled: NDArray[Length]
    ) -> NDArray[Index]:
        """
        Injects new particles into inactive slots in the object pool.
        Returns the indices where the particles were successfully injected.
        """
        num_new = species.shape[0]

        # Find available inactive slots (we use where to get array of indices)
        inactive_indices = np.where(~self._state.is_active)[0]

        if num_new > inactive_indices.shape[0]:
            raise RuntimeError(
                f"Particle pool capacity exceeded. Tried to inject {num_new} "
                f"particles but only {inactive_indices.shape[0]} slots available."
            )

        # Select slots for injection
        target_indices = inactive_indices[:num_new]

        # Generate IDs
        new_ids = np.arange(self._count, self._count + num_new, dtype=ID)
        self._count += num_new

        # Set base arrays in-place
        self._state.is_active[target_indices] = True
        self._state.ID[target_indices] = new_ids
        self._state.species[target_indices] = species
        self._state.energy[target_indices] = energy
        self._state.emission_time[target_indices] = emission_time
        self._state.emission_energy[target_indices] = energy
        self._state.distance_traveled[target_indices] = distance_traveled

        # Set Position
        self._state.position.x[target_indices] = position.x
        self._state.position.y[target_indices] = position.y
        self._state.position.z[target_indices] = position.z

        # Set Direction
        self._state.direction.x[target_indices] = direction.x
        self._state.direction.y[target_indices] = direction.y
        self._state.direction.z[target_indices] = direction.z

        # Set Emission Position
        self._state.emission_position.x[target_indices] = emission_position.x
        self._state.emission_position.y[target_indices] = emission_position.y
        self._state.emission_position.z[target_indices] = emission_position.z

        # Set Emission Direction
        self._state.emission_direction.x[target_indices] = emission_direction.x
        self._state.emission_direction.y[target_indices] = emission_direction.y
        self._state.emission_direction.z[target_indices] = emission_direction.z

        # Invalidate navigation state for reused slots
        import core.particles.particles_soa_kernels as kernel
        kernel.update_navigation_state_inject_kernel(self._navigation_state, target_indices)

        return target_indices

    @property
    def active_indices(self) -> NDArray[Index]:
        """Returns the indices of currently active particles in the pool."""
        return np.nonzero(self._state.is_active)[0]

    def move(self, target_indices: NDArray[Index], distances: NDArray[Float]) -> None:
        """
        Facade for move_kernel, applying distances across target active particles.
        """
        import core.particles.particles_soa_kernels as kernel
        kernel.move_kernel(self._state, target_indices, distances)
        kernel.update_navigation_state_move_kernel(self._navigation_state, target_indices, distances)

    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None:
        """
        Facade for rotate_kernel, applying thetas and phis across target active particles.
        """
        import core.particles.particles_soa_kernels as kernel
        kernel.rotate_kernel(self._state, target_indices, thetas, phis)
        kernel.update_navigation_state_rotate_kernel(self._navigation_state, target_indices)
