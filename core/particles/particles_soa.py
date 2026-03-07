import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species

class Vector3DSoA(NamedTuple):
    x: NDArray[Float]
    y: NDArray[Float]
    z: NDArray[Float]

class ParticleState(NamedTuple):
    is_active: NDArray[np.bool_]
    species: NDArray[Species]

    position: Vector3DSoA
    direction: Vector3DSoA

    energy: NDArray[Energy]
    emission_time: NDArray[Time]
    emission_energy: NDArray[Energy]

    emission_position: Vector3DSoA
    emission_direction: Vector3DSoA

    distance_traveled: NDArray[Length]
    ID: NDArray[ID]

class ParticleBank:
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.count: int = 0
        self.state: ParticleState = self._allocate_pool(capacity)

    def _allocate_pool(self, capacity: int) -> ParticleState:
        return ParticleState(
            is_active=np.zeros(capacity, dtype=np.bool_),
            species=np.empty(capacity, dtype=Species),

            position=Vector3DSoA(
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float)
            ),
            direction=Vector3DSoA(
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float)
            ),

            energy=np.empty(capacity, dtype=Energy),
            emission_time=np.empty(capacity, dtype=Time),
            emission_energy=np.empty(capacity, dtype=Energy),

            emission_position=Vector3DSoA(
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float)
            ),
            emission_direction=Vector3DSoA(
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float),
                np.empty(capacity, dtype=Float)
            ),

            distance_traveled=np.empty(capacity, dtype=Length),
            ID=np.empty(capacity, dtype=ID)
        )

    def move(self, target_indices: NDArray[np.int64], distances: NDArray[Length]) -> None:
        from core.particles.particles_soa_kernels import move_kernel
        move_kernel(self.state, target_indices, distances)

    def rotate(self, target_indices: NDArray[np.int64], thetas: NDArray[Float], phis: NDArray[Float]) -> None:
        from core.particles.particles_soa_kernels import rotate_kernel
        rotate_kernel(self.state, target_indices, thetas, phis)

    def inject(
        self,
        species: NDArray[Species],
        position_x: NDArray[Float],
        position_y: NDArray[Float],
        position_z: NDArray[Float],
        direction_x: NDArray[Float],
        direction_y: NDArray[Float],
        direction_z: NDArray[Float],
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
    ) -> NDArray[np.int64]:
        """
        Injects new particles into inactive slots.
        Returns the indices of the injected particles.
        """
        n_inject = len(species)
        inactive_indices = np.nonzero(~self.state.is_active)[0]

        if len(inactive_indices) < n_inject:
            raise RuntimeError(f"ParticleBank capacity ({self.capacity}) exceeded. Cannot inject {n_inject} particles (only {len(inactive_indices)} slots available).")

        target_indices = inactive_indices[:n_inject]

        self.state.is_active[target_indices] = True
        self.state.species[target_indices] = species

        self.state.position.x[target_indices] = position_x
        self.state.position.y[target_indices] = position_y
        self.state.position.z[target_indices] = position_z

        self.state.direction.x[target_indices] = direction_x
        self.state.direction.y[target_indices] = direction_y
        self.state.direction.z[target_indices] = direction_z

        self.state.energy[target_indices] = energy
        self.state.emission_time[target_indices] = emission_time
        self.state.emission_energy[target_indices] = energy

        self.state.emission_position.x[target_indices] = position_x
        self.state.emission_position.y[target_indices] = position_y
        self.state.emission_position.z[target_indices] = position_z

        self.state.emission_direction.x[target_indices] = direction_x
        self.state.emission_direction.y[target_indices] = direction_y
        self.state.emission_direction.z[target_indices] = direction_z

        self.state.distance_traveled[target_indices] = 0.0

        new_ids = np.arange(self.count, self.count + n_inject, dtype=ID)
        self.state.ID[target_indices] = new_ids
        self.count += n_inject

        return target_indices
