import numpy as np
from typing import NamedTuple, Any
from numpy.typing import NDArray

from core.other.typing_definitions import Energy, Float, ID, Length, Time, Species, Index
from core.particles.vector3d_soa import Vector3DSoA

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

    def _validate(self) -> None:
        sizes = [len(getattr(self, field)) for field in self._fields if field not in ('position', 'direction', 'emission_position', 'emission_direction')]

        for v3d in ('position', 'direction', 'emission_position', 'emission_direction'):
            getattr(self, v3d)._validate()
            sizes.append(getattr(self, v3d).x.size)

        if len(set(sizes)) > 1:
            raise ValueError(f"All arrays in ParticleState must have identical sizes. Got sizes: {sizes}")

class ParticleBank:
    def __init__(self, capacity: int) -> None:
        self._capacity: int = capacity
        self._count: int = 0
        self._state: ParticleState = self._allocate_pool(capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return self._count

    @property
    def state(self) -> ParticleState:
        return self._state

    def _allocate_pool(self, capacity: int) -> ParticleState:
        state = ParticleState(
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
        state._validate()
        return state

    def move(self, target_indices: NDArray[Index], distances: NDArray[Length]) -> None:
        from core.particles.particles_soa_kernels import move_kernel
        move_kernel(self.state, target_indices, distances)

    def rotate(self, target_indices: NDArray[Index], thetas: NDArray[Float], phis: NDArray[Float]) -> None:
        from core.particles.particles_soa_kernels import rotate_kernel
        rotate_kernel(self.state, target_indices, thetas, phis)

    def inject(
        self,
        species: NDArray[Species],
        position: Vector3DSoA,
        direction: Vector3DSoA,
        energy: NDArray[Energy],
        emission_time: NDArray[Time],
    ) -> NDArray[Index]:
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

        self.state.position.x[target_indices] = position.x
        self.state.position.y[target_indices] = position.y
        self.state.position.z[target_indices] = position.z

        self.state.direction.x[target_indices] = direction.x
        self.state.direction.y[target_indices] = direction.y
        self.state.direction.z[target_indices] = direction.z

        self.state.energy[target_indices] = energy
        self.state.emission_time[target_indices] = emission_time
        self.state.emission_energy[target_indices] = energy

        self.state.emission_position.x[target_indices] = position.x
        self.state.emission_position.y[target_indices] = position.y
        self.state.emission_position.z[target_indices] = position.z

        self.state.emission_direction.x[target_indices] = direction.x
        self.state.emission_direction.y[target_indices] = direction.y
        self.state.emission_direction.z[target_indices] = direction.z

        self.state.distance_traveled[target_indices] = 0.0

        new_ids = np.arange(self._count, self._count + n_inject, dtype=ID)
        self.state.ID[target_indices] = new_ids
        self._count += n_inject

        return target_indices
