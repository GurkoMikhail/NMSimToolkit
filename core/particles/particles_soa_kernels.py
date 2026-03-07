import numpy as np
from numba import njit
from numpy.typing import NDArray
from core.other.typing_definitions import Float, Length
from core.particles.particles_soa import ParticleState

@njit(cache=True)
def move_kernel(
    state: ParticleState,
    target_indices: NDArray[np.int64],
    distances: NDArray[Length]
) -> None:
    """
    In-place moves particles based on their directions and the given distances.
    Only processes particles at the `target_indices`.
    """
    for i in range(len(target_indices)):
        idx = target_indices[i]
        d = distances[i]

        state.distance_traveled[idx] += d
        state.position.x[idx] += state.direction.x[idx] * d
        state.position.y[idx] += state.direction.y[idx] * d
        state.position.z[idx] += state.direction.z[idx] * d

@njit(cache=True)
def rotate_kernel(
    state: ParticleState,
    target_indices: NDArray[np.int64],
    thetas: NDArray[Float],
    phis: NDArray[Float]
) -> None:
    """
    In-place rotates directions of particles by theta and phi.
    Only processes particles at the `target_indices`.
    """
    for i in range(len(target_indices)):
        idx = target_indices[i]
        theta = thetas[i]
        phi = phis[i]

        dir_x = state.direction.x[idx]
        dir_y = state.direction.y[idx]
        dir_z = state.direction.z[idx]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        delta1 = sin_theta * np.cos(phi)
        delta2 = sin_theta * np.sin(phi)

        delta = 1.0
        if dir_z < 0:
            delta = -1.0

        b = dir_x * delta1 + dir_y * delta2
        tmp = cos_theta - b / (1.0 + np.abs(dir_z))

        state.direction.x[idx] = dir_x * tmp + delta1
        state.direction.y[idx] = dir_y * tmp + delta2
        state.direction.z[idx] = dir_z * cos_theta - delta * b
