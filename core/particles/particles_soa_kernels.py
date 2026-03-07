import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Float
from core.particles.particles_soa import ParticleState


@njit(cache=True)
def move_kernel(state: ParticleState, target_indices: NDArray[np.int64], distances: NDArray[Float]) -> None:
    """
    In-place kernel that updates distance_traveled and position vectors
    for specific target active particles.
    """
    for j in range(len(target_indices)):
        i = target_indices[j]
        d = distances[j]

        # Update distance traveled
        state.distance_traveled[i] += d

        # Update Position vectors in-place without vector allocation
        state.position.x[i] += state.direction.x[i] * d
        state.position.y[i] += state.direction.y[i] * d
        state.position.z[i] += state.direction.z[i] * d


@njit(cache=True)
def rotate_kernel(
    state: ParticleState,
    target_indices: NDArray[np.int64],
    thetas: NDArray[Float],
    phis: NDArray[Float]
) -> None:
    """
    In-place kernel that applies a sequence of theta and phi rotations
    to the direction vector of specific target particles.
    """
    for j in range(len(target_indices)):
        i = target_indices[j]
        theta = thetas[j]
        phi = phis[j]

        dir_x = state.direction.x[i]
        dir_y = state.direction.y[i]
        dir_z = state.direction.z[i]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        delta1 = sin_theta * np.cos(phi)
        delta2 = sin_theta * np.sin(phi)

        # Delta calculation matching the old logic: np.ones_like(cos_theta) - 2 * (direction[..., 2] < 0)
        delta = 1.0
        if dir_z < 0.0:
            delta = -1.0

        b = dir_x * delta1 + dir_y * delta2

        # In Python: tmp = cos_theta - b / (1 + np.abs(direction[..., 2]))
        abs_z = np.abs(dir_z)
        tmp = cos_theta - b / (1.0 + abs_z)

        # Apply the new rotated components directly to the struct array fields
        state.direction.x[i] = dir_x * tmp + delta1
        state.direction.y[i] = dir_y * tmp + delta2
        state.direction.z[i] = dir_z * cos_theta - delta * b
