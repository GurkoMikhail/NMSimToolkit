import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.particles.kinematic_state import KinematicState
from core.geometry.navigation_state import NavigationState


@njit(inline='always')
def _move_particle(state: KinematicState, p_idx: Index, distance: Float) -> None:
    """
    In-place inline kernel that updates distance_traveled and position vectors
    for a single particle.
    """
    state.distance_traveled[p_idx] += distance
    state.position.x[p_idx] += state.direction.x[p_idx] * distance
    state.position.y[p_idx] += state.direction.y[p_idx] * distance
    state.position.z[p_idx] += state.direction.z[p_idx] * distance

@njit(inline='always')
def _rotate_particle(state: KinematicState, p_idx: Index, theta: Float, phi: Float) -> None:
    """
    In-place inline kernel that applies theta and phi rotations
    to the direction vector of a single particle.
    """
    dir_x = state.direction.x[p_idx]
    dir_y = state.direction.y[p_idx]
    dir_z = state.direction.z[p_idx]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    delta1 = sin_theta * np.cos(phi)
    delta2 = sin_theta * np.sin(phi)

    delta = 1.0
    if dir_z < 0.0:
        delta = -1.0

    b = dir_x * delta1 + dir_y * delta2
    abs_z = np.abs(dir_z)
    tmp = cos_theta - b / (1.0 + abs_z)

    state.direction.x[p_idx] = dir_x * tmp + delta1
    state.direction.y[p_idx] = dir_y * tmp + delta2
    state.direction.z[p_idx] = dir_z * cos_theta - delta * b


@njit(cache=True)
def move_kernel(state: KinematicState, target_indices: NDArray[np.int64], distances: NDArray[Float]) -> None:
    """
    In-place kernel that updates distance_traveled and position vectors
    for specific target active particles.
    """
    for j in range(len(target_indices)):
        i = target_indices[j]
        _move_particle(state, i, distances[j])


@njit(inline='always')
def _invalidate_navigation_state(nav_state: NavigationState, p_idx: Index) -> None:
    """
    In-place inline kernel that invalidates the navigation state for a single particle,
    forcing a fresh geometry search on the next step.
    """
    nav_state.current_volume[p_idx] = -1
    nav_state.boundary_distance[p_idx] = 0.0

@njit(cache=True)
def update_navigation_state_inject_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index]
) -> None:
    for i in range(target_indices.shape[0]):
        p_idx = target_indices[i]
        _invalidate_navigation_state(nav_state, p_idx)

@njit(cache=True)
def update_navigation_state_rotate_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index]
) -> None:
    for i in range(target_indices.shape[0]):
        p_idx = target_indices[i]
        nav_state.boundary_distance[p_idx] = 0.0

@njit(cache=True)
def update_navigation_state_move_kernel(
    nav_state: NavigationState,
    target_indices: NDArray[Index],
    distances: NDArray[Float]
) -> None:
    for i in range(target_indices.shape[0]):
        p_idx = target_indices[i]
        nav_state.boundary_distance[p_idx] -= distances[i]

        if nav_state.boundary_distance[p_idx] <= 0:
            _invalidate_navigation_state(nav_state, p_idx)

@njit(cache=True)
def rotate_kernel(
    state: KinematicState,
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
        _rotate_particle(state, i, thetas[j], phis[j])
