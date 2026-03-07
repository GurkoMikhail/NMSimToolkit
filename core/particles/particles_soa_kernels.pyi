import numpy as np
from numpy.typing import NDArray
from core.other.typing_definitions import Float, Length
from core.particles.particles_soa import ParticleState

def move_kernel(
    state: ParticleState,
    target_indices: NDArray[np.int64],
    distances: NDArray[Length]
) -> None: ...

def rotate_kernel(
    state: ParticleState,
    target_indices: NDArray[np.int64],
    thetas: NDArray[Float],
    phis: NDArray[Float]
) -> None: ...
