import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.geometry.geometry_buffer_soa import GeometryBuffer
from core.particles.particles_soa import ParticleState

def cast_path_kernel(
    geom_buffer: GeometryBuffer,
    state: ParticleState,
    target_indices: NDArray[Index],
    out_distance: NDArray[Float],
    out_material_id: NDArray[np.uint32],
    distance_epsilon: Float = 1e-3
) -> None: ...
