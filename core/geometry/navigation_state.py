import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index


class NavigationState(NamedTuple):
    """
    Structure of Arrays (SoA) to cache geometry navigation state.
    Used for Stateful Navigation, Relocation, and Ray Distance Caching in Woodcock tracking.
    """
    current_volume: NDArray[Index]
    next_volume: NDArray[Index]
    boundary_distance: NDArray[Float]

    def validate(self) -> None:
        arrays = [
            self.current_volume,
            self.next_volume,
            self.boundary_distance
        ]
        for arr in arrays:
            if arr.ndim != 1:
                raise ValueError("NavigationState arrays must be 1-dimensional.")
        length = arrays[0].shape[0]
        for arr in arrays:
            if arr.shape[0] != length:
                raise ValueError("NavigationState arrays must have the same length.")

    @property
    def capacity(self) -> int:
        return self.current_volume.shape[0]
