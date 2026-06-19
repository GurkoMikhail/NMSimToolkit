import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Index, Float


class NavigationState(NamedTuple):
    current_volume: NDArray[Index]
    boundary_distance: NDArray[Float]

    def validate(self) -> None: ...

    @property
    def capacity(self) -> int: ...

    @classmethod
    def allocate(cls, capacity: int) -> 'NavigationState': ...
