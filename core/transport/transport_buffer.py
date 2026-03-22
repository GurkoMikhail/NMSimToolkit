import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Index

class TransportBuffer(NamedTuple):
    """
    SoA buffer for outputting transport logic results per particle.
    """
    process_ids: NDArray[Index]
    material_ids: NDArray[Index]

    def validate(self) -> None:
        if self.process_ids.ndim != 1 or self.material_ids.ndim != 1:
            raise ValueError("All arrays in TransportBuffer must be 1-dimensional.")
        if self.process_ids.shape[0] != self.material_ids.shape[0]:
            raise ValueError("All arrays in TransportBuffer must have the same length.")

    @classmethod
    def allocate(cls, capacity: int) -> 'TransportBuffer':
        buffer = cls(
            process_ids=np.empty(capacity, dtype=Index),
            material_ids=np.empty(capacity, dtype=Index)
        )
        buffer.validate()
        return buffer
