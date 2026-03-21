from typing import NamedTuple
from numpy.typing import NDArray
from core.other.typing_definitions import Index

class TransportBuffer(NamedTuple):
    """
    SoA buffer for outputting transport logic results per particle.
    """
    process_ids: NDArray[Index]
    material_ids: NDArray[Index]
