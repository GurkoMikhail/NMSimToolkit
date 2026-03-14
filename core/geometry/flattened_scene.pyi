from typing import List, Tuple
from numpy.typing import NDArray

from core.geometry.volumes import Volume
from core.other.typing_definitions import Float, Index

class FlattenedScene:
    _flat_list: List[Tuple[Volume, NDArray[Float], Index]]
    def __init__(self, root_volume: Volume) -> None: ...
    @property
    def flat_list(self) -> List[Tuple[Volume, NDArray[Float], Index]]: ...
    def _flatten_scene_graph(self, root_volume: Volume) -> None: ...
