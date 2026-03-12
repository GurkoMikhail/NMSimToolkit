import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index

if TYPE_CHECKING:
    from core.geometry.volumes import Volume


class FlattenedScene:
    """
    Encapsulates the Depth-First Search (DFS) traversal of the OOP Scene Graph.
    Ensures that both GeometryCompiler and PhysicsCompiler process volumes in the exact same order.
    """

    def __init__(self, root_volume: 'Volume'):
        self._flat_list: List[Tuple['Volume', NDArray[Float], Index]] = []
        self._flatten_scene_graph(root_volume)

    @property
    def flat_list(self) -> List[Tuple['Volume', NDArray[Float], Index]]:
        """
        Returns a flattened list where each element is a tuple:
        (Volume, total_transformation_matrix, parent_index)
        """
        return self._flat_list

    def _flatten_scene_graph(self, root_volume: 'Volume') -> None:
        from core.geometry.volumes import TransformableVolume, VolumeWithChilds

        def dfs(volume: 'Volume', parent_matrix: NDArray[Float], parent_index: Index) -> Index:
            if isinstance(volume, TransformableVolume):
                total_matrix = volume.total_transformation_matrix
            else:
                total_matrix = parent_matrix

            current_index = len(self._flat_list)
            self._flat_list.append((volume, total_matrix, parent_index))

            child_count = 0
            if isinstance(volume, VolumeWithChilds):
                for child in volume.childs:
                    child_count += dfs(child, total_matrix, current_index)

            return child_count + 1

        identity_matrix = np.eye(4, dtype=Float)
        dfs(root_volume, identity_matrix, -1)
