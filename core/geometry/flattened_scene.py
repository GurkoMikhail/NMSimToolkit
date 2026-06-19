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

    def __init__(self, root_node: 'CompositeNode'):
        self._flat_list: List[Tuple['Volume', NDArray[Float], Index]] = []
        self._flatten_scene_graph(root_node)

    @property
    def flat_list(self) -> List[Tuple['Volume', NDArray[Float], Index]]:
        """
        Returns a flattened list where each element is a tuple:
        (Volume, total_transformation_matrix, parent_index)
        """
        return self._flat_list

    def _flatten_scene_graph(self, root_node: 'CompositeNode') -> None:
        from core.geometry.volumes import Volume
        from core.scene.nodes import CompositeNode

        def dfs(node: 'CompositeNode', parent_index: Index) -> Index:
            child_count = 0
            current_index = parent_index

            # We only add Volumes to the geometry buffer flat_list
            if isinstance(node, Volume):
                current_index = len(self._flat_list)
                self._flat_list.append((node, node.inverse_global_matrix, parent_index))

            for child in node.childs:
                # Traverse down, passing the current_index to link deeper Volumes to the closest Volume ancestor
                child_count += dfs(child, current_index)

            return child_count + (1 if isinstance(node, Volume) else 0)

        dfs(root_node, -1)
