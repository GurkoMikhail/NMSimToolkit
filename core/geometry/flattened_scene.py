import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.scene.nodes import CompositeNode
import core.geometry.volumes as volumes_mod


class FlattenedScene:
    """
    Инкапсулирует обход графа сцены в глубину (DFS).
    Гарантирует идентичный порядок обработки объемов в GeometryCompiler и PhysicsCompiler.
    """

    def __init__(self, root_node: CompositeNode):
        self._flat_list: List[Tuple['volumes_mod.Volume', NDArray[Float], Index]] = []
        self._flatten_scene_graph(root_node)

    @property
    def flat_list(self) -> List[Tuple['volumes_mod.Volume', NDArray[Float], Index]]:
        """
        Возвращает плоский список кортежей:
        (Volume, общая_матрица_трансформации, parent_index)
        """
        return self._flat_list

    def _flatten_scene_graph(self, root_node: CompositeNode) -> None:
        def dfs(node: CompositeNode, parent_index: Index) -> Index:
            child_count = 0
            current_index = parent_index

            # We only add Volumes to the geometry buffer flat_list
            if isinstance(node, volumes_mod.Volume):
                current_index = len(self._flat_list)
                self._flat_list.append((node, node.inverse_global_matrix, parent_index))

            if isinstance(node, CompositeNode):
                for child_node in node.childs:
                    # Traverse down, passing the current_index to link deeper Volumes to the closest Volume ancestor
                    child_count += dfs(child_node, current_index)

            return child_count + (1 if isinstance(node, volumes_mod.Volume) else 0)

        dfs(root_node, -1)
