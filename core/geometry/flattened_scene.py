"""
Модуль линеаризации графа сцены (FlattenedScene) посредством обхода в глубину (DFS).
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

import core.geometry.volumes as volumes_mod
from core.other.typing_definitions import Float, Index
from core.scene.nodes import CompositeNode, SpatialNode


class FlattenedScene:
    """
    Инкапсулирует обход графа сцены в глубину (DFS).
    Гарантирует идентичный порядок обработки объемов в GeometryCompiler и PhysicsCompiler.
    """

    def __init__(self, root_node: CompositeNode) -> None:
        self._flat_list: List[Tuple[volumes_mod.Volume, NDArray[Float], Index]] = []
        self._flatten_scene_graph(root_node)

    @property
    def flat_list(self) -> List[Tuple[volumes_mod.Volume, NDArray[Float], Index]]:
        """
        Возвращает плоский список кортежей:
        (Volume, обратная_глобальная_матрица_трансформации, parent_index).
        """
        return self._flat_list

    def _flatten_scene_graph(self, root_node: CompositeNode) -> None:
        def dfs(node: SpatialNode, parent_vol_idx: int) -> int:
            child_count = 0
            current_index = parent_vol_idx

            if isinstance(node, volumes_mod.Volume):
                current_index = len(self._flat_list)
                self._flat_list.append((node, node.inverse_global_matrix, Index(parent_vol_idx)))

            if isinstance(node, CompositeNode):
                for child_node in node.childs:
                    child_count += dfs(child_node, current_index)

            return child_count + (1 if isinstance(node, volumes_mod.Volume) else 0)

        dfs(root_node, -1)


__all__ = ["FlattenedScene"]

