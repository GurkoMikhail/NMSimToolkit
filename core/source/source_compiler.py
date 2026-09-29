from typing import List

from core.scene.nodes import CompositeNode
from core.source.sources import Source

class SourceCompiler:
    """
    Компилятор, отвечающий за извлечение плоского списка источников частиц из графа сцены.
    """
    def __init__(self):
        self.active_sources: List[Source] = []

    def compile_scene(self, root_node: CompositeNode) -> List[Source]:
        """
        Обходит граф сцены и извлекает все активные источники излучения.
        Извлекает только листовые источники (не имеющие других источников в качестве потомков).
        """
        self.active_sources = []
        self._extract_sources(root_node)
        return self.active_sources

    def _extract_sources(self, node: CompositeNode) -> bool:
        """
        Возвращает True, если текущий узел является источником и листом среди источников.
        """
        has_source_children = False

        if isinstance(node, CompositeNode):
            for child_node in node.childs:
                is_child_source = self._extract_sources(child_node)
                if is_child_source:
                    has_source_children = True

        if isinstance(node, Source):
            if not has_source_children:
                self.active_sources.append(node)
            return True

        return has_source_children
