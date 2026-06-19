from typing import List

from core.scene.nodes import CompositeNode
from core.source.sources import Source

class SourceCompiler:
    """
    Compiler responsible for extracting a flat list of Sources from a Unified Scene Graph.
    """
    def __init__(self):
        self.active_sources: List[Source] = []

    def compile_scene(self, root_node: CompositeNode) -> List[Source]:
        """
        Traverses the graph and extracts all active Sources.
        Extracts only Leaf Sources (sources that don't have other Sources as children).
        """
        self.active_sources = []
        self._extract_sources(root_node)
        return self.active_sources

    def _extract_sources(self, node: CompositeNode) -> bool:
        """
        Returns True if the current node is a Source and is a leaf in terms of sources.
        """
        has_source_children = False

        for child in node.childs:
            is_child_source = self._extract_sources(child)
            if is_child_source:
                has_source_children = True

        if isinstance(node, Source):
            if not has_source_children:
                self.active_sources.append(node)
            return True

        return has_source_children
