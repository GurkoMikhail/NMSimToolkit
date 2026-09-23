from typing import Any, Dict, List, Optional, Set
import numpy as np

from PySide6.QtCore import QObject, Signal

from core.scene.nodes import SpatialNode, CompositeNode
from gui.viewmodels.node_viewmodel import NodeViewModel, VolumeViewModel, create_node_viewmodel
from core.config.models import (
    SimulationConfig,
    SensitiveVolumeHandlerConfig,
    HistoryAssemblerHandlerConfig,
)


class SceneViewModel(QObject):
    """
    Главная модель представления сцены (MVVM).
    Управляет иерархией NodeViewModel, выбранным узлом и транслирует изменения
    между UI, инспектором свойств и чистым графом сцены ядра.
    """

    scene_loaded = Signal(object)
    node_selected = Signal(object)
    node_added = Signal(object)
    node_removed = Signal(object)

    def __init__(self, root_core_node: Optional[SpatialNode] = None) -> None:
        super().__init__()
        self.root_vm: Optional[NodeViewModel] = None
        self.selected_node: Optional[NodeViewModel] = None
        self._node_map: Dict[int, NodeViewModel] = {}

        if root_core_node is not None:
            self.load_scene(root_core_node)

    def load_scene(self, root_core_node: SpatialNode) -> NodeViewModel:
        """
        Загружает граф сцены из ядра и строит иерархию ViewModel.
        """
        VolumeViewModel.clear_sensitive_volumes()
        self._node_map.clear()
        self.root_vm = create_node_viewmodel(root_core_node)
        self._register_node_recursive(self.root_vm)
        self.select_node(self.root_vm)
        self.scene_loaded.emit(self.root_vm)
        return self.root_vm

    def apply_simulation_config(self, config: SimulationConfig) -> None:
        """
        Применяет параметры конфигурации SimulationConfig к иерархии сцены
        (в частности, синхронизирует статус чувствительных детекторов из data_manager).
        """
        if config.data_manager is not None:
            sensitive_names: Set[str] = set()
            for handler in config.data_manager.handlers:
                if isinstance(handler, (SensitiveVolumeHandlerConfig, HistoryAssemblerHandlerConfig)):
                    sensitive_names.update(handler.sensitive_volumes)
            if sensitive_names:
                for node_vm in self.all_nodes():
                    if isinstance(node_vm, VolumeViewModel) and node_vm.name in sensitive_names:
                        node_vm.is_sensitive_detector = True

    def _register_node_recursive(self, vm: NodeViewModel) -> None:
        self._node_map[id(vm.core_node)] = vm
        for child in vm.children:
            self._register_node_recursive(child)

    def select_node(self, vm: Optional[NodeViewModel]) -> None:
        """
        Выбор активного узла в сцене для инспекции и манипуляций.
        """
        if self.selected_node is not vm:
            self.selected_node = vm
            self.node_selected.emit(vm)

    def add_node(self, parent_vm: NodeViewModel, new_vm: NodeViewModel) -> None:
        """
        Добавляет новый узел в иерархию ViewModel и соответствующий узел ядра.
        """
        parent_vm.add_child(new_vm)
        self._register_node_recursive(new_vm)
        self.selected_node = new_vm
        self.node_added.emit(new_vm)
        self.node_selected.emit(new_vm)

    def remove_node(self, vm: NodeViewModel) -> None:
        """
        Удаляет узел из иерархии сцены.
        """
        if vm.parent_vm is not None:
            parent = vm.parent_vm
            parent.remove_child(vm)
            self._unregister_node_recursive(vm)
            if self.selected_node is vm:
                self.select_node(parent)
            self.node_removed.emit(vm)

    def move_node(self, node_vm: NodeViewModel, new_parent_vm: NodeViewModel, new_index: Optional[int] = None) -> bool:
        """
        Перемещает узел в нового родителя с валидацией циклических зависимостей.
        """
        if node_vm is self.root_vm or node_vm is new_parent_vm:
            return False

        # Проверка на циклы: новый родитель не должен быть потомком перемещаемого узла
        curr: Optional[NodeViewModel] = new_parent_vm
        while curr is not None:
            if curr is node_vm:
                return False
            curr = curr.parent_vm

        try:
            new_parent_vm.add_child(node_vm)
            self._register_node_recursive(node_vm)
            self.selected_node = node_vm
            self.node_added.emit(node_vm)
            self.node_selected.emit(node_vm)
            return True
        except Exception:
            return False

    def _unregister_node_recursive(self, vm: NodeViewModel) -> None:
        core_id = id(vm.core_node)
        if core_id in self._node_map:
            del self._node_map[core_id]
        for child in vm.children:
            self._unregister_node_recursive(child)

    def find_by_name(self, name: str) -> Optional[NodeViewModel]:
        """
        Поиск узла по имени во всей иерархии сцены.
        """
        if self.root_vm is None:
            return None

        stack = [self.root_vm]
        while stack:
            curr = stack.pop()
            if curr.name == name:
                return curr
            stack.extend(curr.children)
        return None

    def find_by_core_node(self, core_node: SpatialNode) -> Optional[NodeViewModel]:
        """
        Быстрый поиск ViewModel по ссылке на объект ядра.
        """
        return self._node_map.get(id(core_node))

    def all_nodes(self) -> List[NodeViewModel]:
        """
        Возвращает плоский список всех узлов сцены.
        """
        if self.root_vm is None:
            return []
        nodes = []
        stack = [self.root_vm]
        while stack:
            curr = stack.pop()
            nodes.append(curr)
            stack.extend(curr.children)
        return nodes
