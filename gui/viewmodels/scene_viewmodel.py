from typing import Any, Dict, List, Optional, Sequence, Set, Union
import numpy as np

from PySide6.QtCore import QObject, Signal

from core.geometry.volumes import Volume
from core.scene.nodes import SpatialNode, CompositeNode
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from core.config.models import (
    SimulationConfig,
    SensitiveVolumeHandlerConfig,
    HistoryAssemblerHandlerConfig,
)


class SceneViewModel(QObject):
    """
    Главная модель представления сцены (MVVM).
    Управляет иерархией NodeViewModel, выбранным узлом, реестром чувствительных объемов
    и транслирует изменения между UI, инспектором свойств и чистым графом сцены ядра.
    """

    scene_loaded = Signal(object)
    node_selected = Signal(object)
    node_added = Signal(object)
    node_removed = Signal(object)
    sensitive_volumes_changed = Signal()

    def __init__(self, root_core_node: Optional[SpatialNode] = None) -> None:
        super().__init__()
        self.root_vm: Optional[NodeViewModel] = None
        self.selected_node: Optional[NodeViewModel] = None
        self._node_map: Dict[int, NodeViewModel] = {}
        self.distribution_registry: Dict[Any, Any] = {}
        self.slots_registry: Dict[Any, Any] = {}
        self._sensitive_volumes: List[str] = []

        if root_core_node is not None:
            self.load_scene(root_core_node)

    def load_scene(
        self,
        root_core_node: SpatialNode,
        distribution_registry: Optional[Dict[Any, Any]] = None,
        slots_registry: Optional[Dict[Any, Any]] = None,
    ) -> NodeViewModel:
        """
        Загружает граф сцены из ядра и строит иерархию ViewModel.
        """
        self.clear_sensitive_volumes()
        self._node_map.clear()
        self.distribution_registry = dict(distribution_registry) if distribution_registry is not None else {}
        self.slots_registry = dict(slots_registry) if slots_registry is not None else {}
        self.root_vm = create_node_viewmodel(root_core_node)
        self._register_node_recursive(self.root_vm)
        self.select_node(self.root_vm)
        self.scene_loaded.emit(self.root_vm)
        return self.root_vm

    @property
    def sensitive_volumes(self) -> List[str]:
        """
        Возвращает упорядоченный список уникальных имен чувствительных объемов сцены.
        """
        return list(self._sensitive_volumes)

    @sensitive_volumes.setter
    def sensitive_volumes(self, names: Sequence[str]) -> None:
        """
        Устанавливает единый список имен чувствительных объемов сцены.
        """
        cleaned_names: List[str] = []
        for name in names:
            name_str = str(name).strip()
            if name_str and name_str not in cleaned_names:
                cleaned_names.append(name_str)
        if self._sensitive_volumes != cleaned_names:
            self._sensitive_volumes = cleaned_names
            self.sensitive_volumes_changed.emit()

    def is_sensitive_volume(self, node_or_name: Union[NodeViewModel, Volume, str]) -> bool:
        """
        Проверяет, зарегистрирован ли узел или имя объема в едином списке чувствительных объемов.
        """
        if isinstance(node_or_name, str):
            volume_name = node_or_name
        elif isinstance(node_or_name, NodeViewModel):
            volume_name = node_or_name.name
        elif isinstance(node_or_name, Volume):
            volume_name = node_or_name.name
        else:
            return False
        return volume_name in self._sensitive_volumes

    def set_volume_sensitive(self, node_or_name: Union[NodeViewModel, Volume, str], is_sensitive: bool) -> None:
        """
        Добавляет или исключает объем из единого списка чувствительных детекторов сцены.
        """
        if isinstance(node_or_name, str):
            volume_name = node_or_name
        elif isinstance(node_or_name, (NodeViewModel, Volume)):
            volume_name = node_or_name.name
        else:
            return

        volume_name = volume_name.strip()
        if not volume_name:
            return

        is_present = volume_name in self._sensitive_volumes
        if is_sensitive and not is_present:
            self._sensitive_volumes.append(volume_name)
            self.sensitive_volumes_changed.emit()
        elif not is_sensitive and is_present:
            self._sensitive_volumes.remove(volume_name)
            self.sensitive_volumes_changed.emit()

    def add_sensitive_volume(self, node_or_name: Union[NodeViewModel, Volume, str]) -> None:
        """Добавляет объем в единый список чувствительных детекторов сцены."""
        self.set_volume_sensitive(node_or_name, True)

    def remove_sensitive_volume(self, node_or_name: Union[NodeViewModel, Volume, str]) -> None:
        """Исключает объем из единого списка чувствительных детекторов сцены."""
        self.set_volume_sensitive(node_or_name, False)

    def clear_sensitive_volumes(self) -> None:
        """Полная очистка единого списка чувствительных детекторов сцены."""
        if self._sensitive_volumes:
            self._sensitive_volumes.clear()
            self.sensitive_volumes_changed.emit()

    def apply_simulation_config(self, config: SimulationConfig) -> None:
        """
        Применяет параметры конфигурации SimulationConfig к иерархии сцены
        (в частности, синхронизирует единый реестр чувствительных детекторов из data_manager).
        """
        if config.data_manager is not None:
            sensitive_names: List[str] = []
            for handler in config.data_manager.handlers:
                if isinstance(handler, (SensitiveVolumeHandlerConfig, HistoryAssemblerHandlerConfig)):
                    for vol_name in handler.sensitive_volumes:
                        if vol_name not in sensitive_names:
                            sensitive_names.append(vol_name)
            if sensitive_names:
                self.sensitive_volumes = sensitive_names

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

    def replace_node(self, target_vm: NodeViewModel, new_vm: NodeViewModel) -> None:
        """
        Замещает узел target_vm новым узлом new_vm в дереве сцены с сохранением:
        - родительского узла и исходной позиции в иерархии;
        - локальной матрицы трансформации;
        - привязок слотов гамма-камеры (casing, detector_box, collimator, crystal, glass_backend).
        """
        if target_vm is None or new_vm is None:
            raise ValueError("target_vm и new_vm не могут быть None")
        if target_vm is new_vm:
            return

        parent_vm = target_vm.parent_vm
        if parent_vm is None:
            raise ValueError("Замена корневого узла без родителя не поддерживается")

        # 1. Сохранение локальной матрицы трансформации
        new_vm.local_matrix = np.copy(target_vm.local_matrix)

        old_name = target_vm.name
        new_name = new_vm.name
        if old_name in self._sensitive_volumes:
            sens_idx = self._sensitive_volumes.index(old_name)
            self._sensitive_volumes[sens_idx] = new_name
            self.sensitive_volumes_changed.emit()

        for node_vm in self.all_nodes():
            if isinstance(node_vm, GammaCameraViewModel):
                slots_config = node_vm.slots
                if slots_config.casing == old_name:
                    slots_config.casing = new_name
                    node_vm.core_node.slots['casing'] = new_name
                if slots_config.detector_box == old_name:
                    slots_config.detector_box = new_name
                    node_vm.core_node.slots['detector_box'] = new_name
                if slots_config.collimator == old_name:
                    slots_config.collimator = new_name
                    node_vm.core_node.slots['collimator'] = new_name
                if slots_config.crystal == old_name:
                    slots_config.crystal = new_name
                    node_vm.core_node.slots['crystal'] = new_name
                if slots_config.glass_backend == old_name:
                    slots_config.glass_backend = new_name
                    node_vm.core_node.slots['glass_backend'] = new_name

        for camera_core_node, slots_config in self.slots_registry.items():
            if slots_config.casing == old_name:
                slots_config.casing = new_name
            if slots_config.detector_box == old_name:
                slots_config.detector_box = new_name
            if slots_config.collimator == old_name:
                slots_config.collimator = new_name
            if slots_config.crystal == old_name:
                slots_config.crystal = new_name
            if slots_config.glass_backend == old_name:
                slots_config.glass_backend = new_name

        # 3. Определение индексов вставки в родительском узле
        child_index = parent_vm.children.index(target_vm) if target_vm in parent_vm.children else len(parent_vm.children)
        core_index = parent_vm.core_node.childs.index(target_vm.core_node) if (
            isinstance(parent_vm.core_node, CompositeNode) and target_vm.core_node in parent_vm.core_node.childs
        ) else len(parent_vm.core_node.childs)

        # 4. Удаление старого узла target_vm
        parent_vm.remove_child(target_vm)
        self._unregister_node_recursive(target_vm)
        self.node_removed.emit(target_vm)

        # 5. Вставка нового узла new_vm на то же место
        if isinstance(parent_vm.core_node, CompositeNode):
            new_vm.core_node.parent = parent_vm.core_node
            parent_vm.core_node.childs.insert(core_index, new_vm.core_node)
        new_vm.parent_vm = parent_vm
        parent_vm.children.insert(child_index, new_vm)
        self._register_node_recursive(new_vm)

        new_vm._notify_transform_changed()
        parent_vm.child_added.emit(new_vm)
        self.node_added.emit(new_vm)
        self.select_node(new_vm)

        # 6. Если узел находится внутри GammaCameraViewModel, синхронизируем размеры и кинематику
        current_ancestor: Optional[NodeViewModel] = parent_vm
        while current_ancestor is not None:
            if isinstance(current_ancestor, GammaCameraViewModel):
                current_ancestor._apply_fixed_constraints_to_subcomponents()
                current_ancestor._rebuild_geometry()
                break
            current_ancestor = current_ancestor.parent_vm

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
        if vm.name in self._sensitive_volumes:
            self._sensitive_volumes.remove(vm.name)
            self.sensitive_volumes_changed.emit()
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
        Возвращает плоский список всех узлов сцены в прямом порядке обхода (pre-order DFS).
        """
        if self.root_vm is None:
            return []
        nodes = []
        stack = [self.root_vm]
        while stack:
            curr = stack.pop()
            nodes.append(curr)
            stack.extend(reversed(curr.children))
        return nodes

