import logging
from typing import Any, Callable, ClassVar, List, Optional, Sequence
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.scene.nodes import SpatialNode, CompositeNode
from gui.viewmodels.decorators import core_field, gui_field

_logger = logging.getLogger(__name__)


class NodeViewModel(QObject):
    """
    Базовая модель представления для узла графа сцены (паттерн MVVM).
    Инкапсулирует SpatialNode или CompositeNode ядра, предоставляя
    Qt-сигналы для синхронизации с 3D-вьюпортом и инспектором свойств.
    """

    property_changed = Signal(str, object)
    child_added = Signal(object)
    child_removed = Signal(object)
    transform_changed = Signal()

    name = core_field('name', default='Node')
    visible = gui_field(default=True)

    _factory: ClassVar[Optional[Callable[[SpatialNode, Optional['NodeViewModel']], 'NodeViewModel']]] = None

    def __init__(self, core_node: SpatialNode, parent_vm: Optional['NodeViewModel'] = None) -> None:
        super().__init__()
        self.core_node = core_node
        self.parent_vm = parent_vm
        self.children: List['NodeViewModel'] = []

        # Инициализация дочерних узлов, если core_node является CompositeNode
        if isinstance(core_node, CompositeNode):
            for child_core in core_node.childs:
                if child_core.parent is not core_node:
                    child_core.parent = core_node
                if NodeViewModel._factory is not None:
                    child_vm = NodeViewModel._factory(child_core, parent_vm=self)
                else:
                    child_vm = NodeViewModel(child_core, parent_vm=self)
                self.children.append(child_vm)

    def _notify_transform_changed(self) -> None:
        """
        Испускает сигнал transform_changed для текущего узла и рекурсивно
        уведомляет всех потомков, так как их эффективная global_matrix изменилась.
        """
        self.transform_changed.emit()
        for child in self.children:
            child._notify_transform_changed()

    @property
    def node_type(self) -> str:
        """
        Человекочитаемый тип узла сцены.
        """
        return self.core_node.__class__.__name__

    @property
    def local_matrix(self) -> np.ndarray:
        return self.core_node.local_matrix

    @local_matrix.setter
    def local_matrix(self, matrix: np.ndarray) -> None:
        self.core_node.local_matrix = np.asarray(matrix, dtype=self.core_node.local_matrix.dtype)
        self.core_node.invalidate_matrix_cache()
        self._notify_transform_changed()
        self.property_changed.emit('local_matrix', self.core_node.local_matrix)

    @property
    def global_matrix(self) -> np.ndarray:
        return self.core_node.global_matrix

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, in_local: bool = False) -> None:
        """
        Перемещение узла с уведомлением подписчиков.
        """
        self.core_node.translate(x=x, y=y, z=z, in_local=in_local)
        self._notify_transform_changed()
        self.property_changed.emit('local_matrix', self.core_node.local_matrix)

    def rotate(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        rotation_center: Sequence[float] = (0.0, 0.0, 0.0),
        in_local: bool = False
    ) -> None:
        """
        Вращение узла с уведомлением подписчиков.
        """
        self.core_node.rotate(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rotation_center=rotation_center,
            in_local=in_local
        )
        self._notify_transform_changed()
        self.property_changed.emit('local_matrix', self.core_node.local_matrix)

    def add_child(self, child_vm: 'NodeViewModel') -> None:
        """
        Добавление дочернего ViewModel и соответствующего узла в ядро.
        """
        if not isinstance(self.core_node, CompositeNode):
            raise TypeError("Cannot add child to a non-composite node")
        if child_vm is self:
            raise ValueError("Cannot add node as a child of itself")

        # Проверка на циклические зависимости
        current_ancestor: Optional['NodeViewModel'] = self
        while current_ancestor is not None:
            if current_ancestor is child_vm:
                raise ValueError("Cannot add an ancestor as a child (cycle detected)")
            current_ancestor = current_ancestor.parent_vm

        # Если узел уже является дочерним для self, повторно не добавляем
        if child_vm.parent_vm is self and child_vm in self.children and child_vm.core_node in self.core_node.childs:
            return

        # Если узел уже имел другого родителя в дереве ViewModel, отсоединяем
        if child_vm.parent_vm is not None and child_vm.parent_vm is not self:
            child_vm.parent_vm.remove_child(child_vm)
        elif child_vm.core_node.parent is not None and child_vm.core_node.parent is not self.core_node:
            if isinstance(child_vm.core_node.parent, CompositeNode):
                child_vm.core_node.parent.remove_child(child_vm.core_node)

        self.core_node.add_child(child_vm.core_node)

        child_vm.parent_vm = self
        if child_vm not in self.children:
            self.children.append(child_vm)
        child_vm._notify_transform_changed()
        self.child_added.emit(child_vm)

    def remove_child(self, child_vm: 'NodeViewModel') -> None:
        """
        Удаление дочернего ViewModel и отсоединение узла из ядра.
        """
        if not isinstance(self.core_node, CompositeNode):
            raise TypeError("Cannot remove child from a non-composite node")
        if child_vm in self.children:
            self.core_node.remove_child(child_vm.core_node)
            child_vm.parent_vm = None
            self.children.remove(child_vm)
            child_vm._notify_transform_changed()
            self.child_removed.emit(child_vm)

    def sync_children_from_core(self) -> None:
        """
        Синхронизация списка children ViewModel со списком core_node.childs
        при изменениях графа со стороны ядра.
        """
        if not isinstance(self.core_node, CompositeNode):
            for child in list(self.children):
                child.parent_vm = None
                if child.core_node.parent is self.core_node:
                    child.core_node.parent = None
                    child.core_node.invalidate_matrix_cache()
                child._notify_transform_changed()
                self.child_removed.emit(child)
            self.children.clear()
            return

        core_child_map = {id(child): child for child in self.core_node.childs}
        current_vms = {id(vm.core_node): vm for vm in list(self.children)}

        # Удаление узлов, которых больше нет в core_node.childs
        for core_id, vm in list(current_vms.items()):
            if core_id not in core_child_map:
                self.children.remove(vm)
                vm.parent_vm = None
                if vm.core_node.parent is self.core_node:
                    vm.core_node.parent = None
                    vm.core_node.invalidate_matrix_cache()
                vm._notify_transform_changed()
                self.child_removed.emit(vm)

        # Добавление новых узлов или актуализация существующих
        for child_core in self.core_node.childs:
            if child_core.parent is not self.core_node:
                child_core.parent = self.core_node

            if id(child_core) not in current_vms:
                if NodeViewModel._factory is not None:
                    child_vm = NodeViewModel._factory(child_core, parent_vm=self)
                else:
                    child_vm = NodeViewModel(child_core, parent_vm=self)
                self.children.append(child_vm)
                child_vm._notify_transform_changed()
                self.child_added.emit(child_vm)
            else:
                existing_vm = current_vms[id(child_core)]
                if existing_vm.parent_vm is not self:
                    existing_vm.parent_vm = self
                existing_vm.sync_children_from_core()

        # Синхронизация порядка children с core_node.childs
        core_order = {id(child): idx for idx, child in enumerate(self.core_node.childs)}
        self.children.sort(key=lambda vm: core_order.get(id(vm.core_node), 0))
