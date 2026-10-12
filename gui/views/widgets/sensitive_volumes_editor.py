import logging
from typing import Any, List, Optional

from PySide6.QtCore import Qt, QPoint, QObject, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QMenu
)

from core.geometry.volumes import Volume
from core.scene.nodes import CompositeNode
from core.scene.dose_grid_node import DoseGridNode
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.data_handler_viewmodel import SensitiveVolumeHandlerViewModel

_logger = logging.getLogger(__name__)


class SensitiveVolumesListWidget(QListWidget):
    """
    Список чувствительных объемов с поддержкой Drag-and-Drop из дерева сцены.
    """

    def __init__(self, editor: 'SensitiveVolumesEditorWidget') -> None:
        super().__init__(editor)
        self.editor = editor
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)

    def _is_valid_drop_candidate(self, node_view_model: Optional[NodeViewModel]) -> bool:
        if node_view_model is None or self.editor.handler_vm is None:
            return False
        scene_view_model = self.editor.scene_vm
        if scene_view_model is not None:
            if node_view_model is scene_view_model.root_vm:
                return False
        core_node = node_view_model.core_node
        if isinstance(core_node, DoseGridNode):
            return False
        if not isinstance(core_node, (Volume, CompositeNode)):
            return False
        return True

    def dragEnterEvent(self, event) -> None:
        from gui.views.scene_tree_widget import SceneTreeWidget
        candidate_vm = SceneTreeWidget._current_dragged_vm
        if candidate_vm is None and event.mimeData().hasText():
            text_name = event.mimeData().text().strip()
            if self.editor.scene_vm is not None:
                candidate_vm = self.editor.scene_vm.find_by_name(text_name)
        if self._is_valid_drop_candidate(candidate_vm):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        from gui.views.scene_tree_widget import SceneTreeWidget
        candidate_vm = SceneTreeWidget._current_dragged_vm
        if candidate_vm is None and event.mimeData().hasText():
            text_name = event.mimeData().text().strip()
            if self.editor.scene_vm is not None:
                candidate_vm = self.editor.scene_vm.find_by_name(text_name)
        if self._is_valid_drop_candidate(candidate_vm):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        from gui.views.scene_tree_widget import SceneTreeWidget
        candidate_vm = SceneTreeWidget._current_dragged_vm
        if candidate_vm is None and event.mimeData().hasText():
            text_name = event.mimeData().text().strip()
            if self.editor.scene_vm is not None:
                candidate_vm = self.editor.scene_vm.find_by_name(text_name)

        if self._is_valid_drop_candidate(candidate_vm) and candidate_vm is not None:
            if self.editor.handler_vm is not None:
                self.editor.handler_vm.add_volume(candidate_vm.name)
            event.acceptProposedAction()
        else:
            event.ignore()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.editor._on_remove_clicked()
        else:
            super().keyPressEvent(event)


class SensitiveVolumesEditorWidget(QWidget):
    """
    Модульный виджет инспекции и управления чувствительными объемами (детекторами)
    для моделей SensitiveVolumeHandlerViewModel и HistoryAssemblerHandlerViewModel.
    """

    volume_selected = Signal(str)

    def __init__(
        self,
        handler_vm: Optional[SensitiveVolumeHandlerViewModel] = None,
        scene_vm: Optional[SceneViewModel] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.handler_vm: Optional[SensitiveVolumeHandlerViewModel] = None
        self.scene_vm: Optional[SceneViewModel] = None

        self._handler_connections: List[Any] = []
        self._scene_connections: List[Any] = []

        self._init_ui()

        if handler_vm is not None:
            self.set_viewmodel(handler_vm)
        if scene_vm is not None:
            self.set_scene_viewmodel(scene_vm)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        header_label = QLabel("Чувствительные объемы (детекторы):")
        header_label.setStyleSheet("font-weight: bold; color: #2ecc71; padding: 2px;")
        main_layout.addWidget(header_label)

        self.list_widget = SensitiveVolumesListWidget(self)
        self.list_widget.setMinimumHeight(110)
        self.list_widget.setToolTip(
            "Список узлов сцены, в которых регистрируются взаимодействия.\n"
            "Перетащите Volume или CompositeNode из дерева сцены сюда."
        )
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        main_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self.btn_add = QPushButton("➕ Добавить объем ▾")
        self.btn_add.setToolTip("Добавить узел сцены в список чувствительных объемов")
        self._add_menu = QMenu(self)
        self._add_menu.aboutToShow.connect(self._populate_add_menu)
        self.btn_add.setMenu(self._add_menu)
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("🗑 Удалить")
        self.btn_remove.setToolTip("Исключить выбранный объем из списка чувствительных")
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        btn_layout.addWidget(self.btn_remove)

        main_layout.addLayout(btn_layout)

    def set_viewmodel(self, handler_vm: Optional[SensitiveVolumeHandlerViewModel]) -> None:
        """Привязывает модель представления обработчика данных с обновлением подписок."""
        for connection in self._handler_connections:
            try:
                QObject.disconnect(connection)
            except (RuntimeError, TypeError):
                pass
        self._handler_connections.clear()

        self.handler_vm = handler_vm
        if self.handler_vm is not None:
            conn_volumes = self.handler_vm.volumes_changed.connect(self._refresh_list)
            conn_changed = self.handler_vm.changed.connect(self._refresh_list)
            self._handler_connections.extend([conn_volumes, conn_changed])

        self._refresh_list()

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        """Привязывает модель сцены для инспекции существующих узлов и их валидации."""
        for connection in self._scene_connections:
            try:
                QObject.disconnect(connection)
            except (RuntimeError, TypeError):
                pass
        self._scene_connections.clear()

        self.scene_vm = scene_vm
        if self.scene_vm is not None:
            conn_loaded = self.scene_vm.scene_loaded.connect(lambda loaded_obj: self._refresh_list())
            conn_added = self.scene_vm.node_added.connect(lambda added_vm: self._refresh_list())
            conn_removed = self.scene_vm.node_removed.connect(lambda removed_vm: self._refresh_list())
            conn_renamed = self.scene_vm.node_renamed.connect(lambda old_name, new_name: self._refresh_list())
            self._scene_connections.extend([conn_loaded, conn_added, conn_removed, conn_renamed])

        self._refresh_list()

    def _refresh_list(self) -> None:
        """Полная перерисовка элементов списка с проверкой валидности наличия узлов в сцене."""
        self.list_widget.clear()

        if self.handler_vm is None:
            return

        volume_names = self.handler_vm.sensitive_volumes
        for volume_name in volume_names:
            node_vm = self.scene_vm.find_by_name(volume_name) if self.scene_vm is not None else None
            if node_vm is not None:
                item = QListWidgetItem(f"🎯 {volume_name} ({node_vm.node_type})")
                item.setToolTip(f"Имя: {volume_name}\nТип: {node_vm.node_type}\nСтатус: Присутствует в сцене")
                item.setData(Qt.UserRole, volume_name)
            else:
                item = QListWidgetItem(f"⚠️ {volume_name} (отсутствует в сцене)")
                item.setToolTip(f"Имя: {volume_name}\nСтатус: Внимание! Узел не найден в текущей сцене.")
                item.setForeground(Qt.yellow)
                item.setData(Qt.UserRole, volume_name)
            self.list_widget.addItem(item)

    def _populate_add_menu(self) -> None:
        """Динамическое построение выпадающего меню доступных узлов сцены."""
        self._add_menu.clear()

        if self.scene_vm is None or self.handler_vm is None:
            disabled_action = self._add_menu.addAction("(Сцена не загружена)")
            disabled_action.setEnabled(False)
            return

        current_sensitive_volumes = set(self.handler_vm.sensitive_volumes)
        all_scene_nodes = self.scene_vm.all_nodes()

        available_nodes: List[NodeViewModel] = []
        for node_vm in all_scene_nodes:
            if node_vm is self.scene_vm.root_vm:
                continue
            if isinstance(node_vm.core_node, DoseGridNode):
                continue
            if isinstance(node_vm.core_node, (Volume, CompositeNode)):
                if node_vm.name not in current_sensitive_volumes:
                    available_nodes.append(node_vm)

        if not available_nodes:
            disabled_action = self._add_menu.addAction("(Нет доступных объемов)")
            disabled_action.setEnabled(False)
            return

        for node_vm in available_nodes:
            action_text = f"{node_vm.name} ({node_vm.node_type})"
            action = self._add_menu.addAction(action_text)
            action.triggered.connect(lambda checked=False, target_name=node_vm.name: self._on_add_volume(target_name))

    def _on_add_volume(self, volume_name: str) -> None:
        if self.handler_vm is not None:
            self.handler_vm.add_volume(volume_name)

    def _on_remove_clicked(self) -> None:
        if self.handler_vm is None:
            return
        selected_items = self.list_widget.selectedItems()
        if not selected_items and self.list_widget.currentItem() is not None:
            selected_items = [self.list_widget.currentItem()]
        for item in selected_items:
            volume_name = item.data(Qt.UserRole)
            if volume_name:
                self.handler_vm.remove_volume(str(volume_name))

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """
        При одиночном клике испускает сигнал выделения объема для 3D-оконтуривания
        без смены активной модели представления PropertyInspector.
        """
        volume_name = item.data(Qt.UserRole)
        if not volume_name:
            return
        self.volume_selected.emit(str(volume_name))

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """
        При двойном клике пользователь явно запрашивает инспекцию узла сцены.
        """
        volume_name = item.data(Qt.UserRole)
        if not volume_name:
            return
        if self.scene_vm is not None:
            node_vm = self.scene_vm.find_by_name(str(volume_name))
            if node_vm is not None:
                self.scene_vm.select_node(node_vm)
