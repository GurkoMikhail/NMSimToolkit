import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import hepunits as units
from PySide6.QtCore import Qt, Signal, QPoint, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
    QTreeWidgetItem, QPushButton, QLabel, QHeaderView,
    QMenu, QMessageBox, QListWidget, QListWidgetItem, QSplitter
)

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.gamma_cameras import GammaCamera
from core.materials.materials import Material, MaterialArray
from core.source.sources import Source, PointSource
from core.scene.nodes import CompositeNode, SpatialNode
from core.scene.dose_grid_node import DoseGridNode
from core.other.typing_definitions import Float
import settings.database_setting as database_setting

_logger = logging.getLogger(__name__)


class SceneTree(QTreeWidget):
    """
    Дерево сцены с поддержкой перетаскивания (Drag and Drop)
    и реорганизации узлов графа сцены через SceneViewModel.
    """

    def __init__(self, owner: 'SceneTreeWidget') -> None:
        super().__init__(owner)
        self.owner = owner
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DragDrop)

    def startDrag(self, supportedActions) -> None:
        src_item = self.currentItem()
        if src_item is not None and self.owner is not None:
            self.owner._dragged_vm = self.owner._vm_by_item.get(src_item)
        try:
            super().startDrag(supportedActions)
        finally:
            if self.owner is not None:
                self.owner._dragged_vm = None

    def dropEvent(self, event) -> None:
        src_item = self.currentItem()
        if not src_item or self.owner.scene_vm is None:
            event.ignore()
            return

        pos = event.position().toPoint()
        target_item = self.itemAt(pos)

        src_vm = self.owner._vm_by_item.get(src_item)
        if src_vm is None or src_vm is self.owner.scene_vm.root_vm:
            event.ignore()
            return

        # Запрет перетаскивания узла внутрь самого себя или своих потомков
        if target_item is not None:
            target_vm = self.owner._vm_by_item.get(target_item)
            if target_vm is not None:
                curr = target_vm
                while curr is not None:
                    if curr is src_vm:
                        event.ignore()
                        return
                    curr = curr.parent_vm

        # Запрет перетаскивания между несовместимыми узлами
        parent_item = target_item if target_item is not None else self.invisibleRootItem()
        new_parent_vm = self.owner._vm_by_item.get(parent_item) if target_item is not None else self.owner.scene_vm.root_vm

        if new_parent_vm is None:
            event.ignore()
            return

        success = self.owner.scene_vm.move_node(src_vm, new_parent_vm)
        if success:
            event.accept()
            self.owner.rebuild_tree()
            self.owner._on_node_selected_externally(src_vm)
        else:
            event.ignore()


class SensitiveVolumesList(QListWidget):
    """
    Список чувствительных объемов с поддержкой Drag-and-Drop из дерева сцены.
    """

    def __init__(self, owner: 'SceneTreeWidget') -> None:
        super().__init__(owner)
        self.owner = owner
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)

    def _is_valid_drop_target(self, vm: Optional[NodeViewModel]) -> bool:
        """
        Проверяет, допустим ли перенос узла в список чувствительных детекторов.
        Запрещены None, узлы без физической геометрии (не VolumeViewModel) и корневой узел сцены.
        """
        if vm is None or not isinstance(vm, VolumeViewModel):
            return False
        if self.owner.scene_vm is not None and vm is self.owner.scene_vm.root_vm:
            return False
        return True

    def dragEnterEvent(self, event) -> None:
        if self._is_valid_drop_target(self.owner._dragged_vm):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._is_valid_drop_target(self.owner._dragged_vm):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        vm = self.owner._dragged_vm
        if self._is_valid_drop_target(vm):
            assert isinstance(vm, VolumeViewModel)
            vm.is_sensitive_detector = True
            event.acceptProposedAction()
        else:
            event.ignore()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Delete:
            self.owner._on_remove_detector_clicked()
        else:
            super().keyPressEvent(event)


class SceneTreeWidget(QWidget):
    """
    Виджет иерархического дерева сцены.
    Отображает структуру узлов (SpatialNode, Volume, GammaCamera и др.),
    позволяет выбирать узлы для инспекции и управлять иерархией.
    Использует обратный индекс _vm_by_item для O(1) поиска ViewModel при смене выделения.
    """

    def __init__(self, scene_vm: Optional[SceneViewModel] = None, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.scene_vm = scene_vm
        self._item_map: Dict[int, QTreeWidgetItem] = {}
        self._vm_by_item: Dict[QTreeWidgetItem, NodeViewModel] = {}
        self._connected_vms: Dict[int, Tuple[NodeViewModel, Any]] = {}
        self._scene_vm_conns: List[Any] = []
        self._dragged_vm: Optional[NodeViewModel] = None
        self._sensitive_item_map: Dict[int, QListWidgetItem] = {}
        self._vm_by_sensitive_item: Dict[int, VolumeViewModel] = {}

        self._init_ui()
        if self.scene_vm is not None:
            self.set_scene_viewmodel(self.scene_vm)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Vertical, self)

        # 1. Верхняя секция: Дерево графа сцены
        tree_container = QWidget(self)
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(4)

        title_label = QLabel("Иерархия сцены (Scene Graph)")
        title_label.setStyleSheet("font-weight: bold; color: #ecf0f1; padding: 2px;")
        tree_layout.addWidget(title_label)

        # Дерево узлов с поддержкой Drag-and-Drop
        self.tree = SceneTree(self)
        self.tree.setHeaderLabels(["Имя узла", "Тип"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        tree_layout.addWidget(self.tree)

        # Кнопки управления узлами
        btn_layout = QHBoxLayout()
        self.btn_add_node = QPushButton("+ Добавить узел ▾")
        self.btn_add_node.setToolTip("Добавить узел сцены выбранного типа")
        self._add_button_menu = QMenu(self)
        self._populate_add_menu(self._add_button_menu, parent_vm=None)
        self.btn_add_node.setMenu(self._add_button_menu)

        self.btn_remove = QPushButton("- Удалить")
        self.btn_remove.setToolTip("Удалить выбранный узел")
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        btn_layout.addWidget(self.btn_add_node)
        btn_layout.addWidget(self.btn_remove)
        tree_layout.addLayout(btn_layout)

        splitter.addWidget(tree_container)

        # 2. Нижняя секция: Список чувствительных объемов (Детекторы)
        sens_container = QWidget(self)
        sens_layout = QVBoxLayout(sens_container)
        sens_layout.setContentsMargins(0, 0, 0, 0)
        sens_layout.setSpacing(4)

        sens_title = QLabel("Чувствительные объёмы (Детекторы)")
        sens_title.setStyleSheet("font-weight: bold; color: #2ecc71; padding: 2px;")
        sens_title.setToolTip("Список объемов, в которых регистрируются взаимодействия для проекции. Перетащите Volume сюда.")
        sens_layout.addWidget(sens_title)

        self.sensitive_list = SensitiveVolumesList(self)
        self.sensitive_list.setToolTip("Перетащите узел Volume из дерева сцены сюда для назначения детектором")
        self.sensitive_list.itemSelectionChanged.connect(self._on_sensitive_selection_changed)
        self.sensitive_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sensitive_list.customContextMenuRequested.connect(self._show_sensitive_context_menu)
        sens_layout.addWidget(self.sensitive_list)

        sens_btn_layout = QHBoxLayout()
        self.btn_remove_detector = QPushButton("- Исключить из детекторов")
        self.btn_remove_detector.setToolTip("Снять статус чувствительного детектора с выбранного объема")
        self.btn_remove_detector.clicked.connect(self._on_remove_detector_clicked)
        sens_btn_layout.addWidget(self.btn_remove_detector)
        sens_layout.addLayout(sens_btn_layout)

        splitter.addWidget(sens_container)
        splitter.setSizes([350, 150])

        main_layout.addWidget(splitter)

    def set_scene_viewmodel(self, scene_vm: SceneViewModel) -> None:
        """
        Подключает модель представления сцены к дереву с корректным отключением старых сигналов.
        """
        for c in self._scene_vm_conns:
            try:
                QObject.disconnect(c)
            except (RuntimeError, TypeError):
                pass
        self._scene_vm_conns.clear()

        self.scene_vm = scene_vm
        if self.scene_vm is not None:
            c1 = self.scene_vm.scene_loaded.connect(self.rebuild_tree)
            c2 = self.scene_vm.node_selected.connect(self._on_node_selected_externally)
            c3 = self.scene_vm.node_added.connect(lambda n: self.rebuild_tree())
            c4 = self.scene_vm.node_removed.connect(lambda n: self.rebuild_tree())
            self._scene_vm_conns.extend([c1, c2, c3, c4])

            if self.scene_vm.root_vm is not None:
                self.rebuild_tree()

    def rebuild_tree(self) -> None:
        """
        Полная перестройка элементов дерева по иерархии ViewModel.
        """
        # Отключаем обработчики сигналов от предыдущей структуры дерева
        for node_id, (vm, conn) in list(self._connected_vms.items()):
            try:
                QObject.disconnect(conn)
            except (RuntimeError, TypeError):
                pass
        self._connected_vms.clear()

        self.tree.clear()
        self._item_map.clear()
        self._vm_by_item.clear()

        if self.scene_vm is None or self.scene_vm.root_vm is None:
            return

        root_item = self._create_tree_item(self.scene_vm.root_vm)
        self.tree.addTopLevelItem(root_item)
        self.tree.expandAll()

        # Восстановление списка чувствительных объемов
        self._refresh_sensitive_volumes_list()

        # Восстановление подсветки выбранного узла
        if self.scene_vm.selected_node is not None:
            self._on_node_selected_externally(self.scene_vm.selected_node)

    def _create_tree_item(self, vm: NodeViewModel) -> QTreeWidgetItem:
        item = QTreeWidgetItem([vm.name, vm.node_type])
        self._item_map[id(vm)] = item
        self._vm_by_item[item] = vm

        node_id = id(vm)
        if node_id not in self._connected_vms:
            conn = vm.property_changed.connect(
                lambda prop, val, n=vm: self._on_node_property_changed(n, prop, val)
            )
            self._connected_vms[node_id] = (vm, conn)

        for child_vm in vm.children:
            child_item = self._create_tree_item(child_vm)
            item.addChild(child_item)

        return item

    def _on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any) -> None:
        if prop_name == 'name':
            item = self._item_map.get(id(node_vm))
            if item is not None:
                item.setText(0, str(value))
            if isinstance(node_vm, VolumeViewModel):
                sens_item = self._sensitive_item_map.get(id(node_vm))
                if sens_item is not None:
                    sens_item.setText(f"🎯 {value} ({node_vm.node_type})")
        elif prop_name == 'is_sensitive_detector' and isinstance(node_vm, VolumeViewModel):
            if bool(value):
                self._add_sensitive_item(node_vm)
            else:
                self._remove_sensitive_item(node_vm)

    def _on_tree_selection_changed(self) -> None:
        selected_items = self.tree.selectedItems()
        if not selected_items or self.scene_vm is None:
            return

        selected_item = selected_items[0]
        # O(1) поиск ViewModel по выбранному QTreeWidgetItem через обратный словарь
        vm = self._vm_by_item.get(selected_item)
        if vm is not None:
            self.scene_vm.select_node(vm)

    def _on_node_selected_externally(self, vm: Optional[NodeViewModel]) -> None:
        if vm is None:
            return

        item = self._item_map.get(id(vm))
        if item is not None and not item.isSelected():
            self.tree.blockSignals(True)
            self.tree.clearSelection()
            item.setSelected(True)
            self.tree.setCurrentItem(item)
            self.tree.blockSignals(False)

        sens_item = self._sensitive_item_map.get(id(vm))
        if sens_item is not None and not sens_item.isSelected():
            self.sensitive_list.blockSignals(True)
            self.sensitive_list.clearSelection()
            sens_item.setSelected(True)
            self.sensitive_list.setCurrentItem(sens_item)
            self.sensitive_list.blockSignals(False)
        elif sens_item is None:
            self.sensitive_list.blockSignals(True)
            self.sensitive_list.clearSelection()
            self.sensitive_list.blockSignals(False)

    def _refresh_sensitive_volumes_list(self) -> None:
        """
        Полное обновление списка чувствительных детекторов по всем узлам ViewModel сцены.
        """
        self.sensitive_list.clear()
        self._sensitive_item_map.clear()
        self._vm_by_sensitive_item.clear()

        if self.scene_vm is None:
            return

        for node_vm in self.scene_vm.all_nodes():
            if isinstance(node_vm, VolumeViewModel) and node_vm.is_sensitive_detector:
                self._add_sensitive_item(node_vm)

    def _add_sensitive_item(self, vm: VolumeViewModel) -> None:
        """
        Добавляет элемент в список чувствительных объемов.
        """
        if id(vm) in self._sensitive_item_map:
            return
        item = QListWidgetItem(f"🎯 {vm.name} ({vm.node_type})")
        item.setToolTip(f"ID: {id(vm)}\nОбъем: {vm.name}\nМатериал: {vm.material_name}")
        self.sensitive_list.addItem(item)
        self._sensitive_item_map[id(vm)] = item
        self._vm_by_sensitive_item[id(item)] = vm

    def _remove_sensitive_item(self, vm: VolumeViewModel) -> None:
        """
        Удаляет элемент из списка чувствительных объемов.
        """
        item = self._sensitive_item_map.pop(id(vm), None)
        if item is not None:
            self._vm_by_sensitive_item.pop(id(item), None)
            row = self.sensitive_list.row(item)
            if row >= 0:
                self.sensitive_list.takeItem(row)

    def _on_sensitive_selection_changed(self) -> None:
        """
        Синхронизация выбора узла при клике в списке чувствительных объемов.
        """
        items = self.sensitive_list.selectedItems()
        if not items or self.scene_vm is None:
            return
        vm = self._vm_by_sensitive_item.get(id(items[0]))
        if vm is not None:
            self.scene_vm.select_node(vm)

    def _on_remove_detector_clicked(self) -> None:
        """
        Снятие статуса детектора с выбранного в списке объема.
        """
        items = self.sensitive_list.selectedItems()
        if not items:
            return
        vm = self._vm_by_sensitive_item.get(id(items[0]))
        if vm is not None:
            vm.is_sensitive_detector = False

    def _show_sensitive_context_menu(self, pos: QPoint) -> None:
        """
        Контекстное меню для элементов списка чувствительных объемов.
        """
        item = self.sensitive_list.itemAt(pos)
        if item is None:
            return
        vm = self._vm_by_sensitive_item.get(id(item))
        if vm is None:
            return

        menu = QMenu(self)
        act_remove = menu.addAction("Исключить из детекторов")
        act_remove.triggered.connect(lambda: self._disable_detector(vm))
        menu.exec(self.sensitive_list.viewport().mapToGlobal(pos))

    def _disable_detector(self, vm: VolumeViewModel) -> None:
        vm.is_sensitive_detector = False

    def _enable_detector(self, vm: VolumeViewModel) -> None:
        vm.is_sensitive_detector = True

    def _show_context_menu(self, pos: QPoint) -> None:
        if self.scene_vm is None:
            return

        item = self.tree.itemAt(pos)
        target_vm = self._vm_by_item.get(item) if item is not None else self.scene_vm.root_vm
        if target_vm is None:
            target_vm = self.scene_vm.root_vm

        menu = QMenu(self)

        # Подменю добавления дочернего узла
        add_submenu = menu.addMenu("+ Добавить дочерний узел")
        self._populate_add_menu(add_submenu, parent_vm=target_vm)

        menu.addSeparator()
        act_child_dose = menu.addAction(f"Создать дочернюю сетку дозы (по размеру {target_vm.name})")
        act_child_dose.triggered.connect(lambda: self._add_dose_grid_node(target_vm))

        if isinstance(target_vm, VolumeViewModel) and target_vm is not self.scene_vm.root_vm:
            menu.addSeparator()
            if target_vm.is_sensitive_detector:
                act_det = menu.addAction("Снять статус чувствительного детектора")
                act_det.triggered.connect(lambda: self._disable_detector(target_vm))
            else:
                act_det = menu.addAction("Назначить чувствительным детектором")
                act_det.triggered.connect(lambda: self._enable_detector(target_vm))

        if target_vm is not self.scene_vm.root_vm:
            menu.addSeparator()
            act_root = menu.addAction("Переместить в корень")
            act_root.triggered.connect(lambda: self._move_node_to_root(target_vm))

            act_del = menu.addAction("Удалить")
            act_del.triggered.connect(lambda: self._remove_specific_node(target_vm))

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _populate_add_menu(self, menu: QMenu, parent_vm: Optional[NodeViewModel] = None) -> None:
        """
        Заполняет меню действиями создания различных типов узлов сцены.
        """
        act_box = menu.addAction("Объем (Параллелепипед / Box)")
        act_box.triggered.connect(lambda: self._add_box_volume(parent_vm))

        act_voxel = menu.addAction("Воксельный фантом (WoodcockVoxelVolume)")
        act_voxel.triggered.connect(lambda: self._add_voxel_volume(parent_vm))

        act_pt_src = menu.addAction("Точечный источник (PointSource)")
        act_pt_src.triggered.connect(lambda: self._add_point_source(parent_vm))

        act_vox_src = menu.addAction("Воксельный источник (Source)")
        act_vox_src.triggered.connect(lambda: self._add_voxel_source(parent_vm))

        act_dose = menu.addAction("Сетка дозы (Dose Scorer)")
        act_dose.triggered.connect(lambda: self._add_dose_grid_node(parent_vm))

        act_group = menu.addAction("Группа / Контейнер (CompositeNode)")
        act_group.triggered.connect(lambda: self._add_composite_node(parent_vm))

        act_gamma_cam = menu.addAction("Гамма-камера ОФЭКТ (GammaCamera)")
        act_gamma_cam.triggered.connect(lambda: self._add_gamma_camera(parent_vm))

    def _get_parent_vm(self, parent_vm: Optional[NodeViewModel]) -> Optional[NodeViewModel]:
        if parent_vm is not None:
            return parent_vm
        if self.scene_vm is not None:
            return self.scene_vm.selected_node or self.scene_vm.root_vm
        return None

    def _get_unique_name(self, base_prefix: str) -> str:
        if self.scene_vm is None:
            return f"{base_prefix}_1"
        existing_names = {node.name for node in self.scene_vm.all_nodes()}
        idx = 1
        while f"{base_prefix}_{idx}" in existing_names:
            idx += 1
        return f"{base_prefix}_{idx}"

    def _add_box_volume(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        vol_name = self._get_unique_name("Volume")
        mat = Material(name="Water")
        geo = Box(100.0, 100.0, 100.0)
        vol = Volume(geometry=geo, material=mat, name=vol_name)
        vol_vm = VolumeViewModel(vol)
        try:
            self.scene_vm.add_node(parent, vol_vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_voxel_volume(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        vol_name = self._get_unique_name("Phantom")
        mat_arr = MaterialArray((16, 16, 16))
        voxel_size = Float(4.0 * units.mm)
        vol = WoodcockVoxelVolume(voxel_size=voxel_size, material_distribution=mat_arr, name=vol_name)
        vm = VoxelVolumeViewModel(vol)
        try:
            self.scene_vm.add_node(parent, vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_point_source(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        src_name = self._get_unique_name("PointSource")
        src = PointSource(
            activity=Float(1e6 * units.becquerel),
            energy=Float(140.5 * units.keV)
        )
        src.name = src_name
        vm = SourceViewModel(src)
        try:
            self.scene_vm.add_node(parent, vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_voxel_source(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        src_name = self._get_unique_name("Source")
        dist = np.ones((16, 16, 16), dtype=Float)
        src = Source(
            distribution=dist,
            activity=Float(1e6 * units.becquerel),
            voxel_size=Float(4.0 * units.mm),
            energy=Float(140.5 * units.keV)
        )
        src.name = src_name
        vm = SourceViewModel(src)
        try:
            self.scene_vm.add_node(parent, vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_composite_node(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        grp_name = self._get_unique_name("Group")
        node = CompositeNode(name=grp_name)
        vm = NodeViewModel(node)
        try:
            self.scene_vm.add_node(parent, vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_gamma_camera(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        cam_name = self._get_unique_name("GammaCamera")
        col_mat = database_setting.material_database.get('Pb', Material(name='Lead'))
        det_mat = database_setting.material_database.get('Sodium Iodide', Material(name='NaI'))
        col = Volume(geometry=Box(400.0, 400.0, 30.0), material=col_mat, name=f"Collimator_{cam_name}")
        det = Volume(geometry=Box(400.0, 400.0, 10.0), material=det_mat, name=f"Detector_{cam_name}")
        cam = GammaCamera(collimator=col, detector=det, name=cam_name)
        vm = GammaCameraViewModel(cam)
        try:
            self.scene_vm.add_node(parent, vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_dose_grid_node(self, parent_vm: Optional[NodeViewModel] = None) -> None:
        """
        Создание узла сетки дозы. Если узел создается для конкретного родителя (гамма-камера,
        объем, фантом и т.д.), его размеры автоматически подгоняются под BoundingBox родителя.
        """
        parent = self._get_parent_vm(parent_vm)
        if self.scene_vm is None or parent is None:
            return

        if isinstance(parent, VolumeViewModel) and parent is not self.scene_vm.root_vm:
            b_size = parent.local_bound
            grid_size = [float(b_size[0]), float(b_size[1]), float(b_size[2])]
            grid_name = self._get_unique_name(f"DoseGrid_{parent.name}")
        elif isinstance(parent, VolumeViewModel):
            b_size = parent.local_bound
            grid_size = [float(b_size[0]), float(b_size[1]), float(b_size[2])]
            grid_name = self._get_unique_name("DoseGrid")
        else:
            grid_size = [100.0, 100.0, 100.0]
            grid_name = self._get_unique_name("DoseGrid")

        voxel_size = 5.0
        node = DoseGridNode(name=grid_name, size=grid_size, dose_voxel_size=voxel_size)
        vm = DoseGridViewModel(node)
        try:
            self.scene_vm.add_node(parent, vm)
        except (TypeError, ValueError) as e:
            QMessageBox.warning(self, "Ошибка добавления", str(e))

    def _add_dose_grid_for_volume(self, volume_vm: NodeViewModel) -> None:
        """
        Быстрое создание дочерней сетки дозы, соответствующей размерам BoundingBox родительского узла.
        """
        self._add_dose_grid_node(volume_vm)

    def _move_node_to_root(self, target_vm: NodeViewModel) -> None:
        if self.scene_vm is None or self.scene_vm.root_vm is None:
            return
        if self.scene_vm.move_node(target_vm, self.scene_vm.root_vm):
            self.rebuild_tree()
            self._on_node_selected_externally(target_vm)

    def _remove_specific_node(self, target_vm: NodeViewModel) -> None:
        if self.scene_vm is None or target_vm is self.scene_vm.root_vm:
            return
        self.scene_vm.remove_node(target_vm)

    def _on_add_box_clicked(self) -> None:
        self._add_box_volume()

    def _on_remove_clicked(self) -> None:
        if self.scene_vm is None or self.scene_vm.selected_node is None:
            return

        target = self.scene_vm.selected_node
        if target is self.scene_vm.root_vm:
            return  # Корневой узел удалять нельзя

        self.scene_vm.remove_node(target)
