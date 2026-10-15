"""
Хост-виджет инспектора свойств узлов сцены, протоколов и обработчиков данных.
Организован на базе компонентного реестра изолированных секций (gui.views.inspectors).
"""
import logging
from typing import Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.viewmodels.data_handler_viewmodel import (
    BaseDataHandlerViewModel,
    DataManagerViewModel,
)
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.procedure_viewmodel import BaseProcedureViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.inspectors import (
    CollimatorInspectorSection,
    DataHandlerInspectorSection,
    DataManagerInspectorSection,
    DoseGridInspectorSection,
    GammaCameraInspectorSection,
    GantryInspectorSection,
    GeneralInspectorSection,
    InspectorRegistry,
    ProcedureInspectorSection,
    SourceInspectorSection,
    TransformInspectorSection,
    VolumeInspectorSection,
    VoxelVolumeInspectorSection,
)

_logger = logging.getLogger(__name__)


class PropertyInspector(QWidget):
    """
    Хост-виджет инспектора параметров выбранного объекта.
    Управляет динамическим монтажом и связыванием специализированных секций
    на базе компонентного реестра InspectorRegistry.
    """

    dose_voxel_size_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None, scene_vm: Optional[SceneViewModel] = None) -> None:
        super().__init__(parent)
        self.current_vm: Optional[Any] = None
        self.scene_vm: Optional[SceneViewModel] = None
        self.procedure_vm: Optional[BaseProcedureViewModel] = None
        self._is_node_vm_connected: bool = False
        self._is_changed_vm_connected: bool = False
        self._registry = InspectorRegistry()

        self._init_ui()
        if scene_vm is not None:
            self.set_scene_viewmodel(scene_vm)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.content_layout = QVBoxLayout(container)
        self.content_layout.setContentsMargins(4, 4, 4, 4)

        # Инстанцирование секций
        self.general_section = GeneralInspectorSection(parent=container)
        self.transform_section = TransformInspectorSection(parent=container)
        self.volume_section = VolumeInspectorSection(parent=container)
        self.dose_grid_section = DoseGridInspectorSection(parent=container)
        self.collimator_section = CollimatorInspectorSection(parent=container)
        self.voxel_section = VoxelVolumeInspectorSection(parent=container)
        self.source_section = SourceInspectorSection(parent=container)
        self.spect_section = GammaCameraInspectorSection(parent=container)
        self.gantry_section = GantryInspectorSection(parent=container)
        self.procedure_section = ProcedureInspectorSection(parent=container)
        self.data_handler_section = DataHandlerInspectorSection(parent=container)
        self.data_manager_section = DataManagerInspectorSection(parent=container)

        sections = [
            self.general_section,
            self.transform_section,
            self.volume_section,
            self.dose_grid_section,
            self.collimator_section,
            self.voxel_section,
            self.source_section,
            self.spect_section,
            self.gantry_section,
            self.procedure_section,
            self.data_handler_section,
            self.data_manager_section,
        ]

        for section in sections:
            self._registry.register_section(section)
            self.content_layout.addWidget(section)
            section.binding_changed.connect(self._on_section_binding_changed)
            section.target_viewmodel_requested.connect(self.set_target_viewmodel)

        self.dose_grid_section.dose_voxel_size_changed.connect(self.dose_voxel_size_changed)

        self.content_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        self.clear_selection()

    def set_procedure_viewmodel(self, procedure_vm: Optional[BaseProcedureViewModel]) -> None:
        """Привязка активной модели протокола для инспектирования параметров."""
        self.procedure_vm = procedure_vm
        for section in self._registry.get_all_sections():
            section.set_procedure_viewmodel(procedure_vm)

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        """Привязка модели сцены для доступа к узлам и контексту валидации."""
        self.scene_vm = scene_vm
        for section in self._registry.get_all_sections():
            section.set_scene_viewmodel(scene_vm)

    def set_target_viewmodel(self, vm: Optional[Any]) -> None:
        """Устанавливает целевую модель представления для инспекции."""
        if self.current_vm is vm:
            return

        if self.current_vm is not None:
            if self._is_node_vm_connected and isinstance(self.current_vm, NodeViewModel):
                self.current_vm.property_changed.disconnect(self._on_property_changed_externally)
                self._is_node_vm_connected = False
            elif self._is_changed_vm_connected and isinstance(
                self.current_vm,
                (BaseProcedureViewModel, BaseDataHandlerViewModel, DataManagerViewModel),
            ):
                self.current_vm.changed.disconnect(self.update_all_fields)
                self._is_changed_vm_connected = False

        self.current_vm = vm
        if self.current_vm is None:
            self.clear_selection()
            return

        if isinstance(self.current_vm, NodeViewModel):
            self.current_vm.property_changed.connect(self._on_property_changed_externally)
            self._is_node_vm_connected = True
        elif isinstance(self.current_vm, (BaseProcedureViewModel, BaseDataHandlerViewModel, DataManagerViewModel)):
            self.current_vm.changed.connect(self.update_all_fields)
            self._is_changed_vm_connected = True

        self.update_all_fields()

    def clear_selection(self) -> None:
        """Сброс и скрытие всех секций при отсутствии выбранного объекта."""
        for section in self._registry.get_all_sections():
            section.unbind()
            section.setVisible(False)

    def update_all_fields(self) -> None:
        """Синхронизация активных секций UI со значениями ViewModel."""
        if self.current_vm is None:
            self.clear_selection()
            return

        active_sections = self._registry.get_active_sections(self.current_vm)
        for section in self._registry.get_all_sections():
            if section in active_sections:
                section.setVisible(True)
                section.setEnabled(True)
                section.bind_viewmodel(self.current_vm)
            else:
                section.unbind()
                section.setVisible(False)

        if isinstance(self.current_vm, NodeViewModel):
            self.volume_section.update_constraint_state(self.current_vm)

    def _on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Трансляция событий изменения свойств активной модели в отображаемые секции."""
        if self.current_vm is None:
            return
        for section in self._registry.get_active_sections(self.current_vm):
            section.on_property_changed_externally(prop_name, new_val)

    def _on_section_binding_changed(self, prop_name: str, var_name: str) -> None:
        """Синхронизация между взаимосвязанными полями в разных секциях при смене привязки."""
        if self.current_vm is None:
            return
        for section in self._registry.get_active_sections(self.current_vm):
            section.on_property_changed_externally("bindings", None)


__all__ = [
    "PropertyInspector",
]

