"""
Секция параметров обработчиков данных (DataHandlerViewModel).
"""
from typing import Any, Optional

from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QLineEdit, QWidget

from gui.viewmodels.data_handler_viewmodel import (
    BaseDataHandlerViewModel,
    DirectStreamHandlerViewModel,
    DoseMapHandlerViewModel,
    HistoryAssemblerHandlerViewModel,
    SensitiveVolumeHandlerViewModel,
)
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.views.inspectors.base_section import BaseInspectorSection
from gui.views.widgets import SensitiveVolumesEditorWidget


class DataHandlerInspectorSection(BaseInspectorSection):
    """Секция настройки обработчиков потоков частиц, чувствительных объемов и карт дозы."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Параметры обработчика данных", parent)

    def _init_ui(self) -> None:
        form_layout = QFormLayout(self)

        self.lbl_dh_type = QLabel()
        form_layout.addRow("Тип:", self.lbl_dh_type)

        self.chk_dh_show_escaped = QCheckBox("Отображать вылетевшие треки")
        self.chk_dh_show_escaped.toggled.connect(self._on_dh_show_escaped_toggled)
        form_layout.addRow(self.chk_dh_show_escaped)

        self.sensitive_volumes_editor = SensitiveVolumesEditorWidget(parent=self)
        form_layout.addRow(self.sensitive_volumes_editor)

        self.chk_dh_initial_states = QCheckBox("Сохранять начальные состояния фотонов")
        self.chk_dh_initial_states.toggled.connect(self._on_dh_initial_states_toggled)
        form_layout.addRow(self.chk_dh_initial_states)

        self.txt_dh_grid_names = QLineEdit()
        self.txt_dh_grid_names.editingFinished.connect(self._on_dh_grid_names_edited)
        form_layout.addRow("Сетки дозы (DoseGrids):", self.txt_dh_grid_names)

        self.txt_dh_shm_name = QLineEdit()
        self.txt_dh_shm_name.editingFinished.connect(self._on_dh_shm_name_edited)
        form_layout.addRow("Имя SharedMemory:", self.txt_dh_shm_name)

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        super().set_scene_viewmodel(scene_vm)
        self.sensitive_volumes_editor.set_scene_viewmodel(scene_vm)

    def can_handle(self, view_model: Any) -> bool:
        return isinstance(view_model, BaseDataHandlerViewModel)

    def unbind(self) -> None:
        super().unbind()
        self.sensitive_volumes_editor.set_viewmodel(None)

    def update_ui_from_vm(self) -> None:
        view_model = self._current_vm
        if not isinstance(view_model, BaseDataHandlerViewModel):
            return

        self.lbl_dh_type.setText(view_model.name)

        is_stream = isinstance(view_model, DirectStreamHandlerViewModel)
        self.chk_dh_show_escaped.setVisible(is_stream)
        if is_stream:
            self.chk_dh_show_escaped.setChecked(bool(view_model.show_escaped_tracks))

        is_sens = isinstance(view_model, (SensitiveVolumeHandlerViewModel, HistoryAssemblerHandlerViewModel))
        self.sensitive_volumes_editor.setVisible(is_sens)
        if is_sens:
            self.sensitive_volumes_editor.set_viewmodel(view_model)
        else:
            self.sensitive_volumes_editor.set_viewmodel(None)

        is_hist = isinstance(view_model, HistoryAssemblerHandlerViewModel)
        self.chk_dh_initial_states.setVisible(is_hist)
        if is_hist:
            self.chk_dh_initial_states.setChecked(bool(view_model.save_initial_states))

        is_dose = isinstance(view_model, DoseMapHandlerViewModel)
        self.txt_dh_grid_names.setVisible(is_dose)
        self.txt_dh_shm_name.setVisible(is_dose)
        if is_dose:
            if not view_model.grid_names and self._scene_vm is not None:
                grid_names = [
                    node_vm.name
                    for node_vm in self._scene_vm.all_nodes()
                    if isinstance(node_vm, DoseGridViewModel)
                ]
                if grid_names:
                    view_model.grid_names = grid_names
            self.txt_dh_grid_names.setText(", ".join(view_model.grid_names))
            self.txt_dh_shm_name.setText(view_model.shm_name)

    def _on_dh_show_escaped_toggled(self, checked: bool) -> None:
        if not self._is_updating_ui and isinstance(self._current_vm, DirectStreamHandlerViewModel):
            self._current_vm.show_escaped_tracks = checked

    def _on_dh_initial_states_toggled(self, checked: bool) -> None:
        if not self._is_updating_ui and isinstance(self._current_vm, HistoryAssemblerHandlerViewModel):
            self._current_vm.save_initial_states = checked

    def _on_dh_grid_names_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self._current_vm, DoseMapHandlerViewModel):
            grids = [
                grid_name.strip()
                for grid_name in self.txt_dh_grid_names.text().split(",")
                if grid_name.strip()
            ]
            self._current_vm.grid_names = grids

    def _on_dh_shm_name_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self._current_vm, DoseMapHandlerViewModel):
            self._current_vm.shm_name = self.txt_dh_shm_name.text().strip()
