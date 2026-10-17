"""
Секция параметров станины томографа (GantryNode / GantryViewModel).
"""
from typing import Any, Optional

from PySide6.QtWidgets import QCheckBox, QFormLayout, QWidget

from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.views.inspectors.base_section import BaseInspectorSection
from gui.views.widgets import BindableDoubleField


class GantryInspectorSection(BaseInspectorSection):
    """Секция настройки угла ротора гентри и видимости кинематических направляющих."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию настройки станины томографа."""
        super().__init__("Параметры станины томографа (Gantry)", parent)

    def _init_ui(self) -> None:
        """Создает поле ввода угла ротора и флажок видимости направляющих."""
        form_layout = QFormLayout(self)

        self.spin_gantry_angle = BindableDoubleField(
            min_val=0.0,
            max_val=360.0,
            step=5.0,
            suffix=" °",
            decimals=2,
        )
        self.spin_gantry_angle.set_variable_provider(self._get_protocol_variables)
        self.spin_gantry_angle.set_variable_created_callback(self._on_variable_created)
        self.spin_gantry_angle.valueChanged.connect(self._on_gantry_angle_changed)
        self.spin_gantry_angle.bindingChanged.connect(self._on_gantry_binding_changed)

        self.chk_wireframe_visible = QCheckBox("Отображать направляющие (Wireframe)")
        self.chk_wireframe_visible.toggled.connect(self._on_gantry_wireframe_toggled)

        form_layout.addRow("Угол поворота ротора θ:", self.spin_gantry_angle)
        form_layout.addRow(self.chk_wireframe_visible)

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления станиной GantryViewModel."""
        return isinstance(view_model, GantryViewModel)

    def unbind(self) -> None:
        """Сбрасывает привязку угла ротора при отвязке модели станины."""
        super().unbind()
        self.spin_gantry_angle.unbind()

    def update_ui_from_vm(self) -> None:
        """Синхронизирует угол ротора, привязку и флажок каркаса из GantryViewModel."""
        if not isinstance(self._current_vm, GantryViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            bindings = self._current_vm.bindings
            if not bindings and self._scene_vm is not None and self._current_vm.core_node in self._scene_vm.bindings_registry:
                bindings = self._scene_vm.bindings_registry[self._current_vm.core_node]
            bound_var = bindings.get("rotation.alpha") or bindings.get("gantry_angle")
            if bound_var:
                self.spin_gantry_angle.bind_variable(bound_var)
            else:
                self.spin_gantry_angle.unbind()
            self.spin_gantry_angle.setValue(float(self._current_vm.gantry_angle))
            self.chk_wireframe_visible.setChecked(bool(self._current_vm.wireframe_visible))
        finally:
            self._is_updating_ui = old_state

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Точечно обновляет угол ротора, каркас или привязку при внешнем изменении."""
        if not isinstance(self._current_vm, GantryViewModel):
            return

        if prop_name == "gantry_angle":
            self._is_updating_ui = True
            try:
                self.spin_gantry_angle.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == "wireframe_visible":
            self._is_updating_ui = True
            try:
                self.chk_wireframe_visible.setChecked(bool(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == "bindings":
            bound_var = self._current_vm.bindings.get("rotation.alpha") or self._current_vm.bindings.get("gantry_angle")
            if bound_var:
                self.spin_gantry_angle.bind_variable(bound_var)
            else:
                self.spin_gantry_angle.unbind()

    def _on_gantry_angle_changed(self, value: float) -> None:
        """Передает новое значение угла поворота ротора в модель GantryViewModel."""
        if self._is_updating_ui or not isinstance(self._current_vm, GantryViewModel):
            return
        self._current_vm.gantry_angle = float(value)

    def _on_gantry_wireframe_toggled(self, checked: bool) -> None:
        """Переключает видимость кинематических направляющих станины."""
        if self._is_updating_ui or not isinstance(self._current_vm, GantryViewModel):
            return
        self._current_vm.wireframe_visible = checked

    def _on_gantry_binding_changed(self, var_name: str) -> None:
        """Обрабатывает изменение привязки поля угла ротора гентри."""
        self._on_field_binding_changed("gantry_angle", var_name)

    def _on_field_binding_changed(self, prop_name: str, var_name: str) -> None:
        """Синхронизирует привязку угла ротора между gantry_angle и rotation.alpha."""
        if self._is_updating_ui or not isinstance(self._current_vm, GantryViewModel):
            return

        clean_var_name = str(var_name) if var_name else ""
        if clean_var_name:
            self._current_vm.bindings["gantry_angle"] = clean_var_name
            self._current_vm.bindings["rotation.alpha"] = clean_var_name
            if self._procedure_vm is not None:
                self._procedure_vm.bind_node_property(self._current_vm.core_node, "gantry_angle", clean_var_name)
                self._procedure_vm.bind_node_property(self._current_vm.core_node, "rotation.alpha", clean_var_name)
            if self._scene_vm is not None:
                if self._current_vm.core_node not in self._scene_vm.bindings_registry:
                    self._scene_vm.bindings_registry[self._current_vm.core_node] = {}
                self._scene_vm.bindings_registry[self._current_vm.core_node]["gantry_angle"] = clean_var_name
                self._scene_vm.bindings_registry[self._current_vm.core_node]["rotation.alpha"] = clean_var_name
        else:
            self._current_vm.bindings.pop("gantry_angle", None)
            self._current_vm.bindings.pop("rotation.alpha", None)
            self._current_vm.bindings.pop("alpha", None)
            if self._procedure_vm is not None:
                self._procedure_vm.unbind_node_property(self._current_vm.core_node, "gantry_angle")
                self._procedure_vm.unbind_node_property(self._current_vm.core_node, "rotation.alpha")
                self._procedure_vm.unbind_node_property(self._current_vm.core_node, "alpha")
            if self._scene_vm is not None and self._current_vm.core_node in self._scene_vm.bindings_registry:
                self._scene_vm.bindings_registry[self._current_vm.core_node].pop("gantry_angle", None)
                self._scene_vm.bindings_registry[self._current_vm.core_node].pop("rotation.alpha", None)
                self._scene_vm.bindings_registry[self._current_vm.core_node].pop("alpha", None)

        self.binding_changed.emit(prop_name, clean_var_name)


__all__ = ["GantryInspectorSection"]
