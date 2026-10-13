"""
Базовый класс секции инспектора свойств.
"""
from typing import Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QWidget

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    CustomSweepProcedureViewModel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel


class BaseInspectorSection(QGroupBox):
    """
    Абстрактная изолированная секция инспектора свойств.
    Управляет собственными виджетами ввода и реактивным связыванием с моделью представления.
    """

    binding_changed = Signal(str, str)
    target_viewmodel_requested = Signal(object)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self._current_vm: Optional[Any] = None
        self._scene_vm: Optional[SceneViewModel] = None
        self._procedure_vm: Optional[BaseProcedureViewModel] = None
        self._is_updating_ui: bool = False
        self._init_ui()

    def _init_ui(self) -> None:
        """Построение виджетов и компоновки секции."""
        raise NotImplementedError

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        """Привязка модели сцены для доступа к узлам и контексту."""
        self._scene_vm = scene_vm

    def set_procedure_viewmodel(self, procedure_vm: Optional[BaseProcedureViewModel]) -> None:
        """Привязка активной модели протокола для синхронизации переменных."""
        self._procedure_vm = procedure_vm

    def bind_viewmodel(self, view_model: Any) -> None:
        """Связывание секции с целевой ViewModel и обновление отображения."""
        if self._current_vm is not view_model:
            self.unbind()
            self._current_vm = view_model
        self.update_ui_from_vm()

    def unbind(self) -> None:
        """Отписка от сигналов текущей ViewModel и очистка локального состояния."""
        self._current_vm = None

    def update_ui_from_vm(self) -> None:
        """Синхронизация значений полей UI из текущей ViewModel."""
        raise NotImplementedError

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Точечное обновление поля UI при изменении свойства ViewModel."""
        pass

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, применима ли данная секция к переданной модели представления."""
        raise NotImplementedError

    def _get_protocol_variables(self) -> list[str]:
        """Возвращает список доступных переменных активного протокола сканирования."""
        if self._procedure_vm is not None:
            variables_dict = self._procedure_vm.get_variables()
            if variables_dict:
                return [
                    variable_key
                    for variable_key in variables_dict.keys()
                    if variable_key not in ("current_angle", "current_time")
                ]
        return ["gantry_angle", "radius", "step_time"]

    def _on_variable_created(self, var_name: str) -> None:
        """Обработчик создания новой переменной сканирования через BindableDoubleField."""
        if isinstance(self._procedure_vm, CustomSweepProcedureViewModel):
            clean_name = str(var_name).strip().replace(" ", "_")
            if clean_name not in self._procedure_vm.all_variables:
                self._procedure_vm.set_variable(clean_name, [0.0], sweep_type="grid")

    def _on_field_binding_changed(self, prop_name: str, var_name: str) -> None:
        """Синхронизация привязки свойства узла в ViewModel, протоколе и сцене."""
        if self._is_updating_ui or not isinstance(self._current_vm, NodeViewModel):
            return

        clean_var_name = str(var_name) if var_name else ""
        if clean_var_name:
            self._current_vm.bindings[prop_name] = clean_var_name
            if self._procedure_vm is not None:
                self._procedure_vm.bind_node_property(self._current_vm.core_node, prop_name, clean_var_name)
            if self._scene_vm is not None:
                if self._current_vm.core_node not in self._scene_vm.bindings_registry:
                    self._scene_vm.bindings_registry[self._current_vm.core_node] = {}
                self._scene_vm.bindings_registry[self._current_vm.core_node][prop_name] = clean_var_name
        else:
            self._current_vm.bindings.pop(prop_name, None)
            if self._procedure_vm is not None:
                self._procedure_vm.unbind_node_property(self._current_vm.core_node, prop_name)
            if self._scene_vm is not None and self._current_vm.core_node in self._scene_vm.bindings_registry:
                self._scene_vm.bindings_registry[self._current_vm.core_node].pop(prop_name, None)

        # Синхронизация между spin_gantry_angle и spin_rot_x для станины GantryViewModel
        if isinstance(self._current_vm, GantryViewModel) and prop_name in ("rotation.alpha", "alpha", "gantry_angle"):
            if clean_var_name:
                self._current_vm.bindings["gantry_angle"] = clean_var_name
                self._current_vm.bindings["rotation.alpha"] = clean_var_name
                if self._procedure_vm is not None:
                    self._procedure_vm.bind_node_property(self._current_vm.core_node, "gantry_angle", clean_var_name)
                    self._procedure_vm.bind_node_property(self._current_vm.core_node, "rotation.alpha", clean_var_name)
                if self._scene_vm is not None:
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
