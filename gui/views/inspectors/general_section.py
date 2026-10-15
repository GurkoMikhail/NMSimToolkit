"""
Секция общих свойств узла сцены (имя и тип).
"""
from typing import Any, Optional

from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit, QWidget

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.views.inspectors.base_section import BaseInspectorSection


class GeneralInspectorSection(BaseInspectorSection):
    """Секция отображения имени и типа узла сцены."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию общих свойств узла сцены."""
        super().__init__("Общие свойства (General)", parent)

    def _init_ui(self) -> None:
        """Создает поля ввода имени и отображения типа узла."""
        form_layout = QFormLayout(self)
        self.txt_name = QLineEdit()
        self.txt_name.editingFinished.connect(self._on_name_changed)
        self.lbl_type = QLabel("None")
        form_layout.addRow("Имя:", self.txt_name)
        form_layout.addRow("Тип узла:", self.lbl_type)

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления узлом сцены NodeViewModel."""
        return isinstance(view_model, NodeViewModel)

    def unbind(self) -> None:
        """Сбрасывает поля имени и типа узла при отвязке модели представления."""
        super().unbind()
        self.txt_name.setText("")
        self.lbl_type.setText("Не выбран")

    def update_ui_from_vm(self) -> None:
        """Обновляет имя и тип узла из текущей NodeViewModel."""
        if isinstance(self._current_vm, NodeViewModel):
            self.txt_name.setText(str(self._current_vm.name))
            self.lbl_type.setText(str(self._current_vm.node_type))

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Обновляет текстовое поле имени при внешнем переименовании узла."""
        if prop_name == "name":
            self.txt_name.setText(str(new_val))

    def _on_name_changed(self) -> None:
        """Переименовывает узел сцены при завершении редактирования поля имени."""
        if not isinstance(self._current_vm, NodeViewModel):
            return
        old_name = self._current_vm.name
        new_name = self.txt_name.text().strip()
        if not new_name or old_name == new_name:
            return
        if self._scene_vm is not None:
            self._scene_vm.rename_node(self._current_vm, new_name)
        else:
            self._current_vm.name = new_name


__all__ = ["GeneralInspectorSection"]
