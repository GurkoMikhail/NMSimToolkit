import logging
from typing import Callable, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QDoubleSpinBox, QToolButton, QMenu,
    QInputDialog, QLineEdit
)

_logger = logging.getLogger(__name__)

# CSS-стили для кнопки связывания, выпадающего меню и подсвеченного поля ввода
DEFAULT_BIND_BUTTON_STYLE = (
    "QToolButton { padding: 2px 4px; border: 1px solid #3c3c3c; border-radius: 3px; "
    "background: #252526; color: #cccccc; }\n"
    "QToolButton:hover { background: #3e3e42; }"
)

ACTIVE_BIND_BUTTON_STYLE = (
    "QToolButton { padding: 2px 6px; border: 1px solid #007acc; border-radius: 3px; "
    "background: #0e639c; color: #ffffff; font-weight: bold; }\n"
    "QToolButton:hover { background: #1177bb; }"
)

POPUP_MENU_STYLE = (
    "QMenu { background-color: #252526; color: #cccccc; border: 1px solid #3c3c3c; }\n"
    "QMenu::item:selected { background-color: #094771; }"
)

BOUND_SPIN_BOX_STYLE = (
    "QDoubleSpinBox { border: 1px solid #007acc; background-color: #1e293b; color: #93c5fd; }"
)


class BindableDoubleField(QWidget):
    """
    Виджет числового поля ввода со встроенной кнопкой связывания (Property Binding)
    с переменными параметрического сканирования или протокола (UE-Style).
    """

    valueChanged = Signal(float)
    bindingChanged = Signal(str)  # Передает имя переменной или пустую строку при отвязке

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 360.0,
        step: float = 1.0,
        suffix: str = "",
        decimals: int = 2,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.bound_variable: Optional[str] = None
        self._variable_provider: Optional[Callable[[], List[str]]] = None
        self._variable_created_callback: Optional[Callable[[str], None]] = None

        self._init_ui(min_val, max_val, step, suffix, decimals)

    def _init_ui(
        self,
        min_val: float,
        max_val: float,
        step: float,
        suffix: str,
        decimals: int,
    ) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setRange(min_val, max_val)
        self.spin_box.setSingleStep(step)
        self.spin_box.setDecimals(decimals)
        if suffix:
            self.spin_box.setSuffix(suffix)
        self.spin_box.valueChanged.connect(self.valueChanged.emit)

        self.btn_bind = QToolButton(self)
        self.btn_bind.setText("🔗")
        self.btn_bind.setToolTip("Связать свойство с переменной протокола сканирования")
        self.btn_bind.setPopupMode(QToolButton.InstantPopup)
        self.btn_bind.setStyleSheet(DEFAULT_BIND_BUTTON_STYLE)
        self._menu = QMenu(self)
        self._menu.setStyleSheet(POPUP_MENU_STYLE)
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self.btn_bind.setMenu(self._menu)
        self._rebuild_menu()

        layout.addWidget(self.spin_box, 1)
        layout.addWidget(self.btn_bind, 0)

    def set_variable_provider(self, provider: Callable[[], List[str]]) -> None:
        """Регистрация функции-поставщика имен доступных переменных протокола."""
        self._variable_provider = provider

    def set_variable_created_callback(self, callback: Callable[[str], None]) -> None:
        """Регистрация обработчика создания новой переменной в протоколе."""
        self._variable_created_callback = callback

    def _rebuild_menu(self) -> None:
        self._menu.clear()

        act_unbind = self._menu.addAction("Без привязки (статическое значение)")
        act_unbind.triggered.connect(self.unbind)
        self._menu.addSeparator()

        available_vars: List[str] = []
        if self._variable_provider is not None:
            try:
                available_vars = self._variable_provider()
            except Exception as error:
                _logger.warning("Ошибка при получении списка доступных переменных протокола: %s", error)

        if not available_vars:
            # Дефолтные типовые переменные
            available_vars = ["gantry_angle", "radius", "step_time"]

        for var_name in available_vars:
            is_current = (self.bound_variable == var_name)
            prefix = "✔ " if is_current else ""
            act_var = self._menu.addAction(f"{prefix}{var_name}")
            act_var.triggered.connect(lambda checked=False, bound_name=var_name: self.bind_variable(bound_name))

        self._menu.addSeparator()
        act_create = self._menu.addAction("➕ Создать переменную протокола...")
        act_create.triggered.connect(self._prompt_create_variable)

    def _prompt_create_variable(self) -> None:
        var_name, ok = QInputDialog.getText(
            self,
            "Новая переменная протокола",
            "Введите имя переменной (например, table_z):",
            QLineEdit.Normal,
            "",
        )
        if ok and var_name.strip():
            cleaned_var = var_name.strip().replace(" ", "_")
            if self._variable_created_callback is not None:
                try:
                    self._variable_created_callback(cleaned_var)
                except Exception as error:
                    _logger.error("Ошибка при создании переменной '%s': %s", cleaned_var, error)
            self.bind_variable(cleaned_var)

    def bind_variable(self, var_name: str) -> None:
        """
        Привязка числового свойства к переменной сканирования.
        Поле ввода переводится в режим только для чтения и визуально подсвечивается.
        """
        str_var = str(var_name)
        if self.bound_variable == str_var:
            return
        self.bound_variable = str_var
        self.spin_box.setReadOnly(True)
        self.btn_bind.setText(f"🔗 {str_var}")
        self.btn_bind.setStyleSheet(ACTIVE_BIND_BUTTON_STYLE)
        self.spin_box.setStyleSheet(BOUND_SPIN_BOX_STYLE)
        self._rebuild_menu()
        self.bindingChanged.emit(self.bound_variable)

    def unbind(self) -> None:
        """
        Отвязка свойства от переменной протокола, возврат к ручному редактированию.
        """
        if self.bound_variable is None:
            return
        self.bound_variable = None
        self.spin_box.setReadOnly(False)
        self.btn_bind.setText("🔗")
        self.btn_bind.setStyleSheet(DEFAULT_BIND_BUTTON_STYLE)
        self.spin_box.setStyleSheet("")
        self._rebuild_menu()
        self.bindingChanged.emit("")

    # Методы-делегаты для QDoubleSpinBox
    def value(self) -> float:
        return float(self.spin_box.value())

    def setValue(self, new_value: float) -> None:
        self.spin_box.setValue(float(new_value))

    def setRange(self, min_val: float, max_val: float) -> None:
        self.spin_box.setRange(float(min_val), float(max_val))

    def setMinimum(self, min_val: float) -> None:
        self.spin_box.setMinimum(float(min_val))

    def setMaximum(self, max_val: float) -> None:
        self.spin_box.setMaximum(float(max_val))

    def setSingleStep(self, step: float) -> None:
        self.spin_box.setSingleStep(float(step))

    def setDecimals(self, decimals: int) -> None:
        self.spin_box.setDecimals(int(decimals))

    def setSuffix(self, suffix: str) -> None:
        self.spin_box.setSuffix(suffix)

    def suffix(self) -> str:
        return self.spin_box.suffix()

    def blockSignals(self, is_blocked: bool) -> bool:
        return self.spin_box.blockSignals(is_blocked)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.spin_box.setEnabled(enabled)
        self.btn_bind.setEnabled(enabled)

    def isReadOnly(self) -> bool:
        return self.spin_box.isReadOnly()

    def setReadOnly(self, is_read_only: bool) -> None:
        if self.bound_variable is not None:
            self.spin_box.setReadOnly(True)
        else:
            self.spin_box.setReadOnly(is_read_only)

    def setToolTip(self, tooltip_text: str) -> None:
        super().setToolTip(tooltip_text)
        self.spin_box.setToolTip(tooltip_text)
