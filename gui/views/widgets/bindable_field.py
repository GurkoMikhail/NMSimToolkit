"""
Виджет числового поля ввода с поддержкой привязки к переменным протокола сканирования
и интеллектуальным парсингом физических единиц измерения.
"""
import re
from typing import Callable, List, Optional, Tuple, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFocusEvent, QKeyEvent, QValidator
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QToolButton,
    QWidget,
)

from core.config.units import (
    get_unit_category,
    normalize_unit_name,
    parse_unit_value,
)

VariableCreatedCallback = Callable[[str], None]

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

_INTERMEDIATE_SPINBOX_PATTERN = re.compile(r"^[+-]?[\d.,eE+\-/*\sA-Za-zА-Яа-яЁё°μÅ]*$")


def infer_unit_from_suffix(suffix: str) -> Optional[str]:
    """
    Определяет каноническую единицу измерения по текстовому суффиксу спинбокса
    (например, ' мм' -> 'mm', ' °' -> 'deg', ' кэВ' -> 'keV', ' с' -> 's', ' МБк' -> 'MBq').
    Возвращает None, если суффикс пуст или не соответствует известной физической единице.
    """
    normalized = normalize_unit_name(suffix)
    if not normalized:
        return None
    try:
        get_unit_category(normalized)
        return normalized
    except ValueError:
        return None


class UnitAwareDoubleSpinBox(QDoubleSpinBox):
    """
    Расширенный QDoubleSpinBox с поддержкой ввода физических величин с единицами измерения
    (например, '25 cm', '0.5 m', '140 keV', 'pi rad', '2 min') и автоматической конвертацией
    в целевую единицу измерения поля при нажатии Enter или потере фокуса.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        default_unit: Optional[str] = None,
    ) -> None:
        """Инициализирует спинбокс с поддержкой парсинга единиц измерения."""
        super().__init__(parent)
        self._explicit_unit: Optional[str] = None
        if default_unit is not None:
            self.set_default_unit(default_unit)

    @property
    def default_unit(self) -> Optional[str]:
        """Возвращает текущую целевую единицу измерения (явно заданную или выведенную из suffix)."""
        if self._explicit_unit is not None:
            return self._explicit_unit
        return infer_unit_from_suffix(self.suffix())

    def set_default_unit(self, unit: Optional[str]) -> None:
        """Устанавливает целевую единицу измерения для конвертации вводимых величин."""
        if unit is None or not unit.strip():
            self._explicit_unit = None
        else:
            self._explicit_unit = normalize_unit_name(unit)

    def _extract_user_input(self, raw_text: str) -> str:
        """Извлекает введенную пользователем часть строки без префикса и штатного суффикса спинбокса."""
        text_body = raw_text
        prefix_str = self.prefix()
        if prefix_str and text_body.startswith(prefix_str):
            text_body = text_body[len(prefix_str):]

        suffix_str = self.suffix()
        if suffix_str and text_body.endswith(suffix_str):
            candidate = text_body[:-len(suffix_str)]
            if candidate.strip():
                text_body = candidate

        return text_body.strip()

    def parse_text(self, raw_text: str) -> float:
        """
        Парсит текстовое выражение из поля спинбокса с учетом default_unit.
        Выбрасывает ValueError, если выражение некорректно или физическая размерность несовместима.
        """
        user_input = self._extract_user_input(raw_text)
        return parse_unit_value(
            user_input,
            default_unit=self.default_unit,
            to_hepunits=False,
        )

    def validate(self, input_text: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """
        Проверяет допустимость текущего текста при посимвольном наборе с клавиатуры.
        Разрешает промежуточный набор буквенных единиц измерения и выражений с pi.
        """
        user_input = self._extract_user_input(input_text)
        if not user_input or user_input in ("+", "-", ".", ",", "+.", "-.", "+,", "-,"):
            return (QValidator.State.Intermediate, input_text, pos)

        try:
            parsed_val = self.parse_text(input_text)
            if self.minimum() <= parsed_val <= self.maximum():
                return (QValidator.State.Acceptable, input_text, pos)
            return (QValidator.State.Intermediate, input_text, pos)
        except ValueError:
            if _INTERMEDIATE_SPINBOX_PATTERN.match(user_input):
                return (QValidator.State.Intermediate, input_text, pos)
            return (QValidator.State.Invalid, input_text, pos)

    def valueFromText(self, text: str) -> float:
        """Преобразует введенный текст (с возможной единицей измерения) в числовое значение спинбокса."""
        try:
            parsed_val = self.parse_text(text)
            return max(self.minimum(), min(self.maximum(), parsed_val))
        except ValueError:
            return float(self.value())

    def fixup(self, input_text: str) -> str:
        """Исправляет или откатывает текст при завершении ввода промежуточного значения."""
        try:
            parsed_val = self.parse_text(input_text)
            clamped = max(self.minimum(), min(self.maximum(), parsed_val))
            return self.textFromValue(clamped)
        except ValueError:
            return self.textFromValue(self.value())

    def _canonical_display_text(self) -> str:
        """Формирует каноническую строку отображения для текущего числового значения спинбокса."""
        return f"{self.prefix()}{self.textFromValue(self.value())}{self.suffix()}"

    def _commit_text_if_needed(self) -> bool:
        """
        Применяет введенное в lineEdit размерное или числовое выражение к значению спинбокса
        и нормализует отображаемый текст. При ошибке парсинга откатывает текст к текущему значению.
        """
        if self.isReadOnly():
            return False
        line_edit = self.lineEdit()
        if line_edit is None:
            return False

        current_text = line_edit.text()
        current_canonical = self._canonical_display_text()
        if current_text == current_canonical:
            return True

        try:
            parsed_val = self.parse_text(current_text)
        except ValueError:
            line_edit.setText(current_canonical)
            return False

        clamped_val = max(self.minimum(), min(self.maximum(), parsed_val))
        if abs(self.value() - clamped_val) > 1e-12:
            self.setValue(clamped_val)
        canonical_display = self._canonical_display_text()
        if line_edit.text() != canonical_display:
            line_edit.setText(canonical_display)
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Обрабатывает подтверждение размерного ввода по нажатию клавиши Enter / Return."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._commit_text_if_needed()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        """Применяет размерный ввод при потере фокуса спинбоксом."""
        self._commit_text_if_needed()
        super().focusOutEvent(event)

    def interpretText(self) -> None:
        """Интерпретирует текущий текст спинбокса с предварительным парсингом единиц."""
        self._commit_text_if_needed()
        super().interpretText()


class BindableDoubleField(QWidget):
    """
    Виджет числового поля ввода со встроенной кнопкой связывания (Property Binding)
    с переменными параметрического сканирования или протокола (UE-Style)
    и поддержкой парсинга размерных выражений.
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
        unit: Optional[str] = None,
    ) -> None:
        """Инициализирует составное числовое поле ввода с кнопкой привязки и поддержкой единиц измерения."""
        super().__init__(parent)
        self.bound_variable: Optional[str] = None
        self._variable_provider: Optional[Callable[[], List[str]]] = None
        self._variable_created_callback: Optional[VariableCreatedCallback] = None

        self._init_ui(min_val, max_val, step, suffix, decimals, unit)

    def _init_ui(
        self,
        min_val: float,
        max_val: float,
        step: float,
        suffix: str,
        decimals: int,
        unit: Optional[str],
    ) -> None:
        """Создает внутренние элементы управления и компоновку."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.spin_box = UnitAwareDoubleSpinBox(self, default_unit=unit)
        self.spin_box.setRange(min_val, max_val)
        self.spin_box.setSingleStep(step)
        self.spin_box.setDecimals(decimals)
        if suffix:
            self.spin_box.setSuffix(suffix)
        self.spin_box.valueChanged.connect(self.valueChanged.emit)

        self.btn_bind = QToolButton(self)
        self.btn_bind.setText("🔗")
        self.btn_bind.setToolTip("Связать свойство с переменной протокола сканирования")
        self.btn_bind.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_bind.setStyleSheet(DEFAULT_BIND_BUTTON_STYLE)
        self._menu = QMenu(self)
        self._menu.setStyleSheet(POPUP_MENU_STYLE)
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self.btn_bind.setMenu(self._menu)
        self._rebuild_menu()

        layout.addWidget(self.spin_box, 1)
        layout.addWidget(self.btn_bind, 0)

    @property
    def default_unit(self) -> Optional[str]:
        """Возвращает текущую целевую единицу измерения поля."""
        return self.spin_box.default_unit

    def unit(self) -> Optional[str]:
        """Возвращает текущую целевую единицу измерения поля."""
        return self.spin_box.default_unit

    def set_unit(self, unit: Optional[str]) -> None:
        """Задает целевую единицу измерения поля для автоматической конвертации ввода."""
        self.spin_box.set_default_unit(unit)

    def set_variable_provider(self, provider: Callable[[], List[str]]) -> None:
        """Регистрация функции-поставщика имен доступных переменных протокола."""
        self._variable_provider = provider

    def set_variable_created_callback(self, callback: VariableCreatedCallback) -> None:
        """Регистрация обработчика создания новой переменной в протоколе."""
        self._variable_created_callback = callback

    def _rebuild_menu(self) -> None:
        """Перестраивает выпадающее меню выбора переменной для привязки."""
        self._menu.clear()

        act_unbind = self._menu.addAction("Без привязки (статическое значение)")
        act_unbind.triggered.connect(self.unbind)
        self._menu.addSeparator()

        available_vars: List[str] = (
            self._variable_provider() if self._variable_provider is not None else []
        )

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
        """Запрашивает у пользователя имя новой переменной протокола через диалоговое окно."""
        var_name, ok = QInputDialog.getText(
            self,
            "Новая переменная протокола",
            "Введите имя переменной (например, table_z):",
            QLineEdit.EchoMode.Normal,
            "",
        )
        if ok and var_name.strip():
            cleaned_var = var_name.strip().replace(" ", "_")
            if self._variable_created_callback is not None:
                self._variable_created_callback(cleaned_var)
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

    def value(self) -> float:
        """Возвращает текущее числовое значение спинбокса."""
        return float(self.spin_box.value())

    def setValue(self, new_value: Union[int, float, str]) -> None:
        """Устанавливает новое значение спинбокса (поддерживает числа и размерные строки)."""
        if isinstance(new_value, str):
            parsed_val = parse_unit_value(
                new_value,
                default_unit=self.spin_box.default_unit,
                to_hepunits=False,
            )
            self.spin_box.setValue(float(parsed_val))
        else:
            self.spin_box.setValue(float(new_value))

    def setRange(self, min_val: float, max_val: float) -> None:
        """Задает допустимый диапазон значений спинбокса."""
        self.spin_box.setRange(float(min_val), float(max_val))

    def setMinimum(self, min_val: float) -> None:
        """Задает минимальное допустимое значение спинбокса."""
        self.spin_box.setMinimum(float(min_val))

    def setMaximum(self, max_val: float) -> None:
        """Задает максимальное допустимое значение спинбокса."""
        self.spin_box.setMaximum(float(max_val))

    def setSingleStep(self, step: float) -> None:
        """Задает шаг изменения значения спинбокса."""
        self.spin_box.setSingleStep(float(step))

    def setDecimals(self, decimals: int) -> None:
        """Задает количество знаков после запятой."""
        self.spin_box.setDecimals(int(decimals))

    def setSuffix(self, suffix: str) -> None:
        """Устанавливает текстовый суффикс спинбокса."""
        self.spin_box.setSuffix(suffix)

    def suffix(self) -> str:
        """Возвращает текущий текстовый суффикс спинбокса."""
        return self.spin_box.suffix()

    def blockSignals(self, is_blocked: bool) -> bool:
        """Блокирует или разблокирует сигналы внутреннего спинбокса."""
        return self.spin_box.blockSignals(is_blocked)

    def setEnabled(self, enabled: bool) -> None:
        """Включает или отключает поле ввода и кнопку привязки."""
        super().setEnabled(enabled)
        self.spin_box.setEnabled(enabled)
        self.btn_bind.setEnabled(enabled)

    def isReadOnly(self) -> bool:
        """Проверяет, находится ли спинбокс в режиме только для чтения."""
        return self.spin_box.isReadOnly()

    def setReadOnly(self, is_read_only: bool) -> None:
        """Переводит спинбокс в режим только для чтения с учетом активной привязки."""
        if self.bound_variable is not None:
            self.spin_box.setReadOnly(True)
        else:
            self.spin_box.setReadOnly(is_read_only)

    def setToolTip(self, tooltip_text: str) -> None:
        """Устанавливает всплывающую подсказку для виджета и спинбокса."""
        super().setToolTip(tooltip_text)
        self.spin_box.setToolTip(tooltip_text)


__all__ = [
    "BindableDoubleField",
    "UnitAwareDoubleSpinBox",
    "infer_unit_from_suffix",
]


