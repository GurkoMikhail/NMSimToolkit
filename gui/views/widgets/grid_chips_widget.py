"""
Модульный клавиатурный чипс-виджет для Grid-переменных протокола (Keyboard-Driven Chips).
Обеспечивает компактное визуальное представление дискретных значений сетки декартова произведения,
быструю навигацию клавишами стрелок, удаление клавишами Backspace/Delete, инлайн-редактирование,
пакетный ввод чисел через запятую и диалог генератора диапазонов.
"""

from typing import Any, Dict, List, Optional, cast
import numpy as np

from PySide6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
    QObject,
)
from PySide6.QtGui import (
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from core.config.units import (
    normalize_unit_name,
    parse_unit_tokens,
    parse_unit_value,
)
from gui.views.widgets.bindable_field import UnitAwareDoubleSpinBox

KNOWN_VARIABLE_DEFAULT_UNITS: Dict[str, str] = {
    "gantry_angle": "deg",
    "current_angle": "deg",
    "start_angle": "deg",
    "end_angle": "deg",
    "radius": "mm",
    "ring_radius": "mm",
    "step_time": "s",
    "start_time": "s",
    "stop_time": "s",
    "current_time": "s",
    "time_per_step": "s",
    "time_per_frame": "s",
    "frame_time": "s",
    "initial_time": "s",
    "frame_duration": "s",
    "activity": "MBq",
    "min_energy": "keV",
}


def infer_variable_default_unit(variable_name: str) -> Optional[str]:
    """Определяет единицу измерения по умолчанию для известных переменных протокола сканирования."""
    cleaned_name = str(variable_name).strip()
    return KNOWN_VARIABLE_DEFAULT_UNITS.get(cleaned_name)

# Стили оформления заголовка и кнопок
TITLE_LABEL_STYLE = "color: #a0c0ff; font-weight: bold;"
DELETE_BUTTON_STYLE = "color: #ff9090; font-weight: bold; padding: 2px 6px;"
GENERATOR_BUTTON_STYLE = "padding: 2px 8px;"
CHIP_CONTAINER_STYLE = "background-color: transparent;"
INPUT_ADD_VALUE_NORMAL_STYLE = (
    "QLineEdit#GridAddValueInput {"
    "  background-color: #212429;"
    "  color: #e2e8f0;"
    "  border: 1px dashed #4b5263;"
    "  border-radius: 4px;"
    "  padding: 1px 6px;"
    "  font-size: 11px;"
    "}"
    "QLineEdit#GridAddValueInput:focus {"
    "  border: 1px solid #528bff;"
    "  background-color: #282c34;"
    "}"
)
INPUT_ADD_VALUE_ERROR_STYLE = (
    "QLineEdit#GridAddValueInput {"
    "  background-color: #3b2222;"
    "  color: #ff9999;"
    "  border: 1.5px solid #ff5555;"
    "  border-radius: 4px;"
    "  padding: 1px 6px;"
    "  font-size: 11px;"
    "}"
)


class FlowLayout(QLayout):
    """
    Потоковый макет (Flow Layout), переносящий дочерние элементы на следующую строку
    при превышении доступной ширины контейнера.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        margin: int = 0,
        horizontal_spacing: int = 4,
        vertical_spacing: int = 4,
    ) -> None:
        super().__init__(parent)
        self._item_list: List[QLayoutItem] = []
        self._horizontal_spacing: int = horizontal_spacing
        self._vertical_spacing: int = vertical_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self) -> None:
        item = self.takeAt(0)
        while item is not None:
            item = self.takeAt(0)

    def addItem(self, item: QLayoutItem) -> None:
        self._item_list.append(item)

    def count(self) -> int:
        return len(self._item_list)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        """Возвращает направления расширения макета."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        """Указывает, что высота макета зависит от его ширины."""
        return True

    def heightForWidth(self, width: int) -> int:
        """Вычисляет необходимую высоту макета для заданной ширины."""
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        """Устанавливает геометрию дочерних элементов макета."""
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        """Возвращает рекомендуемый размер макета."""
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        """Вычисляет минимальный допустимый размер макета."""
        calculated_size = QSize()
        for layout_item in self._item_list:
            calculated_size = calculated_size.expandedTo(layout_item.minimumSize())
        margins = self.contentsMargins()
        calculated_size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return calculated_size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        """Распределяет элементы макета по строкам с переносом по ширине."""
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        current_x = effective_rect.x()
        current_y = effective_rect.y()
        line_height = 0

        for layout_item in self._item_list:
            spacing_x = self._horizontal_spacing
            spacing_y = self._vertical_spacing
            item_size_hint = layout_item.sizeHint()

            next_x = current_x + item_size_hint.width() + spacing_x
            if next_x - spacing_x > effective_rect.right() and line_height > 0:
                current_x = effective_rect.x()
                current_y = current_y + line_height + spacing_y
                next_x = current_x + item_size_hint.width() + spacing_x
                line_height = 0

            if not test_only:
                layout_item.setGeometry(QRect(QPoint(current_x, current_y), item_size_hint))

            current_x = next_x
            line_height = max(line_height, item_size_hint.height())

        return current_y + line_height - rect.y() + margins.bottom()


class _ChipLineEdit(QLineEdit):
    """Внутреннее поле ввода для инлайн-редактирования значения внутри чипса."""

    editing_committed = Signal(float, bool)
    editing_cancelled = Signal(bool)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        default_unit: Optional[str] = None,
    ) -> None:
        """Инициализирует внутреннее поле ввода чипса с поддержкой парсинга единиц измерения."""
        super().__init__(parent)
        self._default_unit: Optional[str] = normalize_unit_name(default_unit) if default_unit else None
        self.setObjectName("GridChipEdit")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "QLineEdit#GridChipEdit {"
            "  background-color: #1a1d21;"
            "  color: #ffffff;"
            "  border: 1px solid #528bff;"
            "  border-radius: 3px;"
            "  font-size: 11px;"
            "  padding: 0px 4px;"
            "  min-width: 48px;"
            "  min-height: 18px;"
            "}"
        )

    def set_default_unit(self, unit: Optional[str]) -> None:
        """Задает целевую единицу измерения для парсинга вводимого текста."""
        self._default_unit = normalize_unit_name(unit) if unit and unit.strip() else None

    def keyPressEvent(self, key_event: QKeyEvent) -> None:
        """Обрабатывает нажатия клавиш Enter и Escape при инлайн-редактировании."""
        if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._commit_edit(refocus=True)
            key_event.accept()
            return
        if key_event.key() == Qt.Key.Key_Escape:
            self.editing_cancelled.emit(True)
            key_event.accept()
            return
        super().keyPressEvent(key_event)

    def focusOutEvent(self, focus_event: QFocusEvent) -> None:
        """Применяет изменения при потере фокуса полем ввода."""
        super().focusOutEvent(focus_event)
        self._commit_edit(refocus=False)

    def _commit_edit(self, refocus: bool = False) -> None:
        """Парсит введенное число или размерное выражение и испускает сигнал завершения или отмены."""
        raw_text = self.text().strip()
        try:
            parsed_number = parse_unit_value(
                raw_text,
                default_unit=self._default_unit,
                to_hepunits=False,
            )
            self.editing_committed.emit(parsed_number, refocus)
        except ValueError:
            self.editing_cancelled.emit(refocus)


class GridChip(QFrame):
    """
    Ультракомпактная капсула значения переменной сетки без крестика ×.
    Поддерживает:
      - Выделение фокусом (клик мыши или клавиатура);
      - Навигацию стрелками Left/Right;
      - Удаление клавишами Delete и Backspace с переносом фокуса;
      - Инлайн-редактирование по Enter или двойному клику мыши с конвертацией единиц.
    """

    value_changed = Signal(float)
    delete_requested = Signal(int)
    navigate_requested = Signal(int)
    clicked = Signal()

    def __init__(
        self,
        value: float,
        parent: Optional[QWidget] = None,
        default_unit: Optional[str] = None,
    ) -> None:
        """Инициализирует капсулу чипса с указанным числовым значением и целевой единицей."""
        super().__init__(parent)
        self._value: float = float(value)
        self._default_unit: Optional[str] = (
            normalize_unit_name(default_unit) if default_unit and default_unit.strip() else None
        )
        self.setObjectName("GridChip")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._stacked_layout = QStackedLayout(self)
        self._stacked_layout.setContentsMargins(0, 0, 0, 0)

        self._label_display = QLabel(self._format_value(self._value))
        self._label_display.setObjectName("GridChipLabel")
        self._label_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_display.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._stacked_layout.addWidget(self._label_display)

        self._edit_input = _ChipLineEdit(default_unit=self._default_unit)
        self._edit_input.editing_committed.connect(self._on_edit_committed)
        self._edit_input.editing_cancelled.connect(self._on_edit_cancelled)
        self._stacked_layout.addWidget(self._edit_input)

        self._apply_styles()

    @property
    def default_unit(self) -> Optional[str]:
        """Возвращает целевую единицу измерения чипса."""
        return self._default_unit

    def set_default_unit(self, unit: Optional[str]) -> None:
        """Устанавливает целевую единицу измерения чипса."""
        self._default_unit = normalize_unit_name(unit) if unit and unit.strip() else None
        self._edit_input.set_default_unit(self._default_unit)

    def _apply_styles(self) -> None:
        """Применяет табличные CSS-стили оформления чипса."""
        self.setStyleSheet(
            "QFrame#GridChip {"
            "  background-color: #2b2d30;"
            "  border: 1px solid #434852;"
            "  border-radius: 4px;"
            "  padding: 2px 6px;"
            "  min-height: 20px;"
            "  max-height: 20px;"
            "}"
            "QFrame#GridChip:hover {"
            "  border: 1px solid #5c6370;"
            "  background-color: #32363d;"
            "}"
            "QFrame#GridChip:focus {"
            "  border: 1.5px solid #528bff;"
            "  background-color: #2e3646;"
            "}"
            "QLabel#GridChipLabel {"
            "  color: #e2e8f0;"
            "  font-size: 11px;"
            "  font-weight: 500;"
            "}"
        )

    @staticmethod
    def _format_value(numeric_value: float) -> str:
        """Компактное форматирование вещественного числа с сохранением точности без лишних нулей."""
        float_value = float(numeric_value)
        if abs(float_value) < 1e-12:
            return "0"
        return f"{float_value:.8g}"

    @property
    def value(self) -> float:
        """Текущее числовое значение чипса."""
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        self._value = float(new_value)
        self._label_display.setText(self._format_value(self._value))

    def start_editing(self) -> None:
        """Переход в режим инлайн-редактирования числа."""
        self._stacked_layout.setCurrentIndex(1)
        self._edit_input.setText(self._format_value(self._value))
        self._edit_input.selectAll()
        self._edit_input.setFocus()

    def _on_edit_committed(self, new_number: float, refocus: bool = True) -> None:
        """Применение отредактированного значения."""
        old_value = self._value
        self._value = float(new_number)
        self._label_display.setText(self._format_value(self._value))
        self._stacked_layout.setCurrentIndex(0)
        if refocus:
            self.setFocus()
        if abs(old_value - self._value) > 1e-9:
            self.value_changed.emit(self._value)

    def _on_edit_cancelled(self, refocus: bool = True) -> None:
        """Отмена редактирования без изменений."""
        self._stacked_layout.setCurrentIndex(0)
        if refocus:
            self.setFocus()

    def mousePressEvent(self, mouse_event: QMouseEvent) -> None:
        """Переводит фокус на чипс при клике левой кнопкой мыши."""
        if mouse_event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.clicked.emit()
        super().mousePressEvent(mouse_event)

    def mouseDoubleClickEvent(self, mouse_event: QMouseEvent) -> None:
        """Открывает режим редактирования при двойном клике левой кнопкой мыши."""
        if mouse_event.button() == Qt.MouseButton.LeftButton:
            self.start_editing()
            mouse_event.accept()
            return
        super().mouseDoubleClickEvent(mouse_event)

    def keyPressEvent(self, key_event: QKeyEvent) -> None:
        """Обрабатывает клавиатурную навигацию, удаление и вход в редактирование."""
        if key_event.key() == Qt.Key.Key_Backspace:
            self.delete_requested.emit(-1)
            key_event.accept()
            return
        if key_event.key() == Qt.Key.Key_Delete:
            self.delete_requested.emit(0)
            key_event.accept()
            return
        if key_event.key() == Qt.Key.Key_Left:
            self.navigate_requested.emit(-1)
            key_event.accept()
            return
        if key_event.key() == Qt.Key.Key_Right:
            self.navigate_requested.emit(1)
            key_event.accept()
            return
        if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.start_editing()
            key_event.accept()
            return
        super().keyPressEvent(key_event)


class GridRangeDialog(QDialog):
    """
    Диалог-генератор диапазона сетки значений.
    Позволяет сгенерировать последовательность дискретных точек
    по заданному шагу или по заданному количеству точек с предпросмотром.
    """

    def __init__(
        self,
        variable_name: str,
        parent: Optional[QWidget] = None,
        default_unit: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Генератор сетки: {variable_name}")
        self.setModal(True)
        self.resize(360, 260)
        resolved_unit = (
            normalize_unit_name(default_unit)
            if default_unit and default_unit.strip()
            else infer_variable_default_unit(variable_name)
        )
        self._default_unit: Optional[str] = resolved_unit
        self._generated_values: List[float] = []

        self._init_ui()
        self._update_preview()

    @property
    def default_unit(self) -> Optional[str]:
        """Целевая единица измерения генератора диапазона."""
        return self._default_unit

    def set_default_unit(self, unit: Optional[str]) -> None:
        """Устанавливает целевую единицу измерения для полей генератора диапазона."""
        self._default_unit = normalize_unit_name(unit) if unit and unit.strip() else None
        self.spin_start.set_default_unit(self._default_unit)
        self.spin_stop.set_default_unit(self._default_unit)
        self.spin_step.set_default_unit(self._default_unit)

    def _init_ui(self) -> None:
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setSpacing(8)

        form_layout = QFormLayout()
        form_layout.setSpacing(6)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("По шагу (Start, Stop, Step)", "step")
        self.combo_mode.addItem("По количеству точек (Start, Stop, Count)", "count")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        form_layout.addRow("Режим:", self.combo_mode)

        self.spin_start = UnitAwareDoubleSpinBox(self, default_unit=self._default_unit)
        self.spin_start.setRange(-100000.0, 100000.0)
        self.spin_start.setDecimals(4)
        self.spin_start.setValue(0.0)
        self.spin_start.valueChanged.connect(self._update_preview)
        form_layout.addRow("Старт:", self.spin_start)

        self.spin_stop = UnitAwareDoubleSpinBox(self, default_unit=self._default_unit)
        self.spin_stop.setRange(-100000.0, 100000.0)
        self.spin_stop.setDecimals(4)
        self.spin_stop.setValue(180.0)
        self.spin_stop.valueChanged.connect(self._update_preview)
        form_layout.addRow("Стоп:", self.spin_stop)

        self.spin_step = UnitAwareDoubleSpinBox(self, default_unit=self._default_unit)
        self.spin_step.setRange(0.0001, 100000.0)
        self.spin_step.setDecimals(4)
        self.spin_step.setValue(10.0)
        self.spin_step.valueChanged.connect(self._update_preview)
        self.lbl_step_row = QLabel("Шаг:")
        form_layout.addRow(self.lbl_step_row, self.spin_step)

        self.spin_count = QSpinBox()
        self.spin_count.setRange(2, 100000)
        self.spin_count.setValue(10)
        self.spin_count.valueChanged.connect(self._update_preview)
        self.lbl_count_row = QLabel("Количество точек:")
        form_layout.addRow(self.lbl_count_row, self.spin_count)

        self.chk_endpoint = QCheckBox("Включая конечное значение (Stop)")
        self.chk_endpoint.setChecked(True)
        self.chk_endpoint.toggled.connect(self._update_preview)
        form_layout.addRow("", self.chk_endpoint)

        self.lbl_preview_title = QLabel("Предпросмотр:")
        self.lbl_preview_text = QLabel("...")
        self.lbl_preview_text.setWordWrap(True)
        self.lbl_preview_text.setStyleSheet("color: #a0c0ff; font-family: monospace; font-size: 11px;")
        form_layout.addRow(self.lbl_preview_title, self.lbl_preview_text)

        dialog_layout.addLayout(form_layout)

        # Кнопки диалога
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        apply_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if apply_button is not None:
            apply_button.setText("Применить")
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button is not None:
            cancel_button.setText("Отмена")

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        dialog_layout.addWidget(button_box)

        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        """Переключает видимость полей шага и числа точек при смене режима генерации."""
        is_step_mode = (self.combo_mode.currentData() == "step")
        self.lbl_step_row.setVisible(is_step_mode)
        self.spin_step.setVisible(is_step_mode)
        self.lbl_count_row.setVisible(not is_step_mode)
        self.spin_count.setVisible(not is_step_mode)
        self._update_preview()

    def _update_preview(self) -> None:
        """Пересчитывает значения диапазона и обновляет текстовый предпросмотр."""
        start_value = float(self.spin_start.value())
        stop_value = float(self.spin_stop.value())
        endpoint_included = bool(self.chk_endpoint.isChecked())
        mode_key = str(self.combo_mode.currentData())

        if mode_key == "count":
            total_points = max(2, int(self.spin_count.value()))
            generated_list = [
                float(point_value)
                for point_value in np.linspace(start_value, stop_value, total_points, endpoint=endpoint_included)
            ]
        else:
            step_value = max(0.0001, float(self.spin_step.value()))
            distance = stop_value - start_value
            if abs(distance) < 1e-9:
                generated_list = [start_value]
            else:
                direction_sign = 1.0 if distance > 0 else -1.0
                number_of_steps = int(np.floor(abs(distance) / step_value + 1e-9))
                generated_list = [
                    round(start_value + direction_sign * step_index * step_value, 9)
                    for step_index in range(number_of_steps + 1)
                ]
                if endpoint_included:
                    if abs(generated_list[-1] - stop_value) > 1e-6:
                        if (direction_sign > 0 and generated_list[-1] < stop_value) or (
                            direction_sign < 0 and generated_list[-1] > stop_value
                        ):
                            generated_list.append(stop_value)
                else:
                    if abs(generated_list[-1] - stop_value) < 1e-6:
                        generated_list.pop()

        self._generated_values = generated_list
        total_generated_count = len(generated_list)

        if total_generated_count <= 8:
            formatted_elements = ", ".join(
                GridChip._format_value(element_value) for element_value in generated_list
            )
        else:
            first_elements = ", ".join(
                GridChip._format_value(element_value) for element_value in generated_list[:4]
            )
            last_elements = ", ".join(
                GridChip._format_value(element_value) for element_value in generated_list[-2:]
            )
            formatted_elements = f"{first_elements}, ..., {last_elements}"

        self.lbl_preview_text.setText(f"[{formatted_elements}] (всего: {total_generated_count} знач.)")

    def get_generated_values(self) -> List[float]:
        """Возвращает полученный список вещественных значений сетки."""
        return list(self._generated_values)


class GridVariableChipsRow(QWidget):
    """
    Виджет-строка для управления переменной декартовой сетки (Grid Variable) на чипсах.
    Включает:
      - Заголовок с именем и счетчиком значений;
      - Контейнер ультракомпактных чипсов со свободным горизонтальным переносом;
      - Инлайн-поле ввода [+ значение] для одиночного или пакетного ввода с поддержкой единиц измерения;
      - Кнопку [⚙ Сетка] генератора диапазона значений;
      - Кнопку [🗑] удаления всей переменной.
    """

    values_changed = Signal(str, list)
    variable_deleted = Signal(str)

    def __init__(
        self,
        var_name: str,
        values: List[float],
        parent: Optional[QWidget] = None,
        default_unit: Optional[str] = None,
    ) -> None:
        """Инициализирует строку чипсов для указанной Grid-переменной."""
        super().__init__(parent)
        self.var_name: str = str(var_name)
        resolved_unit = (
            normalize_unit_name(default_unit)
            if default_unit and default_unit.strip()
            else infer_variable_default_unit(self.var_name)
        )
        self._default_unit: Optional[str] = resolved_unit
        self._values: List[float] = [float(number_element) for number_element in values] if values else [0.0]
        self._chips: List[GridChip] = []
        self._is_updating: bool = False

        self._init_ui()
        self._rebuild_chips()

    @property
    def default_unit(self) -> Optional[str]:
        """Возвращает целевую единицу измерения для данной переменной сетки."""
        return self._default_unit

    def set_default_unit(self, unit: Optional[str]) -> None:
        """Устанавливает целевую единицу измерения для строки и всех ее чипсов."""
        self._default_unit = normalize_unit_name(unit) if unit and unit.strip() else None
        for chip_item in self._chips:
            chip_item.set_default_unit(self._default_unit)

    def _init_ui(self) -> None:
        """Создает элементы интерфейса строки управления Grid-переменной."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(4)

        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"<b>{self.var_name}</b> ({len(self._values)} знач.)")
        self.lbl_title.setStyleSheet(TITLE_LABEL_STYLE)

        self.btn_grid_generator = QPushButton("⚙ Сетка")
        self.btn_grid_generator.setStyleSheet(GENERATOR_BUTTON_STYLE)
        self.btn_grid_generator.setToolTip("Генератор диапазона сетки (старт, стоп, шаг или количество точек)")
        self.btn_grid_generator.clicked.connect(self._on_grid_generator_clicked)

        self.btn_delete_var = QPushButton("🗑")
        self.btn_delete_var.setStyleSheet(DELETE_BUTTON_STYLE)
        self.btn_delete_var.setToolTip(f"Удалить Grid-переменную '{self.var_name}'")
        self.btn_delete_var.clicked.connect(self._on_delete_clicked)

        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_grid_generator)
        header_layout.addWidget(self.btn_delete_var)
        main_layout.addLayout(header_layout)

        self.chips_container = QWidget(self)
        self.chips_container.setStyleSheet(CHIP_CONTAINER_STYLE)
        self.chips_layout = FlowLayout(self.chips_container, margin=0, horizontal_spacing=4, vertical_spacing=4)

        self.input_add_value = QLineEdit(self.chips_container)
        self.input_add_value.setObjectName("GridAddValueInput")
        self.input_add_value.setPlaceholderText("+ значение")
        self.input_add_value.setToolTip(
            "Введите одно или несколько значений через запятую (поддерживаются единицы измерения: 25 cm, 140 keV, pi rad)"
        )
        self.input_add_value.setMinimumWidth(75)
        self.input_add_value.setMaximumWidth(150)
        self.input_add_value.setFixedHeight(22)
        self.input_add_value.setStyleSheet(INPUT_ADD_VALUE_NORMAL_STYLE)
        self.input_add_value.returnPressed.connect(self._on_add_input_entered)
        self.input_add_value.textChanged.connect(self._on_add_input_text_changed)
        self.input_add_value.installEventFilter(self)

        main_layout.addWidget(self.chips_container)

    @property
    def values(self) -> List[float]:
        """Текущий список числовых значений переменной."""
        return list(self._values)

    def count(self) -> int:
        """Количество значений переменной."""
        return len(self._values)

    def __len__(self) -> int:
        """Возвращает длину списка значений переменной."""
        return len(self._values)

    @property
    def table(self) -> Any:
        """Прокси-свойство для совместимости с кодом проверки табличных представлений."""
        class _TableProxy:
            def __init__(self, owner: Any) -> None:
                self._owner = owner

            def rowCount(self) -> int:
                return len(self._owner._values)

        return _TableProxy(self)

    def set_values(self, values: List[float]) -> None:
        """Обновление списка значений сетки с перестроением чипсов."""
        validated_values = [float(number_element) for number_element in values] if values else [0.0]
        if len(validated_values) == len(self._values) and all(
            abs(element_alpha - element_beta) < 1e-7 for element_alpha, element_beta in zip(validated_values, self._values)
        ):
            return
        self._values = validated_values
        self._rebuild_chips()

    def _rebuild_chips(self) -> None:
        """Пересоздает чипсы значений в потоковом макете."""
        self._is_updating = True
        try:
            self.chips_layout.removeWidget(self.input_add_value)

            for existing_chip in list(self._chips):
                self.chips_layout.removeWidget(existing_chip)
                existing_chip.deleteLater()
            self._chips.clear()

            for number_value in self._values:
                chip_widget = GridChip(
                    number_value,
                    parent=self.chips_container,
                    default_unit=self._default_unit,
                )
                chip_widget.delete_requested.connect(self._create_delete_handler(chip_widget))
                chip_widget.navigate_requested.connect(self._create_navigate_handler(chip_widget))
                chip_widget.value_changed.connect(self._create_value_changed_handler(chip_widget))
                self.chips_layout.addWidget(chip_widget)
                self._chips.append(chip_widget)

            self.chips_layout.addWidget(self.input_add_value)
            self.lbl_title.setText(f"<b>{self.var_name}</b> ({len(self._values)} знач.)")
        finally:
            self._is_updating = False

    def _create_delete_handler(self, target_chip: GridChip) -> Any:
        """Создает замыкание обработчика удаления для конкретного чипса."""
        def handler(direction_hint: int = 0) -> None:
            self._on_chip_delete_requested(target_chip, direction_hint)
        return handler

    def _create_navigate_handler(self, target_chip: GridChip) -> Any:
        """Создает замыкание обработчика навигации для конкретного чипса."""
        def handler(direction_offset: int) -> None:
            self._on_chip_navigate_requested(target_chip, direction_offset)
        return handler

    def _create_value_changed_handler(self, target_chip: GridChip) -> Any:
        """Создает замыкание обработчика изменения числа для конкретного чипса."""
        def handler(new_number: float) -> None:
            self._on_chip_value_changed(target_chip, new_number)
        return handler

    def _on_chip_delete_requested(self, target_chip: GridChip, direction_hint: int = 0) -> None:
        """Удаляет выбранный чипс и переносит фокус на соседний элемент."""
        if target_chip not in self._chips:
            return
        if len(self._chips) <= 1:
            QMessageBox.information(
                self,
                "Удаление значения",
                "Переменная сетки должна содержать как минимум одно значение.",
            )
            return

        chip_index = self._chips.index(target_chip)
        self._values.pop(chip_index)
        self._rebuild_chips()

        # Перенос фокуса на соседний чипс с учетом направления клавиши удаления
        if self._chips:
            if direction_hint < 0:
                target_focus_index = max(0, chip_index - 1)
            else:
                target_focus_index = min(chip_index, len(self._chips) - 1)
            self._chips[target_focus_index].setFocus()
        else:
            self.input_add_value.setFocus()

        self.values_changed.emit(self.var_name, list(self._values))

    def _on_chip_navigate_requested(self, target_chip: GridChip, direction_offset: int) -> None:
        """Перемещает клавиатурный фокус между чипсами и полем добавления."""
        if target_chip not in self._chips:
            return
        current_index = self._chips.index(target_chip)
        target_index = current_index + direction_offset
        if direction_offset < 0:
            if target_index >= 0:
                self._chips[target_index].setFocus()
        else:
            if target_index < len(self._chips):
                self._chips[target_index].setFocus()
            elif target_index == len(self._chips):
                self.input_add_value.setFocus()

    def _on_chip_value_changed(self, target_chip: GridChip, new_number: float) -> None:
        """Обновляет значение чипса в списке и уведомляет подписчиков."""
        if target_chip not in self._chips:
            return
        chip_index = self._chips.index(target_chip)
        self._values[chip_index] = float(new_number)
        self.lbl_title.setText(f"<b>{self.var_name}</b> ({len(self._values)} знач.)")
        self.values_changed.emit(self.var_name, list(self._values))

    def _on_add_input_text_changed(self) -> None:
        """Сброс оформления ошибки при начале нового ввода пользователем."""
        self.input_add_value.setStyleSheet(INPUT_ADD_VALUE_NORMAL_STYLE)

    def _on_add_input_entered(self) -> None:
        """Добавляет одно или несколько значений (с поддержкой единиц измерения), введенных в текстовое поле."""
        raw_text = self.input_add_value.text().strip()
        if not raw_text:
            return

        try:
            new_numbers = parse_unit_tokens(
                raw_text,
                default_unit=self._default_unit,
                to_hepunits=False,
            )
        except ValueError:
            self.input_add_value.setStyleSheet(INPUT_ADD_VALUE_ERROR_STYLE)
            return

        self.input_add_value.setStyleSheet(INPUT_ADD_VALUE_NORMAL_STYLE)
        self.input_add_value.clear()
        self._values.extend(new_numbers)
        self._rebuild_chips()
        self.values_changed.emit(self.var_name, list(self._values))
        self.input_add_value.setFocus()

    def _on_grid_generator_clicked(self) -> None:
        """Открывает диалог генератора диапазона значений сетки."""
        generator_dialog = GridRangeDialog(
            self.var_name,
            parent=self,
            default_unit=self._default_unit,
        )
        if generator_dialog.exec() == QDialog.DialogCode.Accepted:
            generated_values = generator_dialog.get_generated_values()
            if generated_values:
                self.set_values(generated_values)
                self.values_changed.emit(self.var_name, list(self._values))

    def _on_delete_clicked(self) -> None:
        """Запрашивает подтверждение и испускает сигнал удаления переменной."""
        user_choice = QMessageBox.question(
            self,
            "Удаление переменной",
            f"Вы уверены, что хотите удалить Grid-переменную '{self.var_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if user_choice == QMessageBox.StandardButton.Yes:
            self.variable_deleted.emit(self.var_name)

    def eventFilter(self, watched_object: QObject, watched_event: QEvent) -> bool:
        """Перехватывает нажатия клавиш в поле добавления значений для навигации к чипсам."""
        if watched_object is self.input_add_value and watched_event.type() == QEvent.Type.KeyPress:
            key_event = cast(QKeyEvent, watched_event)
            if key_event.key() == Qt.Key.Key_Escape:
                self.input_add_value.clear()
                self._on_add_input_text_changed()
                return True
            if key_event.key() == Qt.Key.Key_Left and self.input_add_value.cursorPosition() == 0:
                if self._chips:
                    self._chips[-1].setFocus()
                    return True
            elif key_event.key() == Qt.Key.Key_Backspace and not self.input_add_value.text():
                if self._chips:
                    self._chips[-1].setFocus()
                    return True
        return super().eventFilter(watched_object, watched_event)


__all__ = [
    "FlowLayout",
    "GridChip",
    "GridRangeDialog",
    "GridVariableChipsRow",
    "KNOWN_VARIABLE_DEFAULT_UNITS",
    "infer_variable_default_unit",
]

