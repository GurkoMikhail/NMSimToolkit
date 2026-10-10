"""
Виджет для редактирования отдельных одномерных сеток декартова произведения (Grid Variables, N×1).
Предоставляет графический интерфейс для добавления, изменения и удаления дискретных значений переменной.
"""

from typing import List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)

# Стили для элементов заголовка виджета переменной сетки
GRID_HEADER_TITLE_STYLE = "color: #a0c0ff;"
GRID_DELETE_BUTTON_STYLE = "color: #ff9090;"


class GridVariableWidget(QWidget):
    """
    Виджет расширяемой таблицы N×1 для отдельной переменной сетки (Grid Variable).
    Позволяет добавлять, редактировать и удалять конкретные дискретные значения переменной.
    """

    values_changed = Signal(str, list)   # (var_name, list_of_floats)
    variable_deleted = Signal(str)       # (var_name)

    def __init__(
        self,
        var_name: str,
        values: List[float],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.var_name: str = str(var_name)
        self._values: List[float] = [float(val) for val in values] if values else [0.0]
        self._is_updating: bool = False

        self._init_ui()
        self._populate_table()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(4)

        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"<b>{self.var_name}</b> (Grid: {len(self._values)} знач.)")
        self.lbl_title.setStyleSheet(GRID_HEADER_TITLE_STYLE)

        self.btn_delete_var = QPushButton("🗑 Удалить сетку")
        self.btn_delete_var.setStyleSheet(GRID_DELETE_BUTTON_STYLE)
        self.btn_delete_var.clicked.connect(self._on_delete_clicked)

        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_delete_var)
        main_layout.addLayout(header_layout)

        self.table = QTableWidget(len(self._values), 1)
        self.table.setHorizontalHeaderLabels(["Значение"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMinimumHeight(80)
        self.table.setMaximumHeight(150)
        self.table.cellChanged.connect(self._on_cell_changed)
        main_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add_val = QPushButton("➕ Добавить строку")
        self.btn_add_val.clicked.connect(self._on_add_value_clicked)

        self.btn_remove_val = QPushButton("➖ Удалить строку")
        self.btn_remove_val.clicked.connect(self._on_remove_value_clicked)

        btn_layout.addWidget(self.btn_add_val)
        btn_layout.addWidget(self.btn_remove_val)
        main_layout.addLayout(btn_layout)

    def set_values(self, values: List[float]) -> None:
        """Обновление списка значений сетки без пересоздания виджета."""
        new_values = [float(val) for val in values] if values else [0.0]
        if len(new_values) == len(self._values) and all(
            abs(val_a - val_b) < 1e-7 for val_a, val_b in zip(new_values, self._values)
        ):
            return
        self._values = new_values
        self._populate_table()

    def _populate_table(self) -> None:
        self._is_updating = True
        try:
            self.table.setRowCount(len(self._values))
            for row_index, cell_value in enumerate(self._values):
                table_item = QTableWidgetItem(f"{float(cell_value):.4f}")
                table_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_index, 0, table_item)
            self.lbl_title.setText(f"<b>{self.var_name}</b> (Grid: {len(self._values)} знач.)")
        finally:
            self._is_updating = False

    def _on_cell_changed(self, row_index: int, column_index: int) -> None:
        if self._is_updating or column_index != 0:
            return
        table_item = self.table.item(row_index, column_index)
        if table_item is None:
            return
        try:
            parsed_value = float(table_item.text().strip())
        except ValueError:
            fallback_value = self._values[row_index] if 0 <= row_index < len(self._values) else 0.0
            table_item.setText(f"{fallback_value:.4f}")
            return
        if 0 <= row_index < len(self._values):
            self._values[row_index] = parsed_value
        self.values_changed.emit(self.var_name, list(self._values))

    def _on_add_value_clicked(self) -> None:
        last_value = self._values[-1] if self._values else 0.0
        self._values.append(last_value)
        self._populate_table()
        self.values_changed.emit(self.var_name, list(self._values))

    def _on_remove_value_clicked(self) -> None:
        if len(self._values) <= 1:
            QMessageBox.information(self, "Удаление", "Сетка должна содержать как минимум одно значение.")
            return
        target_row = self.table.currentRow()
        if target_row < 0 or target_row >= len(self._values):
            target_row = len(self._values) - 1
        self._values.pop(target_row)
        self._populate_table()
        self.values_changed.emit(self.var_name, list(self._values))

    def _on_delete_clicked(self) -> None:
        user_choice = QMessageBox.question(
            self,
            "Удаление переменной",
            f"Вы уверены, что хотите удалить Grid-переменную '{self.var_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if user_choice == QMessageBox.Yes:
            self.variable_deleted.emit(self.var_name)


__all__ = ["GridVariableWidget"]
