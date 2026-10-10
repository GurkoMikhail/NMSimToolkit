"""
Тесты стандартных переменных протокола и клавиатурного чипс-виджета GridVariableChipsRow.
Проверяют:
1. Защиту стандартных переменных протокола (gantry_angle, step_time) в StepAndShootProcedureViewModel и PropertyInspector:
   - Флаги Read-Only ячеек (без Qt.ItemIsEditable);
   - Заголовки с замком 🔒 и всплывающими подсказками;
   - Запрет удаления стандартных колонок.
2. Работу GridChip и GridVariableChipsRow:
   - Фокусировку, навигацию Left/Right, удаление Delete/Backspace;
   - Инлайн-редактирование числа;
   - Пакетное добавление чисел через запятую в инлайн-поле;
   - Генератор диапазона сетки GridRangeDialog;
   - Удаление переменной и совместимость прокси-свойства table.rowCount().
"""

import unittest
from unittest.mock import patch
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    StepAndShootProcedureViewModel,
)
from gui.views.property_inspector import PropertyInspector
from gui.views.widgets.grid_chips_widget import (
    GridChip,
    GridRangeDialog,
    GridVariableChipsRow,
)


class TestProtocolStandardVariablesAndGridChips(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qt_application = QApplication.instance()
        if cls.qt_application is None:
            cls.qt_application = QApplication([])

    def test_viewmodel_standard_variables_protocol(self) -> None:
        """
        Проверка интерфейса стандартных переменных протокола во ViewModel.
        """
        base_viewmodel = BaseProcedureViewModel()
        self.assertEqual(base_viewmodel.protocol_standard_variables, [])
        self.assertFalse(base_viewmodel.is_standard_variable("gantry_angle"))
        self.assertFalse(base_viewmodel.is_read_only_variable("gantry_angle"))

        step_shoot_viewmodel = StepAndShootProcedureViewModel(steps=8)
        self.assertIn("gantry_angle", step_shoot_viewmodel.protocol_standard_variables)
        self.assertIn("step_time", step_shoot_viewmodel.protocol_standard_variables)
        self.assertIn("radius", step_shoot_viewmodel.protocol_standard_variables)

        # Проверка флагов защиты
        self.assertTrue(step_shoot_viewmodel.is_standard_variable("gantry_angle"))
        self.assertTrue(step_shoot_viewmodel.is_standard_variable("step_time"))
        self.assertTrue(step_shoot_viewmodel.is_standard_variable("radius"))
        self.assertFalse(step_shoot_viewmodel.is_standard_variable("couch_shift_x"))

        self.assertTrue(step_shoot_viewmodel.is_read_only_variable("gantry_angle"))
        self.assertTrue(step_shoot_viewmodel.is_read_only_variable("step_time"))
        self.assertFalse(step_shoot_viewmodel.is_read_only_variable("radius"))
        self.assertFalse(step_shoot_viewmodel.is_read_only_variable("couch_shift_x"))

        # Проверка обновления стандартных переменных
        step_shoot_viewmodel.steps = 16
        self.assertEqual(len(step_shoot_viewmodel.zipped_variables["gantry_angle"]), 16)
        self.assertEqual(len(step_shoot_viewmodel.zipped_variables["step_time"]), 16)

    def test_property_inspector_read_only_standard_columns_and_deletion_guard(self) -> None:
        """
        Проверка отображения и защиты стандартных колонок в tbl_zipped_variables:
        - Read-Only флаги для gantry_angle и step_time;
        - Префикс 🔒 и tooltip в заголовках;
        - Блокировка удаления стандартных колонок с выводом предупреждения.
        """
        step_shoot_viewmodel = StepAndShootProcedureViewModel(steps=4)
        inspector = PropertyInspector()
        inspector.set_procedure_viewmodel(step_shoot_viewmodel)
        inspector.set_target_viewmodel(step_shoot_viewmodel)

        table_widget = inspector.tbl_zipped_variables
        column_count = table_widget.columnCount()
        self.assertGreaterEqual(column_count, 3)

        # Поиск индексов колонок
        gantry_column_index = -1
        step_time_column_index = -1
        radius_column_index = -1

        for current_column in range(column_count):
            header_item = table_widget.horizontalHeaderItem(current_column)
            self.assertIsNotNone(header_item)
            header_text = header_item.text()
            if "gantry_angle" in header_text:
                gantry_column_index = current_column
                self.assertTrue(header_text.startswith("🔒"))
                self.assertIn("Стандартная переменная протокола", header_item.toolTip())
            elif "step_time" in header_text:
                step_time_column_index = current_column
                self.assertTrue(header_text.startswith("🔒"))
                self.assertIn("Стандартная переменная протокола", header_item.toolTip())
            elif "radius" in header_text:
                radius_column_index = current_column
                self.assertFalse(header_text.startswith("🔒"))

        self.assertNotEqual(gantry_column_index, -1)
        self.assertNotEqual(step_time_column_index, -1)
        self.assertNotEqual(radius_column_index, -1)

        # Проверка флагов ячеек: gantry_angle и step_time не должны быть редактируемыми
        for row_index in range(table_widget.rowCount()):
            gantry_cell = table_widget.item(row_index, gantry_column_index)
            self.assertIsNotNone(gantry_cell)
            self.assertEqual(gantry_cell.flags() & Qt.ItemIsEditable, Qt.ItemFlags(0))
            self.assertTrue(bool(gantry_cell.flags() & Qt.ItemIsEnabled))

            step_time_cell = table_widget.item(row_index, step_time_column_index)
            self.assertIsNotNone(step_time_cell)
            self.assertEqual(step_time_cell.flags() & Qt.ItemIsEditable, Qt.ItemFlags(0))

            radius_cell = table_widget.item(row_index, radius_column_index)
            self.assertIsNotNone(radius_cell)
            self.assertTrue(bool(radius_cell.flags() & Qt.ItemIsEditable))

        # Попытка удалить стандартную колонку gantry_angle должна быть заблокирована с предупреждением
        table_widget.setCurrentCell(0, gantry_column_index)
        initial_columns = table_widget.columnCount()
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
            inspector._on_remove_zipped_col_clicked()
            self.assertEqual(mock_warning.call_count, 1)

        self.assertEqual(table_widget.columnCount(), initial_columns)
        self.assertIn("gantry_angle", step_shoot_viewmodel.zipped_variables)

    def test_grid_chip_interaction_and_editing(self) -> None:
        """
        Проверка виджета GridChip: значение, форматирование, инлайн-редактирование и сигналы.
        """
        chip_widget = GridChip(12.5000)
        self.assertEqual(chip_widget.value, 12.5)
        self.assertEqual(chip_widget._label_display.text(), "12.5")

        # Проверка сигнала инлайн-редактирования
        values_logged = []
        chip_widget.value_changed.connect(lambda new_number: values_logged.append(new_number))

        chip_widget.start_editing()
        self.assertEqual(chip_widget._stacked_layout.currentIndex(), 1)

        # Ввод нового значения
        chip_widget._edit_input.setText("45.25")
        key_press_return = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
        chip_widget._edit_input.keyPressEvent(key_press_return)

        self.assertEqual(chip_widget.value, 45.25)
        self.assertEqual(chip_widget._label_display.text(), "45.25")
        self.assertEqual(chip_widget._stacked_layout.currentIndex(), 0)
        self.assertEqual(values_logged, [45.25])

        # Клавиатурные сигналы Delete и навигации
        delete_logged = []
        navigate_logged = []
        chip_widget.delete_requested.connect(lambda: delete_logged.append(True))
        chip_widget.navigate_requested.connect(lambda offset: navigate_logged.append(offset))

        key_press_delete = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
        chip_widget.keyPressEvent(key_press_delete)
        self.assertEqual(len(delete_logged), 1)

        key_press_left = QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
        chip_widget.keyPressEvent(key_press_left)
        self.assertEqual(navigate_logged, [-1])

        key_press_right = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
        chip_widget.keyPressEvent(key_press_right)
        self.assertEqual(navigate_logged, [-1, 1])

    def test_grid_variable_chips_row_full_lifecycle(self) -> None:
        """
        Проверка строки GridVariableChipsRow:
        - Заголовок и счетчик;
        - Добавление одного значения и нескольких через запятую;
        - Удаление чипса;
        - Защита от удаления последнего чипса;
        - Удаление всей переменной через кнопку;
        - Совместимость с table.rowCount().
        """
        chips_row = GridVariableChipsRow("collimator_tilt", [10.0, 20.0])
        self.assertEqual(chips_row.values, [10.0, 20.0])
        self.assertIn("2 знач.", chips_row.lbl_title.text())
        self.assertEqual(len(chips_row._chips), 2)
        self.assertEqual(chips_row.table.rowCount(), 2)

        changed_history = []
        chips_row.values_changed.connect(lambda var_title, number_list: changed_history.append((var_title, number_list)))

        # Добавление одиночного значения
        chips_row.input_add_value.setText("30.0")
        chips_row._on_add_input_entered()
        self.assertEqual(chips_row.values, [10.0, 20.0, 30.0])
        self.assertIn("3 знач.", chips_row.lbl_title.text())
        self.assertEqual(chips_row.table.rowCount(), 3)

        # Пакетное добавление через запятую
        chips_row.input_add_value.setText("40.0, 50.0, 60.0")
        chips_row._on_add_input_entered()
        self.assertEqual(chips_row.values, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(len(chips_row._chips), 6)
        self.assertIn("6 знач.", chips_row.lbl_title.text())

        # Удаление второго чипса (20.0)
        target_chip = chips_row._chips[1]
        chips_row._on_chip_delete_requested(target_chip)
        self.assertEqual(chips_row.values, [10.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(len(chips_row._chips), 5)

        # Удаление чипсов до 1
        while len(chips_row._chips) > 1:
            chips_row._on_chip_delete_requested(chips_row._chips[-1])
        self.assertEqual(len(chips_row._chips), 1)

        # Попытка удалить последний чипс должна быть отклонена с информационным окном
        with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
            chips_row._on_chip_delete_requested(chips_row._chips[0])
            self.assertEqual(mock_info.call_count, 1)

        self.assertEqual(len(chips_row._chips), 1)
        self.assertEqual(len(chips_row.values), 1)

        # Удаление всей переменной через кнопку корзины
        with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.Yes):
            deleted_vars = []
            chips_row.variable_deleted.connect(lambda removed_title: deleted_vars.append(removed_title))
            chips_row._on_delete_clicked()
            self.assertEqual(deleted_vars, ["collimator_tilt"])

    def test_grid_range_dialog_generation(self) -> None:
        """
        Проверка диалога GridRangeDialog:
        - Генерация по шагу;
        - Генерация по количеству точек;
        - Опция endpoint.
        """
        dialog = GridRangeDialog("test_angle")

        # Режим по шагу: 0 до 30 с шагом 10
        dialog.combo_mode.setCurrentIndex(0)
        dialog.spin_start.setValue(0.0)
        dialog.spin_stop.setValue(30.0)
        dialog.spin_step.setValue(10.0)
        dialog.chk_endpoint.setChecked(True)
        dialog._update_preview()
        self.assertEqual(dialog.get_generated_values(), [0.0, 10.0, 20.0, 30.0])

        # Режим по количеству точек: 0 до 100, 5 точек
        dialog.combo_mode.setCurrentIndex(1)
        dialog.spin_start.setValue(0.0)
        dialog.spin_stop.setValue(100.0)
        dialog.spin_count.setValue(5)
        dialog.chk_endpoint.setChecked(True)
        dialog._update_preview()
        self.assertEqual(dialog.get_generated_values(), [0.0, 25.0, 50.0, 75.0, 100.0])

        # Режим без endpoint
        dialog.chk_endpoint.setChecked(False)
        dialog._update_preview()
        self.assertEqual(dialog.get_generated_values(), [0.0, 20.0, 40.0, 60.0, 80.0])

    def test_grid_chip_small_numbers_and_scientific_notation(self) -> None:
        """
        Проверка сохранения точности малых чисел и экспоненциальной нотации в GridChip.
        Гарантирует, что 0.00001 не превращается в '0'.
        """
        chip_small = GridChip(0.00001)
        self.assertEqual(chip_small.value, 0.00001)
        self.assertEqual(chip_small._label_display.text(), "1e-05")

        chip_submilli = GridChip(0.000123)
        self.assertEqual(chip_submilli.value, 0.000123)
        self.assertEqual(chip_submilli._label_display.text(), "0.000123")

        chip_zero = GridChip(0.0)
        self.assertEqual(chip_zero._label_display.text(), "0")

        chip_neg_zero = GridChip(-0.0)
        self.assertEqual(chip_neg_zero._label_display.text(), "0")

        # Инлайн-редактирование с вводом экспоненциального числа
        chip_small.start_editing()
        chip_small._edit_input.setText("1e-4")
        key_press_return = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
        chip_small._edit_input.keyPressEvent(key_press_return)
        self.assertAlmostEqual(chip_small.value, 0.0001, places=7)
        self.assertEqual(chip_small._label_display.text(), "0.0001")

    def test_grid_range_dialog_ieee754_boundary_precision(self) -> None:
        """
        Проверка устранения погрешностей деления IEEE 754 (0.7 / 0.1, 0.3 / 0.1, 0.6 / 0.2).
        """
        dialog = GridRangeDialog("pitch")
        dialog.combo_mode.setCurrentIndex(0)
        dialog.spin_start.setValue(0.0)
        dialog.spin_stop.setValue(0.7)
        dialog.spin_step.setValue(0.1)

        # endpoint = True
        dialog.chk_endpoint.setChecked(True)
        dialog._update_preview()
        expected_with_endpoint = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.assertEqual(len(dialog.get_generated_values()), 8)
        for actual_val, expected_val in zip(dialog.get_generated_values(), expected_with_endpoint):
            self.assertAlmostEqual(actual_val, expected_val, places=6)

        # endpoint = False
        dialog.chk_endpoint.setChecked(False)
        dialog._update_preview()
        expected_without_endpoint = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual(len(dialog.get_generated_values()), 7)
        for actual_val, expected_val in zip(dialog.get_generated_values(), expected_without_endpoint):
            self.assertAlmostEqual(actual_val, expected_val, places=6)

    def test_grid_chips_row_keyboard_delete_and_error_state(self) -> None:
        """
        Проверка направления удаления Backspace/Delete, методов count/len и сброса ошибки поля ввода.
        """
        chips_row = GridVariableChipsRow("energy_bins", [100.0, 140.0, 180.0])
        self.assertEqual(chips_row.count(), 3)
        self.assertEqual(len(chips_row), 3)

        # Ввод некорректного значения -> подсветка ошибки
        chips_row.input_add_value.setText("not_a_number")
        chips_row._on_add_input_entered()
        self.assertIn("#3b2222", chips_row.input_add_value.styleSheet())
        self.assertEqual(chips_row.count(), 3)

        # Изменение текста сбрасывает стиль ошибки
        chips_row.input_add_value.setText("1")
        self.assertNotIn("#3b2222", chips_row.input_add_value.styleSheet())

        # Нажатие Escape очищает поле
        escape_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        chips_row.eventFilter(chips_row.input_add_value, escape_event)
        self.assertEqual(chips_row.input_add_value.text(), "")

        # Удаление клавишей Backspace (direction_hint = -1) переносит фокус на левый элемент
        target_chip = chips_row._chips[1]  # 140.0
        with patch.object(GridChip, "setFocus") as mock_focus_backspace:
            chips_row._on_chip_delete_requested(target_chip, direction_hint=-1)
            mock_focus_backspace.assert_called_once()
        self.assertEqual(chips_row.values, [100.0, 180.0])

        # Удаление клавишей Delete (direction_hint = 0) оставляет фокус на позиции удаленного чипса (индекс 0)
        target_chip_first = chips_row._chips[0]  # 100.0
        with patch.object(GridChip, "setFocus") as mock_focus_delete:
            chips_row._on_chip_delete_requested(target_chip_first, direction_hint=0)
            mock_focus_delete.assert_called_once()
        self.assertEqual(chips_row.values, [180.0])


if __name__ == "__main__":
    unittest.main()
