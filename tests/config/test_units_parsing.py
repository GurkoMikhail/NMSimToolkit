"""Модульные тесты интеллектуального парсинга единиц измерения (Раздел 11).

Проверяет работу:
- parse_unit_value и parse_unit_tokens в core.config.units;
- UnitAwareDoubleSpinBox и BindableDoubleField в gui.views.widgets.bindable_field;
- GridChip, GridVariableChipsRow и GridRangeDialog в gui.views.widgets.grid_chips_widget.
"""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent
from PySide6.QtWidgets import QApplication

from core.config.units import (
    get_unit_category,
    normalize_unit_name,
    parse_unit_tokens,
    parse_unit_value,
)
from gui.views.widgets.bindable_field import (
    BindableDoubleField,
    UnitAwareDoubleSpinBox,
    infer_unit_from_suffix,
)
from gui.views.widgets.grid_chips_widget import (
    GridChip,
    GridRangeDialog,
    GridVariableChipsRow,
    infer_variable_default_unit,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Глобальный экземпляр QApplication для тестирования виджетов."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    assert isinstance(app, QApplication)
    return app


class TestCoreUnitsParsing:
    """Тесты парсера физических единиц измерения в core.config.units."""

    def test_normalize_unit_aliases_en_and_ru(self) -> None:
        """Проверка нормализации русских, символьных и регистронезависимых алиасов."""
        assert normalize_unit_name("mm") == "mm"
        assert normalize_unit_name("ММ") == "mm"
        assert normalize_unit_name("см") == "cm"
        assert normalize_unit_name("м") == "m"
        assert normalize_unit_name("°") == "deg"
        assert normalize_unit_name("град") == "deg"
        assert normalize_unit_name("рад") == "rad"
        assert normalize_unit_name("kev") == "keV"
        assert normalize_unit_name("кэВ") == "keV"
        assert normalize_unit_name("МэВ") == "MeV"
        assert normalize_unit_name("с") == "s"
        assert normalize_unit_name("мин") == "min"
        assert normalize_unit_name("МБк") == "MBq"
        assert normalize_unit_name("мКи") == "mCi"

        assert get_unit_category("cm") == "length"
        assert get_unit_category("кэВ") == "energy"
        assert get_unit_category("град") == "angle"
        assert get_unit_category("мин") == "time"
        assert get_unit_category("мКи") == "activity"

    def test_infer_unit_from_suffix(self) -> None:
        """Проверка извлечения канонической единицы из суффикса spinbox."""
        assert infer_unit_from_suffix(" mm") == "mm"
        assert infer_unit_from_suffix(" °") == "deg"
        assert infer_unit_from_suffix(" MeV") == "MeV"
        assert infer_unit_from_suffix(" с") == "s"
        assert infer_unit_from_suffix("") is None
        assert infer_unit_from_suffix(" %") is None

    def test_parse_length_to_default_unit_and_hepunits(self) -> None:
        """Конвертация длины (мм, см, м, мкм) в целевую единицу и в hepunits."""
        assert parse_unit_value("25 cm", default_unit="mm") == pytest.approx(250.0)
        assert parse_unit_value("0.25 м", default_unit="mm") == pytest.approx(250.0)
        assert parse_unit_value("25,5 см", default_unit="mm") == pytest.approx(255.0)
        assert parse_unit_value("500 um", default_unit="mm") == pytest.approx(0.5)
        assert parse_unit_value("250 mm", default_unit="cm") == pytest.approx(25.0)
        assert parse_unit_value(120, default_unit="mm") == pytest.approx(120.0)
        assert parse_unit_value("12 cm", default_unit="cm", to_hepunits=True) == pytest.approx(120.0)
        assert parse_unit_value("12", default_unit="cm", to_hepunits=True) == pytest.approx(120.0)

    def test_parse_energy_units(self) -> None:
        """Конвертация энергии (эВ, кэВ, МэВ, ГэВ)."""
        assert parse_unit_value("140.5 keV", default_unit="MeV") == pytest.approx(0.1405)
        assert parse_unit_value("140,5 кэВ", default_unit="MeV") == pytest.approx(0.1405)
        assert parse_unit_value("0.511 МэВ", default_unit="keV") == pytest.approx(511.0)
        assert parse_unit_value("1.5 GeV", to_hepunits=True) == pytest.approx(1500.0)

    def test_parse_time_and_activity_units(self) -> None:
        """Конвертация времени и активности."""
        assert parse_unit_value("1.5 min", default_unit="s") == pytest.approx(90.0)
        assert parse_unit_value("2 мин", default_unit="s") == pytest.approx(120.0)
        assert parse_unit_value("500 ms", default_unit="s") == pytest.approx(0.5)
        assert parse_unit_value("500 мс", default_unit="s") == pytest.approx(0.5)
        assert parse_unit_value("1 s", to_hepunits=True) == pytest.approx(1e9)

        assert parse_unit_value("10 mCi", default_unit="MBq") == pytest.approx(370.0)
        assert parse_unit_value("10 мКи", default_unit="MBq") == pytest.approx(370.0)
        assert parse_unit_value("500 kBq", default_unit="MBq") == pytest.approx(0.5)
        assert parse_unit_value("2 ГБк", default_unit="MBq") == pytest.approx(2000.0)

    def test_parse_angles_and_pi_expressions(self) -> None:
        """Конвертация углов (градусы, радианы) и выражений с константой pi / π."""
        assert parse_unit_value("pi rad", default_unit="deg") == pytest.approx(180.0)
        assert parse_unit_value("π рад", default_unit="deg") == pytest.approx(180.0)
        assert parse_unit_value("pi/2 rad", default_unit="deg") == pytest.approx(90.0)
        assert parse_unit_value("-pi/4 рад", default_unit="deg") == pytest.approx(-45.0)
        assert parse_unit_value("2*pi rad", default_unit="deg") == pytest.approx(360.0)
        assert parse_unit_value("0.5 * pi rad", default_unit="deg") == pytest.approx(90.0)
        assert parse_unit_value("90°", default_unit="deg") == pytest.approx(90.0)
        assert parse_unit_value("45 град", default_unit="deg") == pytest.approx(45.0)
        assert parse_unit_value("180 deg", default_unit="rad") == pytest.approx(math.pi)
        assert parse_unit_value("180 deg", to_hepunits=True) == pytest.approx(math.pi)

    def test_parse_unit_value_rejects_incompatible_dimensions_and_invalid_inputs(self) -> None:
        """Проверка выброса ValueError при несовместимых размерностях и некорректном вводе."""
        with pytest.raises(ValueError, match="Несовместимая физическая размерность"):
            parse_unit_value("50 keV", default_unit="mm")

        with pytest.raises(ValueError, match="Несовместимая физическая размерность"):
            parse_unit_value("25 cm", default_unit="deg")

        with pytest.raises(ValueError, match="Несовместимая физическая размерность"):
            parse_unit_value("10 s", default_unit="MBq")

        with pytest.raises(ValueError, match="Неизвестная единица"):
            parse_unit_value("10 blorgs", default_unit="mm")

        with pytest.raises(ValueError):
            parse_unit_value("", default_unit="mm")

        with pytest.raises(ValueError):
            parse_unit_value(True, default_unit="mm")

    def test_parse_unit_tokens_batch(self) -> None:
        """Пакетный парсинг списка значений с разными единицами одной размерности."""
        lengths = parse_unit_tokens("10 cm, 250 mm; 0.3 m, 15,5 см", default_unit="mm")
        assert lengths == pytest.approx([100.0, 250.0, 300.0, 155.0])

        angles = parse_unit_tokens("0, pi/2 rad, 180°, 1.5*pi рад", default_unit="deg")
        assert angles == pytest.approx([0.0, 90.0, 180.0, 270.0])


class TestBindableDoubleFieldUnitParsing:
    """Тесты ввода размерных строк в UnitAwareDoubleSpinBox и BindableDoubleField."""

    def test_spinbox_converts_unit_text_and_reverts_on_incompatible_unit(
        self, qapp: QApplication
    ) -> None:
        """Ввод '25 cm' в поле с суффиксом ' mm' конвертируется в 250.0, а '50 keV' откатывается."""
        field = BindableDoubleField(min_val=0.0, max_val=2000.0, decimals=2, suffix=" mm")
        assert field.unit() == "mm"
        field.setValue(100.0)
        assert field.value() == pytest.approx(100.0)

        # Программная установка через размерную строку
        field.setValue("25 cm")
        assert field.value() == pytest.approx(250.0)

        # Пользовательский ввод в lineEdit с последующим нажатием Enter
        emitted: list[float] = []
        field.valueChanged.connect(emitted.append)

        line_edit = field.spin_box.lineEdit()
        assert line_edit is not None
        line_edit.setText("0.45 м")
        enter_event = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
        )
        field.spin_box.keyPressEvent(enter_event)

        assert field.value() == pytest.approx(450.0)
        assert len(emitted) >= 1
        assert emitted[-1] == pytest.approx(450.0)

        # Ввод несовместимой единицы измерения ('50 keV' в поле длины ' mm')
        line_edit.setText("50 keV")
        focus_out = QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason)
        field.spin_box.focusOutEvent(focus_out)

        # Значение должно откатиться к последнему валидному (450.0)
        assert field.value() == pytest.approx(450.0)

    def test_angle_and_energy_spinboxes(self, qapp: QApplication) -> None:
        """Проверка конвертации радианов в градусы и кэВ в МэВ в BindableDoubleField."""
        angle_field = BindableDoubleField(
            min_val=-360.0, max_val=360.0, decimals=2, suffix=" °"
        )
        assert angle_field.unit() == "deg"
        line_edit = angle_field.spin_box.lineEdit()
        assert line_edit is not None
        line_edit.setText("pi/2 rad")
        angle_field.spin_box.interpretText()
        assert angle_field.value() == pytest.approx(90.0)

        energy_field = BindableDoubleField(
            min_val=0.0, max_val=10.0, decimals=4, suffix=" MeV"
        )
        assert energy_field.unit() == "MeV"
        energy_edit = energy_field.spin_box.lineEdit()
        assert energy_edit is not None
        energy_edit.setText("140.5 кэВ")
        energy_field.spin_box.interpretText()
        assert energy_field.value() == pytest.approx(0.1405)


class TestGridChipsWidgetUnitParsing:
    """Тесты поддержки единиц измерения в GridChip, GridVariableChipsRow и GridRangeDialog."""

    def test_grid_chip_inline_edit_with_units(self, qapp: QApplication) -> None:
        """Редактирование одиночного чипса с вводом размерной строки и откат при ошибке."""
        assert infer_variable_default_unit("radius") == "mm"
        assert infer_variable_default_unit("gantry_angle") == "deg"
        assert infer_variable_default_unit("frame_time") == "s"

        chip = GridChip(200.0, default_unit="mm")
        edited_values: list[float] = []
        chip.value_changed.connect(edited_values.append)

        chip.start_editing()
        chip._edit_input.setText("25 cm")
        chip._edit_input._commit_edit(refocus=False)

        assert edited_values == [pytest.approx(250.0)]
        assert chip.value == pytest.approx(250.0)

        # При несовместимой размерности сигнал value_changed не должен испускаться
        chip.start_editing()
        chip._edit_input.setText("140 keV")
        chip._edit_input._commit_edit(refocus=False)
        assert len(edited_values) == 1
        assert chip.value == pytest.approx(250.0)

    def test_grid_variable_chips_row_batch_unit_input(self, qapp: QApplication) -> None:
        """Пакетное добавление значений с единицами измерения в строку чипсов."""
        row = GridVariableChipsRow("radius", [100.0])
        assert row.default_unit == "mm"

        row.input_add_value.setText("20 cm, 0.35 m; 150 мм")
        row._on_add_input_entered()

        assert row.values == pytest.approx([100.0, 200.0, 350.0, 150.0])

        angle_row = GridVariableChipsRow("gantry_angle", [0.0])
        assert angle_row.default_unit == "deg"
        angle_row.input_add_value.setText("pi/2 rad, 180°")
        angle_row._on_add_input_entered()
        assert angle_row.values == pytest.approx([0.0, 90.0, 180.0])

    def test_grid_range_dialog_with_unit_aware_spinboxes(self, qapp: QApplication) -> None:
        """Генерация диапазона в GridRangeDialog при вводе значений в разных единицах длины."""
        dialog = GridRangeDialog("radius", default_unit="mm")
        assert isinstance(dialog.spin_start, UnitAwareDoubleSpinBox)
        assert isinstance(dialog.spin_stop, UnitAwareDoubleSpinBox)
        assert isinstance(dialog.spin_step, UnitAwareDoubleSpinBox)

        start_edit = dialog.spin_start.lineEdit()
        stop_edit = dialog.spin_stop.lineEdit()
        step_edit = dialog.spin_step.lineEdit()
        assert start_edit is not None and stop_edit is not None and step_edit is not None

        start_edit.setText("10 cm")
        dialog.spin_start.interpretText()
        stop_edit.setText("0.3 m")
        dialog.spin_stop.interpretText()
        step_edit.setText("5 см")
        dialog.spin_step.interpretText()

        assert dialog.get_generated_values() == pytest.approx([100.0, 150.0, 200.0, 250.0, 300.0])
