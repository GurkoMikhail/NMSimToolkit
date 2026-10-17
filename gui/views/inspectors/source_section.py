"""
Секция параметров источника излучения (SourceNode / SourceViewModel).
"""
import math
import re
from typing import Any, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from gui.models.distribution_import_params import ImportTargetKind
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.views.distribution_import_dialog import DistributionImportDialog
from gui.views.inspectors.base_section import BaseInspectorSection
from gui.views.widgets import BindableDoubleField

_FLOAT_PATTERN = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


def _try_parse_float(text: Optional[str], default_val: float) -> float:
    """Безопасный парсинг вещественного числа по принципу LBYL без возбуждения исключений."""
    if not text:
        return default_val
    clean_text = text.strip().replace(",", ".")
    if _FLOAT_PATTERN.match(clean_text):
        parsed = float(clean_text)
        if math.isfinite(parsed):
            return parsed
    return default_val


class SourceInspectorSection(BaseInspectorSection):
    """Секция настройки спектра энергий, активности, полураспада и 3D-распределения источника."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию параметров источника излучения."""
        super().__init__("Параметры источника (Radiation Source)", parent)

    def _init_ui(self) -> None:
        """Создает элементы управления спектром, активностью, полураспадом и распределением источника."""
        form_layout = QFormLayout(self)

        self.combo_rad_type = QComboBox()
        self.combo_rad_type.addItems(["Gamma", "Beta", "Positron"])
        self.combo_rad_type.currentTextChanged.connect(self._on_source_rad_type_changed)
        form_layout.addRow("Тип излучения:", self.combo_rad_type)

        self.tbl_source_spectrum = QTableWidget(0, 2)
        self.tbl_source_spectrum.setHorizontalHeaderLabels(["Энергия (кэВ)", "Вес / Доля"])
        self.tbl_source_spectrum.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tbl_source_spectrum.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tbl_source_spectrum.setMinimumHeight(120)
        self.tbl_source_spectrum.cellChanged.connect(self._on_spectrum_cell_changed)

        spectrum_btn_layout = QHBoxLayout()
        self.btn_add_spectrum_line = QPushButton("➕ Добавить пик")
        self.btn_add_spectrum_line.clicked.connect(self._on_add_spectrum_row)
        self.btn_remove_spectrum_line = QPushButton("➖ Удалить")
        self.btn_remove_spectrum_line.clicked.connect(self._on_remove_spectrum_row)
        spectrum_btn_layout.addWidget(self.btn_add_spectrum_line)
        spectrum_btn_layout.addWidget(self.btn_remove_spectrum_line)

        form_layout.addRow("Спектр излучения:", self.tbl_source_spectrum)
        form_layout.addRow("", spectrum_btn_layout)

        self.spin_source_activity = BindableDoubleField(
            min_val=0.001,
            max_val=100000.0,
            step=10.0,
            decimals=3,
            unit="MBq",
        )
        self.spin_source_activity.setValue(100.0)
        self.spin_source_activity.set_variable_provider(self._get_protocol_variables)
        self.spin_source_activity.set_variable_created_callback(self._on_variable_created)
        self.spin_source_activity.valueChanged.connect(self._on_source_activity_changed)
        self.spin_source_activity.bindingChanged.connect(
            lambda var_name: self._on_field_binding_changed("activity", var_name)
        )
        form_layout.addRow("Активность (МБк):", self.spin_source_activity)

        self.spin_source_half_life = QDoubleSpinBox()
        self.spin_source_half_life.setRange(0.0, 100000.0)
        self.spin_source_half_life.setSingleStep(1.0)
        self.spin_source_half_life.setValue(6.0)
        self.spin_source_half_life.valueChanged.connect(self._on_source_half_life_changed)
        form_layout.addRow("Период полураспада (ч):", self.spin_source_half_life)

        self.spin_source_voxel_size = QDoubleSpinBox()
        self.spin_source_voxel_size.setRange(0.0, 100.0)
        self.spin_source_voxel_size.setSingleStep(0.5)
        self.spin_source_voxel_size.setValue(4.0)
        self.spin_source_voxel_size.setSuffix(" мм")
        self.spin_source_voxel_size.valueChanged.connect(self._on_source_voxel_size_changed)
        form_layout.addRow("Шаг вокселя источника:", self.spin_source_voxel_size)

        self.txt_source_path = QLineEdit()
        self.txt_source_path.setReadOnly(True)
        self.btn_browse_source = QPushButton("Обзор...")
        self.btn_browse_source.clicked.connect(self._on_browse_source_file)
        src_path_layout = QHBoxLayout()
        src_path_layout.addWidget(self.txt_source_path)
        src_path_layout.addWidget(self.btn_browse_source)
        form_layout.addRow("Файл распределения:", src_path_layout)

        self.lbl_source_shape = QLabel("Точечный источник")
        form_layout.addRow("Сетка источника:", self.lbl_source_shape)

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления источником SourceViewModel."""
        return isinstance(view_model, SourceViewModel)

    def unbind(self) -> None:
        """Сбрасывает привязки и очищает таблицу спектра при отвязке источника."""
        super().unbind()
        self.spin_source_activity.unbind()
        self.tbl_source_spectrum.setRowCount(0)

    def update_ui_from_vm(self) -> None:
        """Синхронизирует все поля секции с текущей моделью SourceViewModel."""
        if not isinstance(self._current_vm, SourceViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            source_type_index = self.combo_rad_type.findText(self._current_vm.radiation_type)
            if source_type_index >= 0:
                self.combo_rad_type.setCurrentIndex(source_type_index)

            bindings = self._current_vm.bindings
            if not bindings and self._scene_vm is not None and self._current_vm.core_node in self._scene_vm.bindings_registry:
                bindings = self._scene_vm.bindings_registry[self._current_vm.core_node]

            self._update_source_spectrum_table()

            bound_act = bindings.get("activity")
            if bound_act:
                self.spin_source_activity.bind_variable(bound_act)
            else:
                self.spin_source_activity.unbind()
            self.spin_source_activity.setValue(float(self._current_vm.activity))
            self.spin_source_half_life.setValue(float(self._current_vm.half_life))
            self.spin_source_voxel_size.setValue(float(self._current_vm.voxel_size))
            self.txt_source_path.setText(str(self._current_vm.file_path))
            dims = self._current_vm.dimensions
            if self._current_vm.is_point_source:
                self.lbl_source_shape.setText("Точечный источник")
            else:
                self.lbl_source_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
        finally:
            self._is_updating_ui = old_state

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        """Точечно обновляет поля секции при внешнем изменении свойств источника."""
        if not isinstance(self._current_vm, SourceViewModel):
            return

        if prop_name == "activity":
            self._set_spin_safe(self.spin_source_activity, float(new_val))
        elif prop_name in ("energy", "spectrum"):
            self._is_updating_ui = True
            try:
                self._update_source_spectrum_table()
            finally:
                self._is_updating_ui = False
        elif prop_name == "half_life":
            self._set_spin_safe(self.spin_source_half_life, float(new_val))
        elif prop_name == "voxel_size":
            self._set_spin_safe(self.spin_source_voxel_size, float(new_val))
        elif prop_name == "file_path":
            self._is_updating_ui = True
            try:
                self.txt_source_path.setText(str(new_val))
                dims = self._current_vm.dimensions
                if self._current_vm.is_point_source:
                    self.lbl_source_shape.setText("Точечный источник")
                else:
                    self.lbl_source_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
            finally:
                self._is_updating_ui = False
        elif prop_name == "bindings":
            bound_act = self._current_vm.bindings.get("activity")
            if bound_act:
                self.spin_source_activity.bind_variable(bound_act)
            else:
                self.spin_source_activity.unbind()

    def _set_spin_safe(self, spin: Any, val: float) -> None:
        """Устанавливает значение спинбокса с защитой от рекурсивных сигналов."""
        self._is_updating_ui = True
        try:
            spin.setValue(val)
        finally:
            self._is_updating_ui = False

    def _on_source_rad_type_changed(self, text: str) -> None:
        """Применяет выбранный тип излучения к модели представления источника."""
        if self._is_updating_ui or not isinstance(self._current_vm, SourceViewModel):
            return
        self._current_vm.radiation_type = text

    def _update_source_spectrum_table(self) -> None:
        """Заполнение строк таблицы дискретными линиями энергий и весов спектра."""
        if not isinstance(self._current_vm, SourceViewModel):
            return
        self.tbl_source_spectrum.blockSignals(True)
        try:
            spectrum = self._current_vm.spectrum
            self.tbl_source_spectrum.setRowCount(len(spectrum))
            for row_index, energy_weight_pair in enumerate(spectrum):
                item_energy = QTableWidgetItem(f"{float(energy_weight_pair[0]):.4f}")
                item_weight = QTableWidgetItem(f"{float(energy_weight_pair[1]):.4f}")
                item_energy.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_weight.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tbl_source_spectrum.setItem(row_index, 0, item_energy)
                self.tbl_source_spectrum.setItem(row_index, 1, item_weight)
        finally:
            self.tbl_source_spectrum.blockSignals(False)

    def _on_spectrum_cell_changed(self, row_index: int, column_index: int) -> None:
        """Синхронизация модифицированного значения энергии или веса в таблице спектра по принципу LBYL."""
        if self._is_updating_ui or not isinstance(self._current_vm, SourceViewModel):
            return
        current_spectrum = self._current_vm.spectrum
        new_spectrum: List[List[float]] = []
        needs_ui_refresh = False

        for current_row_index in range(self.tbl_source_spectrum.rowCount()):
            item_energy = self.tbl_source_spectrum.item(current_row_index, 0)
            item_weight = self.tbl_source_spectrum.item(current_row_index, 1)
            energy_text = item_energy.text().strip() if item_energy is not None else ""
            weight_text = item_weight.text().strip() if item_weight is not None else ""

            prev_energy = (
                float(current_spectrum[current_row_index][0])
                if current_row_index < len(current_spectrum)
                else 140.5
            )
            prev_weight = (
                float(current_spectrum[current_row_index][1])
                if current_row_index < len(current_spectrum)
                else 1.0
            )

            energy_val = _try_parse_float(energy_text, prev_energy)
            if energy_val <= 0.0:
                energy_val = prev_energy if prev_energy > 0.0 else 140.5
                needs_ui_refresh = True

            weight_val = _try_parse_float(weight_text, prev_weight)
            if weight_val < 0.0:
                weight_val = prev_weight if prev_weight >= 0.0 else 1.0
                needs_ui_refresh = True

            new_spectrum.append([energy_val, weight_val])

        if new_spectrum:
            total_weight = sum(line[1] for line in new_spectrum)
            if total_weight <= 0.0:
                new_spectrum[0][1] = 1.0
                needs_ui_refresh = True
            self._current_vm.spectrum = new_spectrum

        if needs_ui_refresh:
            self._update_source_spectrum_table()

    def _on_add_spectrum_row(self) -> None:
        """Добавление новой спектральной линии в источник."""
        if not isinstance(self._current_vm, SourceViewModel):
            return
        current_spectrum = list(self._current_vm.spectrum)
        current_spectrum.append([140.5, 1.0])
        self._current_vm.spectrum = current_spectrum
        self._update_source_spectrum_table()

    def _on_remove_spectrum_row(self) -> None:
        """Удаление выбранной линии спектра из источника."""
        if not isinstance(self._current_vm, SourceViewModel):
            return
        current_spectrum = list(self._current_vm.spectrum)
        selected_rows = sorted(
            {model_index.row() for model_index in self.tbl_source_spectrum.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows and current_spectrum:
            selected_rows = [len(current_spectrum) - 1]
        for target_row_index in selected_rows:
            if 0 <= target_row_index < len(current_spectrum):
                current_spectrum.pop(target_row_index)
        if not current_spectrum:
            current_spectrum = [[140.5, 1.0]]
        self._current_vm.spectrum = current_spectrum
        self._update_source_spectrum_table()

    def _on_source_activity_changed(self, value: float) -> None:
        """Применяет новое значение активности к модели представления источника."""
        if self._is_updating_ui or not isinstance(self._current_vm, SourceViewModel):
            return
        self._current_vm.activity = float(value)

    def _on_source_half_life_changed(self, value: float) -> None:
        """Применяет новое значение периода полураспада к модели представления источника."""
        if self._is_updating_ui or not isinstance(self._current_vm, SourceViewModel):
            return
        self._current_vm.half_life = float(value)

    def _on_source_voxel_size_changed(self, value: float) -> None:
        """Применяет новый шаг вокселя к модели представления источника."""
        if self._is_updating_ui or not isinstance(self._current_vm, SourceViewModel):
            return
        self._current_vm.voxel_size = float(value)

    def _on_browse_source_file(self) -> None:
        """Диалог выбора и загрузки файла 3D-распределения радиофармпрепарата."""
        if not isinstance(self._current_vm, SourceViewModel):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл распределения источника",
            "",
            "Файлы данных (*.npy *.dat *.raw *.bin *.h5);;Все файлы (*.*)",
        )
        if not path:
            return

        phantom_node = None
        if self._scene_vm is not None:
            for node in self._scene_vm.all_nodes():
                if isinstance(node, VoxelVolumeViewModel):
                    phantom_node = node
                    break

        dialog = DistributionImportDialog(
            parent=self,
            file_path=path,
            target_kind=ImportTargetKind.SOURCE,
            scene_phantom_node=phantom_node,
            default_voxel_size=self._current_vm.voxel_size,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            file_path_str = str(params.file_path)
            self.txt_source_path.setText(file_path_str)
            source_v_size = (
                float(params.voxel_size[0])
                if isinstance(params.voxel_size, (tuple, list, np.ndarray))
                else float(params.voxel_size)
            )
            success = self._current_vm.reload_distribution(
                path=file_path_str,
                shape=params.shape,
                order=params.order,
                dtype=params.dtype,
                encoding=params.encoding,
                voxel_size=source_v_size,
                total_activity=params.total_activity,
                noise_threshold=params.noise_threshold,
            )
            if success:
                self._is_updating_ui = True
                try:
                    self.spin_source_voxel_size.setValue(source_v_size)
                    if params.total_activity is not None:
                        self.spin_source_activity.setValue(float(params.total_activity))
                    dims = self._current_vm.dimensions
                    self.lbl_source_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
                finally:
                    self._is_updating_ui = False

                if self._scene_vm is not None:
                    self._scene_vm.update_distribution_path(
                        self._current_vm.core_node,
                        file_path_str,
                        order=params.order,
                        encoding=params.encoding,
                        dtype=params.dtype,
                    )


__all__ = ["SourceInspectorSection"]
