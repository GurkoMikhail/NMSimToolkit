"""
Секция параметров протокола исследования (Step-and-Shoot, SPECT, PET, Custom Sweep).
"""
import math
import re
from typing import Any, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    CustomSweepProcedureViewModel,
    PetProcedureViewModel,
    StepAndShootProcedureViewModel,
)
from gui.views.inspectors.base_section import BaseInspectorSection
from gui.views.widgets import BindableDoubleField, GridVariableChipsRow

_FLOAT_PATTERN = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


class ProcedureInspectorSection(BaseInspectorSection):
    """Секция настройки кинематических протоколов и матриц варьируемых переменных сканирования."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Инициализирует секцию настройки протокола исследования."""
        self._grid_var_widgets: dict[str, GridVariableChipsRow] = {}
        super().__init__("Параметры протокола исследования", parent)

    def _init_ui(self) -> None:
        """Создает элементы управления протоколами сканирования и таблицами переменных."""
        proc_form = QFormLayout(self)
        self.lbl_proc_type = QLabel("ОФЭКТ (SPECT)")
        proc_form.addRow("Тип протокола:", self.lbl_proc_type)

        self.step_and_shoot_widget = self._build_step_and_shoot_ui()
        proc_form.addRow(self.step_and_shoot_widget)

        self.custom_sweep_widget = self._build_custom_sweep_ui()
        proc_form.addRow(self.custom_sweep_widget)

    def _build_step_and_shoot_ui(self) -> QWidget:
        """Построение виджета параметров Step-and-Shoot и PET."""
        widget = QWidget()
        sas_form = QFormLayout(widget)
        sas_form.setContentsMargins(0, 0, 0, 0)

        self.spin_proc_steps = QSpinBox()
        self.spin_proc_steps.setRange(1, 1024)
        self.spin_proc_steps.setValue(32)
        self.spin_proc_steps.valueChanged.connect(self._on_proc_steps_changed)
        sas_form.addRow("Число шагов (Steps):", self.spin_proc_steps)

        self.lbl_proc_total_views = QLabel("64")
        self.lbl_proc_total_views.setStyleSheet("color: #a0c0ff; font-weight: bold;")
        sas_form.addRow("Всего проекций (Total Views):", self.lbl_proc_total_views)

        self.spin_proc_cameras = QSpinBox()
        self.spin_proc_cameras.setRange(1, 16)
        self.spin_proc_cameras.setValue(2)
        self.spin_proc_cameras.valueChanged.connect(self._on_proc_cameras_changed)
        sas_form.addRow("Гамма-камер:", self.spin_proc_cameras)

        self.spin_proc_radius = BindableDoubleField(
            min_val=10.0, max_val=2000.0, step=5.0, decimals=2, unit="mm"
        )
        self.spin_proc_radius.setValue(250.0)
        self.spin_proc_radius.set_variable_provider(self._get_protocol_variables)
        self.spin_proc_radius.set_variable_created_callback(self._on_variable_created)
        self.spin_proc_radius.valueChanged.connect(self._on_proc_radius_changed)
        self.spin_proc_radius.bindingChanged.connect(
            lambda var_name: self._on_procedure_binding_changed("radius", var_name)
        )
        sas_form.addRow("Радиус орбиты (мм):", self.spin_proc_radius)

        self.spin_proc_time = BindableDoubleField(
            min_val=0.001, max_val=3600.0, step=0.5, decimals=3, unit="s"
        )
        self.spin_proc_time.setValue(1.0)
        self.spin_proc_time.set_variable_provider(self._get_protocol_variables)
        self.spin_proc_time.set_variable_created_callback(self._on_variable_created)
        self.spin_proc_time.valueChanged.connect(self._on_proc_time_changed)
        self.spin_proc_time.bindingChanged.connect(
            lambda var_name: self._on_procedure_binding_changed("step_time", var_name)
        )
        sas_form.addRow("Время на ракурс (с):", self.spin_proc_time)

        self.spin_proc_initial_time = QDoubleSpinBox()
        self.spin_proc_initial_time.setRange(0.0, 86400.0)
        self.spin_proc_initial_time.setSingleStep(1.0)
        self.spin_proc_initial_time.setDecimals(3)
        self.spin_proc_initial_time.setValue(0.0)
        self.spin_proc_initial_time.valueChanged.connect(self._on_proc_initial_time_changed)
        sas_form.addRow("Начальное время t₀ (с):", self.spin_proc_initial_time)

        self.chk_proc_sequential_time = QCheckBox("Последовательный отсчет времени по шагам")
        self.chk_proc_sequential_time.setChecked(True)
        self.chk_proc_sequential_time.toggled.connect(self._on_proc_sequential_time_toggled)
        sas_form.addRow(self.chk_proc_sequential_time)

        self.spin_proc_start_angle = QDoubleSpinBox()
        self.spin_proc_start_angle.setRange(0.0, 360.0)
        self.spin_proc_start_angle.setValue(0.0)
        self.spin_proc_start_angle.valueChanged.connect(self._on_proc_start_angle_changed)
        sas_form.addRow("Начальный угол (°):", self.spin_proc_start_angle)

        self.spin_proc_end_angle = QDoubleSpinBox()
        self.spin_proc_end_angle.setRange(0.0, 720.0)
        self.spin_proc_end_angle.setValue(360.0)
        self.spin_proc_end_angle.valueChanged.connect(self._on_proc_end_angle_changed)
        sas_form.addRow("Конечный угол (°):", self.spin_proc_end_angle)

        self.combo_proc_head_mode = QComboBox()
        self.combo_proc_head_mode.addItems(["Симметричный (360°/N)", "L-режим (90°)", "Пользовательский"])
        self.combo_proc_head_mode.currentTextChanged.connect(self._on_proc_head_mode_changed)
        sas_form.addRow("Конфигурация головок:", self.combo_proc_head_mode)

        self.txt_proc_head_angles = QLineEdit()
        self.txt_proc_head_angles.editingFinished.connect(self._on_proc_head_angles_edited)
        sas_form.addRow("Углы головок (°):", self.txt_proc_head_angles)

        self.chk_proc_endpoint = QCheckBox("Включать конечный угол (endpoint)")
        self.chk_proc_endpoint.toggled.connect(self._on_proc_endpoint_toggled)
        sas_form.addRow(self.chk_proc_endpoint)

        return widget

    def _build_custom_sweep_ui(self) -> QWidget:
        """Построение виджета временного интервала и таблиц Zipped и Grid переменных."""
        widget = QWidget()
        cs_layout = QVBoxLayout(widget)
        cs_layout.setContentsMargins(0, 0, 0, 0)
        cs_layout.setSpacing(8)

        # Временной интервал симуляции (start_time / stop_time) с поддержкой Property Binding
        self.grp_time_interval = QGroupBox("Временной интервал симуляции")
        time_form = QFormLayout(self.grp_time_interval)
        time_form.setContentsMargins(4, 6, 4, 6)
        time_form.setSpacing(4)

        self.spin_start_time = BindableDoubleField(
            min_val=0.0, max_val=86400.0, step=1.0, decimals=3, unit="s"
        )
        self.spin_start_time.setValue(0.0)
        self.spin_start_time.set_variable_provider(self._get_protocol_variables)
        self.spin_start_time.set_variable_created_callback(self._on_variable_created)
        self.spin_start_time.valueChanged.connect(self._on_proc_start_time_changed)
        self.spin_start_time.bindingChanged.connect(
            lambda var_name: self._on_procedure_binding_changed("start_time", var_name)
        )
        time_form.addRow("Начало кадра start_time (с):", self.spin_start_time)

        self.spin_stop_time = BindableDoubleField(
            min_val=0.001, max_val=86400.0, step=1.0, decimals=3, unit="s"
        )
        self.spin_stop_time.setValue(1.0)
        self.spin_stop_time.set_variable_provider(self._get_protocol_variables)
        self.spin_stop_time.set_variable_created_callback(self._on_variable_created)
        self.spin_stop_time.valueChanged.connect(self._on_proc_stop_time_changed)
        self.spin_stop_time.bindingChanged.connect(
            lambda var_name: self._on_procedure_binding_changed("stop_time", var_name)
        )
        time_form.addRow("Конец кадра stop_time (с):", self.spin_stop_time)

        cs_layout.addWidget(self.grp_time_interval)

        # Таблица Zipped-переменных
        self.grp_zipped = QGroupBox("Связанные переменные ракурсов (Zipped Variables, K×M)")
        grp_zipped_layout = QVBoxLayout(self.grp_zipped)
        grp_zipped_layout.setContentsMargins(4, 6, 4, 6)
        grp_zipped_layout.setSpacing(4)

        self.tbl_zipped_variables = QTableWidget(0, 0)
        self.tbl_zipped_variables.setMinimumHeight(130)
        self.tbl_zipped_variables.cellChanged.connect(self._on_zipped_cell_changed)
        grp_zipped_layout.addWidget(self.tbl_zipped_variables)

        btn_layout = QHBoxLayout()
        self.btn_add_zipped_col = QPushButton("➕ Переменная (столбец)")
        self.btn_add_zipped_col.clicked.connect(self._on_add_zipped_col_clicked)
        self.btn_remove_zipped_col = QPushButton("🗑 Удалить столбец")
        self.btn_remove_zipped_col.clicked.connect(self._on_remove_zipped_col_clicked)
        self.btn_add_zipped_row = QPushButton("➕ Шаг (строка)")
        self.btn_add_zipped_row.clicked.connect(self._on_add_zipped_row_clicked)
        self.btn_remove_zipped_row = QPushButton("➖ Удалить шаг")
        self.btn_remove_zipped_row.clicked.connect(self._on_remove_zipped_row_clicked)

        for btn in (
            self.btn_add_zipped_col,
            self.btn_remove_zipped_col,
            self.btn_add_zipped_row,
            self.btn_remove_zipped_row,
        ):
            btn_layout.addWidget(btn)
        grp_zipped_layout.addLayout(btn_layout)
        cs_layout.addWidget(self.grp_zipped)

        # Сетки Grid-переменных
        self.grp_grid = QGroupBox("Сетки декартова произведения (Grid Variables, N×1)")
        grp_grid_layout = QVBoxLayout(self.grp_grid)
        grp_grid_layout.setContentsMargins(4, 6, 4, 6)
        grp_grid_layout.setSpacing(4)

        self.btn_add_grid_var = QPushButton("➕ Создать Grid-переменную")
        self.btn_add_grid_var.clicked.connect(self._on_add_grid_var_clicked)
        grp_grid_layout.addWidget(self.btn_add_grid_var)

        self.grid_vars_container = QWidget()
        self.grid_vars_layout = QVBoxLayout(self.grid_vars_container)
        self.grid_vars_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_vars_layout.setSpacing(6)
        grp_grid_layout.addWidget(self.grid_vars_container)
        cs_layout.addWidget(self.grp_grid)

        return widget

    def can_handle(self, view_model: Any) -> bool:
        """Проверяет, является ли модель представления протоколом исследования."""
        return isinstance(view_model, BaseProcedureViewModel)

    def unbind(self) -> None:
        """Отвязывает текущую модель протокола и очищает таблицы переменных."""
        super().unbind()
        self.spin_proc_radius.unbind()
        self.spin_proc_time.unbind()
        self.spin_start_time.unbind()
        self.spin_stop_time.unbind()
        self.tbl_zipped_variables.setRowCount(0)
        self.tbl_zipped_variables.setColumnCount(0)
        for var_widget in list(self._grid_var_widgets.values()):
            self.grid_vars_layout.removeWidget(var_widget)
            var_widget.deleteLater()
        self._grid_var_widgets.clear()

    def _sync_bindable_fields(self, vm: BaseProcedureViewModel) -> None:
        """Синхронизирует состояние привязок BindableDoubleField с текущей ViewModel."""
        b_radius = vm.bindings.get("radius")
        if b_radius:
            self.spin_proc_radius.bind_variable(b_radius)
        else:
            self.spin_proc_radius.unbind()

        b_time = vm.bindings.get("step_time") or vm.bindings.get("time")
        if b_time:
            self.spin_proc_time.bind_variable(b_time)
        else:
            self.spin_proc_time.unbind()

        sim_mgr_bindings = vm.get_property_bindings("simulation_manager")
        b_start_time = vm.bindings.get("start_time") or sim_mgr_bindings.get("start_time")
        if b_start_time:
            self.spin_start_time.bind_variable(b_start_time)
        else:
            self.spin_start_time.unbind()

        b_stop_time = vm.bindings.get("stop_time") or sim_mgr_bindings.get("stop_time")
        if b_stop_time:
            self.spin_stop_time.bind_variable(b_stop_time)
        else:
            self.spin_stop_time.unbind()

    def update_ui_from_vm(self) -> None:
        """Синхронизирует элементы управления протоколом и таблицы с текущей ViewModel."""
        vm = self._current_vm
        if not isinstance(vm, BaseProcedureViewModel):
            return

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            self.spin_start_time.setValue(float(vm.start_time))
            self.spin_stop_time.setValue(float(vm.stop_time))
            self._sync_bindable_fields(vm)

            if isinstance(vm, StepAndShootProcedureViewModel):
                self._show_step_and_shoot(vm)
            elif isinstance(vm, CustomSweepProcedureViewModel):
                self.step_and_shoot_widget.setVisible(False)
                self.custom_sweep_widget.setVisible(True)
                self.lbl_proc_type.setText("Пользовательский скан (Custom Sweep)")
                self._populate_custom_variables_table(vm)
            elif isinstance(vm, PetProcedureViewModel):
                self._show_pet(vm)
            else:
                self.step_and_shoot_widget.setVisible(False)
                self.custom_sweep_widget.setVisible(False)
                self.lbl_proc_type.setText(vm.name)
        finally:
            self._is_updating_ui = old_state

    def _show_step_and_shoot(self, vm: StepAndShootProcedureViewModel) -> None:
        """Синхронизация полей Step-and-Shoot."""
        self.step_and_shoot_widget.setVisible(True)
        self.custom_sweep_widget.setVisible(True)
        self._set_step_controls_visible(True)
        self.lbl_proc_type.setText("Пошаговое сканирование (Step-and-Shoot)")
        self.spin_proc_steps.setValue(int(vm.steps))
        self.spin_proc_cameras.setValue(int(vm.gamma_cameras))
        self.lbl_proc_total_views.setText(str(vm.total_projections))
        self.spin_proc_radius.setValue(float(vm.radius))
        self.spin_proc_time.setValue(float(vm.time_per_step))
        self.spin_proc_initial_time.setValue(float(vm.initial_time))
        self.chk_proc_sequential_time.setChecked(bool(vm.sequential_time))
        self.spin_proc_start_angle.setValue(float(vm.start_angle))
        self.spin_proc_end_angle.setValue(float(vm.end_angle))
        idx = self.combo_proc_head_mode.findText(vm.head_mode)
        if idx >= 0:
            self.combo_proc_head_mode.setCurrentIndex(idx)
        self.txt_proc_head_angles.setText(", ".join(f"{angle:.1f}" for angle in vm.head_angles))
        self.chk_proc_endpoint.setChecked(bool(vm.endpoint))
        self._populate_custom_variables_table(vm)

    def _show_pet(self, vm: PetProcedureViewModel) -> None:
        """Синхронизация полей PET."""
        self.step_and_shoot_widget.setVisible(True)
        self.custom_sweep_widget.setVisible(False)
        self._set_step_controls_visible(False)
        self.lbl_proc_type.setText("ПЭТ (PET)")
        self.spin_proc_steps.setValue(1)
        self.spin_proc_cameras.setValue(int(vm.detector_heads))
        self.lbl_proc_total_views.setText(str(vm.detector_heads))
        self.spin_proc_radius.setValue(float(vm.ring_radius))
        self.spin_proc_time.setValue(float(vm.time_per_frame))

    def _set_step_controls_visible(self, visible: bool) -> None:
        """Управляет видимостью специфичных для Step-and-Shoot угловых и временных настроек."""
        self.spin_proc_initial_time.setVisible(visible)
        self.chk_proc_sequential_time.setVisible(visible)
        self.spin_proc_start_angle.setVisible(visible)
        self.spin_proc_end_angle.setVisible(visible)
        self.combo_proc_head_mode.setVisible(visible)
        self.txt_proc_head_angles.setVisible(visible)
        self.chk_proc_endpoint.setVisible(visible)

    def _populate_custom_variables_table(self, procedure_vm: CustomSweepProcedureViewModel) -> None:
        """Заполнение таблиц Zipped и Grid варьируемых переменных."""
        self._populate_zipped_table(procedure_vm)
        self._populate_grid_tables(procedure_vm)

    def _populate_zipped_table(self, procedure_vm: CustomSweepProcedureViewModel) -> None:
        """Заполнение таблицы связанных переменных из модели представления."""
        tbl = self.tbl_zipped_variables
        tbl.blockSignals(True)
        try:
            zipped_dict = procedure_vm.zipped_variables
            if not zipped_dict:
                tbl.setRowCount(0)
                tbl.setColumnCount(0)
                return

            if isinstance(procedure_vm, StepAndShootProcedureViewModel):
                std_order = ["gantry_angle", "radius", "step_time"]
                other = [
                    var_key
                    for var_key in zipped_dict
                    if var_key not in std_order and var_key not in ("current_angle", "current_time", "start_time", "stop_time")
                ]
                cols = [var_key for var_key in std_order if var_key in zipped_dict] + other
            else:
                cols = [var_key for var_key in zipped_dict if var_key not in ("current_angle", "current_time")]

            rows = len(next(iter(zipped_dict.values()))) if zipped_dict else 0
            tbl.setColumnCount(len(cols))
            tbl.setRowCount(rows)
            tbl.setVerticalHeaderLabels([f"Шаг {idx + 1}" for idx in range(rows)])

            for col_idx, col_name in enumerate(cols):
                is_ro = isinstance(procedure_vm, StepAndShootProcedureViewModel) and col_name in (
                    "gantry_angle",
                    "step_time",
                )
                item = QTableWidgetItem(f"🔒 {col_name}" if is_ro else col_name)
                item.setData(Qt.ItemDataRole.UserRole, col_name)
                if is_ro:
                    item.setToolTip("Стандартная переменная протокола: вычисляется из кинематических параметров")
                tbl.setHorizontalHeaderItem(col_idx, item)

                val_list = zipped_dict.get(col_name, [])
                for row_idx in range(rows):
                    val = val_list[row_idx] if row_idx < len(val_list) else 0.0
                    cell = QTableWidgetItem(f"{float(val):.4f}")
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    if is_ro:
                        cell.setFlags(flags)
                        cell.setForeground(QColor("#888888"))
                        cell.setBackground(QColor("#252830"))
                    else:
                        cell.setFlags(flags | Qt.ItemFlag.ItemIsEditable)
                    tbl.setItem(row_idx, col_idx, cell)
        finally:
            tbl.blockSignals(False)

    def _populate_grid_tables(self, procedure_vm: CustomSweepProcedureViewModel) -> None:
        """Обновление списка и значений чипов Grid-переменных."""
        grid_dict = procedure_vm.grid_variables
        for rem in set(self._grid_var_widgets.keys()) - set(grid_dict.keys()):
            widget = self._grid_var_widgets.pop(rem)
            self.grid_vars_layout.removeWidget(widget)
            widget.deleteLater()

        for idx in reversed(range(self.grid_vars_layout.count())):
            item = self.grid_vars_layout.itemAt(idx)
            child = item.widget() if item else None
            if isinstance(child, QLabel):
                self.grid_vars_layout.takeAt(idx)
                child.deleteLater()

        if not grid_dict:
            lbl = QLabel("Нет созданных Grid-переменных.")
            lbl.setStyleSheet("color: #888888; font-style: italic;")
            self.grid_vars_layout.addWidget(lbl)
            return

        for name, vals in grid_dict.items():
            if name in self._grid_var_widgets:
                self._grid_var_widgets[name].set_values(vals)
            else:
                chips = GridVariableChipsRow(name, vals, parent=self.grid_vars_container)
                chips.values_changed.connect(self._on_grid_values_changed)
                chips.variable_deleted.connect(self._on_grid_deleted)
                self.grid_vars_layout.addWidget(chips)
                self._grid_var_widgets[name] = chips

    def _on_proc_steps_changed(self, steps_val: int) -> None:
        """Обновляет количество шагов сканирования в StepAndShootProcedureViewModel."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.steps = steps_val
            self._is_updating_ui = True
            self.lbl_proc_total_views.setText(str(self._current_vm.total_projections))
            self._is_updating_ui = False
            self._populate_zipped_table(self._current_vm)

    def _on_proc_cameras_changed(self, cam_count: int) -> None:
        """Обновляет число гамма-камер в StepAndShootProcedureViewModel."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.gamma_cameras = cam_count
            self._is_updating_ui = True
            self.lbl_proc_total_views.setText(str(self._current_vm.total_projections))
            self.txt_proc_head_angles.setText(
                ", ".join(f"{angle_val:.1f}" for angle_val in self._current_vm.head_angles)
            )
            self._is_updating_ui = False

    def _on_proc_radius_changed(self, radius: float) -> None:
        """Обновляет радиус орбиты в ViewModel протокола."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.radius = radius
            self._populate_zipped_table(self._current_vm)
        elif not self._is_updating_ui and isinstance(self._current_vm, PetProcedureViewModel):
            self._current_vm.ring_radius = radius

    def _on_proc_time_changed(self, step_time: float) -> None:
        """Обновляет экспозицию на один ракурс/кадр в ViewModel протокола."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.time_per_step = step_time
            self._is_updating_ui = True
            self.spin_start_time.setValue(float(self._current_vm.start_time))
            self.spin_stop_time.setValue(float(self._current_vm.stop_time))
            self._is_updating_ui = False
            self._populate_zipped_table(self._current_vm)
        elif not self._is_updating_ui and isinstance(self._current_vm, PetProcedureViewModel):
            self._current_vm.time_per_frame = step_time

    def _on_proc_initial_time_changed(self, initial_time: float) -> None:
        """Обновляет начальное время смещения t₀ в StepAndShootProcedureViewModel."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.initial_time = initial_time
            self._is_updating_ui = True
            self.spin_start_time.setValue(float(self._current_vm.start_time))
            self.spin_stop_time.setValue(float(self._current_vm.stop_time))
            self._is_updating_ui = False
            self._populate_zipped_table(self._current_vm)

    def _on_proc_sequential_time_toggled(self, checked: bool) -> None:
        """Переключает флаг последовательного отсчета времени по шагам."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.sequential_time = checked
            self._is_updating_ui = True
            self.spin_start_time.setValue(float(self._current_vm.start_time))
            self.spin_stop_time.setValue(float(self._current_vm.stop_time))
            self._is_updating_ui = False
            self._populate_zipped_table(self._current_vm)

    def _on_proc_start_time_changed(self, start_time: float) -> None:
        """Обновляет время начала симуляции в ViewModel протокола."""
        if not self._is_updating_ui and isinstance(self._current_vm, BaseProcedureViewModel):
            self._current_vm.start_time = start_time

    def _on_proc_stop_time_changed(self, stop_time: float) -> None:
        """Обновляет время окончания симуляции в ViewModel протокола."""
        if not self._is_updating_ui and isinstance(self._current_vm, BaseProcedureViewModel):
            self._current_vm.stop_time = stop_time

    def _on_proc_start_angle_changed(self, angle: float) -> None:
        """Обновляет начальный угол вращения гентри."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.start_angle = angle
            self._populate_zipped_table(self._current_vm)

    def _on_proc_end_angle_changed(self, angle: float) -> None:
        """Обновляет конечный угол вращения гентри."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.end_angle = angle
            self._populate_zipped_table(self._current_vm)

    def _on_proc_head_mode_changed(self, text: str) -> None:
        """Обновляет режим взаимного расположения детекторных головок."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.head_mode = text
            self._is_updating_ui = True
            self.spin_proc_cameras.setValue(self._current_vm.gamma_cameras)
            self.txt_proc_head_angles.setText(
                ", ".join(f"{angle_val:.1f}" for angle_val in self._current_vm.head_angles)
            )
            self._is_updating_ui = False

    def _on_proc_head_angles_edited(self) -> None:
        """Разбирает и сохраняет пользовательские углы смещения детекторных головок."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            raw = self.txt_proc_head_angles.text().strip()
            if ";" in raw:
                parts = [part_item.strip().replace(",", ".") for part_item in raw.split(";") if part_item.strip()]
            else:
                parts = [
                    part_item.strip().replace(",", ".")
                    for part_item in re.split(r"[, \t]+", raw)
                    if part_item.strip()
                ]
            angles: list[float] = []
            for part in parts:
                if _FLOAT_PATTERN.match(part):
                    val = float(part)
                    if math.isfinite(val):
                        angles.append(val)
            if angles:
                self._current_vm.head_angles = angles
            else:
                self.txt_proc_head_angles.setText(
                    ", ".join(f"{angle_val:.1f}" for angle_val in self._current_vm.head_angles)
                )

    def _on_proc_endpoint_toggled(self, checked: bool) -> None:
        """Переключает флаг включения конечного угла в диапазон шагов сканирования."""
        if not self._is_updating_ui and isinstance(self._current_vm, StepAndShootProcedureViewModel):
            self._current_vm.endpoint = checked
            self._populate_zipped_table(self._current_vm)

    def _on_grid_values_changed(self, var_name: str, values: List[float]) -> None:
        """Обновляет список значений Grid-переменной в ViewModel."""
        vm = self._current_vm
        if isinstance(vm, CustomSweepProcedureViewModel):
            self._is_updating_ui = True
            try:
                vm.set_variable(var_name, values, sweep_type="grid")
            finally:
                self._is_updating_ui = False

    def _on_grid_deleted(self, var_name: str) -> None:
        """Удаляет Grid-переменную из ViewModel и обновляет панель чипов."""
        vm = self._current_vm
        if isinstance(vm, CustomSweepProcedureViewModel):
            self._is_updating_ui = True
            try:
                vm.remove_variable(var_name)
                self._sync_bindable_fields(vm)
            finally:
                self._is_updating_ui = False
            self._populate_grid_tables(vm)

    def _on_add_grid_var_clicked(self) -> None:
        """Создание новой Grid-переменной через диалоговое окно."""
        vm = self._current_vm
        if not isinstance(vm, CustomSweepProcedureViewModel):
            return
        name_in, ok = QInputDialog.getText(self, "Создать Grid-переменную", "Имя переменной сетки:")
        if not ok or not name_in.strip():
            return
        clean = name_in.strip().replace(" ", "_")
        if clean in vm.all_variables:
            QMessageBox.warning(self, "Ошибка", f"Переменная '{clean}' уже существует.")
            return
        vm.set_variable(clean, [0.0], sweep_type="grid")
        self._populate_grid_tables(vm)

    def _on_zipped_cell_changed(self, row_idx: int, col_idx: int) -> None:
        """Обработка изменения значения ячейки таблицы Zipped-переменных по принципу LBYL."""
        vm = self._current_vm
        if self._is_updating_ui or not isinstance(vm, CustomSweepProcedureViewModel):
            return
        tbl = self.tbl_zipped_variables
        hdr = tbl.horizontalHeaderItem(col_idx)
        cell = tbl.item(row_idx, col_idx)
        if hdr is None or cell is None:
            return
        raw_role = hdr.data(Qt.ItemDataRole.UserRole)
        var_name = str(raw_role) if raw_role else hdr.text().replace("🔒", "").strip()
        if isinstance(vm, StepAndShootProcedureViewModel) and vm.is_read_only_variable(var_name):
            return

        clean_text = cell.text().strip().replace(",", ".").replace(" ", "")
        if not _FLOAT_PATTERN.match(clean_text):
            self._populate_zipped_table(vm)
            return

        val = float(clean_text)
        if not math.isfinite(val):
            self._populate_zipped_table(vm)
            return

        if var_name == "radius":
            val = max(10.0, val)
        elif var_name == "step_time":
            val = max(0.001, val)

        if var_name in vm.zipped_variables:
            values = list(vm.zipped_variables[var_name])
            if 0 <= row_idx < len(values):
                values[row_idx] = val
                self._update_zipped_variable(vm, var_name, values)

    def _update_zipped_variable(self, vm: CustomSweepProcedureViewModel, var_name: str, values: List[float]) -> None:
        """Сохраняет обновленный вектор значений Zipped-переменной в ViewModel."""
        self._is_updating_ui = True
        try:
            vm.set_variable(var_name, values, sweep_type="zipped")
            if isinstance(vm, StepAndShootProcedureViewModel) and len(values) > 0:
                if var_name == "radius":
                    self.spin_proc_radius.setValue(values[0])
                elif var_name == "step_time":
                    self.spin_proc_time.setValue(values[0])
        finally:
            self._is_updating_ui = False

    def _on_add_zipped_col_clicked(self) -> None:
        """Добавление нового столбца связанной переменной."""
        vm = self._current_vm
        if not isinstance(vm, CustomSweepProcedureViewModel):
            return
        col_name, ok = QInputDialog.getText(self, "Добавить Zipped-переменную", "Имя связанной переменной:")
        if not ok or not col_name.strip():
            return
        clean = col_name.strip().replace(" ", "_")
        if clean in vm.all_variables:
            QMessageBox.warning(self, "Ошибка", f"Переменная '{clean}' уже существует.")
            return
        rows = (
            len(next(iter(vm.zipped_variables.values())))
            if vm.zipped_variables
            else (vm.steps if isinstance(vm, StepAndShootProcedureViewModel) else 1)
        )
        vm.set_variable(clean, [0.0] * rows, sweep_type="zipped")
        self._populate_zipped_table(vm)

    def _on_remove_zipped_col_clicked(self) -> None:
        """Удаление выбранного столбца связанной переменной."""
        vm = self._current_vm
        if not isinstance(vm, CustomSweepProcedureViewModel):
            return
        tbl = self.tbl_zipped_variables
        col = tbl.currentColumn() if tbl.currentColumn() >= 0 else tbl.columnCount() - 1
        if col < 0:
            return
        hdr = tbl.horizontalHeaderItem(col)
        if hdr is None:
            return
        raw_role = hdr.data(Qt.ItemDataRole.UserRole)
        var_name = str(raw_role) if raw_role else hdr.text().replace("🔒", "").strip()
        if isinstance(vm, StepAndShootProcedureViewModel) and vm.is_standard_variable(var_name):
            QMessageBox.warning(self, "Запрещено", f"Стандартная переменная '{var_name}' не может быть удалена.")
            return
        self._is_updating_ui = True
        try:
            vm.remove_variable(var_name)
            self._sync_bindable_fields(vm)
        finally:
            self._is_updating_ui = False
        self._populate_zipped_table(vm)

    def _on_add_zipped_row_clicked(self) -> None:
        """Добавление новой строки шага сканирования."""
        vm = self._current_vm
        if not isinstance(vm, CustomSweepProcedureViewModel):
            return
        if not vm.zipped_variables:
            QMessageBox.information(self, "Добавление шага", "Сначала создайте хотя бы одну переменную.")
            return
        vm.add_zipped_step()
        self._sync_step_and_shoot(vm)
        self._populate_zipped_table(vm)

    def _on_remove_zipped_row_clicked(self) -> None:
        """Удаление строки шага сканирования."""
        vm = self._current_vm
        if not isinstance(vm, CustomSweepProcedureViewModel) or not vm.zipped_variables:
            return
        rows = len(next(iter(vm.zipped_variables.values())))
        if rows <= 1:
            QMessageBox.information(self, "Удаление шага", "Должен оставаться как минимум 1 шаг симуляции.")
            return
        target = self.tbl_zipped_variables.currentRow()
        vm.remove_zipped_step(target if 0 <= target < rows else rows - 1)
        self._sync_step_and_shoot(vm)
        self._populate_zipped_table(vm)

    def _sync_step_and_shoot(self, vm: CustomSweepProcedureViewModel) -> None:
        """Синхронизирует счетчик шагов и общего числа проекций для Step-and-Shoot."""
        if isinstance(vm, StepAndShootProcedureViewModel):
            self._is_updating_ui = True
            try:
                self.spin_proc_steps.setValue(vm.steps)
                self.lbl_proc_total_views.setText(str(vm.total_projections))
            finally:
                self._is_updating_ui = False

    def _get_protocol_variables(self) -> List[str]:
        """Возвращает список доступных переменных из активной процедуры."""
        proc_vm: Optional[BaseProcedureViewModel] = None
        if isinstance(self._procedure_vm, BaseProcedureViewModel):
            proc_vm = self._procedure_vm
        elif isinstance(self._current_vm, BaseProcedureViewModel):
            proc_vm = self._current_vm
        if proc_vm is None:
            return []
        result: List[str] = []
        for std_var in proc_vm.protocol_standard_variables:
            if std_var not in result:
                result.append(std_var)
        if isinstance(proc_vm, CustomSweepProcedureViewModel):
            for var_key in proc_vm.all_variables.keys():
                if var_key not in result:
                    result.append(var_key)
        return result

    def _on_variable_created(self, var_name: str) -> None:
        """Создает новую переменную в протоколе при выборе 'Создать переменную...' из поля."""
        proc_vm: Optional[CustomSweepProcedureViewModel] = None
        if isinstance(self._procedure_vm, CustomSweepProcedureViewModel):
            proc_vm = self._procedure_vm
        elif isinstance(self._current_vm, CustomSweepProcedureViewModel):
            proc_vm = self._current_vm
        if proc_vm is not None and var_name not in proc_vm.all_variables:
            proc_vm.set_variable(var_name, [0.0], sweep_type="grid")
            if isinstance(self._current_vm, CustomSweepProcedureViewModel):
                self._populate_grid_tables(self._current_vm)

    def _on_procedure_binding_changed(self, prop_name: str, var_name: str) -> None:
        """Обновляет привязку параметра протокола к переменной в словаре bindings и реестре."""
        if self._is_updating_ui or not isinstance(self._current_vm, BaseProcedureViewModel):
            return
        if var_name:
            self._current_vm.bindings[prop_name] = str(var_name)
            if prop_name in ("start_time", "stop_time"):
                self._current_vm.bind_node_property("simulation_manager", prop_name, str(var_name))
        else:
            self._current_vm.bindings.pop(prop_name, None)
            if prop_name in ("start_time", "stop_time"):
                self._current_vm.unbind_node_property("simulation_manager", prop_name)


__all__ = ["ProcedureInspectorSection"]
