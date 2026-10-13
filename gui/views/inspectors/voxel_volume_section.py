"""
Секция параметров воксельного фантома (WoodcockVoxelVolume / VoxelVolumeViewModel).
"""
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
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
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

import settings.database_setting as database_setting
from gui.models.distribution_import_params import ImportTargetKind
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewport_3d.dicom_colormaps import get_available_colormaps
from gui.viewport_3d.material_palette import get_material_color
from gui.views.distribution_import_dialog import DistributionImportDialog
from gui.views.inspectors.base_section import BaseInspectorSection


class VoxelVolumeInspectorSection(BaseInspectorSection):
    """Секция настройки импорта фантомов, воксельного разрешения, палитры и таблицы NIST."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Параметры фантома (DICOM / Voxel)", parent)

    def _init_ui(self) -> None:
        form_layout = QFormLayout(self)

        self.txt_voxel_path = QLineEdit()
        self.txt_voxel_path.setReadOnly(True)
        self.btn_browse_voxel = QPushButton("Обзор...")
        self.btn_browse_voxel.clicked.connect(self._on_browse_phantom_file)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.txt_voxel_path)
        path_layout.addWidget(self.btn_browse_voxel)
        form_layout.addRow("Файл фантома:", path_layout)

        self.lbl_voxel_shape = QLabel("–")
        form_layout.addRow("Разрешение сетки:", self.lbl_voxel_shape)

        self.spin_voxel_size_x = self._create_voxel_spin()
        self.spin_voxel_size_y = self._create_voxel_spin()
        self.spin_voxel_size_z = self._create_voxel_spin()

        voxel_box = QHBoxLayout()
        for spin in (self.spin_voxel_size_x, self.spin_voxel_size_y, self.spin_voxel_size_z):
            voxel_box.addWidget(spin)
        form_layout.addRow("Шаг вокселей (X, Y, Z):", voxel_box)

        self.combo_colormap = QComboBox()
        self.combo_colormap.addItem("Physical Materials")
        self.combo_colormap.addItems(get_available_colormaps())
        self.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
        form_layout.addRow("Палитра:", self.combo_colormap)

        self.slider_lod = QSlider(Qt.Horizontal)
        self.slider_lod.setRange(1, 10)
        self.slider_lod.setValue(5)
        self.slider_lod.valueChanged.connect(self._on_lod_changed)
        form_layout.addRow("Детализация (LOD):", self.slider_lod)

        self.spin_opacity_thresh = QDoubleSpinBox()
        self.spin_opacity_thresh.setRange(0.0, 1.0)
        self.spin_opacity_thresh.setSingleStep(0.01)
        self.spin_opacity_thresh.setValue(0.05)
        self.spin_opacity_thresh.valueChanged.connect(self._on_opacity_threshold_changed)
        form_layout.addRow("Порог отсечения фона (Air Cutoff):", self.spin_opacity_thresh)

        self.spin_max_opacity = QDoubleSpinBox()
        self.spin_max_opacity.setRange(0.0, 1.0)
        self.spin_max_opacity.setSingleStep(0.05)
        self.spin_max_opacity.setValue(0.40)
        self.spin_max_opacity.valueChanged.connect(self._on_max_opacity_changed)
        form_layout.addRow("Непрозрачность (Opacity):", self.spin_max_opacity)

        self.combo_opacity_preset = QComboBox()
        for preset_label, preset_id in (
            ("Отсечение фона (Air Cutoff)", "air_cutoff"),
            ("Линейный (Linear)", "linear"),
            ("Мягкие ткани (Soft Tissue)", "soft_tissue"),
            ("Рентген / Полупрозрачный (X-Ray)", "xray_translucent"),
            ("Ступенчатый (Step)", "step"),
        ):
            self.combo_opacity_preset.addItem(preset_label, preset_id)
        self.combo_opacity_preset.currentIndexChanged.connect(self._on_opacity_preset_changed)
        form_layout.addRow("Карта прозрачности:", self.combo_opacity_preset)

        self.tbl_material_mapping = QTableWidget()
        self.tbl_material_mapping.setColumnCount(3)
        self.tbl_material_mapping.setHorizontalHeaderLabels(["ID", "Цвет", "Материал NIST"])
        self.tbl_material_mapping.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_material_mapping.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_material_mapping.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_material_mapping.setMinimumHeight(150)
        form_layout.addRow("Таблица материалов:", self.tbl_material_mapping)

    def _create_voxel_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 100.0)
        spin.setSingleStep(0.5)
        spin.setValue(4.0)
        spin.setSuffix(" мм")
        spin.valueChanged.connect(self._on_voxel_size_changed)
        return spin

    def can_handle(self, view_model: Any) -> bool:
        return isinstance(view_model, VoxelVolumeViewModel)

    def unbind(self) -> None:
        super().unbind()
        self.tbl_material_mapping.setRowCount(0)

    def update_ui_from_vm(self) -> None:
        vm = self._current_vm
        if not isinstance(vm, VoxelVolumeViewModel):
            return

        def _apply_vm_values() -> None:
            self.txt_voxel_path.setText(str(vm.file_path))
            self.lbl_voxel_shape.setText(f"{vm.dimensions[0]} × {vm.dimensions[1]} × {vm.dimensions[2]}")
            if len(vm.voxel_size) >= 3:
                self.spin_voxel_size_x.setValue(float(vm.voxel_size[0]))
                self.spin_voxel_size_y.setValue(float(vm.voxel_size[1]))
                self.spin_voxel_size_z.setValue(float(vm.voxel_size[2]))
            colormap_index = self.combo_colormap.findText(vm.colormap_name)
            if colormap_index >= 0:
                self.combo_colormap.setCurrentIndex(colormap_index)
            self._update_voxel_opacity_controls_state(vm.colormap_name)
            self.slider_lod.setValue(int(round(float(vm.lod_factor) * 5.0)))
            self.spin_opacity_thresh.setValue(float(vm.opacity_threshold))
            self.spin_max_opacity.setValue(float(vm.max_opacity))
            preset_idx = self.combo_opacity_preset.findData(vm.opacity_preset)
            if preset_idx >= 0:
                self.combo_opacity_preset.setCurrentIndex(preset_idx)
            self._update_material_mapping_table()

        self._run_ui_update(_apply_vm_values)

    def _run_ui_update(self, callback: Any) -> None:
        self._is_updating_ui = True
        try:
            callback()
        finally:
            self._is_updating_ui = False

    def on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        if not isinstance(self._current_vm, VoxelVolumeViewModel):
            return

        if prop_name == "colormap_name":
            colormap_index = self.combo_colormap.findText(str(new_val))
            if colormap_index >= 0:
                self._run_ui_update(lambda: (
                    self.combo_colormap.setCurrentIndex(colormap_index),
                    self._update_voxel_opacity_controls_state(str(new_val)),
                ))
        elif prop_name == "opacity_threshold":
            self._set_spin_safe(self.spin_opacity_thresh, float(new_val))
        elif prop_name == "max_opacity":
            self._set_spin_safe(self.spin_max_opacity, float(new_val))
        elif prop_name == "lod_factor":
            self._run_ui_update(lambda: self.slider_lod.setValue(int(round(float(new_val) * 5.0))))
        elif prop_name == "file_path":
            dims = self._current_vm.dimensions
            self._run_ui_update(lambda: (
                self.txt_voxel_path.setText(str(new_val)),
                self.lbl_voxel_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}"),
            ))
        elif prop_name == "material_distribution":
            self._update_material_mapping_table()
        elif prop_name == "voxel_size":
            self._update_voxel_size_fields(new_val)

    def _set_spin_safe(self, spin: QDoubleSpinBox, val: float) -> None:
        self._run_ui_update(lambda spin_widget=spin, new_value=val: spin_widget.setValue(new_value))

    def _update_voxel_size_fields(self, new_val: Any) -> None:
        if not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        volume_size = new_val if isinstance(new_val, (list, tuple, np.ndarray)) else self._current_vm.voxel_size
        if len(volume_size) >= 3:
            self._run_ui_update(lambda: (
                self.spin_voxel_size_x.setValue(float(volume_size[0])),
                self.spin_voxel_size_y.setValue(float(volume_size[1])),
                self.spin_voxel_size_z.setValue(float(volume_size[2])),
            ))

    def _update_voxel_opacity_controls_state(self, colormap_name: str) -> None:
        is_physical_mode = (str(colormap_name) == "Physical Materials")
        self.spin_opacity_thresh.setEnabled(not is_physical_mode)
        self.spin_max_opacity.setEnabled(not is_physical_mode)
        self.combo_opacity_preset.setEnabled(not is_physical_mode)

    def _on_colormap_changed(self, cmap_name: str) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        self._current_vm.colormap_name = cmap_name
        self._update_voxel_opacity_controls_state(cmap_name)

    def _on_lod_changed(self, value: int) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        self._current_vm.lod_factor = float(value) / 5.0

    def _on_opacity_threshold_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        self._current_vm.opacity_threshold = float(value)

    def _on_max_opacity_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        self._current_vm.max_opacity = float(value)

    def _on_opacity_preset_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        preset = self.combo_opacity_preset.currentData() or "air_cutoff"
        self._current_vm.opacity_preset = str(preset)

    def _on_voxel_size_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        self._current_vm.voxel_size = (
            float(self.spin_voxel_size_x.value()),
            float(self.spin_voxel_size_y.value()),
            float(self.spin_voxel_size_z.value()),
        )

    def _on_browse_phantom_file(self) -> None:
        """Диалог выбора и загрузки файла 3D-распределения воксельного фантома."""
        if not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл воксельного фантома",
            "",
            "Файлы данных (*.npy *.dat *.raw *.bin *.h5);;Все файлы (*.*)",
        )
        if not path:
            return

        dialog = DistributionImportDialog(
            parent=self,
            file_path=path,
            target_kind=ImportTargetKind.PHANTOM,
            default_voxel_size=self._current_vm.voxel_size,
        )
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_parameters()
            file_path_str = str(params.file_path)
            self.txt_voxel_path.setText(file_path_str)
            success = self._current_vm.reload_distribution(
                path=file_path_str,
                shape=params.shape,
                order=params.order,
                dtype=params.dtype,
                encoding=params.encoding,
                voxel_size=params.voxel_size,
                mapping=params.material_mapping,
                fill_value=params.fill_value,
            )
            if success:
                self._is_updating_ui = True
                try:
                    self.spin_voxel_size_x.setValue(params.voxel_size[0])
                    self.spin_voxel_size_y.setValue(params.voxel_size[1])
                    self.spin_voxel_size_z.setValue(params.voxel_size[2])
                    dims = self._current_vm.dimensions
                    self.lbl_voxel_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
                finally:
                    self._is_updating_ui = False

                if self._scene_vm is not None:
                    self._scene_vm.update_distribution_path(
                        self._current_vm.core_node,
                        file_path_str,
                        mapping=params.material_mapping,
                    )
                self._update_material_mapping_table()

    def _on_mapping_material_changed(self, material_id: int, material_name: str) -> None:
        """Обработчик смены материала NIST в строке таблицы маппинга фантома по принципу LBYL."""
        if self._is_updating_ui or not isinstance(self._current_vm, VoxelVolumeViewModel):
            return
        if material_id < 0 or material_id >= len(self._current_vm.material_list):
            return
        if material_name != "Vacuum" and material_name not in database_setting.material_database:
            return
        self._is_updating_ui = True
        try:
            self._current_vm.set_material_mapping(material_id, material_name)
        finally:
            self._is_updating_ui = False
        color_item = self.tbl_material_mapping.item(material_id, 1)
        if color_item is not None:
            red, green, blue = get_material_color(material_name)
            color_item.setBackground(QBrush(QColor.fromRgbF(red, green, blue)))
        if self._scene_vm is not None:
            mapping_dict = {
                float(idx): mat.name
                for idx, mat in enumerate(self._current_vm.material_list)
            }
            self._scene_vm.update_distribution_mapping(self._current_vm.core_node, mapping_dict)

    def _update_material_mapping_table(self) -> None:
        """Заполнение таблицы сопоставления дискретных ID фантома материалам базы NIST."""
        if not isinstance(self._current_vm, VoxelVolumeViewModel):
            self.tbl_material_mapping.setRowCount(0)
            return
        materials = self._current_vm.material_list
        all_materials = sorted(database_setting.material_database.keys())
        self.tbl_material_mapping.blockSignals(True)
        try:
            self.tbl_material_mapping.setRowCount(len(materials))
            for material_id, material in enumerate(materials):
                id_item = QTableWidgetItem(str(material_id))
                id_item.setTextAlignment(Qt.AlignCenter)
                id_item.setFlags(Qt.ItemIsEnabled)
                self.tbl_material_mapping.setItem(material_id, 0, id_item)

                color_item = QTableWidgetItem()
                red, green, blue = get_material_color(material.name)
                color_item.setBackground(QBrush(QColor.fromRgbF(red, green, blue)))
                color_item.setFlags(Qt.ItemIsEnabled)
                self.tbl_material_mapping.setItem(material_id, 1, color_item)

                material_combo = QComboBox()
                material_combo.addItems(all_materials)
                found_index = material_combo.findText(material.name)
                if found_index >= 0:
                    material_combo.setCurrentIndex(found_index)
                material_combo.currentTextChanged.connect(
                    lambda text, target_material_id=material_id: self._on_mapping_material_changed(
                        target_material_id, text
                    )
                )
                self.tbl_material_mapping.setCellWidget(material_id, 2, material_combo)
        finally:
            self.tbl_material_mapping.blockSignals(False)
