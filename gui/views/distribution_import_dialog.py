import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QDialogButtonBox,
    QWidget, QHeaderView, QMessageBox
)

from core.data.distribution_loader import DistributionLoader
import settings.database_setting as database_setting
from gui.viewport_3d.material_palette import get_material_color
from gui.models.distribution_import_params import DistributionImportParameters, ImportTargetKind

_logger = logging.getLogger(__name__)

SUPPORTED_DTYPES: Dict[str, np.dtype] = {
    "float32 (4 байта)": np.dtype(np.float32),
    "float64 (8 байт)": np.dtype(np.float64),
    "int16 (2 байта, DICOM/HU)": np.dtype(np.int16),
    "uint16 (2 байта)": np.dtype(np.uint16),
    "int32 (4 байта)": np.dtype(np.int32),
    "uint8 (1 байт)": np.dtype(np.uint8),
}


class DistributionImportDialog(QDialog):
    """
    Универсальное диалоговое окно импорта воксельных распределений.
    Обеспечивает ввод метаданных формата (размерность, порядок развертки, тип данных),
    настройку геометрии сетки (шаг вокселей) и конфигурацию маппинга материалов/активности.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        target_kind: ImportTargetKind,
        current_voxel_size: Optional[Union[float, Tuple[float, float, float], Sequence[float]]] = None,
        default_voxel_size: Optional[Union[float, Tuple[float, float, float], Sequence[float]]] = None,
        current_shape: Optional[Tuple[int, ...]] = None,
        existing_mapping: Optional[Dict[float, str]] = None,
        scene_phantom_dims: Optional[Tuple[int, int, int]] = None,
        scene_phantom_voxel_size: Optional[Tuple[float, float, float]] = None,
        scene_phantom_node: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.file_path = Path(file_path)
        self.target_kind = target_kind

        effective_voxel_size = default_voxel_size if default_voxel_size is not None else current_voxel_size
        if effective_voxel_size is None:
            effective_voxel_size = 1.0
        self.initial_voxel_size = effective_voxel_size

        self.initial_shape = current_shape or (128, 128, 92)
        self.existing_mapping = existing_mapping or {}

        if scene_phantom_node is not None:
            if scene_phantom_dims is None:
                scene_phantom_dims = scene_phantom_node.dimensions
            if scene_phantom_voxel_size is None:
                scene_phantom_voxel_size = scene_phantom_node.voxel_size

        self.scene_phantom_dims = scene_phantom_dims
        self.scene_phantom_voxel_size = scene_phantom_voxel_size

        self.metadata = DistributionLoader.inspect_metadata(self.file_path)
        self.is_npy_format = bool(self.metadata.get("is_npy", False))

        kind_title = "воксельного фантома" if target_kind == ImportTargetKind.PHANTOM else "распределения источника"
        self.setWindowTitle(f"Импорт {kind_title} — {self.file_path.name}")
        self.resize(680, 720)
        self.setModal(True)

        self._init_ui()
        self._apply_initial_metadata()
        self._validate_buffer_size()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Информационная плашка файла
        file_size_bytes = int(self.metadata.get("file_size", 0))
        file_size_str = f"{file_size_bytes:,} байт ({file_size_bytes / (1024 * 1024):.2f} МБ)"
        lbl_file_info = QLabel(f"<b>Файл:</b> {self.file_path.name} &nbsp;|&nbsp; <b>Размер:</b> {file_size_str}")
        lbl_file_info.setStyleSheet("background-color: #2b2b2b; padding: 6px; border-radius: 4px; color: #e0e0e0;")
        main_layout.addWidget(lbl_file_info)

        # 1. Блок формата данных (Data Format)
        self.grp_format = QGroupBox("1. Формат сырых данных (Data Format)")
        form_format = QFormLayout(self.grp_format)

        dim_layout = QHBoxLayout()
        self.spin_dim_x = QSpinBox()
        self.spin_dim_x.setRange(1, 8192)
        self.spin_dim_x.setValue(self.initial_shape[0] if len(self.initial_shape) > 0 else 128)
        self.spin_dim_x.valueChanged.connect(self._on_format_changed)

        self.spin_dim_y = QSpinBox()
        self.spin_dim_y.setRange(1, 8192)
        self.spin_dim_y.setValue(self.initial_shape[1] if len(self.initial_shape) > 1 else 128)
        self.spin_dim_y.valueChanged.connect(self._on_format_changed)

        self.spin_dim_z = QSpinBox()
        self.spin_dim_z.setRange(1, 8192)
        self.spin_dim_z.setValue(self.initial_shape[2] if len(self.initial_shape) > 2 else 92)
        self.spin_dim_z.valueChanged.connect(self._on_format_changed)

        dim_layout.addWidget(QLabel("X:"))
        dim_layout.addWidget(self.spin_dim_x)
        dim_layout.addWidget(QLabel("Y:"))
        dim_layout.addWidget(self.spin_dim_y)
        dim_layout.addWidget(QLabel("Z:"))
        dim_layout.addWidget(self.spin_dim_z)
        form_format.addRow("Размерность сетки (вокселей):", dim_layout)

        order_layout = QHBoxLayout()
        self.radio_order_f = QRadioButton("Fortran-order ('F', по столбцам / колумнальный)")
        self.radio_order_c = QRadioButton("C-order ('C', по строкам)")
        self.radio_order_f.setChecked(True)
        self.order_group = QButtonGroup(self)
        self.order_group.addButton(self.radio_order_f)
        self.order_group.addButton(self.radio_order_c)
        self.radio_order_f.toggled.connect(self._on_format_changed)
        order_layout.addWidget(self.radio_order_f)
        order_layout.addWidget(self.radio_order_c)
        form_format.addRow("Порядок развертки в памяти:", order_layout)

        self.combo_dtype = QComboBox()
        for dtype_label, dtype_val in SUPPORTED_DTYPES.items():
            self.combo_dtype.addItem(dtype_label, dtype_val)
        self.combo_dtype.currentIndexChanged.connect(self._on_format_changed)
        form_format.addRow("Тип элементов (Data Type):", self.combo_dtype)

        self.combo_encoding = QComboBox()
        self.combo_encoding.addItem("Бинарный (binary)", "binary")
        self.combo_encoding.addItem("Текстовый ASCII (text)", "text")
        if self.file_path.suffix.lower() in ('.txt', '.csv'):
            self.combo_encoding.setCurrentIndex(1)
        self.combo_encoding.currentIndexChanged.connect(self._on_format_changed)
        form_format.addRow("Режим кодирования:", self.combo_encoding)

        self.lbl_lbyl_status = QLabel("-")
        self.lbl_lbyl_status.setWordWrap(True)
        form_format.addRow("Валидация размера буфера:", self.lbl_lbyl_status)

        main_layout.addWidget(self.grp_format)

        # 2. Блок геометрии (Geometry)
        self.grp_geometry = QGroupBox("2. Геометрия сетки (Geometry)")
        form_geom = QFormLayout(self.grp_geometry)

        if self.target_kind == ImportTargetKind.PHANTOM:
            vox_layout = QHBoxLayout()
            if isinstance(self.initial_voxel_size, (tuple, list, np.ndarray)):
                if len(self.initial_voxel_size) >= 3:
                    initial_v_steps = (float(self.initial_voxel_size[0]), float(self.initial_voxel_size[1]), float(self.initial_voxel_size[2]))
                else:
                    initial_v_steps = (float(self.initial_voxel_size[0]), float(self.initial_voxel_size[0]), float(self.initial_voxel_size[0]))
            else:
                scalar_step = float(self.initial_voxel_size)
                initial_v_steps = (scalar_step, scalar_step, scalar_step)

            self.spin_vox_x = QDoubleSpinBox()
            self.spin_vox_x.setRange(0.01, 100.0)
            self.spin_vox_x.setSingleStep(0.5)
            self.spin_vox_x.setValue(initial_v_steps[0])
            self.spin_vox_x.setSuffix(" мм")
            self.spin_vox_x.valueChanged.connect(self._update_geometry_metrics)

            self.spin_vox_y = QDoubleSpinBox()
            self.spin_vox_y.setRange(0.01, 100.0)
            self.spin_vox_y.setSingleStep(0.5)
            self.spin_vox_y.setValue(initial_v_steps[1])
            self.spin_vox_y.setSuffix(" мм")
            self.spin_vox_y.valueChanged.connect(self._update_geometry_metrics)

            self.spin_vox_z = QDoubleSpinBox()
            self.spin_vox_z.setRange(0.01, 100.0)
            self.spin_vox_z.setSingleStep(0.5)
            self.spin_vox_z.setValue(initial_v_steps[2])
            self.spin_vox_z.setSuffix(" мм")
            self.spin_vox_z.valueChanged.connect(self._update_geometry_metrics)

            vox_layout.addWidget(QLabel("X:"))
            vox_layout.addWidget(self.spin_vox_x)
            vox_layout.addWidget(QLabel("Y:"))
            vox_layout.addWidget(self.spin_vox_y)
            vox_layout.addWidget(QLabel("Z:"))
            vox_layout.addWidget(self.spin_vox_z)
            form_geom.addRow("Шаг вокселей (X, Y, Z):", vox_layout)
        else:
            if isinstance(self.initial_voxel_size, (tuple, list, np.ndarray)):
                initial_v_step = float(self.initial_voxel_size[0])
            else:
                initial_v_step = float(self.initial_voxel_size)

            self.spin_source_voxel_step = QDoubleSpinBox()
            self.spin_source_voxel_step.setRange(0.01, 100.0)
            self.spin_source_voxel_step.setSingleStep(0.5)
            self.spin_source_voxel_step.setValue(initial_v_step)
            self.spin_source_voxel_step.setSuffix(" мм")
            self.spin_source_voxel_step.valueChanged.connect(self._update_geometry_metrics)
            form_geom.addRow("Шаг вокселя источника:", self.spin_source_voxel_step)

            if self.scene_phantom_dims is not None and self.scene_phantom_voxel_size is not None:
                btn_copy_phantom_geom = QPushButton("⇲ Скопировать геометрию фантома сцены")
                btn_copy_phantom_geom.clicked.connect(self._on_copy_phantom_geometry)
                form_geom.addRow(btn_copy_phantom_geom)

        self.lbl_volume_size = QLabel("-")
        form_geom.addRow("Габариты объема (X×Y×Z):", self.lbl_volume_size)
        main_layout.addWidget(self.grp_geometry)

        # 3. Специализированный блок (Маппинг для фантома / Активность для источника)
        if self.target_kind == ImportTargetKind.PHANTOM:
            self._init_phantom_mapping_ui(main_layout)
        else:
            self._init_source_activity_ui(main_layout)

        # Кнопки диалога
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self._update_geometry_metrics()

    def _init_phantom_mapping_ui(self, main_layout: QVBoxLayout) -> None:
        self.grp_mapping = QGroupBox("3. Универсальный маппинг материалов (Material Mapping)")
        map_layout = QVBoxLayout(self.grp_mapping)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Материал фона / по умолчанию:"))
        self.combo_default_material = QComboBox()
        all_materials = ["Vacuum"] + sorted(database_setting.material_database.keys())
        self.combo_default_material.addItems(all_materials)
        self.combo_default_material.setCurrentText("Vacuum")
        top_bar.addWidget(self.combo_default_material)
        top_bar.addStretch()

        self.btn_auto_scan = QPushButton("⚡ Автосканирование значений в файле")
        self.btn_auto_scan.clicked.connect(self._on_auto_scan_materials)
        top_bar.addWidget(self.btn_auto_scan)
        map_layout.addLayout(top_bar)

        self.tbl_mapping = QTableWidget()
        self.tbl_mapping.setColumnCount(5)
        self.tbl_mapping.setHorizontalHeaderLabels([
            "Значение (ID)", "Вокселей", "Доля (%)", "Цвет", "Материал NIST"
        ])
        header = self.tbl_mapping.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.tbl_mapping.setMinimumHeight(180)
        map_layout.addWidget(self.tbl_mapping)

        btn_row_layout = QHBoxLayout()
        btn_add_row = QPushButton("+ Добавить значение")
        btn_add_row.clicked.connect(self._on_add_mapping_row)
        btn_remove_row = QPushButton("- Удалить выбранное")
        btn_remove_row.clicked.connect(self._on_remove_mapping_row)
        btn_row_layout.addWidget(btn_add_row)
        btn_row_layout.addWidget(btn_remove_row)
        btn_row_layout.addStretch()
        map_layout.addLayout(btn_row_layout)

        main_layout.addWidget(self.grp_mapping)

        # Первичное заполнение существующего маппинга
        if self.existing_mapping:
            self._populate_mapping_table(self.existing_mapping)

    def _init_source_activity_ui(self, main_layout: QVBoxLayout) -> None:
        self.grp_source = QGroupBox("3. Параметры распределения активности источника")
        form_source = QFormLayout(self.grp_source)

        self.spin_source_total_activity = QDoubleSpinBox()
        self.spin_source_total_activity.setRange(0.001, 1_000_000.0)
        self.spin_source_total_activity.setSingleStep(10.0)
        self.spin_source_total_activity.setValue(100.0)
        self.spin_source_total_activity.setSuffix(" МБк")
        form_source.addRow("Полная интегральная активность:", self.spin_source_total_activity)

        self.spin_source_threshold = QDoubleSpinBox()
        self.spin_source_threshold.setRange(0.0, 1.0)
        self.spin_source_threshold.setSingleStep(0.01)
        self.spin_source_threshold.setValue(0.0)
        self.spin_source_threshold.setToolTip("Воксели со значением ниже порога отсекаются (устанавливаются в 0)")
        form_source.addRow("Порог отсечения фона (шума):", self.spin_source_threshold)

        self.lbl_source_stats = QLabel("Статистика файла: нажмите 'Проверить файл' для анализа")
        self.lbl_source_stats.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        form_source.addRow(self.lbl_source_stats)

        btn_inspect_source = QPushButton("⚡ Проверить распределение и статистику")
        btn_inspect_source.clicked.connect(self._on_inspect_source_distribution)
        form_source.addRow(btn_inspect_source)

        main_layout.addWidget(self.grp_source)

    def _apply_initial_metadata(self) -> None:
        """Применяет автоматически распознанные метаданные при наличии .npy файла."""
        if self.is_npy_format:
            shape_meta = self.metadata.get("shape")
            order_meta = self.metadata.get("order")
            dtype_meta = self.metadata.get("dtype")

            if shape_meta and len(shape_meta) >= 3:
                self.spin_dim_x.setValue(int(shape_meta[0]))
                self.spin_dim_y.setValue(int(shape_meta[1]))
                self.spin_dim_z.setValue(int(shape_meta[2]))

            if order_meta == "F":
                self.radio_order_f.setChecked(True)
            else:
                self.radio_order_c.setChecked(True)

            if dtype_meta is not None:
                for idx in range(self.combo_dtype.count()):
                    item_dt = self.combo_dtype.itemData(idx)
                    if item_dt == dtype_meta or item_dt == np.dtype(dtype_meta):
                        self.combo_dtype.setCurrentIndex(idx)
                        break

            # Блокируем поля формата для .npy
            self.spin_dim_x.setEnabled(False)
            self.spin_dim_y.setEnabled(False)
            self.spin_dim_z.setEnabled(False)
            self.radio_order_f.setEnabled(False)
            self.radio_order_c.setEnabled(False)
            self.combo_dtype.setEnabled(False)
            self.combo_encoding.setEnabled(False)
            self.grp_format.setTitle("1. Формат данных (.npy — метаданные считаны автоматически)")

    def _on_format_changed(self) -> None:
        self._update_geometry_metrics()
        self._validate_buffer_size()

    def _update_geometry_metrics(self) -> None:
        dim_x = self.spin_dim_x.value()
        dim_y = self.spin_dim_y.value()
        dim_z = self.spin_dim_z.value()

        if self.target_kind == ImportTargetKind.PHANTOM:
            step_x = self.spin_vox_x.value()
            step_y = self.spin_vox_y.value()
            step_z = self.spin_vox_z.value()
        else:
            step_x = step_y = step_z = self.spin_source_voxel_step.value()

        span_x = dim_x * step_x
        span_y = dim_y * step_y
        span_z = dim_z * step_z
        self.lbl_volume_size.setText(
            f"{span_x:.1f} × {span_y:.1f} × {span_z:.1f} мм "
            f"(центр: {-span_x/2:.1f}, {-span_y/2:.1f}, {-span_z/2:.1f} мм)"
        )

    def _validate_buffer_size(self) -> None:
        """Строгая LBYL-проверка размера файла на диске относительно формы и типа данных."""
        if self.is_npy_format:
            self.lbl_lbyl_status.setText("<font color='#4CAF50'>✔ Контейнер NumPy (.npy) валидирован</font>")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            return

        encoding = self.combo_encoding.currentData()
        if encoding == "text":
            self.lbl_lbyl_status.setText("<font color='#FFC107'>ℹ Текстовый режим: проверка количества элементов при чтении</font>")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            return

        dim_x = self.spin_dim_x.value()
        dim_y = self.spin_dim_y.value()
        dim_z = self.spin_dim_z.value()
        total_elements = dim_x * dim_y * dim_z

        chosen_dtype = self.combo_dtype.currentData() or np.dtype(np.float32)
        expected_bytes = total_elements * chosen_dtype.itemsize
        actual_bytes = int(self.metadata.get("file_size", 0))

        if expected_bytes == actual_bytes:
            self.lbl_lbyl_status.setText(
                f"<font color='#4CAF50'>✔ Точное совпадение: {actual_bytes:,} байт "
                f"({dim_x}×{dim_y}×{dim_z} × {chosen_dtype.itemsize} байт)</font>"
            )
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            diff = actual_bytes - expected_bytes
            self.lbl_lbyl_status.setText(
                f"<font color='#F44336'>⚠ Несовпадение размера: файл {actual_bytes:,} байт, "
                f"ожидается {expected_bytes:,} байт (разница: {diff:+,} байт)</font>"
            )
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def _on_copy_phantom_geometry(self) -> None:
        """Синхронизация размеров и шага источника с фантомом сцены."""
        if self.scene_phantom_dims is not None:
            self.spin_dim_x.setValue(int(self.scene_phantom_dims[0]))
            self.spin_dim_y.setValue(int(self.scene_phantom_dims[1]))
            self.spin_dim_z.setValue(int(self.scene_phantom_dims[2]))
        if self.scene_phantom_voxel_size is not None:
            avg_step = float(np.mean(self.scene_phantom_voxel_size))
            self.spin_source_voxel_step.setValue(avg_step)
        self._on_format_changed()

    def _load_preview_data(self) -> Optional[np.ndarray]:
        """Загрузка массива данных с текущими настройками для предпросмотра."""
        shape = (self.spin_dim_x.value(), self.spin_dim_y.value(), self.spin_dim_z.value())
        order = "F" if self.radio_order_f.isChecked() else "C"
        dtype_val = self.combo_dtype.currentData() or np.dtype(np.float32)
        encoding_val = self.combo_encoding.currentData() or "binary"

        try:
            data = DistributionLoader.load(
                self.file_path,
                target_shape=shape,
                order=order,
                dtype=dtype_val,
                encoding=encoding_val,
            )
            return data
        except Exception as load_err:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось прочитать файл с текущими параметрами:\n{load_err}")
            return None

    def _on_auto_scan_materials(self) -> None:
        """Сканирование массива фантома и автоматическое извлечение уникальных значений."""
        data = self._load_preview_data()
        if data is None:
            return

        unique_vals, counts = np.unique(data, return_counts=True)
        total_voxels = data.size

        # Ограничение на случай непрерывных данных
        if len(unique_vals) > 256:
            reply = QMessageBox.question(
                self,
                "Большое число уникальных значений",
                f"В файле обнаружено {len(unique_vals):,} уникальных вещественных значений.\n"
                f"Отобразить первые 50 наиболее частых значений материалов?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return
            sorted_indices = np.argsort(-counts)[:50]
            unique_vals = unique_vals[sorted_indices]
            counts = counts[sorted_indices]

        # Заполняем таблицу
        all_materials = ["Vacuum"] + sorted(database_setting.material_database.keys())
        default_names = ["Vacuum", "Water, Liquid", "Tissue, Soft (ICRU-44)", "Bone, Cortical (ICRU-44)", "Lung (ICRP)", "Adipose Tissue (ICRU-44)"]

        self.tbl_mapping.setRowCount(len(unique_vals))
        for row_idx, (val_item, count_item) in enumerate(zip(unique_vals, counts)):
            val_float = float(val_item)
            val_str = f"{val_float:.4g}"

            item_id = QTableWidgetItem(val_str)
            item_id.setTextAlignment(Qt.AlignCenter)
            self.tbl_mapping.setItem(row_idx, 0, item_id)

            item_count = QTableWidgetItem(f"{count_item:,}")
            item_count.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item_count.setFlags(Qt.ItemIsEnabled)
            self.tbl_mapping.setItem(row_idx, 1, item_count)

            pct_val = (count_item / total_voxels) * 100.0
            item_pct = QTableWidgetItem(f"{pct_val:.2f}%")
            item_pct.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item_pct.setFlags(Qt.ItemIsEnabled)
            self.tbl_mapping.setItem(row_idx, 2, item_pct)

            # Определение имени материала
            assigned_mat = self.existing_mapping.get(val_float)
            if not assigned_mat:
                if row_idx < len(default_names):
                    assigned_mat = default_names[row_idx]
                else:
                    assigned_mat = all_materials[row_idx % len(all_materials)]

            color_item = QTableWidgetItem()
            red, green, blue = get_material_color(assigned_mat)
            color_item.setBackground(QBrush(QColor.fromRgbF(red, green, blue)))
            color_item.setFlags(Qt.ItemIsEnabled)
            self.tbl_mapping.setItem(row_idx, 3, color_item)

            combo_mat = QComboBox()
            combo_mat.addItems(all_materials)
            found_idx = combo_mat.findText(assigned_mat)
            if found_idx >= 0:
                combo_mat.setCurrentIndex(found_idx)
            combo_mat.currentTextChanged.connect(
                lambda text, target_row=row_idx: self._on_table_material_changed(target_row, text)
            )
            self.tbl_mapping.setCellWidget(row_idx, 4, combo_mat)

    def _populate_mapping_table(self, mapping_dict: Dict[float, str]) -> None:
        """Заполнение таблицы из существующего словаря mapping."""
        all_materials = ["Vacuum"] + sorted(database_setting.material_database.keys())
        self.tbl_mapping.setRowCount(len(mapping_dict))
        for row_idx, (val_float, mat_name) in enumerate(sorted(mapping_dict.items())):
            item_id = QTableWidgetItem(f"{float(val_float):.4g}")
            item_id.setTextAlignment(Qt.AlignCenter)
            self.tbl_mapping.setItem(row_idx, 0, item_id)

            item_count = QTableWidgetItem("–")
            item_count.setTextAlignment(Qt.AlignCenter)
            item_count.setFlags(Qt.ItemIsEnabled)
            self.tbl_mapping.setItem(row_idx, 1, item_count)

            item_pct = QTableWidgetItem("–")
            item_pct.setTextAlignment(Qt.AlignCenter)
            item_pct.setFlags(Qt.ItemIsEnabled)
            self.tbl_mapping.setItem(row_idx, 2, item_pct)

            color_item = QTableWidgetItem()
            red, green, blue = get_material_color(mat_name)
            color_item.setBackground(QBrush(QColor.fromRgbF(red, green, blue)))
            color_item.setFlags(Qt.ItemIsEnabled)
            self.tbl_mapping.setItem(row_idx, 3, color_item)

            combo_mat = QComboBox()
            combo_mat.addItems(all_materials)
            found_idx = combo_mat.findText(mat_name)
            if found_idx >= 0:
                combo_mat.setCurrentIndex(found_idx)
            combo_mat.currentTextChanged.connect(
                lambda text, target_row=row_idx: self._on_table_material_changed(target_row, text)
            )
            self.tbl_mapping.setCellWidget(row_idx, 4, combo_mat)

    def _on_table_material_changed(self, row_idx: int, mat_name: str) -> None:
        color_item = self.tbl_mapping.item(row_idx, 3)
        if color_item is not None:
            red, green, blue = get_material_color(mat_name)
            color_item.setBackground(QBrush(QColor.fromRgbF(red, green, blue)))

    def _on_add_mapping_row(self) -> None:
        all_materials = ["Vacuum"] + sorted(database_setting.material_database.keys())
        current_rows = self.tbl_mapping.rowCount()
        self.tbl_mapping.insertRow(current_rows)

        item_id = QTableWidgetItem(str(float(current_rows)))
        item_id.setTextAlignment(Qt.AlignCenter)
        self.tbl_mapping.setItem(current_rows, 0, item_id)

        item_count = QTableWidgetItem("–")
        item_count.setTextAlignment(Qt.AlignCenter)
        item_count.setFlags(Qt.ItemIsEnabled)
        self.tbl_mapping.setItem(current_rows, 1, item_count)

        item_pct = QTableWidgetItem("–")
        item_pct.setTextAlignment(Qt.AlignCenter)
        item_pct.setFlags(Qt.ItemIsEnabled)
        self.tbl_mapping.setItem(current_rows, 2, item_pct)

        default_mat = all_materials[current_rows % len(all_materials)]
        color_item = QTableWidgetItem()
        red, green, blue = get_material_color(default_mat)
        color_item.setBackground(QBrush(QColor.fromRgbF(red, green, blue)))
        color_item.setFlags(Qt.ItemIsEnabled)
        self.tbl_mapping.setItem(current_rows, 3, color_item)

        combo_mat = QComboBox()
        combo_mat.addItems(all_materials)
        found_idx = combo_mat.findText(default_mat)
        if found_idx >= 0:
            combo_mat.setCurrentIndex(found_idx)
        combo_mat.currentTextChanged.connect(
            lambda text, target_row=current_rows: self._on_table_material_changed(target_row, text)
        )
        self.tbl_mapping.setCellWidget(current_rows, 4, combo_mat)

    def _on_remove_mapping_row(self) -> None:
        selected_rows = sorted(set(idx.row() for idx in self.tbl_mapping.selectedIndexes()), reverse=True)
        for row_idx in selected_rows:
            self.tbl_mapping.removeRow(row_idx)

    def _on_inspect_source_distribution(self) -> None:
        """Инспекция распределения активности источника и расчет статистики."""
        data = self._load_preview_data()
        if data is None:
            return
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        sum_val = float(np.sum(data))
        active_count = int(np.count_nonzero(data > self.spin_source_threshold.value()))
        total_count = data.size
        pct_active = (active_count / total_count) * 100.0

        self.lbl_source_stats.setText(
            f"<b>Диапазон:</b> [{min_val:.4g} .. {max_val:.4g}] &nbsp;|&nbsp; "
            f"<b>Сумма:</b> {sum_val:.4g}<br>"
            f"<b>Активных вокселей:</b> {active_count:,} ({pct_active:.1f}%) из {total_count:,}"
        )
        self.lbl_source_stats.setStyleSheet("color: #7fba00; font-size: 11px;")

    def get_import_parameters(self) -> DistributionImportParameters:
        """
        Формирует результирующий датакласс DistributionImportParameters
        со всеми проверенными параметрами импорта.
        """
        shape = (self.spin_dim_x.value(), self.spin_dim_y.value(), self.spin_dim_z.value())
        order = "F" if self.radio_order_f.isChecked() else "C"
        chosen_dtype = self.combo_dtype.currentData() or np.dtype(np.float32)
        encoding = self.combo_encoding.currentData() or "binary"

        if self.target_kind == ImportTargetKind.PHANTOM:
            voxel_size: Union[float, Tuple[float, float, float]] = (
                self.spin_vox_x.value(),
                self.spin_vox_y.value(),
                self.spin_vox_z.value()
            )
            mapping_dict: Dict[float, str] = {}
            for row_idx in range(self.tbl_mapping.rowCount()):
                id_item = self.tbl_mapping.item(row_idx, 0)
                combo_widget = self.tbl_mapping.cellWidget(row_idx, 4)
                if id_item is not None and isinstance(combo_widget, QComboBox):
                    try:
                        val_float = float(id_item.text().strip())
                        mapping_dict[val_float] = combo_widget.currentText()
                    except ValueError:
                        pass
            fill_value = self.combo_default_material.currentText()
            total_activity = None
            noise_threshold = None
        else:
            voxel_size = self.spin_source_voxel_step.value()
            mapping_dict = None
            fill_value = "Vacuum"
            total_activity = self.spin_source_total_activity.value()
            noise_threshold = self.spin_source_threshold.value()

        return DistributionImportParameters(
            target_kind=self.target_kind,
            file_path=self.file_path,
            shape=shape,
            order=order,
            dtype=chosen_dtype,
            encoding=encoding,
            voxel_size=voxel_size,
            mapping=mapping_dict,
            fill_value=fill_value,
            total_activity=total_activity,
            noise_threshold=noise_threshold,
            is_npy=self.is_npy_format,
        )

    def get_parameters(self) -> DistributionImportParameters:
        """Алиас для get_import_parameters."""
        return self.get_import_parameters()
