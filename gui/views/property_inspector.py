import logging
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QSlider, QScrollArea, QPushButton, QCheckBox, QFileDialog
)

from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.direct_collimators import DirectParallelCollimator, CollimatorHoleShape
from core.materials.materials import Material
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.viewport_3d.transform_gizmo import GizmoAxis, GizmoMode
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    SpectProcedureViewModel,
    PetProcedureViewModel,
    CustomSweepProcedureViewModel,
)
from gui.viewmodels.data_handler_viewmodel import (
    BaseDataHandlerViewModel,
    DirectStreamHandlerViewModel,
    SensitiveVolumeHandlerViewModel,
    HistoryAssemblerHandlerViewModel,
    DoseMapHandlerViewModel,
    DataManagerViewModel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.dicom_colormaps import get_available_colormaps
import settings.database_setting as database_setting

_logger = logging.getLogger(__name__)


class PropertyInspector(QWidget):
    """
    Инспектор параметров выбранного узла сцены.
    Обеспечивает двустороннее реактивное связывание между полями ввода
    и свойствами активной ViewModel.
    """

    dose_voxel_size_changed = Signal(float)

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.current_vm: Optional[NodeViewModel] = None
        self.scene_vm: Optional[SceneViewModel] = None
        self._sensitive_conn: Optional[Any] = None
        self._is_updating_ui: bool = False
        self._min_buffer_capacity: int = 1

        self._init_ui()

    def set_scene_viewmodel(self, scene_vm: Optional[SceneViewModel]) -> None:
        """Привязка модели сцены для доступа к узлам и единому реестру детекторов."""
        if self._sensitive_conn is not None:
            try:
                QObject.disconnect(self._sensitive_conn)
            except (RuntimeError, TypeError):
                pass
            self._sensitive_conn = None
        self.scene_vm = scene_vm
        if self.scene_vm is not None:
            self._sensitive_conn = self.scene_vm.sensitive_volumes_changed.connect(self._on_sensitive_volumes_changed)

    def _on_sensitive_volumes_changed(self) -> None:
        """Синхронизация состояния чекбокса детектора при изменении единого реестра сцены."""
        if isinstance(self.current_vm, VolumeViewModel) and self.scene_vm is not None:
            self._is_updating_ui = True
            try:
                self.chk_is_detector.setChecked(self.scene_vm.is_sensitive_volume(self.current_vm))
            finally:
                self._is_updating_ui = False

    def set_min_buffer_capacity(self, min_capacity: int) -> None:
        """Устанавливает нижнюю границу емкости буфера данных (не менее числа частиц)."""
        self._min_buffer_capacity = max(1, int(min_capacity))
        self.spin_dm_buffer.setMinimum(self._min_buffer_capacity)
        if self.spin_dm_buffer.value() < self._min_buffer_capacity:
            self.spin_dm_buffer.setValue(self._min_buffer_capacity)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.content_layout = QVBoxLayout(container)
        self.content_layout.setContentsMargins(4, 4, 4, 4)

        # 1. Секция общих параметров
        self.general_group = QGroupBox("Общие свойства (General)")
        gen_form = QFormLayout(self.general_group)
        self.txt_name = QLineEdit()
        self.txt_name.editingFinished.connect(self._on_name_changed)
        self.lbl_type = QLabel("None")
        gen_form.addRow("Имя:", self.txt_name)
        gen_form.addRow("Тип узла:", self.lbl_type)
        self.content_layout.addWidget(self.general_group)

        # 2. Секция пространственных трансформаций
        self.transform_group = QGroupBox("Положение и ориентация")
        trans_form = QFormLayout(self.transform_group)

        self.spin_x = self._create_coord_spinbox(self._on_transform_changed)
        self.spin_y = self._create_coord_spinbox(self._on_transform_changed)
        self.spin_z = self._create_coord_spinbox(self._on_transform_changed)

        self.spin_rot_x = self._create_rot_spinbox(self._on_transform_changed)
        self.spin_rot_y = self._create_rot_spinbox(self._on_transform_changed)
        self.spin_rot_z = self._create_rot_spinbox(self._on_transform_changed)

        trans_form.addRow("Позиция X (мм):", self.spin_x)
        trans_form.addRow("Позиция Y (мм):", self.spin_y)
        trans_form.addRow("Позиция Z (мм):", self.spin_z)
        trans_form.addRow("Поворот Roll (°):", self.spin_rot_x)
        trans_form.addRow("Поворот Pitch (°):", self.spin_rot_y)
        trans_form.addRow("Поворот Yaw (°):", self.spin_rot_z)
        self.content_layout.addWidget(self.transform_group)

        # 3. Секция геометрического объема (Volume)
        self.volume_group = QGroupBox("Геометрия и материал")
        vol_form = QFormLayout(self.volume_group)
        self.spin_size_x = self._create_size_spinbox(self._on_volume_size_changed)
        self.spin_size_y = self._create_size_spinbox(self._on_volume_size_changed)
        self.spin_size_z = self._create_size_spinbox(self._on_volume_size_changed)

        vol_form.addRow("Размер X (мм):", self.spin_size_x)
        vol_form.addRow("Размер Y (мм):", self.spin_size_y)
        vol_form.addRow("Размер Z (мм):", self.spin_size_z)

        self.combo_material = QComboBox()
        self._populate_materials()
        self.combo_material.currentTextChanged.connect(self._on_material_changed)
        vol_form.addRow("Материал:", self.combo_material)

        self.chk_is_detector = QCheckBox("Чувствительный объем (детектор)")
        self.chk_is_detector.toggled.connect(self._on_is_detector_toggled)
        vol_form.addRow(self.chk_is_detector)
        self.content_layout.addWidget(self.volume_group)

        # 3.1 Секция параметров отдельного узла сетки дозы (DoseGridNode)
        self.dose_grid_group = QGroupBox("Параметры сетки дозы (Dose Scorer)")
        dose_grid_form = QFormLayout(self.dose_grid_group)

        self.btn_fit_dose_grid_to_parent = QPushButton("⇲ Подогнать размер под родителя")
        self.btn_fit_dose_grid_to_parent.setToolTip("Автоматически адаптирует размеры сетки дозы под BoundingBox родительского узла сцены")
        self.btn_fit_dose_grid_to_parent.clicked.connect(self._on_fit_dose_grid_to_parent)
        dose_grid_form.addRow(self.btn_fit_dose_grid_to_parent)

        self.spin_dose_grid_size_x = self._create_size_spinbox(self._on_dose_grid_size_changed)
        self.spin_dose_grid_size_y = self._create_size_spinbox(self._on_dose_grid_size_changed)
        self.spin_dose_grid_size_z = self._create_size_spinbox(self._on_dose_grid_size_changed)

        dose_grid_form.addRow("Размер X (мм):", self.spin_dose_grid_size_x)
        dose_grid_form.addRow("Размер Y (мм):", self.spin_dose_grid_size_y)
        dose_grid_form.addRow("Размер Z (мм):", self.spin_dose_grid_size_z)

        self.spin_dose_grid_voxel = QDoubleSpinBox()
        self.spin_dose_grid_voxel.setRange(0.1, 100.0)
        self.spin_dose_grid_voxel.setSingleStep(0.5)
        self.spin_dose_grid_voxel.setValue(5.0)
        self.spin_dose_grid_voxel.setSuffix(" мм")
        self.spin_dose_grid_voxel.setToolTip("Размер стороны вокселя сетки дозы")
        self.spin_dose_grid_voxel.valueChanged.connect(self._on_dose_grid_voxel_step_changed)
        dose_grid_form.addRow("Шаг вокселя:", self.spin_dose_grid_voxel)

        self.lbl_dose_grid_shape = QLabel("–")
        self.lbl_dose_grid_shape.setToolTip("Разрешение воксельной сетки (Nx × Ny × Nz)")
        dose_grid_form.addRow("Разрешение сетки:", self.lbl_dose_grid_shape)

        self.lbl_dose_grid_memory = QLabel("–")
        self.lbl_dose_grid_memory.setToolTip("Оценка расхода оперативной памяти RAM для сетки")
        dose_grid_form.addRow("Расход памяти RAM:", self.lbl_dose_grid_memory)

        self.chk_dose_grid_active = QCheckBox("Активна для накопления дозы")
        self.chk_dose_grid_active.setChecked(True)
        self.chk_dose_grid_active.toggled.connect(self._on_dose_grid_active_toggled)
        dose_grid_form.addRow(self.chk_dose_grid_active)

        self.content_layout.addWidget(self.dose_grid_group)

        # 3.2 Секция параметров коллиматора (Collimator)
        self.collimator_group = QGroupBox("Параметры коллиматора (Collimator)")
        col_form = QFormLayout(self.collimator_group)

        self.combo_collimator_type = QComboBox()
        self.combo_collimator_type.addItem("Детерминированный (Сквозные каналы)", "direct")
        self.combo_collimator_type.addItem("Параметрический (RayCasting)", "parametric")
        self.combo_collimator_type.currentIndexChanged.connect(self._on_collimator_type_changed)
        col_form.addRow("Тип коллиматора:", self.combo_collimator_type)

        self.lbl_collimator_shape = QLabel("Форма каналов:")
        self.combo_collimator_shape = QComboBox()
        self.combo_collimator_shape.addItem("Гексагональный", CollimatorHoleShape.HEXAGONAL)
        self.combo_collimator_shape.addItem("Квадратный", CollimatorHoleShape.SQUARE)
        self.combo_collimator_shape.addItem("Круглый", CollimatorHoleShape.ROUND)
        self.combo_collimator_shape.currentIndexChanged.connect(self._on_collimator_shape_changed)
        col_form.addRow(self.lbl_collimator_shape, self.combo_collimator_shape)

        self.lbl_hole_material = QLabel("Материал каналов:")
        self.combo_hole_material = QComboBox()
        self._populate_hole_materials()
        self.combo_hole_material.currentIndexChanged.connect(self._on_hole_material_changed)
        col_form.addRow(self.lbl_hole_material, self.combo_hole_material)

        self.lbl_collimator_hole = QLabel("Диаметр отверстий:")
        self.spin_collimator_hole = QDoubleSpinBox()
        self.spin_collimator_hole.setRange(0.01, 50.0)
        self.spin_collimator_hole.setSingleStep(0.1)
        self.spin_collimator_hole.setValue(1.5)
        self.spin_collimator_hole.setSuffix(" мм")
        self.spin_collimator_hole.valueChanged.connect(self._on_collimator_hole_changed)
        col_form.addRow(self.lbl_collimator_hole, self.spin_collimator_hole)

        self.spin_collimator_septa = QDoubleSpinBox()
        self.spin_collimator_septa.setRange(0.001, 20.0)
        self.spin_collimator_septa.setSingleStep(0.05)
        self.spin_collimator_septa.setValue(0.2)
        self.spin_collimator_septa.setSuffix(" мм")
        self.spin_collimator_septa.valueChanged.connect(self._on_collimator_septa_changed)
        col_form.addRow("Толщина септ:", self.spin_collimator_septa)
        self.content_layout.addWidget(self.collimator_group)

        # 4. Секция воксельного фантома (WoodcockVoxelVolume)
        self.voxel_group = QGroupBox("Параметры фантома (DICOM / Voxel)")
        vox_form = QFormLayout(self.voxel_group)

        self.txt_voxel_path = QLineEdit()
        self.txt_voxel_path.setReadOnly(True)
        self.btn_browse_voxel = QPushButton("Обзор...")
        self.btn_browse_voxel.clicked.connect(self._on_browse_phantom_file)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.txt_voxel_path)
        path_layout.addWidget(self.btn_browse_voxel)
        vox_form.addRow("Файл фантома:", path_layout)

        self.lbl_voxel_shape = QLabel("–")
        vox_form.addRow("Разрешение сетки:", self.lbl_voxel_shape)

        self.spin_voxel_size_x = QDoubleSpinBox()
        self.spin_voxel_size_x.setRange(0.01, 100.0)
        self.spin_voxel_size_x.setSingleStep(0.5)
        self.spin_voxel_size_x.setValue(4.0)
        self.spin_voxel_size_x.setSuffix(" мм")
        self.spin_voxel_size_x.valueChanged.connect(self._on_voxel_size_changed)

        self.spin_voxel_size_y = QDoubleSpinBox()
        self.spin_voxel_size_y.setRange(0.01, 100.0)
        self.spin_voxel_size_y.setSingleStep(0.5)
        self.spin_voxel_size_y.setValue(4.0)
        self.spin_voxel_size_y.setSuffix(" мм")
        self.spin_voxel_size_y.valueChanged.connect(self._on_voxel_size_changed)

        self.spin_voxel_size_z = QDoubleSpinBox()
        self.spin_voxel_size_z.setRange(0.01, 100.0)
        self.spin_voxel_size_z.setSingleStep(0.5)
        self.spin_voxel_size_z.setValue(4.0)
        self.spin_voxel_size_z.setSuffix(" мм")
        self.spin_voxel_size_z.valueChanged.connect(self._on_voxel_size_changed)

        v_box = QHBoxLayout()
        v_box.addWidget(self.spin_voxel_size_x)
        v_box.addWidget(self.spin_voxel_size_y)
        v_box.addWidget(self.spin_voxel_size_z)
        vox_form.addRow("Шаг вокселей (X, Y, Z):", v_box)

        self.combo_colormap = QComboBox()
        self.combo_colormap.addItem("Physical Materials")
        self.combo_colormap.addItems(get_available_colormaps())
        self.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
        vox_form.addRow("Палитра:", self.combo_colormap)

        self.slider_lod = QSlider(Qt.Horizontal)
        self.slider_lod.setRange(1, 10)
        self.slider_lod.setValue(5)
        self.slider_lod.valueChanged.connect(self._on_lod_changed)
        vox_form.addRow("Детализация (LOD):", self.slider_lod)

        self.spin_opacity_thresh = QDoubleSpinBox()
        self.spin_opacity_thresh.setRange(0.0, 1.0)
        self.spin_opacity_thresh.setSingleStep(0.01)
        self.spin_opacity_thresh.setValue(0.05)
        self.spin_opacity_thresh.setToolTip("Исключает воксели с коэффициентом ослабления/значением ниже порога из рендеринга, делая фоновый воздух полностью прозрачным")
        self.spin_opacity_thresh.valueChanged.connect(self._on_opacity_threshold_changed)
        vox_form.addRow("Порог отсечения фона (Air Cutoff):", self.spin_opacity_thresh)

        self.spin_max_opacity = QDoubleSpinBox()
        self.spin_max_opacity.setRange(0.0, 1.0)
        self.spin_max_opacity.setSingleStep(0.05)
        self.spin_max_opacity.setValue(0.40)
        self.spin_max_opacity.setToolTip("Максимальная непрозрачность воксельного объема (0 = прозрачно, 1 = плотно)")
        self.spin_max_opacity.valueChanged.connect(self._on_max_opacity_changed)
        vox_form.addRow("Непрозрачность (Opacity):", self.spin_max_opacity)

        self.combo_opacity_preset = QComboBox()
        self.combo_opacity_preset.addItem("Отсечение фона (Air Cutoff)", "air_cutoff")
        self.combo_opacity_preset.addItem("Линейный (Linear)", "linear")
        self.combo_opacity_preset.addItem("Мягкие ткани (Soft Tissue)", "soft_tissue")
        self.combo_opacity_preset.addItem("Рентген / Полупрозрачный (X-Ray)", "xray_translucent")
        self.combo_opacity_preset.addItem("Ступенчатый (Step)", "step")
        self.combo_opacity_preset.currentIndexChanged.connect(self._on_opacity_preset_changed)
        vox_form.addRow("Карта прозрачности:", self.combo_opacity_preset)
        self.content_layout.addWidget(self.voxel_group)

        # 5. Секция источника излучения (Source)
        self.source_group = QGroupBox("Параметры источника (Radiation Source)")
        src_form = QFormLayout(self.source_group)

        self.combo_rad_type = QComboBox()
        self.combo_rad_type.addItems(["Gamma", "Beta", "Positron"])
        self.combo_rad_type.currentTextChanged.connect(self._on_source_rad_type_changed)
        src_form.addRow("Тип излучения:", self.combo_rad_type)

        self.spin_source_energy = QDoubleSpinBox()
        self.spin_source_energy.setRange(1.0, 10000.0)
        self.spin_source_energy.setSingleStep(5.0)
        self.spin_source_energy.setValue(140.5)
        self.spin_source_energy.valueChanged.connect(self._on_source_energy_changed)
        src_form.addRow("Энергия (кэВ):", self.spin_source_energy)

        self.spin_source_activity = QDoubleSpinBox()
        self.spin_source_activity.setRange(0.001, 100000.0)
        self.spin_source_activity.setSingleStep(10.0)
        self.spin_source_activity.setValue(100.0)
        self.spin_source_activity.valueChanged.connect(self._on_source_activity_changed)
        src_form.addRow("Активность (МБк):", self.spin_source_activity)

        self.spin_source_half_life = QDoubleSpinBox()
        self.spin_source_half_life.setRange(0.0, 100000.0)
        self.spin_source_half_life.setSingleStep(1.0)
        self.spin_source_half_life.setValue(6.0)
        self.spin_source_half_life.valueChanged.connect(self._on_source_half_life_changed)
        src_form.addRow("Период полураспада (ч):", self.spin_source_half_life)

        self.spin_source_voxel_size = QDoubleSpinBox()
        self.spin_source_voxel_size.setRange(0.01, 100.0)
        self.spin_source_voxel_size.setSingleStep(0.5)
        self.spin_source_voxel_size.setValue(4.0)
        self.spin_source_voxel_size.setSuffix(" мм")
        self.spin_source_voxel_size.valueChanged.connect(self._on_source_voxel_size_changed)
        src_form.addRow("Шаг вокселя источника:", self.spin_source_voxel_size)

        self.txt_source_path = QLineEdit()
        self.txt_source_path.setReadOnly(True)
        self.btn_browse_source = QPushButton("Обзор...")
        self.btn_browse_source.clicked.connect(self._on_browse_source_file)
        src_path_layout = QHBoxLayout()
        src_path_layout.addWidget(self.txt_source_path)
        src_path_layout.addWidget(self.btn_browse_source)
        src_form.addRow("Файл распределения:", src_path_layout)

        self.lbl_source_shape = QLabel("Точечный источник")
        src_form.addRow("Сетка источника:", self.lbl_source_shape)
        self.content_layout.addWidget(self.source_group)

        # 6. Секция параметров гамма-камеры (GammaCamera)
        self.spect_group = QGroupBox("Параметры гамма-камеры")
        spect_form = QFormLayout(self.spect_group)

        self.spin_detector_size_x = QDoubleSpinBox()
        self.spin_detector_size_x.setRange(10.0, 2000.0)
        self.spin_detector_size_x.setSingleStep(10.0)
        self.spin_detector_size_x.setSuffix(" мм")
        self.spin_detector_size_x.valueChanged.connect(self._on_spect_detector_size_changed)

        self.spin_detector_size_y = QDoubleSpinBox()
        self.spin_detector_size_y.setRange(10.0, 2000.0)
        self.spin_detector_size_y.setSingleStep(10.0)
        self.spin_detector_size_y.setSuffix(" мм")
        self.spin_detector_size_y.valueChanged.connect(self._on_spect_detector_size_changed)

        det_size_layout = QHBoxLayout()
        det_size_layout.addWidget(QLabel("X:"))
        det_size_layout.addWidget(self.spin_detector_size_x)
        det_size_layout.addWidget(QLabel("Y:"))
        det_size_layout.addWidget(self.spin_detector_size_y)

        self.spin_detector_thickness = QDoubleSpinBox()
        self.spin_detector_thickness.setRange(0.1, 200.0)
        self.spin_detector_thickness.setSingleStep(0.5)
        self.spin_detector_thickness.setSuffix(" мм")
        self.spin_detector_thickness.valueChanged.connect(self._on_spect_detector_thickness_changed)

        self.spin_collimator_thickness = QDoubleSpinBox()
        self.spin_collimator_thickness.setRange(1.0, 300.0)
        self.spin_collimator_thickness.setSingleStep(1.0)
        self.spin_collimator_thickness.setSuffix(" мм")
        self.spin_collimator_thickness.valueChanged.connect(self._on_spect_collimator_thickness_changed)

        self.spin_cam_gap = QDoubleSpinBox()
        self.spin_cam_gap.setRange(0.0, 100.0)
        self.spin_cam_gap.setSingleStep(0.5)
        self.spin_cam_gap.setValue(1.0)
        self.spin_cam_gap.setSuffix(" мм")
        self.spin_cam_gap.valueChanged.connect(self._on_spect_cam_gap_changed)

        self.spin_cam_shielding = QDoubleSpinBox()
        self.spin_cam_shielding.setRange(0.0, 200.0)
        self.spin_cam_shielding.setSingleStep(1.0)
        self.spin_cam_shielding.setValue(20.0)
        self.spin_cam_shielding.setSuffix(" мм")
        self.spin_cam_shielding.valueChanged.connect(self._on_spect_cam_shielding_changed)

        self.spin_cam_glass = QDoubleSpinBox()
        self.spin_cam_glass.setRange(0.0, 200.0)
        self.spin_cam_glass.setSingleStep(1.0)
        self.spin_cam_glass.setValue(50.0)
        self.spin_cam_glass.setSuffix(" мм")
        self.spin_cam_glass.valueChanged.connect(self._on_spect_cam_glass_changed)

        self.lbl_housing_size = QLabel("-")

        spect_form.addRow("Активное поле детектора (X×Y):", det_size_layout)
        spect_form.addRow("Толщина кристалла (Z):", self.spin_detector_thickness)
        spect_form.addRow("Толщина коллиматора (Z):", self.spin_collimator_thickness)
        spect_form.addRow("Зазор детектор-коллиматор:", self.spin_cam_gap)
        spect_form.addRow("Толщина свинцовой защиты Pb:", self.spin_cam_shielding)
        spect_form.addRow("Толщина оптического стекла:", self.spin_cam_glass)
        spect_form.addRow("Габариты корпуса (X×Y×Z):", self.lbl_housing_size)
        self.content_layout.addWidget(self.spect_group)

        # 7. Секция станины томографа (GantryNode / GantryViewModel)
        self.gantry_group = QGroupBox("Параметры станины томографа (Gantry)")
        gantry_form = QFormLayout(self.gantry_group)

        self.spin_gantry_angle = QDoubleSpinBox()
        self.spin_gantry_angle.setRange(0.0, 360.0)
        self.spin_gantry_angle.setSingleStep(5.0)
        self.spin_gantry_angle.setSuffix(" °")
        self.spin_gantry_angle.valueChanged.connect(self._on_gantry_angle_changed)

        self.chk_wireframe_visible = QCheckBox("Отображать направляющие (Wireframe)")
        self.chk_wireframe_visible.toggled.connect(self._on_gantry_wireframe_toggled)

        gantry_form.addRow("Угол поворота ротора θ:", self.spin_gantry_angle)
        gantry_form.addRow(self.chk_wireframe_visible)
        self.content_layout.addWidget(self.gantry_group)

        # 9. Секция протоколов / процедур (Procedures)
        self._init_procedure_ui()

        # 10. Секция обработчиков данных (DataHandlers)
        self._init_data_handler_ui()

        # 11. Секция диспетчера данных (DataManager)
        self._init_data_manager_ui()

        self.content_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        self.clear_selection()

    def _populate_materials(self) -> None:
        if database_setting.material_database:
            for name in sorted(database_setting.material_database.keys()):
                self.combo_material.addItem(name)
        else:
            self.combo_material.addItems(["Water", "Air", "Lead", "Vacuum"])

    def _populate_hole_materials(self) -> None:
        self.combo_hole_material.clear()
        self.combo_hole_material.addItem("По умолчанию (от родителя)", None)
        if database_setting.material_database:
            for name in sorted(database_setting.material_database.keys()):
                self.combo_hole_material.addItem(name, name)
        else:
            for name in ["Air", "Vacuum", "Water"]:
                self.combo_hole_material.addItem(name, name)

    def _create_coord_spinbox(self, callback: Any) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-5000.0, 5000.0)
        spin.setSingleStep(1.0)
        spin.valueChanged.connect(callback)
        return spin

    def _create_rot_spinbox(self, callback: Any) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-180.0, 180.0)
        spin.setSingleStep(1.0)
        spin.valueChanged.connect(callback)
        return spin

    def _create_size_spinbox(self, callback: Any) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 5000.0)
        spin.setSingleStep(5.0)
        spin.valueChanged.connect(callback)
        return spin

    def _init_procedure_ui(self) -> None:
        self.procedure_group = QGroupBox("Параметры протокола исследования")
        proc_form = QFormLayout(self.procedure_group)

        self.lbl_proc_type = QLabel("ОФЭКТ (SPECT)")
        proc_form.addRow("Тип протокола:", self.lbl_proc_type)

        self.spin_proc_steps = QSpinBox()
        self.spin_proc_steps.setRange(1, 1024)
        self.spin_proc_steps.setValue(32)
        self.spin_proc_steps.valueChanged.connect(self._on_proc_steps_changed)
        proc_form.addRow("Число шагов (Steps):", self.spin_proc_steps)

        self.lbl_proc_total_views = QLabel("64")
        self.lbl_proc_total_views.setStyleSheet("color: #a0c0ff; font-weight: bold;")
        proc_form.addRow("Всего проекций (Total Views):", self.lbl_proc_total_views)

        self.spin_proc_cameras = QSpinBox()
        self.spin_proc_cameras.setRange(1, 16)
        self.spin_proc_cameras.setValue(2)
        self.spin_proc_cameras.valueChanged.connect(self._on_proc_cameras_changed)
        proc_form.addRow("Гамма-камер:", self.spin_proc_cameras)

        self.spin_proc_radius = QDoubleSpinBox()
        self.spin_proc_radius.setRange(10.0, 2000.0)
        self.spin_proc_radius.setValue(250.0)
        self.spin_proc_radius.setSingleStep(5.0)
        self.spin_proc_radius.valueChanged.connect(self._on_proc_radius_changed)
        proc_form.addRow("Радиус орбиты (мм):", self.spin_proc_radius)

        self.spin_proc_time = QDoubleSpinBox()
        self.spin_proc_time.setRange(0.001, 3600.0)
        self.spin_proc_time.setValue(1.0)
        self.spin_proc_time.setSingleStep(0.5)
        self.spin_proc_time.valueChanged.connect(self._on_proc_time_changed)
        proc_form.addRow("Время на ракурс (с):", self.spin_proc_time)

        self.spin_proc_start_angle = QDoubleSpinBox()
        self.spin_proc_start_angle.setRange(0.0, 360.0)
        self.spin_proc_start_angle.setValue(0.0)
        self.spin_proc_start_angle.valueChanged.connect(self._on_proc_start_angle_changed)
        proc_form.addRow("Начальный угол (°):", self.spin_proc_start_angle)

        self.spin_proc_end_angle = QDoubleSpinBox()
        self.spin_proc_end_angle.setRange(0.0, 720.0)
        self.spin_proc_end_angle.setValue(360.0)
        self.spin_proc_end_angle.valueChanged.connect(self._on_proc_end_angle_changed)
        proc_form.addRow("Конечный угол (°):", self.spin_proc_end_angle)

        self.combo_proc_head_mode = QComboBox()
        self.combo_proc_head_mode.addItems(["Симметричный (360°/N)", "L-режим (90°)", "Пользовательский"])
        self.combo_proc_head_mode.currentTextChanged.connect(self._on_proc_head_mode_changed)
        proc_form.addRow("Конфигурация головок:", self.combo_proc_head_mode)

        self.txt_proc_head_angles = QLineEdit()
        self.txt_proc_head_angles.editingFinished.connect(self._on_proc_head_angles_edited)
        proc_form.addRow("Углы головок (°):", self.txt_proc_head_angles)

        self.chk_proc_endpoint = QCheckBox("Включать конечный угол (endpoint)")
        self.chk_proc_endpoint.toggled.connect(self._on_proc_endpoint_toggled)
        proc_form.addRow(self.chk_proc_endpoint)

        self.content_layout.addWidget(self.procedure_group)

    def _init_data_handler_ui(self) -> None:
        self.data_handler_group = QGroupBox("Параметры обработчика данных")
        dh_form = QFormLayout(self.data_handler_group)

        self.lbl_dh_type = QLabel()
        dh_form.addRow("Тип:", self.lbl_dh_type)

        self.chk_dh_show_escaped = QCheckBox("Отображать вылетевшие треки")
        self.chk_dh_show_escaped.toggled.connect(self._on_dh_show_escaped_toggled)
        dh_form.addRow(self.chk_dh_show_escaped)

        self.txt_dh_vols = QLineEdit()
        self.txt_dh_vols.editingFinished.connect(self._on_dh_vols_edited)
        dh_form.addRow("Чувствительные объемы:", self.txt_dh_vols)

        self.chk_dh_initial_states = QCheckBox("Сохранять начальные состояния фотонов")
        self.chk_dh_initial_states.toggled.connect(self._on_dh_initial_states_toggled)
        dh_form.addRow(self.chk_dh_initial_states)

        self.txt_dh_grid_names = QLineEdit()
        self.txt_dh_grid_names.editingFinished.connect(self._on_dh_grid_names_edited)
        dh_form.addRow("Сетки дозы (DoseGrids):", self.txt_dh_grid_names)

        self.txt_dh_shm_name = QLineEdit()
        self.txt_dh_shm_name.editingFinished.connect(self._on_dh_shm_name_edited)
        dh_form.addRow("Имя SharedMemory:", self.txt_dh_shm_name)

        self.content_layout.addWidget(self.data_handler_group)

    def _init_data_manager_ui(self) -> None:
        self.data_manager_group = QGroupBox("Параметры диспетчера данных (DataManager)")
        dm_form = QFormLayout(self.data_manager_group)

        fn_layout = QHBoxLayout()
        self.txt_dm_filename = QLineEdit()
        self.txt_dm_filename.editingFinished.connect(self._on_dm_filename_edited)
        btn_browse = QPushButton("Обзор...")
        btn_browse.clicked.connect(self._on_dm_browse_clicked)
        fn_layout.addWidget(self.txt_dm_filename)
        fn_layout.addWidget(btn_browse)
        dm_form.addRow("Файл HDF5:", fn_layout)

        self.spin_dm_buffer = QSpinBox()
        self.spin_dm_buffer.setRange(1, 2_147_483_647)
        self.spin_dm_buffer.setSingleStep(1)
        self.spin_dm_buffer.setValue(1)
        self.spin_dm_buffer.valueChanged.connect(self._on_dm_buffer_changed)
        dm_form.addRow("Емкость буфера (частиц):", self.spin_dm_buffer)

        self.content_layout.addWidget(self.data_manager_group)

    def set_target_viewmodel(self, vm: Optional[Any]) -> None:
        """
        Устанавливает целевую модель представления для инспекции (Node, Procedure или DataHandler).
        """
        if self.current_vm is vm:
            return

        if self.current_vm is not None:
            try:
                if isinstance(self.current_vm, NodeViewModel):
                    self.current_vm.property_changed.disconnect(self._on_property_changed_externally)
                    self.current_vm.transform_changed.disconnect(self._update_transform_fields)
                elif isinstance(self.current_vm, (BaseProcedureViewModel, BaseDataHandlerViewModel, DataManagerViewModel)):
                    self.current_vm.changed.disconnect(self.update_all_fields)
            except Exception:
                pass

        self.current_vm = vm
        if self.current_vm is None:
            self.clear_selection()
            return

        if isinstance(self.current_vm, NodeViewModel):
            self.current_vm.property_changed.connect(self._on_property_changed_externally)
            self.current_vm.transform_changed.connect(self._update_transform_fields)
        elif isinstance(self.current_vm, (BaseProcedureViewModel, BaseDataHandlerViewModel, DataManagerViewModel)):
            self.current_vm.changed.connect(self.update_all_fields)

        self.update_all_fields()

    def clear_selection(self) -> None:
        """
        Сброс полей при отсутствии выбранного объекта.
        """
        self._is_updating_ui = True
        self.txt_name.setText("")
        self.lbl_type.setText("Не выбран")
        self.general_group.setEnabled(False)
        self.transform_group.setEnabled(False)
        self.volume_group.setVisible(False)
        self.dose_grid_group.setVisible(False)
        self.collimator_group.setVisible(False)
        self.voxel_group.setVisible(False)
        self.source_group.setVisible(False)
        self.spect_group.setVisible(False)
        self.gantry_group.setVisible(False)
        self.procedure_group.setVisible(False)
        self.data_handler_group.setVisible(False)
        self.data_manager_group.setVisible(False)
        self._is_updating_ui = False

    def update_all_fields(self) -> None:
        """
        Синхронизация всех полей UI со значениями ViewModel.
        """
        if self.current_vm is None:
            return

        self._is_updating_ui = True

        if isinstance(self.current_vm, BaseProcedureViewModel):
            self.general_group.setVisible(False)
            self.transform_group.setVisible(False)
            self.volume_group.setVisible(False)
            self.dose_grid_group.setVisible(False)
            self.collimator_group.setVisible(False)
            self.voxel_group.setVisible(False)
            self.source_group.setVisible(False)
            self.spect_group.setVisible(False)
            self.gantry_group.setVisible(False)
            self.data_handler_group.setVisible(False)
            self.data_manager_group.setVisible(False)
            self.procedure_group.setVisible(True)
            self._update_procedure_fields()
            self._is_updating_ui = False
            return

        if isinstance(self.current_vm, BaseDataHandlerViewModel):
            self.general_group.setVisible(False)
            self.transform_group.setVisible(False)
            self.volume_group.setVisible(False)
            self.dose_grid_group.setVisible(False)
            self.collimator_group.setVisible(False)
            self.voxel_group.setVisible(False)
            self.source_group.setVisible(False)
            self.spect_group.setVisible(False)
            self.gantry_group.setVisible(False)
            self.procedure_group.setVisible(False)
            self.data_manager_group.setVisible(False)
            self.data_handler_group.setVisible(True)
            self._update_data_handler_fields()
            self._is_updating_ui = False
            return

        if isinstance(self.current_vm, DataManagerViewModel):
            self.general_group.setVisible(False)
            self.transform_group.setVisible(False)
            self.volume_group.setVisible(False)
            self.dose_grid_group.setVisible(False)
            self.collimator_group.setVisible(False)
            self.voxel_group.setVisible(False)
            self.source_group.setVisible(False)
            self.spect_group.setVisible(False)
            self.gantry_group.setVisible(False)
            self.procedure_group.setVisible(False)
            self.data_handler_group.setVisible(False)
            self.data_manager_group.setVisible(True)
            self._update_data_manager_fields()
            self._is_updating_ui = False
            return

        # Для NodeViewModel
        self.procedure_group.setVisible(False)
        self.data_handler_group.setVisible(False)
        self.data_manager_group.setVisible(False)
        self.general_group.setVisible(True)
        self.general_group.setEnabled(True)
        self.transform_group.setVisible(True)
        self.transform_group.setEnabled(True)
        self.txt_name.setText(str(self.current_vm.name))
        self.lbl_type.setText(self.current_vm.node_type)

        self._update_transform_fields()

        # Видимость специализированных секций
        is_spect = isinstance(self.current_vm, GammaCameraViewModel)
        is_gantry = isinstance(self.current_vm, GantryViewModel)
        is_vox = isinstance(self.current_vm, VoxelVolumeViewModel)
        is_src = isinstance(self.current_vm, SourceViewModel)
        is_dose_grid = isinstance(self.current_vm, DoseGridViewModel)
        is_col = isinstance(self.current_vm, CollimatorViewModel)
        is_vol = (isinstance(self.current_vm, VolumeViewModel) or is_col) and not is_spect and not is_vox
        is_root_vol = is_vol and (self.current_vm.parent_vm is None or 'world' in str(self.current_vm.name).lower())

        self.volume_group.setVisible(is_vol)
        self.chk_is_detector.setVisible(not is_col)
        self.dose_grid_group.setVisible(is_dose_grid)
        self.collimator_group.setVisible(is_col)
        self.voxel_group.setVisible(is_vox)
        self.source_group.setVisible(is_src)
        self.spect_group.setVisible(is_spect)
        self.gantry_group.setVisible(is_gantry)

        if is_col:
            col_kind = self.current_vm.collimator_kind
            type_index = self.combo_collimator_type.findData(col_kind)
            if type_index >= 0:
                self.combo_collimator_type.setCurrentIndex(type_index)

            is_direct = (col_kind == "direct")
            self.lbl_collimator_shape.setVisible(True)
            self.combo_collimator_shape.setVisible(True)
            self.lbl_hole_material.setVisible(is_direct)
            self.combo_hole_material.setVisible(is_direct)

            current_shape = self.current_vm.hole_shape
            if current_shape == CollimatorHoleShape.SQUARE:
                self.lbl_collimator_hole.setText("Ширина отверстий:")
            else:
                self.lbl_collimator_hole.setText("Диаметр отверстий:")
            self.spin_collimator_hole.setValue(float(self.current_vm.hole_diameter))
            self.spin_collimator_septa.setValue(float(self.current_vm.septa))

            current_shape_val = current_shape.value if isinstance(current_shape, CollimatorHoleShape) else current_shape
            for shape_index in range(self.combo_collimator_shape.count()):
                item_shape = self.combo_collimator_shape.itemData(shape_index)
                if item_shape == current_shape or item_shape == current_shape_val:
                    self.combo_collimator_shape.setCurrentIndex(shape_index)
                    break

            if is_direct:
                hole_mat_name = self.current_vm.hole_material_name
                if hole_mat_name is None:
                    self.combo_hole_material.setCurrentIndex(0)
                else:
                    material_index = self.combo_hole_material.findData(hole_mat_name)
                    if material_index >= 0:
                        self.combo_hole_material.setCurrentIndex(material_index)
                    else:
                        self.combo_hole_material.setCurrentIndex(0)

        if is_dose_grid:
            grid_size = self.current_vm.size
            self.spin_dose_grid_size_x.setValue(float(grid_size[0]))
            self.spin_dose_grid_size_y.setValue(float(grid_size[1]))
            self.spin_dose_grid_size_z.setValue(float(grid_size[2]))
            self.spin_dose_grid_voxel.setValue(float(self.current_vm.dose_voxel_size))
            self.chk_dose_grid_active.setChecked(bool(self.current_vm.is_active))
            self._update_dose_grid_metrics()

            parent_vm = self.current_vm.parent_vm
            if isinstance(parent_vm, (VolumeViewModel, GammaCameraViewModel)):
                self.btn_fit_dose_grid_to_parent.setEnabled(True)
                self.btn_fit_dose_grid_to_parent.setText(f"⇲ Подогнать под {parent_vm.name}")
            else:
                self.btn_fit_dose_grid_to_parent.setEnabled(False)
                self.btn_fit_dose_grid_to_parent.setText("⇲ Подогнать размер под родителя")

        if is_spect:
            detector_dimensions = self.current_vm.detector_size
            self.spin_detector_size_x.setValue(float(detector_dimensions[0]))
            self.spin_detector_size_y.setValue(float(detector_dimensions[1]))
            self.spin_detector_thickness.setValue(float(self.current_vm.detector_thickness))
            self.spin_collimator_thickness.setValue(float(self.current_vm.collimator_thickness))
            self.spin_cam_gap.setValue(float(self.current_vm.gap))
            self.spin_cam_shielding.setValue(float(self.current_vm.shielding_thickness))
            self.spin_cam_glass.setValue(float(self.current_vm.glass_backend_thickness))
            self._update_housing_size_label()

        elif is_gantry:
            self.spin_gantry_angle.setValue(float(self.current_vm.gantry_angle_deg))
            self.chk_wireframe_visible.setChecked(bool(self.current_vm.wireframe_visible))

        elif is_src:
            source_type_index = self.combo_rad_type.findText(self.current_vm.radiation_type)
            if source_type_index >= 0:
                self.combo_rad_type.setCurrentIndex(source_type_index)
            self.spin_source_energy.setValue(float(self.current_vm.energy))
            self.spin_source_activity.setValue(float(self.current_vm.activity))
            self.spin_source_half_life.setValue(float(self.current_vm.half_life))
            self.spin_source_voxel_size.setValue(float(self.current_vm.voxel_size))
            self.txt_source_path.setText(str(self.current_vm.file_path))
            dims = self.current_vm.dimensions
            if self.current_vm.is_point_source:
                self.lbl_source_shape.setText("Точечный источник")
            else:
                self.lbl_source_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")

        elif is_vox:
            self.txt_voxel_path.setText(str(self.current_vm.file_path))
            dims = self.current_vm.dimensions
            self.lbl_voxel_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
            vox_sz = self.current_vm.voxel_size
            if len(vox_sz) >= 3:
                self.spin_voxel_size_x.setValue(float(vox_sz[0]))
                self.spin_voxel_size_y.setValue(float(vox_sz[1]))
                self.spin_voxel_size_z.setValue(float(vox_sz[2]))
            colormap_index = self.combo_colormap.findText(self.current_vm.colormap_name)
            if colormap_index >= 0:
                self.combo_colormap.setCurrentIndex(colormap_index)
            self._update_voxel_opacity_controls_state(self.current_vm.colormap_name)
            self.slider_lod.setValue(int(round(float(self.current_vm.lod_factor) * 5.0)))
            self.spin_opacity_thresh.setValue(float(self.current_vm.opacity_threshold))
            self.spin_max_opacity.setValue(float(self.current_vm.max_opacity))
            preset = self.current_vm.opacity_preset
            preset_index = self.combo_opacity_preset.findData(preset)
            if preset_index >= 0:
                self.combo_opacity_preset.setCurrentIndex(preset_index)

        elif is_vol:
            volume_size = self.current_vm.size
            self.spin_size_x.setValue(float(volume_size[0]))
            self.spin_size_y.setValue(float(volume_size[1]))
            self.spin_size_z.setValue(float(volume_size[2]))
            material_index = self.combo_material.findText(self.current_vm.material_name)
            if material_index >= 0:
                self.combo_material.setCurrentIndex(material_index)
            is_detector = self.scene_vm.is_sensitive_volume(self.current_vm) if self.scene_vm is not None else False
            self.chk_is_detector.setChecked(is_detector)

        self._is_updating_ui = False

    def _update_transform_fields(self) -> None:
        if self.current_vm is None:
            return

        constraint = self.current_vm.get_effective_kinematic_constraint()
        if constraint is not None:
            allowed_trans = constraint.get_allowed_axes(GizmoMode.TRANSLATE)
            allowed_rot = constraint.get_allowed_axes(GizmoMode.ROTATE)
            scale_allowed = constraint.is_scale_allowed()

            self.spin_x.setEnabled(GizmoAxis.X in allowed_trans)
            self.spin_y.setEnabled(GizmoAxis.Y in allowed_trans)
            self.spin_z.setEnabled(GizmoAxis.Z in allowed_trans)

            self.spin_rot_x.setEnabled(GizmoAxis.X in allowed_rot)
            self.spin_rot_y.setEnabled(GizmoAxis.Y in allowed_rot)
            self.spin_rot_z.setEnabled(GizmoAxis.Z in allowed_rot)

            self.spin_size_x.setEnabled(scale_allowed)
            self.spin_size_y.setEnabled(scale_allowed)
            self.spin_size_z.setEnabled(scale_allowed)
        else:
            self.spin_x.setEnabled(True)
            self.spin_y.setEnabled(True)
            self.spin_z.setEnabled(True)

            self.spin_rot_x.setEnabled(True)
            self.spin_rot_y.setEnabled(True)
            self.spin_rot_z.setEnabled(True)

            self.spin_size_x.setEnabled(True)
            self.spin_size_y.setEnabled(True)
            self.spin_size_z.setEnabled(True)

        old_state = self._is_updating_ui
        self._is_updating_ui = True
        try:
            mat = self.current_vm.local_matrix
            self.spin_x.setValue(float(mat[0, 3]))
            self.spin_y.setValue(float(mat[1, 3]))
            self.spin_z.setValue(float(mat[2, 3]))

            rot_mat = mat[:3, :3]
            try:
                if np.all(np.isfinite(rot_mat)):
                    det = np.linalg.det(rot_mat)
                    if det > 1e-6:
                        rotation_obj = Rotation.from_matrix(rot_mat)
                        euler = rotation_obj.as_euler('xyz', degrees=True)
                        if np.all(np.isfinite(euler)):
                            self.spin_rot_x.setValue(float(euler[0]))
                            self.spin_rot_y.setValue(float(euler[1]))
                            self.spin_rot_z.setValue(float(euler[2]))
            except (ValueError, np.linalg.LinAlgError):
                pass
        finally:
            self._is_updating_ui = old_state

    def _on_name_changed(self) -> None:
        if self._is_updating_ui or self.current_vm is None:
            return
        old_name = self.current_vm.name
        new_name = self.txt_name.text().strip()
        if not new_name or old_name == new_name:
            return
        self.current_vm.name = new_name
        if self.scene_vm is not None and old_name in self.scene_vm.sensitive_volumes:
            vols = self.scene_vm.sensitive_volumes
            idx = vols.index(old_name)
            vols[idx] = new_name
            self.scene_vm.sensitive_volumes = vols

    def _on_transform_changed(self) -> None:
        if self._is_updating_ui or self.current_vm is None:
            return
        # Прямое задание смещения и вращения
        mat = np.eye(4, dtype=float)
        try:
            rot = Rotation.from_euler('xyz', [
                self.spin_rot_x.value(),
                self.spin_rot_y.value(),
                self.spin_rot_z.value()
            ], degrees=True)
            mat[:3, :3] = rot.as_matrix()
        except ValueError:
            mat[:3, :3] = self.current_vm.local_matrix[:3, :3]

        mat[0, 3] = self.spin_x.value()
        mat[1, 3] = self.spin_y.value()
        mat[2, 3] = self.spin_z.value()
        self.current_vm.local_matrix = mat

    def _on_volume_size_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, (VolumeViewModel, CollimatorViewModel)):
            return
        self.current_vm.size = [
            self.spin_size_x.value(),
            self.spin_size_y.value(),
            self.spin_size_z.value()
        ]

    def _on_material_changed(self, mat_name: str) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, (VolumeViewModel, CollimatorViewModel)):
            return
        self.current_vm.material_name = mat_name

    def _update_voxel_opacity_controls_state(self, colormap_name: str) -> None:
        """
        Блокирует эвристические регуляторы прозрачности при активном физическом режиме 'Physical Materials'.
        """
        is_physical_mode = (str(colormap_name) == "Physical Materials")
        self.spin_opacity_thresh.setEnabled(not is_physical_mode)
        self.spin_max_opacity.setEnabled(not is_physical_mode)
        self.combo_opacity_preset.setEnabled(not is_physical_mode)

    def _on_colormap_changed(self, cmap_name: str) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        self.current_vm.colormap_name = cmap_name
        self._update_voxel_opacity_controls_state(cmap_name)

    def _on_lod_changed(self, value: int) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        self.current_vm.lod_factor = float(value) / 5.0

    def _on_opacity_threshold_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        self.current_vm.opacity_threshold = float(value)

    def _on_max_opacity_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        self.current_vm.max_opacity = float(value)

    def _on_opacity_preset_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        preset = self.combo_opacity_preset.currentData() or 'air_cutoff'
        self.current_vm.opacity_preset = str(preset)

    def _on_browse_phantom_file(self) -> None:
        if not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл воксельного фантома",
            "",
            "Файлы данных (*.npy *.dat *.raw *.h5);;Все файлы (*.*)"
        )
        if path:
            self.txt_voxel_path.setText(path)
            self.current_vm.reload_distribution(path)

    def _on_browse_source_file(self) -> None:
        if not isinstance(self.current_vm, SourceViewModel):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл распределения источника",
            "",
            "Файлы данных (*.npy *.dat *.raw *.h5);;Все файлы (*.*)"
        )
        if path:
            self.txt_source_path.setText(path)
            self.current_vm.reload_distribution(path)

    def _on_source_rad_type_changed(self, text: str) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, SourceViewModel):
            return
        self.current_vm.radiation_type = text

    def _on_source_energy_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, SourceViewModel):
            return
        self.current_vm.energy = float(value)

    def _on_source_activity_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, SourceViewModel):
            return
        self.current_vm.activity = float(value)

    def _on_source_half_life_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, SourceViewModel):
            return
        self.current_vm.half_life = float(value)

    def _on_is_detector_toggled(self, checked: bool) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VolumeViewModel):
            return
        if self.scene_vm is not None:
            self.scene_vm.set_volume_sensitive(self.current_vm, checked)

    def _update_dose_grid_metrics(self) -> None:
        """
        Расчет разрешения сетки вокселей и объема требуемой оперативной памяти
        для узла DoseGridViewModel.
        """
        if not isinstance(self.current_vm, DoseGridViewModel):
            return
        shape = self.current_vm.grid_shape
        n_voxels = int(shape[0] * shape[1] * shape[2])
        mem_mb = self.current_vm.memory_mb
        self.lbl_dose_grid_shape.setText(f"{shape[0]} × {shape[1]} × {shape[2]} ({n_voxels:,} вокс.)")
        self.lbl_dose_grid_memory.setText(f"{mem_mb:.2f} МБ (float64)")

    def _on_dose_grid_size_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, DoseGridViewModel):
            return
        new_size = (
            self.spin_dose_grid_size_x.value(),
            self.spin_dose_grid_size_y.value(),
            self.spin_dose_grid_size_z.value()
        )
        self.current_vm.size = new_size
        self._update_dose_grid_metrics()

    def _on_dose_grid_voxel_step_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, DoseGridViewModel):
            return
        self.current_vm.dose_voxel_size = self.spin_dose_grid_voxel.value()
        self._update_dose_grid_metrics()

    def _on_dose_grid_active_toggled(self, checked: bool) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, DoseGridViewModel):
            return
        self.current_vm.is_active = checked

    def _on_fit_dose_grid_to_parent(self) -> None:
        """
        Автоматически подгоняет размеры сетки дозы под BoundingBox родительского узла.
        """
        if self._is_updating_ui or not isinstance(self.current_vm, DoseGridViewModel):
            return
        parent_vm = self.current_vm.parent_vm
        if not isinstance(parent_vm, (VolumeViewModel, GammaCameraViewModel)):
            return

        bounding_box_size = parent_vm.local_bound
        new_size = (float(bounding_box_size[0]), float(bounding_box_size[1]), float(bounding_box_size[2]))
        self.current_vm.size = new_size
        self._is_updating_ui = True
        try:
            self.spin_dose_grid_size_x.setValue(new_size[0])
            self.spin_dose_grid_size_y.setValue(new_size[1])
            self.spin_dose_grid_size_z.setValue(new_size[2])
            self._update_dose_grid_metrics()
        finally:
            self._is_updating_ui = False

    def _on_spect_detector_size_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GammaCameraViewModel):
            return
        size_x = float(self.spin_detector_size_x.value())
        size_y = float(self.spin_detector_size_y.value())
        self.current_vm.detector_size = (size_x, size_y)
        self._update_housing_size_label()

    def _on_spect_detector_thickness_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GammaCameraViewModel):
            return
        self.current_vm.detector_thickness = float(value)
        self._update_housing_size_label()

    def _on_spect_collimator_thickness_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GammaCameraViewModel):
            return
        self.current_vm.collimator_thickness = float(value)
        self._update_housing_size_label()

    def _on_gantry_angle_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GantryViewModel):
            return
        self.current_vm.gantry_angle_deg = float(value)

    def _on_gantry_wireframe_toggled(self, checked: bool) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GantryViewModel):
            return
        self.current_vm.wireframe_visible = bool(checked)

    def _update_housing_size_label(self) -> None:
        if isinstance(self.current_vm, GammaCameraViewModel):
            housing_dims = self.current_vm.housing_size
            self.lbl_housing_size.setText(f"{housing_dims[0]:.1f} × {housing_dims[1]:.1f} × {housing_dims[2]:.1f} мм")

    def _on_collimator_hole_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, CollimatorViewModel):
            return
        self.current_vm.hole_diameter = float(value)

    def _on_collimator_septa_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, CollimatorViewModel):
            return
        self.current_vm.septa = float(value)

    def _on_collimator_shape_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, CollimatorViewModel):
            return
        shape_data = self.combo_collimator_shape.itemData(index)
        if shape_data is None:
            return
        try:
            self.current_vm.hole_shape = shape_data
        except NotImplementedError as err:
            _logger.warning(f"Выбранная форма каналов коллиматора еще не реализована: {err}")
            # Возвращаем предыдущее корректное значение
            self._is_updating_ui = True
            try:
                current_shape = self.current_vm.hole_shape
                current_shape_val = current_shape.value if isinstance(current_shape, CollimatorHoleShape) else current_shape
                for shape_index in range(self.combo_collimator_shape.count()):
                    item_shape = self.combo_collimator_shape.itemData(shape_index)
                    if item_shape == current_shape or item_shape == current_shape_val:
                        self.combo_collimator_shape.setCurrentIndex(shape_index)
                        break
            finally:
                self._is_updating_ui = False

    def _on_hole_material_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, CollimatorViewModel):
            return
        selected_hole_mat_name = self.combo_hole_material.itemData(index)
        self.current_vm.hole_material_name = selected_hole_mat_name

    def _on_collimator_type_changed(self, index: int) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, CollimatorViewModel):
            return
        collimator_vm = self.current_vm
        target_kind = self.combo_collimator_type.itemData(index)
        if target_kind == collimator_vm.collimator_kind:
            return

        col_size = np.copy(collimator_vm.size)
        hole_diameter = float(collimator_vm.hole_diameter)
        septa = float(collimator_vm.septa)
        hole_shape = collimator_vm.hole_shape
        material_name = collimator_vm.material_name
        lead_mat = database_setting.material_database.get(material_name, Material(name=material_name))

        if target_kind == "direct":
            new_core = DirectParallelCollimator(
                size=col_size,
                hole_diameter=hole_diameter,
                septa=septa,
                material=lead_mat,
                hole_material=None,
                hole_shape=hole_shape,
                name=collimator_vm.name,
            )
        elif target_kind == "parametric":
            new_core = ParametricParallelCollimator(
                size=col_size,
                hole_diameter=hole_diameter,
                septa=septa,
                material=lead_mat,
                hole_shape=hole_shape,
                name=collimator_vm.name,
            )
        else:
            return

        new_core.local_matrix = np.copy(collimator_vm.local_matrix)
        new_col_vm = CollimatorViewModel(new_core)

        if self.scene_vm is not None:
            self.scene_vm.replace_node(collimator_vm, new_col_vm)
            self.set_target_viewmodel(new_col_vm)

    def _on_voxel_size_changed(self) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, VoxelVolumeViewModel):
            return
        self.current_vm.voxel_size = (
            float(self.spin_voxel_size_x.value()),
            float(self.spin_voxel_size_y.value()),
            float(self.spin_voxel_size_z.value()),
        )

    def _on_source_voxel_size_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, SourceViewModel):
            return
        self.current_vm.voxel_size = float(value)

    def _on_spect_cam_gap_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GammaCameraViewModel):
            return
        self.current_vm.gap = float(value)
        self._update_housing_size_label()

    def _on_spect_cam_shielding_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GammaCameraViewModel):
            return
        self.current_vm.shielding_thickness = float(value)
        self._update_housing_size_label()

    def _on_spect_cam_glass_changed(self, value: float) -> None:
        if self._is_updating_ui or not isinstance(self.current_vm, GammaCameraViewModel):
            return
        self.current_vm.glass_backend_thickness = float(value)
        self._update_housing_size_label()

    def _on_property_changed_externally(self, prop_name: str, new_val: Any) -> None:
        if self._is_updating_ui or self.current_vm is None:
            return

        # Инкрементальное обновление отдельных полей UI без полного пересчета всей формы
        if prop_name == 'name':
            self._is_updating_ui = True
            try:
                self.txt_name.setText(str(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'size' and isinstance(self.current_vm, (VolumeViewModel, CollimatorViewModel)):
            volume_size = new_val if isinstance(new_val, (list, tuple, np.ndarray)) else self.current_vm.size
            self._is_updating_ui = True
            try:
                self.spin_size_x.setValue(float(volume_size[0]))
                self.spin_size_y.setValue(float(volume_size[1]))
                self.spin_size_z.setValue(float(volume_size[2]))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'material_name' and isinstance(self.current_vm, (VolumeViewModel, CollimatorViewModel)):
            material_index = self.combo_material.findText(str(new_val))
            if material_index >= 0:
                self._is_updating_ui = True
                try:
                    self.combo_material.setCurrentIndex(material_index)
                finally:
                    self._is_updating_ui = False
        elif prop_name == 'hole_material_name' and isinstance(self.current_vm, CollimatorViewModel):
            self._is_updating_ui = True
            try:
                hole_material_index = 0
                if new_val is not None:
                    found_mat_index = self.combo_hole_material.findData(str(new_val))
                    if found_mat_index >= 0:
                        hole_material_index = found_mat_index
                self.combo_hole_material.setCurrentIndex(hole_material_index)
            finally:
                self._is_updating_ui = False
        elif prop_name in ('hole_diameter', 'hole_width') and isinstance(self.current_vm, CollimatorViewModel):
            self._is_updating_ui = True
            try:
                self.spin_collimator_hole.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'septa' and isinstance(self.current_vm, CollimatorViewModel):
            self._is_updating_ui = True
            try:
                self.spin_collimator_septa.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'hole_shape' and isinstance(self.current_vm, CollimatorViewModel):
            self._is_updating_ui = True
            try:
                new_val_shape = new_val.value if isinstance(new_val, CollimatorHoleShape) else new_val
                for shape_index in range(self.combo_collimator_shape.count()):
                    item_shape = self.combo_collimator_shape.itemData(shape_index)
                    if item_shape == new_val or item_shape == new_val_shape:
                        self.combo_collimator_shape.setCurrentIndex(shape_index)
                        break
            finally:
                self._is_updating_ui = False
        elif prop_name == 'colormap_name' and isinstance(self.current_vm, VoxelVolumeViewModel):
            colormap_index = self.combo_colormap.findText(str(new_val))
            if colormap_index >= 0:
                self._is_updating_ui = True
                try:
                    self.combo_colormap.setCurrentIndex(colormap_index)
                    self._update_voxel_opacity_controls_state(str(new_val))
                finally:
                    self._is_updating_ui = False
        elif prop_name == 'opacity_threshold' and isinstance(self.current_vm, VoxelVolumeViewModel):
            self._is_updating_ui = True
            try:
                self.spin_opacity_thresh.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'max_opacity' and isinstance(self.current_vm, VoxelVolumeViewModel):
            self._is_updating_ui = True
            try:
                self.spin_max_opacity.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'lod_factor' and isinstance(self.current_vm, VoxelVolumeViewModel):
            self._is_updating_ui = True
            try:
                self.slider_lod.setValue(int(round(float(new_val) * 5.0)))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'file_path':
            self._is_updating_ui = True
            try:
                if isinstance(self.current_vm, VoxelVolumeViewModel):
                    self.txt_voxel_path.setText(str(new_val))
                    dims = self.current_vm.dimensions
                    self.lbl_voxel_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
                elif isinstance(self.current_vm, SourceViewModel):
                    self.txt_source_path.setText(str(new_val))
                    dims = self.current_vm.dimensions
                    if self.current_vm.is_point_source:
                        self.lbl_source_shape.setText("Точечный источник")
                    else:
                        self.lbl_source_shape.setText(f"{dims[0]} × {dims[1]} × {dims[2]}")
            finally:
                self._is_updating_ui = False
        elif prop_name == 'activity' and isinstance(self.current_vm, SourceViewModel):
            self._is_updating_ui = True
            try:
                self.spin_source_activity.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'energy' and isinstance(self.current_vm, SourceViewModel):
            self._is_updating_ui = True
            try:
                self.spin_source_energy.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'half_life' and isinstance(self.current_vm, SourceViewModel):
            self._is_updating_ui = True
            try:
                self.spin_source_half_life.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif isinstance(self.current_vm, GammaCameraViewModel) and prop_name in (
            'detector_size', 'detector_thickness', 'collimator_thickness',
            'gap', 'shielding_thickness', 'glass_backend_thickness', 'housing_size'
        ):
            self._is_updating_ui = True
            try:
                if prop_name == 'detector_size' and isinstance(new_val, (list, tuple, np.ndarray)) and len(new_val) >= 2:
                    self.spin_detector_size_x.setValue(float(new_val[0]))
                    self.spin_detector_size_y.setValue(float(new_val[1]))
                elif prop_name == 'detector_thickness':
                    self.spin_detector_thickness.setValue(float(new_val))
                elif prop_name == 'collimator_thickness':
                    self.spin_collimator_thickness.setValue(float(new_val))
                elif prop_name == 'gap':
                    self.spin_cam_gap.setValue(float(new_val))
                elif prop_name == 'shielding_thickness':
                    self.spin_cam_shielding.setValue(float(new_val))
                elif prop_name == 'glass_backend_thickness':
                    self.spin_cam_glass.setValue(float(new_val))
                self._update_housing_size_label()
            finally:
                self._is_updating_ui = False
        elif isinstance(self.current_vm, GantryViewModel) and prop_name in ('gantry_angle_deg', 'wireframe_visible'):
            self._is_updating_ui = True
            try:
                if prop_name == 'gantry_angle_deg':
                    self.spin_gantry_angle.setValue(float(new_val))
                elif prop_name == 'wireframe_visible':
                    self.chk_wireframe_visible.setChecked(bool(new_val))
            finally:
                self._is_updating_ui = False
        elif prop_name == 'voxel_size':
            self._is_updating_ui = True
            try:
                if isinstance(self.current_vm, VoxelVolumeViewModel):
                    volume_size = new_val if isinstance(new_val, (list, tuple, np.ndarray)) else self.current_vm.voxel_size
                    if len(volume_size) >= 3:
                        self.spin_voxel_size_x.setValue(float(volume_size[0]))
                        self.spin_voxel_size_y.setValue(float(volume_size[1]))
                        self.spin_voxel_size_z.setValue(float(volume_size[2]))
                elif isinstance(self.current_vm, SourceViewModel):
                    self.spin_source_voxel_size.setValue(float(new_val))
            finally:
                self._is_updating_ui = False
        elif isinstance(self.current_vm, DoseGridViewModel):
            if prop_name == 'size':
                grid_size = new_val if isinstance(new_val, (list, tuple, np.ndarray)) else self.current_vm.size
                self._is_updating_ui = True
                try:
                    self.spin_dose_grid_size_x.setValue(float(grid_size[0]))
                    self.spin_dose_grid_size_y.setValue(float(grid_size[1]))
                    self.spin_dose_grid_size_z.setValue(float(grid_size[2]))
                    self._update_dose_grid_metrics()
                finally:
                    self._is_updating_ui = False
            elif prop_name == 'dose_voxel_size':
                self._is_updating_ui = True
                try:
                    self.spin_dose_grid_voxel.setValue(float(new_val))
                    self._update_dose_grid_metrics()
                finally:
                    self._is_updating_ui = False
            elif prop_name in ('grid_shape', 'memory_mb'):
                self._update_dose_grid_metrics()
            elif prop_name == 'is_active':
                self._is_updating_ui = True
                try:
                    self.chk_dose_grid_active.setChecked(bool(new_val))
                finally:
                    self._is_updating_ui = False
        else:
            self.update_all_fields()

    def _update_procedure_fields(self) -> None:
        vm = self.current_vm
        if isinstance(vm, SpectProcedureViewModel):
            self.lbl_proc_type.setText("ОФЭКТ (SPECT)")
            self.spin_proc_steps.setValue(int(vm.steps))
            self.spin_proc_cameras.setValue(int(vm.gamma_cameras))
            self.lbl_proc_total_views.setText(str(vm.total_projections))
            self.spin_proc_radius.setValue(float(vm.radius))
            self.spin_proc_time.setValue(float(vm.time_per_view))
            self.spin_proc_start_angle.setValue(float(vm.start_angle))
            self.spin_proc_end_angle.setValue(float(vm.end_angle))
            idx = self.combo_proc_head_mode.findText(vm.head_mode)
            if idx >= 0:
                self.combo_proc_head_mode.setCurrentIndex(idx)
            self.txt_proc_head_angles.setText(", ".join(f"{a:.1f}" for a in vm.head_angles))
            self.chk_proc_endpoint.setChecked(bool(vm.endpoint))
        elif isinstance(vm, PetProcedureViewModel):
            self.lbl_proc_type.setText("ПЭТ (PET)")
            self.spin_proc_steps.setValue(1)
            self.spin_proc_cameras.setValue(int(vm.detector_heads))
            self.lbl_proc_total_views.setText(str(vm.detector_heads))
            self.spin_proc_radius.setValue(float(vm.ring_radius))
            self.spin_proc_time.setValue(float(vm.time_per_frame))
        elif isinstance(vm, BaseProcedureViewModel):
            self.lbl_proc_type.setText(vm.name)

    def _update_data_handler_fields(self) -> None:
        vm = self.current_vm
        if isinstance(vm, BaseDataHandlerViewModel):
            self.lbl_dh_type.setText(vm.name)

            is_stream = isinstance(vm, DirectStreamHandlerViewModel)
            self.chk_dh_show_escaped.setVisible(is_stream)
            if is_stream:
                self.chk_dh_show_escaped.setChecked(bool(vm.show_escaped_tracks))

            is_sens = isinstance(vm, (SensitiveVolumeHandlerViewModel, HistoryAssemblerHandlerViewModel))
            self.txt_dh_vols.setVisible(is_sens)
            if is_sens:
                if not vm.sensitive_volumes and self.scene_vm is not None and self.scene_vm.sensitive_volumes:
                    vm.sensitive_volumes = self.scene_vm.sensitive_volumes
                self.txt_dh_vols.setText(", ".join(vm.sensitive_volumes))

            is_hist = isinstance(vm, HistoryAssemblerHandlerViewModel)
            self.chk_dh_initial_states.setVisible(is_hist)
            if is_hist:
                self.chk_dh_initial_states.setChecked(bool(vm.save_initial_states))

            is_dose = isinstance(vm, DoseMapHandlerViewModel)
            self.txt_dh_grid_names.setVisible(is_dose)
            self.txt_dh_shm_name.setVisible(is_dose)
            if is_dose:
                # Автоматический подхват доступных сеток дозы со сцены, если в обработчике список пуст
                if not vm.grid_names and self.scene_vm is not None:
                    grid_names = [n.name for n in self.scene_vm.all_nodes() if isinstance(n, DoseGridViewModel)]
                    if grid_names:
                        vm.grid_names = grid_names
                self.txt_dh_grid_names.setText(", ".join(vm.grid_names))
                self.txt_dh_shm_name.setText(vm.shm_name)

    def _update_data_manager_fields(self) -> None:
        vm = self.current_vm
        if isinstance(vm, DataManagerViewModel):
            self.txt_dm_filename.setText(vm.filename)
            effective_min = max(self._min_buffer_capacity, vm.min_buffer_capacity)
            self.spin_dm_buffer.setMinimum(effective_min)
            if vm.buffer_capacity < effective_min:
                vm.buffer_capacity = effective_min
            self.spin_dm_buffer.setValue(int(vm.buffer_capacity))

    # Специфичные слоты для процедур
    def _on_proc_steps_changed(self, steps_value: int) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.steps = steps_value
            self._is_updating_ui = True
            self.lbl_proc_total_views.setText(str(self.current_vm.total_projections))
            self._is_updating_ui = False

    def _on_proc_cameras_changed(self, camera_count: int) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.gamma_cameras = camera_count
            self._is_updating_ui = True
            self.lbl_proc_total_views.setText(str(self.current_vm.total_projections))
            self.txt_proc_head_angles.setText(", ".join(f"{angle_val:.1f}" for angle_val in self.current_vm.head_angles))
            self._is_updating_ui = False

    def _on_proc_radius_changed(self, val: float) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.radius = val

    def _on_proc_time_changed(self, val: float) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.time_per_view = val

    def _on_proc_start_angle_changed(self, val: float) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.start_angle = val

    def _on_proc_end_angle_changed(self, val: float) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.end_angle = val

    def _on_proc_head_mode_changed(self, text: str) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.head_mode = text
            self._is_updating_ui = True
            self.spin_proc_cameras.setValue(self.current_vm.gamma_cameras)
            self.txt_proc_head_angles.setText(", ".join(f"{a:.1f}" for a in self.current_vm.head_angles))
            self._is_updating_ui = False

    def _on_proc_head_angles_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            raw = self.txt_proc_head_angles.text().strip()
            angles = []
            for p in raw.split(','):
                try:
                    angles.append(float(p.strip()))
                except ValueError:
                    pass
            if angles:
                self.current_vm.head_angles = angles

    def _on_proc_endpoint_toggled(self, checked: bool) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, SpectProcedureViewModel):
            self.current_vm.endpoint = checked

    # Специфичные слоты для обработчиков данных
    def _on_dh_show_escaped_toggled(self, checked: bool) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, DirectStreamHandlerViewModel):
            self.current_vm.show_escaped_tracks = checked

    def _on_dh_vols_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, (SensitiveVolumeHandlerViewModel, HistoryAssemblerHandlerViewModel)):
            vols = [v.strip() for v in self.txt_dh_vols.text().split(',') if v.strip()]
            self.current_vm.sensitive_volumes = vols

    def _on_dh_initial_states_toggled(self, checked: bool) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, HistoryAssemblerHandlerViewModel):
            self.current_vm.save_initial_states = checked

    def _on_dh_grid_names_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, DoseMapHandlerViewModel):
            grids = [g.strip() for g in self.txt_dh_grid_names.text().split(',') if g.strip()]
            self.current_vm.grid_names = grids

    def _on_dh_shm_name_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, DoseMapHandlerViewModel):
            self.current_vm.shm_name = self.txt_dh_shm_name.text().strip()

    # Специфичные слоты для DataManager
    def _on_dm_filename_edited(self) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, DataManagerViewModel):
            self.current_vm.filename = self.txt_dm_filename.text().strip()

    def _on_dm_browse_clicked(self) -> None:
        cur_file = self.txt_dm_filename.text() or "simulation_results.h5"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Выбрать HDF5 файл для сохранения результатов",
            cur_file,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)"
        )
        if path:
            self.txt_dm_filename.setText(path)
            if isinstance(self.current_vm, DataManagerViewModel):
                self.current_vm.filename = path

    def _on_dm_buffer_changed(self, val: int) -> None:
        if not self._is_updating_ui and isinstance(self.current_vm, DataManagerViewModel):
            effective_val = max(val, self._min_buffer_capacity)
            self.current_vm.buffer_capacity = effective_val
