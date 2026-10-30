"""
Главное окно графического интерфейса NMSimToolkit.
Объединяет 3D-вьюпорт, док-панели дерева сцены, инспектора свойств, протоколов,
обработчиков данных, очереди задач и панели результатов.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import hepunits as units
from PySide6.QtCore import Qt, QSize, QTimer, QEvent
from PySide6.QtGui import QAction, QKeySequence, QActionGroup
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar,
    QFileDialog, QMessageBox, QLabel, QSpinBox, QComboBox,
    QApplication, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QWidget
)

from gui.viewport_3d.transform_gizmo import GizmoMode, GizmoSpace

from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.property_inspector import PropertyInspector
from gui.views.results_viewer import ResultsViewer
from gui.views.procedure_selector_widget import ProcedureSelectorWidget
from gui.views.data_handler_list_widget import DataHandlerListWidget
from gui.views.simulation_jobs_widget import SimulationJobsWidget
from gui.views.simulation_settings_dialog import SimulationSettingsDialog
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    StepAndShootProcedureViewModel,
    SpectProcedureViewModel,
    procedure_from_config,
)
from gui.viewmodels.data_handler_viewmodel import (
    DataManagerViewModel,
    DirectStreamHandlerViewModel,
    SensitiveVolumeHandlerViewModel,
)
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.controllers.orchestrator_session import OrchestratorSession
from gui.controllers.viewport_controller import SceneViewportController
from gui.models.gui_settings import GuiSimulationSettings
from core.scene.nodes import CompositeNode, SpatialNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.config.exporter import SceneExporter
from core.config.template_binder import ConfigTemplateBinder
from core.config.yaml_loader import load_simulation_config, load_raw_config
from core.config.builder import SceneBuilder
from core.config.models import (
    CustomSweepProtocolConfig,
    DataManagerConfig,
    SimulationManagerConfig,
    StepAndShootProtocolConfig,
)
import settings.database_setting as database_setting

_logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Главное окно графического интерфейса NMSimToolkit.
    Объединяет 3D-вьюпорт, дерево сцены, инспектор свойств и панель результатов
    в модульную архитектуру док-панелей (QDockWidget).
    Синхронизация 3D-вьюпорта делегирована SceneViewportController,
    а параллельный расчет — OrchestratorSession.
    """

    def __init__(self, scene_vm: Optional[SceneViewModel] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NMSimToolkit - Интерактивное 3D моделирование ядерной медицины")
        self.resize(1400, 900)

        self.viewport_controller: Optional[SceneViewportController] = None
        self._connected_procedure_vm: Optional[BaseProcedureViewModel] = None

        # Модель представления сцены
        if scene_vm is None:
            default_root = CompositeNode(name="WorldScene")
            # Базовый объем воды для наглядности
            water_mat = database_setting.material_database.get('Water, Liquid', Material(name='Water, Liquid'))
            default_vol = Volume(geometry=Box(200.0, 200.0, 200.0), material=water_mat, name="WaterPhantom")
            default_root.add_child(default_vol)
            self._scene_vm = SceneViewModel(default_root)
        else:
            self._scene_vm = scene_vm

        self.current_config: Any = None
        self.current_config_path: Optional[str] = None
        self.shm_name: str = "nmsim_gui_proj_shm"
        self.projection_shape = (128, 128)

        self.sim_settings: GuiSimulationSettings = GuiSimulationSettings()
        self.lbl_grid_snap: Optional[QLabel] = None
        self.lbl_angle_snap: Optional[QLabel] = None

        self._init_components()
        self._init_docks()
        self._init_gizmo_actions()
        self._init_menus()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()
        self._update_action_states(running=False, paused=False)

        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.installEventFilter(self)

    @property
    def scene_vm(self) -> SceneViewModel:
        return self._scene_vm

    @scene_vm.setter
    def scene_vm(self, vm: SceneViewModel) -> None:
        self._scene_vm = vm
        if self.viewport_controller is not None:
            self.viewport_controller.set_scene_viewmodel(vm)
        self._sync_detector_metadata()

    def _init_components(self) -> None:
        # Центральный 3D вьюпорт
        self.viewport = VTKViewport(self)
        self.setCentralWidget(self.viewport)

        # Выделенный контроллер синхронизации сцены и 3D-вьюпорта (SRP)
        self.viewport_controller = SceneViewportController(
            viewport=self.viewport,
            scene_vm=self._scene_vm,
            parent=self,
        )

        # Фасад сессии моделирования (Mediator / Session Controller)
        self.session: Optional[OrchestratorSession] = None

        # Дерево сцены
        self.scene_tree = SceneTreeWidget(self.scene_vm, self)

        # Инспектор свойств
        self.property_inspector = PropertyInspector(self)
        self.property_inspector.set_scene_viewmodel(self.scene_vm)

        # Панель результатов
        self.results_viewer = ResultsViewer(self)

        # Модели процедур и диспетчера данных
        self.procedure_vm: BaseProcedureViewModel = SpectProcedureViewModel()
        self.property_inspector.set_procedure_viewmodel(self.procedure_vm)
        self.viewport_controller.procedure_vm = self.procedure_vm
        self.data_manager_vm = DataManagerViewModel()
        self.viewport_controller.data_manager_vm = self.data_manager_vm

        # Диспетчер параллельной оркестрации (OrchestratorSession)
        self.orchestrator_session = OrchestratorSession(
            scene_vm=self.scene_vm,
            procedure_vm=self.procedure_vm,
            data_manager_vm=self.data_manager_vm,
            parent=self,
        )
        self.session = self.orchestrator_session

        # Новые модульные виджеты
        self.procedure_selector = ProcedureSelectorWidget(scene_vm=self.scene_vm, parent=self)
        self.data_handler_list = DataHandlerListWidget(data_manager_vm=self.data_manager_vm, parent=self)
        self.jobs_widget = SimulationJobsWidget(parent=self)

    def _init_docks(self) -> None:
        # 1. Левый док: Дерево сцены
        self.dock_tree = QDockWidget("Иерархия сцены", self)
        self.dock_tree.setWidget(self.scene_tree)
        self.dock_tree.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_tree)

        # 2. Левый док: Процедуры
        self.dock_procedures = self.procedure_selector
        self.dock_procedures.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_procedures)

        # 3. Левый док: Обработчики данных
        self.dock_handlers = self.data_handler_list
        self.dock_handlers.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_handlers)

        # Табификация доков слева
        self.tabifyDockWidget(self.dock_tree, self.dock_procedures)
        self.tabifyDockWidget(self.dock_procedures, self.dock_handlers)

        # 4. Правый док: Инспектор свойств
        self.dock_inspector = QDockWidget("Свойства объекта", self)
        self.dock_inspector.setWidget(self.property_inspector)
        self.dock_inspector.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_inspector)

        # 5. Правый док: Задачи симуляции (Jobs)
        self.dock_jobs = self.jobs_widget
        self.dock_jobs.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_jobs)
        self.tabifyDockWidget(self.dock_inspector, self.dock_jobs)

        # 6. Нижний док: Результаты и телеметрия
        self.dock_results = QDockWidget("Телеметрия и результаты", self)
        self.dock_results.setWidget(self.results_viewer)
        self.dock_results.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_results)

    def _init_gizmo_actions(self) -> None:
        """Инициализация действий и элементов управления интерактивного манипулятора Gizmo."""
        self.gizmo_action_group = QActionGroup(self)
        self.gizmo_action_group.setExclusive(True)

        self.act_gizmo_translate = QAction("☩ Перемещение (W)", self)
        self.act_gizmo_translate.setCheckable(True)
        self.act_gizmo_translate.setChecked(True)
        self.act_gizmo_translate.setShortcuts([QKeySequence(Qt.Key.Key_W), QKeySequence("Ц"), QKeySequence("ц")])
        self.act_gizmo_translate.setToolTip("Режим перемещения объекта вдоль осей или плоскостей (W / Ц)")
        self.act_gizmo_translate.triggered.connect(lambda: self._set_gizmo_mode(GizmoMode.TRANSLATE))
        self.gizmo_action_group.addAction(self.act_gizmo_translate)

        self.act_gizmo_rotate = QAction("⟲ Вращение (E)", self)
        self.act_gizmo_rotate.setCheckable(True)
        self.act_gizmo_rotate.setShortcuts([QKeySequence(Qt.Key.Key_E), QKeySequence("У"), QKeySequence("у")])
        self.act_gizmo_rotate.setToolTip("Режим вращения объекта вокруг координатных осей (E / У)")
        self.act_gizmo_rotate.triggered.connect(lambda: self._set_gizmo_mode(GizmoMode.ROTATE))
        self.gizmo_action_group.addAction(self.act_gizmo_rotate)

        self.act_gizmo_scale = QAction("⤢ Масштаб (R)", self)
        self.act_gizmo_scale.setCheckable(True)
        self.act_gizmo_scale.setShortcuts([QKeySequence(Qt.Key.Key_R), QKeySequence("К"), QKeySequence("к")])
        self.act_gizmo_scale.setToolTip("Режим масштабирования объекта (R / К)")
        self.act_gizmo_scale.triggered.connect(lambda: self._set_gizmo_mode(GizmoMode.SCALE))
        self.gizmo_action_group.addAction(self.act_gizmo_scale)

        self.act_gizmo_space = QAction("🌐 Мировые (Q)", self)
        self.act_gizmo_space.setCheckable(True)
        self.act_gizmo_space.setShortcuts([QKeySequence(Qt.Key.Key_Q), QKeySequence("Й"), QKeySequence("й")])
        self.act_gizmo_space.setToolTip("Переключение системы координат: Мировая / Локальная (Q / Й)")
        self.act_gizmo_space.triggered.connect(self._toggle_gizmo_space)

        self.combo_grid_snap = QComboBox()
        self.combo_grid_snap.addItems(["Выкл", "1 мм", "5 мм", "10 мм", "25 мм", "50 мм", "100 мм"])
        self.combo_grid_snap.setCurrentText("10 мм")
        self.combo_grid_snap.setToolTip("Шаг дискретной координатной сетки привязки (Shift — временное отключение)")
        self.combo_grid_snap.currentTextChanged.connect(self._on_grid_snap_changed)

        self.combo_angle_snap = QComboBox()
        self.combo_angle_snap.addItems(["Выкл", "5°", "10°", "15°", "30°", "45°", "90°"])
        self.combo_angle_snap.setCurrentText("5°")
        self.combo_angle_snap.setToolTip("Шаг угловой привязки вращения (Shift — временное отключение)")
        self.combo_angle_snap.currentTextChanged.connect(self._on_angle_snap_changed)

    def _init_menus(self) -> None:
        menubar = self.menuBar()

        # Меню Файл
        file_menu = menubar.addMenu("&Файл")

        act_new = QAction("&Новая сцена", self)
        act_new.triggered.connect(self._on_new_scene)
        file_menu.addAction(act_new)

        act_open = QAction("&Открыть конфигурацию (YAML)...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open_yaml)
        file_menu.addAction(act_open)

        act_save = QAction("&Сохранить конфигурацию (YAML)...", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save_yaml)
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        act_exit = QAction("&Выход", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Меню Моделирование
        sim_menu = menubar.addMenu("&Моделирование")
        self.act_run = QAction("▶ &Запуск", self)
        self.act_pause = QAction("⏸ Пауза &пула", self)
        self.act_resume = QAction("⏯ Возобновить п&ул", self)
        self.act_pause_focused = QAction("⏸ Пауза &фокуса", self)
        self.act_resume_focused = QAction("⏯ Возобновить фок&ус", self)
        self.act_stop = QAction("⏹ &Остановить", self)
        self.act_step = QAction("⏭ &Шаг фокуса", self)

        sim_menu.addAction(self.act_run)
        sim_menu.addAction(self.act_pause)
        sim_menu.addAction(self.act_resume)
        sim_menu.addSeparator()
        sim_menu.addAction(self.act_pause_focused)
        sim_menu.addAction(self.act_resume_focused)
        sim_menu.addAction(self.act_step)
        sim_menu.addSeparator()
        sim_menu.addAction(self.act_stop)
        sim_menu.addSeparator()

        self.act_settings = QAction("⚙ &Параметры расчета...", self)
        self.act_settings.setToolTip("Настройка параметров моделирования, ОФЭКТ сканирования и треков")
        self.act_settings.triggered.connect(self._on_open_simulation_settings)
        sim_menu.addAction(self.act_settings)

        # Меню Вид
        view_menu = menubar.addMenu("&Вид")
        act_cam_reset = QAction("Сброс камеры", self)
        act_cam_reset.triggered.connect(self.viewport.reset_camera)
        view_menu.addAction(act_cam_reset)

        act_iso = QAction("Изометрия", self)
        act_iso.triggered.connect(self.viewport.view_isometric)
        view_menu.addAction(act_iso)

        act_xy = QAction("Вид сверху (XY)", self)
        act_xy.triggered.connect(self.viewport.view_xy)
        view_menu.addAction(act_xy)

        view_menu.addSeparator()
        self.act_toggle_tracks = QAction("Отображать треки", self)
        self.act_toggle_tracks.setCheckable(True)
        self.act_toggle_tracks.setChecked(True)
        if self.viewport_controller is not None:
            self.act_toggle_tracks.toggled.connect(self.viewport_controller.track_renderer.set_visible)
        view_menu.addAction(self.act_toggle_tracks)

        self.act_toggle_dose = QAction("Отображать карту дозы", self)
        self.act_toggle_dose.setCheckable(True)
        self.act_toggle_dose.setChecked(True)
        if self.viewport_controller is not None:
            self.act_toggle_dose.toggled.connect(self.viewport_controller.dose_renderer.set_visible)
        view_menu.addAction(self.act_toggle_dose)

        view_menu.addSeparator()
        gizmo_menu = view_menu.addMenu("3D Манипулятор (Gizmo)")
        gizmo_menu.addAction(self.act_gizmo_translate)
        gizmo_menu.addAction(self.act_gizmo_rotate)
        gizmo_menu.addAction(self.act_gizmo_scale)
        gizmo_menu.addSeparator()
        gizmo_menu.addAction(self.act_gizmo_space)

        view_menu.addSeparator()
        view_menu.addAction(self.dock_tree.toggleViewAction())
        view_menu.addAction(self.dock_procedures.toggleViewAction())
        view_menu.addAction(self.dock_handlers.toggleViewAction())
        view_menu.addAction(self.dock_inspector.toggleViewAction())
        view_menu.addAction(self.dock_jobs.toggleViewAction())
        view_menu.addAction(self.dock_results.toggleViewAction())

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("Быстрый доступ", self)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        toolbar.addAction(self.act_run)
        toolbar.addAction(self.act_pause)
        toolbar.addAction(self.act_resume)
        toolbar.addSeparator()
        toolbar.addAction(self.act_pause_focused)
        toolbar.addAction(self.act_resume_focused)
        toolbar.addAction(self.act_step)
        toolbar.addSeparator()
        toolbar.addAction(self.act_stop)
        toolbar.addSeparator()
        toolbar.addAction(self.act_settings)
        toolbar.addSeparator()

        act_reset = QAction("⟲ Камера", self)
        act_reset.triggered.connect(self.viewport.reset_camera)
        toolbar.addAction(act_reset)
        toolbar.addSeparator()

        self.lbl_preview_view = QLabel("Шаг:")
        toolbar.addWidget(self.lbl_preview_view)
        self.spin_preview_view = QSpinBox()
        total_steps = self.procedure_vm.steps if isinstance(self.procedure_vm, StepAndShootProcedureViewModel) else 1
        self.spin_preview_view.setRange(1, max(1, total_steps))
        self.spin_preview_view.setValue(1)
        self.spin_preview_view.setToolTip("Предварительный просмотр ориентации детекторных головок ОФЭКТ для выбранного шага")
        self.spin_preview_view.valueChanged.connect(self._on_preview_view_changed)
        toolbar.addWidget(self.spin_preview_view)

        # Панель инструментов: 3D Манипулятор (Transform Gizmo)
        gizmo_toolbar = QToolBar("3D Манипулятор", self)
        gizmo_toolbar.setObjectName("GizmoToolBar")
        gizmo_toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(gizmo_toolbar)

        gizmo_toolbar.addAction(self.act_gizmo_translate)
        gizmo_toolbar.addAction(self.act_gizmo_rotate)
        gizmo_toolbar.addAction(self.act_gizmo_scale)
        gizmo_toolbar.addSeparator()
        gizmo_toolbar.addAction(self.act_gizmo_space)
        gizmo_toolbar.addSeparator()

        self.lbl_grid_snap = QLabel("Сетка:")
        gizmo_toolbar.addWidget(self.lbl_grid_snap)
        gizmo_toolbar.addWidget(self.combo_grid_snap)

        self.lbl_angle_snap = QLabel("Угол:")
        gizmo_toolbar.addWidget(self.lbl_angle_snap)
        gizmo_toolbar.addWidget(self.combo_angle_snap)

    def _init_statusbar(self) -> None:
        status = self.statusBar()
        self.lbl_status = QLabel("Статус: Готов (IDLE)")
        status.addWidget(self.lbl_status)

    def _connect_signals(self) -> None:
        # Связь выбора узла в дереве с инспектором и 3D-манипулятором
        self.scene_vm.node_selected.connect(self.property_inspector.set_target_viewmodel)
        self.scene_vm.node_selected.connect(self._on_node_selected)

        # Связь выбора процедуры и обработчиков с инспектором
        self.procedure_selector.procedure_changed.connect(self._on_procedure_changed)
        self.procedure_selector.procedure_selected.connect(self.property_inspector.set_target_viewmodel)
        self.data_handler_list.handler_selected.connect(self.property_inspector.set_target_viewmodel)
        self.data_handler_list.data_manager_selected.connect(self.property_inspector.set_target_viewmodel)
        self.property_inspector.data_handler_section.sensitive_volumes_editor.volume_selected.connect(
            self._on_sensitive_volume_selected_in_editor
        )

        # Связь генерации задач и управления воркерами
        self.jobs_widget.generate_jobs_requested.connect(self._on_generate_jobs)
        self.jobs_widget.job_preview_requested.connect(self._on_job_preview_requested)
        self.jobs_widget.focused_job_changed.connect(self._on_focused_job_changed)
        self.jobs_widget.pool_size_changed.connect(self._on_pool_size_changed)

        # Подписка виджета задач на сигналы сессии оркестратора
        self.orchestrator_session.jobs_generated.connect(self.jobs_widget.set_jobs)
        self.orchestrator_session.job_started.connect(self.jobs_widget.on_job_started)
        self.orchestrator_session.job_progress.connect(self.jobs_widget.on_job_progress)
        self.orchestrator_session.job_finished.connect(self.jobs_widget.on_job_finished)
        self.orchestrator_session.session_finished.connect(self._on_simulation_finished)
        self.orchestrator_session.session_error.connect(
            lambda err: QMessageBox.critical(self, "Ошибка", f"Сбой расчета:\n{err}")
        )

        # Потоковая визуализация от сфокусированного воркера
        if self.viewport_controller is not None:
            self.orchestrator_session.tracks_received.connect(
                self.viewport_controller.track_renderer.add_tracks_batch
            )
        self.orchestrator_session.projection_received.connect(self.results_viewer.set_projection_data)
        self.orchestrator_session.projection_stack_updated.connect(self._on_projection_stack_updated)
        self.orchestrator_session.spectrum_received.connect(self.results_viewer.set_spectrum_data)
        self.orchestrator_session.detector_metadata_updated.connect(self.results_viewer.set_detector_metadata)
        self.orchestrator_session.dose_volume_received.connect(self._on_dose_volume_received)
        self.orchestrator_session.stats_updated.connect(
            lambda cnt, cps: self.lbl_status.setText(f"Моделирование: {cnt:,} отсчетов ({cps:.1f} CPS)")
        )

        # Раздельные подписки для оптимизированной инкрементальной синхронизации
        self.scene_vm.scene_loaded.connect(self._on_scene_loaded_sync)
        self.scene_vm.node_added.connect(self._on_node_added)
        self.scene_vm.node_removed.connect(self._on_node_removed)
        self.scene_vm.node_renamed.connect(self._on_node_renamed_sync)

        # Синхронизация списка чувствительных детекторов и шкалы спектра с панелью результатов
        self.data_manager_vm.changed.connect(self._sync_detector_metadata)
        self.data_manager_vm.handlers_changed.connect(self._sync_detector_metadata)
        self.scene_vm.scene_loaded.connect(lambda _root: self._sync_detector_metadata())
        self.scene_vm.node_added.connect(lambda _node: self._sync_detector_metadata())
        self.scene_vm.node_removed.connect(lambda _node: self._sync_detector_metadata())
        self.scene_vm.node_property_changed.connect(self._on_scene_node_property_changed)

        # Автоматическая генерация задач при изменении сцены и параметров процедуры
        self.scene_vm.node_added.connect(lambda _node: self._on_generate_jobs())
        self.scene_vm.node_removed.connect(lambda _node: self._on_generate_jobs())
        self._connect_procedure_vm_signals(self.procedure_vm)

        # Управляющие действия моделирования
        self.act_run.triggered.connect(self._on_start_simulation)
        self.act_pause.triggered.connect(self._on_pause_all)
        self.act_resume.triggered.connect(self._on_resume_all)
        self.act_pause_focused.triggered.connect(self._on_pause_focused)
        self.act_resume_focused.triggered.connect(self._on_resume_focused)
        self.act_stop.triggered.connect(self._on_stop_simulation)
        self.act_step.triggered.connect(self._on_step_simulation)

        # Сигналы сессии оркестратора для паузы и возобновления
        self.orchestrator_session.session_paused.connect(self._on_session_paused)
        self.orchestrator_session.session_resumed.connect(self._on_session_resumed)
        self.orchestrator_session.focused_worker_paused.connect(
            lambda tid: self.jobs_widget.on_job_paused(tid, is_local=True)
        )
        self.orchestrator_session.focused_worker_resumed.connect(
            lambda tid: self.jobs_widget.on_job_resumed(tid)
        )

        # Сброс накопления и управление параметрами энергетического спектра из панели результатов
        self.results_viewer.accumulation_cleared.connect(self._on_clear_accumulation)
        self.results_viewer.spectrum_settings_changed.connect(self._on_spectrum_settings_changed)
        self.results_viewer.auto_max_energy_requested.connect(self._on_auto_max_energy_requested)

        # Синхронизация состояния манипулятора Gizmo с панелью инструментов
        if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
            self.viewport_controller.transform_gizmo.mode_changed.connect(self._on_gizmo_mode_changed)
            self.viewport_controller.transform_gizmo.space_changed.connect(self._on_gizmo_space_changed)
            self.viewport_controller.transform_gizmo.status_message_requested.connect(self.lbl_status.setText)

        # Начальная отрисовка сцены во вьюпорте и генерация задач
        if self.viewport_controller is not None:
            self.viewport_controller.sync_viewport_scene()
        self._sync_detector_metadata()
        self._on_generate_jobs()

    def _on_scene_node_property_changed(self, _node_vm: NodeViewModel, prop_name: str, _value: Any) -> None:
        """Реактивный пересчет авто-предела спектра при изменении спектра энергии источника в сцене."""
        if prop_name in ("spectrum", "energy"):
            self._sync_detector_metadata()

    def _on_spectrum_settings_changed(self, max_energy: float, energy_bins: int, auto_max_energy: bool) -> None:
        """Синхронизация параметров энергетического спектра и числа бинов из ResultsViewer в OrchestratorSession."""
        if self.orchestrator_session is None:
            return
        resolved_energy = self.orchestrator_session.set_spectrum_parameters(
            max_energy=float(max_energy) * float(units.keV),
            energy_bins=int(energy_bins),
            auto_max_energy=bool(auto_max_energy),
        )
        if auto_max_energy and self.results_viewer is not None:
            self.results_viewer.set_spectrum_configuration(
                max_energy=float(resolved_energy / units.keV),
                energy_bins=self.orchestrator_session.energy_bins,
                auto_max_energy=True,
            )

    def _on_auto_max_energy_requested(self) -> None:
        """Принудительный пересчет верхней границы спектра по источникам сцены по запросу пользователя."""
        if self.orchestrator_session is not None:
            self.orchestrator_session.auto_max_energy = True
        self._sync_detector_metadata()

    def _on_node_renamed_sync(self, old_name: str, new_name: str) -> None:
        """Обновляет привязки слотов GammaCamera, чувствительные объемы и метаданные детекторов при переименовании узла."""
        if self.scene_vm is not None:
            for node_vm in self.scene_vm.all_nodes():
                if isinstance(node_vm, GammaCameraViewModel):
                    slots_cfg = node_vm.slots
                    cam_core = node_vm.core_node
                    if slots_cfg.casing == old_name:
                        slots_cfg.casing = new_name
                        cam_core.slots["casing"] = new_name
                    if slots_cfg.detector_box == old_name:
                        slots_cfg.detector_box = new_name
                        cam_core.slots["detector_box"] = new_name
                    if slots_cfg.collimator == old_name:
                        slots_cfg.collimator = new_name
                        cam_core.slots["collimator"] = new_name
                    if slots_cfg.crystal == old_name:
                        slots_cfg.crystal = new_name
                        cam_core.slots["crystal"] = new_name
                    if slots_cfg.glass_backend == old_name:
                        slots_cfg.glass_backend = new_name
                        cam_core.slots["glass_backend"] = new_name
            for _cam_core, reg_slots in self.scene_vm.slots_registry.items():
                if reg_slots.casing == old_name:
                    reg_slots.casing = new_name
                if reg_slots.detector_box == old_name:
                    reg_slots.detector_box = new_name
                if reg_slots.collimator == old_name:
                    reg_slots.collimator = new_name
                if reg_slots.crystal == old_name:
                    reg_slots.crystal = new_name
                if reg_slots.glass_backend == old_name:
                    reg_slots.glass_backend = new_name

        self.data_manager_vm.on_node_renamed(old_name, new_name)
        self._sync_detector_metadata()

    def _sync_detector_metadata(self) -> None:
        """
        Собирает имена активных чувствительных объемов детекторов из DataManagerViewModel
        и дерева сцены SceneViewModel, рассчитывает актуальный max_energy и передает их в ResultsViewer.
        """
        if self.results_viewer is None:
            return

        det_names: List[str] = []
        if self.data_manager_vm is not None:
            for handler_vm in self.data_manager_vm.handlers:
                if isinstance(handler_vm, DirectStreamHandlerViewModel):
                    self.sim_settings.show_escaped_tracks = bool(handler_vm.show_escaped_tracks)
                if isinstance(handler_vm, SensitiveVolumeHandlerViewModel) and handler_vm.enabled:
                    for vol_name in handler_vm.sensitive_volumes:
                        clean_name = str(vol_name).strip()
                        if clean_name and clean_name not in det_names:
                            det_names.append(clean_name)

        if not det_names and self.scene_vm is not None:
            for node_vm in self.scene_vm.all_nodes():
                if isinstance(node_vm, GammaCameraViewModel):
                    crystal_vm = node_vm.crystal_vm
                    c_name = str(crystal_vm.name if crystal_vm is not None else (node_vm.slots.crystal or "")).strip()
                    if c_name and c_name not in det_names:
                        det_names.append(c_name)
            if not det_names:
                for node_vm in self.scene_vm.all_nodes():
                    if isinstance(node_vm, VolumeViewModel):
                        name_lower = (node_vm.name or "").lower()
                        if ("detector" in name_lower or "crystal" in name_lower) and "detector_box" not in name_lower:
                            if node_vm.name not in det_names:
                                det_names.append(node_vm.name)

        if self.orchestrator_session is not None:
            self.orchestrator_session.scene_vm = self.scene_vm
            resolved_energy = self.orchestrator_session.resolve_effective_max_energy()
            display_max_energy = float(resolved_energy / units.keV)
            energy_bins = int(self.orchestrator_session.energy_bins)
            auto_max_energy = bool(self.orchestrator_session.auto_max_energy)
        else:
            display_max_energy = 300.0
            energy_bins = 256
            auto_max_energy = True

        self.results_viewer.set_detector_metadata(
            det_names,
            max_energy=display_max_energy,
            energy_bins=energy_bins,
            auto_max_energy=auto_max_energy,
        )

    def _on_scene_loaded_sync(self, _root_vm: Any = None) -> None:
        """Синхронизация 3D-вьюпорта и автошкалы энергетического спектра после полной перезагрузки сцены."""
        if self.viewport_controller is not None:
            self.viewport_controller.sync_viewport_scene()
        if self.orchestrator_session is not None:
            self.orchestrator_session.auto_max_energy = True
        self._sync_detector_metadata()
        if self.results_viewer is not None:
            self.results_viewer.reset_energy_window()

    def _on_node_transform_changed(self, node_vm: NodeViewModel) -> None:
        """Инкрементальное обновление матрицы трансформации актора без пересоздания меша."""
        if self.viewport_controller is not None:
            self.viewport_controller.on_node_transform_changed(node_vm)

    def _on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any) -> None:
        """Инкрементальное обновление параметров актора при смене геометрии или свойств."""
        if self.viewport_controller is not None:
            self.viewport_controller.on_node_property_changed(node_vm, prop_name, value)

    def _on_node_added(self, node_vm: NodeViewModel) -> None:
        """Точечное добавление нового актора в сцену."""
        if self.viewport_controller is not None:
            self.viewport_controller.on_node_added(node_vm)

    def _on_node_removed(self, node_vm: NodeViewModel) -> None:
        """Точечное удаление актора из сцены и очистка удаленных узлов поддерева из обработчиков."""
        if self.viewport_controller is not None:
            self.viewport_controller.on_node_removed(node_vm)
        stack: List[NodeViewModel] = [node_vm]
        while stack:
            curr_vm = stack.pop()
            self.data_manager_vm.on_node_removed(curr_vm.name)
            stack.extend(curr_vm.children)

    def _on_sensitive_volume_selected_in_editor(self, volume_name: str) -> None:
        """
        Подсвечивает выбранный в редакторе детектора объем в 3D-вьюпорте
        без смены фокуса и целевой модели представления PropertyInspector.
        """
        if self.scene_vm is not None and self.viewport_controller is not None:
            node_vm = self.scene_vm.find_by_name(volume_name)
            if node_vm is not None:
                self.viewport_controller.on_node_selected(node_vm)

    def _on_node_selected(self, node_view_model: Optional[NodeViewModel]) -> None:
        if self.dock_inspector.isHidden():
            self.dock_inspector.show()
        self.dock_inspector.raise_()
        if self.viewport_controller is not None:
            self.viewport_controller.on_node_selected(node_view_model)
        self._update_gizmo_toolbar_state(node_view_model)
        if node_view_model is not None:
            constraint = node_view_model.kinematic_constraint
            is_fixed = False
            if constraint is not None:
                has_trans = bool(constraint.get_allowed_axes(GizmoMode.TRANSLATE))
                has_rot = bool(constraint.get_allowed_axes(GizmoMode.ROTATE))
                has_scale = bool(constraint.get_allowed_axes(GizmoMode.SCALE)) and constraint.is_scale_allowed()
                is_fixed = not has_trans and not has_rot and not has_scale

            if is_fixed:
                self.lbl_status.setText(
                    f"Выбран объект '{node_view_model.name}'. Трансформация манипулятором Gizmo заблокирована."
                )
            else:
                self.lbl_status.setText(
                    f"Выбран объект '{node_view_model.name}'. Манипулятор Gizmo: W/Ц — перемещение, "
                    f"E/У — вращение, R/К — масштаб, Q/Й — система координат, Shift — отключение привязки"
                )
        else:
            self.lbl_status.setText("Статус: Готов (IDLE)")

    def _update_gizmo_toolbar_state(self, node_view_model: Optional[NodeViewModel]) -> None:
        """
        Обновление видимости и доступности кнопок тулбара 3D-манипулятора
        в соответствии с кинематическими ограничениями выбранного узла.
        """
        constraint = node_view_model.kinematic_constraint if node_view_model is not None else None

        if constraint is not None:
            allowed_trans_axes = constraint.get_allowed_axes(GizmoMode.TRANSLATE)
            allowed_rot_axes = constraint.get_allowed_axes(GizmoMode.ROTATE)
            allowed_scale_axes = constraint.get_allowed_axes(GizmoMode.SCALE)
            has_translation = bool(allowed_trans_axes)
            has_rotation = bool(allowed_rot_axes)
            has_scale = bool(allowed_scale_axes) and constraint.is_scale_allowed()
            all_disabled = not has_translation and not has_rotation and not has_scale

            self.act_gizmo_translate.setVisible(has_translation)
            self.act_gizmo_rotate.setVisible(has_rotation)
            self.act_gizmo_scale.setVisible(has_scale)

            if all_disabled:
                self.act_gizmo_space.setVisible(False)
                self.combo_grid_snap.setVisible(False)
                self.combo_angle_snap.setVisible(False)
                if self.lbl_grid_snap is not None:
                    self.lbl_grid_snap.setVisible(False)
                if self.lbl_angle_snap is not None:
                    self.lbl_angle_snap.setVisible(False)
            else:
                self.act_gizmo_space.setVisible(True)
                self.combo_grid_snap.setVisible(has_translation or has_scale)
                self.combo_angle_snap.setVisible(has_rotation)
                if self.lbl_grid_snap is not None:
                    self.lbl_grid_snap.setVisible(has_translation or has_scale)
                if self.lbl_angle_snap is not None:
                    self.lbl_angle_snap.setVisible(has_rotation)

                forced_space = constraint.get_forced_space()
                if forced_space is not None:
                    self.act_gizmo_space.setEnabled(False)
                    if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
                        self.viewport_controller.transform_gizmo.space = forced_space
                        self._on_gizmo_space_changed(forced_space)
                else:
                    self.act_gizmo_space.setEnabled(True)

            # Если текущий активный режим манипулятора заблокирован для выбранного узла,
            # переключаем манипулятор на первый разрешенный режим
            if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
                current_mode = self.viewport_controller.transform_gizmo.mode
                if current_mode == GizmoMode.TRANSLATE and not has_translation:
                    if has_rotation:
                        self._set_gizmo_mode(GizmoMode.ROTATE)
                        self.act_gizmo_rotate.setChecked(True)
                    elif has_scale:
                        self._set_gizmo_mode(GizmoMode.SCALE)
                        self.act_gizmo_scale.setChecked(True)
                elif current_mode == GizmoMode.ROTATE and not has_rotation:
                    if has_translation:
                        self._set_gizmo_mode(GizmoMode.TRANSLATE)
                        self.act_gizmo_translate.setChecked(True)
                    elif has_scale:
                        self._set_gizmo_mode(GizmoMode.SCALE)
                        self.act_gizmo_scale.setChecked(True)
                elif current_mode == GizmoMode.SCALE and not has_scale:
                    if has_translation:
                        self._set_gizmo_mode(GizmoMode.TRANSLATE)
                        self.act_gizmo_translate.setChecked(True)
                    elif has_rotation:
                        self._set_gizmo_mode(GizmoMode.ROTATE)
                        self.act_gizmo_rotate.setChecked(True)
        else:
            # При отсутствии ограничений все кнопки видимы и активны
            self.act_gizmo_translate.setVisible(True)
            self.act_gizmo_rotate.setVisible(True)
            self.act_gizmo_scale.setVisible(True)
            self.act_gizmo_space.setVisible(True)
            self.act_gizmo_space.setEnabled(True)
            self.combo_grid_snap.setVisible(True)
            self.combo_angle_snap.setVisible(True)
            if self.lbl_grid_snap is not None:
                self.lbl_grid_snap.setVisible(True)
            if self.lbl_angle_snap is not None:
                self.lbl_angle_snap.setVisible(True)

    def _apply_job_angles_to_viewport(self, context: Dict[str, Any]) -> None:
        """Применяет углы из контекста задачи к гамма-камерам во вьюпорте."""
        if self.viewport_controller is not None:
            self.viewport_controller.apply_job_angles_to_viewport(context, self.procedure_vm)

    def _on_preview_view_changed(self, view_number_1based: int) -> None:
        """Предварительный кинематический поворот детекторов на выбранный шаг ОФЭКТ."""
        base_angle = (
            self.viewport_controller.preview_view(view_number_1based, self.procedure_vm)
            if self.viewport_controller is not None
            else 0.0
        )
        steps_total = self.procedure_vm.steps if isinstance(self.procedure_vm, StepAndShootProcedureViewModel) else 1
        self.lbl_status.setText(f"Предпросмотр ОФЭКТ: Шаг {view_number_1based}/{steps_total} (угол {base_angle:.1f}°)")

    def _on_projection_stack_updated(self, stack: np.ndarray, current_view: int, total_views: int, angle: float) -> None:
        """
        Обработка обновления проекционного стека при пошаговом ОФЭКТ сканировании.
        """
        self.results_viewer.set_projection_stack_data(stack, current_view, total_views, angle)
        self.spin_preview_view.blockSignals(True)
        self.spin_preview_view.setValue(current_view + 1)
        self.spin_preview_view.blockSignals(False)
        self.lbl_status.setText(f"Моделирование ОФЭКТ: Ракурс {current_view + 1}/{total_views} ({angle:.1f}°)")
        self.viewport.render()

    def _on_procedure_parameter_changed(self, param_name: str, _param_val: Any) -> None:
        """Обновляет диапазон шагов предпросмотра при изменении параметров процедуры ОФЭКТ."""
        if param_name in ("steps", "gamma_cameras") and isinstance(
            self.procedure_vm, StepAndShootProcedureViewModel
        ):
            self.spin_preview_view.setRange(1, max(1, int(self.procedure_vm.steps)))

    def _connect_procedure_vm_signals(self, proc_vm: BaseProcedureViewModel) -> None:
        """Переподключает сигналы модели представления процедуры без накопления дубликатов."""
        if self._connected_procedure_vm is not None:
            try:
                self._connected_procedure_vm.changed.disconnect(self._on_generate_jobs)
            except (TypeError, RuntimeError):
                pass
            if isinstance(self._connected_procedure_vm, StepAndShootProcedureViewModel):
                try:
                    self._connected_procedure_vm.parameter_changed.disconnect(
                        self._on_procedure_parameter_changed
                    )
                except (TypeError, RuntimeError):
                    pass

        self._connected_procedure_vm = proc_vm
        proc_vm.changed.connect(self._on_generate_jobs)
        if isinstance(proc_vm, StepAndShootProcedureViewModel):
            self.spin_preview_view.setRange(1, max(1, proc_vm.steps))
            proc_vm.parameter_changed.connect(self._on_procedure_parameter_changed)

    def _on_procedure_changed(self, proc_vm: BaseProcedureViewModel) -> None:
        """
        Смена активной процедуры сканирования.
        """
        self.procedure_vm = proc_vm
        if self.viewport_controller is not None:
            self.viewport_controller.on_procedure_changed(proc_vm)
        self.orchestrator_session.procedure_vm = proc_vm
        self.property_inspector.set_target_viewmodel(proc_vm)
        self.property_inspector.set_procedure_viewmodel(proc_vm)
        self._connect_procedure_vm_signals(proc_vm)
        self._on_generate_jobs()

    def _on_generate_jobs(self) -> None:
        """
        Генерация подзадач для текущей процедуры.
        """
        jobs = self.orchestrator_session.generate_jobs()
        self.jobs_widget.set_jobs(jobs)
        self.lbl_status.setText(f"Сформировано задач: {len(jobs)}")

    def _on_job_preview_requested(self, task_info: Dict[str, Any]) -> None:
        """
        Кинематический предпросмотр ориентации сканера для выбранной задачи.
        """
        self._apply_job_angles_to_viewport(task_info)
        view_idx = task_info.get('view_index', task_info.get('_task_id'))
        if view_idx is not None:
            self.spin_preview_view.blockSignals(True)
            self.spin_preview_view.setValue(int(view_idx) + 1)
            self.spin_preview_view.blockSignals(False)

    def _on_focused_job_changed(self, index: int) -> None:
        """
        Мгновенное переключение фокуса визуализации на указанную задачу без паузы и потери накопленных данных.
        """
        jobs = self.orchestrator_session.jobs
        if 0 <= index < len(jobs):
            self._apply_job_angles_to_viewport(jobs[index])
            self.spin_preview_view.blockSignals(True)
            self.spin_preview_view.setValue(index + 1)
            self.spin_preview_view.blockSignals(False)
        self.orchestrator_session.set_focused_job(index)
        self.lbl_status.setText(f"Фокус визуализации: Задача #{index + 1}")

    def _on_pool_size_changed(self, pool_size: int) -> None:
        """
        Изменение размера пула рабочих процессов.
        """
        val = max(1, int(pool_size))
        self.orchestrator_session.pool_size = val
        self.sim_settings.pool_size = val
        self._update_action_states(
            running=self.orchestrator_session.is_running,
            paused=self.orchestrator_session.is_paused,
            focused_paused=self.orchestrator_session.is_focused_worker_paused,
        )

    def _on_start_simulation(self) -> None:
        """
        Запуск параллельного моделирования через OrchestratorSession.
        """
        if self.orchestrator_session.is_running:
            return

        if self.scene_vm is None or self.scene_vm.root_vm is None:
            QMessageBox.warning(self, "Предупреждение", "Сцена пуста.")
            return

        if self.current_config is not None and self.current_config.simulation_manager is not None:
            cfg_mgr = self.current_config.simulation_manager
            raw_particles = cfg_mgr.particles_number
            if isinstance(raw_particles, (int, float)) and not isinstance(raw_particles, bool):
                self.orchestrator_session.particles_number = int(raw_particles)
            raw_buffer_capacity = cfg_mgr.buffer_capacity
            if raw_buffer_capacity is None:
                self.orchestrator_session.buffer_capacity = None
            elif isinstance(raw_buffer_capacity, (int, float)) and not isinstance(raw_buffer_capacity, bool):
                self.orchestrator_session.buffer_capacity = int(raw_buffer_capacity)
            if not isinstance(
                self.current_config.protocol,
                (StepAndShootProtocolConfig, CustomSweepProtocolConfig),
            ):
                raw_stop_time = cfg_mgr.stop_time
                if isinstance(raw_stop_time, (int, float)) and not isinstance(raw_stop_time, bool):
                    self.orchestrator_session.stop_time = float(raw_stop_time) / float(units.s)
            raw_min_energy = cfg_mgr.min_energy
            if isinstance(raw_min_energy, (int, float)) and not isinstance(raw_min_energy, bool):
                self.orchestrator_session.min_energy = float(raw_min_energy) / float(units.keV)
            raw_batch_capacity = cfg_mgr.batch_capacity
            self.orchestrator_session.batch_capacity = (
                int(raw_batch_capacity)
                if isinstance(raw_batch_capacity, (int, float)) and not isinstance(raw_batch_capacity, bool)
                else None
            )
            self.orchestrator_session.enable_atomic_relaxation = bool(cfg_mgr.enable_atomic_relaxation)
            if isinstance(cfg_mgr.photon_production_cut, (int, float)) and not isinstance(cfg_mgr.photon_production_cut, bool):
                self.orchestrator_session.photon_production_cut = float(cfg_mgr.photon_production_cut)
            if isinstance(cfg_mgr.electron_production_cut, (int, float)) and not isinstance(cfg_mgr.electron_production_cut, bool):
                self.orchestrator_session.electron_production_cut = float(cfg_mgr.electron_production_cut)
            self.orchestrator_session.seed = cfg_mgr.seed
        else:
            self.orchestrator_session.particles_number = int(self.sim_settings.particles_number)
            self.orchestrator_session.buffer_capacity = self.sim_settings.buffer_capacity
            self.orchestrator_session.min_energy = float(self.sim_settings.min_energy)

        self.orchestrator_session.pool_size = int(self.sim_settings.pool_size)

        if not self.orchestrator_session.jobs:
            self._on_generate_jobs()

        if not self.orchestrator_session.jobs:
            QMessageBox.warning(self, "Предупреждение", "Нет задач для выполнения в текущей процедуре.")
            return

        try:
            self._sync_detector_metadata()
            self.orchestrator_session.start()
            self._update_action_states(running=True, paused=False)
            self.lbl_status.setText(f"Моделирование запущено ({len(self.orchestrator_session.jobs)} задач)")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            QMessageBox.critical(self, "Ошибка запуска", f"Не удалось запустить моделирование:\n{exc}")

    def _on_clear_accumulation(self) -> None:
        """
        Полный сброс всех накопленных данных моделирования:
        проекции детектора, спектра, 3D-карты дозы и треков.
        """
        if self.orchestrator_session is not None:
            self.orchestrator_session.clear_accumulation()
        if self.viewport_controller is not None:
            self.viewport_controller.clear_dose_volume()
            self.viewport_controller.clear_tracks()
        self.lbl_status.setText("Накопление данных сброшено")

    def _on_pause_all(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.pause_all()
            self._update_action_states(running=True, paused=True)
            self.lbl_status.setText("Моделирование: весь пул приостановлен")

    def _on_resume_all(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.resume_all()
            self._update_action_states(running=True, paused=False)
            self.lbl_status.setText("Моделирование: весь пул возобновлен")

    def _on_pause_focused(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.pause_focused_worker()
            self._update_action_states(running=True, paused=False, focused_paused=True)
            self.lbl_status.setText("Моделирование: сфокусированный воркер приостановлен")

    def _on_resume_focused(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.resume_focused_worker()
            self._update_action_states(running=True, paused=False, focused_paused=False)
            self.lbl_status.setText("Моделирование: сфокусированный воркер возобновлен")

    def _on_stop_simulation(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.stop()
        self._update_action_states(running=False, paused=False)
        self.lbl_status.setText("Моделирование: остановлено")

    def _on_step_simulation(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.step_focused_worker()

    def _on_session_paused(self) -> None:
        self._update_action_states(running=True, paused=True)
        for row_idx in range(len(self.orchestrator_session.jobs)):
            self.jobs_widget.on_job_paused(row_idx, is_local=False)

    def _on_session_resumed(self) -> None:
        self._update_action_states(running=True, paused=False)
        for row_idx in range(len(self.orchestrator_session.jobs)):
            self.jobs_widget.on_job_resumed(row_idx)

    def _on_simulation_stopped(self) -> None:
        self.lbl_status.setText("Моделирование: остановлено")
        self._update_action_states(running=False, paused=False)

    def _on_simulation_finished(self) -> None:
        self.lbl_status.setText("Моделирование: завершено")
        self._update_action_states(running=False, paused=False)

    def _on_dose_volume_received(self, dose_data: np.ndarray) -> None:
        """
        Прием и отображение очередного снимка 3D-карты дозы.
        Использует геометрические параметры активной сессии или последние сохраненные
        параметры сетки, гарантируя неизменность origin и матриц при остановке симуляции.
        """
        if self.viewport_controller is not None:
            self.viewport_controller.on_dose_volume_received(dose_data, session=self.session)

    def _update_action_states(self, running: bool, paused: bool, focused_paused: bool = False) -> None:
        pool_size = self.orchestrator_session.pool_size
        self.act_run.setEnabled(not running)
        self.act_pause.setEnabled(running and not paused)
        self.act_resume.setEnabled(running and paused)
        # Блокировка паузы фокуса при pool_size == 1
        can_pause_focused = running and not paused and not focused_paused and pool_size > 1
        can_resume_focused = running and not paused and focused_paused and pool_size > 1
        self.act_pause_focused.setEnabled(can_pause_focused)
        self.act_resume_focused.setEnabled(can_resume_focused)
        self.act_stop.setEnabled(running)
        self.act_step.setEnabled(running and (paused or focused_paused or pool_size == 1))
        self.spin_preview_view.setEnabled(not running)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        QTimer.singleShot(50, self.viewport.reset_camera)

    def _on_new_scene(self) -> None:
        root = CompositeNode(name="WorldScene")
        if self.viewport_controller is not None:
            self.viewport_controller.disconnect_all_nodes()
            self.viewport_controller.clear_dose_volume()
            self.viewport_controller.clear_tracks()
        self.scene_vm.load_scene(root)
        if self.data_manager_vm is not None:
            for handler in list(self.data_manager_vm.handlers):
                self.data_manager_vm.remove_handler(handler)
            self.data_handler_list.rebuild_list()
        self._sync_detector_metadata()
        self.viewport.clear_actors()
        self.current_config = None
        self.current_config_path = None
        self.lbl_status.setText("Создана новая пустая сцена")

    def _on_open_yaml(self, filename: Optional[Any] = None) -> None:
        if not filename or isinstance(filename, bool):
            filename, _ = QFileDialog.getOpenFileName(self, "Открыть конфигурацию", "", "YAML files (*.yaml *.yml)")
        if filename:
            try:
                filepath = Path(filename)
                raw_dict = load_raw_config(filepath)
                all_extracted_bindings = ConfigTemplateBinder.extract_bindings(raw_dict)
                scene_bindings = {
                    node_key: node_bindings
                    for node_key, node_bindings in all_extracted_bindings.items()
                    if node_key not in ("simulation_manager", "data_manager")
                }
                cfg = load_simulation_config(filepath, resolve_protocol=True)
                builder = SceneBuilder(base_dir=filepath.parent)
                root_node = builder.build_scene(cfg.scene)
                if self.viewport_controller is not None:
                    self.viewport_controller.disconnect_all_nodes()
                    self.viewport_controller.clear_dose_volume()
                self.scene_vm.load_scene(
                    root_node,
                    distribution_registry=builder.distribution_registry,
                    slots_registry=builder.slots_registry,
                    bindings_registry=scene_bindings,
                    base_dir=filepath.parent,
                )
                self.scene_vm.apply_simulation_config(cfg)
                self.current_config = cfg
                self.current_config_path = str(filepath)

                # Восстановление протокола/процедуры
                if cfg.protocol is not None:
                    proc_vm = procedure_from_config(cfg.protocol)
                    for node_vm in self.scene_vm.all_nodes():
                        if node_vm.bindings:
                            proc_vm.bindings_registry[node_vm.core_node] = dict(node_vm.bindings)
                    if "simulation_manager" in all_extracted_bindings:
                        for prop_name, var_name in all_extracted_bindings["simulation_manager"].items():
                            proc_vm.bind_property("simulation_manager", prop_name, var_name)
                    if "data_manager" in all_extracted_bindings:
                        for prop_name, var_name in all_extracted_bindings["data_manager"].items():
                            if prop_name == "filename":
                                proc_vm.bind_property("data_manager", prop_name, var_name)
                    self.procedure_selector.set_procedure(proc_vm)

                # Восстановление обработчиков данных
                if cfg.data_manager is not None:
                    self.data_manager_vm.load_from_config(cfg.data_manager)
                    raw_dm = raw_dict.get("data_manager")
                    if isinstance(raw_dm, dict) and isinstance(raw_dm.get("filename"), str):
                        self.data_manager_vm.filename = raw_dm["filename"]
                    self.property_inspector.data_manager_section.update_ui_from_vm()
                    self.data_handler_list.rebuild_list()

                # Восстановление параметров менеджера симуляции и пула
                if cfg.simulation_manager is not None:
                    raw_particles = cfg.simulation_manager.particles_number
                    if isinstance(raw_particles, (int, float)) and not isinstance(raw_particles, bool):
                        parsed_particles = int(raw_particles)
                        self.sim_settings.particles_number = parsed_particles
                        self.orchestrator_session.particles_number = parsed_particles

                    # Ёмкость каналов телеметрии: None — авто (= particles_number), иначе не менее particles_number
                    raw_buffer_capacity = cfg.simulation_manager.buffer_capacity
                    if raw_buffer_capacity is None:
                        parsed_buffer_capacity = None
                    elif isinstance(raw_buffer_capacity, (int, float)) and not isinstance(raw_buffer_capacity, bool):
                        parsed_buffer_capacity = max(int(raw_buffer_capacity), self.sim_settings.particles_number)
                    else:
                        raise ValueError(
                            f"Ёмкость буфера телеметрии должна быть числом, получено: {raw_buffer_capacity!r}"
                        )
                    self.sim_settings.buffer_capacity = parsed_buffer_capacity
                    self.orchestrator_session.buffer_capacity = parsed_buffer_capacity

                    if not isinstance(self.procedure_vm, StepAndShootProcedureViewModel) or cfg.protocol is None:
                        raw_start_time = cfg.simulation_manager.start_time
                        if isinstance(raw_start_time, (int, float)) and not isinstance(raw_start_time, bool):
                            self.orchestrator_session.start_time = float(raw_start_time) / float(units.s)
                        raw_stop_time = cfg.simulation_manager.stop_time
                        if isinstance(raw_stop_time, (int, float)) and not isinstance(raw_stop_time, bool):
                            self.orchestrator_session.stop_time = float(raw_stop_time) / float(units.s)
                        self.property_inspector.procedure_section.update_ui_from_vm()

                    raw_min_energy = cfg.simulation_manager.min_energy
                    if isinstance(raw_min_energy, (int, float)) and not isinstance(raw_min_energy, bool):
                        scaled_min_energy = float(raw_min_energy) / float(units.keV)
                        self.sim_settings.min_energy = scaled_min_energy
                        self.orchestrator_session.min_energy = scaled_min_energy

                if cfg.pool_size is not None:
                    self.sim_settings.pool_size = int(cfg.pool_size)
                    self.orchestrator_session.pool_size = int(cfg.pool_size)
                    self.jobs_widget.spin_pool_size.setValue(int(cfg.pool_size))

                self._sync_detector_metadata()
                self._on_generate_jobs()
                self.viewport.reset_camera()
                self.lbl_status.setText(f"Загружена сцена и конфигурация из {filepath.name}")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить YAML:\n{exc}")

    def _on_save_yaml(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить конфигурацию", "simulation_config.yaml", "YAML files (*.yaml *.yml)")
        if filename and self.scene_vm.root_vm is not None:
            try:
                data_mgr = self.data_manager_vm.to_config() if self.data_manager_vm is not None else None
                prev_sim_mgr = (
                    self.current_config.simulation_manager
                    if self.current_config is not None
                    else None
                )
                if self.procedure_vm is not None:
                    for time_prop in ("start_time", "stop_time"):
                        bound_time_var = self.procedure_vm.bindings.get(time_prop)
                        if bound_time_var:
                            self.procedure_vm.bind_property("simulation_manager", time_prop, bound_time_var)

                start_time_cfg = self.orchestrator_session._get_start_time_cfg()
                has_explicit_start_binding = bool(
                    self.procedure_vm is not None
                    and (
                        self.procedure_vm.bindings.get("start_time")
                        or "start_time" in self.procedure_vm.get_property_bindings("simulation_manager")
                    )
                )
                if (
                    not has_explicit_start_binding
                    and self.orchestrator_session.start_time == 0.0
                    and prev_sim_mgr is not None
                    and isinstance(prev_sim_mgr.start_time, (int, float))
                    and float(prev_sim_mgr.start_time) != 0.0
                ):
                    start_time_cfg = prev_sim_mgr.start_time
                stop_time_cfg = self.orchestrator_session._get_stop_time_cfg()
                sim_mgr = SimulationManagerConfig(
                    start_time=start_time_cfg,
                    particles_number=int(self.sim_settings.particles_number),
                    batch_capacity=prev_sim_mgr.batch_capacity if prev_sim_mgr is not None else None,
                    buffer_capacity=self.sim_settings.buffer_capacity,
                    min_energy=float(self.sim_settings.min_energy) * units.keV,
                    stop_time=stop_time_cfg,
                    enable_atomic_relaxation=(
                        prev_sim_mgr.enable_atomic_relaxation if prev_sim_mgr is not None else False
                    ),
                    photon_production_cut=(
                        prev_sim_mgr.photon_production_cut if prev_sim_mgr is not None else 1.0 * units.keV
                    ),
                    electron_production_cut=(
                        prev_sim_mgr.electron_production_cut if prev_sim_mgr is not None else float("inf")
                    ),
                    seed=prev_sim_mgr.seed if prev_sim_mgr is not None else None,
                )
                logging_cfg = self.current_config.logging if self.current_config is not None else None
                protocol_cfg = self.procedure_vm.to_config() if self.procedure_vm is not None else None
                active_bindings = (
                    self.procedure_vm.bindings_registry
                    if (self.procedure_vm and self.procedure_vm.bindings_registry)
                    else self.scene_vm.bindings_registry
                )
                if active_bindings:
                    ConfigTemplateBinder.export_templated_yaml(
                        self.scene_vm.root_vm.core_node,
                        filename,
                        bindings_registry=active_bindings,
                        simulation_manager_cfg=sim_mgr,
                        data_manager_cfg=data_mgr,
                        protocol=protocol_cfg,
                        pool_size=int(self.sim_settings.pool_size),
                        distribution_registry=self.scene_vm.distribution_registry,
                        slots_registry=self.scene_vm.slots_registry,
                        logging_cfg=logging_cfg,
                    )
                else:
                    SceneExporter.export_to_yaml(
                        self.scene_vm.root_vm.core_node,
                        filename,
                        simulation_manager_cfg=sim_mgr,
                        data_manager_cfg=data_mgr,
                        protocol=protocol_cfg,
                        pool_size=int(self.sim_settings.pool_size),
                        distribution_registry=self.scene_vm.distribution_registry,
                        slots_registry=self.scene_vm.slots_registry,
                        logging_cfg=logging_cfg,
                    )
                QMessageBox.information(self, "Успех", f"Конфигурация сохранена в:\n{filename}")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить YAML:\n{exc}")

    def _on_open_simulation_settings(self) -> None:
        """
        Открывает диалоговое окно общих параметров расчета симуляции.
        """
        var_provider = self.procedure_vm.get_variable_names if self.procedure_vm is not None else None
        current_bindings = self.procedure_vm.get_property_bindings("simulation_manager") if self.procedure_vm is not None else {}
        dialog = SimulationSettingsDialog(
            self.sim_settings,
            parent=self,
            variable_provider=var_provider,
            bindings=current_bindings,
        )
        if dialog.exec():
            new_settings = dialog.get_settings()
            self.sim_settings.update(new_settings)
            if self.data_manager_vm is not None:
                for handler_vm in self.data_manager_vm.handlers:
                    if isinstance(handler_vm, DirectStreamHandlerViewModel):
                        handler_vm.show_escaped_tracks = bool(new_settings.show_escaped_tracks)
                self.property_inspector.data_handler_section.update_ui_from_vm()
            # Применение параметров к вычислительной сессии
            self.orchestrator_session.particles_number = new_settings.particles_number
            self.orchestrator_session.buffer_capacity = new_settings.buffer_capacity
            self.orchestrator_session.min_energy = new_settings.min_energy
            self.orchestrator_session.pool_size = new_settings.pool_size
            self.jobs_widget.spin_pool_size.setValue(new_settings.pool_size)

            # Актуализация current_config при его наличии
            if self.current_config is not None and self.current_config.simulation_manager is not None:
                self.current_config.simulation_manager.particles_number = new_settings.particles_number
                self.current_config.simulation_manager.buffer_capacity = new_settings.buffer_capacity
                self.current_config.simulation_manager.min_energy = new_settings.min_energy * units.keV
            if self.current_config is not None:
                self.current_config.pool_size = new_settings.pool_size

            # Сохранение привязок параметров симуляции в процедуре
            new_bindings = dialog.get_bindings()
            if self.procedure_vm is not None:
                for prop_name in ("particles_number", "min_energy"):
                    if prop_name in new_bindings:
                        self.procedure_vm.bind_property("simulation_manager", prop_name, new_bindings[prop_name])
                    else:
                        self.procedure_vm.unbind_property("simulation_manager", prop_name)

            # Передача параметров снаппинга в 3D Gizmo
            if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
                self.viewport_controller.transform_gizmo.grid_snap_step = new_settings.grid_snap_step
                self.viewport_controller.transform_gizmo.angle_snap_step = new_settings.angle_snap_step
                self.viewport_controller.transform_gizmo.scale_snap_step = new_settings.scale_snap_step

            # Передача параметров рентгеновской визуализации во вьюпорт
            if self.viewport_controller is not None:
                self.viewport_controller.set_xray_parameters(
                    energy=new_settings.xray_energy,
                    pseudo_xray_mode=new_settings.pseudo_xray_mode,
                )

            self._on_generate_jobs()
            self.lbl_status.setText("Параметры симуляции обновлены")

    def _set_gizmo_mode(self, mode: GizmoMode) -> None:
        """Переключение активного режима манипулятора Gizmo."""
        if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
            self.viewport_controller.transform_gizmo.mode = mode
            self.viewport.render()

    def _toggle_gizmo_space(self) -> None:
        """Переключение между мировой и локальной системами координат манипулятора."""
        if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
            gizmo = self.viewport_controller.transform_gizmo
            gizmo.space = GizmoSpace.LOCAL if gizmo.space == GizmoSpace.WORLD else GizmoSpace.WORLD
            self.viewport.render()

    def _on_grid_snap_changed(self, text: str) -> None:
        """Обновление шага координатной сетки привязки манипулятора."""
        if self.viewport_controller is None or self.viewport_controller.transform_gizmo is None:
            return
        mapping = {
            "Выкл": 0.0,
            "1 мм": 1.0,
            "5 мм": 5.0,
            "10 мм": 10.0,
            "25 мм": 25.0,
            "50 мм": 50.0,
            "100 мм": 100.0,
        }
        step_val = mapping.get(text, 10.0)
        self.viewport_controller.transform_gizmo.grid_snap_step = step_val

    def _on_angle_snap_changed(self, text: str) -> None:
        """Обновление шага угловой сетки привязки манипулятора."""
        if self.viewport_controller is None or self.viewport_controller.transform_gizmo is None:
            return
        mapping = {
            "Выкл": 0.0,
            "5°": 5.0,
            "10°": 10.0,
            "15°": 15.0,
            "30°": 30.0,
            "45°": 45.0,
            "90°": 90.0,
        }
        angle_val = mapping.get(text, 5.0)
        self.viewport_controller.transform_gizmo.angle_snap_step = angle_val

    def _on_gizmo_mode_changed(self, mode: GizmoMode) -> None:
        """Синхронизация кнопок панели инструментов при смене режима манипулятора."""
        if mode == GizmoMode.TRANSLATE:
            self.act_gizmo_translate.setChecked(True)
        elif mode == GizmoMode.ROTATE:
            self.act_gizmo_rotate.setChecked(True)
        elif mode == GizmoMode.SCALE:
            self.act_gizmo_scale.setChecked(True)
        self.lbl_status.setText(f"Манипулятор Gizmo: режим {mode.value.upper()}")

    def _on_gizmo_space_changed(self, space: GizmoSpace) -> None:
        """Синхронизация кнопки системы координат при смене пространства манипулятора."""
        is_local = (space == GizmoSpace.LOCAL)
        self.act_gizmo_space.setChecked(is_local)
        self.act_gizmo_space.setText("🌐 Локальные (Q)" if is_local else "🌐 Мировые (Q)")
        self.lbl_status.setText(f"Манипулятор Gizmo: система координат {space.value.upper()}")

    def eventFilter(self, watched: Any, event: Any) -> bool:
        """
        Глобальный перехват горячих клавиш W/E/R/Q манипулятора Gizmo во всех дочерних виджетах
        (включая дерево сцены и вьюпорт), кроме текстовых полей ввода.
        """
        if event is not None and event.type() == QEvent.Type.KeyPress:
            focus_widget = QApplication.focusWidget()
            if isinstance(watched, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)) or \
               isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
                return super().eventFilter(watched, event)

            key = event.key()
            text_char = event.text().strip().upper()
            if key == Qt.Key.Key_W or text_char in ('W', 'Ц'):
                if self.act_gizmo_translate.isVisible() and self.act_gizmo_translate.isEnabled():
                    self._set_gizmo_mode(GizmoMode.TRANSLATE)
                    self.act_gizmo_translate.setChecked(True)
                return True
            elif key == Qt.Key.Key_E or text_char in ('E', 'У'):
                if self.act_gizmo_rotate.isVisible() and self.act_gizmo_rotate.isEnabled():
                    self._set_gizmo_mode(GizmoMode.ROTATE)
                    self.act_gizmo_rotate.setChecked(True)
                return True
            elif key == Qt.Key.Key_R or text_char in ('R', 'К'):
                if self.act_gizmo_scale.isVisible() and self.act_gizmo_scale.isEnabled():
                    self._set_gizmo_mode(GizmoMode.SCALE)
                    self.act_gizmo_scale.setChecked(True)
                return True
            elif key == Qt.Key.Key_Q or text_char in ('Q', 'Й'):
                if self.act_gizmo_space.isVisible() and self.act_gizmo_space.isEnabled():
                    self._toggle_gizmo_space()
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event: Any) -> None:
        """Очистка глобальных фильтров событий, сессий, таймеров и 3D-ресурсов при закрытии главного окна."""
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.removeEventFilter(self)
        if self.orchestrator_session is not None:
            self.orchestrator_session.close()
        if self.results_viewer is not None:
            self.results_viewer.close()
        if self.viewport_controller is not None:
            self.viewport_controller.close()
        self.viewport.close()
        super().closeEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        """
        Глобальная обработка горячих клавиш W/E/R/Q для переключения режимов 3D-манипулятора.
        Поддерживает как сканкоды физических клавиш Qt.Key, так и текстовые символы (латиница/кириллица).
        """
        if self.viewport_controller is not None and self.viewport_controller.transform_gizmo is not None:
            gizmo = self.viewport_controller.transform_gizmo
            key_code = event.key()
            if key_code == Qt.Key.Key_W:
                if self.act_gizmo_translate.isVisible() and self.act_gizmo_translate.isEnabled():
                    self._set_gizmo_mode(GizmoMode.TRANSLATE)
                    self.act_gizmo_translate.setChecked(True)
                event.accept()
                return
            elif key_code == Qt.Key.Key_E:
                if self.act_gizmo_rotate.isVisible() and self.act_gizmo_rotate.isEnabled():
                    self._set_gizmo_mode(GizmoMode.ROTATE)
                    self.act_gizmo_rotate.setChecked(True)
                event.accept()
                return
            elif key_code == Qt.Key.Key_R:
                if self.act_gizmo_scale.isVisible() and self.act_gizmo_scale.isEnabled():
                    self._set_gizmo_mode(GizmoMode.SCALE)
                    self.act_gizmo_scale.setChecked(True)
                event.accept()
                return
            elif key_code == Qt.Key.Key_Q:
                if self.act_gizmo_space.isVisible() and self.act_gizmo_space.isEnabled():
                    self._toggle_gizmo_space()
                event.accept()
                return

            key_text = event.text()
            if key_text and gizmo.handle_key_action(key_text):
                self.viewport.render()
                event.accept()
                return
        super().keyPressEvent(event)


__all__ = ["MainWindow"]

