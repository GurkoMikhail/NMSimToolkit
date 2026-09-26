import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyvista as pv
import hepunits as units
from PySide6.QtCore import Qt, QSize, QTimer, QObject
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QLabel, QSpinBox
)

from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.voxel_volume_vm import VoxelVolumeViewModel
from gui.viewmodels.nodes.source_vm import SourceViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.pet_scanner_vm import PetScannerViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.property_inspector import PropertyInspector
from gui.views.results_viewer import ResultsViewer
from gui.views.procedure_selector_widget import ProcedureSelectorWidget
from gui.views.data_handler_list_widget import DataHandlerListWidget
from gui.views.simulation_jobs_widget import SimulationJobsWidget
from gui.views.simulation_settings_dialog import SimulationSettingsDialog
from gui.viewmodels.procedure_viewmodel import BaseProcedureViewModel, SpectProcedureViewModel, procedure_from_config
from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel, DirectStreamHandlerViewModel
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.controllers.orchestrator_session import OrchestratorSession
from gui.controllers.viewport_controller import SceneViewportController
from gui.models.gui_settings import GuiSimulationSettings
from core.scene.nodes import CompositeNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.config.exporter import SceneExporter
from core.config.yaml_loader import load_simulation_config
from core.config.builder import SceneBuilder
from core.config.models import DataManagerConfig, HistoryAssemblerHandlerConfig
from core.config.orchestrator import Orchestrator
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

    def __init__(self, scene_vm: Optional[SceneViewModel] = None, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NMSimToolkit - Интерактивное 3D моделирование ядерной медицины")
        self.resize(1400, 900)

        self.viewport_controller: Optional[SceneViewportController] = None

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

        self._init_components()
        self._init_docks()
        self._init_menus()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()
        self._update_action_states(running=False, paused=False)

    @property
    def scene_vm(self) -> SceneViewModel:
        return self._scene_vm

    @scene_vm.setter
    def scene_vm(self, vm: SceneViewModel) -> None:
        self._scene_vm = vm
        if self.viewport_controller is not None:
            self.viewport_controller.set_scene_viewmodel(vm)

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
        self.procedure_vm = SpectProcedureViewModel()
        self.data_manager_vm = DataManagerViewModel()

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
        self.dock_tree.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_tree)

        # 2. Левый док: Процедуры
        self.dock_procedures = self.procedure_selector
        self.dock_procedures.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_procedures)

        # 3. Левый док: Обработчики данных
        self.dock_handlers = self.data_handler_list
        self.dock_handlers.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_handlers)

        # Табификация доков слева
        self.tabifyDockWidget(self.dock_tree, self.dock_procedures)
        self.tabifyDockWidget(self.dock_procedures, self.dock_handlers)

        # 4. Правый док: Инспектор свойств
        self.dock_inspector = QDockWidget("Свойства объекта", self)
        self.dock_inspector.setWidget(self.property_inspector)
        self.dock_inspector.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_inspector)

        # 5. Правый док: Задачи симуляции (Jobs)
        self.dock_jobs = self.jobs_widget
        self.dock_jobs.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_jobs)
        self.tabifyDockWidget(self.dock_inspector, self.dock_jobs)

        # 6. Нижний док: Результаты и телеметрия
        self.dock_results = QDockWidget("Телеметрия и результаты", self)
        self.dock_results.setWidget(self.results_viewer)
        self.dock_results.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_results)

    def _init_menus(self) -> None:
        menubar = self.menuBar()

        # Меню Файл
        file_menu = menubar.addMenu("&Файл")

        act_new = QAction("&Новая сцена", self)
        act_new.triggered.connect(self._on_new_scene)
        file_menu.addAction(act_new)

        act_open = QAction("&Открыть конфигурацию (YAML)...", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._on_open_yaml)
        file_menu.addAction(act_open)

        act_save = QAction("&Сохранить конфигурацию (YAML)...", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._on_save_yaml)
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        act_exit = QAction("&Выход", self)
        act_exit.setShortcut(QKeySequence.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Меню Моделирование
        sim_menu = menubar.addMenu("&Моделирование")
        self.act_run = QAction("▶ &Запуск", self)
        self.act_pause = QAction("⏸ &Пауза", self)
        self.act_resume = QAction("⏯ &Возобновить", self)
        self.act_stop = QAction("⏹ &Остановить", self)
        self.act_step = QAction("⏭ &Шаг вперед", self)

        sim_menu.addAction(self.act_run)
        sim_menu.addAction(self.act_pause)
        sim_menu.addAction(self.act_resume)
        sim_menu.addAction(self.act_stop)
        sim_menu.addAction(self.act_step)
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
        self.act_toggle_tracks.toggled.connect(self.viewport_controller.track_renderer.set_visible)
        view_menu.addAction(self.act_toggle_tracks)

        self.act_toggle_dose = QAction("Отображать карту дозы", self)
        self.act_toggle_dose.setCheckable(True)
        self.act_toggle_dose.setChecked(True)
        self.act_toggle_dose.toggled.connect(self.viewport_controller.dose_renderer.set_visible)
        view_menu.addAction(self.act_toggle_dose)

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
        toolbar.addAction(self.act_stop)
        toolbar.addAction(self.act_step)
        toolbar.addSeparator()
        toolbar.addAction(self.act_settings)
        toolbar.addSeparator()

        act_reset = QAction("⟲ Камера", self)
        act_reset.triggered.connect(self.viewport.reset_camera)
        toolbar.addAction(act_reset)
        toolbar.addSeparator()

        self.lbl_preview_view = QLabel("Ракурс:")
        toolbar.addWidget(self.lbl_preview_view)
        self.spin_preview_view = QSpinBox()
        self.spin_preview_view.setRange(1, max(1, self.sim_settings.views_number))
        self.spin_preview_view.setValue(1)
        self.spin_preview_view.setToolTip("Предварительный просмотр ориентации детекторных головок ОФЭКТ для выбранного ракурса")
        self.spin_preview_view.valueChanged.connect(self._on_preview_view_changed)
        toolbar.addWidget(self.spin_preview_view)

    def _init_statusbar(self) -> None:
        status = self.statusBar()
        self.lbl_status = QLabel("Статус: Готов (IDLE)")
        status.addWidget(self.lbl_status)

    def _connect_signals(self) -> None:
        # Связь выбора узла в дереве с инспектором и 3D-манипулятором
        self.scene_vm.node_selected.connect(self.property_inspector.set_target_viewmodel)
        self.scene_vm.node_selected.connect(self._on_node_selected)
        self.property_inspector.dose_voxel_size_changed.connect(lambda vs: self.sim_settings.update({'dose_voxel_size': vs}))

        # Связь выбора процедуры и обработчиков с инспектором
        self.procedure_selector.procedure_changed.connect(self._on_procedure_changed)
        self.procedure_selector.procedure_selected.connect(self.property_inspector.set_target_viewmodel)
        self.data_handler_list.handler_selected.connect(self.property_inspector.set_target_viewmodel)
        self.data_handler_list.data_manager_selected.connect(self.property_inspector.set_target_viewmodel)

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
        self.orchestrator_session.session_error.connect(lambda err: QMessageBox.critical(self, "Ошибка", f"Сбой расчета:\n{err}"))

        # Потоковая визуализация от сфокусированного воркера
        self.orchestrator_session.tracks_received.connect(self.viewport_controller.track_renderer.add_tracks_batch)
        self.orchestrator_session.projection_received.connect(self.results_viewer.set_projection_data)
        self.orchestrator_session.projection_stack_updated.connect(self._on_projection_stack_updated)
        self.orchestrator_session.spectrum_received.connect(self.results_viewer.set_spectrum_data)
        self.orchestrator_session.dose_volume_received.connect(self._on_dose_volume_received)
        self.orchestrator_session.stats_updated.connect(
            lambda cnt, cps: self.lbl_status.setText(f"Моделирование: {cnt:,} отсчетов ({cps:.1f} CPS)")
        )

        # Раздельные подписки для оптимизированной инкрементальной синхронизации
        self.scene_vm.scene_loaded.connect(lambda vm: self.viewport_controller.sync_viewport_scene())
        self.scene_vm.node_added.connect(self._on_node_added)
        self.scene_vm.node_removed.connect(self._on_node_removed)

        # Автоматическая генерация задач при изменении сцены и параметров процедуры
        self.scene_vm.node_added.connect(lambda n: self._on_generate_jobs())
        self.scene_vm.node_removed.connect(lambda n: self._on_generate_jobs())
        self.procedure_vm.changed.connect(self._on_generate_jobs)

        # Управляющие действия моделирования
        self.act_run.triggered.connect(self._on_start_simulation)
        self.act_pause.triggered.connect(self._on_pause_simulation)
        self.act_resume.triggered.connect(self._on_resume_simulation)
        self.act_stop.triggered.connect(self._on_stop_simulation)
        self.act_step.triggered.connect(self._on_step_simulation)

        # Сброс накопления из панели результатов
        self.results_viewer.accumulation_cleared.connect(self._on_clear_accumulation)

        # Начальная отрисовка сцены во вьюпорте и генерация задач
        self.viewport_controller.sync_viewport_scene()
        self._on_generate_jobs()

    def _on_node_transform_changed(self, node_vm: NodeViewModel) -> None:
        """Инкрементальное обновление матрицы трансформации актора без пересоздания меша."""
        self.viewport_controller.on_node_transform_changed(node_vm)

    def _on_node_property_changed(self, node_vm: NodeViewModel, prop_name: str, value: Any) -> None:
        """Инкрементальное обновление параметров актора при смене геометрии или свойств."""
        self.viewport_controller.on_node_property_changed(node_vm, prop_name, value)

    def _on_node_added(self, node_vm: NodeViewModel) -> None:
        """Точечное добавление нового актора в сцену."""
        self.viewport_controller.on_node_added(node_vm)

    def _on_node_removed(self, node_vm: NodeViewModel) -> None:
        """Точечное удаление актора из сцены."""
        self.viewport_controller.on_node_removed(node_vm)

    def _on_node_selected(self, vm: Optional[NodeViewModel]) -> None:
        if self.dock_inspector.isHidden():
            self.dock_inspector.show()
        self.dock_inspector.raise_()
        self.viewport_controller.on_node_selected(vm)

    def _apply_job_angles_to_viewport(self, context: Dict[str, Any]) -> None:
        """Применяет углы из контекста задачи к гамма-камерам во вьюпорте."""
        self.viewport_controller.apply_job_angles_to_viewport(context, self.procedure_vm)

    def _on_preview_view_changed(self, view_number_1based: int) -> None:
        """Предварительный кинематический поворот детекторов на выбранный ракурс ОФЭКТ."""
        base_angle = self.viewport_controller.preview_view(view_number_1based, self.procedure_vm)
        views_total = self.procedure_vm.views if isinstance(self.procedure_vm, SpectProcedureViewModel) else 1
        self.lbl_status.setText(f"Предпросмотр ОФЭКТ: Ракурс {view_number_1based}/{views_total} (угол {base_angle:.1f}°)")


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

    def _on_procedure_changed(self, proc_vm: BaseProcedureViewModel) -> None:
        """
        Смена активной процедуры сканирования.
        """
        self.procedure_vm = proc_vm
        self.orchestrator_session.procedure_vm = proc_vm
        self.property_inspector.set_target_viewmodel(proc_vm)
        if isinstance(proc_vm, SpectProcedureViewModel):
            self.spin_preview_view.setRange(1, max(1, proc_vm.views))
            def _on_proc_param_changed(param: str, val: Any) -> None:
                if param in ("views", "views_number"):
                    self.spin_preview_view.setRange(1, max(1, int(val)))
            proc_vm.parameter_changed.connect(_on_proc_param_changed)
        proc_vm.changed.connect(self._on_generate_jobs)
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
        Переключение фокуса визуализации на указанную задачу.
        """
        self.orchestrator_session.set_focused_job(index)
        self.lbl_status.setText(f"Фокус визуализации: Задача #{index + 1}")
        jobs = self.orchestrator_session.jobs
        if 0 <= index < len(jobs):
            self._apply_job_angles_to_viewport(jobs[index])
            self.spin_preview_view.blockSignals(True)
            self.spin_preview_view.setValue(index + 1)
            self.spin_preview_view.blockSignals(False)

    def _on_pool_size_changed(self, pool_size: int) -> None:
        """
        Изменение размера пула рабочих процессов.
        """
        self.orchestrator_session.pool_size = max(1, int(pool_size))

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
            try:
                self.orchestrator_session.particles_number = int(cfg_mgr.particles_number)
            except (TypeError, ValueError):
                pass
            try:
                val_stop = float(cfg_mgr.stop_time)
                self.orchestrator_session.stop_time = val_stop / float(units.s)
            except (TypeError, ValueError):
                pass
            try:
                val_en = float(cfg_mgr.min_energy)
                self.orchestrator_session.min_energy = val_en / float(units.keV)
            except (TypeError, ValueError):
                pass

        if not self.orchestrator_session.jobs:
            self._on_generate_jobs()

        if not self.orchestrator_session.jobs:
            QMessageBox.warning(self, "Предупреждение", "Нет задач для выполнения в текущей процедуре.")
            return

        try:
            self.orchestrator_session.start()
            self._update_action_states(running=True, paused=False)
            self.lbl_status.setText(f"Моделирование запущено ({len(self.orchestrator_session.jobs)} задач)")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка запуска", f"Не удалось запустить моделирование:\n{e}")

    def _on_clear_accumulation(self) -> None:
        """
        Полный сброс всех накопленных данных моделирования:
        проекции детектора, спектра, 3D-карты дозы и треков.
        """
        if self.orchestrator_session is not None:
            self.orchestrator_session.clear_accumulation()
        self.viewport_controller.clear_dose_volume()
        self.viewport_controller.clear_tracks()
        self.lbl_status.setText("Накопление данных сброшено")

    def _on_pause_simulation(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.pause()
            self._update_action_states(running=True, paused=True)
            self.lbl_status.setText("Моделирование: приостановлено")

    def _on_resume_simulation(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.resume()
            self._update_action_states(running=True, paused=False)
            self.lbl_status.setText("Моделирование: возобновлено")

    def _on_stop_simulation(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.stop()
        self._update_action_states(running=False, paused=False)
        self.lbl_status.setText("Моделирование: остановлено")

    def _on_step_simulation(self) -> None:
        if self.orchestrator_session.is_running:
            self.orchestrator_session.step_once()

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
        self.viewport_controller.on_dose_volume_received(dose_data, session=self.session)

    def _update_action_states(self, running: bool, paused: bool) -> None:
        self.act_run.setEnabled(not running)
        self.act_pause.setEnabled(running and not paused)
        self.act_resume.setEnabled(running and paused)
        self.act_stop.setEnabled(running)
        self.act_step.setEnabled(not running or paused)
        self.spin_preview_view.setEnabled(not running)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        QTimer.singleShot(50, self.viewport.reset_camera)

    def closeEvent(self, event: Any) -> None:
        if self.orchestrator_session is not None:
            self.orchestrator_session.close()
        if self.viewport_controller is not None:
            self.viewport_controller.close()
        self.viewport.close()
        super().closeEvent(event)

    def _on_new_scene(self) -> None:
        root = CompositeNode(name="WorldScene")
        if self.viewport_controller is not None:
            self.viewport_controller.disconnect_all_nodes()
            self.viewport_controller.clear_dose_volume()
            self.viewport_controller.clear_tracks()
        self.scene_vm.load_scene(root)
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
                cfg = load_simulation_config(filepath, resolve_protocol=True)
                builder = SceneBuilder(base_dir=filepath.parent)
                root_node = builder.build_scene(cfg.scene)
                if self.viewport_controller is not None:
                    self.viewport_controller.disconnect_all_nodes()
                    self.viewport_controller.clear_dose_volume()
                self.scene_vm.load_scene(root_node, distribution_registry=builder.distribution_registry)
                self.scene_vm.apply_simulation_config(cfg)
                self.current_config = cfg
                self.current_config_path = str(filepath)

                # Восстановление протокола/процедуры
                if cfg.protocol is not None:
                    proc_vm = procedure_from_config(cfg.protocol)
                    self.procedure_selector.set_procedure(proc_vm)
                    self._on_procedure_changed(proc_vm)

                # Восстановление обработчиков данных
                if cfg.data_manager is not None:
                    self.data_manager_vm.load_from_config(cfg.data_manager)
                    self.data_handler_list.rebuild_list()

                # Восстановление параметров менеджера симуляции и пула
                if cfg.simulation_manager is not None:
                    self.sim_settings.particles_number = int(cfg.simulation_manager.particles_number)
                    self.orchestrator_session.particles_number = int(cfg.simulation_manager.particles_number)
                    try:
                        st_sec = float(cfg.simulation_manager.stop_time) / float(units.s)
                        self.sim_settings.stop_time = st_sec
                        self.orchestrator_session.stop_time = st_sec
                    except (TypeError, ValueError):
                        pass
                    try:
                        me_kev = float(cfg.simulation_manager.min_energy) / float(units.keV)
                        self.sim_settings.min_energy = me_kev
                        self.orchestrator_session.min_energy = me_kev
                    except (TypeError, ValueError):
                        pass

                if cfg.pool_size is not None:
                    self.sim_settings.pool_size = int(cfg.pool_size)
                    self.orchestrator_session.pool_size = int(cfg.pool_size)
                    self.jobs_widget.spin_pool_size.setValue(int(cfg.pool_size))

                self._on_generate_jobs()
                self.viewport.reset_camera()
                self.lbl_status.setText(f"Загружена сцена и конфигурация из {filepath.name}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить YAML:\n{e}")

    def _on_save_yaml(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить конфигурацию", "simulation_config.yaml", "YAML files (*.yaml *.yml)")
        if filename and self.scene_vm.root_vm is not None:
            try:
                data_mgr = None
                if self.data_manager_vm is not None and self.data_manager_vm.handlers:
                    handlers_cfg = self.data_manager_vm.to_core_handlers()
                    data_mgr = DataManagerConfig(
                        filename=self.data_manager_vm.filename,
                        handlers=handlers_cfg
                    )
                SceneExporter.export_to_yaml(self.scene_vm.root_vm.core_node, filename, data_manager_cfg=data_mgr)
                QMessageBox.information(self, "Успех", f"Конфигурация сохранена в:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить YAML:\n{e}")

    def _on_open_simulation_settings(self) -> None:
        """
        Открывает диалоговое окно общих параметров расчета симуляции.
        """
        dialog = SimulationSettingsDialog(self.sim_settings, parent=self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            self.sim_settings.update(new_settings)
            # Применение параметров к вычислительной сессии
            self.orchestrator_session.particles_number = new_settings.particles_number
            self.orchestrator_session.stop_time = new_settings.stop_time
            self.orchestrator_session.min_energy = new_settings.min_energy
            self.orchestrator_session.pool_size = new_settings.pool_size
            self.jobs_widget.spin_pool_size.setValue(new_settings.pool_size)
            self.data_manager_vm.buffer_capacity = new_settings.buffer_capacity

            # Синхронизация с обработчиком DirectStreamHandlerViewModel при наличии
            for handler_vm in self.data_manager_vm.handlers:
                if isinstance(handler_vm, DirectStreamHandlerViewModel):
                    handler_vm.show_escaped_tracks = new_settings.show_escaped_tracks

            self._on_generate_jobs()
            self.lbl_status.setText("Параметры симуляции обновлены")
