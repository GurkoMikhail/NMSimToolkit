import unittest
from typing import Any, Dict, List, Optional
import numpy as np
from PySide6.QtWidgets import QApplication

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from settings.database_setting import material_database
from core.scene.nodes import CompositeNode
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    SpectProcedureViewModel,
    PetProcedureViewModel,
    CustomSweepProcedureViewModel,
    create_procedure_viewmodel,
)
from gui.viewmodels.data_handler_viewmodel import (
    DataManagerViewModel,
    DirectStreamHandlerViewModel,
    SensitiveVolumeHandlerViewModel,
    HistoryAssemblerHandlerViewModel,
    DoseMapHandlerViewModel,
)
from gui.views.procedure_selector_widget import ProcedureSelectorWidget
from gui.views.data_handler_list_widget import DataHandlerListWidget
from gui.views.simulation_jobs_widget import SimulationJobsWidget
from gui.views.property_inspector import PropertyInspector
from gui.views.main_window import MainWindow
from gui.controllers.orchestrator_session import OrchestratorSession


# Гарантируем наличие экземпляра QApplication для виджетов Qt
app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestGuiModularization(unittest.TestCase):
    """
    Комплексные тесты для новой модульной архитектуры графического интерфейса:
    - Процедуры моделирования (Spect, Pet, CustomSweep)
    - Обработчики данных (Data Handlers)
    - Параллельный планировщик задач (Jobs Widget)
    - Инспектор свойств (Property Inspector)
    - Интеграция сессии оркестратора (Orchestrator Session) и главного окна (MainWindow)
    """

    def setUp(self) -> None:
        self.root = CompositeNode(name="WorldScene")
        self.scene_vm = SceneViewModel()
        self.scene_vm.load_scene(self.root)

    def test_spect_procedure_viewmodel_and_kinematics(self) -> None:
        """
        Тестирование модели процедуры ОФЭКТ и кинематической синхронизации
        камер GammaCameraViewModel со свойствами процедуры.
        """
        # Создаем детекторные головки
        box_geom = Box(100.0, 80.0, 40.0)
        mat = material_database['Pb']
        cam1 = Volume(geometry=box_geom, material=mat, name="Camera_Head_1")
        cam2 = Volume(geometry=box_geom, material=mat, name="Camera_Head_2")
        self.root.add_child(cam1)
        self.root.add_child(cam2)

        cam_vm1 = GammaCameraViewModel(cam1)
        cam_vm2 = GammaCameraViewModel(cam2)

        proc = SpectProcedureViewModel()
        proc.steps = 32
        proc.start_angle = 15.0
        proc.angular_range = 180.0
        proc.radius = 280.0
        proc.head_angles = [0.0, 90.0]

        self.assertEqual(proc.procedure_type, "SPECT")
        self.assertEqual(proc.steps, 32)
        self.assertEqual(proc.start_angle, 15.0)
        self.assertEqual(proc.angular_range, 180.0)
        self.assertEqual(proc.radius, 280.0)
        self.assertEqual(proc.head_angles, [0.0, 90.0])

        # Проверяем синхронизацию камер со свойствами процедуры
        proc.sync_cameras([cam_vm1, cam_vm2])
        self.assertAlmostEqual(cam_vm1.orbit_radius, 280.0)
        self.assertAlmostEqual(cam_vm2.orbit_radius, 280.0)
        self.assertAlmostEqual(cam_vm1.orbit_angle, 15.0)
        self.assertAlmostEqual(cam_vm2.orbit_angle, 105.0)

    def test_procedure_factory_and_types(self) -> None:
        """
        Тестирование фабрики процедур.
        """
        spect = create_procedure_viewmodel("SPECT")
        self.assertIsInstance(spect, SpectProcedureViewModel)

        pet = create_procedure_viewmodel("PET")
        self.assertIsInstance(pet, PetProcedureViewModel)
        self.assertEqual(pet.procedure_type, "PET")

        custom = create_procedure_viewmodel("Custom Sweep")
        self.assertIsInstance(custom, CustomSweepProcedureViewModel)
        self.assertEqual(custom.procedure_type, "CustomSweep")

        with self.assertRaises(ValueError):
            create_procedure_viewmodel("UnknownProcedure")

    def test_data_manager_viewmodel(self) -> None:
        """
        Тестирование управления обработчиками данных и конвертации в конфиги ядра.
        """
        dm = DataManagerViewModel()
        self.assertEqual(dm.filename, "simulation_results.h5")
        self.assertEqual(len(dm.handlers), 4)  # По умолчанию: стрим, история, сенситив, доза

        # Добавляем еще один обработчик
        custom_handler = DirectStreamHandlerViewModel(enabled=True)
        dm.add_handler(custom_handler)
        self.assertEqual(len(dm.handlers), 5)

        # Конвертация в core-конфигурации
        core_configs = dm.to_core_handlers()
        self.assertEqual(len(core_configs), 5)

        # Удаление
        dm.remove_handler(custom_handler)
        self.assertEqual(len(dm.handlers), 4)

    def test_procedure_selector_widget(self) -> None:
        """
        Тестирование виджета выбора процедур.
        """
        widget = ProcedureSelectorWidget(scene_vm=self.scene_vm)
        received_procs = []
        widget.procedure_changed.connect(lambda p: received_procs.append(p))

        # Переключаем комбобокс на ПЭТ
        widget.combo_type.setCurrentIndex(1)
        self.assertEqual(len(received_procs), 1)
        self.assertIsInstance(received_procs[0], PetProcedureViewModel)
        self.assertIsInstance(widget.active_procedure, PetProcedureViewModel)

    def test_data_handler_list_widget(self) -> None:
        """
        Тестирование виджета списка обработчиков данных.
        """
        dm = DataManagerViewModel()
        widget = DataHandlerListWidget(data_manager_vm=dm)

        selected_items = []
        widget.handler_selected.connect(lambda h: selected_items.append(h))

        # Выбираем первый обработчик
        widget.list_widget.setCurrentRow(0)
        self.assertEqual(len(selected_items), 1)
        self.assertIsInstance(selected_items[0], DirectStreamHandlerViewModel)

    def test_simulation_jobs_widget(self) -> None:
        """
        Тестирование виджета отображения и выбора подзадач симуляции.
        """
        widget = SimulationJobsWidget()

        jobs = [
            {'view_index': 0, 'angle': 0.0},
            {'view_index': 1, 'angle': 180.0},
        ]
        widget.set_jobs(jobs)
        self.assertEqual(widget.table.rowCount(), 2)

        # Проверка фокуса
        focused = []
        widget.focused_job_changed.connect(lambda fid: focused.append(fid))

        # Переключаем фокус
        widget._on_focus_toggled(1, True)
        self.assertEqual(widget.focused_index, 1)
        self.assertIn(1, focused)

        # Обновление прогресса
        widget.on_job_progress(0, 0.5, 500, 100.0)
        p_bar = widget._progress_bars[0]
        self.assertEqual(p_bar.value(), 50)

        # Обновление состояния
        widget.on_job_started(0)
        item_state = widget.table.item(0, 3)
        self.assertEqual(item_state.text(), "Выполняется")

    def test_orchestrator_session_task_generation(self) -> None:
        """
        Тестирование сессии оркестратора: генерация задач и управление фокусом.
        """
        proc = SpectProcedureViewModel(steps=4, gamma_cameras=1)

        dm = DataManagerViewModel()
        session = OrchestratorSession(
            scene_vm=self.scene_vm,
            procedure_vm=proc,
            data_manager_vm=dm,
        )

        jobs = session.generate_jobs()
        self.assertEqual(len(jobs), 4)

        # Установка фокуса
        session.set_focused_job(2)
        self.assertEqual(session.focused_job_index, 2)

    def test_property_inspector_procedural_and_data_sections(self) -> None:
        """
        Тестирование инспектора свойств при выборе процедур и обработчиков.
        """
        inspector = PropertyInspector()

        # 1. Инспекция процедуры ОФЭКТ
        proc = SpectProcedureViewModel()
        inspector.set_target_viewmodel(proc)
        self.assertFalse(inspector.procedure_group.isHidden())
        self.assertTrue(inspector.data_handler_group.isHidden())

        # 2. Инспекция обработчика прямого стрима
        stream_handler = DirectStreamHandlerViewModel()
        inspector.set_target_viewmodel(stream_handler)
        self.assertTrue(inspector.procedure_group.isHidden())
        self.assertFalse(inspector.data_handler_group.isHidden())

        # 3. Инспекция диспетчера данных
        dm = DataManagerViewModel()
        inspector.set_target_viewmodel(dm)
        self.assertFalse(inspector.data_manager_group.isHidden())

    def test_main_window_dock_integration(self) -> None:
        """
        Тестирование инициализации главного окна с шестью док-панелями.
        """
        window = MainWindow(scene_vm=self.scene_vm)
        try:
            # Проверяем наличие всех доков
            self.assertIsNotNone(window.dock_tree)
            self.assertIsNotNone(window.dock_procedures)
            self.assertIsNotNone(window.dock_handlers)
            self.assertIsNotNone(window.dock_inspector)
            self.assertIsNotNone(window.dock_jobs)
            self.assertIsNotNone(window.dock_results)

            # Проверяем генерацию задач через окно
            window._on_generate_jobs()
            self.assertGreater(len(window.orchestrator_session.jobs), 0)
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
