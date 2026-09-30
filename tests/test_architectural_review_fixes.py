import unittest
import numpy as np
from pydantic import ValidationError
from PySide6.QtWidgets import QApplication

from core.scene.nodes import CompositeNode
from core.geometry.pet_scanners import PetScanner
from core.scene.gamma_camera_node import GammaCameraNode
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from gui.models.gui_settings import GuiSimulationSettings
from gui.views.simulation_settings_dialog import SimulationSettingsDialog
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.pet_scanner_vm import PetScannerViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.procedure_viewmodel import SpectProcedureViewModel
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.controllers.viewport_controller import SceneViewportController, IDoseGeometryProvider

app = QApplication.instance() or QApplication([])


class TestArchitecturalReviewFixes(unittest.TestCase):
    """
    Глубокие тесты для проверки устранения замечаний архитектурного ревью:
    1. Строгая Pydantic-валидация и метод update() в GuiSimulationSettings.
    2. Совместимость SimulationSettingsDialog с GuiSimulationSettings.
    3. Корректное наследование PetScannerViewModel (NodeViewModel, а не VolumeViewModel),
       Single Source of Truth для полей (diameter, axial_length, num_sectors)
       и отсутствие падений при синхронизации с SceneViewportController.
    4. Строгий протокол IDoseGeometryProvider без getattr в SceneViewportController.
    5. Граничный случай: динамическое добавление и удаление нескольких многоголовочных
       систем ОФЭКТ в дереве сцены на лету во время активного предпросмотра ракурсов кинематики.
    """

    def setUp(self) -> None:
        self.root = CompositeNode(name="WorldScene")
        self.scene_vm = SceneViewModel(self.root)
        self.viewport = VTKViewport()
        self.viewport_ctrl = SceneViewportController(self.viewport, self.scene_vm)

    def tearDown(self) -> None:
        self.viewport_ctrl.close()
        self.viewport.close()

    # -------------------------------------------------------------------------
    # 1. GuiSimulationSettings Pydantic модель
    # -------------------------------------------------------------------------
    def test_gui_simulation_settings_validation_and_update(self) -> None:
        """Проверка строгой валидации диапазонов Pydantic и метода update()."""
        settings = GuiSimulationSettings()
        self.assertEqual(settings.particles_number, 5000)
        self.assertEqual(settings.min_energy, 1.0)
        self.assertEqual(settings.grid_snap_step, 10.0)
        self.assertEqual(settings.angle_snap_step, 15.0)
        self.assertEqual(settings.scale_snap_step, 1.0)

        # Успешное обновление
        settings.update({
            'particles_number': 10000,
            'pool_size': 4,
            'grid_snap_step': 20.0,
            'angle_snap_step': 15.0,
            'scale_snap_step': 0.2,
        })
        self.assertEqual(settings.particles_number, 10000)
        self.assertEqual(settings.pool_size, 4)
        self.assertEqual(settings.grid_snap_step, 20.0)
        self.assertEqual(settings.angle_snap_step, 15.0)
        self.assertEqual(settings.scale_snap_step, 0.2)

        # Ошибки валидации инвариантов
        with self.assertRaises(ValidationError):
            settings.particles_number = 0  # ge=1

        with self.assertRaises(ValidationError):
            settings.min_energy = -1.0  # ge=0.0

        with self.assertRaises(ValidationError):
            settings.pool_size = 0  # ge=1

        with self.assertRaises(ValidationError):
            settings.grid_snap_step = -5.0  # ge=0.0

        # Словарный интерфейс (маппинг)
        self.assertEqual(settings['grid_snap_step'], 20.0)
        self.assertEqual(settings.get('angle_snap_step'), 15.0)
        self.assertEqual(settings.get('unknown_key', 'def'), 'def')
        self.assertIn('particles_number', settings.to_dict())

        # Строгая валидация update() (LBYL / fail-fast)
        # Проверка, что удаленные поля (views_number, stop_time, dose_voxel_size) теперь вызывают KeyError
        with self.assertRaises(KeyError):
            settings.update({'views_number': 64})

        with self.assertRaises(KeyError):
            settings.update({'stop_time': 10.0})

        with self.assertRaises(KeyError):
            settings.update({'dose_voxel_size': 2.5})

        with self.assertRaises(KeyError):
            settings.update({'completely_unknown_key': 123})

        with self.assertRaises(TypeError):
            settings.update("invalid_argument_type")

    def test_simulation_settings_dialog_with_gui_settings(self) -> None:
        """Проверка передачи GuiSimulationSettings в SimulationSettingsDialog без TypeError."""
        settings = GuiSimulationSettings(particles_number=12345, grid_snap_step=25.0)
        dialog = SimulationSettingsDialog(settings)
        self.assertEqual(dialog.spin_particles.value(), 12345)
        self.assertAlmostEqual(dialog.spin_grid_snap.value(), 25.0)

        ret = dialog.get_settings()
        self.assertIsInstance(ret, GuiSimulationSettings)
        self.assertEqual(ret.particles_number, 12345)
        self.assertAlmostEqual(ret.grid_snap_step, 25.0)

        # Проверка отказа от legacy dict
        with self.assertRaises(TypeError):
            SimulationSettingsDialog({"particles_number": 5000})  # type: ignore

    # -------------------------------------------------------------------------
    # 2. PetScannerViewModel и SceneViewportController
    # -------------------------------------------------------------------------
    def test_pet_scanner_viewmodel_hierarchy_and_viewport_sync(self) -> None:
        """
        Проверка наследования PetScannerViewModel от NodeViewModel (не VolumeViewModel),
        Single Source of Truth для геометрических параметров и интеграции со вьюпортом.
        """
        pet_core = PetScanner(
            name="PET_WholeBody",
            diameter=650.0,
            axial_length=220.0,
            num_sectors=28
        )
        vm = create_node_viewmodel(pet_core)

        # Строгая иерархия
        self.assertIsInstance(vm, PetScannerViewModel)
        self.assertIsInstance(vm, NodeViewModel)
        self.assertNotIsInstance(vm, VolumeViewModel)

        # Чтение свойств
        self.assertEqual(vm.diameter, 650.0)
        self.assertEqual(vm.axial_length, 220.0)
        self.assertEqual(vm.num_sectors, 28)
        np.testing.assert_array_equal(vm.size, [650.0, 650.0, 220.0])

        # Синхронизация в расчетное ядро (Single Source of Truth)
        vm.diameter = 700.0
        self.assertEqual(pet_core.diameter, 700.0)
        vm.axial_length = 250.0
        self.assertEqual(pet_core.axial_length, 250.0)
        vm.num_sectors = 32
        self.assertEqual(pet_core.num_sectors, 32)

        # Синхронизация с 3D-вьюпортом (не должно быть AttributeError: 'PetScanner' has no attribute 'size')
        self.viewport_ctrl.add_or_update_node_actor(vm)
        self.assertAlmostEqual(self.viewport_ctrl.pet_manipulator.diameter, 700.0)
        self.assertAlmostEqual(self.viewport_ctrl.pet_manipulator.axial_length, 250.0)
        self.assertEqual(self.viewport_ctrl.pet_manipulator.num_sectors, 32)

        # Выбор узла активирует манипулятор ПЭТ и скрывает ОФЭКТ
        self.viewport_ctrl.on_node_selected(vm)
        self.viewport_ctrl.on_node_selected(None)

    # -------------------------------------------------------------------------
    # 3. IDoseGeometryProvider без duck typing
    # -------------------------------------------------------------------------
    def test_dose_geometry_provider_contract(self) -> None:
        """Проверка контракта IDoseGeometryProvider при приеме дозы во вьюпорт."""
        class MockSession(IDoseGeometryProvider):
            @property
            def dose_origin(self):
                return (-50.0, -50.0, -50.0)
            @property
            def dose_voxel_size(self):
                return 4.0
            @property
            def dose_transform_matrix(self):
                return np.eye(4)

        session = MockSession()
        self.assertIsInstance(session, IDoseGeometryProvider)

        dose_dummy = np.zeros((10, 10, 10), dtype=np.float32)
        self.viewport_ctrl.on_dose_volume_received(dose_dummy, session=session)

        self.assertEqual(self.viewport_ctrl.active_dose_origin, (-50.0, -50.0, -50.0))
        self.assertEqual(self.viewport_ctrl.active_dose_voxel_size, 4.0)
        self.assertIsNotNone(self.viewport_ctrl.active_dose_transform_matrix)

    # -------------------------------------------------------------------------
    # 4. Граничный случай: Динамическое добавление/удаление ОФЭКТ-камер при preview
    # -------------------------------------------------------------------------
    def test_dynamic_spect_cameras_during_preview(self) -> None:
        """
        Проверка добавления и удаления нескольких многоголовочных ОФЭКТ систем
        в дереве сцены на лету во время активного предпросмотра ракурсов кинематики.
        """
        mat = Material(name="Lead")

        def _make_cam(name: str) -> GammaCameraNode:
            cam_node, _ = create_default_gamma_camera(name=name)
            return cam_node

        cam1 = _make_cam("Camera_Head_1")
        cam2 = _make_cam("Camera_Head_2")
        self.root.add_child(cam1)
        self.root.add_child(cam2)
        self.scene_vm.load_scene(self.root)
        self.viewport_ctrl.sync_viewport_scene()

        proc = SpectProcedureViewModel()
        proc.steps = 8
        proc.start_angle = 0.0
        proc.end_angle = 360.0
        proc.gamma_cameras = 2
        proc.head_angles = [0.0, 90.0]
        proc.radius = 250.0

        # Предпросмотр ракурса 2 (для 2 головок из 16 ракурсов: 8 позиций гантри, шаг 45.0°)
        base_ang = self.viewport_ctrl.preview_view(2, proc)
        self.assertAlmostEqual(base_ang, 45.0)

        # Динамическое добавление еще двух камер на лету
        cam3 = _make_cam("Camera_Head_3")
        cam4 = _make_cam("Camera_Head_4")
        self.root.add_child(cam3)
        self.root.add_child(cam4)
        self.scene_vm.load_scene(self.root)
        self.viewport_ctrl.sync_viewport_scene()

        proc.steps = 4
        proc.gamma_cameras = 4
        proc.head_angles = [0.0, 90.0, 180.0, 270.0]

        # Предпросмотр ракурса 4 с 4 камерами (4 позиции гантри, шаг 90°, ракурс 4 -> индекс 3 -> 270°)
        base_ang4 = self.viewport_ctrl.preview_view(4, proc)
        self.assertAlmostEqual(base_ang4, 270.0)

        # Динамическое удаление камер
        self.root.remove_child(cam3)
        self.root.remove_child(cam4)
        self.scene_vm.load_scene(self.root)
        self.viewport_ctrl.sync_viewport_scene()

        # Предпросмотр ракурса 1
        base_ang_back = self.viewport_ctrl.preview_view(1, proc)
        self.assertAlmostEqual(base_ang_back, 0.0)
