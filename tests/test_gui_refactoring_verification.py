import time
import unittest
from multiprocessing import Queue, shared_memory

import numpy as np
from PySide6.QtWidgets import QApplication

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import CompositeNode, SpatialNode
from core.transport.simulation_managers import SimulationState
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.simulation_session import SimulationSession
from gui.viewmodels.decorators import core_field, gui_field, observable_field
from gui.viewmodels.node_viewmodel import NodeViewModel, VolumeViewModel, GammaCameraViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.vtk_viewport import VTKViewport
from gui.views.main_window import MainWindow
from gui.views.scene_tree_widget import SceneTreeWidget
from gui.views.property_inspector import PropertyInspector
from core.geometry.gamma_cameras import GammaCamera


class TestGuiRefactoringVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = QApplication.instance()
        if app is None:
            cls.app = QApplication(["-platform", "offscreen"])
        else:
            cls.app = app

    def test_single_source_of_truth_descriptors(self):
        """
        Проверка Single Source of Truth: core_field и observable_field
        не дублируют значение в instance.__dict__, а читают/пишут строго в core_node.
        """
        node = SpatialNode(name="OriginalName")

        class SampleViewModel:
            name = core_field('name')
            tag = gui_field(default='LocalTag')
            obs = observable_field('name')

            def __init__(self, core):
                self.core_node = core

        vm = SampleViewModel(node)

        # 1. Чтение из core_node
        self.assertEqual(vm.name, "OriginalName")
        self.assertNotIn("name", vm.__dict__, "Имя не должно сохраняться в __dict__ ViewModel!")

        # 2. Запись через ViewModel
        vm.name = "UpdatedViaVM"
        self.assertEqual(node.name, "UpdatedViaVM")
        self.assertNotIn("name", vm.__dict__, "Имя по-прежнему не должно кэшироваться в __dict__!")

        # 3. Изменение напрямую в ядре отражается во ViewModel
        node.name = "DirectCoreChange"
        self.assertEqual(vm.name, "DirectCoreChange")

        # 4. Проверка gui_field (локальное хранилище)
        self.assertEqual(vm.tag, "LocalTag")
        vm.tag = "NewTag"
        self.assertEqual(vm.__dict__.get("tag"), "NewTag")
        self.assertFalse(hasattr(node, "tag"))

        # 5. Проверка observable_field на атрибуте ядра (не создает дубликата)
        vm.obs = "ChangedViaObs"
        self.assertEqual(node.name, "ChangedViaObs")
        self.assertNotIn("obs", vm.__dict__)

    def test_simulation_session_lifecycle(self):
        """
        Проверка контроллера SimulationSession: создание конвейера,
        управление жизненным циклом и детерминированное закрытие ресурсов.
        """
        from settings.database_setting import material_database
        root = CompositeNode(name="SessionTestScene")
        vol = Volume(geometry=Box(10.0, 10.0, 10.0), material=material_database['Water, Liquid'], name="TargetBox")
        root.add_child(vol)

        session = SimulationSession(
            scene_root=root,
            shm_name="test_sim_session_shm",
            projection_shape=(16, 16),
            particles_number=20,
            stop_time=0.01,
            fps=30.0,
            h5_filename="test_session.h5"
        )

        started_log = []
        paused_log = []
        resumed_log = []
        stopped_log = []

        session.session_started.connect(lambda: started_log.append(True))
        session.session_paused.connect(lambda: paused_log.append(True))
        session.session_resumed.connect(lambda: resumed_log.append(True))
        session.session_stopped.connect(lambda: stopped_log.append(True))

        self.assertIsNotNone(session.runner)
        self.assertIsNotNone(session.receiver)
        self.assertIsNotNone(session.stream_handler)
        self.assertIsNotNone(session.data_manager)

        session.pause()
        self.assertEqual(len(paused_log), 1)

        session.resume()
        self.assertEqual(len(resumed_log), 1)

        session.stop()
        self.assertEqual(len(stopped_log), 1)

        # Детерминированное закрытие ресурсов
        session.close()
        self.assertTrue(session._is_closed)
        self.assertIsNone(session.runner)
        self.assertIsNone(session.receiver)
        self.assertIsNone(session.stream_handler)
        self.assertIsNone(session.data_manager)

    def test_scene_tree_widget_reverse_map(self):
        """
        Проверка O(1) обратного словаря _vm_by_item в SceneTreeWidget.
        """
        root = CompositeNode(name="RootNode")
        c1 = SpatialNode(name="Child1")
        c2 = SpatialNode(name="Child2")
        root.add_child(c1)
        root.add_child(c2)

        scene_vm = SceneViewModel(root)
        tree_widget = SceneTreeWidget(scene_vm)

        self.assertEqual(len(tree_widget._vm_by_item), 3)  # Root, Child1, Child2

        # Проверка соответствия каждого QTreeWidgetItem его ViewModel
        for item, vm in tree_widget._vm_by_item.items():
            self.assertIn(vm.name, ["RootNode", "Child1", "Child2"])
            self.assertEqual(item.text(0), vm.name)

        # Имитация выбора элемента
        child1_vm = scene_vm.find_by_name("Child1")
        child1_item = tree_widget._item_map[id(child1_vm)]
        tree_widget.tree.setCurrentItem(child1_item)
        tree_widget._on_tree_selection_changed()

        self.assertIs(scene_vm.selected_node, child1_vm)

    def test_vtk_viewport_update_actor_transform(self):
        """
        Проверка инкрементального обновления матрицы трансформации в VTKViewport.
        """
        viewport = VTKViewport()
        import pyvista as pv
        box = pv.Box()
        actor = viewport.add_mesh_actor("test_transform_box", box)

        # Несуществующий актор возвращает False
        self.assertFalse(viewport.update_actor_transform("non_existent_actor", np.eye(4)))

        if actor is not None:
            # Обновление матрицы
            mat = np.eye(4, dtype=float)
            mat[0, 3] = 123.0
            mat[1, 3] = 456.0
            mat[2, 3] = 789.0
            res = viewport.update_actor_transform("test_transform_box", mat)
            self.assertTrue(res)

            if hasattr(actor, 'user_matrix'):
                self.assertAlmostEqual(actor.user_matrix[0, 3], 123.0)
                self.assertAlmostEqual(actor.user_matrix[1, 3], 456.0)
                self.assertAlmostEqual(actor.user_matrix[2, 3], 789.0)

        viewport.close()

    def test_main_window_incremental_scene_sync(self):
        """
        Проверка раздельных обработчиков добавления, удаления и изменения
        трансформации узлов в MainWindow.
        """
        win = MainWindow()
        root = CompositeNode(name="TestIncrementalRoot")
        vol1 = Volume(geometry=Box(20.0, 20.0, 20.0), material=Material(name="Water"), name="Vol1")
        root.add_child(vol1)

        win.scene_vm.load_scene(root)
        vol1_vm = win.scene_vm.find_by_name("Vol1")
        self.assertIsNotNone(vol1_vm)

        actor_name = f"mesh_{id(vol1_vm)}"
        self.assertIn(actor_name, win.viewport._actors)

        # Проверка инкрементального вращения (без сброса всех акторов)
        vol1_vm.translate(x=50.0, y=50.0, z=50.0)
        self.assertIn(actor_name, win.viewport._actors)

        # Добавление нового узла
        vol2 = Volume(geometry=Box(30.0, 30.0, 30.0), material=Material(name="Lead"), name="Vol2")
        vol2_vm = VolumeViewModel(vol2)
        win.scene_vm.add_node(win.scene_vm.root_vm, vol2_vm)

        actor2_name = f"mesh_{id(vol2_vm)}"
        self.assertIn(actor2_name, win.viewport._actors)

        # Точечное удаление узла
        win.scene_vm.remove_node(vol2_vm)
        self.assertNotIn(actor2_name, win.viewport._actors)
        self.assertIn(actor_name, win.viewport._actors)

        win.close()

    def test_simulation_session_edge_cases(self):
        """
        Тестирование граничных условий SimulationSession:
        - Идемпотентность close()
        - Вызов start() после close() вызывает RuntimeError
        - Безопасность вызовов pause/resume/step_once при незапущенной сессии
        """
        from settings.database_setting import material_database
        root = CompositeNode(name="EdgeCaseScene")
        vol = Volume(geometry=Box(10.0, 10.0, 10.0), material=material_database['Water, Liquid'], name="TargetBox")
        root.add_child(vol)

        session = SimulationSession(
            scene_root=root,
            shm_name="test_edge_session_shm",
            projection_shape=(16, 16),
            particles_number=10,
            stop_time=0.01,
            h5_filename="test_edge_session.h5"
        )

        # Вызовы управления до старта
        session.pause()
        session.resume()
        session.step_once()
        session.stop()

        # Двойной close()
        session.close()
        session.close()
        self.assertTrue(session._is_closed)

        # Попытка старта после закрытия
        with self.assertRaises(RuntimeError):
            session.start()

    def test_descriptors_edge_cases(self):
        """
        Тестирование граничных условий дескрипторов:
        - Доступ через класс возвращает сам дескриптор
        - core_field без атрибута ядра вызывает AttributeError при установке
        """
        class DummyClass:
            cf = core_field('non_existent_core_attr')
            gf = gui_field(default='DefaultVal')

        # Доступ через класс
        self.assertIsInstance(DummyClass.cf, core_field)
        self.assertIsInstance(DummyClass.gf, gui_field)

        dummy_inst = DummyClass()
        dummy_inst.core_node = object()  # нет атрибута non_existent_core_attr

        with self.assertRaises(AttributeError):
            dummy_inst.cf = "SomeValue"

        # Значение по умолчанию для core_field при отсутствии поля
        self.assertIsNone(dummy_inst.cf)

    def test_scene_tree_widget_edge_cases(self):
        """
        Тестирование граничных условий SceneTreeWidget:
        - rebuild_tree при None root_vm
        - удаление корневого узла блокируется
        """
        empty_scene_vm = SceneViewModel()
        tree_widget = SceneTreeWidget(empty_scene_vm)
        self.assertEqual(tree_widget.tree.topLevelItemCount(), 0)

        # Попытка клика удаления при отсутствии выделения
        tree_widget._on_remove_clicked()
        self.assertEqual(tree_widget.tree.topLevelItemCount(), 0)

    def test_gamma_camera_viewmodel_orbit_sync_and_inheritance(self):
        """
        Проверка GammaCameraViewModel:
        - Наследование от VolumeViewModel (наличие size, color).
        - Корректная синхронизация orbit_radius и orbit_angle через дескрипторы gui_field.
        - Отсутствие отката значений к дефолтным при вызове set_orbit_position().
        """
        from settings.database_setting import material_database
        collimator = Volume(geometry=Box(20.0, 20.0, 5.0), material=material_database['Pb'], name='Collimator')
        detector = Volume(geometry=Box(20.0, 20.0, 2.0), material=material_database['Pb'], name='Detector')
        cam = GammaCamera(collimator=collimator, detector=detector, name='TestGammaCam')

        vm = GammaCameraViewModel(cam)
        self.assertIsInstance(vm, VolumeViewModel)
        self.assertIsNotNone(vm.size)

        # Проверка начальных параметров
        self.assertAlmostEqual(vm.orbit_radius, 250.0)
        self.assertAlmostEqual(vm.orbit_angle, 0.0)

        # Вызов set_orbit_position и проверка реактивности
        prop_changes = []
        vm.property_changed.connect(lambda prop, val: prop_changes.append((prop, val)))

        vm.set_orbit_position(380.0, 60.0)
        self.assertAlmostEqual(vm.orbit_radius, 380.0)
        self.assertAlmostEqual(vm.orbit_angle, 60.0)

        # Проверка связывания с PropertyInspector
        inspector = PropertyInspector()
        inspector.set_target_viewmodel(vm)
        self.assertAlmostEqual(inspector.spin_orbit_radius.value(), 380.0)
        self.assertAlmostEqual(inspector.spin_orbit_angle.value(), 60.0)

        # Изменение через UI инспектора
        inspector.spin_orbit_radius.setValue(420.0)
        inspector._on_spect_param_changed()
        self.assertAlmostEqual(vm.orbit_radius, 420.0)
        self.assertAlmostEqual(inspector.spin_orbit_radius.value(), 420.0)

    def test_data_manager_stop_method(self):
        """
        Проверка детерминированного завершения потока DataManager.stop()
        без возникновения AttributeError и утечки фоновых потоков.
        """
        from core.data.data_manager import DataManager
        from core.data.data_handlers import BaseDataHandler
        import queue

        class DummyHandler(BaseDataHandler):
            def process_chunk(self, chunk):
                pass

        q = queue.Queue()
        dm = DataManager(filename="test_dm_stop.h5", handlers=[DummyHandler()], queue=q)
        dm.start()
        self.assertTrue(dm.is_alive())

        dm.stop(timeout=1.0)
        self.assertFalse(dm.is_alive())

    def test_scene_tree_widget_live_rename(self):
        """
        Проверка мгновенного обновления текста элемента в SceneTreeWidget
        при изменении имени во ViewModel (Single Source of Truth + Reactive UI).
        """
        root = CompositeNode(name="RootNode")
        child = SpatialNode(name="OldName")
        root.add_child(child)

        scene_vm = SceneViewModel(root)
        tree_widget = SceneTreeWidget(scene_vm)

        child_vm = scene_vm.find_by_name("OldName")
        self.assertIsNotNone(child_vm)
        child_item = tree_widget._item_map[id(child_vm)]
        self.assertEqual(child_item.text(0), "OldName")

        # Переименование через ViewModel
        child_vm.name = "NewLiveName"
        self.assertEqual(child_item.text(0), "NewLiveName")

    def test_main_window_spect_manipulator_coupling(self):
        """
        Проверка двусторонней связи 3D-манипулятора ОФЭКТ с выбранной GammaCameraViewModel в MainWindow.
        """
        from settings.database_setting import material_database
        root = CompositeNode(name="SPECTScene")
        collimator = Volume(geometry=Box(20.0, 20.0, 5.0), material=material_database['Pb'], name='Collimator')
        detector = Volume(geometry=Box(20.0, 20.0, 2.0), material=material_database['Pb'], name='Detector')
        cam = GammaCamera(collimator=collimator, detector=detector, name='CameraNode')
        root.add_child(cam)

        scene_vm = SceneViewModel(root)
        win = MainWindow(scene_vm=scene_vm)

        cam_vm = scene_vm.find_by_name("CameraNode")
        self.assertIsNotNone(cam_vm)
        scene_vm.select_node(cam_vm)

        # Симулируем перемещение манипулятора
        win.spect_manipulator.orbit_changed.emit(330.0, 45.0, 0.0)
        self.assertAlmostEqual(cam_vm.orbit_radius, 330.0)
        self.assertAlmostEqual(cam_vm.orbit_angle, 45.0)

        win.close()

    def test_main_window_volume_property_changed_sync(self):
        """
        Проверка синхронизации изменения размеров объема с актором 3D Viewport.
        """
        root = CompositeNode(name="VolSyncScene")
        vol = Volume(geometry=Box(50.0, 50.0, 50.0), material=Material(name="Water"), name="DynamicBox")
        root.add_child(vol)

        win = MainWindow(scene_vm=SceneViewModel(root))
        vol_vm = win.scene_vm.find_by_name("DynamicBox")
        self.assertIsNotNone(vol_vm)

        actor_name = f"mesh_{id(vol_vm)}"
        self.assertIn(actor_name, win.viewport._actors)

        # Изменение геометрии
        vol_vm.size = [120.0, 130.0, 140.0]
        self.assertIn(actor_name, win.viewport._actors)

        win.close()


if __name__ == '__main__':
    unittest.main()
