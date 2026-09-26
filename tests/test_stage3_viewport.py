import os
import tempfile
import unittest
import numpy as np

from gui.viewport_3d.dicom_colormaps import (
    get_available_colormaps,
    get_colormap_lut,
    import_lut_file,
)
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.viewport_3d.pet_manipulator import PETManipulator
from gui.viewport_3d.track_renderer import TrackRenderer


class TestStage3Viewport(unittest.TestCase):
    def test_dicom_colormaps(self):
        """Проверка генерации DICOM-палитр ядерной медицины."""
        cmaps = get_available_colormaps()
        self.assertIn('Hot Iron', cmaps)
        self.assertIn('Rainbow', cmaps)
        self.assertIn('GE Color', cmaps)
        self.assertIn('PET 20 Step', cmaps)

        lut = get_colormap_lut('Hot Iron', num_colors=256)
        self.assertEqual(lut.shape, (256, 4))
        self.assertTrue(np.all(lut >= 0.0))
        self.assertTrue(np.all(lut <= 1.0))

    def test_lut_file_import(self):
        """Проверка импорта стороннего LUT-файла."""
        with tempfile.NamedTemporaryFile(suffix=".lut", delete=False, mode='w') as tmp:
            tmp.write("0.0 0.0 0.0\n0.5 0.5 0.5\n1.0 1.0 1.0\n")
            tmp_path = tmp.name

        try:
            data = import_lut_file(tmp_path)
            self.assertEqual(data.shape, (3, 4))
            self.assertAlmostEqual(data[0, 0], 0.0)
            self.assertAlmostEqual(data[2, 0], 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_spect_manipulator_kinematics(self):
        """Проверка кинематических ограничений манипулятора ОФЭКТ."""
        manipulator = SPECTManipulator(viewport=None, initial_radius=200.0, initial_angle=0.0)

        events = []
        manipulator.orbit_changed.connect(lambda r, a, z: events.append((r, a, z)))

        # Поворот на 90 градусов
        manipulator.set_orbit_parameters(radius=250.0, angle_deg=90.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], (250.0, 90.0, 0.0))

        # При угле 90 градусов X=0, Y=250
        x, y, z = manipulator.get_cartesian_position()
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 250.0, places=5)

        mat = manipulator.get_orientation_matrix()
        self.assertEqual(mat.shape, (4, 4))
        self.assertAlmostEqual(mat[1, 3], 250.0)

    def test_pet_manipulator_geometry(self):
        """Проверка манипулятора геометрии кольца ПЭТ."""
        pet = PETManipulator(viewport=None, diameter=600.0, axial_length=200.0)

        events = []
        pet.geometry_changed.connect(lambda d, l, s: events.append((d, l, s)))

        pet.set_parameters(diameter=700.0, axial_length=220.0, num_sectors=32)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], (700.0, 220.0, 32))
        self.assertEqual(pet.diameter, 700.0)

    def test_track_renderer_batch(self):
        """Проверка накопления треков в буфере TrackRenderer."""
        renderer = TrackRenderer(viewport=None, max_points=100)

        batch = {
            'pos_x': np.array([10.0, 20.0, 30.0]),
            'pos_y': np.array([0.0, 0.0, 0.0]),
            'pos_z': np.array([5.0, 5.0, 5.0]),
            'process_id': np.array([1, 1, 2]),
            'particle_id': np.array([1, 1, 2])
        }

        renderer.add_tracks_batch(batch)
        self.assertEqual(len(renderer._point_buffer), 3)

        renderer.clear()
        self.assertEqual(len(renderer._point_buffer), 0)

    def test_transform_gizmo_modes_and_spaces(self):
        """Проверка переключения режимов и систем координат в TransformGizmo."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoMode, GizmoSpace

        gizmo = TransformGizmo(viewport=None)
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)
        self.assertEqual(gizmo.space, GizmoSpace.WORLD)

        mode_events = []
        gizmo.mode_changed.connect(lambda m: mode_events.append(m))
        gizmo.mode = GizmoMode.ROTATE
        self.assertEqual(len(mode_events), 1)
        self.assertEqual(mode_events[0], GizmoMode.ROTATE)

        space_events = []
        gizmo.space_changed.connect(lambda s: space_events.append(s))
        gizmo.space = GizmoSpace.LOCAL
        self.assertEqual(len(space_events), 1)
        self.assertEqual(space_events[0], GizmoSpace.LOCAL)

    def test_transform_gizmo_snapping(self):
        """Проверка ступенчатого снаппинга координат, углов и масштаба (UE-Style)."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo

        gizmo = TransformGizmo(
            viewport=None,
            grid_snap_step=10.0,
            angle_snap_step=15.0,
            scale_snap_step=0.25
        )

        # 1. Линейный снаппинг
        self.assertAlmostEqual(gizmo.snap_coordinate(12.3), 10.0)
        self.assertAlmostEqual(gizmo.snap_coordinate(17.8), 20.0)
        # Shift отключает снаппинг
        self.assertAlmostEqual(gizmo.snap_coordinate(12.3, shift_modifier=True), 12.3)

        # 2. Угловой снаппинг
        self.assertAlmostEqual(gizmo.snap_angle(28.0), 30.0)
        self.assertAlmostEqual(gizmo.snap_angle(37.0), 30.0)
        self.assertAlmostEqual(gizmo.snap_angle(37.0, shift_modifier=True), 37.0)

        # 3. Масштабный снаппинг
        self.assertAlmostEqual(gizmo.snap_scale_factor(1.18), 1.25)
        self.assertAlmostEqual(gizmo.snap_scale_factor(1.05), 1.0)
        self.assertAlmostEqual(gizmo.snap_scale_factor(1.18, shift_modifier=True), 1.18)

    def test_transform_gizmo_apply_transformations(self):
        """Проверка применения трансформаций к целевому NodeViewModel."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoSpace
        from core.scene.nodes import SpatialNode
        from gui.viewmodels.nodes.base_node_vm import NodeViewModel

        core_node = SpatialNode(name="TestObject")
        vm = NodeViewModel(core_node)
        gizmo = TransformGizmo(viewport=None, target_node=vm, grid_snap_step=5.0)

        # 1. Перемещение с снаппингом
        gizmo.apply_translation(np.array([12.1, 7.6, -3.2]))
        # 12.1 -> 10.0, 7.6 -> 10.0, -3.2 -> -5.0
        pos = vm.local_matrix[0:3, 3]
        self.assertAlmostEqual(pos[0], 10.0)
        self.assertAlmostEqual(pos[1], 10.0)
        self.assertAlmostEqual(pos[2], -5.0)

        # 2. Вращение вокруг Z на 90 градусов
        gizmo.angle_snap_step = 45.0
        gizmo.apply_rotation('z', 88.0)  # snaps to 90.0
        # Базисный вектор X (столбец 0) должен повернуться в (0, 1, 0)
        rot_col0 = vm.local_matrix[0:3, 0]
        self.assertAlmostEqual(rot_col0[0], 0.0, places=5)
        self.assertAlmostEqual(rot_col0[1], 1.0, places=5)

        # 3. Масштабирование
        gizmo.scale_snap_step = 0.5
        gizmo.apply_scale(np.array([1.9, 1.9, 1.9]))  # snaps to 2.0
        norm_x = np.linalg.norm(vm.local_matrix[0:3, 0])
        self.assertAlmostEqual(norm_x, 2.0, places=5)

    def test_transform_gizmo_hotkeys(self):
        """Проверка горячих клавиш W/E/R/Q."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoMode, GizmoSpace

        gizmo = TransformGizmo(viewport=None)
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)

        self.assertTrue(gizmo.handle_key_action('E'))
        self.assertEqual(gizmo.mode, GizmoMode.ROTATE)

        self.assertTrue(gizmo.handle_key_action('R'))
        self.assertEqual(gizmo.mode, GizmoMode.SCALE)

        self.assertTrue(gizmo.handle_key_action('W'))
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)

        self.assertEqual(gizmo.space, GizmoSpace.WORLD)
        self.assertTrue(gizmo.handle_key_action('Q'))
        self.assertEqual(gizmo.space, GizmoSpace.LOCAL)
        self.assertTrue(gizmo.handle_key_action('Q'))
        self.assertEqual(gizmo.space, GizmoSpace.WORLD)

        # Проверка русской раскладки (кириллица: Ц, У, К, Й)
        self.assertTrue(gizmo.handle_key_action('У'))  # Русская 'e' -> Rotate
        self.assertEqual(gizmo.mode, GizmoMode.ROTATE)
        self.assertTrue(gizmo.handle_key_action('к'))  # Русская 'r' -> Scale
        self.assertEqual(gizmo.mode, GizmoMode.SCALE)
        self.assertTrue(gizmo.handle_key_action('ц'))  # Русская 'w' -> Translate
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)
        self.assertTrue(gizmo.handle_key_action('й'))  # Русская 'q' -> Local
        self.assertEqual(gizmo.space, GizmoSpace.LOCAL)
        self.assertTrue(gizmo.handle_key_action('Й'))  # Русская 'q' -> World
        self.assertEqual(gizmo.space, GizmoSpace.WORLD)

    def test_transform_gizmo_mouse_interaction_and_picking(self):
        """Проверка перехвата событий мыши VTK, сброса камеры (SetAbortFlag) и эмиссии сигналов."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoAxis, GizmoMode
        from core.scene.nodes import SpatialNode
        from gui.viewmodels.nodes.base_node_vm import NodeViewModel

        core_node = SpatialNode(name="InteractTarget")
        node_vm = NodeViewModel(core_node)

        class MockProperty:
            def __init__(self):
                self.color = (1.0, 0.0, 0.0)

            def SetColor(self, red: float, green: float, blue: float):
                self.color = (red, green, blue)

        class MockActor:
            def __init__(self):
                self.prop = MockProperty()

            def GetProperty(self):
                return self.prop

        class MockInteractorObj:
            def __init__(self):
                self.abort_flag = 0

            def SetAbortFlag(self, flag: int):
                self.abort_flag = flag

        class MockIren:
            def __init__(self):
                self.mouse_pos = (150, 150)
                self.shift_key = 0

            def GetEventPosition(self):
                return self.mouse_pos

            def GetShiftKey(self):
                return self.shift_key

            def AddObserver(self, event_name, callback, priority=0.0):
                return 42

            def RemoveObserver(self, tag):
                pass

        class MockPlotter:
            def __init__(self):
                self.iren = MockIren()
                self.renderer = None

        class MockViewport:
            def __init__(self):
                self.plotter = MockPlotter()
                self.rendered = False

            @property
            def interactor(self):
                return self.plotter.iren

            def add_mesh_actor(self, actor_name, mesh_polydata, color="#FFFFFF", opacity=1.0, reset_camera=False):
                return MockActor()

            def remove_actor(self, actor_name):
                pass

            def render(self):
                self.rendered = True

        mock_viewport = MockViewport()
        gizmo = TransformGizmo(viewport=mock_viewport, target_node=node_vm)

        mock_x_actor = MockActor()
        gizmo._mesh_actors["gizmo_translate_x"] = mock_x_actor
        gizmo._actor_axis_map["gizmo_translate_x"] = GizmoAxis.X

        # Подменяем _pick_gizmo_axis для детерминированного теста
        gizmo._pick_gizmo_axis = lambda event_x_val, event_y_val: (
            GizmoAxis.X, "gizmo_translate_x"
        ) if (event_x_val, event_y_val) == (150, 150) else (GizmoAxis.NONE, None)

        started_events = []
        ended_events = []
        gizmo.transform_started.connect(lambda: started_events.append(True))
        gizmo.transform_ended.connect(lambda: ended_events.append(True))

        mock_interactor_obj = MockInteractorObj()

        # 1. Нажатие левой кнопки мыши
        gizmo._on_left_button_press(mock_interactor_obj, "LeftButtonPressEvent")
        self.assertTrue(gizmo.is_dragging)
        self.assertEqual(gizmo.active_axis, GizmoAxis.X)
        self.assertEqual(mock_interactor_obj.abort_flag, 1)
        self.assertEqual(len(started_events), 1)

        # 2. Перемещение мыши при перетаскивании
        mock_interactor_obj.abort_flag = 0
        mock_viewport.plotter.iren.mouse_pos = (170, 150)
        gizmo._compute_screen_to_world_delta = lambda last_coord_x, last_coord_y, current_coord_x, current_coord_y: np.array([20.0, 0.0, 0.0], dtype=np.float64)
        gizmo._on_mouse_move(mock_interactor_obj, "MouseMoveEvent")
        self.assertEqual(mock_interactor_obj.abort_flag, 1)
        self.assertTrue(mock_viewport.rendered)

        # 3. Отпускание кнопки мыши
        mock_interactor_obj.abort_flag = 0
        gizmo._on_left_button_release(mock_interactor_obj, "LeftButtonReleaseEvent")
        self.assertFalse(gizmo.is_dragging)
        self.assertEqual(gizmo.active_axis, GizmoAxis.NONE)
        self.assertEqual(mock_interactor_obj.abort_flag, 1)
        self.assertEqual(len(ended_events), 1)

    def test_transform_gizmo_hover_and_highlight(self):
        """Проверка подсветки активного элемента при наведении (hover) и сброса цвета."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoAxis

        class MockProperty:
            def __init__(self):
                self.color = (1.0, 0.0, 0.0)

            def SetColor(self, red: float, green: float, blue: float):
                self.color = (red, green, blue)

        class MockActor:
            def __init__(self):
                self.prop = MockProperty()

            def GetProperty(self):
                return self.prop

        class MockViewport:
            def __init__(self):
                self.plotter = None
                self.rendered = False

            @property
            def interactor(self):
                return None

            def add_mesh_actor(self, *args, **kwargs):
                return None

            def remove_actor(self, *args, **kwargs):
                pass

            def render(self):
                self.rendered = True

        gizmo = TransformGizmo(viewport=MockViewport())
        mock_actor = MockActor()
        gizmo._mesh_actors["gizmo_axis_x"] = mock_actor
        gizmo._actor_axis_map["gizmo_axis_x"] = GizmoAxis.X
        gizmo._colors[GizmoAxis.X] = "#FF0000"

        gizmo._pick_gizmo_axis = lambda event_x_val, event_y_val: (
            GizmoAxis.X, "gizmo_axis_x"
        ) if event_x_val > 100 else (GizmoAxis.NONE, None)

        # Наведение мыши
        gizmo._process_hover(150, 50)
        self.assertEqual(gizmo._hovered_actor_name, "gizmo_axis_x")
        self.assertEqual(mock_actor.prop.color, (1.0, 1.0, 0.0))

        # Увод мыши
        gizmo._process_hover(50, 50)
        self.assertIsNone(gizmo._hovered_actor_name)
        self.assertAlmostEqual(mock_actor.prop.color[0], 1.0)
        self.assertAlmostEqual(mock_actor.prop.color[1], 0.0)
        self.assertAlmostEqual(mock_actor.prop.color[2], 0.0)

    def test_transform_gizmo_hierarchical_transforms_and_cleanup(self):
        """Проверка корректной трансформации в глобальной системе при наличии родительских узлов и очистки ресурсов."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoSpace
        from core.scene.nodes import SpatialNode, CompositeNode
        from gui.viewmodels.nodes.base_node_vm import NodeViewModel

        parent_node = CompositeNode(name="Parent")
        child_node = SpatialNode(name="Child")
        parent_node.add_child(child_node)

        parent_vm = NodeViewModel(parent_node)
        child_vm = NodeViewModel(child_node)
        child_vm.parent_vm = parent_vm

        # Поворачиваем родительский узел на 90 градусов вокруг Z
        parent_rot = np.eye(4, dtype=np.float64)
        parent_rot[0, 0] = 0.0
        parent_rot[0, 1] = -1.0
        parent_rot[1, 0] = 1.0
        parent_rot[1, 1] = 0.0
        parent_vm.local_matrix = parent_rot

        gizmo = TransformGizmo(viewport=None, target_node=child_vm, grid_snap_step=0.0)
        gizmo.space = GizmoSpace.WORLD

        # Перемещаем дочерний узел в мировом пространстве по оси X на 10 единиц
        # Так как родитель повернут на 90 градусов (мировой X это родительский -Y),
        # в локальной системе ребенка смещение должно быть направлено по -Y
        gizmo.apply_translation(np.array([10.0, 0.0, 0.0]))
        child_pos = child_vm.local_matrix[0:3, 3]
        self.assertAlmostEqual(child_pos[0], 0.0, places=5)
        self.assertAlmostEqual(child_pos[1], -10.0, places=5)

        # Проверка закрытия и отсоединения
        gizmo.close()
        self.assertIsNone(gizmo.target_node)
        self.assertEqual(len(gizmo._observer_tags), 0)
        self.assertEqual(len(gizmo._actor_names), 0)

    def test_transform_gizmo_continuous_drag_snapping(self):
        """Проверка устранения мертвой зоны снаппинга при непрерывном пошаговом драге."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo, GizmoAxis
        from core.scene.nodes import SpatialNode
        from gui.viewmodels.nodes.base_node_vm import NodeViewModel

        core_node = SpatialNode(name="DragTarget")
        node_vm = NodeViewModel(core_node)

        class MockPlotter:
            def __init__(self):
                self.renderer = None

        class MockViewport:
            def __init__(self):
                self.plotter = MockPlotter()
                self.rendered = False

            @property
            def interactor(self):
                return None

            def add_mesh_actor(self, *args, **kwargs):
                return None

            def remove_actor(self, *args, **kwargs):
                pass

            def render(self):
                self.rendered = True

        mock_vp = MockViewport()
        gizmo = TransformGizmo(viewport=mock_vp, target_node=node_vm, grid_snap_step=10.0)
        gizmo._active_axis = GizmoAxis.X
        gizmo._is_dragging = True
        gizmo._initial_drag_matrix = node_vm.local_matrix.copy()

        # Моделируем обратную проекцию луча камеры: каждый шаг мыши дает +1.5 мм по оси X
        gizmo._compute_screen_to_world_delta = lambda lx, ly, cx, cy: np.array([1.5, 0.0, 0.0], dtype=np.float64)

        # 1. Сдвиг на 1.5 мм: в пределах полушага (5 мм), должно остаться 0.0
        gizmo._process_drag(0, 0, 10, 0, shift_modifier=False)
        self.assertAlmostEqual(node_vm.local_matrix[0, 3], 0.0)

        # 2. Еще 3 шага по 1.5 мм: суммарно 1.5 * 4 = 6.0 мм (> 5.0 мм), должно привязаться к 10.0 мм
        for _ in range(3):
            gizmo._process_drag(0, 0, 10, 0, shift_modifier=False)
        self.assertAlmostEqual(node_vm.local_matrix[0, 3], 10.0)

        # 3. Еще 4 шага по 1.5 мм: суммарно 6.0 + 6.0 = 12.0 мм, должно оставаться 10.0 мм
        for _ in range(4):
            gizmo._process_drag(0, 0, 10, 0, shift_modifier=False)
        self.assertAlmostEqual(node_vm.local_matrix[0, 3], 10.0)

        # 4. Еще 3 шага по 1.5 мм: суммарно 12.0 + 4.5 = 16.5 мм (> 15.0 мм), должно привязаться к 20.0 мм
        for _ in range(3):
            gizmo._process_drag(0, 0, 10, 0, shift_modifier=False)
        self.assertAlmostEqual(node_vm.local_matrix[0, 3], 20.0)

        # 5. С зажатым Shift привязка отключается (непрерывный сдвиг)
        gizmo._process_drag(0, 0, 10, 0, shift_modifier=True)
        # Суммарно 16.5 + 1.5 = 18.0 мм без снаппинга
        self.assertAlmostEqual(node_vm.local_matrix[0, 3], 18.0)

    def test_transform_gizmo_volume_adaptive_size(self):
        """Проверка адаптивного масштабирования манипулятора под размеры геометрического объема."""
        from gui.viewport_3d.transform_gizmo import TransformGizmo
        from core.geometry.volumes import Volume
        from core.geometry.geometries import Box
        from core.materials.materials import Material
        from gui.viewmodels.nodes.volume_vm import VolumeViewModel

        # Создаем большой объем (сторона 400 мм -> полуразмер 200 мм)
        mat = Material(name="TestMat")
        vol = Volume(geometry=Box(400.0, 400.0, 400.0), material=mat, name="BigBox")
        vol_vm = VolumeViewModel(vol)

        class MockPlotter:
            def __init__(self):
                self.renderer = None

        class MockViewport:
            def __init__(self):
                self.plotter = MockPlotter()
                self.rendered = False

            @property
            def interactor(self):
                return None

            def add_mesh_actor(self, *args, **kwargs):
                return None

            def remove_actor(self, *args, **kwargs):
                pass

            def render(self):
                self.rendered = True

        mock_vp = MockViewport()
        gizmo = TransformGizmo(viewport=mock_vp, target_node=vol_vm, gizmo_size=80.0)
        # При размере объема 400x400x400 полуразмер 200, адаптивный размер должен быть >= 200 * 1.35 = 270
        built_sizes = []
        original_build = gizmo._build_translate_visuals
        def _intercept_build(center, dir_x, dir_y, dir_z, size):
            built_sizes.append(size)
            original_build(center, dir_x, dir_y, dir_z, size)
        gizmo._build_translate_visuals = _intercept_build
        gizmo.update_visuals()

        self.assertTrue(len(built_sizes) > 0)
        self.assertGreaterEqual(built_sizes[0], 270.0)

    def test_transform_gizmo_native_vtk_abort_event(self):
        """Проверка безопасного сброса событий как для mock-объектов, так и для нативного VTK (GetCommand)."""
        import vtk
        from gui.viewport_3d.transform_gizmo import TransformGizmo

        class MockViewport:
            @property
            def interactor(self):
                return None
            @property
            def plotter(self):
                return None

        gizmo = TransformGizmo(viewport=MockViewport())

        # 1. Mock-объект с методом SetAbortFlag
        class MockWithAbort:
            def __init__(self):
                self.flag = 0
            def SetAbortFlag(self, f):
                self.flag = f

        caller_mock = MockWithAbort()
        gizmo._abort_event(caller_mock, 'LeftButtonPressEvent')
        self.assertEqual(caller_mock.flag, 1)

        # 2. Нативный VTK vtkGenericRenderWindowInteractor (не имеет SetAbortFlag, использует GetCommand)
        iren = vtk.vtkGenericRenderWindowInteractor()
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        iren.SetRenderWindow(rw)
        iren.Initialize()

        tag = iren.AddObserver('LeftButtonPressEvent', lambda c, e: None, 10.0)
        gizmo._observer_tags['LeftButtonPressEvent'] = tag

        # Вызов не должен выбрасывать AttributeError: object has no attribute 'SetAbortFlag'
        gizmo._abort_event(iren, 'LeftButtonPressEvent')
        cmd = iren.GetCommand(tag)
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.GetAbortFlag(), 1)

    def test_main_window_gizmo_hotkeys_with_tree_focus(self):
        """Проверка переключения режимов манипулятора клавишами W/E/R/Q при нахождении фокуса в дереве сцены."""
        from PySide6.QtWidgets import QApplication, QLineEdit
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from gui.views.main_window import MainWindow
        from gui.viewport_3d.transform_gizmo import GizmoMode, GizmoSpace
        import sys

        app = QApplication.instance() or QApplication(sys.argv)
        mw = MainWindow()
        mw.show()
        mw.activateWindow()
        app.processEvents()

        tree = mw.scene_tree.tree
        tree.setFocus()
        app.processEvents()
        self.assertTrue(tree.hasFocus() or mw.focusWidget() is tree)

        gizmo = mw.viewport_controller.transform_gizmo
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)
        self.assertEqual(gizmo.space, GizmoSpace.WORLD)

        # 1. Нажатие 'E' (вращение) при фокусе в дереве сцены
        QTest.keyClick(tree, Qt.Key.Key_E)
        self.assertEqual(gizmo.mode, GizmoMode.ROTATE)
        self.assertTrue(mw.act_gizmo_rotate.isChecked())

        # 2. Нажатие 'R' (масштаб) при фокусе в дереве сцены
        QTest.keyClick(tree, Qt.Key.Key_R)
        self.assertEqual(gizmo.mode, GizmoMode.SCALE)
        self.assertTrue(mw.act_gizmo_scale.isChecked())

        # 3. Нажатие 'W' (перемещение) при фокусе в дереве сцены
        QTest.keyClick(tree, Qt.Key.Key_W)
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)
        self.assertTrue(mw.act_gizmo_translate.isChecked())

        # 4. Нажатие 'Q' (переключение локальных/мировых координат)
        QTest.keyClick(tree, Qt.Key.Key_Q)
        self.assertEqual(gizmo.space, GizmoSpace.LOCAL)
        QTest.keyClick(tree, Qt.Key.Key_Q)
        self.assertEqual(gizmo.space, GizmoSpace.WORLD)

        # 5. Русская раскладка: клавиша 'У' (E) -> ROTATE, 'К' (R) -> SCALE, 'Ц' (W) -> TRANSLATE
        QTest.keyClicks(tree, 'у')
        self.assertEqual(gizmo.mode, GizmoMode.ROTATE)
        QTest.keyClicks(tree, 'к')
        self.assertEqual(gizmo.mode, GizmoMode.SCALE)
        QTest.keyClicks(tree, 'ц')
        self.assertEqual(gizmo.mode, GizmoMode.TRANSLATE)
        QTest.keyClicks(tree, 'й')
        self.assertEqual(gizmo.space, GizmoSpace.LOCAL)
        QTest.keyClicks(tree, 'й')
        self.assertEqual(gizmo.space, GizmoSpace.WORLD)

        # 6. Проверка, что ввод текста в QLineEdit НЕ перехватывается фильтром
        edit = QLineEdit(mw)
        edit.setFocus()
        app.processEvents()
        self.assertTrue(edit.hasFocus() or mw.focusWidget() is edit)
        QTest.keyClicks(edit, 'w')
        # Режим манипулятора не должен измениться, а текст 'w' должен появиться в edit
        self.assertEqual(edit.text(), 'w')

        mw.close()


if __name__ == '__main__':
    unittest.main()
