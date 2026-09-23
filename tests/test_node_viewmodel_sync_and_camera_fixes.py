import unittest
import numpy as np

from PySide6.QtWidgets import QApplication
import sys

APP = QApplication.instance() or QApplication(sys.argv)

from core.scene.nodes import SpatialNode, CompositeNode
from core.geometry.volumes import Volume
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from gui.viewmodels.node_viewmodel import (
    NodeViewModel,
    VolumeViewModel,
    VoxelVolumeViewModel,
    GammaCameraViewModel,
    create_node_viewmodel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.spect_manipulator import SPECTManipulator
from gui.viewport_3d.voxel_volume_renderer import VoxelVolumeRenderer
from gui.views.main_window import MainWindow


class TestNodeViewModelSyncAndCameraFixes(unittest.TestCase):
    """
    Тесты для верификации устранения рассинхронов NodeViewModel с ядром,
    кинематики GammaCamera, центрирования воксельного объема и распространения transform_changed.
    """

    # -------------------------------------------------------------------------
    # 1. Центрирование воксельного фантома и актуализация origin
    # -------------------------------------------------------------------------
    def test_voxel_volume_renderer_default_centering(self):
        """Проверка автоматического центрирования сетки (origin=None)."""
        renderer = VoxelVolumeRenderer(viewport=None)
        data = np.zeros((10, 20, 30), dtype=np.float32)
        spacing = (2.0, 3.0, 4.0)

        renderer.set_volume_data(data, voxel_size=spacing, origin=None)

        self.assertIsNotNone(renderer.grid)
        expected_origin = (-0.5 * 10 * 2.0, -0.5 * 20 * 3.0, -0.5 * 30 * 4.0)
        np.testing.assert_allclose(renderer.grid.origin, expected_origin)
        np.testing.assert_allclose(renderer.grid.spacing, spacing)

    def test_voxel_volume_renderer_inplace_origin_update(self):
        """Проверка актуализации origin при in-place обновлении одинаковой размерности."""
        renderer = VoxelVolumeRenderer(viewport=None)
        data = np.zeros((10, 10, 10), dtype=np.float32)

        # Первый вызов
        renderer.set_volume_data(data, voxel_size=1.0, origin=None)
        self.assertEqual(renderer.grid.origin, (-5.0, -5.0, -5.0))

        # Симулируем наличие volume_actor для входа в in-place ветку
        renderer.volume_actor = object()

        # In-place обновление с другим origin
        custom_origin = (10.0, 20.0, 30.0)
        renderer.set_volume_data(data, voxel_size=1.0, origin=custom_origin)
        self.assertEqual(renderer.grid.origin, custom_origin)

    def test_voxel_volume_viewmodel_properties(self):
        """Проверка свойств voxel_size, dimensions и origin в VoxelVolumeViewModel."""
        mat_arr = MaterialArray((10, 20, 30))
        mat_arr.element_list = [Material(name="Water")]
        voxel_vol = WoodcockVoxelVolume(voxel_size=2.0, material_distribution=mat_arr, name="Phantom")
        vm = VoxelVolumeViewModel(voxel_vol)

        self.assertEqual(vm.dimensions, (10, 20, 30))
        np.testing.assert_allclose(vm.origin, (-10.0, -20.0, -30.0))

        # Сеттер voxel_size
        vm.voxel_size = [4.0, 4.0, 4.0]
        np.testing.assert_allclose(vm.voxel_size, [4.0, 4.0, 4.0])
        np.testing.assert_allclose(vm.origin, (-20.0, -40.0, -60.0))

    # -------------------------------------------------------------------------
    # 2. Кинематика GammaCamera и селекция без паразитных мутаций
    # -------------------------------------------------------------------------
    def test_spect_manipulator_emit_signal_flag(self):
        """Проверка управления эмиссией сигнала в set_orbit_parameters."""
        manipulator = SPECTManipulator(viewport=None, initial_radius=200.0, initial_angle=0.0)

        emitted = []
        manipulator.orbit_changed.connect(lambda r, a, z: emitted.append((r, a, z)))

        # emit_signal=False: сигнал не должен испускаться
        manipulator.set_orbit_parameters(300.0, 45.0, z=10.0, emit_signal=False)
        self.assertEqual(len(emitted), 0)
        self.assertEqual(manipulator.radius, 300.0)
        self.assertEqual(manipulator.angle_deg, 45.0)
        self.assertEqual(manipulator.z_pos, 10.0)

        # emit_signal=True: сигнал испускается
        manipulator.set_orbit_parameters(350.0, 90.0, z=20.0, emit_signal=True)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0], (350.0, 90.0, 20.0))

    def test_gamma_camera_viewmodel_initialization_from_matrix(self):
        """Проверка извлечения orbit_radius, orbit_angle, orbit_z из local_matrix ядра."""
        col = Volume(geometry=Box(100, 100, 30), material=Material(name="Lead"))
        det = Volume(geometry=Box(100, 100, 10), material=Material(name="NaI"))
        cam = GammaCamera(collimator=col, detector=det)

        # Задаем камере матрицу трансформации: радиус лицевой поверхности 329.05, угол 90 град, z = 15.0
        half_th = cam.size[2] / 2.0
        target_mat = GammaCameraViewModel.compute_orbit_matrix(
            radius=329.05, angle_deg=90.0, z=15.0, half_thickness=half_th
        )
        cam.local_matrix = target_mat

        vm = GammaCameraViewModel(cam)
        self.assertAlmostEqual(vm.orbit_radius, 329.05, places=2)
        self.assertAlmostEqual(vm.orbit_angle, 90.0, places=2)
        self.assertAlmostEqual(vm.orbit_z, 15.0, places=2)

    def test_gamma_camera_kinematics_detector_normal_points_to_center(self):
        """
        Проверка кинематики: лицевая нормаль детектора (+Z) должна смотреть
        строго в центр орбиты (0, 0, z) при любом угле поворота.
        """
        for angle in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0, 315.0]:
            radius = 280.0
            z = 25.0
            mat = GammaCameraViewModel.compute_orbit_matrix(radius=radius, angle_deg=angle, z=z)

            # 1. Позиция камеры
            cam_pos = mat[:3, 3]
            expected_pos = np.array([radius * np.cos(np.radians(angle)), radius * np.sin(np.radians(angle)), z])
            np.testing.assert_allclose(cam_pos, expected_pos, atol=1e-5)

            # 2. Нормаль детектора (+Z, столбец 2)
            detector_normal = mat[:3, 2]
            center = np.array([0.0, 0.0, z])
            to_center = center - cam_pos
            to_center_unit = to_center / np.linalg.norm(to_center)

            # Скалярное произведение нормали и направления на центр должно быть в точности 1.0
            dot_prod = float(np.dot(detector_normal, to_center_unit))
            self.assertAlmostEqual(dot_prod, 1.0, places=5,
                                   msg=f"Нормаль детектора не смотрит в центр на угле {angle}")

            # 3. Ортогональность и определитель вращения +1
            rot = mat[:3, :3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=5)

    def test_selection_does_not_mutate_camera_matrix(self):
        """Проверка: селекция GammaCamera в MainWindow не перезаписывает ее матрицу."""
        win = MainWindow()
        col = Volume(geometry=Box(100, 100, 30), material=Material(name="Lead"))
        det = Volume(geometry=Box(100, 100, 10), material=Material(name="NaI"))
        cam = GammaCamera(collimator=col, detector=det)

        # Камера на радиусе 400 мм (по лицевой поверхности) и угле 120 град
        half_th = cam.size[2] / 2.0
        expected_matrix = GammaCameraViewModel.compute_orbit_matrix(
            radius=400.0, angle_deg=120.0, z=50.0, half_thickness=half_th
        )
        cam.local_matrix = np.copy(expected_matrix)

        root = CompositeNode(name="World")
        root.add_child(cam)
        win.scene_vm.load_scene(root)

        cam_vm = win.scene_vm.find_by_core_node(cam)
        self.assertIsNotNone(cam_vm)

        # Выделяем камеру
        win.scene_vm.select_node(cam_vm)

        # Проверяем, что матрица НЕ изменилась
        np.testing.assert_allclose(cam_vm.local_matrix, expected_matrix, atol=1e-5)
        self.assertAlmostEqual(cam_vm.orbit_radius, 400.0, places=2)
        self.assertAlmostEqual(cam_vm.orbit_angle, 120.0, places=2)
        self.assertAlmostEqual(cam_vm.orbit_z, 50.0, places=2)
        win.close()

    # -------------------------------------------------------------------------
    # 3. Распространение transform_changed на дочерние узлы
    # -------------------------------------------------------------------------
    def test_transform_changed_propagation_to_descendants(self):
        """
        Проверка: изменение local_matrix (в сеттере, translate, rotate)
        уведомляет всех потомков через transform_changed.
        """
        root = CompositeNode(name="Root")
        child = CompositeNode(name="Child")
        grandchild = SpatialNode(name="Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        root_vm = create_node_viewmodel(root)
        child_vm = root_vm.children[0]
        grandchild_vm = child_vm.children[0]

        child_events = []
        grandchild_events = []
        child_vm.transform_changed.connect(lambda: child_events.append(True))
        grandchild_vm.transform_changed.connect(lambda: grandchild_events.append(True))

        # 1. Сеттер local_matrix
        new_mat = np.eye(4)
        new_mat[0, 3] = 100.0
        root_vm.local_matrix = new_mat

        self.assertEqual(len(child_events), 1)
        self.assertEqual(len(grandchild_events), 1)
        np.testing.assert_allclose(child_vm.global_matrix[0, 3], 100.0)
        np.testing.assert_allclose(grandchild_vm.global_matrix[0, 3], 100.0)

        # 2. translate
        root_vm.translate(x=50.0)
        self.assertEqual(len(child_events), 2)
        self.assertEqual(len(grandchild_events), 2)
        np.testing.assert_allclose(child_vm.global_matrix[0, 3], 150.0)

        # 3. rotate
        root_vm.rotate(gamma=np.radians(90))
        self.assertEqual(len(child_events), 3)
        self.assertEqual(len(grandchild_events), 3)

    # -------------------------------------------------------------------------
    # 4. Ревью рассинхронов в NodeViewModel и граничные случаи
    # -------------------------------------------------------------------------
    def test_add_child_to_non_composite_raises_type_error(self):
        """Проверка: попытка добавить дочерний узел к не-Composite узлу выбрасывает TypeError."""
        leaf_core = SpatialNode(name="Leaf")
        child_core = SpatialNode(name="Child")

        leaf_vm = NodeViewModel(leaf_core)
        child_vm = NodeViewModel(child_core)

        with self.assertRaises(TypeError):
            leaf_vm.add_child(child_vm)

        self.assertEqual(len(leaf_vm.children), 0)
        self.assertIsNone(child_vm.parent_vm)

    def test_add_child_self_or_cycle_prevention(self):
        """Проверка защиты от добавления узла в себя и циклических зависимостей."""
        node_a = CompositeNode(name="A")
        node_b = CompositeNode(name="B")

        vm_a = NodeViewModel(node_a)
        vm_b = NodeViewModel(node_b)

        # Добавление в себя
        with self.assertRaises(ValueError):
            vm_a.add_child(vm_a)

        # Добавляем B к A
        vm_a.add_child(vm_b)
        self.assertIn(vm_b, vm_a.children)

        # Попытка добавить предка A в потомка B (цикл)
        with self.assertRaises(ValueError):
            vm_b.add_child(vm_a)

    def test_reparenting_node_removes_from_old_parent(self):
        """Проверка переподключения узла: узел корректно удаляется из старого родителя."""
        parent1 = CompositeNode(name="P1")
        parent2 = CompositeNode(name="P2")
        child = SpatialNode(name="C")

        p1_vm = NodeViewModel(parent1)
        p2_vm = NodeViewModel(parent2)
        c_vm = NodeViewModel(child)

        p1_vm.add_child(c_vm)
        self.assertEqual(len(p1_vm.children), 1)
        self.assertIs(c_vm.parent_vm, p1_vm)

        # Переподключаем к p2
        p2_vm.add_child(c_vm)
        self.assertEqual(len(p1_vm.children), 0)
        self.assertEqual(len(p2_vm.children), 1)
        self.assertIs(c_vm.parent_vm, p2_vm)
        self.assertIs(child.parent, parent2)

    def test_sync_children_from_core(self):
        """Проверка актуализации списка children из ядра при внешних изменениях."""
        comp = CompositeNode(name="Root")
        child1 = SpatialNode(name="C1")
        child2 = SpatialNode(name="C2")
        comp.add_child(child1)

        root_vm = create_node_viewmodel(comp)
        self.assertEqual(len(root_vm.children), 1)

        # Напрямую модифицируем ядро
        comp.add_child(child2)
        root_vm.sync_children_from_core()
        self.assertEqual(len(root_vm.children), 2)
        self.assertEqual(root_vm.children[1].name, "C2")

        # Удаляем из ядра
        comp.remove_child(child1)
        root_vm.sync_children_from_core()
        self.assertEqual(len(root_vm.children), 1)
        self.assertEqual(root_vm.children[0].name, "C2")

    def test_volume_viewmodel_material_and_size_invalidation(self):
        """Проверка инвалидации геометрии Volume при смене материала или размера."""
        vol = Volume(geometry=Box(50, 50, 50), material=Material(name="Water"), name="Vol")
        # Формируем кэш скомпилированной геометрии
        _ = vol.geometry_buffer
        self.assertIsNotNone(vol._geometry_buffer)

        vm = VolumeViewModel(vol)
        # Смена размера должна сбросить кэш geometry_buffer
        vm.size = [60, 60, 60]
        self.assertIsNone(vol._geometry_buffer)

        # Повторно кэшируем
        _ = vol.geometry_buffer
        self.assertIsNotNone(vol._geometry_buffer)

        # Смена материала на Vacuum
        vm.material_name = "Vacuum"
        self.assertEqual(vm.material_name, "Vacuum")
        self.assertIsNone(vol._geometry_buffer)

    def test_gamma_camera_zero_radius_and_angle_normalization(self):
        """Проверка сохранения дефолтных параметров при r <= 1e-4 и нормализации углов."""
        col = Volume(geometry=Box(100, 100, 30), material=Material(name="Lead"))
        det = Volume(geometry=Box(100, 100, 10), material=Material(name="NaI"))
        cam = GammaCamera(collimator=col, detector=det)
        vm = GammaCameraViewModel(cam)

        # Начальное положение камеры в начале координат - сохраняет дефолтный радиус 250 мм
        self.assertAlmostEqual(vm.orbit_radius, 250.0, places=2)

        # Перемещаем камеру вдоль оси Z (r = 0, z = 10)
        vm.translate(z=10.0)
        self.assertAlmostEqual(vm.orbit_radius, 250.0, places=2)
        self.assertAlmostEqual(vm.orbit_z, 10.0, places=2)

        # Проверяем нормализацию угла 360 градусов -> 0.0 на орбите
        mat_360 = GammaCameraViewModel.compute_orbit_matrix(
            radius=200.0, angle_deg=360.0, half_thickness=vm.half_thickness
        )
        vm.local_matrix = mat_360
        self.assertAlmostEqual(vm.orbit_radius, 200.0, places=2)
        self.assertAlmostEqual(vm.orbit_angle, 0.0, places=5)

    def test_add_child_idempotency_and_core_reparenting(self):
        """Проверка идемпотентности add_child и очистки родителя ядра без parent_vm."""
        root = CompositeNode(name="Root")
        child = SpatialNode(name="Child")
        root_vm = create_node_viewmodel(root)
        child_vm = create_node_viewmodel(child)

        root_vm.add_child(child_vm)
        self.assertEqual(len(root_vm.children), 1)

        added_signals = []
        root_vm.child_added.connect(lambda c: added_signals.append(c))

        # Повторный вызов add_child не должен дублировать узел или испускать сигнал
        root_vm.add_child(child_vm)
        self.assertEqual(len(root_vm.children), 1)
        self.assertEqual(len(added_signals), 0)

        # Репарентинг из ядра без parent_vm
        other_root = CompositeNode(name="OtherRoot")
        other_root.add_child(child)
        self.assertIs(child.parent, other_root)

        # Добавляем в root_vm узел child_vm, у которого в ядре parent был other_root
        child_vm.parent_vm = None
        root_vm.add_child(child_vm)
        self.assertIs(child.parent, root)
        self.assertNotIn(child, other_root.childs)

    def test_sync_children_from_core_recursive_and_unparenting(self):
        """Проверка рекурсивной синхронизации потомков и сброса parent при удалении из childs."""
        comp = CompositeNode(name="Root")
        sub_comp = CompositeNode(name="Sub")
        grandchild = SpatialNode(name="GC")

        comp.add_child(sub_comp)
        root_vm = create_node_viewmodel(comp)

        # Добавляем внука в ядро
        sub_comp.add_child(grandchild)
        root_vm.sync_children_from_core()
        sub_vm = root_vm.children[0]
        self.assertEqual(len(sub_vm.children), 1)
        self.assertEqual(sub_vm.children[0].name, "GC")

        # Прямое удаление внука из списка childs
        sub_comp.childs.remove(grandchild)
        root_vm.sync_children_from_core()
        self.assertEqual(len(sub_vm.children), 0)
        self.assertIsNone(grandchild.parent)

    def test_volume_invalidate_geometry_propagation_to_ancestor_volumes(self):
        """Проверка распространения инвалидации геометрии Volume вверх к предкам."""
        world = Volume(geometry=Box(500, 500, 500), material=Material(name="Air"), name="World")
        box = Volume(geometry=Box(100, 100, 100), material=Material(name="Water"), name="Box")
        world.add_child(box)

        _ = world.geometry_buffer
        self.assertIsNotNone(world._geometry_buffer)

        box_vm = VolumeViewModel(box)
        box_vm.size = [120, 120, 120]

        # Кэш предка world должен быть инвалидирован
        self.assertIsNone(world._geometry_buffer)

    def test_voxel_volume_size_and_voxel_size_sync(self):
        """Проверка синхронизации size и voxel_size в VoxelVolumeViewModel."""
        mat_arr = MaterialArray((10, 10, 10))
        mat_arr.element_list = [Material(name="Water")]
        voxel_vol = WoodcockVoxelVolume(voxel_size=2.0, material_distribution=mat_arr, name="Phantom")
        vm = VoxelVolumeViewModel(voxel_vol)

        np.testing.assert_allclose(vm.size, [20.0, 20.0, 20.0])

        # Изменение voxel_size обновляет geometry.size ядра
        vm.voxel_size = [3.0, 3.0, 3.0]
        np.testing.assert_allclose(vm.size, [30.0, 30.0, 30.0])
        np.testing.assert_allclose(voxel_vol.geometry.size, [30.0, 30.0, 30.0])

    def test_volume_viewmodel_unknown_material_raises_key_error(self):
        """Проверка: задание неизвестного материала выбрасывает KeyError."""
        vol = Volume(geometry=Box(50, 50, 50), material=Material(name="Water"), name="Vol")
        vm = VolumeViewModel(vol)

        with self.assertRaises(KeyError):
            vm.material_name = "CompletelyNonExistentMaterial999"

    def test_main_window_recursive_actor_lifecycle_and_voxel_size(self):
        """Проверка рекурсивного добавления/удаления акторов и реакции на voxel_size в MainWindow."""
        win = MainWindow()
        world = Volume(geometry=Box(500, 500, 500), material=Material(name="Air"), name="World")
        col = Volume(geometry=Box(100, 100, 30), material=Material(name="Lead"), name="Col")
        det = Volume(geometry=Box(100, 100, 10), material=Material(name="NaI"), name="Det")
        cam = GammaCamera(collimator=col, detector=det, name="Cam")

        world.add_child(cam)
        win.scene_vm.load_scene(world)

        # Проверяем, что все дочерние узлы камеры получили акторы в вьюпорте
        cam_vm = win.scene_vm.find_by_name("Cam")
        self.assertIsNotNone(cam_vm)
        for child in cam_vm.children:
            actor_name = f"mesh_{id(child)}"
            self.assertIn(actor_name, win.viewport._actors)

        # Удаление камеры должно удалить акторы всех ее потомков
        win.scene_vm.remove_node(cam_vm)
        for child in cam_vm.children:
            actor_name = f"mesh_{id(child)}"
            self.assertNotIn(actor_name, win.viewport._actors)

        win.close()


if __name__ == '__main__':
    unittest.main()
