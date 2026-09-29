"""
Модульные тесты для узла поворотной станины томографа GantryNode.
Проверяют кинематику вращения, сохранение локальных матриц дочерних детекторов,
пересчет глобальных матриц, а также экспорт и сборку через SceneExporter / SceneBuilder.
"""

import unittest
import numpy as np

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.gamma_cameras import GammaCamera
from core.materials.materials import Material
from core.scene.gantry_node import GantryNode
from core.scene.nodes import SpatialNode, CompositeNode
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.models import GantryConfig, RotateConfig


class TestGantryNode(unittest.TestCase):
    """Набор тестов кинематики и сериализации узла GantryNode."""

    def setUp(self) -> None:
        self.material = Material(name="Lead", density=11.34)

    def test_gantry_initialization_and_angle(self) -> None:
        """Проверка начальной инициализации GantryNode и установки угла вращения."""
        gantry = GantryNode(name="TestGantry")
        self.assertEqual(gantry.name, "TestGantry")
        self.assertAlmostEqual(gantry.gantry_angle, 0.0)

        # Поворот на 90 градусов (pi / 2)
        gantry.set_rotation_angle(np.pi / 2)
        self.assertAlmostEqual(gantry.gantry_angle, np.pi / 2)

        # Поворот на 180 градусов (pi)
        gantry.set_rotation_angle(np.pi)
        self.assertAlmostEqual(abs(gantry.gantry_angle), np.pi)

    def test_gantry_child_cameras_kinematics(self) -> None:
        """
        Проверка кинематики: локальные матрицы камер остаются неизменными,
        а глобальные координаты камер пересчитываются синхронно с поворотом станины.
        """
        root_world = CompositeNode(name="World")
        gantry = GantryNode(name="Gantry")
        root_world.add_child(gantry)

        collimator_1 = Volume(geometry=Box(400.0, 400.0, 30.0), material=self.material, name="Collimator_1")
        detector_1 = Volume(geometry=Box(400.0, 400.0, 10.0), material=self.material, name="Detector_1")
        camera_1 = GammaCamera(collimator=collimator_1, detector=detector_1, name="Camera_1")
        camera_1.set_orbit_position(radius=250.0, angle_deg=0.0, z=0.0)

        collimator_2 = Volume(geometry=Box(400.0, 400.0, 30.0), material=self.material, name="Collimator_2")
        detector_2 = Volume(geometry=Box(400.0, 400.0, 10.0), material=self.material, name="Detector_2")
        camera_2 = GammaCamera(collimator=collimator_2, detector=detector_2, name="Camera_2")
        camera_2.set_orbit_position(radius=250.0, angle_deg=180.0, z=0.0)

        gantry.add_child(camera_1)
        gantry.add_child(camera_2)

        initial_camera_1_local = np.copy(camera_1.local_matrix)
        initial_camera_2_local = np.copy(camera_2.local_matrix)

        initial_pos_1 = camera_1.global_matrix[0:3, 3]
        initial_pos_2 = camera_2.global_matrix[0:3, 3]

        # При угле гантри 0°: камера 1 смещена по оси X на расстояние радиуса + смещение центра
        self.assertGreater(initial_pos_1[0], 200.0)
        self.assertAlmostEqual(initial_pos_1[1], 0.0, places=3)

        # Камера 2 смещена в противоположную сторону по X
        self.assertLess(initial_pos_2[0], -200.0)
        self.assertAlmostEqual(initial_pos_2[1], 0.0, places=3)

        # Поворачиваем гантри на 90 градусов (pi / 2 радиан)
        gantry.set_rotation_angle(np.pi / 2)

        # Локальные матрицы камер обязаны остаться строго неизменными!
        np.testing.assert_allclose(camera_1.local_matrix, initial_camera_1_local)
        np.testing.assert_allclose(camera_2.local_matrix, initial_camera_2_local)

        # Мировые координаты камер пересчитались: камера 1 теперь на положительной оси Y!
        rotated_pos_1 = camera_1.global_matrix[0:3, 3]
        rotated_pos_2 = camera_2.global_matrix[0:3, 3]

        self.assertAlmostEqual(rotated_pos_1[0], 0.0, places=3)
        self.assertGreater(rotated_pos_1[1], 200.0)

        self.assertAlmostEqual(rotated_pos_2[0], 0.0, places=3)
        self.assertLess(rotated_pos_2[1], -200.0)

    def test_gantry_export_and_builder_roundtrip(self) -> None:
        """Проверка экспорта GantryNode в GantryConfig и восстановления через SceneBuilder."""
        gantry = GantryNode(name="ExportedGantry")
        gantry.rotate(alpha=np.pi / 3)  # поворот на 60 градусов

        child_node = SpatialNode(name="MarkerNode")
        child_node.translate(x=100.0, y=0.0, z=0.0)
        gantry.add_child(child_node)

        # Экспорт в конфигурацию
        exported_config = SceneExporter.export_scene(gantry)
        self.assertIsInstance(exported_config, GantryConfig)
        self.assertEqual(exported_config.type, "Gantry")
        self.assertEqual(exported_config.name, "ExportedGantry")
        self.assertEqual(len(exported_config.children), 1)

        # Сборка из конфигурации
        builder = SceneBuilder()
        built_gantry = builder.build_scene(exported_config)

        self.assertIsInstance(built_gantry, GantryNode)
        self.assertEqual(built_gantry.name, "ExportedGantry")
        self.assertEqual(len(built_gantry.childs), 1)
        self.assertAlmostEqual(built_gantry.gantry_angle, np.pi / 3, places=4)

    def test_nested_gantry_hierarchy_kinematics(self) -> None:
        """
        Проверка кинематики при сложной составной иерархии:
        World -> Room (смещение и поворот) -> GantryNode -> GammaCamera.
        Проверяет каскадный пересчет глобальных матриц M_world = M_room @ M_gantry @ M_cam.
        """
        root_world = CompositeNode(name="World")
        room_node = CompositeNode(name="DiagnosticRoom")
        room_node.translate(x=100.0, y=200.0, z=50.0)
        room_node.rotate(alpha=np.pi / 6)  # поворот кабинета на 30°
        root_world.add_child(room_node)

        gantry = GantryNode(name="MountedGantry")
        room_node.add_child(gantry)

        collimator = Volume(geometry=Box(400.0, 400.0, 30.0), material=self.material, name="Collimator")
        detector = Volume(geometry=Box(400.0, 400.0, 10.0), material=self.material, name="Detector")
        camera = GammaCamera(collimator=collimator, detector=detector, name="MountedCamera")
        camera.set_orbit_position(radius=250.0, angle_deg=0.0, z=0.0)
        gantry.add_child(camera)

        initial_camera_local = np.copy(camera.local_matrix)

        # Вычисляем ожидаемую мировую позицию до поворота гантри:
        # В координатах кабинета камера на (270, 0, 0).
        # Поворот кабинета на 30° + смещение (100, 200, 50).
        expected_world_before = room_node.global_matrix @ gantry.local_matrix @ camera.local_matrix
        np.testing.assert_allclose(camera.global_matrix, expected_world_before, atol=1e-5)

        # Поворачиваем гантри на 90° (pi / 2)
        gantry.set_rotation_angle(np.pi / 2)

        # Локальная матрица камеры не должна измениться
        np.testing.assert_allclose(camera.local_matrix, initial_camera_local)

        # Глобальная матрица обязана корректно объединить трансформацию кабинета и поворот гантри
        expected_world_after = room_node.global_matrix @ gantry.local_matrix @ camera.local_matrix
        np.testing.assert_allclose(camera.global_matrix, expected_world_after, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
