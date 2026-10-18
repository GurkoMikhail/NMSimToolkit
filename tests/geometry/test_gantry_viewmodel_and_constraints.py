"""
Тесты интеграции GUI: GantryViewModel, кинематические ограничения GantryKinematicConstraint,
CameraMountKinematicConstraint и синхронизация в SpectProcedureViewModel.
"""

import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from core.scene.gamma_camera_node import GammaCameraNode
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from gui.viewmodels import create_default_gamma_camera_vm
from core.geometry.geometries import Box
from core.geometry.spect_kinematics import compute_orbit_matrix, extract_orbit_parameters
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.gantry_node import GantryNode
from core.scene.nodes import CompositeNode
from typing import Any, Optional
from gui.controllers.viewport_controller import SceneViewportController
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.procedure_viewmodel import SpectProcedureViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewport_3d.kinematic_constraints import (
    IKinematicConstraint,
    GantryKinematicConstraint,
    CameraMountKinematicConstraint,
    SpectOrbitKinematicConstraint,
    IRotatableProcedure,
)
from gui.viewport_3d.transform_gizmo import GizmoAxis, GizmoMode, GizmoSpace, TransformGizmo

app = QApplication.instance() or QApplication([])


class TestGantryViewModelAndConstraints(unittest.TestCase):
    """Тестирование связки GantryViewModel, ограничений и процедур ОФЭКТ."""

    def setUp(self) -> None:
        self.material = Material(name="Lead", density=11.34)

    def test_factory_creates_gantry_viewmodel(self) -> None:
        """Проверка того, что фабрика create_node_viewmodel создает GantryViewModel для GantryNode."""
        core_gantry = GantryNode(name="TestGantryCore")
        gantry_vm = create_node_viewmodel(core_gantry)
        self.assertIsInstance(gantry_vm, GantryViewModel)
        self.assertEqual(gantry_vm.name, "TestGantryCore")
        self.assertEqual(gantry_vm.node_type, "GantryNode")

    def test_gantry_viewmodel_angle_property_and_signals(self) -> None:
        """Проверка изменения угла ротора через свойство gantry_angle и испускания сигналов."""
        core_gantry = GantryNode(name="RotaryGantry")
        gantry_vm = GantryViewModel(core_gantry)

        signals_received = []
        gantry_vm.property_changed.connect(lambda name, val: signals_received.append((name, val)))
        transform_signals = []
        gantry_vm.transform_changed.connect(lambda: transform_signals.append(True))

        gantry_vm.gantry_angle = 45.0
        self.assertAlmostEqual(gantry_vm.gantry_angle, 45.0)
        self.assertAlmostEqual(core_gantry.gantry_angle, np.radians(45.0))
        self.assertTrue(len(signals_received) >= 1)
        self.assertTrue(len(transform_signals) >= 1)

    def test_spect_procedure_sync_with_scene_mounts_gantry(self) -> None:
        """Проверка того, что SpectProcedureViewModel.sync_with_scene монтирует детекторы внутрь GantryViewModel."""
        root_node = CompositeNode(name="World")
        scene_vm = SceneViewModel(root_node)

        # Добавляем одну исходную камеру
        cam_vm = create_default_gamma_camera_vm(name="InitialCamera")
        scene_vm.add_node(scene_vm.root_vm, cam_vm)

        procedure = SpectProcedureViewModel(steps=32, gamma_cameras=2, radius=280.0, start_angle=30.0)
        procedure.sync_with_scene(scene_vm)

        # В сцене должен появиться GantryViewModel
        all_nodes = scene_vm.all_nodes()
        gantry_vms = [node for node in all_nodes if isinstance(node, GantryViewModel)]
        self.assertEqual(len(gantry_vms), 1)
        gantry_vm = gantry_vms[0]

        # Камеры должны быть дочерними для GantryViewModel
        camera_vms = [node for node in all_nodes if isinstance(node, GammaCameraViewModel)]
        self.assertEqual(len(camera_vms), 2)
        for cam in camera_vms:
            self.assertIs(cam.parent_vm, gantry_vm)

        # Начальный угол станины равен start_angle
        self.assertAlmostEqual(gantry_vm.gantry_angle, 30.0)

    def test_spect_procedure_returns_specialized_constraints(self) -> None:
        """Проверка возврата GantryKinematicConstraint и CameraMountKinematicConstraint."""
        procedure = SpectProcedureViewModel()
        core_gantry = GantryNode()
        gantry_vm = GantryViewModel(core_gantry)

        cam_vm = create_default_gamma_camera_vm()

        gantry_constraint = procedure.get_kinematic_constraint_for_node(gantry_vm)
        camera_constraint = procedure.get_kinematic_constraint_for_node(cam_vm)

        self.assertIsInstance(gantry_constraint, GantryKinematicConstraint)
        self.assertIsInstance(camera_constraint, CameraMountKinematicConstraint)
        self.assertIsInstance(camera_constraint, SpectOrbitKinematicConstraint)

    def test_gantry_kinematic_constraint_rules(self) -> None:
        """Проверка правил констрейта станины: 1-DOF вращение вокруг оси Z, запрет перемещений и масштабирования."""
        procedure = SpectProcedureViewModel(start_angle=0.0)
        core_gantry = GantryNode()
        gantry_vm = GantryViewModel(core_gantry)
        constraint = GantryKinematicConstraint(procedure_vm=procedure, gantry_vm=gantry_vm)

        self.assertFalse(constraint.is_scale_allowed())
        self.assertEqual(constraint.get_forced_space(), GizmoSpace.WORLD)
        self.assertEqual(constraint.get_allowed_axes(GizmoMode.ROTATE), {GizmoAxis.Z})
        self.assertEqual(constraint.get_allowed_axes(GizmoMode.TRANSLATE), set())

        # Проверка блокировки линейного перемещения
        world_delta, trans_data = constraint.filter_translation(
            target_node=gantry_vm,
            proposed_world_delta=np.array([10.0, 20.0, 30.0]),
            initial_matrix=np.eye(4),
        )
        np.testing.assert_allclose(world_delta, [0.0, 0.0, 0.0])

        # Проверка вращения: снаппинг к 5 градусам
        rot_axis, snapped_angle, rot_data = constraint.filter_rotation(
            target_node=gantry_vm,
            axis=np.array([0.0, 0.0, 1.0]),
            proposed_angle=23.4,
            initial_matrix=np.eye(4),
        )
        self.assertAlmostEqual(snapped_angle, 25.0)
        self.assertAlmostEqual(rot_data["angle"], 25.0)

        # При коммите угол фиксируется в процедуре
        constraint.on_transform_committed(gantry_vm, rot_data)
        self.assertAlmostEqual(procedure.start_angle, 25.0)

    def test_rotatable_procedure_protocol_compliance(self) -> None:
        """Проверка соответствия SpectProcedureViewModel протоколу IRotatableProcedure."""
        procedure = SpectProcedureViewModel(start_angle=45.0)
        self.assertIsInstance(procedure, IRotatableProcedure)
        self.assertAlmostEqual(procedure.start_angle, 45.0)

    def test_camera_mount_radial_translation(self) -> None:
        """Проверка радиального вылета каретки: изменение радиуса всех головок с сохранением ориентации станины."""
        root_node = CompositeNode(name="World")
        scene_vm = SceneViewModel(root_node)

        cam1_vm = create_default_gamma_camera_vm(name="Camera_1")
        scene_vm.add_node(scene_vm.root_vm, cam1_vm)

        procedure = SpectProcedureViewModel(steps=32, gamma_cameras=2, radius=250.0, start_angle=0.0)
        procedure.sync_with_scene(scene_vm)

        gantry_vms = [node for node in scene_vm.all_nodes() if isinstance(node, GantryViewModel)]
        self.assertEqual(len(gantry_vms), 1)
        gantry_vm = gantry_vms[0]

        camera_vms = [node for node in scene_vm.all_nodes() if isinstance(node, GammaCameraViewModel)]
        self.assertEqual(len(camera_vms), 2)
        cam1_target = camera_vms[0]
        cam2_sibling = camera_vms[1]

        constraint = procedure.get_kinematic_constraint_for_node(cam1_target)
        self.assertIsInstance(constraint, CameraMountKinematicConstraint)

        # Радиальное перемещение по нормали (ось Z детектора) на +50 мм
        initial_cam_matrix = cam1_target.local_matrix.copy()
        proposed_delta = np.array([50.0, 0.0, 0.0])  # смещение вдоль оси X (радиальное для детектора на 0°)

        filtered_delta, changed_data = constraint.filter_translation(
            target_node=cam1_target,
            proposed_world_delta=proposed_delta,
            initial_matrix=initial_cam_matrix,
            active_axis=GizmoAxis.Z,
        )

        self.assertEqual(changed_data["action"], "radial")
        self.assertAlmostEqual(changed_data["radius"], 300.0)

        # Применение трансформации к целевой камере и оповещение констрейнта
        cam1_target.local_matrix = changed_data["matrix"]
        constraint.on_transform_changed(cam1_target, changed_data)

        # Спаренная головка cam2 также синхронизирует радиус до 300 мм
        cam2_radius = float(np.hypot(cam2_sibling.local_matrix[0, 3], cam2_sibling.local_matrix[1, 3])) - cam2_sibling.half_thickness
        self.assertAlmostEqual(cam2_radius, 300.0)
        self.assertAlmostEqual(procedure.radius, 300.0)
        # Станина не должна вращаться при радиальном смещении
        self.assertAlmostEqual(gantry_vm.gantry_angle, 0.0)

    def test_camera_mount_tangential_translation_rotates_gantry_without_double_rotation(self) -> None:
        """
        Проверка тангенциального перехвата ('потянуть за ручку аппарата'):
        вращается родительский узел GantryViewModel, а локальные углы монтажа камер
        остаются строго неизменными, исключая паразитное двойное вращение.
        """
        root_node = CompositeNode(name="World")
        scene_vm = SceneViewModel(root_node)

        cam1_vm = create_default_gamma_camera_vm(name="Camera_1")
        scene_vm.add_node(scene_vm.root_vm, cam1_vm)

        procedure = SpectProcedureViewModel(steps=32, gamma_cameras=2, radius=250.0, start_angle=0.0)
        procedure.sync_with_scene(scene_vm)

        gantry_vm = [node for node in scene_vm.all_nodes() if isinstance(node, GantryViewModel)][0]
        camera_vms = [node for node in scene_vm.all_nodes() if isinstance(node, GammaCameraViewModel)]
        cam1_target = camera_vms[0]
        cam2_sibling = camera_vms[1]

        # Исходные локальные углы монтажа: 0° и 180°
        cam1_angle = float(np.degrees(np.arctan2(cam1_target.local_matrix[1, 3], cam1_target.local_matrix[0, 3])) % 360.0)
        cam2_angle = float(np.degrees(np.arctan2(cam2_sibling.local_matrix[1, 3], cam2_sibling.local_matrix[0, 3])) % 360.0)
        self.assertAlmostEqual(cam1_angle, 0.0)
        self.assertAlmostEqual(cam2_angle, 180.0)

        constraint = procedure.get_kinematic_constraint_for_node(cam1_target)

        # Тангенциальное воздействие: смещение вдоль касательной Y для камеры на 0°
        # delta_y = radius * delta_rad => для ~20°: 270 мм * radians(20) ~ 94.2 мм
        effective_radius = float(np.hypot(cam1_target.local_matrix[0, 3], cam1_target.local_matrix[1, 3]))
        proposed_tangent_delta = np.array([0.0, effective_radius * np.radians(20.0), 0.0])

        initial_cam_matrix = cam1_target.local_matrix.copy()
        filtered_delta, changed_data = constraint.filter_translation(
            target_node=cam1_target,
            proposed_world_delta=proposed_tangent_delta,
            initial_matrix=initial_cam_matrix,
            active_axis=GizmoAxis.X,
        )

        self.assertEqual(changed_data["action"], "tangential")
        self.assertAlmostEqual(changed_data["gantry_angle"], 20.0)

        # КРИТИЧЕСКИЙ ИНВАРИАНТ: локальная матрица детектора НЕ должна повернуться внутри станины!
        # Ее угол должен остаться 0° (исходное место монтажа на рельсе).
        cam1_target.local_matrix = changed_data["matrix"]
        cam1_angle_after = float(np.degrees(np.arctan2(cam1_target.local_matrix[1, 3], cam1_target.local_matrix[0, 3])) % 360.0)
        self.assertAlmostEqual(cam1_angle_after, 0.0)

        # Оповещение констрейнта транслирует перемещение в поворот станины
        constraint.on_transform_changed(cam1_target, changed_data)
        self.assertAlmostEqual(gantry_vm.gantry_angle, 20.0)
        self.assertAlmostEqual(procedure.start_angle, 20.0)

        # Фиксация перемещения
        constraint.on_transform_committed(cam1_target, changed_data)
        self.assertAlmostEqual(procedure.start_angle, 20.0)

        # Проверка мировых координат:
        # Камера 1 в мировых координатах повернута на 20° (станина 20° + локально 0°)
        # Камера 2 в мировых координатах повернута на 200° (станина 20° + локально 180°)
        # Взаимный угол между детекторами строго равен 180°!
        pos1_world = cam1_target.global_matrix[0:2, 3]
        pos2_world = cam2_sibling.global_matrix[0:2, 3]

        angle_world_1 = float(np.degrees(np.arctan2(pos1_world[1], pos1_world[0])) % 360.0)
        angle_world_2 = float(np.degrees(np.arctan2(pos2_world[1], pos2_world[0])) % 360.0)

        self.assertAlmostEqual(angle_world_1, 20.0, places=3)
        self.assertAlmostEqual(angle_world_2, 200.0, places=3)
        angle_difference = (angle_world_2 - angle_world_1) % 360.0
        self.assertAlmostEqual(angle_difference, 180.0, places=3)

    def test_spect_l_mode_switching_with_portrait_detectors(self) -> None:
        """
        Проверка интерактивного переключения геометрии 180° -> 90° (L-режим)
        при детекторах, развернутых оператором в книжную ориентацию (Portrait).
        Проверяет, что при смене режима взаимный угол становится 90°, а книжная ориентация сохраняется.
        """
        root_node = CompositeNode(name="World")
        scene_vm = SceneViewModel(root_node)

        cam1_vm = create_default_gamma_camera_vm(name="Camera_1")
        scene_vm.add_node(scene_vm.root_vm, cam1_vm)

        procedure = SpectProcedureViewModel(steps=32, gamma_cameras=2, radius=250.0, start_angle=0.0)
        procedure.sync_with_scene(scene_vm)

        gantry_vm = [node for node in scene_vm.all_nodes() if isinstance(node, GantryViewModel)][0]
        camera_vms = [node for node in scene_vm.all_nodes() if isinstance(node, GammaCameraViewModel)]
        cam1 = camera_vms[0]
        cam2 = camera_vms[1]

        # Разворачиваем обе камеры в книжную ориентацию (Portrait: поворот на 90° вокруг нормали Z)
        constraint_cam1 = procedure.get_kinematic_constraint_for_node(cam1)
        _, _, rot_data_1 = constraint_cam1.filter_rotation(
            target_node=cam1,
            axis=cam1.local_matrix[0:3, 2],
            proposed_angle=90.0,
            initial_matrix=cam1.local_matrix,
        )
        cam1.local_matrix = rot_data_1["matrix"]

        constraint_cam2 = procedure.get_kinematic_constraint_for_node(cam2)
        _, _, rot_data_2 = constraint_cam2.filter_rotation(
            target_node=cam2,
            axis=cam2.local_matrix[0:3, 2],
            proposed_angle=90.0,
            initial_matrix=cam2.local_matrix,
        )
        cam2.local_matrix = rot_data_2["matrix"]

        # Проверяем, что обе камеры находятся в Portrait режиме (ось стола Y детектора перпендикулярна оси Z аппарата)
        self.assertAlmostEqual(abs(float(np.dot(cam1.local_matrix[0:3, 1], [0.0, 0.0, 1.0]))), 0.0, places=3)
        self.assertAlmostEqual(abs(float(np.dot(cam2.local_matrix[0:3, 1], [0.0, 0.0, 1.0]))), 0.0, places=3)

        # Переключаем процедуру в L-режим (90°)
        procedure.head_mode = "L-режим (90°)"
        self.assertEqual(procedure.head_angles, [0.0, 90.0])
        procedure.sync_with_scene(scene_vm)

        # Проверяем углы монтажа на станине: 0° и 90°
        cam1_angle = float(np.degrees(np.arctan2(cam1.local_matrix[1, 3], cam1.local_matrix[0, 3])) % 360.0)
        cam2_angle = float(np.degrees(np.arctan2(cam2.local_matrix[1, 3], cam2.local_matrix[0, 3])) % 360.0)
        self.assertAlmostEqual(cam1_angle, 0.0)
        self.assertAlmostEqual(cam2_angle, 90.0)

        # Проверяем, что портретная ориентация сохранилась после переключения геометрии
        self.assertAlmostEqual(abs(float(np.dot(cam1.local_matrix[0:3, 1], [0.0, 0.0, 1.0]))), 0.0, places=3)
        self.assertAlmostEqual(abs(float(np.dot(cam2.local_matrix[0:3, 1], [0.0, 0.0, 1.0]))), 0.0, places=3)

    def test_base_node_vm_kinematic_constraints_hierarchy(self) -> None:
        """
        Проверка контракта кинематических ограничений в NodeViewModel:
        - get_self_kinematic_constraint / set_self_kinematic_constraint
        - get_child_kinematic_constraint / set_child_kinematic_constraint
        - приоритет родительского ограничения в get_effective_kinematic_constraint
        - очистка кэша специфических ограничений потомков при удалении узла.
        """
        parent_core = CompositeNode(name="ParentNode")
        child_core = CompositeNode(name="ChildNode")
        parent_view_model = NodeViewModel(parent_core)
        child_view_model = NodeViewModel(child_core)
        parent_view_model.add_child(child_view_model)

        # 1. По умолчанию ограничения отсутствуют
        self.assertIsNone(child_view_model.get_self_kinematic_constraint())
        self.assertIsNone(parent_view_model.get_child_kinematic_constraint(child_view_model))
        self.assertIsNone(child_view_model.get_effective_kinematic_constraint())

        # 2. Установка собственного ограничения узла
        class DummySelfConstraint(IKinematicConstraint):
            def is_scale_allowed(self) -> bool:
                return False

        self_constraint = DummySelfConstraint()
        child_view_model.set_self_kinematic_constraint(self_constraint)
        self.assertIs(child_view_model.get_self_kinematic_constraint(), self_constraint)
        self.assertIs(child_view_model.get_effective_kinematic_constraint(), self_constraint)

        # 3. Установка ограничения потомков по умолчанию на родителе
        class DummyParentChildConstraint(IKinematicConstraint):
            def is_scale_allowed(self) -> bool:
                return True

        parent_child_constraint = DummyParentChildConstraint()
        parent_view_model.set_child_kinematic_constraint(parent_child_constraint)
        self.assertIs(parent_view_model.get_child_kinematic_constraint(child_view_model), parent_child_constraint)
        # Приоритет родительского ограничения над собственным!
        self.assertIs(child_view_model.get_effective_kinematic_constraint(), parent_child_constraint)

        # 4. Установка специализированного ограничения для конкретного потомка
        class DummySpecificChildConstraint(IKinematicConstraint):
            pass

        specific_constraint = DummySpecificChildConstraint()
        parent_view_model.set_child_kinematic_constraint(specific_constraint, child_vm=child_view_model)
        self.assertIs(parent_view_model.get_child_kinematic_constraint(child_view_model), specific_constraint)
        self.assertIs(child_view_model.get_effective_kinematic_constraint(), specific_constraint)

        # 5. При удалении потомка специфическое ограничение очищается
        parent_view_model.remove_child(child_view_model)
        self.assertIsNone(child_view_model.parent_vm)
        self.assertIs(child_view_model.get_effective_kinematic_constraint(), self_constraint)

    def test_gantry_and_camera_effective_kinematic_constraints(self) -> None:
        """
        Проверка того, что GantryViewModel автоматически предоставляет:
        - GantryKinematicConstraint для самой станины через get_effective_kinematic_constraint
        - CameraMountKinematicConstraint для дочерних GammaCameraViewModel через родительское ограничение.
        """
        gantry_core = GantryNode(name="Gantry_Test")
        gantry_view_model = GantryViewModel(gantry_core)

        # Станина возвращает GantryKinematicConstraint
        gantry_effective_constraint = gantry_view_model.get_effective_kinematic_constraint()
        self.assertIsInstance(gantry_effective_constraint, GantryKinematicConstraint)
        self.assertIs(gantry_effective_constraint.gantry_vm, gantry_view_model)

        # Создаем гамма-камеру
        camera_view_model = create_default_gamma_camera_vm(name="MountedCamera")

        # До монтирования на станину у камеры нет эффективного ограничения
        self.assertIsNone(camera_view_model.get_effective_kinematic_constraint())

        # Монтируем камеру на станину
        gantry_view_model.add_child(camera_view_model)

        # Теперь эффективное ограничение камеры диктуется станиной и является CameraMountKinematicConstraint!
        camera_effective_constraint = camera_view_model.get_effective_kinematic_constraint()
        self.assertIsInstance(camera_effective_constraint, CameraMountKinematicConstraint)
        self.assertIs(camera_effective_constraint.gantry_vm, gantry_view_model)
        self.assertIs(camera_effective_constraint.camera_vm, camera_view_model)

        # Размонтируем камеру со станины
        gantry_view_model.remove_child(camera_view_model)
        self.assertIsNone(camera_view_model.get_effective_kinematic_constraint())

    def test_scene_viewport_controller_node_selected_uses_effective_constraint_without_procedure(self) -> None:
        """
        Проверка того, что SceneViewportController.on_node_selected корректно накладывает
        кинематические ограничения напрямую из графа сцены (Gantry / Camera), даже если
        в контроллере процедура не назначена (procedure_vm is None).
        """
        root_core = CompositeNode(name="SceneRoot")
        scene_view_model = SceneViewModel(root_core)

        gantry_core = GantryNode(name="MainGantry")
        gantry_view_model = GantryViewModel(gantry_core)
        scene_view_model.add_node(scene_view_model.root_vm, gantry_view_model)

        camera_view_model = create_default_gamma_camera_vm(name="ChildCamera")
        scene_view_model.add_node(gantry_view_model, camera_view_model)

        simple_volume = Volume(geometry=Box(200.0, 200.0, 200.0), material=self.material, name="SimplePhantom")
        phantom_view_model = NodeViewModel(simple_volume)
        scene_view_model.add_node(scene_view_model.root_vm, phantom_view_model)

        class MockSignal:
            def connect(self, slot_function: Any) -> None:
                pass

            def disconnect(self, slot_function: Any = None) -> None:
                pass

            def emit(self, *arguments: Any) -> None:
                pass

        class MockPlotter:
            def __init__(self) -> None:
                self.renderer = None

        class MockViewport:
            def __init__(self) -> None:
                self.plotter = MockPlotter()
                self.rendered = False
                self._actors = {}
                self.camera_interaction_started = MockSignal()
                self.camera_interaction_ended = MockSignal()
                self.camera_moved = MockSignal()

            @property
            def interactor(self) -> None:
                return None

            def add_mesh_actor(self, actor_name: str, mesh: Any, **kwargs: Any) -> None:
                self._actors[actor_name] = mesh

            def add_actor(self, actor_name: str, actor: Any, **kwargs: Any) -> None:
                self._actors[actor_name] = actor

            def remove_actor(self, actor_name: str) -> None:
                self._actors.pop(actor_name, None)

            def update_actor_transform(self, actor_name: str, matrix: Any) -> None:
                pass

            def set_node_local_transform(self, node_id: int, local_matrix: Any, parent_id: Optional[int] = None) -> None:
                pass

            def attach_actor_to_node(self, actor_name: str, node_id: int) -> bool:
                return True

            def remove_node_transform(self, node_id: int) -> None:
                pass

            def request_render(self) -> None:
                self.rendered = True

            def begin_update(self) -> None:
                pass

            def end_update(self) -> None:
                self.rendered = True

            def render(self) -> None:
                self.rendered = True

        mock_viewport = MockViewport()
        viewport_controller = SceneViewportController(viewport=mock_viewport, scene_vm=scene_view_model)
        # Процедура намеренно НЕ задана
        self.assertIsNone(viewport_controller.procedure_vm)

        # 1. Выделяем станину GantryViewModel
        viewport_controller.on_node_selected(gantry_view_model)
        gizmo = viewport_controller.transform_gizmo
        self.assertIsNotNone(gizmo)
        self.assertIsInstance(gizmo.constraint, GantryKinematicConstraint)

        # 2. Выделяем дочернюю камеру GammaCameraViewModel
        viewport_controller.on_node_selected(camera_view_model)
        self.assertIsInstance(gizmo.constraint, CameraMountKinematicConstraint)

        # 3. Выделяем обычный объемный узел - кинематических ограничений нет (свободный 6-DOF)
        viewport_controller.on_node_selected(phantom_view_model)
        self.assertIsNone(gizmo.constraint)

    def test_two_way_reactive_observer_synchronization(self) -> None:
        """
        Проверка двусторонней реактивной синхронизации (Observer) между SpectProcedureViewModel,
        станиной GantryViewModel и гамма-камерами GammaCameraViewModel:
        - поворот станины обновляет start_angle процедуры
        - изменение start_angle процедуры обновляет угол станины
        - перемещение каретки камеры обновляет radius процедуры
        - изменение radius процедуры перемещает все каретки камер
        - динамическое добавление и удаление камер корректно подписывает и отписывает слушателей.
        """
        root_core = CompositeNode(name="WorldRoot")
        scene_view_model = SceneViewModel(root_core)

        initial_camera_view_model = create_default_gamma_camera_vm(name="Camera_Init")
        scene_view_model.add_node(scene_view_model.root_vm, initial_camera_view_model)

        procedure = SpectProcedureViewModel(steps=32, gamma_cameras=2, radius=240.0, start_angle=10.0)
        procedure.sync_with_scene(scene_view_model)

        gantry_view_models = [node for node in scene_view_model.all_nodes() if isinstance(node, GantryViewModel)]
        self.assertEqual(len(gantry_view_models), 1)
        gantry_view_model = gantry_view_models[0]

        camera_view_models = [node for node in scene_view_model.all_nodes() if isinstance(node, GammaCameraViewModel)]
        self.assertEqual(len(camera_view_models), 2)
        camera_view_model_first = camera_view_models[0]
        camera_view_model_second = camera_view_models[1]

        # 1. Поворот станины -> обновление procedure.start_angle
        gantry_view_model.gantry_angle = 45.0
        self.assertAlmostEqual(procedure.start_angle, 45.0)

        # 2. Изменение procedure.start_angle -> обновление угла станины
        procedure.start_angle = 75.0
        self.assertAlmostEqual(gantry_view_model.gantry_angle, 75.0)

        # 3. Изменение радиуса камеры -> обновление procedure.radius
        cam1_angle = float(np.degrees(np.arctan2(camera_view_model_first.local_matrix[1, 3], camera_view_model_first.local_matrix[0, 3])) % 360.0)
        camera_view_model_first.local_matrix = compute_orbit_matrix(
            radius=320.0,
            angle=cam1_angle,
            z=float(camera_view_model_first.local_matrix[2, 3]),
            half_thickness=camera_view_model_first.half_thickness,
        )
        self.assertAlmostEqual(procedure.radius, 320.0)

        # 4. Изменение procedure.radius -> обновление радиуса обеих камер
        procedure.radius = 290.0
        cam1_rad = float(np.hypot(camera_view_model_first.local_matrix[0, 3], camera_view_model_first.local_matrix[1, 3])) - camera_view_model_first.half_thickness
        cam2_rad = float(np.hypot(camera_view_model_second.local_matrix[0, 3], camera_view_model_second.local_matrix[1, 3])) - camera_view_model_second.half_thickness
        self.assertAlmostEqual(cam1_rad, 290.0)
        self.assertAlmostEqual(cam2_rad, 290.0)

        # 5. Динамическое добавление 3-й камеры на станину
        third_camera_view_model = create_default_gamma_camera_vm(name="Camera_3")
        third_camera_view_model.local_matrix = compute_orbit_matrix(
            radius=290.0, angle=240.0, z=0.0, half_thickness=third_camera_view_model.half_thickness
        )
        scene_view_model.add_node(gantry_view_model, third_camera_view_model)

        # Изменение радиуса добавленной камеры обновляет процедуру
        third_camera_view_model.local_matrix = compute_orbit_matrix(
            radius=330.0, angle=240.0, z=0.0, half_thickness=third_camera_view_model.half_thickness
        )
        self.assertAlmostEqual(procedure.radius, 330.0)

        # 6. Динамическое удаление камеры со станины отписывает её
        scene_view_model.remove_node(third_camera_view_model)
        # Изменение радиуса удаленной камеры больше не влияет на процедуру
        third_camera_view_model.local_matrix = compute_orbit_matrix(
            radius=150.0, angle=240.0, z=0.0, half_thickness=third_camera_view_model.half_thickness
        )
        self.assertAlmostEqual(procedure.radius, 330.0)

    def test_extract_orbit_parameters_roundtrip_and_edge_cases(self) -> None:
        """
        Проверка обратимости пары compute_orbit_matrix <-> extract_orbit_parameters
        на различных углах орбиты, крена (roll), Z-позициях, толщинах и вырожденном R -> 0.
        """
        for radius_val in (50.0, 250.0, 480.0):
            for angle_val in (0.0, 37.5, 90.0, 180.0, 270.0, 359.5):
                for z_val in (-120.0, 0.0, 85.0):
                    for roll_val in (-45.0, 0.0, 22.5, 135.0):
                        for ht_val in (0.0, 42.5):
                            mat = compute_orbit_matrix(
                                radius=radius_val,
                                angle=angle_val,
                                z=z_val,
                                roll=roll_val,
                                half_thickness=ht_val,
                            )
                            ext_r, ext_angle, ext_z, ext_roll = extract_orbit_parameters(mat, half_thickness=ht_val)
                            self.assertAlmostEqual(ext_r, radius_val, places=6)
                            self.assertAlmostEqual(ext_angle, angle_val % 360.0, places=6)
                            self.assertAlmostEqual(ext_z, z_val, places=6)
                            self.assertAlmostEqual(ext_roll, roll_val % 360.0, places=6)

        # Вырожденный случай в начале координат (R_center = 0, half_thickness > 0 -> R клэмпится в 0.0)
        identity_mat = np.eye(4, dtype=np.float64)
        identity_mat[2, 3] = 15.0
        ext_r0, ext_a0, ext_z0, _ = extract_orbit_parameters(identity_mat, half_thickness=30.0)
        self.assertEqual(ext_r0, 0.0)
        self.assertEqual(ext_a0, 0.0)
        self.assertEqual(ext_z0, 15.0)

    def test_cacheless_global_matrix_and_local_coordinate_transforms(self) -> None:
        """
        Проверка прямого бескэшевого вычисления SpatialNode.global_matrix,
        обратимости convert_to_global_position / convert_to_local_position
        и актуальности компиляции Volume.geometry_buffer при мутации дерева.
        """
        parent = Volume(geometry=Box(500.0, 500.0, 500.0), material=self.material, name="ParentVol")
        child = Volume(geometry=Box(100.0, 100.0, 100.0), material=self.material, name="ChildVol")
        grandchild = Volume(geometry=Box(20.0, 20.0, 20.0), material=self.material, name="GrandChildVol")
        parent.add_child(child)
        child.add_child(grandchild)

        parent.local_matrix = compute_orbit_matrix(radius=200.0, angle=35.0, z=40.0, roll=15.0)
        child.local_matrix = compute_orbit_matrix(radius=80.0, angle=120.0, z=-25.0, roll=-30.0)
        grandchild.local_matrix = compute_orbit_matrix(radius=10.0, angle=210.0, z=5.0, roll=10.0)

        # 1. Прямое вычисление global_matrix без кэширования
        expected_global = parent.local_matrix @ child.local_matrix @ grandchild.local_matrix
        np.testing.assert_allclose(grandchild.global_matrix, expected_global, atol=1e-12)

        # 2. Обратимость позиций и направлений через convert_to_global_* / convert_to_local_*
        local_pts = np.array([[1.0, -2.5, 3.0], [-4.0, 0.5, -1.5]], dtype=np.float64)
        global_pts = grandchild.convert_to_global_position(local_pts)
        recovered_pts = grandchild.convert_to_local_position(global_pts)
        np.testing.assert_allclose(recovered_pts, local_pts, atol=1e-11)

        local_dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        global_dirs = grandchild.convert_to_global_direction(local_dirs)
        recovered_dirs = grandchild.convert_to_local_direction(global_dirs)
        np.testing.assert_allclose(recovered_dirs, local_dirs, atol=1e-12)

        # 3. Немедленное отражение мутаций родителя в global_matrix потомков и geometry_buffer
        parent.translate(10.0, 0.0, 0.0)
        np.testing.assert_allclose(
            grandchild.global_matrix,
            parent.local_matrix @ child.local_matrix @ grandchild.local_matrix,
            atol=1e-12,
        )

        buf = parent.geometry_buffer
        self.assertEqual(buf.shape[0], 3)
        expected_inv_grandchild = np.linalg.inv(grandchild.global_matrix)
        np.testing.assert_allclose(
            [
                buf[2]["transform"]["translation"]["x"],
                buf[2]["transform"]["translation"]["y"],
                buf[2]["transform"]["translation"]["z"],
            ],
            expected_inv_grandchild[:3, 3],
            atol=1e-11,
        )

    def test_gamma_camera_vm_slot_caching_and_invalidation(self) -> None:
        """
        Проверка O(1) кэширования ссылок на слоты в GammaCameraViewModel и сброса кэша при мутации дерева.
        """
        camera_vm = create_default_gamma_camera_vm(name="CacheTestCam")
        casing = camera_vm.casing_vm
        det_box = camera_vm.detector_box_vm
        collimator = camera_vm.collimator_vm
        crystal = camera_vm.crystal_vm
        glass = camera_vm.glass_backend_vm

        self.assertIsNotNone(casing)
        self.assertIsNotNone(det_box)
        self.assertIsNotNone(collimator)
        self.assertIsNotNone(crystal)
        self.assertIsNotNone(glass)

        # Повторные обращения возвращают закэшированные экземпляры из _slot_vm_cache
        self.assertEqual(len(camera_vm._slot_vm_cache), 5)
        self.assertIs(camera_vm.crystal_vm, crystal)
        self.assertIs(camera_vm.collimator_vm, collimator)

        # При sync_children_from_core кэш инвалидируется и лениво заполняется по мере обращения к слотам
        camera_vm.sync_children_from_core()
        self.assertEqual(len(camera_vm._slot_vm_cache), 0)
        crystal = camera_vm.crystal_vm
        self.assertIsNotNone(crystal)
        self.assertEqual(len(camera_vm._slot_vm_cache), 1)
        casing = camera_vm.casing_vm
        det_box = camera_vm.detector_box_vm
        collimator = camera_vm.collimator_vm
        glass = camera_vm.glass_backend_vm
        self.assertEqual(len(camera_vm._slot_vm_cache), 5)

        # Отсоединение узла из родительского контейнера (parent_vm = None) должно инвалидировать слот в кэше
        assert casing is not None and det_box is not None and crystal is not None and collimator is not None and glass is not None
        det_box.remove_child(crystal)
        self.assertIsNone(camera_vm.crystal_vm)
        self.assertNotIn("crystal", camera_vm._slot_vm_cache)

        det_box.remove_child(collimator)
        self.assertIsNone(camera_vm.collimator_vm)
        self.assertNotIn("collimator", camera_vm._slot_vm_cache)

        # Отсоединение промежуточного контейнера det_box от casing без отсоединения glass от det_box
        # (glass.parent_vm is det_box != None) должно инвалидировать слот glass_backend за счёт _is_descendant_slot_vm!
        casing.remove_child(det_box)
        self.assertIsNotNone(glass.parent_vm)
        self.assertIsNone(camera_vm.glass_backend_vm)
        self.assertNotIn("glass_backend", camera_vm._slot_vm_cache)

    def test_tangential_drag_breaks_feedback_loop_and_preserves_second_camera_roll(self) -> None:
        """
        Проверка разрыва петли обратной связи Gizmo <-> Constraint <-> ProcedureVM:
        - при синхронизации радиуса сохраняется угол roll второй камеры;
        - при касательном перетаскивании камеры на гантри матрица самой камеры не перезаписывается.
        """
        root_core = CompositeNode(name="World")
        scene_vm = SceneViewModel(root_core)
        assert scene_vm.root_vm is not None
        cam_init = create_default_gamma_camera_vm(name="CamInit")
        scene_vm.add_node(scene_vm.root_vm, cam_init)

        procedure = SpectProcedureViewModel(steps=16, gamma_cameras=2, radius=250.0, start_angle=0.0)
        procedure.sync_with_scene(scene_vm)

        gantry_vm = [n for n in scene_vm.all_nodes() if isinstance(n, GantryViewModel)][0]
        cameras = [n for n in scene_vm.all_nodes() if isinstance(n, GammaCameraViewModel)]
        cam1, cam2 = cameras[0], cameras[1]

        # Задаём второй камере ненулевой угол собственного крена roll = 33.0 град
        _, cam2_angle, cam2_z, _ = extract_orbit_parameters(cam2.local_matrix, half_thickness=cam2.half_thickness)
        cam2.local_matrix = compute_orbit_matrix(
            radius=250.0,
            angle=cam2_angle,
            z=cam2_z,
            roll=33.0,
            half_thickness=cam2.half_thickness,
        )

        # Изменяем радиус процедуры -> roll второй камеры обязан сохраниться равным 33.0 град!
        procedure.radius = 300.0
        _, _, _, cam2_roll_after = extract_orbit_parameters(cam2.local_matrix, half_thickness=cam2.half_thickness)
        self.assertAlmostEqual(cam2_roll_after, 33.0, places=5)

        # Подсчитываем число эмиссий local_transform_changed на камерах и гантри при касательном перетаскивании
        cam1_transforms: list[int] = []
        cam2_transforms: list[int] = []
        gantry_transforms: list[int] = []
        cam1.local_transform_changed.connect(lambda: cam1_transforms.append(1))
        cam2.local_transform_changed.connect(lambda: cam2_transforms.append(1))
        gantry_vm.local_transform_changed.connect(lambda: gantry_transforms.append(1))

        constraint = cam1.get_effective_kinematic_constraint()
        gizmo = TransformGizmo(viewport=None, target_node=cam1)
        gizmo.set_constraint(constraint)
        gizmo._active_axis = GizmoAxis.X  # Локальная ось X камеры направлена по касательной к орбите
        gizmo._is_dragging = True
        gizmo._initial_drag_matrix = cam1.local_matrix.copy()

        # Касательное смещение вдоль мирового +Y (по касательной к орбите при угле 0 град)
        changed = gizmo._apply_drag_translation(np.array([0.0, 50.0, 0.0], dtype=np.float64), shift_modifier=False)
        self.assertTrue(changed)
        # Локальные матрицы камер на роторе НЕ перезаписываются (0 сигналов), вращается только гантри (1 сигнал)!
        self.assertEqual(len(cam1_transforms), 0)
        self.assertEqual(len(cam2_transforms), 0)
        self.assertEqual(len(gantry_transforms), 1)

    def test_multi_step_tangential_drag_and_shift_individual_radial_move(self) -> None:
        """
        Проверка отсутствия квадратичного накопления угла гантри при многошаговом касательном перетаскивании
        и корректности индивидуального радиального сдвига камеры при зажатом Shift.
        """
        root_core = CompositeNode(name="World")
        scene_vm = SceneViewModel(root_core)
        assert scene_vm.root_vm is not None
        cam_init = create_default_gamma_camera_vm(name="CamInit")
        scene_vm.add_node(scene_vm.root_vm, cam_init)

        procedure = SpectProcedureViewModel(steps=16, gamma_cameras=2, radius=250.0, start_angle=30.0)
        procedure.sync_with_scene(scene_vm)

        gantry_vm = [n for n in scene_vm.all_nodes() if isinstance(n, GantryViewModel)][0]
        cameras = [n for n in scene_vm.all_nodes() if isinstance(n, GammaCameraViewModel)]
        cam1, cam2 = cameras[0], cameras[1]

        constraint = cam1.get_effective_kinematic_constraint()
        assert constraint is not None
        gizmo = TransformGizmo(viewport=None, target_node=cam1)
        gizmo.set_constraint(constraint)

        # 1. Многошаговое тангенциальное перетаскивание от начального угла станины 30°
        gizmo._active_axis = GizmoAxis.X
        gizmo._is_dragging = True
        gizmo._initial_drag_matrix = cam1.local_matrix.copy()

        effective_radius = float(np.hypot(cam1.local_matrix[0, 3], cam1.local_matrix[1, 3]))
        # В мировых координатах касательный вектор для камеры 1 при повороте станины на 30°: (-sin(30°), cos(30°), 0)
        world_tangent = np.array([-np.sin(np.radians(30.0)), np.cos(np.radians(30.0)), 0.0], dtype=np.float64)

        # Шаг 1: кумулятивный сдвиг на +5° -> угол станины должен стать 35°
        changed_1 = gizmo._apply_drag_translation(
            world_tangent * (effective_radius * np.radians(5.0)),
            shift_modifier=False,
        )
        self.assertTrue(changed_1)
        self.assertAlmostEqual(gantry_vm.gantry_angle, 35.0, places=5)

        # Суб-шаг (+6° суммарно, снап к тем же 35°) -> Snap Gate обязан вернуть False!
        changed_sub = gizmo._apply_drag_translation(
            world_tangent * (effective_radius * np.radians(6.0)),
            shift_modifier=False,
        )
        self.assertFalse(changed_sub)
        self.assertAlmostEqual(gantry_vm.gantry_angle, 35.0, places=5)

        # Шаг 2: кумулятивный сдвиг на +10° от старта -> угол станины 40° (а не 35° + 10° = 45°!)
        changed_2 = gizmo._apply_drag_translation(
            world_tangent * (effective_radius * np.radians(10.0)),
            shift_modifier=False,
        )
        self.assertTrue(changed_2)
        self.assertAlmostEqual(gantry_vm.gantry_angle, 40.0, places=5)

        # Шаг 3: кумулятивный сдвиг на +15° от старта -> угол станины 45°
        changed_3 = gizmo._apply_drag_translation(
            world_tangent * (effective_radius * np.radians(15.0)),
            shift_modifier=False,
        )
        self.assertTrue(changed_3)
        self.assertAlmostEqual(gantry_vm.gantry_angle, 45.0, places=5)
        assert gizmo._last_changed_data is not None
        constraint.on_transform_committed(cam1, gizmo._last_changed_data)
        gizmo._is_dragging = False
        self.assertAlmostEqual(procedure.start_angle, 45.0, places=5)

        # 2. Индивидуальный радиальный сдвиг камеры 1 с зажатым Shift (shift_modifier=True):
        # камера 2 и procedure.radius должны остаться на 250.0 мм!
        gizmo._active_axis = GizmoAxis.Z
        gizmo._is_dragging = True
        gizmo._initial_drag_matrix = cam1.local_matrix.copy()
        world_radial = np.array([np.cos(np.radians(45.0)), np.sin(np.radians(45.0)), 0.0], dtype=np.float64)

        changed_shift = gizmo._apply_drag_translation(world_radial * 37.5, shift_modifier=True)
        self.assertTrue(changed_shift)
        cam1_radius, _, _, _ = extract_orbit_parameters(cam1.local_matrix, half_thickness=cam1.half_thickness)
        cam2_radius, _, _, _ = extract_orbit_parameters(cam2.local_matrix, half_thickness=cam2.half_thickness)
        self.assertAlmostEqual(cam1_radius, 290.0, places=4)
        self.assertAlmostEqual(cam2_radius, 250.0, places=4)
        self.assertAlmostEqual(procedure.radius, 250.0, places=4)


if __name__ == "__main__":
    unittest.main()

