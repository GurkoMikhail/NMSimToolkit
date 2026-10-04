"""
Модульные тесты для интеграции DirectParallelCollimator в GUI:
- DirectParallelCollimatorViewModel и кинематические констрейнты дочерних объемов;
- Интеграция с GammaCameraViewModel (управление размерами и толщиной);
- Механизм замены узлов в сцене (SceneViewModel.replace_node);
- Аппаратный 3D-рендерер каналов CollimatorHoleRenderer.
"""

import sys
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication(sys.argv)

import hepunits as units
import settings.database_setting as database_setting
from core.geometry.direct_collimators import (
    DirectParallelCollimator,
    CollimatorHoleShape,
)
from core.geometry.geometries import Box, PeriodicHexPrism
from core.geometry.volumes import Volume
from core.scene.gamma_camera_node import GammaCameraNode
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from gui.viewmodels import create_default_gamma_camera_vm
from gui.viewmodels.nodes.collimator_vm import CollimatorViewModel
from gui.viewmodels.nodes.factory import create_node_viewmodel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.scene_viewmodel import SceneViewModel
from core.geometry.parametric_collimators import ParametricParallelCollimator
from gui.views.property_inspector import PropertyInspector
from gui.viewport_3d.collimator_hole_renderer import CollimatorHoleRenderer
from gui.viewport_3d.transform_gizmo import GizmoMode


class TestDirectCollimatorGUI(unittest.TestCase):
    """
    Тестирование компонентов пользовательского интерфейса, моделей представления
    и рендерера для DirectParallelCollimator.
    """

    def setUp(self) -> None:
        self.lead_material = database_setting.material_database["Pb"]
        self.vacuum_material = database_setting.material_database["Vacuum"]
        self.collimator = DirectParallelCollimator(
            size=[400.0, 400.0, 35.0],
            hole_diameter=1.5,
            septa=0.2,
            material=self.lead_material,
            hole_material=self.vacuum_material,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="MyDirectCollimator",
        )

    def test_direct_collimator_viewmodel_creation_and_properties(self) -> None:
        """Проверка инициализации DirectParallelCollimatorViewModel через фабрику и её свойств."""
        collimator_vm = create_node_viewmodel(self.collimator)
        self.assertIsInstance(collimator_vm, CollimatorViewModel)
        self.assertEqual(collimator_vm.name, "MyDirectCollimator")
        self.assertAlmostEqual(collimator_vm.hole_diameter, 1.5)
        self.assertAlmostEqual(collimator_vm.septa, 0.2)
        self.assertEqual(collimator_vm.hole_shape, CollimatorHoleShape.HEXAGONAL)
        self.assertEqual(collimator_vm.material_name, "Pb")
        self.assertEqual(collimator_vm.hole_material_name, "Vacuum")

        # Реактивное обновление параметров
        collimator_vm.hole_diameter = 2.0
        self.assertAlmostEqual(self.collimator.hole_diameter, 2.0)
        collimator_vm.septa = 0.3
        self.assertAlmostEqual(self.collimator.septa, 0.3)

    def test_fixed_constraints_on_child_volumes(self) -> None:
        """
        Проверка назначения FixedSubcomponentKinematicConstraint дочерним объемам:
        пользователь не может сместить channels относительно lead_body через манипулятор.
        """
        collimator_vm = create_node_viewmodel(self.collimator)
        self.assertIsInstance(collimator_vm, CollimatorViewModel)

        # У самого коллиматора нет фиксирующего констрейнта (если он на верхнем уровне)
        self.assertIsNone(collimator_vm.kinematic_constraint)

        # Все дочерние объемы (корпус и каналы) заблокированы
        self.assertGreaterEqual(len(collimator_vm.children), 1)
        for child_vm in collimator_vm.children:
            constraint = child_vm.get_effective_kinematic_constraint()
            self.assertIsNotNone(constraint)
            # Все степени свободы перемещения, вращения и масштабирования заблокированы
            self.assertEqual(len(constraint.get_allowed_axes(GizmoMode.TRANSLATE)), 0)
            self.assertEqual(len(constraint.get_allowed_axes(GizmoMode.ROTATE)), 0)
            self.assertFalse(constraint.is_scale_allowed())

    def test_gamma_camera_collimator_management(self) -> None:
        """
        Проверка управления DirectParallelCollimator из GammaCameraViewModel:
        гамма-камера задает размеры active detector_size и collimator_thickness.
        """
        camera_vm = create_default_gamma_camera_vm(name="SpectCamera")
        scene_vm = SceneViewModel(root_core_node=Volume(name="World", geometry=Box(1000.0, 1000.0, 1000.0), material=self.lead_material))
        scene_vm.add_node(scene_vm.root_vm, camera_vm)

        # Замещаем оригинальный параметрический коллиматор на DirectParallelCollimator
        old_collimator_vm = camera_vm.collimator_vm
        self.assertIsNotNone(old_collimator_vm)

        new_direct_collimator = DirectParallelCollimator(
            size=[camera_vm.detector_size[0], camera_vm.detector_size[1], camera_vm.collimator_thickness],
            hole_diameter=1.4,
            septa=0.18,
            material=self.lead_material,
            hole_material=self.vacuum_material,
            name="DirectCollimatorSwapped",
        )
        new_collimator_vm = create_node_viewmodel(new_direct_collimator)

        scene_vm.replace_node(old_collimator_vm, new_collimator_vm)

        # Камера теперь видит новый DirectParallelCollimatorViewModel в своем слоте
        self.assertIs(camera_vm.collimator_vm, new_collimator_vm)
        self.assertEqual(camera_vm.slots.collimator, "DirectCollimatorSwapped")

        # Изменение detector_size в гамма-камере должно актуализировать размеры коллиматора
        camera_vm.detector_size = (520.0, 390.0)
        self.assertAlmostEqual(new_collimator_vm.size[0], 520.0)
        self.assertAlmostEqual(new_collimator_vm.size[1], 390.0)
        self.assertAlmostEqual(new_direct_collimator.size[0], 520.0)
        self.assertAlmostEqual(new_direct_collimator.size[1], 390.0)

        # Изменение collimator_thickness в гамма-камере должно актуализировать толщину коллиматора
        camera_vm.collimator_thickness = 42.0
        self.assertAlmostEqual(new_collimator_vm.size[2], 42.0)
        self.assertAlmostEqual(new_direct_collimator.size[2], 42.0)

    def test_collimator_hole_renderer_instancing(self) -> None:
        """Проверка работы CollimatorHoleRenderer и создания VTK актора с GPU-инстансингом."""
        class MockViewport:
            def __init__(self) -> None:
                self.actors = {}
            def add_actor(self, name: str, actor: Any) -> Any:
                self.actors[name] = actor
                return actor
            def remove_actor(self, name: str) -> None:
                self.actors.pop(name, None)
            def update_actor_transform(self, name: str, matrix: np.ndarray) -> bool:
                return True

        mock_viewport = MockViewport()
        renderer = CollimatorHoleRenderer(viewport=mock_viewport)
        prism_geometry = PeriodicHexPrism(
            size=[100.0, 100.0, 20.0],
            hole_diameter=2.0,
            septa=0.4,
        )

        actor = renderer.render_holes(
            actor_name="test_holes",
            geometry=prism_geometry,
            global_matrix=np.eye(4),
            hole_color=(0.8, 0.8, 0.8),
            hole_opacity=0.7,
        )

        self.assertIsNotNone(actor)
        self.assertIn("test_holes", renderer.actors)
        self.assertIn("test_holes", mock_viewport.actors)

        # Проверка кэширования геометрии и обновления
        actor_cached = renderer.render_holes(
            actor_name="test_holes",
            geometry=prism_geometry,
            global_matrix=np.eye(4),
            hole_color=(0.9, 0.9, 0.9),
            hole_opacity=0.8,
        )
        self.assertIs(actor_cached, actor)

        # Удаление
        renderer.remove_actor("test_holes")
        self.assertNotIn("test_holes", renderer.actors)
        self.assertNotIn("test_holes", mock_viewport.actors)

    def test_collimator_viewmodel_for_parametric(self) -> None:
        """Проверка работы CollimatorViewModel для параметрического коллиматора ядра."""
        param_collimator = ParametricParallelCollimator(
            size=[380.0, 380.0, 32.0],
            hole_diameter=1.6,
            septa=0.22,
            material=self.lead_material,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="ParamCollimator",
        )
        col_vm = create_node_viewmodel(param_collimator)
        self.assertIsInstance(col_vm, CollimatorViewModel)
        self.assertEqual(col_vm.collimator_kind, "parametric")
        self.assertEqual(col_vm.collimator_type, "Параметрический (RayCasting)")
        self.assertEqual(col_vm.hole_shape, CollimatorHoleShape.HEXAGONAL)
        self.assertFalse(hasattr(col_vm, 'is_sensitive_detector'))
        self.assertEqual(col_vm.material_name, "Pb")
        self.assertIsNone(col_vm.hole_material_name)

        # Реактивная смена формы
        col_vm.hole_shape = CollimatorHoleShape.SQUARE
        self.assertEqual(param_collimator.hole_shape, CollimatorHoleShape.SQUARE)

        # Смена размеров отверстий и септ
        col_vm.hole_width = 2.0
        self.assertAlmostEqual(param_collimator.hole_diameter, 2.0)
        self.assertAlmostEqual(col_vm.hole_diameter, 2.0)

        col_vm.septa = 0.3
        self.assertAlmostEqual(param_collimator.septa, 0.3)
        self.assertAlmostEqual(col_vm.septa, 0.3)

    def test_collimator_property_inspector_integration(self) -> None:
        """Проверка интерактивного переключения типов коллиматора через PropertyInspector."""
        inspector = PropertyInspector()
        world_volume = Volume(name="World", geometry=Box(1000.0, 1000.0, 1000.0), material=self.lead_material)
        scene_vm = SceneViewModel(root_core_node=world_volume)
        inspector.set_scene_viewmodel(scene_vm)

        direct_col = DirectParallelCollimator(
            size=[400.0, 400.0, 35.0],
            hole_diameter=1.5,
            septa=0.2,
            material=self.lead_material,
            hole_material=None,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="CollimatorToSwap",
        )
        col_vm = create_node_viewmodel(direct_col)
        scene_vm.add_node(scene_vm.root_vm, col_vm)

        inspector.set_target_viewmodel(col_vm)

        # В инспекторе выбран детерминированный коллиматор
        self.assertEqual(inspector.combo_collimator_type.currentData(), "direct")
        self.assertFalse(inspector.collimator_group.isHidden())
        self.assertFalse(inspector.combo_collimator_shape.isHidden())
        self.assertFalse(inspector.combo_hole_material.isHidden())

        # Переключаем тип коллиматора на параметрический
        param_index = inspector.combo_collimator_type.findData("parametric")
        self.assertGreaterEqual(param_index, 0)
        inspector.combo_collimator_type.setCurrentIndex(param_index)

        # Проверяем, что узел в сцене заменен на параметрический
        swapped_vm = scene_vm.find_by_name("CollimatorToSwap")
        self.assertIsNotNone(swapped_vm)
        self.assertIsInstance(swapped_vm, CollimatorViewModel)
        self.assertEqual(swapped_vm.collimator_kind, "parametric")
        self.assertIsInstance(swapped_vm.core_node, ParametricParallelCollimator)
        # Геометрия сохранена
        self.assertAlmostEqual(swapped_vm.hole_diameter, 1.5)
        self.assertAlmostEqual(swapped_vm.septa, 0.2)
        self.assertEqual(swapped_vm.hole_shape, CollimatorHoleShape.HEXAGONAL)


if __name__ == '__main__':
    unittest.main()

