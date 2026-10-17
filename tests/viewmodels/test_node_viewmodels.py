import unittest

from core.config.models import StepAndShootProtocolConfig
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import SpatialNode, CompositeNode
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel
from gui.viewmodels.procedure_viewmodel import (
    CustomSweepProcedureViewModel,
    SpectProcedureViewModel,
    StepAndShootProcedureViewModel,
    procedure_from_config,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel


class TestStage2ViewModels(unittest.TestCase):
    def test_core_and_gui_field_reactivity(self):
        """Проверка дескрипторов core_field и gui_field и двусторонней синхронизации."""
        core_node = SpatialNode()
        vm = NodeViewModel(core_node)

        changed_log = []
        vm.property_changed.connect(lambda name, val: changed_log.append((name, val)))

        # 1. core_field
        vm.name = "PhantomNode"
        self.assertEqual(core_node.name, "PhantomNode")
        self.assertEqual(len(changed_log), 1)
        self.assertEqual(changed_log[0], ("name", "PhantomNode"))

        # 2. gui_field
        vm.visible = False
        self.assertFalse(vm.visible)
        self.assertEqual(len(changed_log), 2)
        self.assertEqual(changed_log[1], ("visible", False))

    def test_node_viewmodel_transformations(self):
        """Проверка трансформаций через ViewModel и сигналов."""
        core_node = SpatialNode()
        vm = NodeViewModel(core_node)

        matrix_log = []
        vm.transform_changed.connect(lambda: matrix_log.append(True))

        vm.translate(x=10.0, y=20.0, z=30.0)
        self.assertEqual(len(matrix_log), 1)
        self.assertAlmostEqual(core_node.local_matrix[0, 3], 10.0)
        self.assertAlmostEqual(core_node.local_matrix[1, 3], 20.0)
        self.assertAlmostEqual(core_node.local_matrix[2, 3], 30.0)

    def test_volume_viewmodel_properties(self):
        """Проверка VolumeViewModel: изменение геометрии и материалов."""
        mat = Material(name="Lead")
        geo = Box(10.0, 20.0, 30.0)
        vol = Volume(geometry=geo, material=mat, name="Collimator")
        vm = VolumeViewModel(vol)

        self.assertEqual(vm.material_name, "Lead")
        vm.size = [50.0, 60.0, 70.0]
        self.assertAlmostEqual(vol.geometry.size[0], 50.0)
        self.assertAlmostEqual(vol.geometry.size[1], 60.0)
        self.assertAlmostEqual(vol.geometry.size[2], 70.0)

    def test_scene_viewmodel_hierarchy(self):
        """Проверка иерархии сцены, выбора, добавления и удаления узлов."""
        root = CompositeNode()
        child1 = SpatialNode()
        child1.name = "Child1"
        root.add_child(child1)

        scene_vm = SceneViewModel(root)
        self.assertIsNotNone(scene_vm.root_vm)
        self.assertEqual(len(scene_vm.root_vm.children), 1)

        # Выбор узла
        selected = []
        scene_vm.node_selected.connect(lambda node: selected.append(node.name))
        scene_vm.select_node(scene_vm.root_vm.children[0])
        self.assertEqual(selected, ["Child1"])

        # Поиск узла
        found = scene_vm.find_by_name("Child1")
        self.assertIsNotNone(found)
        self.assertEqual(found.core_node, child1)

        # Добавление нового узла
        child2_core = SpatialNode()
        child2_core.name = "Child2"
        child2_vm = NodeViewModel(child2_core)
        scene_vm.add_node(scene_vm.root_vm, child2_vm)
        self.assertEqual(len(scene_vm.root_vm.children), 2)
        self.assertIn(child2_core, root.childs)

        # Удаление узла
        scene_vm.remove_node(child2_vm)
        self.assertEqual(len(scene_vm.root_vm.children), 1)
        self.assertNotIn(child2_core, root.childs)

    def test_spect_procedure_steps_and_total_projections(self):
        """Проверка полей steps, total_projections и синхронизации с конфигурацией."""
        proc = SpectProcedureViewModel()
        self.assertEqual(proc.steps, 32)
        self.assertEqual(proc.gamma_cameras, 2)
        self.assertEqual(proc.total_projections, 64)
        with self.assertRaises(AttributeError):
            _ = proc.views

        # Изменение steps
        proc.steps = 16
        self.assertEqual(proc.steps, 16)
        self.assertEqual(proc.total_projections, 32)

        # Изменение gamma_cameras
        proc.gamma_cameras = 4
        self.assertEqual(proc.steps, 16)
        self.assertEqual(proc.total_projections, 64)

        # Конвертация в StepAndShootProtocolConfig
        cfg = proc.to_config()
        self.assertIsInstance(cfg, StepAndShootProtocolConfig)
        self.assertEqual(cfg.steps, 16)

        # Восстановление из StepAndShootProtocolConfig
        loaded_vm = procedure_from_config(cfg)
        self.assertIsInstance(loaded_vm, SpectProcedureViewModel)
        self.assertEqual(loaded_vm.steps, 16)

    def test_procedure_start_stop_time_and_bindings(self):
        """Проверка свойств start_time/stop_time в CustomSweepProcedureViewModel и синхронизации привязок."""
        sweep_vm = CustomSweepProcedureViewModel(
            zipped_variables={"t_start": [0.0, 10.0], "t_stop": [10.0, 30.0]},
            start_time=2.5,
            stop_time=12.5,
        )
        self.assertAlmostEqual(sweep_vm.start_time, 2.5)
        self.assertAlmostEqual(sweep_vm.stop_time, 12.5)

        sweep_vm.bind_property("simulation_manager", "start_time", "t_start")
        sweep_vm.bind_property("simulation_manager", "stop_time", "t_stop")
        self.assertEqual(sweep_vm.bindings.get("start_time"), "t_start")
        self.assertEqual(sweep_vm.bindings.get("stop_time"), "t_stop")

        sweep_vm.remove_variable("t_start")
        self.assertNotIn("start_time", sweep_vm.bindings)
        self.assertNotIn("start_time", sweep_vm.get_property_bindings("simulation_manager"))
        self.assertEqual(sweep_vm.bindings.get("stop_time"), "t_stop")

        sas_vm = StepAndShootProcedureViewModel(steps=4, time_per_step=5.0, initial_time=10.0, sequential_time=True)
        self.assertAlmostEqual(sas_vm.start_time, 10.0)
        self.assertAlmostEqual(sas_vm.stop_time, 15.0)


if __name__ == '__main__':
    unittest.main()
