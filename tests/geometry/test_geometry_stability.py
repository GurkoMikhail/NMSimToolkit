import threading
import unittest
import numpy as np

import hepunits as units
from core.scene.nodes import SpatialNode, CompositeNode
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.direct_collimators import CollimatorHoleShape
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.source.sources import Source
from core.transport.simulation_managers import SimulationManager, SimulationState
from core.config.builder import SceneBuilder
from core.config.models import SimulationConfig, ParametricParallelCollimatorConfig
import settings.database_setting as settings


class TestCoreHotfixStability(unittest.TestCase):
    """
    Набор регрессионных тестов для проверки критических исправлений ядра (hotfix/core-stability).
    """

    def test_simulation_manager_in_worker_thread(self):
        """
        Проверка: SimulationManager не должен падать с ValueError (signal only works in main thread)
        при инициализации и запуске внутри фонового рабочего потока.
        """
        worker_error = None

        def run_in_thread():
            nonlocal worker_error
            try:
                root = CompositeNode()
                mgr = SimulationManager(scene=root, particles_number=10)
                self.assertEqual(mgr.state, SimulationState.IDLE)
                mgr.pause()
                self.assertEqual(mgr.state, SimulationState.PAUSED)
                mgr.resume()
                self.assertEqual(mgr.state, SimulationState.RUNNING)
                mgr.stop()
                self.assertEqual(mgr.state, SimulationState.STOPPED)
            except Exception as e:
                worker_error = e

        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join(timeout=5.0)

        self.assertIsNone(worker_error, f"Ошибка при создании SimulationManager в потоке: {worker_error}")

    def test_source_zero_activity_and_normalization(self):
        """
        Проверка: Источник излучения с нулевой активностью должен выбрасывать ValueError (DbC-контракт),
        а для ненулевого распределения вероятности в voxel_probabilities должны строго суммироваться к 1.0.
        """
        # 1. Нулевое распределение
        zero_dist = np.zeros((4, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            Source(distribution=zero_dist, voxel_size=2.0 * units.mm)

        # 2. Неоднородное распределение и динамическое обновление
        dist = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
        src2 = Source(distribution=dist, voxel_size=1.0 * units.mm)
        self.assertAlmostEqual(float(np.sum(src2.voxel_probabilities)), 1.0, places=6)

        # Проверка сеттера voxel_size
        src2.voxel_size = 2.5 * units.mm
        self.assertAlmostEqual(float(src2.voxel_size), 2.5 * units.mm)
        self.assertAlmostEqual(float(np.sum(src2.voxel_probabilities)), 1.0, places=6)

    def test_scene_builder_square_collimator(self):
        """
        Проверка: SceneBuilder должен корректно создавать ParametricParallelCollimator
        с формой каналов SQUARE и параметром septa.
        """
        builder = SceneBuilder()
        cfg = ParametricParallelCollimatorConfig(
            name="test_collimator",
            size=[100.0, 100.0, 40.0],
            hole_diameter=1.5,
            septa=0.2,
            material="Pb",
            hole_shape="square",
        )
        collimator = builder.build_scene(cfg)
        self.assertIsInstance(collimator, ParametricParallelCollimator)
        self.assertEqual(collimator.hole_shape, CollimatorHoleShape.SQUARE)
        self.assertAlmostEqual(collimator.size[2], 40.0)
        self.assertAlmostEqual(collimator.septa, 0.2)
        self.assertEqual(collimator.name, "test_collimator")

    def test_material_aliases_in_builder(self):
        """
        Проверка: SceneBuilder должен корректно разрешать канонические материалы ('Water, Liquid', 'Pb')
        и выбрасывать ValueError для неизвестных.
        """
        builder = SceneBuilder()
        pb = builder._get_material("Pb")
        self.assertEqual(pb.name, "Pb")

        water = builder._get_material("Water, Liquid")
        self.assertEqual(water.name, "Water, Liquid")
        
        with self.assertRaises(ValueError):
            builder._get_material("UnknownMaterial")

    def test_composite_node_add_and_remove_child(self):
        """
        Проверка: CompositeNode.add_child не должен дублировать дочерние узлы,
        а remove_child должен корректно сбрасывать ссылку parent и инвалидировать матрицы.
        """
        parent = CompositeNode(name="Parent")
        child = SpatialNode(name="Child")

        parent.add_child(child)
        self.assertEqual(len(parent.childs), 1)
        self.assertIs(child.parent, parent)

        # Повторное добавление того же узла не должно дублировать его
        parent.add_child(child)
        self.assertEqual(len(parent.childs), 1)

        # Удаление узла
        parent.remove_child(child)
        self.assertEqual(len(parent.childs), 0)
        self.assertIsNone(child.parent)

    def test_woodcock_cfunc_invalidation(self):
        """
        Проверка: При изменении параметров коллиматора, его материала или родительского объема
        кэш _cfunc должен сбрасываться и перекомпилироваться с актуальными ID материалов.
        """
        lead = settings.material_database["Pb"]
        water = settings.material_database["Water, Liquid"]
        vacuum = settings.material_database["Vacuum"]
        collimator = ParametricParallelCollimator(
            size=np.array([100.0, 100.0, 40.0]),
            hole_diameter=2.0,
            septa=0.5,
            material=lead,
            hole_shape=CollimatorHoleShape.SQUARE,
        )
        # Получаем скомпилированную функцию (без родителя отверстия заполнены Vacuum)
        cfunc_ptr = collimator.material_cfunc
        self.assertIsNotNone(cfunc_ptr)
        self.assertIsNotNone(collimator._cfunc)
        self.assertEqual(int(collimator._cfunc(0.0, 0.0, 0.0)), int(vacuum.ID))
        self.assertEqual(int(collimator._cfunc(1.1, 0.0, 0.0)), int(lead.ID))

        # Изменяем диаметр отверстия через сеттер
        collimator.hole_diameter = 2.2
        self.assertIsNone(collimator._cfunc, "Кэш _cfunc должен сбрасываться при изменении параметров")

        # При повторном запросе должна скомпилироваться новая функция
        new_cfunc_ptr = collimator.material_cfunc
        self.assertIsNotNone(new_cfunc_ptr)

        # Прямая замена материала корпуса через collimator.material сбрасывает _cfunc
        sodium_iodide = settings.material_database["Sodium Iodide"]
        collimator.material = sodium_iodide
        self.assertIsNone(collimator._cfunc, "Кэш _cfunc должен сбрасываться при прямой смене material")
        _ = collimator.material_cfunc
        self.assertEqual(int(collimator._cfunc(1.2, 0.0, 0.0)), int(sodium_iodide.ID))

        # Добавление коллиматора в родительский Volume сбрасывает _cfunc и обновляет материал отверстий
        parent_box = Volume(geometry=Box(200.0, 200.0, 200.0), material=water, name="ParentBox")
        parent_box.add_child(collimator)
        self.assertIsNone(collimator._cfunc, "Кэш _cfunc должен сбрасываться при изменении parent")
        _ = collimator.material_cfunc
        self.assertEqual(int(collimator._cfunc(0.0, 0.0, 0.0)), int(water.ID))

        # Смена материала уже привязанного родительского Volume актуализирует @cfunc при чтении material_cfunc
        parent_box.material = lead
        _ = collimator.material_cfunc
        self.assertEqual(int(collimator._cfunc(0.0, 0.0, 0.0)), int(lead.ID))

    def test_volume_cascade_geometry_invalidation(self):
        """
        Проверка: Изменение размера дочернего Volume должно сразу отражаться в geometry_buffer родителя.
        """
        vac = settings.material_database["Vacuum"]
        parent_vol = Volume(geometry=Box(200.0, 200.0, 200.0), material=vac, name="ParentVolume")
        child_vol = Volume(geometry=Box(50.0, 50.0, 50.0), material=vac, name="ChildVolume")
        parent_vol.add_child(child_vol)

        # Вызываем компиляцию буфера геометрии
        buf = parent_vol.geometry_buffer
        np.testing.assert_allclose(
            [buf[1]["shape_data"]["param_0"], buf[1]["shape_data"]["param_1"], buf[1]["shape_data"]["param_2"]],
            [25.0, 25.0, 25.0],
        )

        # Меняем размер дочернего объема
        child_vol.size = np.array([80.0, 80.0, 80.0])

        # Буфер родителя сразу отражает актуальные полуразмеры дочернего объема
        updated_buf = parent_vol.geometry_buffer
        np.testing.assert_allclose(
            [
                updated_buf[1]["shape_data"]["param_0"],
                updated_buf[1]["shape_data"]["param_1"],
                updated_buf[1]["shape_data"]["param_2"],
            ],
            [40.0, 40.0, 40.0],
        )


if __name__ == '__main__':
    unittest.main()
