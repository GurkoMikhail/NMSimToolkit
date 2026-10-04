"""
Модульные тесты для универсального параметрического коллиматора (ParametricParallelCollimator):
- Поддержка форм каналов CollimatorHoleShape (HEXAGONAL, SQUARE, ROUND);
- Согласованность свойств hole_diameter, hole_width, septa и hole_shape;
- Динамический пересчет констант решетки и инвалидация кэша геометрии;
- Динамическое наследование материала каналов от родительского объема;
- Компиляция Numba-кернелов и векторизованная трассировка геометрии;
- Сериализация и десериализация сцены с ParametricParallelCollimator.
"""

import unittest
import numpy as np
import hepunits as units

import settings.database_setting as database_setting
from core.geometry.direct_collimators import (
    CollimatorHoleShape,
    DirectParallelCollimator,
)
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import (
    ParametricParallelCollimator,
)
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.config.models import ParametricParallelCollimatorConfig
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter


class TestParametricCollimator(unittest.TestCase):
    """
    Тестирование универсального ParametricParallelCollimator и динамического наследования материалов.
    """

    def setUp(self) -> None:
        self.lead_material = database_setting.material_database["Pb"]
        self.vacuum_material = database_setting.material_database["Vacuum"]
        self.air_material = database_setting.material_database.get("Air, Dry (near sea level)", Material(name="Air"))
        self.water_material = database_setting.material_database.get("Water, Liquid", Material(name="Water"))

    def test_parametric_collimator_shapes(self) -> None:
        """Проверка инициализации параметрического коллиматора с различными формами каналов."""
        # Гексагональный (по умолчанию)
        hex_collimator = ParametricParallelCollimator(
            size=[400.0, 400.0, 30.0],
            hole_diameter=1.5,
            septa=0.2,
            material=self.lead_material,
        )
        self.assertEqual(hex_collimator.hole_shape, CollimatorHoleShape.HEXAGONAL)
        self.assertAlmostEqual(hex_collimator.hole_diameter, 1.5)
        self.assertAlmostEqual(hex_collimator.hole_width, 1.5)
        self.assertAlmostEqual(hex_collimator.septa, 0.2)

        # Квадратный
        square_collimator = ParametricParallelCollimator(
            size=[400.0, 400.0, 30.0],
            hole_diameter=1.8,
            septa=0.25,
            material=self.lead_material,
            hole_shape=CollimatorHoleShape.SQUARE,
        )
        self.assertEqual(square_collimator.hole_shape, CollimatorHoleShape.SQUARE)
        self.assertAlmostEqual(square_collimator.hole_width, 1.8)
        self.assertAlmostEqual(square_collimator.hole_diameter, 1.8)

        # Круглый (NotImplementedError)
        with self.assertRaises(NotImplementedError):
            ParametricParallelCollimator(
                size=[400.0, 400.0, 30.0],
                hole_diameter=1.5,
                septa=0.2,
                hole_shape=CollimatorHoleShape.ROUND,
            )

        # Невалидная форма
        with self.assertRaises(ValueError):
            ParametricParallelCollimator(
                size=[400.0, 400.0, 30.0],
                hole_diameter=1.5,
                septa=0.2,
                hole_shape="invalid_shape",
            )

    def test_parametric_collimator_property_setters_and_sync(self) -> None:
        """Проверка сеттеров параметров, согласованности hole_diameter/hole_width и пересчета."""
        collimator = ParametricParallelCollimator(
            size=[400.0, 400.0, 30.0],
            hole_diameter=1.5,
            septa=0.2,
            material=self.lead_material,
        )

        # Изменение hole_diameter
        collimator.hole_diameter = 2.0
        self.assertAlmostEqual(collimator.hole_diameter, 2.0)
        self.assertAlmostEqual(collimator.hole_width, 2.0)

        # Изменение hole_width
        collimator.hole_width = 2.4
        self.assertAlmostEqual(collimator.hole_diameter, 2.4)
        self.assertAlmostEqual(collimator.hole_width, 2.4)

        # Изменение septa
        collimator.septa = 0.35
        self.assertAlmostEqual(collimator.septa, 0.35)

        # Смена формы на SQUARE
        collimator.hole_shape = CollimatorHoleShape.SQUARE
        self.assertEqual(collimator.hole_shape, CollimatorHoleShape.SQUARE)
        self.assertAlmostEqual(collimator._square_period, 2.75)

        # Валидация недопустимых значений
        with self.assertRaises(ValueError):
            collimator.hole_diameter = -1.0
        with self.assertRaises(ValueError):
            collimator.septa = 0.0

    def test_parametric_collimator_numba_compilation_and_tracing(self) -> None:
        """Проверка компиляции кернелов Numba и функции _parametric_function для HEXAGONAL и SQUARE."""
        collimator = ParametricParallelCollimator(
            size=[100.0, 100.0, 20.0],
            hole_diameter=2.0,
            septa=0.5,
            material=self.lead_material,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
        )

        cfunc_hex = collimator._compile_cfunc()
        self.assertTrue(callable(cfunc_hex))

        # Тест векторизованной трассировки точек
        test_points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
        ], dtype=float)
        in_channels, effective_mat = collimator._parametric_function(test_points)
        self.assertEqual(len(in_channels), 2)
        self.assertIsInstance(effective_mat, Material)

        # Переключение на квадратные каналы
        collimator.hole_shape = CollimatorHoleShape.SQUARE
        cfunc_square = collimator._compile_cfunc()
        self.assertTrue(callable(cfunc_square))
        in_channels_sq, _ = collimator._parametric_function(test_points)
        self.assertEqual(len(in_channels_sq), 2)

    def test_parametric_collimator_material_inheritance(self) -> None:
        """Проверка динамического разрешения материала каналов от родительского объема."""
        collimator = ParametricParallelCollimator(
            size=[200.0, 200.0, 30.0],
            hole_diameter=1.5,
            septa=0.2,
            material=self.lead_material,
        )
        # Без родителя fallback на Vacuum
        self.assertEqual(collimator._resolve_hole_material().name, "Vacuum")

        # Добавление в родительский Volume с Air
        parent_box = Volume(
            name="DetectorBox",
            geometry=Box(300.0, 300.0, 100.0),
            material=self.air_material,
        )
        parent_box.add_child(collimator)

        # Теперь материал каналов динамически берется от родителя
        self.assertEqual(collimator._resolve_hole_material().name, self.air_material.name)

    def test_direct_collimator_material_inheritance_and_set_parent(self) -> None:
        """Проверка динамического наследования материалов каналов в DirectParallelCollimator."""
        direct_col = DirectParallelCollimator(
            size=[300.0, 300.0, 35.0],
            hole_diameter=1.6,
            septa=0.22,
            material=self.lead_material,
            hole_material=None,  # Наследование от родителя
        )
        # Без родителя каналы инициализируются Vacuum
        self.assertEqual(direct_col.hole_material.name, "Vacuum")
        self.assertIsNone(direct_col.explicit_hole_material)

        # Назначение родительского объема через set_parent
        parent_container = Volume(
            name="AirBox",
            geometry=Box(400.0, 400.0, 150.0),
            material=self.air_material,
        )
        direct_col.set_parent(parent_container)

        # Материал каналов автоматически обновился на материал родителя (Air)
        self.assertEqual(direct_col.hole_material.name, self.air_material.name)

        # Явное переопределение материала каналов
        direct_col.hole_material = self.water_material
        self.assertEqual(direct_col.hole_material.name, self.water_material.name)
        self.assertEqual(direct_col.explicit_hole_material.name, self.water_material.name)

        # Сброс в режим наследования передачей None
        direct_col.hole_material = None
        self.assertIsNone(direct_col.explicit_hole_material)
        self.assertEqual(direct_col.hole_material.name, self.air_material.name)

    def test_config_builder_exporter_parametric_collimator(self) -> None:
        """Проверка экспорта и восстановления ParametricParallelCollimator с hole_shape через SceneBuilder/Exporter."""
        collimator = ParametricParallelCollimator(
            size=[350.0, 350.0, 28.0],
            hole_diameter=1.75,
            septa=0.28,
            material=self.lead_material,
            hole_shape=CollimatorHoleShape.SQUARE,
            name="ConfigSquareCollimator",
        )

        exported_config = SceneExporter.export_node(collimator)
        self.assertIsInstance(exported_config, ParametricParallelCollimatorConfig)
        self.assertEqual(exported_config.hole_shape, "square")
        self.assertAlmostEqual(exported_config.hole_diameter, 1.75)
        self.assertAlmostEqual(exported_config.septa, 0.28)

        # Восстановление через SceneBuilder
        builder = SceneBuilder()
        restored_node = builder.build_scene(exported_config)
        self.assertIsInstance(restored_node, ParametricParallelCollimator)
        self.assertEqual(restored_node.hole_shape, CollimatorHoleShape.SQUARE)
        self.assertAlmostEqual(restored_node.hole_diameter, 1.75)
        self.assertAlmostEqual(restored_node.septa, 0.28)

    def test_collimator_hole_renderer_square_prism(self) -> None:
        """Проверка построения прототипа квадратного канала и центров квадратной решетки."""
        from gui.viewport_3d.collimator_hole_renderer import (
            create_hollow_square_prism_prototype,
            generate_square_hole_centers,
            create_hole_prototype,
        )

        # 1. Прототип квадратной призмы
        square_poly = create_hollow_square_prism_prototype(hole_width=2.0, height_z=30.0)
        self.assertEqual(square_poly.GetNumberOfPoints(), 8)
        self.assertEqual(square_poly.GetNumberOfCells(), 4)

        # 2. Фабрика прототипов
        poly_from_factory = create_hole_prototype(
            shape=CollimatorHoleShape.SQUARE,
            hole_diameter=2.0,
            height_z=30.0,
        )
        self.assertEqual(poly_from_factory.GetNumberOfPoints(), 8)

        # 3. Генерация центров
        centers = generate_square_hole_centers(
            collimator_size=[100.0, 100.0, 30.0],
            hole_width=2.0,
            septa=0.5,
        )
        self.assertGreater(len(centers), 0)
        self.assertEqual(centers.shape[1], 3)
        # Проверяем, что все центры лежат в пределах полуразмеров
        self.assertTrue(np.all(np.abs(centers[:, 0]) <= 50.0))
        self.assertTrue(np.all(np.abs(centers[:, 1]) <= 50.0))
        self.assertTrue(np.all(centers[:, 2] == 0.0))


if __name__ == "__main__":
    unittest.main()
