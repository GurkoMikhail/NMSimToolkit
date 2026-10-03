"""
Тесты для детерминированного параметрического коллиматора DirectParallelCollimator
и геометрического примитива PeriodicHexPrism.
"""

import unittest
import numpy as np
import hepunits as units

import settings.database_setting as settings
from core.geometry.geometries import Box, PeriodicHexPrism, ShapeDataDType
from core.geometry.direct_collimators import DirectParallelCollimator
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume, GeometryBufferDType
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.flattened_scene import FlattenedScene
from core.geometry.geometry_kernels import _hex_prism_intersect, _get_intersection, _trace_single_ray
from core.scene.nodes import CompositeNode
from core.other.typing_definitions import Float


class TestPeriodicHexPrism(unittest.TestCase):
    """Тестирование геометрического примитива PeriodicHexPrism."""

    def setUp(self) -> None:
        self.collimator_size = (100.0 * units.mm, 100.0 * units.mm, 40.0 * units.mm)
        self.hole_diameter = 1.5 * units.mm
        self.septa = 0.2 * units.mm
        self.prism = PeriodicHexPrism(
            size=self.collimator_size,
            hole_diameter=self.hole_diameter,
            septa=self.septa
        )

    def test_parameters_and_properties(self) -> None:
        """Проверка базовых геометрических параметров и свойств."""
        self.assertAlmostEqual(self.prism.hole_diameter, 1.5 * units.mm)
        self.assertAlmostEqual(self.prism.septa, 0.2 * units.mm)
        self.assertEqual(len(self.prism.size), 3)
        self.assertAlmostEqual(self.prism.size[2], 40.0 * units.mm)
        self.assertAlmostEqual(self.prism.half_size[2], 20.0 * units.mm)

    def test_write_shape_data(self) -> None:
        """Проверка корректности записи формы в NumPy буфер ShapeDataDType."""
        shape_buffer = np.zeros(1, dtype=ShapeDataDType)
        self.prism.write_shape_data(shape_buffer, 0)

        self.assertEqual(shape_buffer[0]['shape'], 1)
        expected_x_period = 1.5 * units.mm + 0.2 * units.mm
        expected_y_period = np.sqrt(3.0) * expected_x_period
        expected_channel_half_width = 0.5 * 1.5 * units.mm
        expected_channel_side_limit = 1.5 * units.mm
        expected_half_height_z = 20.0 * units.mm
        expected_cell_half_width = 0.5 * expected_x_period

        self.assertAlmostEqual(shape_buffer[0]['param_0'], expected_x_period)
        self.assertAlmostEqual(shape_buffer[0]['param_1'], expected_y_period)
        self.assertAlmostEqual(shape_buffer[0]['param_2'], expected_channel_half_width)
        self.assertAlmostEqual(shape_buffer[0]['param_3'], expected_channel_side_limit)
        self.assertAlmostEqual(shape_buffer[0]['param_4'], expected_half_height_z)
        self.assertAlmostEqual(shape_buffer[0]['param_5'], expected_cell_half_width)

    def test_setter_updates(self) -> None:
        """Проверка динамического обновления параметров через сеттеры."""
        self.prism.hole_diameter = 2.0 * units.mm
        self.prism.septa = 0.4 * units.mm

        shape_buffer = np.zeros(1, dtype=ShapeDataDType)
        self.prism.write_shape_data(shape_buffer, 0)

        new_x_period = 2.4 * units.mm
        self.assertAlmostEqual(shape_buffer[0]['param_0'], new_x_period)
        self.assertAlmostEqual(shape_buffer[0]['param_2'], 1.0 * units.mm)


class TestHexPrismKernel(unittest.TestCase):
    """Тестирование аналитического кернела _hex_prism_intersect."""

    def setUp(self) -> None:
        self.hole_diameter = 1.5 * units.mm
        self.septa = 0.2 * units.mm
        self.thickness_z = 40.0 * units.mm
        self.prism = PeriodicHexPrism(
            size=(100.0 * units.mm, 100.0 * units.mm, self.thickness_z),
            hole_diameter=self.hole_diameter,
            septa=self.septa
        )
        shape_buffer = np.zeros(1, dtype=ShapeDataDType)
        self.prism.write_shape_data(shape_buffer, 0)
        self.shape_data = shape_buffer[0]

    def test_ray_inside_channel_along_z(self) -> None:
        """Луч находится в центре канала и летит вдоль оси Z до торца."""
        local_position_x = 0.0
        local_position_y = 0.0
        local_position_z = 0.0
        local_direction_x = 0.0
        local_direction_y = 0.0
        local_direction_z = 1.0

        time_min, time_max = _hex_prism_intersect(
            local_position_x, local_position_y, local_position_z,
            local_direction_x, local_direction_y, local_direction_z,
            self.shape_data
        )

        # Точка внутри канала: time_min == 0.0, time_max == расстояние до торца (20 мм)
        self.assertAlmostEqual(time_min, 0.0)
        self.assertAlmostEqual(time_max, 20.0 * units.mm)

    def test_ray_inside_channel_transverse_to_wall(self) -> None:
        """Луч находится в центре канала и летит перпендикулярно оси к вертикальной стенке гексагона."""
        local_position_x = 0.0
        local_position_y = 0.0
        local_position_z = 0.0
        local_direction_x = 1.0
        local_direction_y = 0.0
        local_direction_z = 0.0

        time_min, time_max = _hex_prism_intersect(
            local_position_x, local_position_y, local_position_z,
            local_direction_x, local_direction_y, local_direction_z,
            self.shape_data
        )

        # Выход через вертикальную грань x = channel_half_width = 0.75 мм
        self.assertAlmostEqual(time_min, 0.0)
        self.assertAlmostEqual(time_max, 0.75 * units.mm)

    def test_ray_in_lead_septa_towards_channel(self) -> None:
        """Луч находится в свинцовой септе и летит в сторону канала."""
        # Канал с центром в (0, 0) имеет стенку при x = 0.75 мм.
        # Поместим частицу в x = 0.80 мм (в септу) и направим луч влево к каналу (-1, 0, 0)
        local_position_x = 0.80 * units.mm
        local_position_y = 0.0
        local_position_z = 0.0
        local_direction_x = -1.0
        local_direction_y = 0.0
        local_direction_z = 0.0

        time_min, time_max = _hex_prism_intersect(
            local_position_x, local_position_y, local_position_z,
            local_direction_x, local_direction_y, local_direction_z,
            self.shape_data
        )

        # В септе: time_min == расстояние до входа в канал (0.05 мм), time_max == inf
        self.assertAlmostEqual(time_min, 0.05 * units.mm)
        self.assertEqual(time_max, np.inf)

    def test_ray_in_lead_septa_towards_cell_boundary(self) -> None:
        """Луч находится в свинцовой септе и летит вправо к границе ячейки периодичности."""
        # Период x_period = 1.7 мм, cell_half_width = 0.85 мм.
        # Точка в x = 0.80 мм, летит вправо (1, 0, 0).
        # До границы ячейки расстояние = 0.85 - 0.80 = 0.05 мм.
        local_position_x = 0.80 * units.mm
        local_position_y = 0.0
        local_position_z = 0.0
        local_direction_x = 1.0
        local_direction_y = 0.0
        local_direction_z = 0.0

        time_min, time_max = _hex_prism_intersect(
            local_position_x, local_position_y, local_position_z,
            local_direction_x, local_direction_y, local_direction_z,
            self.shape_data
        )

        # До границы ячейки Вороного: time_min == 0.05 мм, time_max == inf
        self.assertAlmostEqual(time_min, 0.05 * units.mm)
        self.assertEqual(time_max, np.inf)

    def test_ray_outside_z_moving_away(self) -> None:
        """Луч находится за торцом по оси Z и улетает наружу."""
        local_position_x = 0.0
        local_position_y = 0.0
        local_position_z = 30.0 * units.mm  # При полувысоте 20 мм
        local_direction_x = 0.0
        local_direction_y = 0.0
        local_direction_z = 1.0

        time_min, time_max = _hex_prism_intersect(
            local_position_x, local_position_y, local_position_z,
            local_direction_x, local_direction_y, local_direction_z,
            self.shape_data
        )

        self.assertEqual(time_min, np.inf)
        self.assertEqual(time_max, -np.inf)


class TestDirectParallelCollimator(unittest.TestCase):
    """Тестирование класса DirectParallelCollimator и его интеграции в граф сцены."""

    def setUp(self) -> None:
        self.size = (120.0 * units.mm, 100.0 * units.mm, 35.0 * units.mm)
        self.hole_diameter = 1.5 * units.mm
        self.septa = 0.2 * units.mm
        self.collimator = DirectParallelCollimator(
            size=self.size,
            hole_diameter=self.hole_diameter,
            septa=self.septa,
            name="TestCollimator"
        )

    def test_node_hierarchy(self) -> None:
        """Проверка структуры графа сцены и типов узлов."""
        # Коллиматор является CompositeNode, но НЕ Volume
        self.assertIsInstance(self.collimator, CompositeNode)
        self.assertNotIsInstance(self.collimator, Volume)

        # Внутри создана иерархия lead_body -> channels
        self.assertIsInstance(self.collimator.lead_body, Volume)
        self.assertIsInstance(self.collimator.channels, Volume)
        self.assertEqual(self.collimator.lead_body.material.name, 'Pb')
        self.assertEqual(self.collimator.channels.material.name, 'Vacuum')

        self.assertIs(self.collimator.channels.parent, self.collimator.lead_body)
        self.assertIs(self.collimator.lead_body.parent, self.collimator)

    def test_flattened_scene_and_compiler(self) -> None:
        """Проверка корректной компиляции иерархии в GeometryBuffer через FlattenedScene."""
        flattened = FlattenedScene(self.collimator)
        flat_list = flattened.flat_list

        # В плоский буфер попадают ровно два Volume: свинцовый корпус и воздушные каналы
        self.assertEqual(len(flat_list), 2)
        lead_volume, _, lead_parent_index = flat_list[0]
        channels_volume, _, channels_parent_index = flat_list[1]

        self.assertIs(lead_volume, self.collimator.lead_body)
        self.assertIs(channels_volume, self.collimator.channels)
        self.assertEqual(lead_parent_index, -1)
        self.assertEqual(channels_parent_index, 0)

        # Компиляция в буфер геометрии
        compiler = GeometryCompiler()
        geometry_buffer = compiler.compile_scene(self.collimator)

        self.assertEqual(geometry_buffer.shape[0], 2)
        # 0-й элемент: свинцовый Box
        self.assertEqual(geometry_buffer[0]['shape_data']['shape'], 0)
        self.assertEqual(geometry_buffer[0]['miss_index'], 2)  # перепрыгивает дочерний channels
        # 1-й элемент: воздушный PeriodicHexPrism
        self.assertEqual(geometry_buffer[1]['shape_data']['shape'], 1)
        self.assertEqual(geometry_buffer[1]['parent_index'], 0)

    def test_single_pass_painters_raycasting(self) -> None:
        """
        Проверка работы Single-Pass Painter's Algorithm в _trace_single_ray
        для прямого коллиматора.
        """
        compiler = GeometryCompiler()
        geometry_buffer = compiler.compile_scene(self.collimator)

        # Сценарий 1: Фотон летит по оси Z сквозь центр канала (Air/Vacuum)
        # Позиция (0, 0, 0), направление (0, 0, 1)
        closest_distance, detected_volume_index = _trace_single_ray(
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            geometry_buffer
        )

        # Painter's Algorithm обязан распознать среду как каналы (индекс 1)
        # и установить расстояние до выхода из коллиматора (17.5 мм)
        self.assertEqual(detected_volume_index, 1)
        self.assertAlmostEqual(closest_distance, 17.5 * units.mm)

        # Сценарий 2: Фотон летит внутри свинцовой септы
        # Точка (0.80 мм, 0, 0), направление влево к каналу (-1, 0, 0)
        closest_distance_septa, detected_volume_septa = _trace_single_ray(
            0.80 * units.mm, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            geometry_buffer
        )

        # В септе среда должна остаться свинцом (индекс 0), а дистанция - до входа в канал (0.05 мм)
        self.assertEqual(detected_volume_septa, 0)
        self.assertAlmostEqual(closest_distance_septa, 0.05 * units.mm)

        # Сценарий 3: Фотон летит снаружи коллиматора к его поверхности
        # Позиция (0, 0, -50 мм), направление к коллиматору (0, 0, 1)
        closest_distance_outside, detected_volume_outside = _trace_single_ray(
            0.0, 0.0, -50.0 * units.mm,
            0.0, 0.0, 1.0,
            geometry_buffer
        )

        # Снаружи коллиматора объем не детектируется (-1), а расстояние до входа в Box = 50 - 17.5 = 32.5 мм
        self.assertEqual(detected_volume_outside, -1)
        self.assertAlmostEqual(closest_distance_outside, 32.5 * units.mm)

    def test_equivalence_of_channel_identification(self) -> None:
        """
        Проверка 100% геометрической эквивалентности распознавания каналов
        между оригинальным ParametricParallelCollimator и новым PeriodicHexPrism.
        """
        orig_collimator = ParametricParallelCollimator(
            size=self.size,
            hole_diameter=self.hole_diameter,
            septa=self.septa,
            material=settings.material_database['Pb']
        )
        parametric_func = orig_collimator._compile_cfunc()

        # Проверим сетку точек в диапазоне одного периода
        test_x_values = np.linspace(-1.5 * units.mm, 1.5 * units.mm, 31)
        test_y_values = np.linspace(-2.5 * units.mm, 2.5 * units.mm, 31)

        shape_buffer = np.zeros(1, dtype=ShapeDataDType)
        self.collimator.channels.geometry.write_shape_data(shape_buffer, 0)
        shape_data = shape_buffer[0]

        lead_material_id = settings.material_database['Pb'].ID
        vacuum_material_id = settings.material_database['Vacuum'].ID

        for coordinate_x in test_x_values:
            for coordinate_y in test_y_values:
                # Оригинальная параметрическая функция
                expected_mat_id = parametric_func(coordinate_x, coordinate_y, 0.0)

                # Проверка через кернел _hex_prism_intersect
                # Если частица внутри канала, то при движении вдоль Z time_min == 0
                time_min, _ = _hex_prism_intersect(
                    coordinate_x, coordinate_y, 0.0,
                    0.0, 0.0, 1.0,
                    shape_data
                )

                if expected_mat_id == vacuum_material_id:
                    self.assertAlmostEqual(
                        time_min, 0.0,
                        msg=f"Несовпадение канала в точке ({coordinate_x}, {coordinate_y}): ожидался Vacuum"
                    )
                else:
                    self.assertGreater(
                        time_min, 0.0,
                        msg=f"Несовпадение септы в точке ({coordinate_x}, {coordinate_y}): ожидался Pb"
                    )

    def test_lbyl_parameter_validation(self) -> None:
        """Проверка контрактов LBYL: выброс ValueError при неположительных параметрах."""
        valid_size = (100.0, 100.0, 30.0)

        # PeriodicHexPrism
        with self.assertRaises(ValueError):
            PeriodicHexPrism(size=(0.0, 100.0, 30.0), hole_diameter=1.5, septa=0.2)
        with self.assertRaises(ValueError):
            PeriodicHexPrism(size=valid_size, hole_diameter=-1.0, septa=0.2)
        with self.assertRaises(ValueError):
            PeriodicHexPrism(size=valid_size, hole_diameter=1.5, septa=0.0)

        prism = PeriodicHexPrism(size=valid_size, hole_diameter=1.5, septa=0.2)
        with self.assertRaises(ValueError):
            prism.hole_diameter = -0.5
        with self.assertRaises(ValueError):
            prism.septa = 0.0

        # DirectParallelCollimator
        with self.assertRaises(ValueError):
            DirectParallelCollimator(size=(-10.0, 100.0, 30.0), hole_diameter=1.5, septa=0.2)
        with self.assertRaises(ValueError):
            DirectParallelCollimator(size=valid_size, hole_diameter=0.0, septa=0.2)
        with self.assertRaises(ValueError):
            DirectParallelCollimator(size=valid_size, hole_diameter=1.5, septa=-0.1)

        with self.assertRaises(ValueError):
            self.collimator.hole_diameter = 0.0
        with self.assertRaises(ValueError):
            self.collimator.septa = -0.2
        with self.assertRaises(ValueError):
            self.collimator.size = (100.0, -50.0, 30.0)


if __name__ == '__main__':
    unittest.main()

