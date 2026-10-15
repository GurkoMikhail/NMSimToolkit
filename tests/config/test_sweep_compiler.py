"""
Модульные тесты для компилятора параметров и протоколов SweepCompiler и ConfigTemplateBinder.
Проверяют трансляцию протоколов ОФЭКТ/StepAndShoot в CustomSweepProtocolConfig,
построение декартовой матрицы задач, рекурсивную интерполяцию переменных и привязки шаблонов.
"""

import unittest

import hepunits as units
import numpy as np

from core.config.models import (
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
)
from core.config.sweep_compiler import SweepCompiler
from core.config.template_binder import ConfigTemplateBinder


class TestSweepCompiler(unittest.TestCase):
    """Набор тестов для компилятора SweepCompiler и ConfigTemplateBinder."""

    def test_compile_none_protocol(self) -> None:
        """Проверка компиляции пустого протокола (None)."""
        sweep = SweepCompiler.compile_protocol(None)
        self.assertIsInstance(sweep, CustomSweepProtocolConfig)
        self.assertEqual(sweep.grid_variables, {})
        self.assertEqual(sweep.zipped_variables, {})

    def test_compile_custom_sweep_passthrough(self) -> None:
        """Проверка сквозной передачи уже скомпилированного CustomSweepProtocolConfig."""
        existing_sweep = CustomSweepProtocolConfig(
            grid_variables={"energy": [100.0, 140.0]},
            zipped_variables={"time": [1.0, 2.0]},
        )
        compiled_sweep = SweepCompiler.compile_protocol(existing_sweep)
        self.assertIs(compiled_sweep, existing_sweep)

    def test_compile_step_and_shoot_protocol(self) -> None:
        """Проверка компиляции протокола StepAndShoot."""
        protocol = StepAndShootProtocolConfig(
            steps=4,
            start_angle=0.0 * units.rad,
            end_angle=np.pi * units.rad,
            time_per_step=5.0 * units.s,
            endpoint=True,
        )
        sweep = SweepCompiler.compile_protocol(protocol)
        self.assertIn("gantry_angle", sweep.zipped_variables)
        self.assertIn("current_angle", sweep.zipped_variables)
        self.assertIn("current_time", sweep.zipped_variables)

        angles = sweep.zipped_variables["gantry_angle"]
        times = sweep.zipped_variables["current_time"]

        self.assertEqual(len(angles), 4)
        np.testing.assert_allclose(angles, [0.0, np.pi / 3, 2 * np.pi / 3, np.pi])
        np.testing.assert_allclose(times, [5.0 * units.s] * 4)

    def test_compile_step_and_shoot_cw_vs_ccw(self) -> None:
        """Регрессионный тест: при rotation_direction='CCW' углы убывают, а при 'CW' — возрастают."""
        protocol_cw = StepAndShootProtocolConfig(
            steps=4,
            start_angle=0.0 * units.rad,
            end_angle=np.pi * units.rad,
            rotation_direction="CW",
            time_per_step=2.0 * units.s,
            endpoint=True,
        )
        protocol_ccw = StepAndShootProtocolConfig(
            steps=4,
            start_angle=0.0 * units.rad,
            end_angle=np.pi * units.rad,
            rotation_direction="CCW",
            time_per_step=2.0 * units.s,
            endpoint=True,
        )
        sweep_cw = SweepCompiler.compile_protocol(protocol_cw)
        sweep_ccw = SweepCompiler.compile_protocol(protocol_ccw)

        angles_cw = sweep_cw.zipped_variables["gantry_angle"]
        angles_ccw = sweep_ccw.zipped_variables["gantry_angle"]

        np.testing.assert_allclose(angles_cw, [0.0, np.pi / 3, 2 * np.pi / 3, np.pi])
        np.testing.assert_allclose(angles_ccw, [0.0, -np.pi / 3, -2 * np.pi / 3, -np.pi])
        self.assertTrue(all(angles_cw[idx] < angles_cw[idx + 1] for idx in range(3)))
        self.assertTrue(all(angles_ccw[idx] > angles_ccw[idx + 1] for idx in range(3)))

    def test_compile_step_and_shoot_with_radius(self) -> None:
        """Проверка компиляции протокола StepAndShoot с массивом радиусов (Body Contour)."""
        radii = [150.0 * units.mm, 160.0 * units.mm, 170.0 * units.mm, 180.0 * units.mm]
        protocol = StepAndShootProtocolConfig(
            steps=4,
            start_angle=0.0 * units.rad,
            end_angle=np.pi * units.rad,
            time_per_step=2.0 * units.s,
            radius=radii,
            endpoint=False,
        )
        sweep = SweepCompiler.compile_protocol(protocol)
        self.assertEqual(len(sweep.zipped_variables["gantry_angle"]), 4)
        self.assertIn("radius", sweep.zipped_variables)
        self.assertEqual(len(sweep.zipped_variables["radius"]), 4)
        np.testing.assert_allclose(sweep.zipped_variables["radius"], [150.0, 160.0, 170.0, 180.0])

    def test_generate_job_matrix_cartesian_and_zipped(self) -> None:
        """Проверка генерации декартова произведения сетки и синхронной сшивки."""
        sweep = CustomSweepProtocolConfig(
            grid_variables={"energy": [100.0, 200.0]},
            zipped_variables={"angle": [0.0, 90.0, 180.0], "time": [10.0, 20.0, 30.0]},
        )
        job_matrix = SweepCompiler.generate_job_matrix(sweep)

        self.assertEqual(len(job_matrix), 6)
        self.assertEqual(job_matrix[0], {"energy": 100.0, "angle": 0.0, "time": 10.0})
        self.assertEqual(job_matrix[1], {"energy": 100.0, "angle": 90.0, "time": 20.0})
        self.assertEqual(job_matrix[2], {"energy": 100.0, "angle": 180.0, "time": 30.0})
        self.assertEqual(job_matrix[3], {"energy": 200.0, "angle": 0.0, "time": 10.0})

    def test_inject_variables_typed_and_string(self) -> None:
        """Проверка подстановки числовых значений и строковой интерполяции."""
        context = {"gantry_angle": 1.570796, "time_val": 100.0}
        data_structure = {
            "rotation": "${gantry_angle}",
            "filename": "output_${time_val}.hdf",
            "nested_list": ["${gantry_angle}", 42.0],
            "static_text": "no_variables_here",
        }
        resolved = SweepCompiler.inject_variables(data_structure, context)

        self.assertIsInstance(resolved["rotation"], float)
        self.assertAlmostEqual(resolved["rotation"], 1.570796)
        self.assertEqual(resolved["filename"], "output_100.0.hdf")
        self.assertAlmostEqual(resolved["nested_list"][0], 1.570796)
        self.assertEqual(resolved["nested_list"][1], 42.0)
        self.assertEqual(resolved["static_text"], "no_variables_here")

    def test_inject_variables_fail_fast_and_multi_macro(self) -> None:
        """
        Регрессионный тест SweepCompiler.inject_variables:
        1. Подстановка нескольких макросов в одной строке ('${a}_${b}').
        2. Выброс ValueError при неразрешенном одиночном ('${unknown}') и составном ('prefix_${unknown}') макросе.
        """
        context: dict[str, float] = {"a": 1.0, "b": 42.0}
        multi_res = SweepCompiler.inject_variables({"label": "${a}_${b}"}, context)
        self.assertEqual(multi_res["label"], "1.0_42.0")

        with self.assertRaises(ValueError):
            SweepCompiler.inject_variables({"param": "${unknown}"}, context)

        with self.assertRaises(ValueError):
            SweepCompiler.inject_variables({"param": "prefix_${unknown}"}, context)

    def test_template_binder_matrix_transform_no_duplication(self) -> None:
        """
        Регрессионный тест ConfigTemplateBinder._apply_bindings_to_dict:
        Наложение динамической привязки на узел с MatrixTransformConfig раскладывает матрицу
        в элементарные translate/rotate и перезаписывает привязанный параметр без удвоения трансформации.
        """
        mat_4x4 = [
            [1.0, 0.0, 0.0, 250.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        raw_config_dict: dict[str, object] = {
            "scene": {
                "type": "Volume",
                "name": "World",
                "children": [
                    {
                        "type": "Volume",
                        "name": "DetectorHead",
                        "transformations": [
                            {"type": "matrix", "matrix": mat_4x4},
                        ],
                    }
                ],
            }
        }
        bindings_map = {
            "DetectorHead": {"x": "radius"},
        }

        ConfigTemplateBinder._apply_bindings_to_dict(raw_config_dict, bindings_map)

        scene_dict = raw_config_dict["scene"]
        assert isinstance(scene_dict, dict)
        detector_dict = scene_dict["children"][0]
        transforms = detector_dict["transformations"]

        # Матрица должна быть заменена на разложенные элементарные трансформации без остатка type='matrix'
        matrix_entries = [tr for tr in transforms if tr.get("type") == "matrix"]
        translate_entries = [tr for tr in transforms if tr.get("type") == "translate"]
        self.assertEqual(len(matrix_entries), 0)
        self.assertEqual(len(translate_entries), 1)
        self.assertEqual(translate_entries[0]["x"], "${radius} mm")
        self.assertAlmostEqual(float(translate_entries[0]["y"]), 10.0)
        self.assertAlmostEqual(float(translate_entries[0]["z"]), -5.0)


if __name__ == "__main__":
    unittest.main()


