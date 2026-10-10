"""
Модульные тесты для компилятора параметров и протоколов SweepCompiler.
Проверяют трансляцию протоколов ОФЭКТ/StepAndShoot в CustomSweepProtocolConfig,
построение декартовой матрицы задач и рекурсивную интерполяцию переменных.
"""

import unittest
import numpy as np
import hepunits as units

from core.config.models import (
    CustomSweepProtocolConfig,
    StepAndShootProtocolConfig,
)
from core.config.sweep_compiler import SweepCompiler


class TestSweepCompiler(unittest.TestCase):
    """Набор тестов для компилятора SweepCompiler."""

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

        # 2 значения energy * 3 пары (angle, time) = 6 задач
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


if __name__ == "__main__":
    unittest.main()
