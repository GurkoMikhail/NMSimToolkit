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
    SpectProtocolConfig,
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
            views=4,
            gamma_cameras=1,
            start_angle=0.0 * units.rad,
            end_angle=np.pi * units.rad,
            time_per_view=5.0 * units.s,
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

    def test_compile_spect_protocol_multi_head(self) -> None:
        """Проверка компиляции протокола ОФЭКТ с несколькими детекторными головками."""
        protocol = SpectProtocolConfig(
            views=8,
            gamma_cameras=2,
            start_angle=0.0 * units.rad,
            end_angle=np.pi * units.rad,
            time_per_view=2.0 * units.s,
            head_angles=[0.0 * units.rad, np.pi * units.rad],
            endpoint=False,
        )
        sweep = SweepCompiler.compile_protocol(protocol)
        # При 8 ракурсах и 2 головках число позиций гантри = 8 // 2 = 4
        self.assertEqual(len(sweep.zipped_variables["gantry_angle"]), 4)
        self.assertIn("head_0_angle", sweep.zipped_variables)
        self.assertIn("head_1_angle", sweep.zipped_variables)

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
