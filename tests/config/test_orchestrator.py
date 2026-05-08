import unittest
import numpy as np

from core.config.orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.raw_config = {
            "protocol": {
                "type": "StepAndShoot",
                "views": 3,
                "gamma_cameras": 1,
                "start_angle": "0 deg",
                "end_angle": "180 deg",
                "time_per_view": "10 s"
            },
            "settings": {
                "stop_time": "${current_time}",
                "particles_number": 5000,
                "min_energy": "0.5 MeV"
            },
            "data_manager": {
                "filename": "output.h5",
                "handlers": [],
                "buffer_capacity": 1000
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {
                    "type": "Box",
                    "x": "10.0 cm",
                    "y": "10.0 cm",
                    "z": "10.0 cm"
                },
                "material": "Air, Dry (near sea level)",
                "transformations": [
                    {
                        "type": "rotate",
                        "alpha": "${current_angle}",
                        "beta": 0.0,
                        "gamma": 0.0
                    }
                ],
                "children": []
            }
        }

    def test_invalid_zipped_lengths(self):
        from core.config.models import CustomSweepProtocolConfig
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            CustomSweepProtocolConfig(
                zipped_variables={
                    "var1": [1.0, 2.0],
                    "var2": [1.0]
                }
            )

    def test_orchestrator_job_generation(self):
        from core.config.models import CustomSweepProtocolConfig
        sweep = CustomSweepProtocolConfig(
            grid_variables={
                "energy": [100.0, 200.0]
            },
            zipped_variables={
                "angle": [0.0, 90.0, 180.0],
                "time": [10.0, 20.0, 30.0]
            }
        )
        orchestrator = Orchestrator(self.raw_config) # Reusing raw config since we only test the internal _generate_job_list
        tasks = orchestrator._generate_job_list(sweep)

        # 2 grid items * 3 zipped items = 6 tasks
        self.assertEqual(len(tasks), 6)

        # Checking cross product
        self.assertEqual(tasks[0], {"energy": 100.0, "angle": 0.0, "time": 10.0})
        self.assertEqual(tasks[1], {"energy": 100.0, "angle": 90.0, "time": 20.0})
        self.assertEqual(tasks[2], {"energy": 100.0, "angle": 180.0, "time": 30.0})
        self.assertEqual(tasks[3], {"energy": 200.0, "angle": 0.0, "time": 10.0})

    def test_orchestrator_compilation_and_injection(self):
        orchestrator = Orchestrator(self.raw_config)

        # Test protocol compilation
        sweep_config = orchestrator.compile_protocol()
        self.assertEqual(sweep_config.type, "CustomSweep")
        self.assertIn("current_angle", sweep_config.zipped_variables)
        self.assertIn("current_time", sweep_config.zipped_variables)

        angles = sweep_config.zipped_variables["current_angle"]
        times = sweep_config.zipped_variables["current_time"]

        self.assertEqual(len(angles), 3)
        self.assertEqual(len(times), 3)

        # start_angle was 0 deg (0 rad), end_angle was 180 deg (pi rad)
        self.assertTrue(np.isclose(angles[0], 0.0))
        self.assertTrue(np.isclose(angles[1], np.pi / 2))
        self.assertTrue(np.isclose(angles[2], np.pi))

        # time_per_view was 10 s (1e10 ns)
        self.assertTrue(np.isclose(times[0], 1e10))

        # Test full pipeline execution (generates list of validated SimulationConfigs)
        results = orchestrator.run()
        self.assertEqual(len(results), 3)

        # Verify the 2nd task (which should be 90 deg / pi/2 rad)
        context, final_config = results[1]
        self.assertTrue(np.isclose(context["current_angle"], np.pi / 2))
        self.assertTrue(np.isclose(final_config.scene.transformations[0].alpha, np.pi / 2))
        self.assertTrue(np.isclose(final_config.settings.stop_time, 1e10))

if __name__ == '__main__':
    unittest.main()
