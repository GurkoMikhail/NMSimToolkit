import unittest
import os
import yaml
import numpy as np
from pathlib import Path

from core.config.models import SimulationConfig, VolumeConfig, BoxConfig, SimulationManagerConfig, DataManagerConfig
from core.config.yaml_loader import load_simulation_config
from core.config.yaml_dumper import dump_simulation_config
from core.config.builder import SceneBuilder
from core.geometry.volumes import Volume

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_yaml = "test_config.yaml"
        self.config_dict = {
            "settings": {
                "stop_time": 2.0,
                "particles_number": 5000,
                "min_energy": 500.0
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
                    "x": 10.0,
                    "y": 10.0,
                    "z": 10.0
                },
                "material": "Air, Dry (near sea level)",
                "transformations": [
                    {
                        "type": "translate",
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0
                    }
                ],
                "children": [
                    {
                        "type": "Volume",
                        "name": "Detector",
                        "geometry": {
                            "type": "Box",
                            "x": 2.0,
                            "y": 2.0,
                            "z": 2.0
                        },
                        "material": "Pb"
                    }
                ]
            }
        }
        with open(self.test_yaml, "w") as f:
            yaml.dump(self.config_dict, f)

    def tearDown(self):
        if os.path.exists(self.test_yaml):
            os.remove(self.test_yaml)
        if os.path.exists("dumped_test_config.yaml"):
            os.remove("dumped_test_config.yaml")

    def test_pydantic_validation(self):
        config = SimulationConfig.model_validate(self.config_dict)
        self.assertEqual(config.settings.stop_time, 2.0)
        self.assertEqual(config.scene.type, "Volume")
        self.assertEqual(len(config.scene.transformations), 1)
        self.assertEqual(config.scene.transformations[0].type, "translate")
        self.assertEqual(config.scene.transformations[0].x, 1.0)
        self.assertEqual(len(config.scene.children), 1)
        self.assertEqual(config.scene.children[0].type, "Volume")

    def test_yaml_loader(self):
        config = load_simulation_config(self.test_yaml)
        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.scene.name, "World")

    def test_yaml_dumper(self):
        config = load_simulation_config(self.test_yaml)
        dump_simulation_config(config, "dumped_test_config.yaml")

        with open("dumped_test_config.yaml", "r") as f:
            content = f.read()

        # Verify materials block is generated and anchored
        self.assertIn("Materials:", content)
        self.assertIn("&air", content)
        self.assertIn("&pb", content)
        self.assertIn("*air", content)
        self.assertIn("*pb", content)

        # Reloading should yield identical model
        reloaded_config = load_simulation_config("dumped_test_config.yaml")
        self.assertEqual(config.model_dump(), reloaded_config.model_dump())

    def test_scene_builder(self):
        config = load_simulation_config(self.test_yaml)
        builder = SceneBuilder()
        root_node = builder.build_scene(config.scene)

        self.assertIsInstance(root_node, Volume)
        self.assertEqual(root_node.name, "World")
        self.assertEqual(root_node.geometry.size[0], 10.0)
        self.assertEqual(root_node.material.name, "Air, Dry (near sea level)")

        self.assertEqual(len(root_node.childs), 1)
        child = root_node.childs[0]
        self.assertIsInstance(child, Volume)
        self.assertEqual(child.name, "Detector")
        self.assertEqual(child.material.name, "Pb")

if __name__ == '__main__':
    unittest.main()
