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
                        "type": "WoodcockVoxelVolume",
                        "name": "Phantom",
                        "voxel_size": 0.5,
                        "distribution": {
                            "format": "numpy",
                            "path": "dummy_dist.npy",
                            "fill_value": "Air, Dry (near sea level)",
                            "mapping": {
                                1.0: "Water, Liquid",
                                2.0: "Pb"
                            }
                        }
                    },
                    {
                        "type": "Source",
                        "name": "Source",
                        "voxel_size": 0.1,
                        "distribution": {
                            "format": "raw",
                            "path": "source.dat",
                            "shape": [1, 2, 2],
                            "order": "C",
                            "fill_value": 0.0,
                            "mapping": {
                                1.0: 100.0,
                                2.0: 200.0
                            }
                        }
                    }
                ]
            }
        }
        with open(self.test_yaml, "w") as f:
            yaml.dump(self.config_dict, f)

        dummy_dist = np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=float)
        np.save("dummy_dist.npy", dummy_dist)

        np.savetxt("source.dat", dummy_dist.flatten())

    def tearDown(self):
        if os.path.exists(self.test_yaml):
            os.remove(self.test_yaml)
        if os.path.exists("dumped_test_config.yaml"):
            os.remove("dumped_test_config.yaml")
        if os.path.exists("dummy_dist.npy"):
            os.remove("dummy_dist.npy")
        if os.path.exists("source.dat"):
            os.remove("source.dat")

    def test_pydantic_validation(self):
        config = SimulationConfig.model_validate(self.config_dict)
        self.assertEqual(config.settings.stop_time, 2.0)
        self.assertEqual(config.scene.type, "Volume")
        self.assertEqual(len(config.scene.transformations), 1)
        self.assertEqual(config.scene.transformations[0].type, "translate")
        self.assertEqual(config.scene.transformations[0].x, 1.0)
        self.assertEqual(len(config.scene.children), 2)

        child = config.scene.children[0]
        self.assertEqual(child.type, "WoodcockVoxelVolume")
        self.assertEqual(child.distribution.format, "numpy")
        self.assertEqual(child.distribution.mapping[1.0], "Water, Liquid")
        self.assertEqual(child.distribution.fill_value, "Air, Dry (near sea level)")

        child2 = config.scene.children[1]
        self.assertEqual(child2.type, "Source")
        self.assertEqual(child2.distribution.format, "raw")
        self.assertEqual(child2.distribution.mapping[1.0], 100.0)
        self.assertEqual(child2.distribution.fill_value, 0.0)

    def test_yaml_loader(self):
        config = load_simulation_config(self.test_yaml)
        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.scene.name, "World")

    def test_yaml_dumper(self):
        config = load_simulation_config(self.test_yaml)
        dump_simulation_config(config, "dumped_test_config.yaml")

        with open("dumped_test_config.yaml", "r") as f:
            content = f.read()

        # Verify materials block is generated and anchored.
        # We are asserting that Water, Air, and Pb are handled
        self.assertIn("Materials:", content)
        self.assertIn("&air", content)
        self.assertIn("&water", content)
        self.assertIn("&pb", content)
        self.assertIn("*pb", content)

        # Reloading should yield identical model
        reloaded_config = load_simulation_config("dumped_test_config.yaml")
        self.assertEqual(config.model_dump(), reloaded_config.model_dump())

    def test_scene_builder(self):
        config = load_simulation_config(self.test_yaml)
        builder = SceneBuilder()
        from core.geometry.voxel_volumes import WoodcockVoxelVolume
        from core.source.sources import Source
        root_node = builder.build_scene(config.scene)

        self.assertIsInstance(root_node, Volume)
        self.assertEqual(root_node.name, "World")
        self.assertEqual(root_node.geometry.size[0], 10.0)
        self.assertEqual(root_node.material.name, "Air, Dry (near sea level)")

        self.assertEqual(len(root_node.childs), 2)
        child = root_node.childs[0]
        self.assertIsInstance(child, WoodcockVoxelVolume)
        self.assertEqual(child.name, "Phantom")
        self.assertEqual(child.material_distribution.shape, (1, 2, 2))

        # Ensure mapping correctly applied
        ids = child.material_distribution.ID
        # 0.0 -> Air (from fill value)
        mat_db = builder._get_material("Air, Dry (near sea level)")
        self.assertEqual(ids[0, 0, 0], mat_db.ID)

        child2 = root_node.childs[1]
        self.assertIsInstance(child2, Source)
        self.assertEqual(child2.distribution.shape, (1, 2, 2))
        self.assertEqual(child2.distribution[0, 0, 0], 0.0)
        # 100 is converted to probability since the source object normalizes the distribution upon init
        self.assertTrue(np.isclose(child2.distribution[0, 0, 1], 100.0 / 300.0))
        self.assertEqual(child2.initial_activity, 300.0) # The total activity defaults to sum of distribution before normalization

if __name__ == '__main__':
    unittest.main()
