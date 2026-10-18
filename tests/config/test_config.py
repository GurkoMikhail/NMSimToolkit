import unittest
import os
import yaml
import numpy as np
from pathlib import Path
import hepunits as units

from core.config.models import (
    BoxConfig,
    DataManagerConfig,
    SimulationConfig,
    SimulationManagerConfig,
    SourceConfig,
    SourceMetadataConfig,
    VolumeConfig,
)
from core.config.yaml_loader import load_simulation_config
from core.config.yaml_dumper import dump_simulation_config
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.source.sources import Source


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_yaml = "test_config.yaml"
        self.config_dict = {
            "simulation_manager": {
                "stop_time": "2.0 s",
                "particles_number": 5000,
                "min_energy": "0.5 MeV"
            },
            "data_manager": {
                "filename": "output.h5",
                "handlers": []
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
                        "type": "translate",
                        "x": "10.0 mm",
                        "y": 2.0,
                        "z": 3.0
                    },
                    {
                        "type": "rotate",
                        "alpha": "90 deg",
                        "beta": "180.0 deg",
                        "gamma": 0.0
                    }
                ],
                "children": [
                    {
                        "type": "WoodcockVoxelVolume",
                        "name": "Phantom",
                        "voxel_size": "5.0 mm",
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
                        "voxel_size": "10.0 mm",
                        "energy": "140.5 keV",
                        "activity": "300.0 MBq",
                        "radionuclide": "99mTc",
                        "phantom_type": "NEMA IEC",
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
        with open(self.test_yaml, "w") as file_handle:
            yaml.dump(self.config_dict, file_handle)

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
        self.assertTrue(np.isclose(config.simulation_manager.stop_time, 2e9))
        self.assertEqual(config.simulation_manager.min_energy, 0.5)
        self.assertEqual(config.scene.type, "Volume")
        self.assertEqual(config.scene.geometry.x, 100.0)
        self.assertEqual(len(config.scene.transformations), 2)
        self.assertEqual(config.scene.transformations[0].type, "translate")
        self.assertEqual(config.scene.transformations[0].x, 10.0)
        self.assertEqual(config.scene.transformations[1].type, "rotate")
        self.assertTrue(np.isclose(config.scene.transformations[1].alpha, np.pi / 2))
        self.assertTrue(np.isclose(config.scene.transformations[1].beta, np.pi))
        self.assertEqual(len(config.scene.children), 2)

        child = config.scene.children[0]
        self.assertEqual(child.type, "WoodcockVoxelVolume")
        self.assertEqual(child.voxel_size, 5.0)
        self.assertEqual(child.distribution.format, "numpy")
        self.assertEqual(child.distribution.mapping[1.0], "Water, Liquid")
        self.assertEqual(child.distribution.fill_value, "Air, Dry (near sea level)")

        child2 = config.scene.children[1]
        self.assertEqual(child2.type, "Source")
        self.assertEqual(child2.energy, 0.1405)
        self.assertTrue(np.isclose(child2.activity, 300e6 * units.Bq))
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

        with open("dumped_test_config.yaml", "r") as file_handle:
            content = file_handle.read()

        self.assertIn("Materials:", content)
        self.assertIn("&air", content)
        self.assertIn("&water", content)
        self.assertIn("&pb", content)
        self.assertIn("*pb", content)

        reloaded_config = load_simulation_config("dumped_test_config.yaml")
        self.assertEqual(config.model_dump(), reloaded_config.model_dump())

    def test_scene_builder(self):
        config = load_simulation_config(self.test_yaml)
        builder = SceneBuilder()
        root_node = builder.build_scene(config.scene)

        self.assertIsInstance(root_node, Volume)
        self.assertEqual(root_node.name, "World")
        self.assertEqual(root_node.geometry.size[0], 100.0)
        self.assertEqual(root_node.material.name, "Air, Dry (near sea level)")

        self.assertEqual(len(root_node.childs), 2)
        child = root_node.childs[0]
        self.assertIsInstance(child, WoodcockVoxelVolume)
        self.assertEqual(child.name, "Phantom")
        self.assertEqual(child.material_distribution.shape, (1, 2, 2))

        ids = child.material_distribution.ID
        mat_db = builder._get_material("Air, Dry (near sea level)")
        self.assertEqual(ids[0, 0, 0], mat_db.ID)

        child2 = root_node.childs[1]
        self.assertIsInstance(child2, Source)
        self.assertEqual(child2.distribution.shape, (1, 2, 2))
        self.assertEqual(child2.distribution[0, 0, 0], 0.0)
        self.assertTrue(np.isclose(child2.distribution[0, 0, 1], 100.0 / 300.0))
        self.assertTrue(np.isclose(child2.initial_activity, 300e6 * units.Bq))

    def test_source_metadata_registry_roundtrip_and_no_dynamic_attrs(self) -> None:
        """
        Регрессионный тест Roundtrip для SourceMetadataConfig (radionuclide, phantom_type):
        1. SceneBuilder сохраняет метаданные во внешнем реестре source_metadata_registry и в SceneExporter.
        2. Экземпляр Source в ядре не загрязняется динамическими атрибутами radionuclide и phantom_type.
        3. SceneExporter.export_node восстанавливает radionuclide и phantom_type в SourceConfig.
        """
        config = load_simulation_config(self.test_yaml)
        builder = SceneBuilder()
        root_node = builder.build_scene(config.scene)
        assert isinstance(root_node, Volume)

        source_node = root_node.childs[1]
        assert isinstance(source_node, Source)

        # Проверка отсутствия динамических атрибутов на экземпляре Source
        self.assertNotIn("radionuclide", source_node.__dict__)
        self.assertNotIn("phantom_type", source_node.__dict__)

        # Проверка регистрации в SceneBuilder.source_metadata_registry и SceneExporter
        self.assertIn(source_node, builder.source_metadata_registry)
        meta_cfg = builder.source_metadata_registry[source_node]
        self.assertIsInstance(meta_cfg, SourceMetadataConfig)
        self.assertEqual(meta_cfg.radionuclide, "99mTc")
        self.assertEqual(meta_cfg.phantom_type, "NEMA IEC")

        exporter_meta = SceneExporter.get_source_metadata(source_node)
        assert exporter_meta is not None
        self.assertEqual(exporter_meta.radionuclide, "99mTc")
        self.assertEqual(exporter_meta.phantom_type, "NEMA IEC")

        # Проверка обратного экспорта через реестр метаданных
        exported_root = SceneExporter.export_node(
            root_node,
            distribution_registry=builder.distribution_registry,
            source_metadata_registry=builder.source_metadata_registry,
        )
        assert isinstance(exported_root, VolumeConfig)
        exported_source_cfg = exported_root.children[1]
        assert isinstance(exported_source_cfg, SourceConfig)
        self.assertEqual(exported_source_cfg.radionuclide, "99mTc")
        self.assertEqual(exported_source_cfg.phantom_type, "NEMA IEC")

    def test_load_nema_1_cam_yaml(self):
        config = load_simulation_config("examples/nema_1_cam.yaml")
        self.assertIsInstance(config, SimulationConfig)
        builder = SceneBuilder()
        root_node = builder.build_scene(config.scene)
        self.assertIsNotNone(root_node)
        self.assertEqual(root_node.name, "Simulation_volume")


if __name__ == '__main__':
    unittest.main()

