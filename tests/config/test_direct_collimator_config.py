"""
Модульные тесты для сохранения, загрузки и валидации DirectParallelCollimator
в декларативных конфигурациях YAML и Pydantic-моделях.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import hepunits as units

from core.config.models import (
    DirectParallelCollimatorConfig,
    SimulationConfig,
    SimulationManagerConfig,
    DataManagerConfig,
    VolumeConfig,
    BoxConfig,
)
from core.config.builder import SceneBuilder
from core.config.exporter import SceneExporter
from core.config.yaml_loader import load_simulation_config
from core.config.yaml_dumper import dump_simulation_config
from core.geometry.direct_collimators import (
    DirectParallelCollimator,
    CollimatorHoleShape,
)
from core.geometry.geometries import PeriodicHexPrism, Box
from core.geometry.volumes import Volume
import settings.database_setting as database_setting


class TestDirectParallelCollimatorConfig(unittest.TestCase):
    """
    Тестирование Pydantic-модели, фабрики SceneBuilder, экспортера SceneExporter
    и YAML сериализации для DirectParallelCollimator.
    """

    def setUp(self) -> None:
        self.collimator_size = (400.0 * units.mm, 400.0 * units.mm, 30.0 * units.mm)
        self.hole_diameter = 2.5 * units.mm
        self.septa = 0.5 * units.mm

    def test_pydantic_model_validation(self) -> None:
        """Проверка валидации полей DirectParallelCollimatorConfig."""
        config = DirectParallelCollimatorConfig(
            name="TestCollimator",
            size=(400.0 * units.mm, 400.0 * units.mm, 30.0 * units.mm),
            hole_diameter=2.5 * units.mm,
            septa=0.5 * units.mm,
            material="Pb",
            hole_material="Vacuum",
            hole_shape="hexagonal",
        )
        self.assertEqual(config.type, "DirectParallelCollimator")
        self.assertEqual(config.name, "TestCollimator")
        self.assertAlmostEqual(config.hole_diameter, 2.5 * units.mm)
        self.assertAlmostEqual(config.septa, 0.5 * units.mm)
        self.assertEqual(config.material, "Pb")
        self.assertEqual(config.hole_material, "Vacuum")
        self.assertEqual(config.hole_shape, "hexagonal")

    def test_builder_creates_direct_collimator(self) -> None:
        """Проверка построения DirectParallelCollimator через SceneBuilder."""
        config = DirectParallelCollimatorConfig(
            name="MyCollimator",
            size=(400.0 * units.mm, 400.0 * units.mm, 30.0 * units.mm),
            hole_diameter=2.5 * units.mm,
            septa=0.5 * units.mm,
            material="Pb",
            hole_material="Vacuum",
            hole_shape="hexagonal",
        )
        builder = SceneBuilder()
        node = builder.build_scene(config)

        self.assertIsInstance(node, DirectParallelCollimator)
        self.assertEqual(node.name, "MyCollimator")
        self.assertAlmostEqual(node.hole_diameter, 2.5 * units.mm)
        self.assertAlmostEqual(node.septa, 0.5 * units.mm)
        self.assertEqual(node.hole_shape, CollimatorHoleShape.HEXAGONAL)
        self.assertEqual(node.material.name, "Pb")
        self.assertEqual(node.hole_material.name, "Vacuum")

        # Проверка внутренней структуры технических детей
        self.assertIsInstance(node.lead_body, Volume)
        self.assertIsInstance(node.lead_body.geometry, Box)
        self.assertIsInstance(node.channels, Volume)
        self.assertIsInstance(node.channels.geometry, PeriodicHexPrism)

    def test_exporter_serializes_monolithic_node(self) -> None:
        """
        Проверка экспорта DirectParallelCollimator:
        узел должен экспортироваться монолитно БЕЗ разворачивания технических детей в children.
        """
        lead_material = database_setting.material_database["Pb"]
        vacuum_material = database_setting.material_database["Vacuum"]
        collimator = DirectParallelCollimator(
            size=[400.0, 400.0, 30.0],
            hole_diameter=2.5,
            septa=0.5,
            material=lead_material,
            hole_material=vacuum_material,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="ExportedCollimator",
        )

        exported_config = SceneExporter.export_node(collimator)

        self.assertIsInstance(exported_config, DirectParallelCollimatorConfig)
        self.assertEqual(exported_config.name, "ExportedCollimator")
        self.assertAlmostEqual(exported_config.hole_diameter, 2.5)
        self.assertAlmostEqual(exported_config.septa, 0.5)
        self.assertEqual(exported_config.material, "Pb")
        self.assertEqual(exported_config.hole_material, "Vacuum")
        self.assertEqual(exported_config.hole_shape, "hexagonal")
        # Технические дети НЕ должны быть развернуты в children
        self.assertEqual(len(exported_config.children), 0)

    def test_full_yaml_roundtrip(self) -> None:
        """Проверка полного цикла сериализации/десериализации YAML."""
        lead_material = database_setting.material_database["Pb"]
        vacuum_material = database_setting.material_database["Vacuum"]
        collimator = DirectParallelCollimator(
            size=[400.0, 400.0, 30.0],
            hole_diameter=2.5,
            septa=0.5,
            material=lead_material,
            hole_material=vacuum_material,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="RoundtripCollimator",
        )

        collimator_config = SceneExporter.export_node(collimator)

        sim_config = SimulationConfig(
            simulation_manager=SimulationManagerConfig(
                particles_number=1000,
            ),
            data_manager=DataManagerConfig(
                filename="roundtrip_test.h5",
                handlers=[],
            ),
            scene=collimator_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "collimator_scene.yaml"
            dump_simulation_config(sim_config, yaml_path)

            self.assertTrue(yaml_path.is_file())

            loaded_sim_config = load_simulation_config(yaml_path)
            self.assertIsInstance(loaded_sim_config.scene, DirectParallelCollimatorConfig)
            self.assertEqual(loaded_sim_config.scene.name, "RoundtripCollimator")
            self.assertAlmostEqual(loaded_sim_config.scene.hole_diameter, 2.5)
            self.assertAlmostEqual(loaded_sim_config.scene.septa, 0.5)

            builder = SceneBuilder()
            restored_scene = builder.build_scene(loaded_sim_config.scene)
            self.assertIsInstance(restored_scene, DirectParallelCollimator)
            self.assertAlmostEqual(restored_scene.hole_diameter, 2.5)
            self.assertAlmostEqual(restored_scene.septa, 0.5)
            self.assertEqual(restored_scene.hole_shape, CollimatorHoleShape.HEXAGONAL)


if __name__ == '__main__':
    unittest.main()
