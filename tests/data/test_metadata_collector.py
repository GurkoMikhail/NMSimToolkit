"""
Модуль модульных тестов для сборщика процедурных метаданных (metadata_collector).
Проверяет изоляцию провайдеров, отсутствие утиной типизации, корректность размерностей
HepUnits и чистоту классов DataHandler.
"""

import unittest
import numpy as np
import hepunits as units

from core.data.metadata_collector import (
    ProtocolMetadataProvider,
    KinematicsMetadataProvider,
    DetectorMetadataProvider,
    ProcedureMetadataCollector,
)
from core.config.models import (
    SpectProtocolConfig,
    StepAndShootProtocolConfig,
    CustomSweepProtocolConfig,
)
from core.data.data_handlers import SensitiveVolumeHandler, HistoryAssemblerHandler
from core.geometry.geometries import Box
from core.materials import Material
from core.geometry.volumes import Volume
from core.scene.gantry_node import GantryNode
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.nodes import CompositeNode


class TestMetadataCollector(unittest.TestCase):
    """
    Набор тестов для провайдеров метаданных и фасадного сборщика.
    """

    def setUp(self) -> None:
        self.vacuum = Material(name="Vacuum", density=0.0001)

    def test_protocol_metadata_provider_spect(self) -> None:
        """
        Проверка извлечения метаданных из SpectProtocolConfig.
        """
        protocol = SpectProtocolConfig(
            views=64,
            gamma_cameras=2,
            time_per_view=2.0 * units.s,
            radius=200.0 * units.mm,
        )
        provider = ProtocolMetadataProvider()
        result = provider.collect(protocol=protocol, context={"view_index": 5}, task_id=5)

        self.assertEqual(result["modality"], "SPECT")
        self.assertEqual(result["view_index"], 5)
        self.assertEqual(result["views_total"], 64)
        self.assertAlmostEqual(result["exposure_time"], 2.0 * 1e9)  # 2.0 s в нс
        self.assertAlmostEqual(result["orbit_radius"], 200.0)      # 200.0 мм

    def test_protocol_metadata_provider_step_and_shoot(self) -> None:
        """
        Проверка извлечения метаданных из StepAndShootProtocolConfig.
        """
        protocol = StepAndShootProtocolConfig(
            views=16,
            start_angle=0.0 * units.deg,
            end_angle=180.0 * units.deg,
            time_per_view=0.5 * units.s,
            radius=150.0 * units.mm,
        )
        provider = ProtocolMetadataProvider()
        result = provider.collect(protocol=protocol, context={}, task_id=3)

        self.assertEqual(result["modality"], "StepAndShoot")
        self.assertEqual(result["view_index"], 3)
        self.assertEqual(result["views_total"], 16)
        self.assertAlmostEqual(result["exposure_time"], 0.5 * 1e9)
        self.assertAlmostEqual(result["orbit_radius"], 150.0)

    def test_protocol_metadata_provider_custom_sweep(self) -> None:
        """
        Проверка извлечения метаданных из CustomSweepProtocolConfig.
        """
        protocol = CustomSweepProtocolConfig(
            zipped_variables={"gantry_angle": [0.1, 0.2, 0.3]}
        )
        provider = ProtocolMetadataProvider()
        result = provider.collect(protocol=protocol, context={"view_index": 1}, task_id=1)

        self.assertEqual(result["modality"], "CustomSweep")
        self.assertEqual(result["view_index"], 1)
        self.assertEqual(result["views_total"], 3)

    def test_kinematics_metadata_provider_with_gantry(self) -> None:
        """
        Проверка извлечения угла и имени GantryNode.
        """
        root = CompositeNode(name="World")
        gantry = GantryNode(name="SPECT_Gantry")
        gantry.set_rotation_angle(np.pi / 4.0)
        root.add_child(gantry)

        provider = KinematicsMetadataProvider()
        result = provider.collect(root_scene=root)

        self.assertEqual(result["gantry_name"], "SPECT_Gantry")
        self.assertAlmostEqual(result["gantry_angle"], float(np.pi / 4.0), places=5)

    def test_detector_metadata_provider(self) -> None:
        """
        Проверка извлечения параметров детектора и кристалла через GammaCameraNode slots.
        """
        root = CompositeNode(name="World")
        camera = GammaCameraNode(
            name="Camera_1",
            slots={"crystal": "NaI_Crystal"},
        )
        crystal = Volume(
            name="NaI_Crystal",
            geometry=Box(x=400.0, y=300.0, z=10.0),
            material=self.vacuum,
            tags=["crystal", "detector"],
        )
        camera.add_child(crystal)
        root.add_child(camera)

        provider = DetectorMetadataProvider()
        result = provider.collect(root_scene=root)

        self.assertIn("Camera_1", result)
        camera_data = result["Camera_1"]
        self.assertEqual(camera_data["camera_name"], "Camera_1")
        self.assertEqual(camera_data["crystal_name"], "NaI_Crystal")
        self.assertEqual(camera_data["global_matrix"].shape, (4, 4))
        self.assertIn("tags", camera_data)

    def test_procedure_metadata_collector_facade(self) -> None:
        """
        Проверка работы фасадного класса ProcedureMetadataCollector.
        """
        root = CompositeNode(name="World")
        gantry = GantryNode(name="Gantry")
        gantry.set_rotation_angle(0.5)
        root.add_child(gantry)

        camera = GammaCameraNode(name="Head1", slots={"crystal": "Crystal1"})
        crystal = Volume(
            name="Crystal1",
            geometry=Box(x=100.0, y=100.0, z=10.0),
            material=self.vacuum,
        )
        camera.add_child(crystal)
        gantry.add_child(camera)

        protocol = SpectProtocolConfig(views=32)

        collector = ProcedureMetadataCollector()
        meta = collector.collect(
            root_scene=root,
            protocol=protocol,
            context={"view_index": 2},
            task_id=2,
        )

        self.assertIn("protocol", meta)
        self.assertIn("kinematics", meta)
        self.assertIn("detectors", meta)

        self.assertEqual(meta["protocol"]["modality"], "SPECT")
        self.assertEqual(meta["kinematics"]["gantry_name"], "Gantry")
        self.assertIn("Head1", meta["detectors"])

    def test_data_handlers_clean_architecture(self) -> None:
        """
        Проверка, что SensitiveVolumeHandler и HistoryAssemblerHandler
        полностью очищены от _build_volume_metadata и volume_metadata.
        """
        vol = Volume(
            name="TestDet",
            geometry=Box(x=10.0, y=10.0, z=10.0),
            material=self.vacuum,
        )
        handler = SensitiveVolumeHandler(sensitive_volumes=[vol])
        self.assertFalse(hasattr(handler, "volume_metadata"))
        self.assertFalse(hasattr(handler, "_build_volume_metadata"))
        self.assertFalse(hasattr(handler, "_apply_volume_attributes"))

        history_handler = HistoryAssemblerHandler(sensitive_volumes=[vol])
        self.assertFalse(hasattr(history_handler, "volume_metadata"))
        self.assertFalse(hasattr(history_handler, "_build_volume_metadata"))


if __name__ == "__main__":
    unittest.main()
