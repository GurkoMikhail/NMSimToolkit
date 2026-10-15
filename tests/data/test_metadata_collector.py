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
    GeometryMetadataProvider,
    GantryMetadataProvider,
    SourceMetadataProvider,
    ProcedureMetadataCollector,
)
from core.config.models import (
    StepAndShootProtocolConfig,
    CustomSweepProtocolConfig,
    SimulationManagerConfig,
)
from core.data.data_handlers import SensitiveVolumeHandler, HistoryAssemblerHandler
from core.geometry.direct_collimators import CollimatorHoleShape
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.materials import Material
from core.geometry.volumes import Volume
from core.scene.gantry_node import GantryNode
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.nodes import CompositeNode
from core.source.sources import Source


class TestMetadataCollector(unittest.TestCase):
    """
    Набор тестов для провайдеров метаданных и фасадного сборщика.
    """

    def setUp(self) -> None:
        self.vacuum = Material(name="Vacuum", density=0.0001)

    def test_protocol_metadata_provider_step_and_shoot_contour(self) -> None:
        """
        Проверка извлечения метаданных из StepAndShootProtocolConfig с массивом радиусов.
        """
        protocol = StepAndShootProtocolConfig(
            steps=64,
            time_per_step=2.0 * units.s,
            radius=[200.0 * units.mm] * 64,
        )
        provider = ProtocolMetadataProvider()
        result = provider.collect(protocol=protocol, context={"view_index": 5}, task_id=5)

        self.assertEqual(result["modality"], "StepAndShoot")
        self.assertEqual(result["view_index"], 5)
        self.assertEqual(result["views_total"], 64)
        self.assertAlmostEqual(result["exposure_time"], 2.0)
        self.assertAlmostEqual(result["orbit_radius"], 200.0)

    def test_protocol_metadata_provider_step_and_shoot(self) -> None:
        """
        Проверка извлечения метаданных из StepAndShootProtocolConfig со скалярным радиусом и без радиуса.
        """
        protocol = StepAndShootProtocolConfig(
            steps=16,
            start_angle=0.0 * units.deg,
            end_angle=180.0 * units.deg,
            time_per_step=0.5 * units.s,
            radius=150.0 * units.mm,
        )
        provider = ProtocolMetadataProvider()
        result = provider.collect(protocol=protocol, context={}, task_id=3)

        self.assertEqual(result["modality"], "StepAndShoot")
        self.assertEqual(result["view_index"], 3)
        self.assertEqual(result["views_total"], 16)
        self.assertAlmostEqual(result["exposure_time"], 0.5)
        self.assertAlmostEqual(result["orbit_radius"], 150.0)

        protocol_no_radius = StepAndShootProtocolConfig(
            steps=16,
            time_per_step=1.5 * units.s,
            radius=None,
        )
        result_no_radius = provider.collect(protocol=protocol_no_radius, context={}, task_id=0)
        self.assertNotIn("orbit_radius", result_no_radius)
        self.assertAlmostEqual(result_no_radius["exposure_time"], 1.5)

    def test_protocol_metadata_provider_custom_sweep(self) -> None:
        """
        Проверка извлечения метаданных из CustomSweepProtocolConfig без нулевых заглушек.
        """
        protocol = CustomSweepProtocolConfig(
            zipped_variables={"gantry_angle": [0.1, 0.2, 0.3]}
        )
        provider = ProtocolMetadataProvider()
        result = provider.collect(protocol=protocol, context={"view_index": 1}, task_id=1)

        self.assertEqual(result["modality"], "CustomSweep")
        self.assertEqual(result["view_index"], 1)
        self.assertEqual(result["views_total"], 3)
        self.assertNotIn("exposure_time", result)
        self.assertNotIn("orbit_radius", result)

        static_result = provider.collect(protocol=None, context={}, task_id=0)
        self.assertEqual(static_result["modality"], "Static")
        self.assertNotIn("exposure_time", static_result)
        self.assertNotIn("orbit_radius", static_result)

    def test_kinematics_metadata_provider_with_and_without_gantry(self) -> None:
        """
        Проверка извлечения угла и имени GantryNode, а также отсутствия фиктивных полей без GantryNode.
        """
        root = CompositeNode(name="World")
        gantry = GantryNode(name="SPECT_Gantry")
        gantry.set_rotation_angle(np.pi / 4.0)
        root.add_child(gantry)

        provider = KinematicsMetadataProvider()
        result = provider.collect(root_scene=root)

        self.assertEqual(result["gantry_name"], "SPECT_Gantry")
        self.assertAlmostEqual(result["gantry_angle"], float(np.pi / 4.0), places=5)

        empty_root = CompositeNode(name="EmptyWorld")
        self.assertEqual(provider.collect(root_scene=empty_root), {})
        context_result = provider.collect(root_scene=empty_root, context={"gantry_angle": 0.75})
        self.assertEqual(context_result, {"gantry_angle": 0.75})
        self.assertNotIn("gantry_name", context_result)

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

    def test_geometry_metadata_provider(self) -> None:
        """
        Проверка извлечения параметров объемов сцены во вложенный словарь volumes
        и отсутствия нулевых заглушек при отсутствии камер.
        """
        root = CompositeNode(name="World")
        phantom = Volume(
            name="Phantom",
            geometry=Box(x=200.0, y=200.0, z=200.0),
            material=self.vacuum,
            tags=["phantom", "target"],
        )
        root.add_child(phantom)

        provider = GeometryMetadataProvider()
        result = provider.collect(root_scene=root)

        self.assertEqual(result["detector_heads_count"], 0)
        self.assertNotIn("crystal_dimensions", result)
        self.assertNotIn("collimator_type", result)
        self.assertNotIn("Phantom", result)
        self.assertIn("volumes", result)
        self.assertIn("Phantom", result["volumes"])
        phantom_record = result["volumes"]["Phantom"]
        self.assertEqual(phantom_record["name"], "Phantom")
        self.assertEqual(phantom_record["material"], "Vacuum")
        self.assertEqual(phantom_record["geometry_type"], "Box")
        self.assertEqual(phantom_record["global_matrix"].shape, (4, 4))
        self.assertIn("tags", phantom_record)

    def test_dual_camera_no_duplicate_keys_and_gantry_specialization(self) -> None:
        """
        Проверка замечаний 1-4 из metadata_feedback.md:
        - отсутствие дублирующих ключей pose_matrix_GammaCamera_N и crystal_dimensions_GammaCamera_N;
        - корректная запись orbit_radius, frame_duration и angular_step для StepAndShoot;
        - полное отсутствие orbit_radius, angular_step, rotation_direction для CustomSweep.
        """
        root = CompositeNode(name="World")
        gantry = GantryNode(name="Gantry")
        root.add_child(gantry)

        for idx in (1, 2):
            cam = GammaCameraNode(
                name=f"GammaCamera_{idx}",
                slots={"crystal": f"Crystal_{idx}", "collimator": f"Collimator_{idx}"},
            )
            crystal = Volume(
                name=f"Crystal_{idx}",
                geometry=Box(x=540.0, y=400.0, z=9.5),
                material=self.vacuum,
            )
            collimator = ParametricParallelCollimator(
                size=[540.0, 400.0, 35.0],
                hole_diameter=1.5,
                septa=0.2,
                material=self.vacuum,
                hole_shape=CollimatorHoleShape.HEXAGONAL,
                name=f"Collimator_{idx}",
            )
            cam.add_child(crystal)
            cam.add_child(collimator)
            gantry.add_child(cam)

        geom_provider = GeometryMetadataProvider()
        geom_meta = geom_provider.collect(root_scene=root)
        self.assertEqual(geom_meta["detector_heads_count"], 2)
        np.testing.assert_allclose(geom_meta["crystal_dimensions"], [540.0, 400.0, 9.5])
        self.assertNotIn("crystal_dimensions_GammaCamera_1", geom_meta)
        self.assertNotIn("crystal_dimensions_GammaCamera_2", geom_meta)
        self.assertNotIn("collimator_hole_diameter_GammaCamera_1", geom_meta)
        self.assertNotIn("detector_1", geom_meta)

        gantry_provider = GantryMetadataProvider()
        step_protocol = StepAndShootProtocolConfig(
            steps=60,
            start_angle=0.0 * units.deg,
            end_angle=180.0 * units.deg,
            endpoint=False,
            time_per_step=30.0 * units.s,
            radius=258.75 * units.mm,
        )
        gantry_meta = gantry_provider.collect(root_scene=root, protocol=step_protocol)
        self.assertAlmostEqual(gantry_meta["orbit_radius"], 258.75)
        self.assertAlmostEqual(gantry_meta["frame_duration"], 30.0)
        self.assertAlmostEqual(gantry_meta["angular_step"], 3.0)
        self.assertIn("pose_matrix_detector_1", gantry_meta)
        self.assertIn("pose_matrix_detector_2", gantry_meta)
        self.assertEqual(gantry_meta["detector_1_name"], "GammaCamera_1")
        self.assertEqual(gantry_meta["detector_2_name"], "GammaCamera_2")
        self.assertNotIn("pose_matrix_GammaCamera_1", gantry_meta)
        self.assertNotIn("pose_matrix_GammaCamera_2", gantry_meta)

        sweep_protocol = CustomSweepProtocolConfig(
            zipped_variables={"gantry_angle": [0.0, float(np.deg2rad(3.0))]}
        )
        sim_cfg = SimulationManagerConfig(particles_number=100, start_time=0.0, stop_time=30.0 * units.s)
        gantry_sweep_meta = gantry_provider.collect(
            root_scene=root,
            protocol=sweep_protocol,
            context={"gantry_angle": float(np.deg2rad(3.0))},
            sim_config=sim_cfg,
        )
        self.assertNotIn("orbit_radius", gantry_sweep_meta)
        self.assertNotIn("angular_step", gantry_sweep_meta)
        self.assertNotIn("rotation_direction", gantry_sweep_meta)
        self.assertAlmostEqual(gantry_sweep_meta["frame_duration"], 30.0)

    def test_source_metadata_half_life_rounding(self) -> None:
        """
        Проверка замечания 5 из metadata_feedback.md:
        округление периода полураспада 6.89 суток до точного 595296.0 с без артефактов IEEE-754.
        """
        root = CompositeNode(name="World")
        tb161_source = Source(
            distribution=np.ones((2, 2, 2)),
            activity=100.0 * units.MBq,
            half_life=6.89 * units.day,
        )
        root.add_child(tb161_source)

        provider = SourceMetadataProvider()
        source_meta = provider.collect(root_scene=root)
        self.assertEqual(source_meta["half_life"], 595296.0)
        self.assertEqual(source_meta["simulated_activity"], 100_000_000.0)
        self.assertNotIn("radionuclide", source_meta)
        self.assertNotIn("phantom_type", source_meta)

    def test_procedure_metadata_collector_facade(self) -> None:
        """
        Проверка работы фасадного класса ProcedureMetadataCollector (без дублирующего acquisition).
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

        protocol = StepAndShootProtocolConfig(steps=32)

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
        self.assertIn("geometry", meta)
        self.assertNotIn("acquisition", meta)

        self.assertEqual(meta["protocol"]["modality"], "StepAndShoot")
        self.assertEqual(meta["kinematics"]["gantry_name"], "Gantry")
        self.assertIn("Head1", meta["detectors"])
        self.assertIn("volumes", meta["geometry"])
        self.assertIn("Crystal1", meta["geometry"]["volumes"])

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
        self.assertNotIn("volume_metadata", handler.__dict__)
        self.assertNotIn("_build_volume_metadata", SensitiveVolumeHandler.__dict__)
        self.assertNotIn("_apply_volume_attributes", SensitiveVolumeHandler.__dict__)

        history_handler = HistoryAssemblerHandler(sensitive_volumes=[vol])
        self.assertNotIn("volume_metadata", history_handler.__dict__)
        self.assertNotIn("_build_volume_metadata", HistoryAssemblerHandler.__dict__)


if __name__ == "__main__":
    unittest.main()
