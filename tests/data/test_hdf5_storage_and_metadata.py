"""
Модуль тестирования оптимизации формата хранения HDF5 и паспорта метаданных (NMSimToolkit).

Проверяет:
1. Приведение volume_id и material_id к uint32 при записи в HDF5;
2. Обязательное сохранение массива scattering_angles (N, 2);
3. Полное исключение PROCESS_MAP из обработчиков данных и динамическое формирование
   attrs["mapping"] для process_code (uint8) и species_code (uint8);
4. Настраиваемую точность floating_point_precision (float32 vs float64);
5. Поддержку компрессии gzip/lzf и чанкирования датасетов;
6. Структурирование 4 канонических групп metadata (gantry, geometry, source, simulation)
   с единицами измерения hepunits и сохранением обратной совместимости tasks/task_{id}.
"""

import json
import os
import pathlib
import tempfile
import unittest

import h5py
import hepunits as units
import numpy as np

from core.config.models import (
    BoxConfig,
    CustomSweepProtocolConfig,
    DataManagerConfig,
    DirectParallelCollimatorConfig,
    GammaCameraConfig,
    GantryConfig,
    NumpyDistributionConfig,
    SimulationConfig,
    SimulationManagerConfig,
    SourceConfig,
    StepAndShootProtocolConfig,
    VolumeConfig,
    VoxelSpatialConfig,
)
from core.data.data_handlers import (
    DirectStreamHandler,
    HistoryAssemblerHandler,
    SensitiveVolumeHandler,
)
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.data.metadata_collector import ProcedureMetadataCollector
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.physics.interaction_buffers import ChunkType
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PairProduction,
    PhotoelectricEffect,
    get_process_mapping,
    get_species_mapping,
)
from core.config.simulation_worker import simulation_worker_task
from core.scene.dose_grid_node import DoseGridNode
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode
from core.scene.nodes import CompositeNode
from core.source.sources import Source
from core.transport.simulation_managers import SimulationManager


class TestHdf5StorageAndMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "output_spec.h5")
        self.air = Material(name="Air", density=0.0012)
        self.nai = Material(name="NaI", density=3.67)
        self.lead = Material(name="Pb", density=11.34)

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir.name):
            self.temp_dir.cleanup()

    def test_process_map_completely_eliminated_and_dynamic_mapping(self) -> None:
        """
        Проверка: PROCESS_MAP полностью исключен из класса SensitiveVolumeHandler,
        а маппинг процессов извлекается динамически из классов core.physics.processes.
        """
        self.assertNotIn(
            "PROCESS_MAP",
            SensitiveVolumeHandler.__dict__,
            "Статический словарь PROCESS_MAP должен быть полностью исключен из SensitiveVolumeHandler!",
        )

        proc_mapping = get_process_mapping()
        self.assertIn(0, proc_mapping)
        self.assertIn(1, proc_mapping)
        self.assertIn(2, proc_mapping)
        self.assertIn(3, proc_mapping)
        self.assertEqual(proc_mapping[0], PhotoelectricEffect.__name__)
        self.assertEqual(proc_mapping[1], ComptonScattering.__name__)
        self.assertEqual(proc_mapping[2], CoherentScattering.__name__)
        self.assertEqual(proc_mapping[3], PairProduction.__name__)

        species_mapping = get_species_mapping()
        self.assertEqual(species_mapping[0], "Photon")
        self.assertEqual(species_mapping[1], "Electron")
        self.assertEqual(species_mapping[2], "Positron")
        self.assertEqual(species_mapping[3], "Alpha")

    def test_uint32_ids_and_scattering_angles_preserved(self) -> None:
        """
        Проверка: volume_id и material_id ВСЕГДА сохраняются как uint32,
        scattering_angles сохраняется как (N, 2),
        process_code и species_code сохраняются как uint8 с attrs['mapping'].
        """
        world = Volume(geometry=Box(500.0, 500.0, 500.0), material=self.air, name="World")
        crystal = Volume(geometry=Box(400.0, 400.0, 10.0), material=self.nai, name="Crystal")
        world.add_child(crystal)

        handler = SensitiveVolumeHandler(
            sensitive_volumes=[crystal],
            floating_point_precision="float32",
        )
        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        crystal_id = handler.target_volumes[0]

        test_events_count = 5
        handler.process_chunk({
            "type": ChunkType.INTERACTIONS,
            "data": {
                "particle_ID": np.arange(100, 100 + test_events_count, dtype=np.uint64),
                "volume_id": np.full(test_events_count, crystal_id, dtype=np.int64),
                "pos_x": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
                "pos_y": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                "pos_z": np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64),
                "dir_x": np.zeros(test_events_count, dtype=np.float64),
                "dir_y": np.zeros(test_events_count, dtype=np.float64),
                "dir_z": np.ones(test_events_count, dtype=np.float64),
                "energy_deposit": np.array([140.0, 70.0, 30.0, 10.0, 5.0], dtype=np.float64),
                "process_id": np.array([0, 1, 2, 0, 1], dtype=np.uint8),
                "scattering_theta": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
                "scattering_phi": np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64),
                "material_id": np.array([12, 12, 12, 12, 12], dtype=np.int64),
                "Z": np.full(test_events_count, 53, dtype=np.int8),
                "species": np.array([0, 0, 1, 0, 2], dtype=np.uint8),
                "distance_traveled": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            },
        })

        handler.finalize()

        with h5py.File(self.h5_path, "w") as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, "r") as h5_file:
            c_group = h5_file["interactions/Crystal"]

            # 1. Проверка uint32 для volume_id и material_id
            self.assertEqual(c_group["volume_id"].dtype, np.uint32)
            self.assertEqual(c_group["material_id"].dtype, np.uint32)
            self.assertEqual(int(c_group["volume_id"][0]), int(crystal_id))
            self.assertEqual(int(c_group["material_id"][0]), 12)

            # 2. Проверка scattering_angles
            self.assertIn("scattering_angles", c_group)
            self.assertEqual(c_group["scattering_angles"].shape, (test_events_count, 2))
            self.assertEqual(c_group["scattering_angles"].dtype, np.float32)
            self.assertEqual(c_group["scattering_angles"].attrs["units"], "rad")
            np.testing.assert_allclose(
                c_group["scattering_angles"][0],
                [0.1, 1.1],
                atol=1e-5,
            )

            # 3. Проверка process_code uint8 + mapping
            self.assertIn("process_code", c_group)
            self.assertEqual(c_group["process_code"].dtype, np.uint8)
            self.assertIn("mapping", c_group["process_code"].attrs)
            saved_proc_map = json.loads(c_group["process_code"].attrs["mapping"])
            self.assertEqual(saved_proc_map["0"], "PhotoelectricEffect")
            self.assertEqual(saved_proc_map["1"], "ComptonScattering")

            # 4. Проверка species_code uint8 + mapping
            self.assertIn("species_code", c_group)
            self.assertEqual(c_group["species_code"].dtype, np.uint8)
            self.assertIn("mapping", c_group["species_code"].attrs)
            saved_spec_map = json.loads(c_group["species_code"].attrs["mapping"])
            self.assertEqual(saved_spec_map["0"], "Photon")
            self.assertEqual(saved_spec_map["1"], "Electron")

            # 5. Проверка float32 по умолчанию
            self.assertEqual(c_group["global_position"].dtype, np.float32)
            self.assertEqual(c_group["global_direction"].dtype, np.float32)
            self.assertEqual(c_group["energy_deposit"].dtype, np.float32)
            self.assertEqual(c_group["distance_traveled"].dtype, np.float32)

            # 6. Проверка отсутствия старых полей
            self.assertNotIn("process_name", c_group)
            self.assertNotIn("species", c_group)

    def test_float64_precision_mode(self) -> None:
        """
        Проверка: при floating_point_precision='float64' вещественные массивы сохраняются в float64.
        """
        world = Volume(geometry=Box(500.0, 500.0, 500.0), material=self.air, name="World")
        crystal = Volume(geometry=Box(400.0, 400.0, 10.0), material=self.nai, name="Crystal")
        world.add_child(crystal)

        handler = SensitiveVolumeHandler(
            sensitive_volumes=[crystal],
            floating_point_precision="float64",
        )
        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        crystal_id = handler.target_volumes[0]
        handler.process_chunk({
            "type": ChunkType.INTERACTIONS,
            "data": {
                "particle_ID": np.array([1], dtype=np.uint64),
                "volume_id": np.array([crystal_id], dtype=np.int64),
                "pos_x": np.array([12.3456789], dtype=np.float64),
                "pos_y": np.array([23.4567891], dtype=np.float64),
                "pos_z": np.array([34.5678912], dtype=np.float64),
                "dir_x": np.array([0.0], dtype=np.float64),
                "dir_y": np.array([0.0], dtype=np.float64),
                "dir_z": np.array([1.0], dtype=np.float64),
                "energy_deposit": np.array([140.5], dtype=np.float64),
                "process_id": np.array([0], dtype=np.uint8),
                "scattering_theta": np.array([0.123456], dtype=np.float64),
                "scattering_phi": np.array([0.654321], dtype=np.float64),
                "material_id": np.array([9], dtype=np.int64),
                "Z": np.array([53], dtype=np.int8),
                "species": np.array([0], dtype=np.uint8),
                "distance_traveled": np.array([10.5], dtype=np.float64),
            },
        })

        handler.finalize()

        with h5py.File(self.h5_path, "w") as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, "r") as h5_file:
            c_group = h5_file["interactions/Crystal"]
            self.assertEqual(c_group["global_position"].dtype, np.float64)
            self.assertEqual(c_group["global_direction"].dtype, np.float64)
            self.assertEqual(c_group["energy_deposit"].dtype, np.float64)
            self.assertEqual(c_group["distance_traveled"].dtype, np.float64)
            self.assertEqual(c_group["scattering_angles"].dtype, np.float64)
            self.assertEqual(c_group["volume_id"].dtype, np.uint32)
            self.assertEqual(c_group["material_id"].dtype, np.uint32)

    def test_compression_and_chunking_enabled(self) -> None:
        """
        Проверка: включение сжатия compression='gzip', compression_opts=4
        применяется ко всем сохраняемым датасетам.
        """
        stream_handler = DirectStreamHandler(
            compression="gzip",
            compression_opts=4,
        )
        written_ops = []
        stream_handler.set_writer_callback(lambda op: written_ops.append(op))

        stream_handler.process_chunk({
            "type": ChunkType.INITIAL_STATES,
            "data": {
                "particle_ID": np.arange(100, dtype=np.uint64),
                "emission_position": np.zeros((100, 3), dtype=np.float32),
                "emission_direction": np.ones((100, 3), dtype=np.float32),
                "emission_energy": np.full(100, 140.0, dtype=np.float32),
                "emission_time": np.zeros(100, dtype=np.float32),
            },
        })

        stream_handler.finalize()

        with h5py.File(self.h5_path, "w") as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, "r") as h5_file:
            init_group = h5_file["initial_states"]
            self.assertIn("emission_energy", init_group)
            dset = init_group["emission_energy"]
            self.assertEqual(dset.compression, "gzip")
            self.assertEqual(dset.compression_opts, 4)
            self.assertIsNotNone(dset.chunks)
            self.assertEqual(dset.attrs["units"], "MeV")

    def test_canonical_metadata_hierarchy_and_backward_compatibility(self) -> None:
        """
        Проверка сбора метаданных во все 4 канонические группы:
        /metadata/gantry, /metadata/geometry, /metadata/source, /metadata/simulation
        и сохранение обратной совместимости с /metadata/tasks/task_{id}.
        """
        # Построение сцены
        world = CompositeNode(name="World")

        gantry = GantryNode(name="Gantry")
        gantry.set_rotation_angle(np.pi / 3.0)  # 60 градусов

        camera = GammaCameraNode(
            name="Detector_1",
            slots={"crystal": "Crystal_1", "collimator": "Collimator_1"},
        )
        crystal = Volume(
            name="Crystal_1",
            geometry=Box(x=500.0, y=400.0, z=9.5),
            material=self.nai,
        )
        collimator = ParametricParallelCollimator(
            size=[500.0, 400.0, 35.0],
            hole_diameter=1.2,
            septa=0.2,
            material=self.lead,
            hole_shape=CollimatorHoleShape.HEXAGONAL,
            name="Collimator_1",
        )
        camera.add_child(crystal)
        camera.add_child(collimator)
        gantry.add_child(camera)
        world.add_child(gantry)

        source_node = Source.from_voxels(
            distribution=np.ones((2, 2, 2)),
            activity=50.0 * units.MBq,
            half_life=6.0 * units.hour,
            name="Source_Nuclide",
        )
        world.add_child(source_node)

        # Конфигурация
        protocol_cfg = StepAndShootProtocolConfig(
            steps=30,
            start_angle=0.0 * units.deg,
            end_angle=360.0 * units.deg,
            rotation_direction="CCW",
            time_per_step=15.0 * units.s,
            radius=250.0 * units.mm,
        )
        sim_mgr_cfg = SimulationManagerConfig(
            particles_number=50_000,
            physics_list="StandardEM_SPECT",
        )
        source_pydantic_cfg = SourceConfig(
            spatial=VoxelSpatialConfig(distribution=NumpyDistributionConfig(path="source.npy")),
            radionuclide="161Tb",
            phantom_type="NEMA IEC Body Phantom",
        )

        collector = ProcedureMetadataCollector()
        metadata_dict = collector.collect(
            root_scene=world,
            protocol=protocol_cfg,
            context={"view_index": 5, "gantry_angle": np.pi / 3.0},
            task_id=5,
            sim_config=sim_mgr_cfg,
            scene_config=source_pydantic_cfg,
            random_seed=12345,
            precision_mode="single",
        )

        # Инициализация и запись через DataManager
        dm = DataManager(
            filename=self.h5_path,
            handlers=[],
            metadata=metadata_dict,
            task_id=5,
        )
        dm.initialize_metadata()
        dm.finalize_metadata(status="completed")

        with h5py.File(self.h5_path, "r") as h5_file:
            self.assertIn("metadata", h5_file)
            meta = h5_file["metadata"]

            # 1. Группа gantry
            self.assertIn("gantry", meta)
            gantry_grp = meta["gantry"]
            self.assertAlmostEqual(float(gantry_grp["gantry_angle"][()]), np.pi / 3.0)
            self.assertEqual(gantry_grp["gantry_angle"].attrs["units"], "rad")
            self.assertAlmostEqual(float(gantry_grp["orbit_radius"][()]), 250.0)
            self.assertEqual(gantry_grp["orbit_radius"].attrs["units"], "mm")
            self.assertEqual(gantry_grp.attrs["rotation_direction"], "CCW")
            self.assertAlmostEqual(float(gantry_grp["frame_duration"][()]), 15.0)
            self.assertIn("pose_matrix_detector_1", gantry_grp)
            self.assertNotIn("pose_matrix_Detector_1", gantry_grp)
            self.assertEqual(gantry_grp.attrs["detector_1_name"], "Detector_1")
            self.assertEqual(gantry_grp["pose_matrix_detector_1"].shape, (4, 4))

            # 2. Группа geometry
            self.assertIn("geometry", meta)
            geom_grp = meta["geometry"]
            self.assertEqual(int(geom_grp["detector_heads_count"][()]), 1)
            np.testing.assert_allclose(
                geom_grp["crystal_dimensions"][()],
                [500.0, 400.0, 9.5],
            )
            self.assertEqual(geom_grp["crystal_dimensions"].attrs["units"], "mm")
            self.assertEqual(geom_grp.attrs["collimator_type"], "Parallel Hole")
            self.assertAlmostEqual(float(geom_grp["collimator_hole_diameter"][()]), 1.2)
            self.assertAlmostEqual(float(geom_grp["collimator_hole_length"][()]), 35.0)
            self.assertAlmostEqual(float(geom_grp["collimator_septa_thickness"][()]), 0.2)
            self.assertEqual(geom_grp.attrs["collimator_hole_shape"], "hexagonal")
            self.assertEqual(geom_grp.attrs["collimator_material"], "Pb")
            self.assertNotIn("crystal_dimensions_Detector_1", geom_grp)
            self.assertIn("volumes", geom_grp)
            self.assertIn("Crystal_1", geom_grp["volumes"])

            # 2b. Группа protocol
            self.assertIn("protocol", meta)
            proto_grp = meta["protocol"]
            self.assertAlmostEqual(float(proto_grp["exposure_time"][()]), 15.0)
            self.assertEqual(proto_grp["exposure_time"].attrs["units"], "s")
            self.assertAlmostEqual(float(proto_grp["orbit_radius"][()]), 250.0)
            self.assertEqual(proto_grp["orbit_radius"].attrs["units"], "mm")

            # 3. Группа source
            self.assertIn("source", meta)
            src_grp = meta["source"]
            self.assertEqual(src_grp.attrs["radionuclide"], "161Tb")
            self.assertEqual(src_grp.attrs["phantom_type"], "NEMA IEC Body Phantom")
            self.assertAlmostEqual(float(src_grp["half_life"][()]), 6.0 * 3600.0)
            self.assertAlmostEqual(float(src_grp["simulated_activity"][()]), 50.0 * 1e6)
            self.assertEqual(src_grp["half_life"].attrs["units"], "s")
            self.assertEqual(src_grp["simulated_activity"].attrs["units"], "Bq")

            # 4. Группа simulation
            self.assertIn("simulation", meta)
            sim_grp = meta["simulation"]
            self.assertEqual(sim_grp.attrs["code_name"], "NMSimToolkit")
            self.assertEqual(sim_grp.attrs["code_version"], "0.2.0")
            self.assertEqual(sim_grp.attrs["physics_list"], "StandardEM_SPECT")
            self.assertEqual(int(sim_grp["simulated_histories"][()]), 50_000)
            self.assertEqual(int(sim_grp["random_seed"][()]), 12345)
            self.assertEqual(sim_grp.attrs["precision_mode"], "single")
            self.assertEqual(sim_grp.attrs["status"], "completed")

            # 5. Запись задачи tasks/task_5 без дублирования корневых секций
            self.assertIn("tasks", meta)
            self.assertIn("task_5", meta["tasks"])
            task_5_group = meta["tasks/task_5"]
            self.assertEqual(task_5_group.attrs["task_id"], 5)
            self.assertEqual(task_5_group.attrs["status"], "completed")
            self.assertIn("start_time", task_5_group.attrs)
            self.assertIn("completion_time", task_5_group.attrs)
            self.assertNotIn("acquisition", task_5_group)
            self.assertNotIn("gantry", task_5_group)
            self.assertNotIn("geometry", task_5_group)
            self.assertNotIn("source", task_5_group)
            self.assertNotIn("simulation", task_5_group)
            self.assertIn("protocol", task_5_group)
            self.assertIn("kinematics", task_5_group)
            self.assertIn("detectors", task_5_group)

    def test_direct_stream_handler_raw_chunk_formatting(self) -> None:
        """
        Проверка: DirectStreamHandler корректно приводит «сырые» SoA-буферы симулятора
        к канонической структуре HDF5 (uint32 для volume_id и material_id, scattering_angles (N, 2),
        process_code и species_code uint8, emission_position и emission_direction (N, 3)).
        """
        handler = DirectStreamHandler(
            floating_point_precision="float32",
            compression="gzip",
            compression_opts=4,
        )
        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        # 1. Посылаем сырой чанк initial_states с раздельными координатами
        handler.process_chunk({
            "type": ChunkType.INITIAL_STATES,
            "data": {
                "particle_ID": np.array([501, 502], dtype=np.uint64),
                "emission_time": np.array([0.0, 1.0], dtype=np.float64),
                "emission_energy": np.array([140.5, 140.5], dtype=np.float64),
                "pos_x": np.array([10.0, 20.0], dtype=np.float64),
                "pos_y": np.array([30.0, 40.0], dtype=np.float64),
                "pos_z": np.array([50.0, 60.0], dtype=np.float64),
                "dir_x": np.array([0.0, 0.0], dtype=np.float64),
                "dir_y": np.array([0.0, 0.0], dtype=np.float64),
                "dir_z": np.array([1.0, 1.0], dtype=np.float64),
            },
        })

        # 2. Посылаем сырой чанк interactions из SoA буфера
        handler.process_chunk({
            "type": ChunkType.INTERACTIONS,
            "data": {
                "particle_ID": np.array([501, 502], dtype=np.uint64),
                "volume_id": np.array([10, 20], dtype=np.int64),
                "material_id": np.array([1, 2], dtype=np.int64),
                "pos_x": np.array([11.0, 21.0], dtype=np.float64),
                "pos_y": np.array([31.0, 41.0], dtype=np.float64),
                "pos_z": np.array([51.0, 61.0], dtype=np.float64),
                "dir_x": np.array([0.1, 0.2], dtype=np.float64),
                "dir_y": np.array([0.3, 0.4], dtype=np.float64),
                "dir_z": np.array([0.9, 0.8], dtype=np.float64),
                "energy_deposit": np.array([10.0, 20.0], dtype=np.float64),
                "distance_traveled": np.array([5.0, 6.0], dtype=np.float64),
                "process_id": np.array([1, 2], dtype=np.int32),
                "species": np.array([0, 1], dtype=np.int32),
                "Z": np.array([53, 82], dtype=np.int32),
                "scattering_theta": np.array([0.15, 0.25], dtype=np.float64),
                "scattering_phi": np.array([0.35, 0.45], dtype=np.float64),
            },
        })

        handler.finalize()

        with h5py.File(self.h5_path, "w") as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, "r") as h5_file:
            # Проверка initial_states
            self.assertIn("initial_states", h5_file)
            init_grp = h5_file["initial_states"]
            self.assertIn("emission_position", init_grp)
            self.assertEqual(init_grp["emission_position"].shape, (2, 3))
            self.assertEqual(init_grp["emission_position"].dtype, np.float32)
            self.assertIn("emission_direction", init_grp)
            self.assertEqual(init_grp["emission_direction"].shape, (2, 3))
            self.assertEqual(init_grp["emission_direction"].dtype, np.float32)
            self.assertIn("emission_time", init_grp)
            self.assertEqual(init_grp["emission_time"].dtype, np.float64)
            self.assertEqual(init_grp["emission_time"].attrs["units"], "ns")
            self.assertIn("emission_energy", init_grp)
            self.assertEqual(init_grp["emission_energy"].dtype, np.float32)

            # Проверка interactions/All_Volumes
            self.assertIn("interactions/All_Volumes", h5_file)
            vol_grp = h5_file["interactions/All_Volumes"]
            self.assertEqual(vol_grp["volume_id"].dtype, np.uint32)
            self.assertEqual(vol_grp["material_id"].dtype, np.uint32)
            self.assertEqual(vol_grp["process_code"].dtype, np.uint8)
            self.assertEqual(vol_grp["species_code"].dtype, np.uint8)
            self.assertEqual(vol_grp["global_position"].shape, (2, 3))
            self.assertEqual(vol_grp["global_position"].dtype, np.float32)
            self.assertEqual(vol_grp["global_direction"].shape, (2, 3))
            self.assertEqual(vol_grp["global_direction"].dtype, np.float32)
            self.assertEqual(vol_grp["scattering_angles"].shape, (2, 2))
            self.assertEqual(vol_grp["scattering_angles"].dtype, np.float32)
            self.assertEqual(vol_grp["scattering_angles"].attrs["units"], "rad")
            self.assertIn("mapping", vol_grp["process_code"].attrs)
            self.assertIn("mapping", vol_grp["species_code"].attrs)

    def test_strict_no_duck_typing_in_core(self) -> None:
        """
        Проверка: в каталоге core/ полностью отсутствуют вызовы рефлексии
        в соответствии с правилом 02_code_contracts.md.
        """
        core_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "core"
        py_files = list(core_dir.rglob("*.py"))
        self.assertGreater(len(py_files), 10, "Каталог core должен содержать файлы исходного кода Python")

        forbidden_get = "get" + "attr("
        forbidden_has = "has" + "attr("
        for py_path in py_files:
            content = py_path.read_text(encoding="utf-8")
            self.assertNotIn(
                forbidden_get,
                content,
                f"Обнаружен запрещенный вызов рефлексии в файле {py_path}",
            )
            self.assertNotIn(
                forbidden_has,
                content,
                f"Обнаружен запрещенный вызов рефлексии в файле {py_path}",
            )

    def test_history_assembler_finalize_and_dose_map_accumulation(self) -> None:
        """
        Регрессионный тест HistoryAssemblerHandler.finalize() и DoseMapHandler.finalize():
        1. HistoryAssemblerHandler.finalize() сбрасывает в HDF5 истории всех частиц,
           попавших в детектор, но не получивших чанк dead_particles до завершения симуляции,
           и полностью очищает внутренние накопители.
        2. Два последовательных вызова DoseMapHandler.finalize() в один HDF5-файл
           суммируют энерговыделение в существующем датасете /dose/<name>, а не затирают его.
        """
        world = Volume(geometry=Box(500.0, 500.0, 500.0), material=self.air, name="World")
        crystal = Volume(geometry=Box(400.0, 400.0, 10.0), material=self.nai, name="Crystal")
        world.add_child(crystal)

        assembler = HistoryAssemblerHandler(sensitive_volumes=[crystal], save_initial_states=True)
        written_ops = []
        assembler.set_writer_callback(lambda op: written_ops.append(op))

        crystal_id = assembler.target_volumes[0]

        # Посылаем начальное состояние и взаимодействие частицы 777 без чанка dead_particles
        assembler.process_chunk({
            "type": ChunkType.INITIAL_STATES,
            "data": {
                "particle_ID": np.array([777, 888], dtype=np.uint64),
                "emission_time": np.array([0.0, 1.0], dtype=np.float64),
                "emission_energy": np.array([0.1405, 0.1405], dtype=np.float64),
                "pos_x": np.array([0.0, 0.0], dtype=np.float64),
                "pos_y": np.array([0.0, 0.0], dtype=np.float64),
                "pos_z": np.array([0.0, 0.0], dtype=np.float64),
                "dir_x": np.array([0.0, 0.0], dtype=np.float64),
                "dir_y": np.array([0.0, 0.0], dtype=np.float64),
                "dir_z": np.array([1.0, 1.0], dtype=np.float64),
            },
        })
        assembler.process_chunk({
            "type": ChunkType.INTERACTIONS,
            "data": {
                "particle_ID": np.array([777], dtype=np.uint64),
                "volume_id": np.array([crystal_id], dtype=np.int64),
                "material_id": np.array([1], dtype=np.int64),
                "pos_x": np.array([1.0], dtype=np.float64),
                "pos_y": np.array([2.0], dtype=np.float64),
                "pos_z": np.array([3.0], dtype=np.float64),
                "dir_x": np.array([0.0], dtype=np.float64),
                "dir_y": np.array([0.0], dtype=np.float64),
                "dir_z": np.array([1.0], dtype=np.float64),
                "energy_deposit": np.array([0.1405], dtype=np.float64),
                "distance_traveled": np.array([10.0], dtype=np.float64),
                "process_id": np.array([0], dtype=np.uint8),
                "species": np.array([0], dtype=np.uint8),
                "Z": np.array([53], dtype=np.int8),
                "scattering_theta": np.array([0.0], dtype=np.float64),
                "scattering_phi": np.array([0.0], dtype=np.float64),
            },
        })

        # До вызова finalize() записи еще не было
        self.assertEqual(len(written_ops), 0)
        assembler.finalize()
        self.assertGreater(len(written_ops), 0)
        self.assertEqual(len(assembler.initial_states_chunks), 0)
        self.assertEqual(len(assembler.interactions_chunks), 0)
        self.assertEqual(len(assembler.scored_particles), 0)

        # Проверяем также суммирование дозы при двух вызовах DoseMapHandler.finalize()
        dose_node = DoseGridNode(name="PhantomDose", size=(20.0, 20.0, 20.0), dose_voxel_size=10.0)
        world.add_child(dose_node)
        dose_handler = DoseMapHandler(grid_nodes=[dose_node], create_shm=False)
        dose_handler.set_writer_callback(lambda op: written_ops.append(op))

        try:
            dose_handler.process_chunk({
                "type": ChunkType.INTERACTIONS,
                "data": {
                    "pos_x": np.array([5.0], dtype=np.float64),
                    "pos_y": np.array([5.0], dtype=np.float64),
                    "pos_z": np.array([5.0], dtype=np.float64),
                    "energy_deposit": np.array([2.5], dtype=np.float64),
                },
            })
            # Первый вызов finalize() записывает накопленную дозу 2.5 МэВ
            dose_handler.finalize()
            # Второй вызов finalize() (например, для следующего пакета/задачи) должен прибавить еще 2.5 МэВ -> 5.0 МэВ
            dose_handler.finalize()

            with h5py.File(self.h5_path, "w") as h5_file:
                for op in written_ops:
                    op(h5_file)

            with h5py.File(self.h5_path, "r") as h5_file:
                self.assertIn("initial_states", h5_file)
                self.assertEqual(list(h5_file["initial_states/particle_ID"][:]), [777])
                self.assertIn("interactions/Crystal", h5_file)
                self.assertEqual(list(h5_file["interactions/Crystal/particle_ID"][:]), [777])

                self.assertIn("dose/PhantomDose", h5_file)
                saved_dose = h5_file["dose/PhantomDose"][:]
                self.assertAlmostEqual(float(np.sum(saved_dose)), 5.0, places=6)
                self.assertAlmostEqual(float(saved_dose[1, 1, 1]), 5.0, places=6)
        finally:
            dose_handler.close()

    def test_metadata_specialization_feedback_remarks_in_hdf5(self) -> None:
        """
        Сквозная проверка всех 5 замечаний из docs/plans/metadata_feedback.md в HDF5-файле:
        1-3. Отсутствие нулевых orbit_radius, angular_step, frame_duration при CustomSweep,
             и точные значения при StepAndShoot;
        4. Отсутствие дубликатов pose_matrix_GammaCamera_N и crystal_dimensions_GammaCamera_N;
        5. Точное округление half_life = 595296.0 с для 161Tb (6.89 сут).
        """
        world = CompositeNode(name="World")
        gantry = GantryNode(name="Gantry")
        world.add_child(gantry)

        for idx in (1, 2):
            camera = GammaCameraNode(
                name=f"GammaCamera_{idx}",
                slots={"crystal": f"Crystal_{idx}", "collimator": f"Collimator_{idx}"},
            )
            crystal = Volume(
                name=f"Crystal_{idx}",
                geometry=Box(x=540.0, y=400.0, z=9.5),
                material=self.nai,
            )
            collimator = ParametricParallelCollimator(
                size=[540.0, 400.0, 35.0],
                hole_diameter=1.5,
                septa=0.2,
                material=self.lead,
                hole_shape=CollimatorHoleShape.HEXAGONAL,
                name=f"Collimator_{idx}",
            )
            camera.add_child(crystal)
            camera.add_child(collimator)
            gantry.add_child(camera)

        tb_source = Source.from_voxels(
            distribution=np.ones((2, 2, 2)),
            activity=100.0 * units.MBq,
            half_life=6.89 * units.day,
        )
        world.add_child(tb_source)

        collector = ProcedureMetadataCollector()

        # А. Протокол CustomSweep без заданного времени экспозиции -> нулевые поля НЕ создаются
        sweep_protocol = CustomSweepProtocolConfig(
            zipped_variables={"gantry_angle": [0.0, float(np.deg2rad(3.0))]}
        )
        sweep_meta = collector.collect(
            root_scene=world,
            protocol=sweep_protocol,
            context={"gantry_angle": 0.0, "view_index": 0},
            task_id=0,
        )
        sweep_h5_path = os.path.join(self.temp_dir.name, "sweep_spec.h5")
        dm_sweep = DataManager(filename=sweep_h5_path, handlers=[], metadata=sweep_meta, task_id=0)
        dm_sweep.initialize_metadata()

        with h5py.File(sweep_h5_path, "r") as h5_file:
            gantry_grp = h5_file["metadata/gantry"]
            proto_grp = h5_file["metadata/protocol"]
            geom_grp = h5_file["metadata/geometry"]
            src_grp = h5_file["metadata/source"]

            self.assertNotIn("orbit_radius", gantry_grp)
            self.assertNotIn("orbit_radius", gantry_grp.attrs)
            self.assertNotIn("angular_step", gantry_grp)
            self.assertNotIn("rotation_direction", gantry_grp.attrs)
            self.assertNotIn("frame_duration", gantry_grp)
            self.assertNotIn("exposure_time", proto_grp)
            self.assertNotIn("orbit_radius", proto_grp)

            self.assertIn("pose_matrix_detector_1", gantry_grp)
            self.assertIn("pose_matrix_detector_2", gantry_grp)
            self.assertNotIn("pose_matrix_GammaCamera_1", gantry_grp)
            self.assertNotIn("pose_matrix_GammaCamera_2", gantry_grp)

            self.assertIn("crystal_dimensions", geom_grp)
            self.assertNotIn("crystal_dimensions_GammaCamera_1", geom_grp)
            self.assertNotIn("crystal_dimensions_GammaCamera_2", geom_grp)
            self.assertNotIn("collimator_hole_diameter_GammaCamera_1", geom_grp)

            self.assertEqual(float(src_grp["half_life"][()]), 595296.0)
            self.assertEqual(float(src_grp.attrs["half_life"]), 595296.0)

    def test_simulation_manager_unified_rng(self) -> None:
        """
        Проверка единства генератора случайных чисел (self.rng) в SimulationManager:
        пропагатор, все физические процессы и поток эмиссии используют один и тот же
        экземпляр генератора, а rng, initial_particle_id и rng_state корректно инициализируют состояние.
        """
        world = Volume(
            geometry=Box(200.0, 200.0, 200.0),
            material=Material(name="Air, Dry (near sea level)", density=0.0012),
            name="World",
        )
        src1 = Source.from_point(activity=5000.0 * units.Bq, energy=140.5 * units.keV)
        src2 = Source.from_point(activity=3000.0 * units.Bq, energy=140.5 * units.keV)
        world.add_child(src1)
        world.add_child(src2)

        explicit_rng = np.random.default_rng(555)
        mgr_explicit = SimulationManager(
            scene=world,
            stop_time=0.01 * units.s,
            particles_number=50,
            rng=explicit_rng,
        )
        self.assertIs(mgr_explicit.rng, explicit_rng)
        self.assertIs(mgr_explicit.propagator.rng, explicit_rng)

        ref_rng = np.random.default_rng(777)
        _ = ref_rng.random(10)
        saved_state = ref_rng.bit_generator.state
        expected_next_values = ref_rng.random(5)

        mgr = SimulationManager(
            scene=world,
            stop_time=0.01 * units.s,
            particles_number=50,
            seed=123,
            resume_state={"initial_particle_id": 420, "rng_state": saved_state},
        )

        self.assertIs(mgr.propagator.rng, mgr.rng)
        self.assertEqual(mgr.source_buffer.num_channels, 2)
        self.assertEqual(len(mgr.active_sources), 2)
        self.assertNotIn("emission_rng", mgr.__dict__)
        self.assertNotIn("transport_rng", mgr.__dict__)
        self.assertNotIn("emission_rng_ctx", mgr.__dict__)
        self.assertNotIn("transport_rng_ctx", mgr.__dict__)

        self.assertEqual(mgr.bank.count, 420)
        np.testing.assert_array_equal(mgr.rng.random(5), expected_next_values)

        mgr_resumed = SimulationManager(
            scene=world,
            start_time=100.0 * units.ns,
            stop_time=0.01 * units.s,
            particles_number=50,
            seed=123,
            resume_state={
                "next_particle_id": 420,
                "rng_state": saved_state,
                "t_ref": 0.0,
                "t_last": 95.0 * units.ns,
                "pending_arrival_time": 105.0 * units.ns,
            },
        )
        self.assertEqual(float(mgr_resumed.t_ref), 0.0)
        self.assertEqual(float(mgr_resumed.t_last), float(95.0 * units.ns))
        self.assertEqual(float(mgr_resumed.pending_arrival_time), float(105.0 * units.ns))
        np.testing.assert_array_equal(mgr_resumed.rng.random(5), expected_next_values)

    @staticmethod
    def _build_resume_test_task_dict(h5_filename: str, stop_time_str: str) -> dict:
        """Формирует конфигурацию задачи симуляции для проверки бесшовного досчёта."""
        return {
            "protocol": {
                "type": "CustomSweep",
                "zipped_variables": {"gantry_angle": [0.0]},
            },
            "simulation_manager": {
                "start_time": "0.0 s",
                "stop_time": stop_time_str,
                "particles_number": 50,
                "min_energy": "1.0 keV",
            },
            "data_manager": {
                "filename": h5_filename,
                "floating_point_precision": "float64",
                "handlers": [
                    {"type": "DirectStreamHandler"},
                    {"type": "SensitiveVolumeHandler", "sensitive_volumes": ["Crystal"]},
                ],
            },
            "scene": {
                "type": "Volume",
                "name": "World",
                "geometry": {"type": "Box", "x": "200.0 mm", "y": "200.0 mm", "z": "200.0 mm"},
                "material": "Air, Dry (near sea level)",
                "children": [
                    {
                        "type": "Volume",
                        "name": "Crystal",
                        "geometry": {"type": "Box", "x": "120.0 mm", "y": "120.0 mm", "z": "25.0 mm"},
                        "material": "Sodium Iodide",
                        "transformations": [{"type": "translate", "x": 0.0, "y": 0.0, "z": "30.0 mm"}],
                    },
                    {
                        "type": "Source",
                        "name": "PointSrc",
                        "spatial": {"type": "point"},
                        "activity": "10000 Bq",
                        "half_life": "6.0 h",
                        "channels": [
                            {
                                "species": "photon",
                                "spectrum": {"type": "discrete", "lines": "140.5 keV"},
                            }
                        ],
                    },
                ],
            },
        }

    def test_stateful_resume_bitwise_identity(self) -> None:
        """
        Проверка бесшовного двухэтапного досчёта симуляции (Stateful Resume):
        1. Побитовое совпадение первого этапа с непрерывным эталонным расчётом;
        2. Сквозная непрерывность particle_ID и actual_start_time == actual_stop_time;
        3. Корректная кумулятивная агрегация frame_duration и simulated_histories;
        4. 100% побитовая воспроизводимость (np.array_equal) двухэтапного досчёта.
        """
        ref_h5 = os.path.join(self.temp_dir.name, "ref.h5")
        resumed_h5 = os.path.join(self.temp_dir.name, "resumed.h5")
        resumed_repro_h5 = os.path.join(self.temp_dir.name, "resumed_repro.h5")
        fixed_seed = 12345

        # 1. Эталонный непрерывный прогон [0.0, 0.02 s] -> ref.h5
        ref_task = self._build_resume_test_task_dict(ref_h5, "0.02 s")
        simulation_worker_task((ref_task, fixed_seed, None, None, None, None, (), 0))

        # 2. Двухэтапный прогон: сначала [0.0, 0.01 s], затем досчёт до [0.0, 0.02 s] -> resumed.h5
        stage1_task = self._build_resume_test_task_dict(resumed_h5, "0.01 s")
        simulation_worker_task((stage1_task, fixed_seed, None, None, None, None, (), 0))

        stage2_task = self._build_resume_test_task_dict(resumed_h5, "0.02 s")
        simulation_worker_task((stage2_task, 99999, None, None, None, None, (), 0))

        # 3. Независимый повтор двухэтапного прогона для проверки 100% побитовой детерминированности
        repro_stage1 = self._build_resume_test_task_dict(resumed_repro_h5, "0.01 s")
        simulation_worker_task((repro_stage1, fixed_seed, None, None, None, None, (), 0))

        repro_stage2 = self._build_resume_test_task_dict(resumed_repro_h5, "0.02 s")
        simulation_worker_task((repro_stage2, 77777, None, None, None, None, (), 0))

        with (
            h5py.File(ref_h5, "r") as ref_file,
            h5py.File(resumed_h5, "r") as res_file,
            h5py.File(resumed_repro_h5, "r") as repro_file,
        ):
            # Проверка паспортов задач в resumed.h5
            self.assertIn("task_0", res_file["metadata/tasks"])
            self.assertIn("task_1", res_file["metadata/tasks"])
            task_0_group = res_file["metadata/tasks/task_0"]
            task_1_group = res_file["metadata/tasks/task_1"]
            self.assertEqual(task_0_group.attrs["status"], "completed")
            self.assertEqual(task_1_group.attrs["status"], "completed")
            self.assertEqual(int(task_0_group.attrs["seed"]), fixed_seed)
            self.assertEqual(int(task_1_group.attrs["seed"]), fixed_seed)
            self.assertEqual(float(task_1_group.attrs["actual_start_time"]), float(task_0_group.attrs["actual_stop_time"]))
            self.assertEqual(int(task_1_group.attrs["particle_id_start"]), int(task_0_group.attrs["particle_id_end"]))
            for req_attr in ("rng_state", "t_ref", "t_last", "pending_arrival_time"):
                self.assertIn(req_attr, task_0_group.attrs)
                self.assertIn(req_attr, task_1_group.attrs)
            self.assertNotIn("emission_rng_state", task_0_group.attrs)
            self.assertNotIn("transport_rng_state", task_0_group.attrs)
            self.assertEqual(float(task_0_group.attrs["t_ref"]), 0.0)
            self.assertEqual(float(task_1_group.attrs["t_ref"]), 0.0)
            self.assertGreaterEqual(float(task_0_group.attrs["pending_arrival_time"]), float(0.01 * units.s))
            self.assertGreaterEqual(float(task_1_group.attrs["pending_arrival_time"]), float(0.02 * units.s))
            self.assertTrue(np.all(ref_file["initial_states/emission_time"][:] < float(0.02 * units.s)))
            self.assertTrue(np.all(res_file["initial_states/emission_time"][:] < float(0.02 * units.s)))

            stage1_particles = int(task_0_group.attrs["particle_id_end"])
            total_resumed_particles = int(task_1_group.attrs["particle_id_end"])
            self.assertGreater(stage1_particles, 0)
            self.assertGreater(total_resumed_particles, stage1_particles)

            # Проверка сквозной нумерации particle_ID взаимодействовавших частиц без коллизий между task_0 и task_1
            res_particle_ids = res_file["initial_states/particle_ID"][:]
            self.assertEqual(len(np.unique(res_particle_ids)), len(res_particle_ids))
            self.assertTrue(np.any(res_particle_ids < stage1_particles))
            self.assertTrue(np.any(res_particle_ids >= stage1_particles))
            self.assertTrue(np.all(res_particle_ids < total_resumed_particles))

            # Проверка кумулятивных метаданных в resumed.h5
            expected_total_histories = (
                int(task_0_group.attrs["simulated_histories"]) + int(task_1_group.attrs["simulated_histories"])
            )
            self.assertEqual(expected_total_histories, total_resumed_particles)
            self.assertEqual(int(res_file["metadata/simulation/simulated_histories"][()]), total_resumed_particles)
            self.assertEqual(int(res_file["metadata/simulation"].attrs["simulated_histories"]), total_resumed_particles)

            expected_total_duration = round(
                float(task_0_group.attrs["frame_duration"]) + float(task_1_group.attrs["frame_duration"]), 6
            )
            self.assertAlmostEqual(float(res_file["metadata/gantry/frame_duration"][()]), expected_total_duration, places=6)
            self.assertAlmostEqual(float(res_file["metadata/gantry"].attrs["frame_duration"]), expected_total_duration, places=6)

            # Побитовое совпадение начальных состояний частиц до дренажа последнего батча этапа 1 с эталоном ref.h5
            completed_before_drain_id = max(1, stage1_particles - 50)
            ref_mask_prefix = ref_file["initial_states/particle_ID"][:] < completed_before_drain_id
            res_mask_prefix = res_file["initial_states/particle_ID"][:] < completed_before_drain_id
            self.assertGreater(int(np.count_nonzero(res_mask_prefix)), 0)
            for dset_name in ref_file["initial_states"].keys():
                arr_ref_prefix = ref_file["initial_states"][dset_name][:][ref_mask_prefix]
                arr_res_prefix = res_file["initial_states"][dset_name][:][res_mask_prefix]
                self.assertTrue(
                    np.array_equal(arr_ref_prefix, arr_res_prefix),
                    f"Префикс первого этапа initial_states/{dset_name} не совпал побитово с ref.h5!",
                )

            # 100% побитовое совпадение всех массивов initial_states/* и interactions/* при повторном досчёте
            for dset_name in res_file["initial_states"].keys():
                self.assertTrue(
                    np.array_equal(res_file["initial_states"][dset_name][:], repro_file["initial_states"][dset_name][:]),
                    f"Массив initial_states/{dset_name} не совпал побитово при повторном досчёте!",
                )

            for vol_name in res_file["interactions"].keys():
                grp_res = res_file["interactions"][vol_name]
                grp_repro = repro_file["interactions"][vol_name]
                self.assertGreater(len(grp_res["particle_ID"]), 0)
                for dset_name in grp_res.keys():
                    self.assertTrue(
                        np.array_equal(grp_res[dset_name][:], grp_repro[dset_name][:]),
                        f"Массив interactions/{vol_name}/{dset_name} не совпал побитово при повторном досчёте!",
                    )

    def test_skip_already_completed_time_range(self) -> None:
        """
        Проверка мгновенного пропуска задачи, если целевой интервал времени stop_time
        уже полностью рассчитан в HDF5-файле.
        """
        resumed_h5 = os.path.join(self.temp_dir.name, "skip_test.h5")
        task_cfg = self._build_resume_test_task_dict(resumed_h5, "0.01 s")
        simulation_worker_task((task_cfg, 12345, None, None, None, None, (), 0))

        with h5py.File(resumed_h5, "r") as h5_before:
            histories_before = int(h5_before["metadata/simulation/simulated_histories"][()])
            init_count_before = len(h5_before["initial_states/particle_ID"])
            tasks_count_before = len(h5_before["metadata/tasks"])

        # Повторный запуск с тем же stop_time = 0.01 s должен быть пропущен
        simulation_worker_task((task_cfg, 12345, None, None, None, None, (), 0))

        with h5py.File(resumed_h5, "r") as h5_after:
            self.assertEqual(int(h5_after["metadata/simulation/simulated_histories"][()]), histories_before)
            self.assertEqual(len(h5_after["initial_states/particle_ID"]), init_count_before)
            self.assertEqual(len(h5_after["metadata/tasks"]), tasks_count_before)

    def test_multi_step_protocol_shared_hdf5_resume_and_no_collision(self) -> None:
        """
        Проверка многошагового протокола в одном общем HDF5-файле:
        1. Второй ракурс (view_index=1) НЕ пропускается после завершения первого (view_index=0)
           и начинает нумерацию particle_ID с max_particle_id_end первого ракурса без коллизий;
        2. При увеличении stop_time каждый ракурс досчитывается строго со своего чекпоинта по context.
        """
        shared_h5 = os.path.join(self.temp_dir.name, "multi_step_shared.h5")

        view0_stage1 = self._build_resume_test_task_dict(shared_h5, "0.01 s")
        view0_stage1["_context"] = {"view_index": 0, "gantry_angle": 0.0}
        simulation_worker_task((view0_stage1, 11111, None, None, None, None, (), 0))

        view1_stage1 = self._build_resume_test_task_dict(shared_h5, "0.01 s")
        view1_stage1["_context"] = {"view_index": 1, "gantry_angle": 1.5707963}
        simulation_worker_task((view1_stage1, 22222, None, None, None, None, (), 1))

        with h5py.File(shared_h5, "r") as h5_stage1:
            self.assertIn("task_0", h5_stage1["metadata/tasks"])
            self.assertIn("task_1", h5_stage1["metadata/tasks"])
            task_0_group = h5_stage1["metadata/tasks/task_0"]
            task_1_group = h5_stage1["metadata/tasks/task_1"]
            self.assertEqual(float(task_0_group.attrs["actual_start_time"]), 0.0)
            self.assertEqual(float(task_1_group.attrs["actual_start_time"]), 0.0)
            self.assertEqual(int(task_0_group.attrs["particle_id_start"]), 0)
            self.assertEqual(int(task_1_group.attrs["particle_id_start"]), int(task_0_group.attrs["particle_id_end"]))
            stage1_pids = h5_stage1["initial_states/particle_ID"][:]
            self.assertEqual(len(np.unique(stage1_pids)), len(stage1_pids))

        # Досчёт первого ракурса (view_index=0) до 0.02 s
        view0_stage2 = self._build_resume_test_task_dict(shared_h5, "0.02 s")
        view0_stage2["_context"] = {"view_index": 0, "gantry_angle": 0.0}
        simulation_worker_task((view0_stage2, 99999, None, None, None, None, (), 0))

        with h5py.File(shared_h5, "r") as h5_stage2:
            self.assertIn("task_2", h5_stage2["metadata/tasks"])
            task_0_group = h5_stage2["metadata/tasks/task_0"]
            task_1_group = h5_stage2["metadata/tasks/task_1"]
            task_2_group = h5_stage2["metadata/tasks/task_2"]
            self.assertEqual(int(task_2_group.attrs["seed"]), 11111)
            self.assertEqual(float(task_2_group.attrs["actual_start_time"]), float(task_0_group.attrs["actual_stop_time"]))
            self.assertEqual(int(task_2_group.attrs["particle_id_start"]), int(task_1_group.attrs["particle_id_end"]))
            stage2_pids = h5_stage2["initial_states/particle_ID"][:]
            self.assertEqual(len(np.unique(stage2_pids)), len(stage2_pids))

    def test_inspect_resume_state_fail_fast_on_missing_attributes(self) -> None:
        """
        Проверка Fail-Fast валидации в DataManager.inspect_resume_state:
        отсутствие любого из обязательных атрибутов возобновления (rng_state,
        t_ref, t_last, pending_arrival_time) в завершенной задаче вызывает ValueError.
        """
        base_h5 = os.path.join(self.temp_dir.name, "fail_fast_base.h5")
        task_cfg = self._build_resume_test_task_dict(base_h5, "0.005 s")
        simulation_worker_task((task_cfg, 54321, None, None, None, None, (), 0))

        resume_info = DataManager.inspect_resume_state(base_h5)
        self.assertIsNotNone(resume_info)
        assert resume_info is not None
        self.assertEqual(resume_info["batch_capacity"], 50)

        required_attrs = (
            "rng_state",
            "t_ref",
            "t_last",
            "pending_arrival_time",
            "batch_capacity",
        )
        for missing_attr in required_attrs:
            with h5py.File(base_h5, "a") as h5_file:
                attrs = h5_file["metadata/tasks/task_0"].attrs
                saved_attr_val = attrs[missing_attr]
                del attrs[missing_attr]

            with self.assertRaises(ValueError, msg=f"Ожидался ValueError при отсутствии {missing_attr}"):
                DataManager.inspect_resume_state(base_h5)

            with h5py.File(base_h5, "a") as h5_file:
                h5_file["metadata/tasks/task_0"].attrs[missing_attr] = saved_attr_val

        with self.assertRaises(ValueError):
            DataManager.inspect_resume_state(base_h5, batch_capacity=999)

        mismatch_task = self._build_resume_test_task_dict(base_h5, "0.01 s")
        mismatch_task["simulation_manager"]["batch_capacity"] = 999
        with self.assertRaises(ValueError):
            simulation_worker_task((mismatch_task, 54321, None, None, None, None, (), 0))


if __name__ == "__main__":
    unittest.main()


