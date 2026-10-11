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
import tempfile
import unittest

import h5py
import hepunits as units
import numpy as np

from core.config.models import (
    BoxConfig,
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
)
from core.data.data_handlers import (
    DirectStreamHandler,
    HistoryAssemblerHandler,
    SensitiveVolumeHandler,
)
from core.data.data_manager import DataManager
from core.data.metadata_collector import ProcedureMetadataCollector
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PairProduction,
    PhotoelectricEffect,
    get_process_mapping,
    get_species_mapping,
)
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode
from core.scene.nodes import CompositeNode
from core.source.sources import Source


class TestHdf5StorageAndMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "output_spec.h5")
        self.air = Material(name="Air", density=0.0012)
        self.nai = Material(name="NaI", density=3.67)
        self.lead = Material(name="Pb", density=11.34)

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except (OSError, RuntimeError):
            pass

    def test_process_map_completely_eliminated_and_dynamic_mapping(self) -> None:
        """
        Проверка: PROCESS_MAP полностью исключен из класса SensitiveVolumeHandler,
        а маппинг процессов извлекается динамически из классов core.physics.processes.
        """
        self.assertFalse(
            hasattr(SensitiveVolumeHandler, "PROCESS_MAP"),
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
            "type": "interactions",
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
            "type": "interactions",
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
            "type": "initial_states",
            "data": {
                "particle_ID": np.arange(100, dtype=np.uint64),
                "emission_position": np.zeros((100, 3), dtype=np.float32),
                "emission_direction": np.ones((100, 3), dtype=np.float32),
                "emission_energy": np.full(100, 140.0, dtype=np.float32),
                "emission_time": np.zeros(100, dtype=np.float32),
            },
        })

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

        source_node = Source(
            distribution=np.ones((2, 2, 2)),
            activity=50.0 * units.MBq,
            half_life=6.0 * units.hour,
        )
        source_node.name = "Source_Nuclide"
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
            distribution=NumpyDistributionConfig(path="source.npy"),
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
            self.assertAlmostEqual(float(gantry_grp["gantry_angle_deg"][()]), 60.0)
            self.assertEqual(gantry_grp["gantry_angle"].attrs["units"], "rad")
            self.assertAlmostEqual(float(gantry_grp["orbit_radius"][()]), 250.0)
            self.assertEqual(gantry_grp["orbit_radius"].attrs["units"], "mm")
            self.assertEqual(gantry_grp.attrs["rotation_direction"], "CCW")
            self.assertAlmostEqual(float(gantry_grp["frame_duration"][()]), 15.0)
            self.assertIn("pose_matrix_Detector_1", gantry_grp)
            self.assertIn("pose_matrix_detector_1", gantry_grp)
            self.assertEqual(gantry_grp["pose_matrix_Detector_1"].shape, (4, 4))

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

            # 5. Обратная совместимость с задачами tasks/task_5
            self.assertIn("tasks", meta)
            self.assertIn("task_5", meta["tasks"])
            t_grp = meta["tasks/task_5"]
            self.assertEqual(t_grp.attrs["task_id"], 5)
            self.assertEqual(t_grp.attrs["status"], "completed")
            self.assertIn("start_time", t_grp.attrs)
            self.assertIn("completion_time", t_grp.attrs)
            self.assertIn("acquisition", t_grp)
            self.assertIn("protocol", t_grp["acquisition"])
            self.assertIn("kinematics", t_grp["acquisition"])
            self.assertIn("detectors", t_grp["acquisition"])
            self.assertIn("geometry", t_grp["acquisition"])

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
            "type": "initial_states",
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
            "type": "interactions",
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
        Проверка: в каталоге core/ полностью отсутствуют вызовы getattr и hasattr
        в соответствии с правилом 02_code_contracts.md.
        """
        import pathlib
        core_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "core"
        py_files = list(core_dir.rglob("*.py"))
        self.assertGreater(len(py_files), 10, "Каталог core должен содержать файлы исходного кода Python")

        for py_path in py_files:
            content = py_path.read_text(encoding="utf-8")
            self.assertNotIn(
                "getattr(",
                content,
                f"Обнаружен запрещенный вызов getattr( в файле {py_path}",
            )
            self.assertNotIn(
                "hasattr(",
                content,
                f"Обнаружен запрещенный вызов hasattr( в файле {py_path}",
            )


if __name__ == "__main__":
    unittest.main()
