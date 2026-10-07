import os
import tempfile
import unittest
import numpy as np
import h5py
import hepunits as units

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import CompositeNode
from core.data.data_handlers import SensitiveVolumeHandler, HistoryAssemblerHandler
from core.data.data_manager import DataManager
from core.config.yaml_loader import load_simulation_config
from core.config.builder import SceneBuilder
from core.config.simulation_worker import _find_nodes_by_names


class TestDataHandlers(unittest.TestCase):
    """
    Набор тестов для проверки корректности распределения взаимодействий по группам HDF5
    в SensitiveVolumeHandler и HistoryAssemblerHandler.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "test_output.h5")

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except (OSError, RuntimeError):
            pass

    def test_sensitive_volume_handler_nested_geometry(self) -> None:
        """
        Проверка SensitiveVolumeHandler: взаимодействие в чувствительном объеме внутри
        иерархии узлов сохраняется именно в группу чувствительного объема с локальными координатами.
        """
        world = Volume(geometry=Box(1000.0, 1000.0, 1000.0), material=Material(name="Air"), name="World")
        gantry = CompositeNode(name="Gantry")
        detector_box = Volume(geometry=Box(500.0, 500.0, 50.0), material=Material(name="Air"), name="DetBox")
        crystal = Volume(geometry=Box(400.0, 400.0, 10.0), material=Material(name="NaI"), name="Crystal")

        # Сдвиг кристалла в локальных координатах детектора
        crystal.translate(z=10.0)
        detector_box.add_child(crystal)
        # Сдвиг всей головки в мировых координатах
        detector_box.translate(y=300.0)
        gantry.add_child(detector_box)
        world.add_child(gantry)

        handler = SensitiveVolumeHandler(sensitive_volumes=[crystal])

        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        # Индекс кристалла в flat_list
        crystal_id = handler.target_volumes[0]

        # Имитируем два события: одно в кристалле, одно вне его (в воздухе мира)
        handler.process_chunk({
            'type': 'interactions',
            'data': {
                'particle_ID': np.array([1, 2], dtype=np.uint64),
                'volume_id': np.array([crystal_id, 0], dtype=np.uint32),
                'pos_x': np.array([0.0, 100.0], dtype=np.float32),
                'pos_y': np.array([300.0, 100.0], dtype=np.float32),
                'pos_z': np.array([10.0, 100.0], dtype=np.float32),
                'dir_x': np.array([0.0, 0.0], dtype=np.float32),
                'dir_y': np.array([0.0, 0.0], dtype=np.float32),
                'dir_z': np.array([1.0, 1.0], dtype=np.float32),
                'energy_deposit': np.array([140.0, 10.0], dtype=np.float32),
                'process_id': np.array([1, 0], dtype=np.uint16),
                'scattering_theta': np.zeros(2, dtype=np.float32),
                'scattering_phi': np.zeros(2, dtype=np.float32),
                'material_id': np.zeros(2, dtype=np.uint32),
                'Z': np.ones(2, dtype=np.uint32),
                'species': np.zeros(2, dtype=np.uint8),
                'distance_traveled': np.ones(2, dtype=np.float32),
            }
        })

        handler.finalize()

        with h5py.File(self.h5_path, 'a') as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, 'r') as h5_file:
            self.assertIn("interactions", h5_file)
            # Группа Crystal должна существовать
            self.assertIn("Crystal", h5_file["interactions"])
            # События вне кристалла (World) не должны сохраняться в SensitiveVolumeHandler
            self.assertNotIn("World", h5_file["interactions"])

            crystal_group = h5_file["interactions/Crystal"]
            self.assertEqual(len(crystal_group["particle_ID"]), 1)
            self.assertEqual(crystal_group["particle_ID"][0], 1)

            # Проверяем отсутствие устаревших датасетов local_position и local_direction
            self.assertNotIn("local_position", crystal_group)
            self.assertNotIn("local_direction", crystal_group)

            # Проверяем метаданные группы
            self.assertEqual(crystal_group.attrs["role"], "detector")
            self.assertIn("tags", crystal_group.attrs)
            tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in crystal_group.attrs["tags"]]
            self.assertIn("detector", tags)
            self.assertIn("crystal", tags)

            self.assertAlmostEqual(crystal_group.attrs["orbit_radius_mm"], 300.0, places=3)
            self.assertAlmostEqual(crystal_group.attrs["detector_angle_deg"], 90.0, places=3)

            # Проверяем вычисление истинно локальных координат через pose_matrix
            pose_matrix = crystal_group.attrs["pose_matrix"]
            inv_pose = np.linalg.inv(pose_matrix)
            global_pos = np.array(crystal_group["global_position"][0])
            local_pos_calc = (np.append(global_pos, 1.0) @ inv_pose.T)[:3]
            np.testing.assert_allclose(local_pos_calc, [0.0, 0.0, 0.0], atol=1e-3)

    def test_history_assembler_dual_detector_groups(self) -> None:
        """
        Проверка HistoryAssemblerHandler: взаимодействия в двух кристаллах и фантоме
        сохраняются в 3 разные группы с соответствующими локальными координатами.
        """
        yaml_path = "tb161_nema_protocol direct.yaml"
        cfg = load_simulation_config(yaml_path)
        root_scene = SceneBuilder().build_scene(cfg.scene)
        sensitive_vols = _find_nodes_by_names(root_scene, cfg.data_manager.handlers[0].sensitive_volumes)

        handler = HistoryAssemblerHandler(sensitive_volumes=sensitive_vols, save_initial_states=True)

        # Проверяем, что в объемы для записи попадают оба кристалла и корень мира
        vols_to_write_names = [v.name for v in handler._get_volumes_to_write()]
        self.assertIn("Detector_1", vols_to_write_names)
        self.assertIn("Detector_2", vols_to_write_names)
        self.assertIn("Simulation_volume", vols_to_write_names)

        # Проверяем сопоставление volume_mapping
        # volume_id 1 -> Phantom (мапится на Simulation_volume)
        # volume_id 6 -> Detector_1 (мапится на Detector_1)
        # volume_id 12 -> Detector_2 (мапится на Detector_2)
        self.assertEqual(handler.volume_mapping[1].name, "Simulation_volume")
        self.assertEqual(handler.volume_mapping[6].name, "Detector_1")
        self.assertEqual(handler.volume_mapping[12].name, "Detector_2")

        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        # Имитируем частицу 101, рассеявшуюся в фантоме (1) и поглощенную в Detector_1 (6)
        # и частицу 102, рассеявшуюся в фантоме (1) и поглощенную в Detector_2 (12)
        handler.process_chunk({
            'type': 'interactions',
            'data': {
                'particle_ID': np.array([101, 101, 102, 102], dtype=np.uint64),
                'volume_id': np.array([1, 6, 1, 12], dtype=np.uint32),
                'pos_x': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'pos_y': np.array([0.0, 329.05, 0.0, -329.05], dtype=np.float32),
                'pos_z': np.array([0.0, 30.45, 0.0, 30.45], dtype=np.float32),
                'dir_x': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'dir_y': np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32),
                'dir_z': np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
                'energy_deposit': np.array([5.0, 70.0, 6.0, 69.0], dtype=np.float32),
                'process_id': np.array([1, 2, 1, 2], dtype=np.uint16),
                'scattering_theta': np.zeros(4, dtype=np.float32),
                'scattering_phi': np.zeros(4, dtype=np.float32),
                'material_id': np.zeros(4, dtype=np.uint32),
                'Z': np.ones(4, dtype=np.uint32),
                'species': np.zeros(4, dtype=np.uint8),
                'distance_traveled': np.ones(4, dtype=np.float32),
            }
        })

        # Завершение треков частиц
        handler.process_chunk({
            'type': 'dead_particles',
            'data': np.array([101, 102], dtype=np.uint64)
        })

        handler.finalize()

        with h5py.File(self.h5_path, 'a') as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, 'r') as h5_file:
            self.assertIn("interactions", h5_file)
            inter_group = h5_file["interactions"]

            # Должны быть ровно 3 группы: Detector_1, Detector_2 и Simulation_volume
            self.assertIn("Detector_1", inter_group)
            self.assertIn("Detector_2", inter_group)
            self.assertIn("Simulation_volume", inter_group)

            # Проверяем отсутствие устаревших локальных датасетов
            for vol_name in ("Detector_1", "Detector_2", "Simulation_volume"):
                self.assertNotIn("local_position", inter_group[vol_name])
                self.assertNotIn("local_direction", inter_group[vol_name])

            # Проверяем семантические роли
            self.assertEqual(inter_group["Detector_1"].attrs["role"], "detector")
            self.assertEqual(inter_group["Detector_2"].attrs["role"], "detector")
            self.assertEqual(inter_group["Simulation_volume"].attrs["role"], "scatter_history")

            # Проверяем кинематические параметры детекторов
            self.assertAlmostEqual(inter_group["Detector_1"].attrs["orbit_radius_mm"], 298.6, places=1)
            self.assertAlmostEqual(inter_group["Detector_2"].attrs["orbit_radius_mm"], 298.6, places=1)
            self.assertAlmostEqual(inter_group["Detector_1"].attrs["detector_angle_deg"], 90.0, places=1)
            self.assertAlmostEqual(inter_group["Detector_2"].attrs["detector_angle_deg"], 270.0, places=1)

            # В Detector_1 должно быть 1 событие (частица 101, volume_id 6)
            self.assertEqual(len(inter_group["Detector_1/particle_ID"]), 1)
            self.assertEqual(inter_group["Detector_1/volume_id"][0], 6)

            # В Detector_2 должно быть 1 событие (частица 102, volume_id 12)
            self.assertEqual(len(inter_group["Detector_2/particle_ID"]), 1)
            self.assertEqual(inter_group["Detector_2/volume_id"][0], 12)

            # В Simulation_volume должны быть 2 события рассеяния в фантоме (volume_id 1)
            self.assertEqual(len(inter_group["Simulation_volume/particle_ID"]), 2)
            np.testing.assert_array_equal(inter_group["Simulation_volume/volume_id"], [1, 1])

    def test_data_manager_metadata_and_context(self) -> None:
        """
        Проверка DataManager: запись метаданных задачи и процедурного контекста в HDF5.
        """
        context_data = {
            "gantry_angle": 45.0,
            "scan_time": 60.0,
            "projection_index": 5,
        }
        task_metadata = {
            "task_id": "test_step_05",
            "protocol_type": "spect_step_and_shoot",
            "context": context_data,
        }
        dm = DataManager(filename=self.h5_path, handlers=[], metadata=task_metadata)
        dm.write_metadata()

        with h5py.File(self.h5_path, "r") as h5_file:
            self.assertIn("metadata", h5_file)
            meta_group = h5_file["metadata"]
            self.assertEqual(meta_group.attrs["task_id"], "test_step_05")
            self.assertEqual(meta_group.attrs["protocol_type"], "spect_step_and_shoot")

            self.assertIn("context", meta_group)
            ctx_group = meta_group["context"]
            self.assertAlmostEqual(ctx_group.attrs["gantry_angle"], 45.0)
            self.assertAlmostEqual(ctx_group.attrs["scan_time"], 60.0)
            self.assertEqual(ctx_group.attrs["projection_index"], 5)


if __name__ == '__main__':
    unittest.main()
