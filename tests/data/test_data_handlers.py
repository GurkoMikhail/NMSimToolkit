import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from multiprocessing import shared_memory
import numpy as np
import h5py
import hepunits as units

from core.geometry.flattened_scene import FlattenedScene
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index
from core.particles.particles import ParticleBank
from core.physics.interaction_buffers import ChunkType, EscapedParticlesBuffer
from core.scene.nodes import CompositeNode
from core.data.data_handlers import (
    BaseDataHandler,
    DirectStreamHandler,
    SensitiveVolumeHandler,
    HistoryAssemblerHandler,
)
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.config.yaml_loader import load_simulation_config
from core.config.builder import SceneBuilder
from core.config.simulation_worker import _find_nodes_by_names
from core.transport.ipc_pause_bridge import IpcPauseBridge
from core.transport.simulation_managers import SimulationManager, SimulationState
from core.transport.transport_kernels import _push_to_escaped_particles_kernel
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.stream_handlers import (
    ControlBufferSHM,
    FocusedPauseProxy,
    GuiStreamDataHandler,
    TelemetryRingBufferSHM,
    create_gui_stream_handler,
    get_task_dose_shm_name,
    get_task_projection_shm_name,
    get_task_telemetry_shm_name,
)


class TestDataHandlers(unittest.TestCase):
    """
    Набор тестов для проверки корректности распределения взаимодействий по группам HDF5
    в SensitiveVolumeHandler и HistoryAssemblerHandler.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "test_output.h5")

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir.name):
            self.temp_dir.cleanup()

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
            'type': ChunkType.INTERACTIONS,
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

            # Проверяем корректность записанных глобальных координат
            global_pos = np.array(crystal_group["global_position"][0])
            np.testing.assert_allclose(global_pos, [0.0, 300.0, 10.0], atol=1e-3)

    def test_history_assembler_dual_detector_groups(self) -> None:
        """
        Проверка HistoryAssemblerHandler: взаимодействия в двух кристаллах и фантоме
        сохраняются в 3 разные группы с соответствующими локальными координатами.
        """
        yaml_path = "tb161_nema_protocol.yaml"
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
            'type': ChunkType.INTERACTIONS,
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
            'type': ChunkType.DEAD_PARTICLES,
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
        Проверка DataManager: запись метаданных задачи и процедурного контекста в HDF5
        в изолированную подгруппу tasks/task_{task_id}.
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
        dm = DataManager(filename=self.h5_path, handlers=[], metadata=task_metadata, task_id="test_step_05")
        dm.initialize_metadata()
        dm.finalize_metadata(status="completed")

        with h5py.File(self.h5_path, "r") as h5_file:
            self.assertIn("metadata", h5_file)
            meta_group = h5_file["metadata"]
            self.assertIn("created_time", meta_group.attrs)

            # Проверка изолированной подгруппы tasks/task_test_step_05
            self.assertIn("tasks", meta_group)
            self.assertIn("task_test_step_05", meta_group["tasks"])
            task_subgroup = meta_group["tasks/task_test_step_05"]
            self.assertEqual(task_subgroup.attrs["task_id"], "test_step_05")
            self.assertEqual(task_subgroup.attrs["protocol_type"], "spect_step_and_shoot")
            self.assertEqual(task_subgroup.attrs["status"], "completed")
            self.assertIn("start_time", task_subgroup.attrs)
            self.assertIn("completion_time", task_subgroup.attrs)
            self.assertIn("elapsed_real_time", task_subgroup.attrs)
            self.assertEqual(task_subgroup.attrs["elapsed_real_time_units"], "s")
            self.assertIn("context", task_subgroup)
            self.assertAlmostEqual(task_subgroup["context"].attrs["gantry_angle"], 45.0)
            self.assertAlmostEqual(task_subgroup["context"].attrs["scan_time"], 60.0)
            self.assertEqual(task_subgroup["context"].attrs["projection_index"], 5)

    def test_early_metadata_initialization(self) -> None:
        """
        Проверка ранней инициализации метаданных: фиксация статуса 'in_progress'
        и отметки времени старта до начала моделирования.
        """
        context_data = {"current_angle": 1.23}
        task_metadata = {
            "task_id": 42,
            "context": context_data,
        }
        dm = DataManager(filename=self.h5_path, handlers=[], metadata=task_metadata, task_id=42)

        # Выполняем раннюю инициализацию
        dm.initialize_metadata()

        # Проверяем состояние файла в процессе выполнения симуляции
        with h5py.File(self.h5_path, "r") as h5_file:
            self.assertIn("metadata", h5_file)
            meta_group = h5_file["metadata"]
            self.assertIn("created_time", meta_group.attrs)

            task_group = meta_group["tasks/task_42"]
            self.assertEqual(task_group.attrs["status"], "in_progress")
            self.assertEqual(task_group.attrs["task_id"], 42)
            self.assertIn("start_time", task_group.attrs)
            self.assertNotIn("completion_time", task_group.attrs)
            self.assertAlmostEqual(task_group["context"].attrs["current_angle"], 1.23)

        # Завершаем моделирование
        dm.finalize_metadata(status="completed")

        with h5py.File(self.h5_path, "r") as h5_file:
            meta_group = h5_file["metadata"]
            task_group = meta_group["tasks/task_42"]
            self.assertEqual(task_group.attrs["status"], "completed")
            self.assertIn("completion_time", task_group.attrs)
            self.assertIn("elapsed_real_time", task_group.attrs)

    def test_simulation_failure_metadata_recording(self) -> None:
        """
        Проверка фиксации сбоя симуляции: обновление статуса на 'failed'
        и запись диагностического сообщения об ошибке.
        """
        task_metadata = {"task_id": 99, "simulation_type": "monte_carlo"}
        dm = DataManager(filename=self.h5_path, handlers=[], metadata=task_metadata, task_id=99)
        dm.initialize_metadata()

        error_message = "RuntimeError: Particle geometry boundary overflow detected"
        dm.finalize_metadata(status="failed", error=error_message)

        with h5py.File(self.h5_path, "r") as h5_file:
            meta_group = h5_file["metadata"]
            task_group = meta_group["tasks/task_99"]
            self.assertEqual(task_group.attrs["status"], "failed")
            self.assertEqual(task_group.attrs["error"], error_message)
            self.assertIn("completion_time", task_group.attrs)

    def test_multi_task_metadata_isolation(self) -> None:
        """
        Проверка изоляции метаданных нескольких задач, записываемых в один файл HDF5.
        Данные каждой задачи сохраняются в /metadata/tasks/task_{task_id} и не перезаписывают друг друга.
        """
        file_lock = threading.Lock()

        # Задача 0: первая ОФЭКТ проекция
        task_0_meta = {
            "task_id": 0,
            "context": {"view_index": 0, "gantry_angle": 0.0},
        }
        dm_task_0 = DataManager(
            filename=self.h5_path,
            handlers=[],
            lock=file_lock,
            metadata=task_0_meta,
            task_id=0,
        )
        dm_task_0.initialize_metadata()
        dm_task_0.finalize_metadata(status="completed")

        # Задача 1: вторая ОФЭКТ проекция
        task_1_meta = {
            "task_id": 1,
            "context": {"view_index": 1, "gantry_angle": np.pi / 2.0},
        }
        dm_task_1 = DataManager(
            filename=self.h5_path,
            handlers=[],
            lock=file_lock,
            metadata=task_1_meta,
            task_id=1,
        )
        dm_task_1.initialize_metadata()
        dm_task_1.finalize_metadata(status="completed")

        # Проверка содержимого файла HDF5
        with h5py.File(self.h5_path, "r") as h5_file:
            self.assertIn("metadata/tasks", h5_file)
            tasks_group = h5_file["metadata/tasks"]

            self.assertIn("task_0", tasks_group)
            self.assertIn("task_1", tasks_group)

            task_0_group = tasks_group["task_0"]
            task_1_group = tasks_group["task_1"]

            # Проверяем, что параметры задачи 0 сохранены и НЕ затерты задачей 1
            self.assertEqual(task_0_group.attrs["task_id"], 0)
            self.assertEqual(task_0_group.attrs["status"], "completed")
            self.assertAlmostEqual(task_0_group["context"].attrs["gantry_angle"], 0.0)
            self.assertEqual(task_0_group["context"].attrs["view_index"], 0)

            # Проверяем параметры задачи 1
            self.assertEqual(task_1_group.attrs["task_id"], 1)
            self.assertEqual(task_1_group.attrs["status"], "completed")
            self.assertAlmostEqual(task_1_group["context"].attrs["gantry_angle"], np.pi / 2.0)
            self.assertEqual(task_1_group["context"].attrs["view_index"], 1)

            # Проверяем атрибуты контейнера метаданных
            meta_group = h5_file["metadata"]
            self.assertIn("created_time", meta_group.attrs)

    def test_sensitive_volume_handler_composite_node_and_pruning(self) -> None:
        """
        Проверка SensitiveVolumeHandler при передаче CompositeNode:
        1. Избыточные дочерние узлы отсекаются алгоритмом _filter_topmost_nodes.
        2. Все внутренние объемы CompositeNode консолидируются в единую группу детектора.
        """
        world = Volume(geometry=Box(1000.0, 1000.0, 1000.0), material=Material(name="Air"), name="World")
        head = CompositeNode(name="DetectorHead")
        crystal_front = Volume(geometry=Box(400.0, 400.0, 10.0), material=Material(name="NaI"), name="CrystalFront")
        crystal_back = Volume(geometry=Box(400.0, 400.0, 10.0), material=Material(name="NaI"), name="CrystalBack")
        head.add_child(crystal_front)
        head.add_child(crystal_back)
        world.add_child(head)

        # Передаем и родительский CompositeNode, и дочерний Volume
        handler = SensitiveVolumeHandler(sensitive_volumes=[head, crystal_front])

        # Проверяем, что crystal_front отсечен, а остался только DetectorHead
        self.assertEqual(len(handler.sensitive_volumes), 1)
        self.assertEqual(handler.sensitive_volumes[0].name, "DetectorHead")

        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        # Находим внутренние volume_id для обоих кристаллов
        flat_list = FlattenedScene(world).flat_list
        front_id = next(vol_idx for vol_idx, (vol_node, _, _) in enumerate(flat_list) if vol_node is crystal_front)
        back_id = next(vol_idx for vol_idx, (vol_node, _, _) in enumerate(flat_list) if vol_node is crystal_back)

        # Имитируем взаимодействия в обоих кристаллах
        handler.process_chunk({
            'type': ChunkType.INTERACTIONS,
            'data': {
                'particle_ID': np.array([101, 102], dtype=np.uint64),
                'volume_id': np.array([front_id, back_id], dtype=np.uint32),
                'pos_x': np.array([10.0, 20.0], dtype=np.float32),
                'pos_y': np.array([0.0, 0.0], dtype=np.float32),
                'pos_z': np.array([0.0, 5.0], dtype=np.float32),
                'dir_x': np.array([0.0, 0.0], dtype=np.float32),
                'dir_y': np.array([0.0, 0.0], dtype=np.float32),
                'dir_z': np.array([1.0, 1.0], dtype=np.float32),
                'energy_deposit': np.array([140.0, 70.0], dtype=np.float32),
                'process_id': np.array([1, 1], dtype=np.uint16),
                'scattering_theta': np.zeros(2, dtype=np.float32),
                'distance_traveled': np.ones(2, dtype=np.float32),
            }
        })

        handler.finalize()

        with h5py.File(self.h5_path, 'a') as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, 'r') as h5_file:
            self.assertIn("interactions", h5_file)
            # Должна быть создана группа "DetectorHead"
            self.assertIn("DetectorHead", h5_file["interactions"])
            det_group = h5_file["interactions/DetectorHead"]
            # Оба события записаны в DetectorHead
            self.assertEqual(len(det_group["particle_ID"]), 2)
            self.assertListEqual(list(det_group["particle_ID"]), [101, 102])

    def test_filter_topmost_nodes_contracts_and_edge_cases(self) -> None:
        """
        Проверка контракта DbC и краевых случаев _filter_topmost_nodes:
        1. Передача не-SpatialNode объектов вызывает TypeError.
        2. Трехуровневая иерархия A -> B -> C корректно сворачивается до верхнего предка.
        3. Несвязанные ветви графа сцены не отсекают друг друга.
        """
        node_a = CompositeNode(name="LevelA")
        node_b = CompositeNode(name="LevelB")
        node_c = Volume(geometry=Box(10.0, 10.0, 10.0), material=Material(name="Air"), name="LevelC")
        node_b.add_child(node_c)
        node_a.add_child(node_b)

        disjoint_node = Volume(geometry=Box(20.0, 20.0, 20.0), material=Material(name="Air"), name="Disjoint")

        # 1. Трехуровневая иерархия в обратном порядке: [C, A] -> [A]
        filtered_1 = SensitiveVolumeHandler._filter_topmost_nodes([node_c, node_a])
        self.assertEqual(len(filtered_1), 1)
        self.assertEqual(filtered_1[0].name, "LevelA")

        # 3. Полный список с дубликатами [A, B, C, A, B] -> [A]
        filtered_2 = SensitiveVolumeHandler._filter_topmost_nodes([node_a, node_b, node_c, node_a, node_b])
        self.assertEqual(len(filtered_2), 1)
        self.assertEqual(filtered_2[0].name, "LevelA")

        # 4. Несвязанные ветви [A, Disjoint] -> [A, Disjoint]
        filtered_3 = SensitiveVolumeHandler._filter_topmost_nodes([node_a, disjoint_node])
        self.assertEqual(len(filtered_3), 2)
        self.assertListEqual([node_item.name for node_item in filtered_3], ["LevelA", "Disjoint"])

        # 5. Пустой список -> []
        self.assertEqual(SensitiveVolumeHandler._filter_topmost_nodes([]), [])

    def test_interactions_fallback_coordinates_shape(self) -> None:
        """
        Проверка корректной генерации резервных массивов координат и направлений
        при отсутствии pos_x/dir_x в словаре взаимодействий (предотвращение IndexError).
        """
        world = Volume(geometry=Box(100.0, 100.0, 100.0), material=Material(name="Air"), name="World")
        crystal = Volume(geometry=Box(50.0, 50.0, 10.0), material=Material(name="NaI"), name="Crystal")
        world.add_child(crystal)

        handler = SensitiveVolumeHandler(sensitive_volumes=[crystal])
        crystal_id = handler.target_volumes[0]

        written_ops = []
        handler.set_writer_callback(lambda op: written_ops.append(op))

        # Передаем словарь без pos_x, pos_y, pos_z, dir_x, dir_y, dir_z
        handler.process_chunk({
            'type': ChunkType.INTERACTIONS,
            'data': {
                'particle_ID': np.array([77, 88], dtype=np.uint64),
                'volume_id': np.array([crystal_id, crystal_id], dtype=np.uint32),
                'energy_deposit': np.array([10.0, 20.0], dtype=np.float32),
                'process_id': np.array([1, 1], dtype=np.uint16),
            }
        })
        handler.finalize()

        with h5py.File(self.h5_path, 'a') as h5_file:
            for op in written_ops:
                op(h5_file)

        with h5py.File(self.h5_path, 'r') as h5_file:
            crystal_group = h5_file["interactions/Crystal"]
            self.assertEqual(crystal_group["global_position"].shape, (2, 3))
            self.assertEqual(crystal_group["global_direction"].shape, (2, 3))

    def test_consumed_chunk_types_contract_and_filtering(self) -> None:
        """
        Проверка контракта BaseDataHandler.consumed_chunk_types:
        1. Подкласс без реализации consumed_chunk_types вызывает TypeError при инстанциировании.
        2. Все обработчики декларируют строгие неизменяемые множества потребляемых типов чанков.
        3. DataManager агрегирует подписки и не вызывает process_chunk для неподписанных типов.
        """
        world = Volume(geometry=Box(100.0, 100.0, 100.0), material=Material(name="Air"), name="World")
        det_vol = Volume(geometry=Box(50.0, 50.0, 10.0), material=Material(name="NaI"), name="Crystal")
        world.add_child(det_vol)

        class IncompleteHandler(BaseDataHandler):
            def process_chunk(self, chunk):
                pass

        with self.assertRaises(TypeError):
            IncompleteHandler()  # type: ignore[abstract]

        direct = DirectStreamHandler()
        self.assertEqual(
            direct.consumed_chunk_types,
            frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
        )

        sensitive = SensitiveVolumeHandler(sensitive_volumes=[det_vol])
        self.assertEqual(sensitive.consumed_chunk_types, frozenset({ChunkType.INTERACTIONS}))

        hist_with_init = HistoryAssemblerHandler(sensitive_volumes=[det_vol], save_initial_states=True)
        self.assertEqual(
            hist_with_init.consumed_chunk_types,
            frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS, ChunkType.DEAD_PARTICLES}),
        )

        hist_no_init = HistoryAssemblerHandler(sensitive_volumes=[det_vol], save_initial_states=False)
        self.assertEqual(
            hist_no_init.consumed_chunk_types,
            frozenset({ChunkType.INTERACTIONS, ChunkType.DEAD_PARTICLES}),
        )

        dose = DoseMapHandler(grid_nodes=[], create_shm=False)
        self.assertEqual(dose.consumed_chunk_types, frozenset({ChunkType.INTERACTIONS}))

        gui_no_esc = GuiStreamDataHandler(show_escaped_tracks=False)
        self.assertEqual(
            gui_no_esc.consumed_chunk_types,
            frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS}),
        )

        gui_with_esc = GuiStreamDataHandler(show_escaped_tracks=True)
        self.assertEqual(
            gui_with_esc.consumed_chunk_types,
            frozenset({ChunkType.INITIAL_STATES, ChunkType.INTERACTIONS, ChunkType.ESCAPED_PARTICLES}),
        )

        received_types = []

        class InteractionsOnlySpyHandler(BaseDataHandler):
            @property
            def consumed_chunk_types(self):
                return frozenset({ChunkType.INTERACTIONS})

            def process_chunk(self, chunk):
                received_types.append(chunk["type"])

        spy = InteractionsOnlySpyHandler()
        chunk_queue: queue.Queue = queue.Queue()
        dm = DataManager(filename=self.h5_path, handlers=[spy, dose], queue=chunk_queue, swmr=False)
        self.assertEqual(dm.consumed_chunk_types, frozenset({ChunkType.INTERACTIONS}))

        chunk_queue.put({"type": ChunkType.INITIAL_STATES, "data": {"particle_ID": np.array([1], dtype=np.int64)}})
        chunk_queue.put({"type": ChunkType.ESCAPED_PARTICLES, "data": {"particle_ID": np.array([1], dtype=np.int64)}})
        chunk_queue.put({
            "type": ChunkType.INTERACTIONS,
            "data": {"pos_x": np.array([], dtype=np.float64), "energy_deposit": np.array([], dtype=np.float64)},
        })
        chunk_queue.put("stop")
        dm.run()

        self.assertEqual(received_types, [ChunkType.INTERACTIONS])

    def test_escaped_particles_buffering_and_subscription_gating(self) -> None:
        """
        Проверка гейтинга _record_finished_histories по подписке и буферизации EscapedParticlesBuffer:
        1. Без подписки на ChunkType.ESCAPED_PARTICLES емкость буфера равна 0 и чанки не создаются.
        2. С подпиской на ChunkType.ESCAPED_PARTICLES и ChunkType.INITIAL_STATES для непрореагировавших
           вылетевших частиц автоматически дозаписываются начальные состояния в initial_states, а точки
           вылета накапливаются в escaped_particles и сбрасываются при подготовке емкости шага.
        """
        world = Volume(geometry=Box(100.0, 100.0, 100.0), material=Material(name="Air"), name="World")

        # 1. Тест записи в EscapedParticlesBuffer через Numba-ядро _push_to_escaped_particles_kernel
        esc_buf = EscapedParticlesBuffer.allocate(4)
        esc_buf.validate()
        test_bank = ParticleBank.allocate(5)
        test_bank.initial_state.ID[:] = np.array([100, 200, 300, 400, 500], dtype=Index)
        test_bank.state.position.x[:] = np.array([20.0, 40.0, 60.0, 80.0, 100.0], dtype=Float)
        test_bank.state.position.y[:] = np.array([21.0, 41.0, 61.0, 81.0, 101.0], dtype=Float)
        test_bank.state.position.z[:] = np.array([22.0, 42.0, 62.0, 82.0, 102.0], dtype=Float)

        _push_to_escaped_particles_kernel(
            test_bank.initial_state,
            test_bank.state,
            np.array([1, 3], dtype=Index),
            esc_buf,
        )
        self.assertEqual(esc_buf.cursor_value, 2)
        self.assertEqual(esc_buf.remaining_capacity, 2)

        flushed = esc_buf.flush(clear=True)
        self.assertEqual(esc_buf.cursor_value, 0)
        np.testing.assert_array_equal(flushed["particle_ID"], np.array([200, 400], dtype=Index))
        np.testing.assert_array_equal(flushed["pos_x"], np.array([40.0, 80.0], dtype=Float))
        np.testing.assert_array_equal(flushed["pos_y"], np.array([41.0, 81.0], dtype=Float))
        np.testing.assert_array_equal(flushed["pos_z"], np.array([42.0, 82.0], dtype=Float))
        self.assertNotIn("birth_x", flushed)
        self.assertNotIn("has_interacted", flushed)

        # 2. Проверка нулевой емкости неактивных буферов без подписки на ChunkType.ESCAPED_PARTICLES
        sim_no_esc = SimulationManager(
            scene=world,
            particles_number=4,
            buffer_capacity=10,
            stop_time=Float(1.0 * units.s),
            consumed_chunk_types=frozenset({ChunkType.INTERACTIONS}),
        )
        self.assertEqual(sim_no_esc.data_buffer.interactions.capacity, 10)
        self.assertEqual(sim_no_esc.data_buffer.initial_states.capacity, 0)
        self.assertEqual(sim_no_esc.data_buffer.dead_particles.capacity, 0)
        self.assertEqual(sim_no_esc.data_buffer.escaped_particles.capacity, 0)

        sim_no_esc.bank.navigation_state.current_volume[:2] = -1
        sim_no_esc.bank.state.is_active[:2] = True
        sim_no_esc.bank.navigation_state.boundary_distance[:2] = 1.0

        dead_idx = sim_no_esc._apply_invalidators(np.array([0, 1], dtype=Index))
        self.assertEqual(len(dead_idx), 2)
        self.assertNotIn(ChunkType.ESCAPED_PARTICLES, sim_no_esc.consumed_chunk_types)

        # 3. Проверка буферизации при наличии подписки на ChunkType.ESCAPED_PARTICLES и ChunkType.INITIAL_STATES
        sim_with_esc = SimulationManager(
            scene=world,
            particles_number=4,
            buffer_capacity=4,
            stop_time=Float(1.0 * units.s),
            consumed_chunk_types=frozenset({ChunkType.INITIAL_STATES, ChunkType.ESCAPED_PARTICLES}),
        )
        sim_with_esc.bank.navigation_state.current_volume[:4] = -1
        sim_with_esc.bank.initial_state.ID[:4] = np.array([11, 22, 33, 44], dtype=Index)
        # Частицы 0 и 2 не взаимодействовали (direction == emission_direction)
        # Частицы 1 и 3 провзаимодействовали (direction != emission_direction)
        sim_with_esc.bank.initial_state.emission_energy[:4] = 140.0
        sim_with_esc.bank.state.energy[:4] = np.array([140.0, 100.0, 140.0, 90.0], dtype=Float)
        sim_with_esc.bank.initial_state.emission_direction.x[:4] = 0.0
        sim_with_esc.bank.initial_state.emission_direction.y[:4] = 0.0
        sim_with_esc.bank.initial_state.emission_direction.z[:4] = 1.0
        sim_with_esc.bank.state.direction.x[:4] = np.array([0.0, 1.0, 0.0, 0.0], dtype=Float)
        sim_with_esc.bank.state.direction.y[:4] = np.array([0.0, 0.0, 0.0, 1.0], dtype=Float)
        sim_with_esc.bank.state.direction.z[:4] = np.array([1.0, 0.0, 1.0, 0.0], dtype=Float)

        # Первая порция из 2 частиц меньше емкости буфера (4) -> накапливается в буфере, очередь пуста
        sim_with_esc._record_finished_histories(np.array([0, 1], dtype=Index))
        self.assertEqual(sim_with_esc.data_buffer.escaped_particles.cursor_value, 2)
        # В initial_states записана только непрореагировавшая частица 0 (ID=11)
        self.assertEqual(sim_with_esc.data_buffer.initial_states.cursor_value, 1)
        self.assertEqual(int(sim_with_esc.data_buffer.initial_states.particle_ID[0]), 11)
        self.assertTrue(sim_with_esc.queue.empty())

        # Перед вторым шагом (или при записи еще 3 частиц) _ensure_step_capacity сбрасывает буферы,
        # когда remaining_capacity (2) < active_count (3), сохраняя причинно-следственный порядок
        sim_with_esc._record_finished_histories(np.array([2, 3], dtype=Index))
        self.assertEqual(sim_with_esc.data_buffer.escaped_particles.cursor_value, 4)
        sim_with_esc._ensure_step_capacity(1)
        self.assertEqual(sim_with_esc.data_buffer.escaped_particles.cursor_value, 0)
        self.assertFalse(sim_with_esc.queue.empty())
        chunk_init = sim_with_esc.queue.get_nowait()
        self.assertEqual(chunk_init["type"], ChunkType.INITIAL_STATES)
        np.testing.assert_array_equal(chunk_init["data"]["particle_ID"], np.array([11, 33], dtype=Index))
        chunk_esc = sim_with_esc.queue.get_nowait()
        self.assertEqual(chunk_esc["type"], ChunkType.ESCAPED_PARTICLES)
        np.testing.assert_array_equal(chunk_esc["data"]["particle_ID"], np.array([11, 22, 33, 44], dtype=Index))

    def test_gui_stream_handler_inherits_sensitive_volume_and_filters_phantom(self) -> None:
        """
        Проверка GuiStreamDataHandler:
        1. Наследуется от SensitiveVolumeHandler и автоматически вычисляет target_volumes через FlattenedScene.
        2. Взаимодействия в фантоме (вне Detector_1/Detector_2) не попадают в 2D-проекцию и detector_energy_deposit,
           тогда как 3D-координаты всех точек исправно передаются в track_queue.
        3. При батче только с фоновыми взаимодействиями в фантоме IPCReceiver не добавляет их в спектр детектора.
        """
        self.assertTrue(issubclass(GuiStreamDataHandler, SensitiveVolumeHandler))

        yaml_path = "tb161_nema_protocol.yaml"
        cfg = load_simulation_config(yaml_path)
        root_scene = SceneBuilder().build_scene(cfg.scene)
        sensitive_vols = _find_nodes_by_names(root_scene, cfg.data_manager.handlers[0].sensitive_volumes)

        shm_name = f"test_gui_sv_shm_{os.getpid()}"
        track_q: queue.Queue = queue.Queue()

        handler = GuiStreamDataHandler(
            track_queue=track_q,
            shm_name=shm_name,
            projection_shape=(64, 64),
            create_shm=True,
            sensitive_volumes=sensitive_vols,
        )

        try:
            # 1. Проверяем автоматический расчет target_volumes (Detector_1=6, Detector_2=12) и защиту от записи в HDF5
            self.assertEqual(set(handler.target_volumes), {6, 12})
            self.assertEqual(handler.sensitive_volume_ids, {6, 12})
            written_ops = []
            handler.set_writer_callback(lambda op: written_ops.append(op))
            self.assertIsNone(handler.writer_callback)

            # 2. Батч ТОЛЬКО с взаимодействиями в воде фантома (volume_id=1) прямо перед детектором (x=0, z=0)
            handler.process_chunk({
                "type": ChunkType.INTERACTIONS,
                "data": {
                    "particle_ID": np.array([101, 102], dtype=np.uint64),
                    "volume_id": np.array([1, 1], dtype=np.uint32),
                    "pos_x": np.array([0.0, 10.0], dtype=np.float32),
                    "pos_y": np.array([0.0, 5.0], dtype=np.float32),
                    "pos_z": np.array([0.0, 10.0], dtype=np.float32),
                    "process_id": np.array([1, 1], dtype=np.uint16),
                    "energy_deposit": np.array([0.05, 0.08], dtype=np.float32),
                },
            })

            # Проекция должна остаться строго нулевой (события из фантома отсечены)
            proj_after_phantom = handler.get_projection_snapshot()
            self.assertIsNotNone(proj_after_phantom)
            self.assertEqual(float(np.sum(proj_after_phantom)), 0.0)

            # В track_queue 3D-точки фантома переданы, но detector_energy_deposit — пустой массив (не None)
            item_phantom = track_q.get_nowait()
            self.assertEqual(len(item_phantom["pos_x"]), 2)
            self.assertIsNotNone(item_phantom["detector_energy_deposit"])
            self.assertEqual(len(item_phantom["detector_energy_deposit"]), 0)

            # Проверяем, что IPCReceiver не засоряет спектр энерговыделениями из фантома
            receiver = IPCReceiver()
            try:
                receiver._process_track_item(item_phantom)
                self.assertEqual(len(receiver._accumulated_energies), 0)

                # 3. Батч со смешанными взаимодействиями: в фантоме (1) и в обоих кристаллах (6 и 12)
                det1_pos = sensitive_vols[0].global_matrix[:3, 3]
                det2_pos = sensitive_vols[1].global_matrix[:3, 3]
                handler.process_chunk({
                    "type": ChunkType.INTERACTIONS,
                    "data": {
                        "particle_ID": np.array([201, 201, 202], dtype=np.uint64),
                        "volume_id": np.array([1, 6, 12], dtype=np.uint32),
                        "pos_x": np.array([0.0, det1_pos[0], det2_pos[0]], dtype=np.float32),
                        "pos_y": np.array([0.0, det1_pos[1], det2_pos[1]], dtype=np.float32),
                        "pos_z": np.array([0.0, det1_pos[2], det2_pos[2]], dtype=np.float32),
                        "process_id": np.array([1, 2, 2], dtype=np.uint16),
                        "energy_deposit": np.array([0.04, 0.10, 0.14], dtype=np.float32),
                    },
                })

                proj_after_det = handler.get_projection_snapshot()
                self.assertIsNotNone(proj_after_det)
                self.assertEqual(float(np.sum(proj_after_det)), 2.0)

                item_mixed = track_q.get_nowait()
                self.assertEqual(len(item_mixed["pos_x"]), 3)
                np.testing.assert_allclose(item_mixed["detector_energy_deposit"], [0.10, 0.14], rtol=1e-5)

                receiver._process_track_item(item_mixed)
                self.assertEqual(len(receiver._accumulated_energies), 1)
                np.testing.assert_allclose(
                    receiver._accumulated_energies[0],
                    np.array([0.10, 0.14], dtype=np.float32) / float(units.keV),
                    rtol=1e-5,
                )
            finally:
                receiver.close()

            # 4. Проверка фабрики create_gui_stream_handler
            task_dict = cfg.model_dump(mode="python")
            created = create_gui_stream_handler(
                root_scene=root_scene,
                task_dict=task_dict,
                task_id=0,
                telemetry_queue=track_q,
                focused_job_index=0,
                focused_shm_name=shm_name,
                projection_shape=(64, 64),
            )
            self.assertEqual(len(created), 1)
            created_handler = created[0]
            self.assertIsInstance(created_handler, GuiStreamDataHandler)
            self.assertEqual(created_handler.sensitive_volume_ids, {6, 12})
            created_handler.close()
        finally:
            handler.close()

    def test_telemetry_ring_buffer_shm(self) -> None:
        """
        Проверка TelemetryRingBufferSHM:
        1. Запись точек треков (write_tracks) и инкрементальное чтение по курсору (read_tracks_since).
        2. Кольцевое переполнение буфера треков (n > track_capacity) с сохранением последних точек.
        3. Запись и чтение энергетического спектра (write_spectrum, read_spectrum) с фильтрацией нулей.
        """
        shm_name = f"test_telem_ring_{os.getpid()}"
        writer = TelemetryRingBufferSHM(shm_name=shm_name, create=True, track_capacity=5, spectrum_capacity=4)
        reader = TelemetryRingBufferSHM(shm_name=shm_name, create=False)

        try:
            self.assertEqual(reader.track_capacity, 5)
            self.assertEqual(reader.spectrum_capacity, 4)
            self.assertEqual(reader.track_write_cursor, 0)
            self.assertEqual(reader.spectrum_write_cursor, 0)

            # 1. Запись первых 3 точек треков
            c1 = writer.write_tracks(
                pos_x=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                pos_y=np.array([10.0, 20.0, 30.0], dtype=np.float32),
                pos_z=np.array([100.0, 200.0, 300.0], dtype=np.float32),
                particle_id=np.array([11, 12, 13], dtype=np.int64),
                process_id=np.array([-2, 1, 2], dtype=np.int16),
            )
            self.assertEqual(c1, 3)

            read_cur, tracks_batch_1 = reader.read_tracks_since(0)
            self.assertEqual(read_cur, 3)
            self.assertIsNotNone(tracks_batch_1)
            assert tracks_batch_1 is not None
            np.testing.assert_allclose(tracks_batch_1["pos_x"], [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(tracks_batch_1["particle_id"], [11, 12, 13])
            np.testing.assert_array_equal(tracks_batch_1["process_id"], [-2, 1, 2])

            # Повторное чтение с тем же курсором возвращает None
            read_cur_same, empty_batch = reader.read_tracks_since(read_cur)
            self.assertEqual(read_cur_same, 3)
            self.assertIsNone(empty_batch)

            # 2. Запись еще 4 точек (суммарно 7 > track_capacity=5 -> переполнение кольца)
            c2 = writer.write_tracks(
                pos_x=np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                pos_y=np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float32),
                pos_z=np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32),
                particle_id=np.array([14, 15, 16, 17], dtype=np.int64),
                process_id=np.array([1, 1, -1, 2], dtype=np.int16),
            )
            self.assertEqual(c2, 7)

            # Чтение дельты с курсора 3 возвращает ровно 4 новые точки (4, 5, 6, 7)
            read_cur_2, delta_batch = reader.read_tracks_since(3)
            self.assertEqual(read_cur_2, 7)
            assert delta_batch is not None
            np.testing.assert_allclose(delta_batch["pos_x"], [4.0, 5.0, 6.0, 7.0])
            np.testing.assert_array_equal(delta_batch["particle_id"], [14, 15, 16, 17])

            # Чтение всех точек с 0 возвращает последние 5 точек (3, 4, 5, 6, 7)
            _, all_tracks = reader.read_all_tracks()
            assert all_tracks is not None
            np.testing.assert_allclose(all_tracks["pos_x"], [3.0, 4.0, 5.0, 6.0, 7.0])
            np.testing.assert_array_equal(all_tracks["particle_id"], [13, 14, 15, 16, 17])

            # 3. Запись батча, сразу превышающего емкость кольца (6 > 5)
            c3 = writer.write_tracks(
                pos_x=np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32),
                pos_y=np.zeros(6, dtype=np.float32),
                pos_z=np.zeros(6, dtype=np.float32),
                particle_id=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
                process_id=np.zeros(6, dtype=np.int16),
            )
            self.assertEqual(c3, 13)
            _, overflow_tracks = reader.read_all_tracks()
            assert overflow_tracks is not None
            np.testing.assert_allclose(overflow_tracks["pos_x"], [11.0, 12.0, 13.0, 14.0, 15.0])

            # 4. Запись и чтение спектра (отрицательные и нулевые значения отбрасываются)
            s1 = writer.write_spectrum(np.array([0.0, -1.0, 0.140, 0.070, 0.035], dtype=np.float32))
            self.assertEqual(s1, 3)
            spec_cur, spec_vals = reader.read_spectrum()
            self.assertEqual(spec_cur, 3)
            np.testing.assert_allclose(spec_vals, [0.140, 0.070, 0.035], rtol=1e-5)

            # Переполнение спектра (емкость 4, добавляем еще 2 -> всего 5)
            s2 = writer.write_spectrum(np.array([0.200, 0.511], dtype=np.float32))
            self.assertEqual(s2, 5)
            spec_cur_2, spec_vals_2 = reader.read_spectrum()
            self.assertEqual(spec_cur_2, 5)
            np.testing.assert_allclose(spec_vals_2, [0.070, 0.035, 0.200, 0.511], rtol=1e-5)
        finally:
            reader.close(unlink=False)
            writer.close(unlink=True)

    def test_control_buffer_shm_and_focused_pause_proxy(self) -> None:
        """
        Проверка ControlBufferSHM и FocusedPauseProxy:
        1. Динамическое переключение focused_job_index на лету между task_id=0 и task_id=1.
        2. Локальная пауза фокуса тормозит только текущий focused_job_index, не затрагивая остальные задачи.
        3. Одиночный шаг trigger_step() пропускает ровно один шаг для сфокусированной задачи.
        4. Глобальная пауза останавливает все задачи независимо от фокуса.
        """
        ctrl_name = f"test_ctrl_shm_{os.getpid()}"
        ctrl = ControlBufferSHM(shm_name=ctrl_name, create=True, initial_focused_job=0)
        proxy_0 = FocusedPauseProxy(ctrl_shm_name=ctrl_name, task_id=0)
        proxy_1 = FocusedPauseProxy(ctrl_shm_name=ctrl_name, task_id=1)

        try:
            # Изначально оба воркера активны
            self.assertTrue(proxy_0.is_set())
            self.assertTrue(proxy_1.is_set())

            # Ставим локальную паузу сфокусированного воркера (сейчас фокус на task_id=0)
            ctrl.focused_run = False
            self.assertFalse(proxy_0.is_set())
            self.assertTrue(proxy_1.is_set())

            # Одиночный шаг для сфокусированного воркера 0
            ctrl.trigger_step()
            self.assertTrue(proxy_0.is_set())
            self.assertFalse(proxy_0.is_set())

            # Переключаем фокус на лету на task_id=1 при активной локальной паузе фокуса
            ctrl.focused_job_index = 1
            # Теперь воркер 0 мгновенно освобожден и бежит дальше, а воркер 1 встал на паузу!
            self.assertTrue(proxy_0.is_set())
            self.assertFalse(proxy_1.is_set())

            # Одиночный шаг теперь срабатывает для воркера 1
            ctrl.trigger_step()
            self.assertTrue(proxy_1.is_set())
            proxy_1.on_step_completed()
            self.assertFalse(proxy_1.is_set())

            # Возобновляем сфокусированный воркер
            ctrl.focused_run = True
            self.assertTrue(proxy_0.is_set())
            self.assertTrue(proxy_1.is_set())

            # Глобальная пауза останавливает всех
            ctrl.global_run = False
            self.assertFalse(proxy_0.is_set())
            self.assertFalse(proxy_1.is_set())
        finally:
            proxy_0.close()
            proxy_1.close()
            ctrl.close(unlink=True)

    def test_create_gui_stream_handler_per_task_shm_isolation(self) -> None:
        """
        Проверка create_gui_stream_handler с per_task_shm=True:
        1. Обработчики создаются для всех задач (task_id=0 и task_id=1), даже если focused_job_index=0.
        2. Каждая задача пишет 2D-проекцию, 3D-треки и спектр в собственные изолированные сегменты SharedMemory.
        3. Пакеты треков при наличии TelemetryRingBufferSHM не засоряют multiprocessing.Queue.
        """
        yaml_path = "tb161_nema_protocol.yaml"
        cfg = load_simulation_config(yaml_path)
        root_scene = SceneBuilder().build_scene(cfg.scene)

        base_shm = f"test_per_task_proj_{os.getpid()}"
        ctrl_shm_name = f"test_per_task_ctrl_{os.getpid()}"
        shape = (32, 32)
        nbytes = int(np.prod(shape) * np.dtype(np.float32).itemsize)

        ctrl = ControlBufferSHM(shm_name=ctrl_shm_name, create=True, initial_focused_job=0)
        proj_shms = []
        telem_rings = []
        handlers = []
        track_q: queue.Queue = queue.Queue()

        try:
            for tid in (0, 1):
                p_name = get_task_projection_shm_name(base_shm, tid)
                t_name = get_task_telemetry_shm_name(base_shm, tid)
                p_shm = shared_memory.SharedMemory(name=p_name, create=True, size=nbytes)
                np.ndarray(shape, dtype=np.float32, buffer=p_shm.buf).fill(0.0)
                proj_shms.append(p_shm)
                t_ring = TelemetryRingBufferSHM(shm_name=t_name, create=True, track_capacity=100, spectrum_capacity=100)
                telem_rings.append(t_ring)

            for tid in (0, 1):
                task_dict = cfg.model_dump(mode="python")
                created = create_gui_stream_handler(
                    root_scene=root_scene,
                    task_dict=task_dict,
                    task_id=tid,
                    telemetry_queue=track_q,
                    focused_job_index=0,
                    focused_shm_name=base_shm,
                    projection_shape=shape,
                    extra_args={"per_task_shm": True, "ctrl_shm_name": ctrl_shm_name},
                )
                self.assertEqual(len(created), 1)
                handler_item = created[0]
                self.assertIsInstance(handler_item, GuiStreamDataHandler)
                assert isinstance(handler_item, GuiStreamDataHandler)
                self.assertEqual(handler_item.shm_name, get_task_projection_shm_name(base_shm, tid))
                self.assertEqual(handler_item.telemetry_shm_name, get_task_telemetry_shm_name(base_shm, tid))
                handlers.append(handler_item)

            det1_pos = handlers[0].sensitive_volumes[0].global_matrix[:3, 3]

            # Задача 0 регистрирует 1 фотон (0.140 МэВ), Задача 1 регистрирует 2 фотона (0.100 и 0.140 МэВ)
            handlers[0].process_chunk({
                "type": ChunkType.INTERACTIONS,
                "data": {
                    "particle_ID": np.array([10], dtype=np.uint64),
                    "volume_id": np.array([6], dtype=np.uint32),
                    "pos_x": np.array([det1_pos[0]], dtype=np.float32),
                    "pos_y": np.array([det1_pos[1]], dtype=np.float32),
                    "pos_z": np.array([det1_pos[2]], dtype=np.float32),
                    "process_id": np.array([2], dtype=np.uint16),
                    "energy_deposit": np.array([0.140], dtype=np.float32),
                },
            })
            handlers[1].process_chunk({
                "type": ChunkType.INTERACTIONS,
                "data": {
                    "particle_ID": np.array([20, 21], dtype=np.uint64),
                    "volume_id": np.array([6, 6], dtype=np.uint32),
                    "pos_x": np.array([det1_pos[0], det1_pos[0]], dtype=np.float32),
                    "pos_y": np.array([det1_pos[1], det1_pos[1]], dtype=np.float32),
                    "pos_z": np.array([det1_pos[2], det1_pos[2]], dtype=np.float32),
                    "process_id": np.array([2, 2], dtype=np.uint16),
                    "energy_deposit": np.array([0.100, 0.140], dtype=np.float32),
                },
            })

            # Очередь треков пуста (треки пошли через SHM кольцо)
            self.assertTrue(track_q.empty())

            # Проверяем изолированные проекции
            snap_0 = handlers[0].get_projection_snapshot()
            snap_1 = handlers[1].get_projection_snapshot()
            assert snap_0 is not None and snap_1 is not None
            self.assertEqual(float(np.sum(snap_0)), 1.0)
            self.assertEqual(float(np.sum(snap_1)), 2.0)

            # Проверяем изолированные кольца треков и спектров
            cur_0, tracks_0 = telem_rings[0].read_all_tracks()
            cur_1, tracks_1 = telem_rings[1].read_all_tracks()
            self.assertEqual(cur_0, 1)
            self.assertEqual(cur_1, 2)
            assert tracks_0 is not None and tracks_1 is not None
            np.testing.assert_array_equal(tracks_0["particle_id"], [10])
            np.testing.assert_array_equal(tracks_1["particle_id"], [20, 21])

            _, spec_0 = telem_rings[0].read_spectrum()
            _, spec_1 = telem_rings[1].read_spectrum()
            np.testing.assert_allclose(spec_0, [0.140], rtol=1e-5)
            np.testing.assert_allclose(spec_1, [0.100, 0.140], rtol=1e-5)

            # Проверяем именование пер-тасковых сегментов дозы
            self.assertEqual(get_task_dose_shm_name("nmsim_dose_shm", 2), "nmsim_dose_shm_task_2")
            self.assertEqual(
                get_task_dose_shm_name("nmsim_dose_shm", 2, grid_name="GridA", multi_grid=True),
                "nmsim_dose_shm_task_2_GridA",
            )
        finally:
            for handler_item in handlers:
                handler_item.close()
            for t_ring in telem_rings:
                t_ring.close(unlink=True)
            for p_shm in proj_shms:
                try:
                    p_shm.close()
                    p_shm.unlink()
                except (FileNotFoundError, OSError):
                    pass
            ctrl.close(unlink=True)

    def test_cooperative_stop_via_control_buffer_and_clean_hdf5_close(self) -> None:
        """
        Проверка кооперативной остановки воркера через ControlBufferSHM + FocusedPauseProxy + IpcPauseBridge
        и открытия HDF5 без Superblock v3 флагов блокировки при swmr=False.
        """
        clean_path = Path(self.temp_dir.name) / "cooperative_stop_clean.h5"
        dm = DataManager(
            filename=clean_path,
            handlers=[],
            swmr=False,
            metadata={"simulation": {"simulated_histories": 10}},
            task_id=0,
        )
        dm.initialize_metadata()
        dm.finalize_metadata(status="completed")

        # При swmr=False используется стандартный формат HDF5 (Superblock v0/v2 без флагов консистентности v3)
        with open(clean_path, "rb") as raw_f:
            header = raw_f.read(16)
            self.assertLess(header[8], 3)

        ctrl_name = f"test_coop_stop_ctrl_{os.getpid()}"
        ctrl = ControlBufferSHM(shm_name=ctrl_name, create=True, initial_focused_job=0)
        proxy = FocusedPauseProxy(ctrl_shm_name=ctrl_name, task_id=0)

        class _DummySimManager:
            def __init__(self) -> None:
                self.state = SimulationState.RUNNING
                self.paused = False
                self.stopped = False

            def pause(self) -> None:
                self.paused = True
                self.state = SimulationState.PAUSED

            def resume(self) -> None:
                self.paused = False
                self.state = SimulationState.RUNNING

            def stop(self) -> None:
                self.stopped = True
                self.state = SimulationState.STOPPED

        dummy_mgr = _DummySimManager()
        bridge = IpcPauseBridge(
            manager=dummy_mgr,
            ipc_event=proxy,
            poll_interval=0.01,
            stop_check=proxy.check_stop_requested,
        )
        bridge.start()
        try:
            ctrl.global_run = False
            time.sleep(0.03)
            self.assertTrue(dummy_mgr.paused)
            self.assertFalse(proxy.is_set())

            # Запрос кооперативной остановки разблокирует proxy и вызывает manager.stop()
            ctrl.stop_requested = True
            self.assertTrue(proxy.check_stop_requested())
            self.assertTrue(proxy.is_set())
            self.assertTrue(proxy.wait(timeout=0.05))
            time.sleep(0.03)
            self.assertTrue(dummy_mgr.stopped)
        finally:
            bridge.stop_bridge()
            bridge.join(timeout=0.5)
            proxy.close()
            ctrl.close(unlink=True)


if __name__ == '__main__':
    unittest.main()

