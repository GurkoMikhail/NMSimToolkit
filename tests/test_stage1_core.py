import os
import queue
import tempfile
import threading
import time
import unittest
from multiprocessing import Queue

import numpy as np

from core.config.exporter import SceneExporter
from core.config.models import SimulationConfig
from core.config.orchestrator import IpcPauseBridge
from core.data.data_manager import DataManager
from gui.controllers.stream_handlers import GuiStreamDataHandler
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.nodes import SpatialNode, CompositeNode
from core.transport.simulation_managers import SimulationManager, SimulationState


class TestStage1Core(unittest.TestCase):
    def test_simulation_state_and_events(self):
        """Проверка состояний и методов управления SimulationManager."""
        # Создаем минимальный узел сцены
        scene = CompositeNode()
        manager = SimulationManager(scene=scene, particles_number=10)
        self.assertEqual(manager.state, SimulationState.IDLE)
        self.assertFalse(manager.is_stopped)
        self.assertFalse(manager.has_active_particles)
        self.assertFalse(manager.has_pending_sources)

        manager.pause()
        self.assertEqual(manager.state, SimulationState.PAUSED)
        self.assertFalse(manager.is_stopped)

        manager.resume()
        self.assertEqual(manager.state, SimulationState.RUNNING)
        self.assertFalse(manager.is_stopped)

        manager.step_once()
        self.assertEqual(manager.state, SimulationState.RUNNING)
        self.assertFalse(manager.is_stopped)

        manager.stop()
        self.assertEqual(manager.state, SimulationState.STOPPED)
        self.assertTrue(manager.is_stopped)

        # Проверка терминальности STOPPED: pause/resume не могут перевести в PAUSED/RUNNING
        manager.pause()
        self.assertEqual(manager.state, SimulationState.STOPPED)
        self.assertTrue(manager.is_stopped)
        manager.resume()
        self.assertEqual(manager.state, SimulationState.STOPPED)
        self.assertTrue(manager.is_stopped)

    def test_gui_stream_data_handler(self):
        """Проверка GuiStreamDataHandler с очередью и разделяемой памятью."""
        q = Queue()
        shm_name = f"test_nmsim_shm_proj_{os.getpid()}"
        handler = GuiStreamDataHandler(
            track_queue=q,
            shm_name=shm_name,
            projection_shape=(32, 32),
            create_shm=True,
            detector_size=100.0,
            sensitive_volume_ids={0},
            max_tracks_per_batch=100
        )

        try:
            chunk = {
                'type': 'interactions',
                'data': {
                    'pos_x': np.array([0.0, 10.0, -10.0], dtype=np.float32),
                    'pos_y': np.array([0.0, 10.0, -10.0], dtype=np.float32),
                    'pos_z': np.array([50.0, 50.0, 50.0], dtype=np.float32),
                    'process_id': np.array([1, 1, 2], dtype=np.int32),
                    'particle_ID': np.array([101, 102, 103], dtype=np.int64),
                    'energy_deposit': np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    'volume_id': np.array([0, 0, 0], dtype=np.int32),
                }
            }

            handler.process_chunk(chunk)

            # Проверяем получение данных в очереди треков с надежным ожиданием
            try:
                item = q.get(timeout=2.0)
            except queue.Empty:
                self.fail("Таймаут: пакет данных треков не был получен из очереди за 2.0 секунды")

            self.assertEqual(item['type'], 'tracks')
            self.assertEqual(len(item['pos_x']), 3)
            self.assertEqual(len(item['particle_id']), 3)

            # Проверяем накопление проекции
            proj = handler.get_projection_snapshot()
            self.assertIsNotNone(proj)
            self.assertEqual(proj.shape, (32, 32))
            self.assertGreater(np.sum(proj), 0)

            # Проверка фильтрации FOV: взаимодействие вне поля зрения не должно накапливаться
            prev_sum = float(np.sum(proj))
            out_of_bounds_chunk = {
                'type': 'interactions',
                'data': {
                    'pos_x': np.array([500.0], dtype=np.float32),
                    'pos_y': np.array([500.0], dtype=np.float32),
                    'pos_z': np.array([0.0], dtype=np.float32),
                    'particle_id': np.array([200], dtype=np.int64),
                    'volume_id': np.array([0], dtype=np.int32),
                }
            }
            handler.process_chunk(out_of_bounds_chunk)
            proj_after = handler.get_projection_snapshot()
            self.assertEqual(float(np.sum(proj_after)), prev_sum)

            # Проверка устойчивости к пустым и некорректным чанкам
            handler.process_chunk({'type': 'invalid', 'data': None})
            handler.process_chunk({'type': 'interactions', 'data': {}})
        finally:
            handler.close()
            q.close()
            q.join_thread()

    def test_data_manager_swmr_flag(self):
        """Проверка инициализации DataManager с флагом swmr."""
        dm = DataManager(filename="test_swmr.h5", handlers=[], swmr=True)
        self.assertTrue(dm.swmr)

    def test_scene_exporter(self):
        """Проверка обратной сериализации графа сцены в SimulationConfig и YAML."""
        mat = Material(name="Water")
        geo = Box(x=100.0, y=100.0, z=50.0)
        vol = Volume(geometry=geo, material=mat, name="WaterBox")
        vol.translate(x=10.0, y=20.0, z=30.0)

        cfg = SceneExporter.export_to_config(vol)
        self.assertIsInstance(cfg, SimulationConfig)
        self.assertEqual(cfg.scene.name, "WaterBox")
        self.assertEqual(len(cfg.scene.transformations), 1)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            SceneExporter.export_to_yaml(vol, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def test_simulation_manager_buffer_capacity_invariant(self):
        """Проверка инварианта: buffer_capacity тихо поднимается до particles_number при передаче меньшего значения."""
        scene = CompositeNode()
        # buffer_capacity < particles_number: тихий clamp, исключения нет
        mgr_clamped = SimulationManager(scene=scene, particles_number=5000, buffer_capacity=1000)
        self.assertEqual(mgr_clamped.data_buffer.interactions.capacity, 5000)

        # buffer_capacity >= particles_number: значение сохраняется как есть
        mgr = SimulationManager(scene=scene, particles_number=5000, buffer_capacity=5000)
        self.assertEqual(mgr.data_buffer.interactions.capacity, 5000)

        # При buffer_capacity=None ёмкость автоматически равна particles_number
        mgr_auto = SimulationManager(scene=scene, particles_number=150000, buffer_capacity=None)
        self.assertEqual(mgr_auto.data_buffer.interactions.capacity, 150000)

    def test_causal_flush_ordering(self):
        """Проверка причинно-следственного порядка сброса: initial_states -> interactions -> dead_particles."""
        scene = CompositeNode()
        received_chunks = []
        test_queue = queue.Queue()
        mgr = SimulationManager(scene=scene, particles_number=10, buffer_capacity=10, queue=test_queue)

        # Имитируем накопление данных во всех трех буферах
        mgr.data_buffer.initial_states.particle_ID[0] = 1
        mgr.data_buffer.initial_states.emission_time[0] = 0.0
        mgr.data_buffer.initial_states.emission_energy[0] = 140.0
        mgr.data_buffer.initial_states.emission_position.x[0] = 0.0
        mgr.data_buffer.initial_states.emission_position.y[0] = 0.0
        mgr.data_buffer.initial_states.emission_position.z[0] = 0.0
        mgr.data_buffer.initial_states.emission_direction.x[0] = 0.0
        mgr.data_buffer.initial_states.emission_direction.y[0] = 0.0
        mgr.data_buffer.initial_states.emission_direction.z[0] = 1.0
        mgr.data_buffer.initial_states.cursor[0] = 1

        mgr.data_buffer.interactions.process_id[0] = 1
        mgr.data_buffer.interactions.volume_id[0] = 0
        mgr.data_buffer.interactions.material_id[0] = 1
        mgr.data_buffer.interactions.particle_ID[0] = 1
        mgr.data_buffer.interactions.energy_deposit[0] = 20.0
        mgr.data_buffer.interactions.scattering_theta[0] = 0.1
        mgr.data_buffer.interactions.scattering_phi[0] = 0.2
        mgr.data_buffer.interactions.distance_traveled[0] = 5.0
        mgr.data_buffer.interactions.species[0] = 0
        mgr.data_buffer.interactions.Z[0] = 7.4
        mgr.data_buffer.interactions.position.x[0] = 0.0
        mgr.data_buffer.interactions.position.y[0] = 0.0
        mgr.data_buffer.interactions.position.z[0] = 5.0
        mgr.data_buffer.interactions.direction.x[0] = 0.0
        mgr.data_buffer.interactions.direction.y[0] = 0.0
        mgr.data_buffer.interactions.direction.z[0] = 1.0
        mgr.data_buffer.interactions.cursor[0] = 1

        mgr.data_buffer.dead_particles.append(np.array([1], dtype=np.int32))

        # Вызов flush_dead_particles должен сначала сбросить initial_states и interactions
        mgr.flush_dead_particles()

        while not test_queue.empty():
            received_chunks.append(test_queue.get())

        chunk_types = [c['type'] for c in received_chunks if isinstance(c, dict)]
        self.assertEqual(chunk_types, ['initial_states', 'interactions', 'dead_particles'])

    def test_flush_all_ordering(self):
        """Проверка причинно-следственного порядка сброса через flush_all: initial_states -> interactions -> dead_particles."""
        scene = CompositeNode()
        received_chunks = []
        test_queue = queue.Queue()
        mgr = SimulationManager(scene=scene, particles_number=10, buffer_capacity=10, queue=test_queue)

        # Заполняем буферы данными
        mgr.data_buffer.initial_states.particle_ID[0] = 42
        mgr.data_buffer.initial_states.cursor[0] = 1

        mgr.data_buffer.interactions.particle_ID[0] = 42
        mgr.data_buffer.interactions.cursor[0] = 1

        mgr.data_buffer.dead_particles.append(np.array([42], dtype=np.int32))

        # Вызов flush_all должен произвести сброс всех трех буферов в правильном порядке
        mgr.flush_all()

        while not test_queue.empty():
            received_chunks.append(test_queue.get())

        chunk_types = [c['type'] for c in received_chunks if isinstance(c, dict)]
        self.assertEqual(chunk_types, ['initial_states', 'interactions', 'dead_particles'])

    def test_ipc_pause_bridge(self):
        """Проверка адаптера межпроцессной паузы IpcPauseBridge."""
        scene = CompositeNode()
        manager = SimulationManager(scene=scene, particles_number=10)
        ipc_event = threading.Event()
        ipc_event.set()

        bridge = IpcPauseBridge(manager=manager, ipc_event=ipc_event, poll_interval_seconds=0.01)
        bridge.start()

        try:
            # Исходно менеджер не на паузе
            self.assertEqual(manager.state, SimulationState.IDLE)

            # Симулируем сигнал паузы из другого процесса
            ipc_event.clear()
            time.sleep(0.04)
            self.assertEqual(manager.state, SimulationState.PAUSED)

            # Симулируем снятие паузы
            ipc_event.set()
            time.sleep(0.04)
            self.assertEqual(manager.state, SimulationState.RUNNING)
        finally:
            bridge.stop_bridge()
            bridge.join(timeout=1.0)
            self.assertFalse(bridge.is_alive())


if __name__ == '__main__':
    unittest.main()
