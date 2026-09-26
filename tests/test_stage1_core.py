import os
import queue
import tempfile
import unittest
from multiprocessing import Queue

import numpy as np

from core.config.exporter import SceneExporter
from core.config.models import SimulationConfig
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
        mgr = SimulationManager(scene=scene, particles_number=10)
        self.assertEqual(mgr.state, SimulationState.IDLE)

        mgr.pause()
        self.assertEqual(mgr.state, SimulationState.PAUSED)

        mgr.resume()
        self.assertEqual(mgr.state, SimulationState.RUNNING)

        mgr.stop()
        self.assertEqual(mgr.state, SimulationState.STOPPED)
        self.assertTrue(mgr._stop_event.is_set())

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


if __name__ == '__main__':
    unittest.main()
