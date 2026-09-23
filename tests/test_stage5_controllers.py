import time
import unittest
from multiprocessing import Queue, shared_memory

import numpy as np

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(['-platform', 'offscreen'])
except ImportError:
    try:
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication(['-platform', 'offscreen'])
    except ImportError:
        app = None

from core.scene.nodes import CompositeNode
from core.transport.simulation_managers import SimulationManager, SimulationState
from gui.controllers.simulation_runner import SimulationRunner
from gui.controllers.ipc_receiver import IPCReceiver


class TestStage5Controllers(unittest.TestCase):
    def test_simulation_runner_controls(self):
        """Проверка управляющих команд SimulationRunner."""
        scene = CompositeNode()
        mgr = SimulationManager(scene=scene, particles_number=10)
        runner = SimulationRunner(manager=mgr)

        paused_signals = []
        resumed_signals = []
        stopped_signals = []

        runner.simulation_paused.connect(lambda: paused_signals.append(True))
        runner.simulation_resumed.connect(lambda: resumed_signals.append(True))
        runner.simulation_stopped.connect(lambda: stopped_signals.append(True))

        runner.pause()
        self.assertEqual(len(paused_signals), 1)
        self.assertEqual(mgr.state, SimulationState.PAUSED)

        runner.resume()
        self.assertEqual(len(resumed_signals), 1)
        self.assertEqual(mgr.state, SimulationState.RUNNING)

        runner.stop()
        self.assertEqual(len(stopped_signals), 1)
        self.assertEqual(mgr.state, SimulationState.STOPPED)

    def test_ipc_receiver_stream(self):
        """Проверка передачи треков и проекций через IPCReceiver."""
        q = Queue()
        shm_name = "test_ipc_receiver_shm"
        shape = (16, 16)
        nbytes = int(np.prod(shape) * 4)

        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
            buf = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
            buf.fill(42.0)

            receiver = IPCReceiver(
                track_queue=q,
                shm_name=shm_name,
                projection_shape=shape,
                fps=60.0
            )

            tracks_received = []
            proj_received = []

            receiver.tracks_received.connect(lambda t: tracks_received.append(t))
            receiver.projection_received.connect(lambda p: proj_received.append(p))

            receiver.start()

            # Отправляем тестовый пакет треков в очередь
            sample_batch = {
                'type': 'tracks',
                'pos_x': np.array([1.0, 2.0]),
                'pos_y': np.array([3.0, 4.0]),
                'pos_z': np.array([5.0, 6.0]),
                'process_id': np.array([1, 2]),
                'energy_deposit': np.array([0.5, 0.8])
            }
            q.put(sample_batch)

            # Ожидаем обработку в цикле IPCReceiver с обработкой событий Qt
            for _ in range(30):
                if app is not None:
                    app.processEvents()
                if len(tracks_received) > 0 and len(proj_received) > 0:
                    break
                time.sleep(0.02)

            receiver.stop()
            receiver.close()

            self.assertGreater(len(tracks_received), 0)
            self.assertEqual(tracks_received[0]['type'], 'tracks')

            self.assertGreater(len(proj_received), 0)
            self.assertEqual(proj_received[0].shape, shape)
            self.assertAlmostEqual(proj_received[0][0, 0], 42.0)

        finally:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
