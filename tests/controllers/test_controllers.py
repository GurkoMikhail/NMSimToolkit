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

    def test_focused_pause_proxy_behavior(self):
        """Проверка FocusedPauseProxy для двухуровневой паузы и пошагового выполнения."""
        import threading
        from gui.controllers.stream_handlers import FocusedPauseProxy

        global_pause = threading.Event()
        focused_pause = threading.Event()
        step_trigger = threading.Event()

        # По умолчанию воркеры запущены (события взведены)
        global_pause.set()
        focused_pause.set()

        proxy = FocusedPauseProxy(
            global_pause_event=global_pause,
            focused_pause_event=focused_pause,
            step_trigger_event=step_trigger
        )

        # 1. По умолчанию воркер не на паузе (is_set() возвращает True -> разрешено выполнение)
        self.assertTrue(proxy.is_set())

        # 2. Глобальная пауза всего пула (сброс флага активности)
        global_pause.clear()
        self.assertFalse(proxy.is_set())
        global_pause.set()
        self.assertTrue(proxy.is_set())

        # 3. Локальная пауза сфокусированного воркера
        focused_pause.clear()
        self.assertFalse(proxy.is_set())

        # 4. Одиночный шаг частиц (Particle Batch Step) при нахождении на паузе
        step_trigger.set()
        # Первый вызов is_set() потребляет триггер и возвращает True (шаг разрешен)
        self.assertTrue(proxy.is_set())
        # Следующий вызов is_set() снова возвращает False (шаг завершен, пауза восстановлена)
        self.assertFalse(proxy.is_set())
        # Триггер шага сброшен
        self.assertFalse(step_trigger.is_set())

    def test_orchestrator_session_two_level_pause_controls(self):
        """Проверка методов и сигналов двухуровневой паузы в OrchestratorSession."""
        import threading
        from gui.controllers.orchestrator_session import OrchestratorSession

        session = OrchestratorSession(pool_size=2)
        session._is_running = True
        session._global_pause_event = threading.Event()
        session._global_pause_event.set()
        session._focused_pause_event = threading.Event()
        session._focused_pause_event.set()
        session._step_trigger_event = threading.Event()

        paused_signals = []
        resumed_signals = []
        focused_paused_signals = []
        focused_resumed_signals = []

        session.session_paused.connect(lambda: paused_signals.append(True))
        session.session_resumed.connect(lambda: resumed_signals.append(True))
        session.focused_worker_paused.connect(lambda tid: focused_paused_signals.append(tid))
        session.focused_worker_resumed.connect(lambda tid: focused_resumed_signals.append(tid))

        # 1. Глобальная пауза пула
        session.pause_all()
        self.assertTrue(session.is_paused)
        self.assertFalse(session._global_pause_event.is_set())
        self.assertEqual(len(paused_signals), 1)

        # 2. Возобновление пула
        session.resume_all()
        self.assertFalse(session.is_paused)
        self.assertTrue(session._global_pause_event.is_set())
        self.assertEqual(len(resumed_signals), 1)

        # 3. Локальная пауза сфокусированного воркера
        session.set_focused_job(1)
        session.pause_focused_worker()
        self.assertTrue(session.is_focused_worker_paused)
        self.assertFalse(session._focused_pause_event.is_set())
        self.assertEqual(focused_paused_signals, [1])

        # 4. Шаг сфокусированного воркера
        session.step_focused_worker()
        if session._step_trigger_event is not None:
            self.assertTrue(session._step_trigger_event.is_set())

        # 5. Возобновление сфокусированного воркера
        session.resume_focused_worker()
        self.assertFalse(session.is_focused_worker_paused)
        self.assertTrue(session._focused_pause_event.is_set())
        self.assertEqual(focused_resumed_signals, [1])


if __name__ == '__main__':
    unittest.main()
