import threading
import time
import unittest
from multiprocessing import Queue, shared_memory

import hepunits as units
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
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.orchestrator_session import OrchestratorSession
from gui.controllers.simulation_runner import SimulationRunner
from gui.controllers.stream_handlers import (
    ControlBufferSHM,
    FocusedPauseProxy,
    TelemetryRingBufferSHM,
    get_task_dose_shm_name,
    get_task_projection_shm_name,
    get_task_telemetry_shm_name,
)


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
        telemetry_queue: Queue = Queue()
        shm_name = "test_ipc_receiver_shm"
        shape = (16, 16)
        nbytes = int(np.prod(shape) * 4)

        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
            buf = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
            buf.fill(42.0)

            receiver = IPCReceiver(
                track_queue=telemetry_queue,
                shm_name=shm_name,
                projection_shape=shape,
                fps=60.0
            )

            tracks_received = []
            proj_received = []

            receiver.tracks_received.connect(lambda track_batch: tracks_received.append(track_batch))
            receiver.projection_received.connect(lambda proj_arr: proj_received.append(proj_arr))

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
            telemetry_queue.put(sample_batch)

            # Ожидаем обработку в цикле IPCReceiver с обработкой событий Qt
            for _ in range(30):
                if app is not None:
                    app.processEvents()
                if len(tracks_received) > 0 and len(proj_received) > 0:
                    break
                time.sleep(0.02)

            receiver.stop()

            self.assertGreater(len(tracks_received), 0)
            self.assertEqual(tracks_received[0]['type'], 'tracks')

            self.assertGreater(len(proj_received), 0)
            self.assertEqual(proj_received[0].shape, shape)
            self.assertAlmostEqual(float(proj_received[0][0, 0]), 42.0)

            proj_received.clear()
            receiver.close()
            del buf

        finally:
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, OSError, BufferError):
                pass

    def test_focused_pause_proxy_behavior(self):
        """Проверка FocusedPauseProxy для двухуровневой паузы и пошагового выполнения."""
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

    def test_per_task_shm_focus_switching(self):
        """
        Проверка пер-тасковых сегментов SharedMemory и мгновенного переключения фокуса в IPCReceiver
        между задачами без потерь накопленных данных (проекция, треки, спектр, 3D-доза).
        """
        base_proj_shm = "test_switch_proj_shm"
        base_dose_shm = "test_switch_dose_shm"
        ctrl_shm_name = "test_switch_ctrl_shm"
        proj_shape = (8, 8)
        dose_shape = (4, 4, 4)
        proj_nbytes = int(np.prod(proj_shape) * np.dtype(np.float32).itemsize)
        dose_nbytes = int(np.prod(dose_shape) * np.dtype(np.float32).itemsize)

        proj_shms = []
        dose_shms = []
        telem_rings = []
        ctrl_shm = None
        receiver = None
        received_proj = []
        received_tracks = []
        received_spec = []
        received_dose = []

        try:
            ctrl_shm = ControlBufferSHM(shm_name=ctrl_shm_name, create=True, initial_focused_job=0)

            for tid in range(2):
                p_name = get_task_projection_shm_name(base_proj_shm, tid)
                p_shm = shared_memory.SharedMemory(name=p_name, create=True, size=proj_nbytes)
                p_buf = np.ndarray(proj_shape, dtype=np.float32, buffer=p_shm.buf)
                p_buf.fill(10.0 if tid == 0 else 75.0)
                del p_buf
                proj_shms.append(p_shm)

                t_name = get_task_telemetry_shm_name(base_proj_shm, tid)
                ring = TelemetryRingBufferSHM(shm_name=t_name, create=True, track_capacity=200, spectrum_capacity=200)
                if tid == 0:
                    ring.write_tracks(
                        pos_x=np.array([1.0, 2.0], dtype=np.float32),
                        pos_y=np.array([3.0, 4.0], dtype=np.float32),
                        pos_z=np.array([5.0, 6.0], dtype=np.float32),
                        particle_id=np.array([101, 101], dtype=np.int64),
                        process_id=np.array([-2, 1], dtype=np.int16),
                    )
                    ring.write_spectrum(np.array([0.140 * units.MeV, 0.140 * units.MeV], dtype=np.float32))
                else:
                    ring.write_tracks(
                        pos_x=np.array([10.0, 20.0, 30.0], dtype=np.float32),
                        pos_y=np.array([11.0, 21.0, 31.0], dtype=np.float32),
                        pos_z=np.array([12.0, 22.0, 32.0], dtype=np.float32),
                        particle_id=np.array([202, 202, 202], dtype=np.int64),
                        process_id=np.array([-2, 1, 2], dtype=np.int16),
                    )
                    ring.write_spectrum(np.array([0.511 * units.MeV], dtype=np.float32))
                telem_rings.append(ring)

                d_name = get_task_dose_shm_name(base_dose_shm, tid, multi_grid=False)
                d_shm = shared_memory.SharedMemory(name=d_name, create=True, size=dose_nbytes)
                d_buf = np.ndarray(dose_shape, dtype=np.float32, buffer=d_shm.buf, order="F")
                d_buf.fill(1.5 if tid == 0 else 9.5)
                del d_buf
                dose_shms.append(d_shm)

            receiver = IPCReceiver(
                shm_name=base_proj_shm,
                projection_shape=proj_shape,
                dose_shm_name=base_dose_shm,
                dose_grid_shape=dose_shape,
                focused_job_index=0,
                num_tasks=2,
                per_task_shm=True,
                base_dose_shm_name=base_dose_shm,
            )

            receiver.projection_received.connect(lambda proj_arr: received_proj.append(proj_arr.copy()))
            receiver.tracks_received.connect(received_tracks.append)
            receiver.spectrum_received.connect(received_spec.append)
            receiver.dose_volume_received.connect(lambda dose_arr: received_dose.append(dose_arr.copy()))

            # 1. Опрос при фокусе на задаче 0
            receiver._poll_once()
            self.assertEqual(len(received_proj), 1)
            self.assertAlmostEqual(float(received_proj[-1][0, 0]), 10.0)
            self.assertEqual(len(received_tracks), 1)
            self.assertTrue(received_tracks[-1].get("reset"))
            np.testing.assert_allclose(received_tracks[-1]["pos_x"], [1.0, 2.0])
            self.assertEqual(len(received_spec), 1)
            self.assertEqual(len(received_spec[-1]), 2)
            self.assertAlmostEqual(float(received_spec[-1][0]), 140.0, places=2)
            self.assertEqual(len(received_dose), 1)
            self.assertAlmostEqual(float(received_dose[-1][0, 0, 0]), 1.5)

            # 2. Переключение фокуса на задачу 1
            receiver.set_focused_job(1)
            receiver._poll_once()
            self.assertEqual(len(received_proj), 2)
            self.assertAlmostEqual(float(received_proj[-1][0, 0]), 75.0)
            self.assertEqual(len(received_tracks), 2)
            self.assertTrue(received_tracks[-1].get("reset"))
            np.testing.assert_allclose(received_tracks[-1]["pos_x"], [10.0, 20.0, 30.0])
            self.assertEqual(len(received_spec), 2)
            self.assertEqual(len(received_spec[-1]), 1)
            self.assertAlmostEqual(float(received_spec[-1][0]), 511.0, places=2)
            self.assertEqual(len(received_dose), 2)
            self.assertAlmostEqual(float(received_dose[-1][0, 0, 0]), 9.5)

            # 3. Возврат фокуса на задачу 0 — все накопленные данные задачи 0 сохранены без потерь
            receiver.set_focused_job(0)
            receiver._poll_once()
            self.assertEqual(len(received_proj), 3)
            self.assertAlmostEqual(float(received_proj[-1][0, 0]), 10.0)
            self.assertEqual(len(received_tracks), 3)
            self.assertTrue(received_tracks[-1].get("reset"))
            np.testing.assert_allclose(received_tracks[-1]["pos_x"], [1.0, 2.0])
            self.assertEqual(len(received_spec), 3)
            self.assertEqual(len(received_spec[-1]), 2)
            self.assertAlmostEqual(float(received_spec[-1][0]), 140.0, places=2)
            self.assertEqual(len(received_dose), 3)
            self.assertAlmostEqual(float(received_dose[-1][0, 0, 0]), 1.5)

        finally:
            received_proj.clear()
            received_dose.clear()
            if receiver is not None:
                receiver.close()
            if ctrl_shm is not None:
                ctrl_shm.close(unlink=True)
            for ring in telem_rings:
                ring.close(unlink=True)
            for p_shm in proj_shms:
                try:
                    p_shm.close()
                    p_shm.unlink()
                except (FileNotFoundError, OSError):
                    pass
            for d_shm in dose_shms:
                try:
                    d_shm.close()
                    d_shm.unlink()
                except (FileNotFoundError, OSError):
                    pass


if __name__ == '__main__':
    unittest.main()

