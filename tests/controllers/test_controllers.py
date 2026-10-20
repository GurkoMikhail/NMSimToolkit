import threading
import time
from typing import Any, Dict, List
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

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.scene.gamma_camera_node import GammaCameraNode
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
    safe_close_shm,
)
from gui.viewmodels.data_handler_viewmodel import (
    DataManagerViewModel,
    SensitiveVolumeHandlerViewModel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel


class TestStage5Controllers(unittest.TestCase):
    def test_simulation_runner_controls(self) -> None:
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

    def test_ipc_receiver_stream(self) -> None:
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

    def test_focused_pause_proxy_behavior(self) -> None:
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

    def test_orchestrator_session_two_level_pause_controls(self) -> None:
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

    def test_per_task_shm_focus_switching(self) -> None:
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
        received_proj: List[np.ndarray] = []
        received_tracks: List[Dict[str, Any]] = []
        received_spec: List[np.ndarray] = []
        received_dose: List[np.ndarray] = []

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

    def test_ipc_receiver_4d_cube_and_orchestrator_detector_metadata(self) -> None:
        """
        Проверка передачи 4D гиперспектрального куба (N_det, E_bins, Rows, Cols) через IPCReceiver
        и формирования detector_names / сигнала detector_metadata_updated в OrchestratorSession.
        """
        dm_vm = DataManagerViewModel(handlers=[
            SensitiveVolumeHandlerViewModel(sensitive_volumes=["Crystal1", "Crystal2"], enabled=True),
        ])
        session = OrchestratorSession(data_manager_vm=dm_vm, projection_shape=(32, 32))
        self.assertEqual(session.detector_names, ["Crystal1", "Crystal2"])
        self.assertEqual(session.energy_bins, 256)
        self.assertAlmostEqual(session.max_energy, 0.3)

        meta_received = []
        session.detector_metadata_updated.connect(
            lambda names, max_e: meta_received.append((list(names), float(max_e)))
        )
        session.detector_metadata_updated.emit(session.detector_names, float(session.max_energy / units.keV))
        self.assertEqual(meta_received, [(["Crystal1", "Crystal2"], 300.0)])

        base_proj_shm = "test_4d_ipc_proj_shm"
        cube_shape = (2, 256, 16, 16)
        cube_nbytes = int(np.prod(cube_shape) * np.dtype(np.float32).itemsize)
        p_name = get_task_projection_shm_name(base_proj_shm, 0)
        t_name = get_task_telemetry_shm_name(base_proj_shm, 0)
        p_shm = None
        t_ring = None
        receiver = None
        received_4d = []

        try:
            p_shm = shared_memory.SharedMemory(name=p_name, create=True, size=cube_nbytes)
            p_buf = np.ndarray(cube_shape, dtype=np.float32, buffer=p_shm.buf)
            p_buf.fill(0.0)
            p_buf[0, 119, 4, 5] = 3.0
            p_buf[1, 119, 8, 9] = 7.0
            del p_buf

            t_ring = TelemetryRingBufferSHM(shm_name=t_name, create=True, track_capacity=64, spectrum_capacity=64)
            receiver = IPCReceiver(
                shm_name=base_proj_shm,
                projection_shape=cube_shape,
                focused_job_index=0,
                num_tasks=1,
                per_task_shm=True,
            )
            receiver.projection_received.connect(lambda arr: received_4d.append(arr.copy()))
            receiver._poll_once()

            self.assertEqual(len(received_4d), 1)
            self.assertEqual(received_4d[0].shape, cube_shape)
            self.assertAlmostEqual(float(received_4d[0][0, 119, 4, 5]), 3.0)
            self.assertAlmostEqual(float(received_4d[0][1, 119, 8, 9]), 7.0)
            self.assertAlmostEqual(float(np.sum(received_4d[0])), 10.0)
        finally:
            received_4d.clear()
            if receiver is not None:
                receiver.close()
            if t_ring is not None:
                t_ring.close(unlink=True)
            if p_shm is not None:
                try:
                    p_shm.close()
                    p_shm.unlink()
                except (FileNotFoundError, OSError):
                    pass

    def test_shm_recreation_with_smaller_locked_segment_winerror_183(self) -> None:
        """
        Регрессионный тест на ошибку Windows [WinError 183] File exists: 'nmsim_gui_proj_shm_task_16':
        если сегмент SharedMemory с тем же именем и меньшим размером удерживается открытым дескриптором,
        _create_or_reset_shm и _resolve_non_conflicting_base_shm_name корректно выделяют буфер нужного
        размера без выброса FileExistsError.
        """
        base_name = "nmsim_gui_proj_shm_win183_test"
        task_16_name = get_task_projection_shm_name(base_name, 16)
        small_nbytes = 4096
        large_nbytes = 65536
        telem_nbytes = TelemetryRingBufferSHM.compute_byte_size(64, 64)

        locked_small_shm = None
        new_shm = None
        resolved_task_shm = None
        try:
            locked_small_shm = shared_memory.SharedMemory(name=task_16_name, create=True, size=small_nbytes)
            self.assertGreaterEqual(locked_small_shm.size, small_nbytes)

            # 1. Прямой вызов _create_or_reset_shm при открытом меньшем сегменте не падает с FileExistsError
            new_shm = OrchestratorSession._create_or_reset_shm(task_16_name, large_nbytes)
            self.assertGreaterEqual(new_shm.size, large_nbytes)

            # 2. _resolve_non_conflicting_base_shm_name обнаруживает конфликт на task_16 и переключает поколение префикса
            resolved_base = OrchestratorSession._resolve_non_conflicting_base_shm_name(
                base_name=base_name,
                num_tasks=17,
                proj_nbytes=large_nbytes,
                telem_nbytes=telem_nbytes,
            )
            self.assertNotEqual(resolved_base, base_name)
            resolved_task_16 = get_task_projection_shm_name(resolved_base, 16)
            resolved_task_shm = OrchestratorSession._create_or_reset_shm(resolved_task_16, large_nbytes)
            self.assertEqual(resolved_task_shm.name, resolved_task_16)
            self.assertGreaterEqual(resolved_task_shm.size, large_nbytes)
        finally:
            safe_close_shm(resolved_task_shm, unlink=True)
            safe_close_shm(new_shm, unlink=True)
            safe_close_shm(locked_small_shm, unlink=True)

    def test_detector_names_scene_resolution_and_ipc_retained_arrays(self) -> None:
        """
        Проверка:
        1. IPCReceiver при эмиссии projection_received и dose_volume_received передает независимые копии,
           поэтому сохранение ссылки на массив у получателя сигнала не вызывает BufferError при close()
           и не блокирует пересоздание сегмента большего размера под тем же именем.
        2. OrchestratorSession.detector_names отфильтровывает устаревшие имена детекторов, отсутствующие
           в текущей сцене, и находит кристаллы GammaCameraNode (включая слот 'crystal').
        """
        base_proj_shm = "test_retained_proj_shm"
        p_name = get_task_projection_shm_name(base_proj_shm, 0)
        t_name = get_task_telemetry_shm_name(base_proj_shm, 0)
        small_shape = (1, 16, 8, 8)
        large_shape = (2, 32, 16, 16)
        small_nbytes = int(np.prod(small_shape) * 4)
        large_nbytes = int(np.prod(large_shape) * 4)

        p_shm = None
        t_ring = None
        receiver = None
        recreated_shm = None
        retained_projections: List[np.ndarray] = []

        try:
            p_shm = shared_memory.SharedMemory(name=p_name, create=True, size=small_nbytes)
            arr_init = np.ndarray(small_shape, dtype=np.float32, buffer=p_shm.buf)
            arr_init.fill(5.0)
            del arr_init

            t_ring = TelemetryRingBufferSHM(shm_name=t_name, create=True, track_capacity=32, spectrum_capacity=32)
            receiver = IPCReceiver(
                shm_name=base_proj_shm,
                projection_shape=small_shape,
                focused_job_index=0,
                num_tasks=1,
                per_task_shm=True,
            )
            # Сохраняем полученный массив НАПРЯМУЮ без .copy() в слоте получателя
            receiver.projection_received.connect(retained_projections.append)
            receiver._poll_once()
            self.assertEqual(len(retained_projections), 1)

            # Закрываем receiver и исходный сегмент при живой ссылке в retained_projections
            receiver.close()
            receiver = None
            t_ring.close(unlink=True)
            t_ring = None
            safe_close_shm(p_shm, unlink=True)
            p_shm = None

            # Пересоздаем сегмент большего размера с ТЕМ ЖЕ ИМЕНЕМ p_name — не должно быть FileExistsError
            recreated_shm = shared_memory.SharedMemory(name=p_name, create=True, size=large_nbytes)
            self.assertEqual(recreated_shm.name, p_name)
            self.assertGreaterEqual(recreated_shm.size, large_nbytes)
        finally:
            retained_projections.clear()
            if receiver is not None:
                receiver.close()
            if t_ring is not None:
                t_ring.close(unlink=True)
            safe_close_shm(p_shm, unlink=True)
            safe_close_shm(recreated_shm, unlink=True)

        # Проверка согласования detector_names с узлами сцены
        world = CompositeNode(name="World")
        cam = GammaCameraNode(name="GammaCam_1")
        custom_crystal = Volume(
            geometry=Box(400.0, 400.0, 9.5),
            material=Material(name="NaI"),
            name="NaI_Scintillator_Head1",
        )
        cam.add_child(custom_crystal)
        cam.slots["crystal"] = "NaI_Scintillator_Head1"
        world.add_child(cam)

        scene_vm = SceneViewModel(root_core_node=world)
        dm_vm = DataManagerViewModel(handlers=[
            SensitiveVolumeHandlerViewModel(sensitive_volumes=["Stale_Removed_Detector"], enabled=True),
        ])
        session = OrchestratorSession(scene_vm=scene_vm, data_manager_vm=dm_vm)
        self.assertEqual(session.detector_names, ["NaI_Scintillator_Head1"])

        # Дефолтная сцена без детекторов (например, только WaterPhantom) обязана возвращать [],
        # даже если в DataManagerViewModel остались устаревшие имена ["Detector_1", "Detector_2"]
        empty_world = CompositeNode(name="WorldScene")
        phantom_vol = Volume(
            geometry=Box(200.0, 200.0, 200.0),
            material=Material(name="Water"),
            name="WaterPhantom",
        )
        empty_world.add_child(phantom_vol)
        empty_scene_vm = SceneViewModel(root_core_node=empty_world)
        stale_dm_vm = DataManagerViewModel(handlers=[
            SensitiveVolumeHandlerViewModel(sensitive_volumes=["Detector_1", "Detector_2"], enabled=True),
        ])
        empty_session = OrchestratorSession(scene_vm=empty_scene_vm, data_manager_vm=stale_dm_vm)
        self.assertEqual(empty_session.detector_names, [])

    def test_dose_shm_conflict_resolution_and_detector_count_switching(self) -> None:
        """
        Проверка:
        1. _resolve_non_conflicting_dose_shm_name обходит удерживаемый в ОС сегмент дозы меньшего размера
           (nmsim_dose_shm_task_16) без FileExistsError (WinError 183).
        2. Многократное изменение числа детекторов (0 -> 1 -> 2 -> 4 -> 1) и числа энергетических каналов
           при последовательных аллокациях SharedMemory не вызывает конфликтов размеров.
        """
        base_dose_name = "test_win183_dose_shm"
        task_16_dose_name = get_task_dose_shm_name(base_dose_name, 16, grid_name="DoseGrid_0", multi_grid=False)
        small_dose_bytes = 8 * 8 * 8 * 4
        large_dose_shape = (16, 16, 16)
        large_dose_bytes = 16 * 16 * 16 * 4
        dose_grids_info = [
            {
                "name": "DoseGrid_0",
                "shm_name": base_dose_name,
                "grid_shape": large_dose_shape,
                "dtype": "float32",
            }
        ]

        locked_dose_shm = None
        resolved_dose_shm = None
        created_shms = []
        try:
            locked_dose_shm = shared_memory.SharedMemory(
                name=task_16_dose_name,
                create=True,
                size=small_dose_bytes,
            )
            resolved_base_dose = OrchestratorSession._resolve_non_conflicting_dose_shm_name(
                base_dose_name,
                num_tasks=17,
                dose_grids_info=dose_grids_info,
            )
            self.assertNotEqual(resolved_base_dose, base_dose_name)
            self.assertTrue(resolved_base_dose.startswith(f"{base_dose_name}_g"))

            resolved_task_16_dose = get_task_dose_shm_name(
                resolved_base_dose, 16, grid_name="DoseGrid_0", multi_grid=False
            )
            resolved_dose_shm = OrchestratorSession._create_or_reset_shm(
                resolved_task_16_dose,
                large_dose_bytes,
            )
            self.assertGreaterEqual(resolved_dose_shm.size, large_dose_bytes)

            # Последовательные переключения конфигурации (0 -> 1 -> 2 -> 4 -> 1 детекторов и разные E_bins)
            base_proj = "test_multi_switch_proj_shm"
            telem_bytes = TelemetryRingBufferSHM.compute_byte_size(64)
            configs = [
                (1, 64, 16, 16),
                (1, 128, 16, 16),
                (2, 128, 16, 16),
                (4, 256, 16, 16),
                (1, 64, 16, 16),
            ]
            for shape in configs:
                req_bytes = int(np.prod(shape) * 4)
                active_base = OrchestratorSession._resolve_non_conflicting_base_shm_name(
                    base_proj,
                    num_tasks=2,
                    proj_nbytes=req_bytes,
                    telem_nbytes=telem_bytes,
                )
                for tid in range(2):
                    t_name = get_task_projection_shm_name(active_base, tid)
                    shm_seg = OrchestratorSession._create_or_reset_shm(t_name, req_bytes)
                    self.assertGreaterEqual(shm_seg.size, req_bytes)
                    created_shms.append(shm_seg)
        finally:
            for seg in created_shms:
                safe_close_shm(seg, unlink=True)
            safe_close_shm(resolved_dose_shm, unlink=True)
            safe_close_shm(locked_dose_shm, unlink=True)


if __name__ == '__main__':
    unittest.main()


