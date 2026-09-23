import logging
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Set, Tuple

import hepunits as units
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.other.typing_definitions import Float
from core.data.data_manager import DataManager
from core.data.dose_map_handler import DoseMapHandler
from core.data.data_handlers import (
    BaseDataHandler,
    DirectStreamHandler,
    SensitiveVolumeHandler,
    HistoryAssemblerHandler
)
from core.data.stream_handlers import GuiStreamDataHandler
from core.geometry.flattened_scene import FlattenedScene
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.volumes import Volume
from core.scene.dose_grid_node import DoseGridNode
from core.scene.nodes import CompositeNode, SpatialNode
from core.transport.simulation_managers import SimulationManager
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.simulation_runner import SimulationRunner
from gui.viewmodels.node_viewmodel import GammaCameraViewModel, VolumeViewModel

_logger = logging.getLogger(__name__)


class SimulationSession(QObject):
    """
    Фасад сессии моделирования (Mediator / Session Controller).
    Изолирует сборку вычислительного и телеметрического конвейера,
    управляет жизненным циклом фоновых процессов и потоков, детерминированно
    освобождает ресурсы межпроцессного взаимодействия (IPC).
    """

    session_started = Signal()
    session_paused = Signal()
    session_resumed = Signal()
    session_stopped = Signal()
    session_finished = Signal()
    session_error = Signal(str)

    tracks_received = Signal(dict)
    projection_received = Signal(object)
    projection_stack_updated = Signal(object, int, int, float)  # stack_3d, current_view, total_views, angle_deg
    spectrum_received = Signal(object)
    stats_updated = Signal(int, float)
    dose_volume_received = Signal(object)

    def __init__(
        self,
        scene_root: Any,
        shm_name: str = "nmsim_gui_proj_shm",
        projection_shape: Tuple[int, int] = (128, 128),
        particles_number: int = 5000,
        stop_time: float = 1.0,
        views_number: int = 1,
        angular_range: float = 360.0,
        fov_size: float = 400.0,
        sensitive_volume_ids: Optional[Set[int]] = None,
        max_tracks_per_batch: int = 1000,
        show_escaped_tracks: bool = False,
        buffer_capacity: Optional[int] = None,
        camera_vm: Optional[Any] = None,
        fps: float = 30.0,
        h5_filename: str = "gui_simulation.h5",
        dose_accumulation_enabled: bool = True,
        dose_grid_shape: Optional[Tuple[int, int, int]] = None,
        dose_voxel_size: float = 5.0,
        dose_origin: Optional[Tuple[float, float, float]] = None,
        dose_shm_name: Optional[str] = None,
        save_history: bool = True,
        save_initial_states: bool = True,
        save_sensitive_interactions: bool = True,
        save_direct_stream: bool = False,
        min_energy: float = 1.0,
        seed: Optional[int] = None,
        swmr: bool = True,
        gamma_cameras_number: int = 1,
        head_angle_offsets: Optional[List[float]] = None,
        start_angle: float = 0.0,
        orbit_radius: float = 250.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.scene_root = scene_root
        self.shm_name = shm_name
        self.projection_shape = projection_shape
        self.particles_number = int(particles_number)
        self.stop_time = float(stop_time)
        self.views_number = max(1, int(views_number))
        self.angular_range = float(angular_range)
        self.start_angle = float(start_angle)
        self.orbit_radius = float(orbit_radius)
        self.fov_size = fov_size
        self.sensitive_volume_ids = sensitive_volume_ids
        self.max_tracks_per_batch = int(max_tracks_per_batch)
        self.show_escaped_tracks = bool(show_escaped_tracks)
        self.buffer_capacity = buffer_capacity
        self.camera_vm = camera_vm
        self.fps = fps
        self.h5_filename = h5_filename
        self.save_history = save_history
        self.save_initial_states = save_initial_states
        self.save_sensitive_interactions = save_sensitive_interactions
        self.save_direct_stream = save_direct_stream
        self.min_energy = min_energy
        self.seed = seed
        self.swmr = swmr
        self.gamma_cameras_number = max(1, int(gamma_cameras_number))
        self.head_angle_offsets = head_angle_offsets

        self.current_view_index: int = 0
        self.projections_stack: np.ndarray = np.zeros(
            (self.views_number, self.projection_shape[0], self.projection_shape[1]),
            dtype=np.float32
        )

        self.dose_accumulation_enabled = dose_accumulation_enabled
        self.dose_grid_shape = dose_grid_shape
        self.dose_voxel_size = dose_voxel_size
        self.dose_origin = dose_origin
        self.dose_shm_name = dose_shm_name or f"nmsim_dose_shm_{abs(id(self))}"
        self.dose_transform_matrix: Optional[np.ndarray] = None

        self.track_queue: Optional[Queue] = None
        self.stream_handler: Optional[GuiStreamDataHandler] = None
        self.dose_handler: Optional[Any] = None
        self.manager: Optional[SimulationManager] = None
        self.data_manager: Optional[DataManager] = None
        self.ipc_receiver: Optional[IPCReceiver] = None
        self.simulation_runner: Optional[SimulationRunner] = None

        self._is_closed: bool = False
        self._setup_pipeline()

    def _find_dose_grid_nodes(self) -> List[Any]:
        """
        Рекурсивный поиск всех активных узлов DoseGridNode в иерархии сцены.
        """
        if self.scene_root is None:
            return []

        grids: List[Any] = []
        visited = set()

        def _traverse(node: Any) -> None:
            if node is None or id(node) in visited:
                return
            visited.add(id(node))
            if isinstance(node, DoseGridNode) and node.is_active:
                grids.append(node)
            if isinstance(node, CompositeNode):
                for c in node.childs:
                    _traverse(c)

        _traverse(self.scene_root)
        return grids

    def _find_detector_info(self) -> Tuple[Optional[Set[int]], Optional[Any]]:
        """
        Находит в сцене узел детектора и возвращает (множество_volume_ids, detector_volume).
        """
        if self.scene_root is None or not isinstance(self.scene_root, SpatialNode):
            return None, None

        flat_list = FlattenedScene(self.scene_root).flat_list

        # 1. Если sensitive_volume_ids уже задан явно, сопоставляем с flat_list
        if self.sensitive_volume_ids:
            for vid in self.sensitive_volume_ids:
                if 0 <= vid < len(flat_list):
                    return set(self.sensitive_volume_ids), flat_list[vid][0]

        # 2. Поиск объемов, помеченных в GUI через VolumeViewModel как детекторы
        gui_sens_ids = set()
        gui_target_vol = None
        sensitive_vols = VolumeViewModel.get_sensitive_volumes()
        for idx, (vol, _, _) in enumerate(flat_list):
            if isinstance(vol, Volume) and vol in sensitive_vols:
                gui_sens_ids.add(idx)
                if gui_target_vol is None:
                    gui_target_vol = vol
        if gui_sens_ids:
            return gui_sens_ids, gui_target_vol

        # 3. Поиск детектора через GammaCamera.detector в иерархии сцены
        target_vol = None
        def find_camera_detector(node):
            if isinstance(node, GammaCamera):
                return node.detector
            if isinstance(node, CompositeNode):
                for child in node.childs:
                    found = find_camera_detector(child)
                    if found is not None:
                        return found
            return None

        target_vol = find_camera_detector(self.scene_root)

        # 4. Если GammaCamera не найдена, ищем объем со словом 'detector' в имени
        if target_vol is None:
            for vol, _, _ in flat_list:
                if 'detector' in str(vol.name).lower():
                    target_vol = vol
                    break

        if target_vol is not None:
            sensitive_ids = set()
            target_vols = {target_vol}
            def collect_children(v):
                if isinstance(v, CompositeNode):
                    for ch in v.childs:
                        if isinstance(ch, Volume):
                            target_vols.add(ch)
                            collect_children(ch)
            collect_children(target_vol)

            for idx, (vol, _, parent_idx) in enumerate(flat_list):
                if vol in target_vols:
                    sensitive_ids.add(idx)
                elif parent_idx != -1 and parent_idx in sensitive_ids:
                    sensitive_ids.add(idx)
            if sensitive_ids:
                return sensitive_ids, target_vol

        return None, None

    def _create_simulation_manager(self) -> SimulationManager:
        """
        Фабричный метод создания SimulationManager с заданными параметрами.
        """
        gui_buffer_capacity = self.buffer_capacity if self.buffer_capacity is not None else max(int(self.particles_number), 1000)
        return SimulationManager(
            scene=self.scene_root,
            particles_number=self.particles_number,
            stop_time=self.stop_time,
            buffer_capacity=gui_buffer_capacity,
            min_energy=Float(self.min_energy * units.keV),
            seed=self.seed,
            queue=Queue(maxsize=64),
        )

    def _build_data_handlers(self) -> List[BaseDataHandler]:
        """
        Формирует список всех активных обработчиков данных для DataManager.
        Гарантирует, что все чувствительные объемы принадлежат строго текущей сцене.
        """
        handlers: List[BaseDataHandler] = [self.stream_handler]
        if self.dose_handler is not None:
            handlers.append(self.dose_handler)

        sensitive_vols: List[Volume] = []
        current_root = self.scene_root.root if isinstance(self.scene_root, SpatialNode) else self.scene_root

        # 1. Сбор детекторов всех гамма-камер в текущей сцене
        def _collect_camera_detectors(node: Any) -> List[Volume]:
            dets: List[Volume] = []
            if isinstance(node, GammaCamera) and isinstance(node.detector, Volume):
                dets.append(node.detector)
            if isinstance(node, CompositeNode):
                for ch in node.childs:
                    dets.extend(_collect_camera_detectors(ch))
            return dets

        if isinstance(self.scene_root, SpatialNode):
            for cam_det in _collect_camera_detectors(self.scene_root):
                if cam_det not in sensitive_vols:
                    sensitive_vols.append(cam_det)

        # 2. Объем, найденный _find_detector_info (если не гамма-камера)
        _, detector_vol = self._find_detector_info()
        if isinstance(detector_vol, Volume) and detector_vol.root is current_root and detector_vol not in sensitive_vols:
            sensitive_vols.append(detector_vol)

        # 3. Объемы, явно помеченные в GUI через VolumeViewModel (строго в текущей сцене)
        for v in VolumeViewModel.get_sensitive_volumes():
            if isinstance(v, Volume) and v.root is current_root and v not in sensitive_vols:
                sensitive_vols.append(v)

        if self.save_history and sensitive_vols:
            self.history_handler = HistoryAssemblerHandler(
                sensitive_volumes=sensitive_vols,
                save_initial_states=self.save_initial_states
            )
            handlers.append(self.history_handler)
        elif self.save_sensitive_interactions and sensitive_vols:
            self.sensitive_handler = SensitiveVolumeHandler(sensitive_volumes=sensitive_vols)
            handlers.append(self.sensitive_handler)

        if self.save_direct_stream:
            self.direct_stream_h5_handler = DirectStreamHandler()
            handlers.append(self.direct_stream_h5_handler)

        return handlers

    def _setup_pipeline(self) -> None:
        """
        Инициализация конвейера:
        SimulationManager -> Queue -> GuiStreamDataHandler -> (SharedMemory + track_queue)
        -> IPCReceiver -> SimulationSession Signals
        """
        # 1. Создание очереди треков и обработчика телеметрии ядра
        self.track_queue = Queue()

        auto_sensitive_ids, detector_vol = self._find_detector_info()
        sensitive_ids = self.sensitive_volume_ids if self.sensitive_volume_ids is not None else auto_sensitive_ids

        self.stream_handler = GuiStreamDataHandler(
            track_queue=self.track_queue,
            shm_name=self.shm_name,
            projection_shape=self.projection_shape,
            create_shm=True,
            sensitive_volume_ids=sensitive_ids,
            detector_volume=detector_vol,
            fov_size=self.fov_size,
            max_tracks_per_batch=self.max_tracks_per_batch,
            show_escaped_tracks=self.show_escaped_tracks,
        )

        # 2. Менеджер моделирования с конфигурируемой емкостью буфера
        self.manager = self._create_simulation_manager()

        # Инициализация обработчика 3D-накопления дозы строго по узлам DoseGridNode в сцене
        if self.dose_accumulation_enabled:
            dose_grid_nodes = self._find_dose_grid_nodes()
            if dose_grid_nodes:
                self.dose_handler = DoseMapHandler(
                    shm_name=self.dose_shm_name,
                    create_shm=True,
                    grid_nodes=dose_grid_nodes,
                )
                first_entry = self.dose_handler.entries[0]
                self.dose_grid_shape = first_entry.grid_shape
                self.dose_voxel_size = first_entry.voxel_size[0]
                self.dose_origin = first_entry.origin
                self.dose_shm_name = first_entry.shm_name
                self.dose_transform_matrix = first_entry.node.global_matrix.copy() if first_entry.node is not None else None
            else:
                self.dose_handler = None
                self.dose_transform_matrix = None
        else:
            self.dose_handler = None
            self.dose_transform_matrix = None

        handlers = self._build_data_handlers()

        self.data_manager = DataManager(
            filename=self.h5_filename,
            handlers=handlers,
            queue=self.manager.queue,
            swmr=self.swmr,
        )

        # 3. Контроллер фонового приема IPC-данных
        self.ipc_receiver = IPCReceiver(
            track_queue=self.track_queue,
            shm_name=self.shm_name,
            projection_shape=self.projection_shape,
            fps=self.fps,
            dose_shm_name=self.dose_shm_name if self.dose_accumulation_enabled else None,
            dose_grid_shape=self.dose_grid_shape,
            dose_fps=2.0,
            parent=self,
        )
        self.ipc_receiver.tracks_received.connect(self.tracks_received)
        self.ipc_receiver.projection_received.connect(self._on_projection_received)
        self.ipc_receiver.spectrum_received.connect(self.spectrum_received)
        self.ipc_receiver.dose_volume_received.connect(self.dose_volume_received)
        self.ipc_receiver.stats_updated.connect(self.stats_updated)

        # 4. Контроллер потока моделирования SimulationRunner
        self.simulation_runner = SimulationRunner(manager=self.manager, parent=self)
        self.simulation_runner.simulation_started.connect(self.session_started)
        self.simulation_runner.simulation_paused.connect(self.session_paused)
        self.simulation_runner.simulation_resumed.connect(self.session_resumed)
        self.simulation_runner.simulation_stopped.connect(self._on_runner_stopped)
        self.simulation_runner.simulation_finished.connect(self._on_runner_finished)
        self.simulation_runner.simulation_error.connect(self.session_error)

    @property
    def is_running(self) -> bool:
        return self.simulation_runner is not None and self.simulation_runner.is_running

    @property
    def runner(self) -> Optional[SimulationRunner]:
        return self.simulation_runner

    @property
    def receiver(self) -> Optional[IPCReceiver]:
        return self.ipc_receiver

    def start(self) -> None:
        """
        Запуск телеметрии и процесса симуляции.
        """
        if self._is_closed:
            raise RuntimeError("Попытка запуска закрытой сессии SimulationSession")
        if self.data_manager is not None and not self.data_manager.is_alive():
            self.data_manager.start()
        if self.ipc_receiver is not None:
            self.ipc_receiver.start()
        if self.simulation_runner is not None:
            self.simulation_runner.start()

    def pause(self) -> None:
        """
        Приостановка процесса моделирования.
        """
        if self.simulation_runner is not None:
            self.simulation_runner.pause()

    def resume(self) -> None:
        """
        Возобновление процесса моделирования.
        """
        if self.simulation_runner is not None:
            self.simulation_runner.resume()

    def stop(self) -> None:
        """
        Остановка моделирования и телеметрии.
        """
        if self.simulation_runner is not None:
            self.simulation_runner.stop()
            self.simulation_runner.wait(1000)
        if self.data_manager is not None and self.data_manager.is_alive():
            self.data_manager.join(timeout=1.0)
        if self.ipc_receiver is not None:
            self.ipc_receiver.stop(timeout_ms=500)

    def step_once(self) -> None:
        """
        Пошаговое продвижение симуляции на одну итерацию.
        """
        if self.simulation_runner is not None:
            self.simulation_runner.step_once()

    def _on_runner_stopped(self) -> None:
        if self.simulation_runner is not None:
            self.simulation_runner.wait(1000)
        if self.data_manager is not None and self.data_manager.is_alive():
            self.data_manager.join(timeout=1.0)
        if self.ipc_receiver is not None:
            self.ipc_receiver.stop(timeout_ms=500)
        self.session_stopped.emit()

    def _on_projection_received(self, proj_data: np.ndarray) -> None:
        if self.views_number > 0 and 0 <= self.current_view_index < self.views_number:
            self.projections_stack[self.current_view_index] = np.array(proj_data, copy=True)
            angle = (self.current_view_index / self.views_number) * self.angular_range
            self.projection_stack_updated.emit(self.projections_stack, self.current_view_index, self.views_number, angle)
        self.projection_received.emit(proj_data)

    def _rotate_camera_to_view(self, view_index: int) -> None:
        """
        Поворачивает гамма-камеры на заданный угол проекции для ОФЭКТ сканирования (поддержка 1, 2, 4 или N головок).
        """
        base_angle = (view_index / self.views_number) * self.angular_range + self.start_angle

        # Сбор всех гамма-камер в сцене
        cameras: List[GammaCamera] = []
        def _collect_cams(node: Any) -> None:
            if isinstance(node, GammaCamera):
                cameras.append(node)
            if isinstance(node, CompositeNode):
                for ch in node.childs:
                    _collect_cams(ch)
        _collect_cams(self.scene_root)

        gc = max(1, len(cameras), self.gamma_cameras_number)
        if self.head_angle_offsets is not None and len(self.head_angle_offsets) == len(cameras):
            offsets = self.head_angle_offsets
        else:
            step_off = 360.0 / gc if gc > 0 else 0.0
            offsets = [step_off * i for i in range(len(cameras))]

        if isinstance(self.camera_vm, GammaCameraViewModel) and len(cameras) <= 1:
            rad = float(self.camera_vm.orbit_radius)
            z = float(self.camera_vm.orbit_z)
            self.camera_vm.set_orbit_position(rad, base_angle, z)
        else:
            for cam, off in zip(cameras, offsets):
                ang = (base_angle + off) % 360.0
                half_th = float(cam.size[2]) / 2.0 if len(cam.size) >= 3 and cam.size[2] > 0 else 0.0
                rad = self.orbit_radius
                if isinstance(self.camera_vm, GammaCameraViewModel) and self.camera_vm.core_node is cam:
                    rad = float(self.camera_vm.orbit_radius)
                cam.local_matrix = GammaCameraViewModel.compute_orbit_matrix(rad, ang, 0.0, half_thickness=half_th)
                cam.invalidate_matrix_cache()

    def preview_view(self, view_index: int) -> float:
        """
        Предварительная установка ориентации камер на заданный ракурс до запуска симуляции.
        Возвращает базовый угол поворота в градусах.
        """
        base_angle = (view_index / self.views_number) * self.angular_range + self.start_angle
        self._rotate_camera_to_view(view_index)
        return float(base_angle % 360.0)

    def _on_runner_finished(self) -> None:
        # Если запущено многоракурсное сканирование (SPECT) и есть еще проекции
        if self.views_number > 1 and self.current_view_index + 1 < self.views_number and not self._is_closed:
            # Сохраняем текущий срез
            if self.stream_handler is not None and self.stream_handler._projection_array is not None:
                self.projections_stack[self.current_view_index] = np.array(self.stream_handler._projection_array, copy=True)

            self.current_view_index += 1
            self._rotate_camera_to_view(self.current_view_index)

            # Очищаем проекцию детектора для нового ракурса
            if self.stream_handler is not None:
                self.stream_handler.clear()

            # Запускаем расчет следующего ракурса
            self.manager = self._create_simulation_manager()
            handlers = self._build_data_handlers()

            self.data_manager = DataManager(
                filename=self.h5_filename,
                handlers=handlers,
                queue=self.manager.queue,
                swmr=self.swmr,
            )
            self.simulation_runner = SimulationRunner(manager=self.manager, parent=self)
            self.simulation_runner.simulation_started.connect(self.session_started)
            self.simulation_runner.simulation_paused.connect(self.session_paused)
            self.simulation_runner.simulation_resumed.connect(self.session_resumed)
            self.simulation_runner.simulation_stopped.connect(self._on_runner_stopped)
            self.simulation_runner.simulation_finished.connect(self._on_runner_finished)
            self.simulation_runner.simulation_error.connect(self.session_error)

            self.simulation_runner.start()
            return

        if self.simulation_runner is not None:
            self.simulation_runner.wait(1000)
        if self.data_manager is not None and self.data_manager.is_alive():
            self.data_manager.join(timeout=1.0)
        if self.ipc_receiver is not None:
            self.ipc_receiver.stop(timeout_ms=500)

        # Актуализация локальных массивов dose_data узлов DoseGridNode
        if self.dose_handler is not None:
            for entry in self.dose_handler.entries:
                if entry.node is not None and entry._dose_grid is not None:
                    entry.node.dose_data = entry._dose_grid.copy()

        self.session_finished.emit()

    def clear_accumulation(self) -> None:
        """
        Сброс всех накопленных данных моделирования (проекции детектора, спектра и 3D-карты дозы).
        """
        if self.stream_handler is not None:
            self.stream_handler.clear()
        if self.dose_handler is not None:
            self.dose_handler.clear()
            for entry in self.dose_handler.entries:
                if entry.node is not None:
                    entry.node.clear()
        if self.ipc_receiver is not None:
            self.ipc_receiver.clear_accumulation()

    def get_dose_snapshot(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Возвращает снимок 3D-массива дозы для указанной или единственной сетки.
        """
        if self.dose_handler is not None:
            return self.dose_handler.get_dose_snapshot(name=name)
        return None

    def get_dose_snapshots(self) -> Dict[str, np.ndarray]:
        """
        Возвращает снимки всех активных сеток дозы {имя: ndarray}.
        """
        if self.dose_handler is not None:
            return self.dose_handler.get_dose_snapshots()
        return {}

    def close(self) -> None:
        """
        Детерминированное освобождение всех ресурсов сессии:
        остановка потоков, закрытие дескрипторов очереди и shared memory,
        остановка HDF5 DataManager.
        """
        if self._is_closed:
            return
        self._is_closed = True

        if self.simulation_runner is not None:
            try:
                self.simulation_runner.stop()
                self.simulation_runner.wait(500)
            except Exception as e:
                _logger.debug(f"Ошибка остановки simulation_runner: {e}")
            self.simulation_runner = None

        if self.ipc_receiver is not None:
            try:
                self.ipc_receiver.close()
            except Exception as e:
                _logger.debug(f"Ошибка закрытия ipc_receiver: {e}")
            self.ipc_receiver = None

        if self.stream_handler is not None:
            try:
                self.stream_handler.close()
            except Exception as e:
                _logger.debug(f"Ошибка закрытия stream_handler: {e}")
            self.stream_handler = None

        if self.dose_handler is not None:
            try:
                self.dose_handler.close()
            except Exception as e:
                _logger.debug(f"Ошибка закрытия dose_handler: {e}")
            self.dose_handler = None

        if self.data_manager is not None:
            try:
                self.data_manager.stop()
            except Exception as e:
                _logger.debug(f"Ошибка остановки data_manager: {e}")
            self.data_manager = None

        if self.track_queue is not None:
            try:
                self.track_queue.close()
                self.track_queue.cancel_join_thread()
            except Exception as e:
                _logger.debug(f"Ошибка закрытия track_queue: {e}")
            self.track_queue = None
