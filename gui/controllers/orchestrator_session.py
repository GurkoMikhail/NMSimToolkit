import logging
from multiprocessing import Manager, shared_memory
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import hepunits as units
from PySide6.QtCore import QObject, QThread, Signal

from core.config.exporter import SceneExporter
from core.config.models import (
    SimulationConfig,
    SimulationManagerConfig,
    DataManagerConfig,
)
from core.config.orchestrator import Orchestrator
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.stream_handlers import create_gui_stream_handler
from gui.viewmodels.data_handler_viewmodel import DataManagerViewModel
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    SpectProcedureViewModel,
    PetProcedureViewModel,
    CustomSweepProcedureViewModel,
)
from gui.viewmodels.scene_viewmodel import SceneViewModel
from gui.viewmodels.nodes.dose_grid_vm import DoseGridViewModel

_logger = logging.getLogger(__name__)


class _OrchestratorWorkerThread(QThread):
    """
    Фоновый рабочий поток Qt для выполнения пула задач Orchestrator.run().
    Предотвращает зависание графического интерфейса при параллельных вычислениях.
    """

    completed = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        orchestrator: Orchestrator,
        telemetry_queue: Any,
        focused_job_index: Optional[int],
        focused_shm_name: Optional[str],
        projection_shape: Tuple[int, int],
        global_pause_event: Optional[Any] = None,
        focused_pause_event: Optional[Any] = None,
        step_trigger_event: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.telemetry_queue = telemetry_queue
        self.focused_job_index = focused_job_index
        self.focused_shm_name = focused_shm_name
        self.projection_shape = projection_shape
        self.global_pause_event = global_pause_event
        self.focused_pause_event = focused_pause_event
        self.step_trigger_event = step_trigger_event

    def run(self) -> None:
        try:
            results = self.orchestrator.run(
                telemetry_queue=self.telemetry_queue,
                extra_handler_factory=create_gui_stream_handler,
                extra_handler_args=(
                    self.telemetry_queue,
                    self.focused_job_index,
                    self.focused_shm_name,
                    self.projection_shape,
                    self.global_pause_event,
                    self.focused_pause_event,
                    self.step_trigger_event,
                ),
            )
            self.completed.emit(results)
        except Exception as exc:
            _logger.exception(f"Сбой при выполнении пула задач оркестратора: {exc}")
            self.failed.emit(str(exc))


class OrchestratorSession(QObject):
    """
    Контроллер сессии параллельной оркестрации задач (Job Dispatcher / Session Controller).
    Управляет генерацией задач на основе процедур, распределением вычислений,
    сбором телеметрии и эксклюзивным стримингом для выбранного воркера.
    """

    session_started = Signal()
    session_paused = Signal()
    session_resumed = Signal()
    session_stopped = Signal()
    session_finished = Signal()
    session_error = Signal(str)

    jobs_generated = Signal(list)
    job_started = Signal(int)
    job_progress = Signal(int, float, int, float)  # task_id, progress, counts, cps
    job_finished = Signal(int)

    focused_worker_paused = Signal(int)
    focused_worker_resumed = Signal(int)

    tracks_received = Signal(dict)
    projection_received = Signal(object)
    projection_stack_updated = Signal(object, int, int, float)
    spectrum_received = Signal(object)
    stats_updated = Signal(int, float)
    dose_volume_received = Signal(object)

    def __init__(
        self,
        scene_vm: Optional[SceneViewModel] = None,
        procedure_vm: Optional[BaseProcedureViewModel] = None,
        data_manager_vm: Optional[DataManagerViewModel] = None,
        pool_size: int = 1,
        particles_number: int = 5000,
        stop_time: Optional[float] = None,
        min_energy: float = 1.0,
        shm_name: str = "nmsim_gui_proj_shm",
        projection_shape: Tuple[int, int] = (128, 128),
        focused_job_index: int = 0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.scene_vm = scene_vm
        self.procedure_vm = procedure_vm or SpectProcedureViewModel()
        if stop_time is not None:
            self.stop_time = float(stop_time)
        self.data_manager_vm = data_manager_vm or DataManagerViewModel()
        self.pool_size = max(1, int(pool_size))
        self.particles_number = int(particles_number)
        self.min_energy = float(min_energy)
        self.shm_name = shm_name
        self.projection_shape = projection_shape
        self.focused_job_index: int = int(focused_job_index)

        self._jobs: List[Dict[str, Any]] = []
        self._is_running: bool = False
        self._is_paused: bool = False
        self._is_focused_paused: bool = False

        self._worker_thread: Optional[_OrchestratorWorkerThread] = None
        self._ipc_receiver: Optional[IPCReceiver] = None
        self._mp_manager: Optional[Any] = None
        self._telemetry_queue: Optional[Any] = None
        self._shm: Optional[shared_memory.SharedMemory] = None

        self._global_pause_event: Optional[Any] = None
        self._focused_pause_event: Optional[Any] = None
        self._step_trigger_event: Optional[Any] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def is_focused_paused(self) -> bool:
        return self._is_focused_paused

    @property
    def is_focused_worker_paused(self) -> bool:
        return self._is_focused_paused

    @property
    def stop_time(self) -> float:
        """Время экспозиции/моделирования, определяемое активной процедурой (SSOT)."""
        if isinstance(self.procedure_vm, SpectProcedureViewModel):
            return float(self.procedure_vm.time_per_view)
        elif isinstance(self.procedure_vm, PetProcedureViewModel):
            return float(self.procedure_vm.time_per_frame)
        return 1.0

    @stop_time.setter
    def stop_time(self, new_stop_time: float) -> None:
        """Синхронизация времени с активной моделью процедуры."""
        validated_time = max(0.001, float(new_stop_time))
        if isinstance(self.procedure_vm, SpectProcedureViewModel):
            self.procedure_vm.time_per_view = validated_time
        elif isinstance(self.procedure_vm, PetProcedureViewModel):
            self.procedure_vm.time_per_frame = validated_time

    def _get_stop_time_cfg(self) -> Any:
        """Вычисляет конфигурацию времени остановки для экспортера ядра."""
        if isinstance(self.procedure_vm, CustomSweepProcedureViewModel):
            all_sweep_vars = set(self.procedure_vm.grid_variables.keys()) | set(self.procedure_vm.zipped_variables.keys())
            if "stop_time" in all_sweep_vars:
                return "${stop_time} s"
            return 1.0 * units.s
        elif isinstance(self.procedure_vm, SpectProcedureViewModel):
            return float(self.procedure_vm.time_per_view) * units.s
        elif isinstance(self.procedure_vm, PetProcedureViewModel):
            return float(self.procedure_vm.time_per_frame) * units.s
        return 1.0 * units.s

    @property
    def jobs(self) -> List[Dict[str, Any]]:
        return list(self._jobs)

    @property
    def dose_origin(self) -> Optional[Tuple[float, float, float]]:
        if self.scene_vm is not None:
            for node in self.scene_vm.all_nodes():
                if isinstance(node, DoseGridViewModel) and node.is_active:
                    return node.origin
        return None

    @property
    def dose_voxel_size(self) -> Optional[float]:
        """Размер вокселя первой активной сетки дозы в сцене или None, если сеток нет."""
        if self.scene_vm is not None:
            for node in self.scene_vm.all_nodes():
                if isinstance(node, DoseGridViewModel) and node.is_active:
                    return float(node.dose_voxel_size)
        return None

    @property
    def dose_transform_matrix(self) -> Optional[np.ndarray]:
        if self.scene_vm is not None:
            for node in self.scene_vm.all_nodes():
                if isinstance(node, DoseGridViewModel) and node.is_active:
                    return node.global_matrix
        return None

    def set_focused_job(self, index: int) -> None:
        """Переключение фокуса визуализации на задачу с указанным индексом."""
        self.focused_job_index = max(0, int(index))

    def clear_accumulation(self) -> None:
        """Сброс накопленных данных моделирования."""
        pass

    def step_once(self) -> None:
        """Выполнение одного расчетного шага сфокусированным воркером."""
        self.step_focused_worker()

    def generate_jobs(self) -> List[Dict[str, Any]]:
        """
        Генерация плоского списка задач симуляции на основе геометрии сцены и активной процедуры.
        """
        if self.scene_vm is None or self.scene_vm.root_vm is None:
            self._jobs = []
            self.jobs_generated.emit([])
            return []

        # Синхронизация геометрии процедуры со сценой перед экспортом
        self.procedure_vm.sync_with_scene(self.scene_vm)

        # Сборка базовой конфигурации
        root_core = self.scene_vm.root_vm.core_node
        start_time_cfg: Any = 0.0 * units.ns
        stop_time_cfg: Any = self._get_stop_time_cfg()
        if isinstance(self.procedure_vm, CustomSweepProcedureViewModel):
            all_sweep_vars = set(self.procedure_vm.grid_variables.keys()) | set(self.procedure_vm.zipped_variables.keys())
            if "start_time" in all_sweep_vars:
                start_time_cfg = "${start_time} s"

        sim_mgr_cfg = SimulationManagerConfig(
            particles_number=self.particles_number,
            start_time=start_time_cfg,
            stop_time=stop_time_cfg,
            min_energy=self.min_energy * units.keV,
        )
        data_mgr_cfg = self.data_manager_vm.to_config()

        dist_registry = self.scene_vm.distribution_registry if self.scene_vm else None
        sim_config = SceneExporter.export_to_config(
            root_node=root_core,
            simulation_manager_cfg=sim_mgr_cfg,
            data_manager_cfg=data_mgr_cfg,
            distribution_registry=dist_registry,
            pool_size=self.pool_size,
        )
        sim_config.protocol = self.procedure_vm.to_config()

        orchestrator = Orchestrator(sim_config)
        sweep_config = orchestrator.compile_protocol()
        self._jobs = orchestrator._generate_job_list(sweep_config)
        self.jobs_generated.emit(self._jobs)
        return self._jobs

    def start(self) -> None:
        """
        Запуск параллельной симуляции через пул процессов Orchestrator.
        """
        if self._is_running:
            return

        if not self._jobs:
            self.generate_jobs()

        if not self._jobs:
            self.session_error.emit("Список задач симуляции пуст.")
            return

        # 1. Синхронизация сцены
        if self.scene_vm is not None:
            self.procedure_vm.sync_with_scene(self.scene_vm)

        root_core = self.scene_vm.root_vm.core_node if self.scene_vm and self.scene_vm.root_vm else None
        if root_core is None:
            self.session_error.emit("Корневой узел сцены отсутствует.")
            return

        start_time_cfg: Any = 0.0 * units.ns
        stop_time_cfg: Any = self._get_stop_time_cfg()
        if isinstance(self.procedure_vm, CustomSweepProcedureViewModel):
            all_sweep_vars = set(self.procedure_vm.grid_variables.keys()) | set(self.procedure_vm.zipped_variables.keys())
            if "start_time" in all_sweep_vars:
                start_time_cfg = "${start_time} s"

        sim_mgr_cfg = SimulationManagerConfig(
            particles_number=self.particles_number,
            start_time=start_time_cfg,
            stop_time=stop_time_cfg,
            min_energy=self.min_energy * units.keV,
        )
        data_mgr_cfg = self.data_manager_vm.to_config()
        dist_registry = self.scene_vm.distribution_registry if self.scene_vm else None
        sim_config = SceneExporter.export_to_config(
            root_node=root_core,
            simulation_manager_cfg=sim_mgr_cfg,
            data_manager_cfg=data_mgr_cfg,
            distribution_registry=dist_registry,
            pool_size=self.pool_size,
        )
        sim_config.protocol = self.procedure_vm.to_config()

        # 2. Инициализация IPC и SharedMemory для сфокусированного воркера
        self._cleanup_ipc()

        nbytes = int(np.prod(self.projection_shape) * np.dtype(np.float32).itemsize)
        try:
            self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=nbytes)
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

        # Очистка буфера перед стартом
        np_buf = np.ndarray(self.projection_shape, dtype=np.float32, buffer=self._shm.buf)
        np_buf.fill(0.0)

        self._mp_manager = Manager()
        self._telemetry_queue = self._mp_manager.Queue()
        self._global_pause_event = self._mp_manager.Event()
        self._global_pause_event.set()
        self._focused_pause_event = self._mp_manager.Event()
        self._focused_pause_event.set()
        self._step_trigger_event = self._mp_manager.Event()
        self._step_trigger_event.clear()

        # 3. Запуск приемника телеметрии IPCReceiver
        self._ipc_receiver = IPCReceiver(
            track_queue=self._telemetry_queue,
            shm_name=self.shm_name,
            projection_shape=self.projection_shape,
            projection_dtype=np.float32,
            fps=30.0,
            parent=self,
        )
        self._ipc_receiver.tracks_received.connect(self.tracks_received)
        self._ipc_receiver.projection_received.connect(self.projection_received)
        self._ipc_receiver.spectrum_received.connect(self.spectrum_received)
        self._ipc_receiver.stats_updated.connect(self.stats_updated)
        self._ipc_receiver.dose_volume_received.connect(self.dose_volume_received)
        self._ipc_receiver.task_event.connect(self._on_task_event)
        self._ipc_receiver.start()

        # 4. Запуск Orchestrator в фоновом потоке
        orchestrator = Orchestrator(sim_config)
        self._worker_thread = _OrchestratorWorkerThread(
            orchestrator=orchestrator,
            telemetry_queue=self._telemetry_queue,
            focused_job_index=self.focused_job_index,
            focused_shm_name=self.shm_name,
            projection_shape=self.projection_shape,
            global_pause_event=self._global_pause_event,
            focused_pause_event=self._focused_pause_event,
            step_trigger_event=self._step_trigger_event,
            parent=self,
        )
        self._worker_thread.completed.connect(self._on_worker_completed)
        self._worker_thread.failed.connect(self._on_worker_failed)

        self._is_running = True
        self._is_paused = False
        self._is_focused_paused = False
        self._worker_thread.start()
        self.session_started.emit()

    def _on_task_event(self, event: Dict[str, Any]) -> None:
        """Обработка событий жизненного цикла задач из очереди телеметрии."""
        etype = event.get('type')
        task_id = int(event.get('task_id', 0))
        if etype == 'task_started':
            self.job_started.emit(task_id)
        elif etype == 'task_finished':
            self.job_finished.emit(task_id)
        elif etype == 'task_progress':
            prog = float(event.get('progress', 0.0))
            cnt = int(event.get('counts', 0))
            cps = float(event.get('cps', 0.0))
            self.job_progress.emit(task_id, prog, cnt, cps)
        elif etype == 'task_error':
            err = str(event.get('error', 'Неизвестная ошибка'))
            self.session_error.emit(f"Ошибка задачи #{task_id}: {err}")

    def _on_worker_completed(self, results: Any) -> None:
        self._is_running = False
        self._stop_ipc_receiver()
        self._cleanup_ipc()
        self.session_finished.emit()

    def _on_worker_failed(self, error_msg: str) -> None:
        self._is_running = False
        self._stop_ipc_receiver()
        self._cleanup_ipc()
        self.session_error.emit(error_msg)

    def pause(self) -> None:
        """Приостановка всего пула процессов."""
        self.pause_all()

    def pause_all(self) -> None:
        """Глобальная приостановка всех воркеров пула."""
        if self._is_running and not self._is_paused:
            self._is_paused = True
            if self._global_pause_event is not None:
                self._global_pause_event.clear()
            self.session_paused.emit()

    def resume(self) -> None:
        """Возобновление работы пула процессов."""
        self.resume_all()

    def resume_all(self) -> None:
        """Глобальное возобновление работы всех воркеров пула."""
        if self._is_running and self._is_paused:
            self._is_paused = False
            if self._global_pause_event is not None:
                self._global_pause_event.set()
            self.session_resumed.emit()

    def pause_focused_worker(self) -> None:
        """Локальная приостановка сфокусированного процесса для покадрового анализа."""
        if self._is_running and not self._is_focused_paused:
            self._is_focused_paused = True
            if self._focused_pause_event is not None:
                self._focused_pause_event.clear()
            self.focused_worker_paused.emit(self.focused_job_index)

    def resume_focused_worker(self) -> None:
        """Возобновление работы сфокусированного процесса."""
        if self._is_running and self._is_focused_paused:
            self._is_focused_paused = False
            if self._focused_pause_event is not None:
                self._focused_pause_event.set()
            self.focused_worker_resumed.emit(self.focused_job_index)

    def step_focused_worker(self) -> None:
        """Выполнение одного дискретного расчетного шага пачки частиц сфокусированным воркером."""
        if self._is_running and self._step_trigger_event is not None:
            self._step_trigger_event.set()

    def stop(self) -> None:
        """Остановка всех процессов оркестратора и освобождение ресурсов."""
        was_running = self._is_running
        self._is_running = False
        self._is_paused = False
        self._is_focused_paused = False

        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.terminate()
            self._worker_thread.wait(2000)
            self._worker_thread = None

        self._stop_ipc_receiver()
        self._cleanup_ipc()
        if was_running:
            self.session_stopped.emit()

    def _stop_ipc_receiver(self) -> None:
        if self._ipc_receiver is not None:
            self._ipc_receiver.stop()
            self._ipc_receiver.wait(1000)
            self._ipc_receiver.close()
            self._ipc_receiver = None

    def _cleanup_ipc(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except (FileNotFoundError, OSError):
                pass
            self._shm = None

        if self._mp_manager is not None:
            try:
                self._mp_manager.shutdown()
            except Exception:
                pass
            self._mp_manager = None
        self._telemetry_queue = None
        self._global_pause_event = None
        self._focused_pause_event = None
        self._step_trigger_event = None

    def close(self) -> None:
        self.stop()
