import gc
import logging
from multiprocessing import Manager, shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import hepunits as units
from PySide6.QtCore import QObject, QThread, Signal

from core.config.exporter import SceneExporter
from core.config.template_binder import ConfigTemplateBinder
from core.config.models import (
    SimulationConfig,
    SimulationManagerConfig,
    DataManagerConfig,
    DoseMapHandlerConfig,
)
from core.config.orchestrator import Orchestrator
from core.config.sweep_compiler import SweepCompiler
from core.scene.nodes import CompositeNode, SpatialNode
from core.source.sources import Source
from gui.controllers.ipc_receiver import IPCReceiver
from gui.controllers.stream_handlers import (
    ControlBufferSHM,
    DEFAULT_CTRL_SHM_NAME,
    TelemetryRingBufferSHM,
    _find_all_detector_nodes,
    _find_scene_nodes_by_names,
    create_gui_stream_handler,
    create_or_reuse_shm,
    get_task_dose_shm_name,
    get_task_projection_shm_name,
    get_task_telemetry_shm_name,
    is_shm_name_conflicting,
    safe_close_shm,
)
from gui.viewmodels.data_handler_viewmodel import (
    DataManagerViewModel,
    DoseMapHandlerViewModel,
    SensitiveVolumeHandlerViewModel,
)
from gui.viewmodels.procedure_viewmodel import (
    BaseProcedureViewModel,
    StepAndShootProcedureViewModel,
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
        projection_shape: Union[Tuple[int, int], Tuple[int, int, int, int], Tuple[int, ...]],
        global_pause_event: Optional[Any] = None,
        focused_pause_event: Optional[Any] = None,
        step_trigger_event: Optional[Any] = None,
        per_task_shm: bool = True,
        ctrl_shm_name: Optional[str] = DEFAULT_CTRL_SHM_NAME,
        max_energy: float = 0.3,
        energy_bins: int = 256,
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
        self.per_task_shm = bool(per_task_shm)
        self.ctrl_shm_name = ctrl_shm_name
        self.max_energy = float(max_energy)
        self.energy_bins = int(energy_bins)

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
                    {
                        "per_task_shm": self.per_task_shm,
                        "ctrl_shm_name": self.ctrl_shm_name,
                        "max_energy": self.max_energy,
                        "energy_bins": self.energy_bins,
                    },
                ),
            )
            self.completed.emit(results)
        except (OSError, RuntimeError, ValueError, KeyError, BufferError, TypeError) as exc:
            _logger.exception(f"Сбой при выполнении пула задач оркестратора: {exc}")
            self.failed.emit(str(exc))


class OrchestratorSession(QObject):
    """
    Контроллер сессии параллельной оркестрации задач (Job Dispatcher / Session Controller).
    Управляет генерацией задач на основе процедур, распределением вычислений,
    пер-тасковыми буферами SharedMemory и мгновенным переключением фокуса между задачами.
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
    detector_metadata_updated = Signal(list, float)

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
        projection_shape: Union[Tuple[int, int], Tuple[int, int, int, int], Tuple[int, ...]] = (128, 128),
        focused_job_index: int = 0,
        max_track_points: int = 10000,
        energy_bins: int = 256,
        max_energy: Optional[float] = None,
        auto_max_energy: Optional[bool] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.scene_vm = scene_vm
        self.procedure_vm = procedure_vm or StepAndShootProcedureViewModel()
        if stop_time is not None:
            self.stop_time = float(stop_time)
        self.data_manager_vm = data_manager_vm or DataManagerViewModel()
        self.pool_size = max(1, int(pool_size))
        self.particles_number = int(particles_number)
        self.min_energy = float(min_energy)
        self.shm_name = shm_name
        self.projection_shape = projection_shape
        self.energy_bins: int = max(1, int(energy_bins))
        if auto_max_energy is not None:
            self.auto_max_energy: bool = bool(auto_max_energy)
        else:
            self.auto_max_energy = max_energy is None
        raw_max_e = float(max_energy) if (max_energy is not None and float(max_energy) > 0.0) else 0.3
        self._max_energy: float = raw_max_e
        if self.auto_max_energy and self.scene_vm is not None:
            self._max_energy = self.compute_auto_max_energy()
        self._active_cube_shape: Tuple[int, ...] = tuple(int(dim) for dim in projection_shape)
        self.focused_job_index: int = int(focused_job_index)
        self.max_track_points: int = int(max_track_points)
        self.num_tasks: int = 0

        self._jobs: List[Dict[str, Any]] = []
        self._is_running: bool = False
        self._is_paused: bool = False
        self._is_focused_paused: bool = False
        self._is_generating_jobs: bool = False

        self._worker_thread: Optional[_OrchestratorWorkerThread] = None
        self._ipc_receiver: Optional[IPCReceiver] = None
        self._mp_manager: Optional[Any] = None
        self._telemetry_queue: Optional[Any] = None
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._proj_shms: List[shared_memory.SharedMemory] = []
        self._telem_rings: List[TelemetryRingBufferSHM] = []
        self._ctrl_shm: Optional[ControlBufferSHM] = None
        self._dose_shms: List[shared_memory.SharedMemory] = []
        self._active_dose_grids_info: List[Dict[str, Any]] = []
        self._base_dose_shm_name: str = "nmsim_dose_shm"
        self._completed_task_snapshots: Dict[int, Dict[str, Any]] = {}

        self._global_pause_event: Optional[Any] = None
        self._focused_pause_event: Optional[Any] = None
        self._step_trigger_event: Optional[Any] = None

    @property
    def max_energy(self) -> float:
        """Верхняя граница шкалы спектра в базовых единицах hepunits (МэВ)."""
        return self._max_energy

    @max_energy.setter
    def max_energy(self, value: float) -> None:
        val = float(value)
        if val > 0.0:
            self._max_energy = val
            self.auto_max_energy = False

    def compute_auto_max_energy(self, margin_factor: float = 1.2) -> float:
        """
        Вычисляет верхнюю границу шкалы спектра (в МэВ) по реальному максимуму энергии
        всех источников текущей сцены с запасом margin_factor (~20% сверх максимальной линии).
        При отсутствии источников возвращает 0.3 МэВ (300 кэВ).
        """
        default_max = float(300.0 * units.keV)
        if self.scene_vm is None or self.scene_vm.root_vm is None:
            return default_max
        root_core = self.scene_vm.root_vm.core_node
        if root_core is None:
            return default_max

        all_energies: List[float] = []

        stack: List[SpatialNode] = [root_core]
        while stack:
            node = stack.pop()
            if isinstance(node, Source):
                energy_table = node.energy
                if (
                    isinstance(energy_table, np.ndarray)
                    and energy_table.size > 0
                    and energy_table.dtype.names is not None
                    and "energy" in energy_table.dtype.names
                ):
                    energies = np.asarray(energy_table["energy"], dtype=float).ravel()
                    valid_mask = energies > 0.0
                    if np.any(valid_mask):
                        valid_e = energies[valid_mask]
                        if "probability" in energy_table.dtype.names:
                            probs = np.asarray(energy_table["probability"], dtype=float).ravel()[valid_mask]
                            pos_prob_mask = probs > 0.0
                            if np.any(pos_prob_mask):
                                valid_e = valid_e[pos_prob_mask]
                        all_energies.extend(float(val) for val in valid_e)
            if isinstance(node, CompositeNode):
                stack.extend(node.childs)

        if not all_energies:
            return default_max

        peak_energy = max(all_energies)
        padded_energy = max(float(10.0 * units.keV), peak_energy * float(margin_factor))
        if padded_energy <= 120.0 * units.keV:
            step = float(5.0 * units.keV)
        elif padded_energy <= 500.0 * units.keV:
            step = float(10.0 * units.keV)
        else:
            step = float(25.0 * units.keV)
        return float(np.ceil(round(padded_energy / step, 9)) * step)

    def resolve_effective_max_energy(self) -> float:
        """
        Актуализирует и возвращает верхнюю границу спектра (в МэВ):
        в режиме auto_max_energy пересчитывает по источникам сцены, иначе возвращает текущее значение.
        """
        if self.auto_max_energy:
            self._max_energy = self.compute_auto_max_energy()
        return self._max_energy

    def set_spectrum_parameters(
        self,
        max_energy: Optional[float] = None,
        energy_bins: Optional[int] = None,
        auto_max_energy: Optional[bool] = None,
    ) -> float:
        """
        Устанавливает параметры энергетического спектра (max_energy в МэВ, число бинов energy_bins
        и флаг автоподбора auto_max_energy) и возвращает актуальный max_energy в МэВ.
        """
        if auto_max_energy is not None:
            self.auto_max_energy = bool(auto_max_energy)
        if energy_bins is not None:
            self.energy_bins = max(1, int(energy_bins))
        if self.auto_max_energy:
            self._max_energy = self.compute_auto_max_energy()
        elif max_energy is not None and float(max_energy) > 0.0:
            self._max_energy = float(max_energy)
        return self._max_energy

    @property
    def detector_names(self) -> List[str]:
        """
        Упорядоченный список имен чувствительных объемов (детекторов) из активных обработчиков данных,
        согласованный с узлами текущего дерева сцены, с резервным поиском детекторов/кристаллов.
        """
        configured_names: List[str] = []
        if self.data_manager_vm is not None:
            for handler_vm in self.data_manager_vm.handlers:
                if isinstance(handler_vm, SensitiveVolumeHandlerViewModel) and handler_vm.enabled:
                    for vol_name in handler_vm.sensitive_volumes:
                        clean_name = str(vol_name).strip()
                        if clean_name and clean_name not in configured_names:
                            configured_names.append(clean_name)

        if self.scene_vm is not None and self.scene_vm.root_vm is not None:
            root_core = self.scene_vm.root_vm.core_node
            if root_core is not None:
                if configured_names:
                    matched_nodes = _find_scene_nodes_by_names(root_core, configured_names)
                    if matched_nodes:
                        matched_names: List[str] = []
                        for node in matched_nodes:
                            if node.name and node.name not in matched_names:
                                matched_names.append(node.name)
                        return matched_names
                scene_detectors = _find_all_detector_nodes(
                    root_core,
                    slots_registry=self.scene_vm.slots_registry,
                )
                scene_names: List[str] = []
                for det_node in scene_detectors:
                    if det_node.name and det_node.name not in scene_names:
                        scene_names.append(det_node.name)
                return scene_names

        return configured_names

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def is_focused_worker_paused(self) -> bool:
        return self._is_focused_paused

    @property
    def start_time(self) -> float:
        """Время начала моделирования (секунды), определяемое активной процедурой (SSOT)."""
        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            return float(self.procedure_vm.initial_time)
        elif isinstance(self.procedure_vm, BaseProcedureViewModel):
            return float(self.procedure_vm.start_time)
        return 0.0

    @start_time.setter
    def start_time(self, new_start_time: float) -> None:
        """Синхронизация времени начала с активной моделью процедуры."""
        validated_time = max(0.0, float(new_start_time))
        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            self.procedure_vm.initial_time = validated_time
        elif isinstance(self.procedure_vm, BaseProcedureViewModel):
            self.procedure_vm.start_time = validated_time

    @property
    def stop_time(self) -> float:
        """Время экспозиции/окончания моделирования, определяемое активной процедурой (SSOT)."""
        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            return float(self.procedure_vm.time_per_step)
        elif isinstance(self.procedure_vm, PetProcedureViewModel):
            return float(self.procedure_vm.time_per_frame)
        elif isinstance(self.procedure_vm, BaseProcedureViewModel):
            return float(self.procedure_vm.stop_time)
        return 1.0

    @stop_time.setter
    def stop_time(self, new_stop_time: float) -> None:
        """Синхронизация времени с активной моделью процедуры."""
        validated_time = max(0.001, float(new_stop_time))
        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            self.procedure_vm.time_per_step = validated_time
        elif isinstance(self.procedure_vm, PetProcedureViewModel):
            self.procedure_vm.time_per_frame = validated_time
        elif isinstance(self.procedure_vm, BaseProcedureViewModel):
            self.procedure_vm.stop_time = validated_time

    def _format_time_macro(self, var_name: str) -> str:
        """Форматирует макрос переменной времени с учетом типа протокола."""
        clean_var = str(var_name).strip()
        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel) and clean_var in (
            "start_time",
            "stop_time",
            "step_time",
            "current_time",
        ):
            return f"${{{clean_var}}}"
        return f"${{{clean_var}}} s"

    def _get_start_time_cfg(self) -> Any:
        """Вычисляет конфигурацию времени начала для экспортера ядра."""
        if isinstance(self.procedure_vm, BaseProcedureViewModel):
            sim_mgr_bindings = self.procedure_vm.get_property_bindings("simulation_manager")
            bound_var = self.procedure_vm.bindings.get("start_time") or sim_mgr_bindings.get("start_time")
            if bound_var:
                return self._format_time_macro(bound_var)

        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            return "${start_time}"
        elif isinstance(self.procedure_vm, CustomSweepProcedureViewModel):
            all_sweep_vars = set(self.procedure_vm.grid_variables.keys()) | set(self.procedure_vm.zipped_variables.keys())
            if "start_time" in all_sweep_vars:
                return "${start_time} s"
            return float(self.procedure_vm.start_time) * units.s
        elif isinstance(self.procedure_vm, BaseProcedureViewModel):
            return float(self.procedure_vm.start_time) * units.s
        return 0.0 * units.ns

    def _get_stop_time_cfg(self) -> Any:
        """Вычисляет конфигурацию времени остановки для экспортера ядра."""
        if isinstance(self.procedure_vm, BaseProcedureViewModel):
            sim_mgr_bindings = self.procedure_vm.get_property_bindings("simulation_manager")
            bound_var = self.procedure_vm.bindings.get("stop_time") or sim_mgr_bindings.get("stop_time")
            if bound_var:
                return self._format_time_macro(bound_var)

        if isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
            return "${stop_time}"
        elif isinstance(self.procedure_vm, CustomSweepProcedureViewModel):
            all_sweep_vars = set(self.procedure_vm.grid_variables.keys()) | set(self.procedure_vm.zipped_variables.keys())
            if "stop_time" in all_sweep_vars:
                return "${stop_time} s"
            return float(self.procedure_vm.stop_time) * units.s
        elif isinstance(self.procedure_vm, PetProcedureViewModel):
            return float(self.procedure_vm.time_per_frame) * units.s
        elif isinstance(self.procedure_vm, BaseProcedureViewModel):
            return float(self.procedure_vm.stop_time) * units.s
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

    def _emit_cached_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Эмитирует сохраненные данные завершенной задачи в GUI при переключении фокуса после расчета."""
        tracks_data = snapshot.get("tracks")
        if isinstance(tracks_data, dict):
            batch_copy = {
                field_key: (np.array(field_val, copy=True) if isinstance(field_val, np.ndarray) else field_val)
                for field_key, field_val in tracks_data.items()
            }
            batch_copy["reset"] = True
            self.tracks_received.emit(batch_copy)

        spec_data = snapshot.get("spectrum")
        if isinstance(spec_data, np.ndarray):
            self.spectrum_received.emit(spec_data.copy())

        proj_data = snapshot.get("projection")
        if isinstance(proj_data, np.ndarray):
            proj_copy = proj_data.copy()
            self.projection_received.emit(proj_copy)
            self.stats_updated.emit(int(np.sum(proj_copy)), 0.0)

        dose_data = snapshot.get("dose")
        if isinstance(dose_data, dict):
            self.dose_volume_received.emit({grid_key: grid_val.copy() for grid_key, grid_val in dose_data.items()})
        elif isinstance(dose_data, np.ndarray):
            self.dose_volume_received.emit(dose_data.copy())

    def _resolve_shm_proj_shape(self, shm_size: int) -> Tuple[int, ...]:
        """Определяет форму массива проекции (4D куб или 2D срез), соответствующую размеру сегмента SharedMemory."""
        itemsize = np.dtype(np.float32).itemsize
        cube_bytes = int(np.prod(self._active_cube_shape) * itemsize)
        if shm_size >= cube_bytes:
            return self._active_cube_shape
        return tuple(int(dim) for dim in self.projection_shape)

    def _capture_task_snapshots(self) -> None:
        """Сохраняет финальные снимки всех задач из пер-тасковых сегментов SharedMemory перед их очисткой."""
        self._completed_task_snapshots.clear()
        multi_grid = len(self._active_dose_grids_info) > 1

        for task_id in range(self.num_tasks):
            snap: Dict[str, Any] = {}
            if task_id < len(self._proj_shms):
                try:
                    proj_shm = self._proj_shms[task_id]
                    target_shape = self._resolve_shm_proj_shape(proj_shm.size)
                    proj_arr = np.ndarray(target_shape, dtype=np.float32, buffer=proj_shm.buf)
                    snap["projection"] = proj_arr.copy()
                    del proj_arr
                except (OSError, ValueError, BufferError, TypeError):
                    pass

            if task_id < len(self._telem_rings):
                ring = self._telem_rings[task_id]
                try:
                    _cur, track_batch = ring.read_all_tracks()
                    if track_batch is not None:
                        track_copy: Dict[str, Any] = {
                            field_key: np.array(field_val, copy=True)
                            for field_key, field_val in track_batch.items()
                        }
                        track_copy["reset"] = True
                        snap["tracks"] = track_copy
                    else:
                        snap["tracks"] = {
                            "pos_x": np.empty(0, dtype=np.float32),
                            "pos_y": np.empty(0, dtype=np.float32),
                            "pos_z": np.empty(0, dtype=np.float32),
                            "particle_id": np.empty(0, dtype=np.int64),
                            "process_id": np.empty(0, dtype=np.int16),
                            "reset": True,
                        }
                except (OSError, ValueError, BufferError, TypeError):
                    pass

                try:
                    _scur, raw_energies = ring.read_spectrum()
                    if len(raw_energies) > 0:
                        snap["spectrum"] = (raw_energies / np.float32(units.keV)).astype(np.float32)
                    else:
                        snap["spectrum"] = np.empty(0, dtype=np.float32)
                except (OSError, ValueError, BufferError, TypeError):
                    pass

            if self._active_dose_grids_info:
                if multi_grid:
                    dose_dict: Dict[str, np.ndarray] = {}
                    for info in self._active_dose_grids_info:
                        g_name = info["name"]
                        g_shape = info["grid_shape"]
                        g_dtype = np.dtype(info.get("dtype", "float32"))
                        t_dose_name = get_task_dose_shm_name(
                            self._base_dose_shm_name, task_id, grid_name=g_name, multi_grid=True
                        )
                        try:
                            d_shm = shared_memory.SharedMemory(name=t_dose_name, create=False)
                            d_arr: Optional[np.ndarray] = None
                            try:
                                d_arr = np.ndarray(g_shape, dtype=g_dtype, buffer=d_shm.buf, order="F")
                                if float(np.max(d_arr)) > 0.0:
                                    dose_dict[g_name] = d_arr.copy(order="K")
                                else:
                                    dose_dict[g_name] = np.zeros(g_shape, dtype=g_dtype, order="F")
                            finally:
                                d_arr = None
                                d_shm.close()
                        except (OSError, ValueError, BufferError, TypeError):
                            pass
                    if dose_dict:
                        snap["dose"] = dose_dict
                else:
                    info = self._active_dose_grids_info[0]
                    g_shape = info["grid_shape"]
                    g_dtype = np.dtype(info.get("dtype", "float32"))
                    t_dose_name = get_task_dose_shm_name(
                        self._base_dose_shm_name, task_id, grid_name=info["name"], multi_grid=False
                    )
                    try:
                        d_shm = shared_memory.SharedMemory(name=t_dose_name, create=False)
                        d_arr_single: Optional[np.ndarray] = None
                        try:
                            d_arr_single = np.ndarray(g_shape, dtype=g_dtype, buffer=d_shm.buf, order="F")
                            if float(np.max(d_arr_single)) > 0.0:
                                snap["dose"] = d_arr_single.copy(order="K")
                            else:
                                snap["dose"] = np.zeros(g_shape, dtype=g_dtype, order="F")
                        finally:
                            d_arr_single = None
                            d_shm.close()
                    except (OSError, ValueError, BufferError, TypeError):
                        pass

            self._completed_task_snapshots[task_id] = snap

    def set_focused_job(self, index: int) -> None:
        """Переключение фокуса визуализации на задачу с указанным индексом."""
        self.focused_job_index = max(0, int(index))
        if self._ctrl_shm is not None:
            self._ctrl_shm.focused_job_index = self.focused_job_index
        if self._ipc_receiver is not None:
            self._ipc_receiver.set_focused_job(self.focused_job_index)
        elif self.focused_job_index in self._completed_task_snapshots:
            self._emit_cached_snapshot(self._completed_task_snapshots[self.focused_job_index])

    def clear_accumulation(self) -> None:
        """Сброс накопленных данных моделирования в IPCReceiver и буфере SharedMemory."""
        if self._ipc_receiver is not None:
            self._ipc_receiver.clear_accumulation()
        else:
            if 0 <= self.focused_job_index < len(self._proj_shms):
                try:
                    t_shm = self._proj_shms[self.focused_job_index]
                    target_shape = self._resolve_shm_proj_shape(t_shm.size)
                    t_view = np.ndarray(target_shape, dtype=np.float32, buffer=t_shm.buf)
                    t_view.fill(0.0)
                    del t_view
                except (OSError, ValueError, BufferError, TypeError):
                    pass
            if 0 <= self.focused_job_index < len(self._telem_rings):
                try:
                    self._telem_rings[self.focused_job_index].clear()
                except (OSError, ValueError, BufferError, TypeError):
                    pass
            if self._shm is not None:
                try:
                    target_shape = self._resolve_shm_proj_shape(self._shm.size)
                    np_buf = np.ndarray(target_shape, dtype=np.float32, buffer=self._shm.buf)
                    np_buf.fill(0.0)
                    del np_buf
                except (OSError, ValueError, BufferError, TypeError):
                    pass
            for dose_shm in self._dose_shms:
                try:
                    d_view = np.ndarray((dose_shm.size,), dtype=np.uint8, buffer=dose_shm.buf)
                    d_view.fill(0)
                    del d_view
                except (OSError, ValueError, BufferError, TypeError):
                    pass

    def step_once(self) -> None:
        """Выполнение одного расчетного шага сфокусированным воркером."""
        self.step_focused_worker()

    @staticmethod
    def _create_or_reset_shm(shm_name: str, nbytes: int) -> shared_memory.SharedMemory:
        """Создает или переоткрывает сегмент SharedMemory нужного размера без ошибки WinError 183."""
        shm, _ = create_or_reuse_shm(shm_name, nbytes)
        return shm

    @staticmethod
    def _resolve_non_conflicting_base_shm_name(
        base_name: str,
        num_tasks: int,
        proj_nbytes: int,
        telem_nbytes: int,
    ) -> str:
        """
        Подбирает префикс имени SharedMemory для сессии так, чтобы ни базовый сегмент,
        ни пер-тасковые сегменты проекции/телеметрии не конфликтовали с удерживаемыми в ОС
        сегментами меньшего размера.
        """
        gc.collect()
        for gen in range(0, 1024):
            candidate = base_name if gen == 0 else f"{base_name}_g{gen}"
            if is_shm_name_conflicting(candidate, proj_nbytes):
                continue
            conflict_found = False
            for task_id in range(num_tasks):
                t_proj_name = get_task_projection_shm_name(candidate, task_id)
                if is_shm_name_conflicting(t_proj_name, proj_nbytes):
                    conflict_found = True
                    break
                t_telem_name = get_task_telemetry_shm_name(candidate, task_id)
                if is_shm_name_conflicting(t_telem_name, telem_nbytes):
                    conflict_found = True
                    break
            if not conflict_found:
                return candidate
        return base_name

    @staticmethod
    def _resolve_non_conflicting_dose_shm_name(
        base_dose_name: str,
        num_tasks: int,
        dose_grids_info: List[Dict[str, Any]],
    ) -> str:
        """
        Подбирает префикс имени SharedMemory для 3D-карт дозы так, чтобы ни базовые,
        ни пер-тасковые сегменты не конфликтовали с удерживаемыми в ОС сегментами меньшего размера.
        """
        if not dose_grids_info:
            return base_dose_name
        gc.collect()
        multi_grid = len(dose_grids_info) > 1
        for gen in range(0, 1024):
            candidate = base_dose_name if gen == 0 else f"{base_dose_name}_g{gen}"
            conflict_found = False
            for info in dose_grids_info:
                grid_name = info["name"]
                grid_shape = info["grid_shape"]
                grid_dtype = np.dtype(info.get("dtype", "float32"))
                dose_nbytes = int(np.prod(grid_shape) * grid_dtype.itemsize)
                base_grid_shm = f"{candidate}_{grid_name}" if multi_grid else candidate
                if is_shm_name_conflicting(base_grid_shm, dose_nbytes):
                    conflict_found = True
                    break
                for task_id in range(num_tasks):
                    t_dose_name = get_task_dose_shm_name(
                        candidate,
                        task_id,
                        grid_name=grid_name,
                        multi_grid=multi_grid,
                    )
                    if is_shm_name_conflicting(t_dose_name, dose_nbytes):
                        conflict_found = True
                        break
                if conflict_found:
                    break
            if not conflict_found:
                return candidate
        return base_dose_name

    def _collect_dose_ipc_config(self) -> Tuple[Optional[str], Tuple[int, int, int], List[Dict[str, Any]]]:
        """Собирает метаданные активных сеток дозы для подключения IPCReceiver к SharedMemory."""
        base_shm_name = "nmsim_dose_shm"
        if self.data_manager_vm is not None:
            for handler_vm in self.data_manager_vm.handlers:
                if isinstance(handler_vm, DoseMapHandlerViewModel) and handler_vm.shm_name:
                    base_shm_name = str(handler_vm.shm_name)
                    break

        active_dose_nodes: List[DoseGridViewModel] = []
        if self.scene_vm is not None:
            for node_vm in self.scene_vm.all_nodes():
                if isinstance(node_vm, DoseGridViewModel) and node_vm.is_active:
                    active_dose_nodes.append(node_vm)

        if not active_dose_nodes:
            return None, (64, 64, 64), []

        dose_grids_info: List[Dict[str, Any]] = []
        for idx, dose_node_vm in enumerate(active_dose_nodes):
            grid_name = dose_node_vm.name or f"DoseGrid_{idx}"
            grid_shm = f"{base_shm_name}_{grid_name}" if len(active_dose_nodes) > 1 else base_shm_name
            dose_grids_info.append({
                "name": grid_name,
                "shm_name": grid_shm,
                "grid_shape": dose_node_vm.grid_shape,
                "dtype": dose_node_vm.dtype,
            })

        first_grid = active_dose_nodes[0]
        single_shm_name = base_shm_name if len(active_dose_nodes) == 1 else None
        return single_shm_name, first_grid.grid_shape, dose_grids_info

    def generate_jobs(self) -> List[Dict[str, Any]]:
        """
        Генерация плоского списка задач симуляции на основе геометрии сцены и активной процедуры.
        """
        if self._is_generating_jobs:
            return self._jobs

        if self.scene_vm is None or self.scene_vm.root_vm is None:
            self._jobs = []
            self.jobs_generated.emit([])
            return []

        self._is_generating_jobs = True
        try:
            if (
                isinstance(self.procedure_vm, StepAndShootProcedureViewModel)
                and self.procedure_vm._observed_scene is not self.scene_vm
            ):
                self.procedure_vm.sync_with_scene(self.scene_vm)
            elif not isinstance(self.procedure_vm, StepAndShootProcedureViewModel):
                self.procedure_vm.sync_with_scene(self.scene_vm)

            sweep_config = SweepCompiler.compile_protocol(self.procedure_vm.to_config())
            self._jobs = SweepCompiler.generate_job_matrix(sweep_config)
            self.jobs_generated.emit(self._jobs)
            return self._jobs
        finally:
            self._is_generating_jobs = False

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

        start_time_cfg: Any = self._get_start_time_cfg()
        stop_time_cfg: Any = self._get_stop_time_cfg()

        sim_mgr_cfg = SimulationManagerConfig(
            particles_number=self.particles_number,
            start_time=start_time_cfg,
            stop_time=stop_time_cfg,
            min_energy=self.min_energy * units.keV,
        )
        data_mgr_cfg = self.data_manager_vm.to_config()
        dist_registry = self.scene_vm.distribution_registry if self.scene_vm else None
        slots_reg = self.scene_vm.slots_registry if self.scene_vm else None
        bindings_reg = (
            self.procedure_vm.bindings_registry
            if (self.procedure_vm and self.procedure_vm.bindings_registry)
            else (self.scene_vm.bindings_registry if self.scene_vm else None)
        )
        sim_config = SceneExporter.export_to_config(
            root_node=root_core,
            simulation_manager_cfg=sim_mgr_cfg,
            data_manager_cfg=data_mgr_cfg,
            distribution_registry=dist_registry,
            slots_registry=slots_reg,
            pool_size=self.pool_size,
        )
        sim_config.protocol = self.procedure_vm.to_config()
        if bindings_reg:
            bound_cfg = ConfigTemplateBinder.apply_bindings(sim_config, bindings_reg)
            if isinstance(bound_cfg, SimulationConfig):
                sim_config = bound_cfg

        self.resolve_effective_max_energy()
        det_names = self.detector_names
        display_max_energy = float(self.max_energy / units.keV)
        self.detector_metadata_updated.emit(det_names, display_max_energy)

        if len(self.projection_shape) == 4:
            cube_shape: Tuple[int, ...] = (
                int(self.projection_shape[0]),
                int(self.projection_shape[1]),
                int(self.projection_shape[2]),
                int(self.projection_shape[3]),
            )
        else:
            rows = int(self.projection_shape[-2])
            cols = int(self.projection_shape[-1])
            cube_shape = (max(1, len(det_names)), self.energy_bins, rows, cols)
        self._active_cube_shape = cube_shape

        # 2. Инициализация IPC и пер-тасковых сегментов SharedMemory
        self._stop_ipc_receiver()
        self._cleanup_ipc()
        self._completed_task_snapshots.clear()

        num_tasks = max(1, len(self.jobs))
        self.num_tasks = num_tasks

        try:
            self._ctrl_shm = ControlBufferSHM(
                shm_name=DEFAULT_CTRL_SHM_NAME,
                create=True,
                initial_focused_job=self.focused_job_index,
            )
            active_ctrl_shm_name = self._ctrl_shm.shm_name

            nbytes = int(np.prod(cube_shape) * np.dtype(np.float32).itemsize)
            telem_nbytes = TelemetryRingBufferSHM.compute_byte_size(self.max_track_points)
            active_shm_name = self._resolve_non_conflicting_base_shm_name(
                self.shm_name,
                num_tasks,
                nbytes,
                telem_nbytes,
            )

            self._shm = self._create_or_reset_shm(active_shm_name, nbytes)
            main_proj_view = np.ndarray(cube_shape, dtype=np.float32, buffer=self._shm.buf)
            main_proj_view.fill(0.0)
            del main_proj_view

            self._mp_manager = Manager()
            self._telemetry_queue = self._mp_manager.Queue()
            self._global_pause_event = self._mp_manager.Event()
            self._global_pause_event.set()
            self._focused_pause_event = self._mp_manager.Event()
            self._focused_pause_event.set()
            self._step_trigger_event = self._mp_manager.Event()
            self._step_trigger_event.clear()

            dose_shm_name, dose_grid_shape, dose_grids_info = self._collect_dose_ipc_config()
            base_dose_shm_name = "nmsim_dose_shm"
            if self.data_manager_vm is not None:
                for handler_vm in self.data_manager_vm.handlers:
                    if isinstance(handler_vm, DoseMapHandlerViewModel) and handler_vm.shm_name:
                        base_dose_shm_name = str(handler_vm.shm_name)
                        break

            multi_grid = len(dose_grids_info) > 1
            base_dose_shm_name = self._resolve_non_conflicting_dose_shm_name(
                base_dose_shm_name,
                num_tasks,
                dose_grids_info,
            )
            self._base_dose_shm_name = base_dose_shm_name
            for info in dose_grids_info:
                info["shm_name"] = f"{base_dose_shm_name}_{info['name']}" if multi_grid else base_dose_shm_name
            if len(dose_grids_info) == 1:
                dose_shm_name = base_dose_shm_name
            self._active_dose_grids_info = list(dose_grids_info)

            if dose_grids_info:
                has_dose_handler_cfg = False
                for handler_cfg in sim_config.data_manager.handlers:
                    if isinstance(handler_cfg, DoseMapHandlerConfig):
                        handler_cfg.shm_name = base_dose_shm_name
                        has_dose_handler_cfg = True
                if not has_dose_handler_cfg and base_dose_shm_name != "nmsim_dose_shm":
                    sim_config.data_manager.handlers.append(
                        DoseMapHandlerConfig(type="DoseMapHandler", shm_name=base_dose_shm_name)
                    )

            for task_id in range(num_tasks):
                t_proj_name = get_task_projection_shm_name(active_shm_name, task_id)
                t_proj_shm = self._create_or_reset_shm(t_proj_name, nbytes)
                t_proj_view = np.ndarray(cube_shape, dtype=np.float32, buffer=t_proj_shm.buf)
                t_proj_view.fill(0.0)
                del t_proj_view
                self._proj_shms.append(t_proj_shm)

                t_telem_name = get_task_telemetry_shm_name(active_shm_name, task_id)
                ring = TelemetryRingBufferSHM(
                    shm_name=t_telem_name,
                    create=True,
                    track_capacity=self.max_track_points,
                )
                self._telem_rings.append(ring)

                for info in dose_grids_info:
                    grid_shape = info["grid_shape"]
                    grid_dtype = np.dtype(info.get("dtype", "float32"))
                    dose_nbytes = int(np.prod(grid_shape) * grid_dtype.itemsize)
                    t_dose_name = get_task_dose_shm_name(
                        base_dose_shm_name,
                        task_id,
                        grid_name=info["name"],
                        multi_grid=multi_grid,
                    )
                    t_dose_shm = self._create_or_reset_shm(t_dose_name, dose_nbytes)
                    t_dose_view = np.ndarray(grid_shape, dtype=grid_dtype, buffer=t_dose_shm.buf, order="F")
                    t_dose_view.fill(0.0)
                    del t_dose_view
                    self._dose_shms.append(t_dose_shm)

            for info in dose_grids_info:
                grid_shape = info["grid_shape"]
                grid_dtype = np.dtype(info.get("dtype", "float32"))
                dose_nbytes = int(np.prod(grid_shape) * grid_dtype.itemsize)
                dose_shm = self._create_or_reset_shm(info["shm_name"], dose_nbytes)
                dose_view = np.ndarray(grid_shape, dtype=grid_dtype, buffer=dose_shm.buf, order="F")
                dose_view.fill(0.0)
                del dose_view
                self._dose_shms.append(dose_shm)

            first_dose_dtype = (
                np.dtype(dose_grids_info[0].get("dtype", "float32"))
                if dose_grids_info
                else np.dtype(np.float32)
            )

            # 3. Запуск приемника телеметрии IPCReceiver
            self._ipc_receiver = IPCReceiver(
                track_queue=self._telemetry_queue,
                shm_name=active_shm_name,
                projection_shape=cube_shape,
                projection_dtype=np.dtype(np.float32),
                fps=30.0,
                dose_shm_name=dose_shm_name,
                dose_grid_shape=dose_grid_shape,
                dose_dtype=first_dose_dtype,
                dose_grids_info=dose_grids_info,
                focused_job_index=self.focused_job_index,
                num_tasks=num_tasks,
                per_task_shm=True,
                base_dose_shm_name=base_dose_shm_name,
                parent=self,
            )
            self._ipc_receiver.tracks_received.connect(self.tracks_received)
            self._ipc_receiver.projection_received.connect(self._on_ipc_proj_received)
            self._ipc_receiver.spectrum_received.connect(self._on_ipc_spec_received)
            self._ipc_receiver.stats_updated.connect(self.stats_updated)
            self._ipc_receiver.dose_volume_received.connect(self._on_ipc_dose_received)
            self._ipc_receiver.task_event.connect(self._on_task_event)
            self._ipc_receiver.start()

            # 4. Запуск Orchestrator в фоновом потоке
            orchestrator = Orchestrator(sim_config)
            self._worker_thread = _OrchestratorWorkerThread(
                orchestrator=orchestrator,
                telemetry_queue=self._telemetry_queue,
                focused_job_index=self.focused_job_index,
                focused_shm_name=active_shm_name,
                projection_shape=cube_shape,
                global_pause_event=self._global_pause_event,
                focused_pause_event=self._focused_pause_event,
                step_trigger_event=self._step_trigger_event,
                per_task_shm=True,
                ctrl_shm_name=active_ctrl_shm_name,
                max_energy=self.max_energy,
                energy_bins=self.energy_bins,
                parent=self,
            )
            self._worker_thread.completed.connect(self._on_worker_completed)
            self._worker_thread.failed.connect(self._on_worker_failed)

            self._is_running = True
            self._is_paused = False
            self._is_focused_paused = False
            self._worker_thread.start()
            self.session_started.emit()
        except Exception:
            self._stop_ipc_receiver()
            self._cleanup_ipc()
            raise

    def _on_ipc_proj_received(self, proj_data: Any) -> None:
        """Синхронно доставляет кадр 2D-проекции в GUI и освобождает слот backpressure в IPCReceiver."""
        try:
            self.projection_received.emit(proj_data)
        finally:
            if self._ipc_receiver is not None:
                self._ipc_receiver.mark_proj_delivered()

    def _on_ipc_spec_received(self, spec_data: Any) -> None:
        """Синхронно доставляет массив спектра в GUI и освобождает слот backpressure в IPCReceiver."""
        try:
            self.spectrum_received.emit(spec_data)
        finally:
            if self._ipc_receiver is not None:
                self._ipc_receiver.mark_spec_delivered()

    def _on_ipc_dose_received(self, dose_data: Any) -> None:
        """Синхронно доставляет 3D-карту дозы в GUI и освобождает слот backpressure в IPCReceiver."""
        try:
            self.dose_volume_received.emit(dose_data)
        finally:
            if self._ipc_receiver is not None:
                self._ipc_receiver.mark_dose_delivered()

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
        self._capture_task_snapshots()
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
        if self._ctrl_shm is not None:
            self._ctrl_shm.global_run = False
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
        if self._ctrl_shm is not None:
            self._ctrl_shm.global_run = True
            self._ctrl_shm.focused_run = True
        if self._is_running and self._is_paused:
            self._is_paused = False
            self._is_focused_paused = False
            if self._global_pause_event is not None:
                self._global_pause_event.set()
            if self._focused_pause_event is not None:
                self._focused_pause_event.set()
            self.session_resumed.emit()

    def pause_focused_worker(self) -> None:
        """Локальная приостановка сфокусированного процесса для покадрового анализа."""
        if self._ctrl_shm is not None:
            self._ctrl_shm.focused_run = False
        if self._is_running and not self._is_focused_paused:
            self._is_focused_paused = True
            if self._focused_pause_event is not None:
                self._focused_pause_event.clear()
            self.focused_worker_paused.emit(self.focused_job_index)

    def resume_focused_worker(self) -> None:
        """Возобновление работы сфокусированного процесса."""
        if self._ctrl_shm is not None:
            self._ctrl_shm.focused_run = True
        if self._is_running and self._is_focused_paused:
            self._is_focused_paused = False
            if self._focused_pause_event is not None:
                self._focused_pause_event.set()
            self.focused_worker_resumed.emit(self.focused_job_index)

    def step_focused_worker(self) -> None:
        """Выполнение одного дискретного расчетного шага пачки частиц сфокусированным воркером."""
        if self._ctrl_shm is not None:
            self._ctrl_shm.focused_run = False
            self._ctrl_shm.trigger_step()
        if self._is_running and self._step_trigger_event is not None:
            self._step_trigger_event.set()

    def stop(self) -> None:
        """Кооперативная остановка всех процессов оркестратора и освобождение ресурсов."""
        was_running = self._is_running
        self._is_running = False
        self._is_paused = False
        self._is_focused_paused = False

        if self._ctrl_shm is not None:
            try:
                self._ctrl_shm.stop_requested = True
            except (OSError, ValueError, BufferError):
                pass
        if self._global_pause_event is not None:
            try:
                self._global_pause_event.set()
            except (OSError, ValueError, RuntimeError):
                pass
        if self._focused_pause_event is not None:
            try:
                self._focused_pause_event.set()
            except (OSError, ValueError, RuntimeError):
                pass

        if self._worker_thread is not None:
            if self._worker_thread.isRunning():
                if not self._worker_thread.wait(3000):
                    try:
                        self._worker_thread.orchestrator.terminate_pool()
                    except (OSError, ValueError, RuntimeError):
                        pass
                    if not self._worker_thread.wait(1000):
                        self._worker_thread.terminate()
                        self._worker_thread.wait(1000)
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
        if self._ctrl_shm is not None:
            try:
                self._ctrl_shm.close(unlink=True)
            except (FileNotFoundError, OSError, ValueError, BufferError):
                pass
            self._ctrl_shm = None

        proj_shms = list(self._proj_shms)
        self._proj_shms.clear()
        for proj_shm in proj_shms:
            safe_close_shm(proj_shm, unlink=True)

        telem_rings = list(self._telem_rings)
        self._telem_rings.clear()
        for ring in telem_rings:
            try:
                ring.close(unlink=True)
            except (FileNotFoundError, OSError, ValueError, BufferError):
                pass

        if self._shm is not None:
            main_shm = self._shm
            self._shm = None
            safe_close_shm(main_shm, unlink=True)

        dose_shms = list(self._dose_shms)
        self._dose_shms.clear()
        for dose_shm in dose_shms:
            safe_close_shm(dose_shm, unlink=True)

        if self._mp_manager is not None:
            try:
                self._mp_manager.shutdown()
            except (OSError, RuntimeError, ValueError):
                pass
            self._mp_manager = None
        self._telemetry_queue = None
        self._global_pause_event = None
        self._focused_pause_event = None
        self._step_trigger_event = None

    def close(self) -> None:
        self.stop()


__all__ = [
    "OrchestratorSession",
]

