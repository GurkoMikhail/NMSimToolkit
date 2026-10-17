import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import hepunits as units
from PySide6.QtCore import QObject, Signal

from core.config.models import (
    AnyProtocolConfig,
    StepAndShootProtocolConfig,
    CustomSweepProtocolConfig,
)
from core.scene.gamma_camera_node import GammaCameraNode
from gui.factories.gamma_camera_factory import create_default_gamma_camera
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.geometry.spect_kinematics import compute_orbit_matrix
from core.scene.gantry_node import GantryNode
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.gamma_camera_vm import GammaCameraViewModel
from gui.viewmodels.nodes.gantry_vm import GantryViewModel
from gui.viewport_3d.kinematic_constraints import (
    IKinematicConstraint,
    SpectOrbitKinematicConstraint,
    CameraMountKinematicConstraint,
    GantryKinematicConstraint,
)

_logger = logging.getLogger(__name__)


class BaseProcedureViewModel(QObject):
    """
    Абстрактный базовый класс модели представления процедуры / протокола исследования.
    Обеспечивает реактивное оповещение UI об изменении параметров протокола
    и синхронизацию с графом сцены.
    """

    changed = Signal()
    parameter_changed = Signal(str, object)
    bindings_changed = Signal()

    def __init__(
        self,
        name: str = "Procedure",
        procedure_type: str = "Base",
        parent: Optional[QObject] = None,
        start_time: float = 0.0,
        stop_time: float = 1.0,
    ) -> None:
        super().__init__(parent)
        self._name: str = name
        self._procedure_type: str = procedure_type
        self._start_time: float = max(0.0, float(start_time))
        self._stop_time: float = max(0.001, float(stop_time))
        # Единый реестр привязок свойств узлов сцены к переменным процедуры
        # Ключ: SpatialNode ядра (или NodeViewModel), значение: словарь {имя_свойства: имя_переменной}
        self.bindings_registry: Dict[Any, Dict[str, str]] = {}
        # Скалярные привязки параметров самой процедуры (например, radius -> cam_radius, start_time -> t_start)
        self.bindings: Dict[str, str] = {}
        self._is_syncing_with_scene: bool = False

    def get_variables(self) -> Dict[str, List[float]]:
        """
        Возвращает словарь всех доступных переменных протокола и списков их значений: {имя: [значения]}.
        """
        return {}

    def get_variable_names(self) -> List[str]:
        """Возвращает список имен всех доступных переменных процедуры."""
        return list(self.get_variables().keys())

    def bind_node_property(self, node: Any, prop_name: str, var_name: str) -> None:
        """Связывает свойство узла с переменной процедуры в едином реестре."""
        if not var_name:
            self.unbind_node_property(node, prop_name)
            return
        clean_var = str(var_name)
        if node == "simulation_manager" and prop_name in ("start_time", "stop_time"):
            self.bindings[prop_name] = clean_var
        if node not in self.bindings_registry:
            self.bindings_registry[node] = {}
        if self.bindings_registry[node].get(prop_name) == clean_var:
            return
        self.bindings_registry[node][prop_name] = clean_var
        self.bindings_changed.emit()
        if not self._is_syncing_with_scene:
            self.changed.emit()

    def unbind_node_property(self, node: Any, prop_name: str) -> None:
        """Удаляет привязку свойства узла из единого реестра."""
        if node == "simulation_manager" and prop_name in ("start_time", "stop_time"):
            self.bindings.pop(prop_name, None)
        if node in self.bindings_registry:
            if prop_name not in self.bindings_registry[node]:
                return
            self.bindings_registry[node].pop(prop_name, None)
            if not self.bindings_registry[node]:
                self.bindings_registry.pop(node, None)
            self.bindings_changed.emit()
            if not self._is_syncing_with_scene:
                self.changed.emit()

    def get_node_bindings(self, node: Any) -> Dict[str, str]:
        """Возвращает словарь привязок для указанного узла сцены."""
        return dict(self.bindings_registry.get(node, {}))

    def bind_property(self, target: Any, prop_name: str, var_name: str) -> None:
        """Универсальное связывание свойства узла сцены или раздела настроек с переменной."""
        self.bind_node_property(target, prop_name, var_name)

    def unbind_property(self, target: Any, prop_name: str) -> None:
        """Универсальное удаление связывания свойства."""
        self.unbind_node_property(target, prop_name)

    def get_property_bindings(self, target: Any) -> Dict[str, str]:
        """Возвращает привязки для указанного целевого узла или раздела."""
        return self.get_node_bindings(target)

    @property
    def protocol_standard_variables(self) -> List[str]:
        """Список стандартных переменных протокола."""
        return []

    def is_standard_variable(self, variable_name: str) -> bool:
        """Проверяет, является ли переменная стандартной переменной протокола."""
        return False

    def is_read_only_variable(self, variable_name: str) -> bool:
        """Проверяет, защищена ли переменная от ручного редактирования в таблице."""
        return False

    @property
    def name(self) -> str:
        """Пользовательское наименование процедуры."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name != value:
            self._name = str(value)
            self.changed.emit()
            self.parameter_changed.emit("name", self._name)

    @property
    def procedure_type(self) -> str:
        """Тип процедуры (например, SPECT, PET, CustomSweep)."""
        return self._procedure_type

    @property
    def start_time(self) -> float:
        """Статическое время начала интервала симуляции (секунды)."""
        return self._start_time

    @start_time.setter
    def start_time(self, new_start_time: float) -> None:
        validated_start = max(0.0, float(new_start_time))
        if self._start_time != validated_start:
            self._start_time = validated_start
            self.changed.emit()
            self.parameter_changed.emit("start_time", validated_start)

    @property
    def stop_time(self) -> float:
        """Статическое время окончания интервала симуляции (секунды)."""
        return self._stop_time

    @stop_time.setter
    def stop_time(self, new_stop_time: float) -> None:
        validated_stop = max(0.001, float(new_stop_time))
        if self._stop_time != validated_stop:
            self._stop_time = validated_stop
            self.changed.emit()
            self.parameter_changed.emit("stop_time", validated_stop)

    def to_config(self) -> AnyProtocolConfig:
        """Конвертация параметров процедуры в Pydantic-конфигурацию ядра."""
        raise NotImplementedError("to_config() должен быть реализован в подклассе.")

    def sync_with_scene(self, scene_vm: Any) -> None:
        """Синхронизация параметров процедуры с геометрией сцены (SceneViewModel)."""
        pass

    def get_kinematic_constraint_for_node(
        self,
        node_vm: NodeViewModel
    ) -> Optional[IKinematicConstraint]:
        """
        Возвращает кинематическое ограничение для выбранного узла в контексте
        данной процедуры. По умолчанию ограничений нет (возвращается None -> свободный 6-DOF).
        """
        return None


class CustomSweepProcedureViewModel(BaseProcedureViewModel):
    """
    Модель представления пользовательского протокола сканирования (параметрический скан).
    """

    def __init__(
        self,
        grid_variables: Optional[Dict[str, List[float]]] = None,
        zipped_variables: Optional[Dict[str, List[float]]] = None,
        name: str = "Пользовательский скан (Custom Sweep)",
        procedure_type: str = "CustomSweep",
        parent: Optional[QObject] = None,
        start_time: float = 0.0,
        stop_time: float = 1.0,
    ) -> None:
        super().__init__(
            name=name,
            procedure_type=procedure_type,
            parent=parent,
            start_time=start_time,
            stop_time=stop_time,
        )
        self._grid_variables: Dict[str, List[float]] = dict(grid_variables) if grid_variables else {}
        self._zipped_variables: Dict[str, List[float]] = dict(zipped_variables) if zipped_variables else {}

    @property
    def grid_variables(self) -> Dict[str, List[float]]:
        return dict(self._grid_variables)

    @grid_variables.setter
    def grid_variables(self, new_grid_vars: Dict[str, List[float]]) -> None:
        self._grid_variables = dict(new_grid_vars)
        self.changed.emit()
        self.parameter_changed.emit("grid_variables", self._grid_variables)

    @property
    def zipped_variables(self) -> Dict[str, List[float]]:
        return dict(self._zipped_variables)

    @zipped_variables.setter
    def zipped_variables(self, new_zipped_vars: Dict[str, List[float]]) -> None:
        self._zipped_variables = dict(new_zipped_vars)
        self.changed.emit()
        self.parameter_changed.emit("zipped_variables", self._zipped_variables)

    @property
    def all_variables(self) -> Dict[str, List[float]]:
        """Объединенный словарь всех варьируемых переменных процедуры: {имя: [значения]}."""
        return {**self._grid_variables, **self._zipped_variables}

    def get_variables(self) -> Dict[str, List[float]]:
        """Возвращает все варьируемые переменные и их значения."""
        return self.all_variables

    def get_variable_values(self, var_name: str) -> List[float]:
        """Возвращает список конкретных значений для указанной переменной."""
        clean = str(var_name).strip()
        if clean in self._grid_variables:
            return list(self._grid_variables[clean])
        elif clean in self._zipped_variables:
            return list(self._zipped_variables[clean])
        return []

    def set_variable(
        self,
        var_name: str,
        values: List[float],
        sweep_type: str = "grid",
    ) -> None:
        """Добавляет или обновляет переменную и список ее значений."""
        clean_name = str(var_name).strip().replace(" ", "_")
        float_values = [float(val) for val in values]
        if sweep_type == "zipped":
            self._grid_variables.pop(clean_name, None)
            self._zipped_variables[clean_name] = float_values
        else:
            self._zipped_variables.pop(clean_name, None)
            self._grid_variables[clean_name] = float_values
        self.changed.emit()
        self.parameter_changed.emit(clean_name, float_values)

    def remove_variable(self, var_name: str) -> None:
        """Удаляет переменную и очищает связанные с ней привязки из единого реестра и словаря bindings."""
        clean_name = str(var_name).strip()
        removed = self._grid_variables.pop(clean_name, None) is not None
        removed = (self._zipped_variables.pop(clean_name, None) is not None) or removed
        if removed:
            scalar_props_to_remove = [
                prop_key for prop_key, bound_var_name in self.bindings.items()
                if bound_var_name == clean_name
            ]
            for prop_key in scalar_props_to_remove:
                self.bindings.pop(prop_key, None)

            nodes_to_clean: List[Any] = []
            for node_key, bindings_map in list(self.bindings_registry.items()):
                properties_to_remove = [
                    prop_name for prop_name, bound_var_name in bindings_map.items()
                    if bound_var_name == clean_name
                ]
                for prop_name in properties_to_remove:
                    bindings_map.pop(prop_name, None)
                if not bindings_map:
                    nodes_to_clean.append(node_key)
            for node_key in nodes_to_clean:
                self.bindings_registry.pop(node_key, None)
            self.bindings_changed.emit()
            self.changed.emit()
            self.parameter_changed.emit("variable_removed", clean_name)

    def add_zipped_step(self, step_values: Optional[Dict[str, float]] = None) -> None:
        """Добавляет один новый шаг (строку) во все связанные переменные zipped_variables."""
        if not self._zipped_variables:
            return
        step_payload = step_values or {}
        for var_name, value_list in self._zipped_variables.items():
            if var_name in step_payload:
                new_value = float(step_payload[var_name])
            else:
                new_value = float(value_list[-1]) if value_list else 0.0
            value_list.append(new_value)
        self.changed.emit()
        self.parameter_changed.emit("zipped_variables", self._zipped_variables)

    def remove_zipped_step(self, step_index: int = -1) -> None:
        """Удаляет шаг (строку) по индексу из всех связанных переменных zipped_variables."""
        if not self._zipped_variables:
            return
        total_steps = len(next(iter(self._zipped_variables.values())))
        if total_steps <= 1:
            return
        target_index = step_index if step_index >= 0 else total_steps - 1
        if 0 <= target_index < total_steps:
            for value_list in self._zipped_variables.values():
                value_list.pop(target_index)
            self.changed.emit()
            self.parameter_changed.emit("zipped_variables", self._zipped_variables)

    def to_config(self) -> Union[CustomSweepProtocolConfig, StepAndShootProtocolConfig]:
        return CustomSweepProtocolConfig(
            type="CustomSweep",
            grid_variables=self._grid_variables,
            zipped_variables=self._zipped_variables
        )


class StepAndShootProcedureViewModel(CustomSweepProcedureViewModel):
    """
    Модель представления протокола пошагового сканирования (Step-and-Shoot).
    Управляет числом шагов гантри, радиусом орбиты,
    временем экспозиции на шаг и сектором сканирования.
    Наследует CustomSweepProcedureViewModel, автоматически наполняя
    стандартные переменные протокола (gantry_angle, radius, step_time) и поддерживая
    дополнительные пользовательские переменные.
    """

    PROTOCOL_STANDARD_VARIABLES: tuple[str, ...] = (
        "gantry_angle",
        "start_time",
        "stop_time",
        "step_time",
        "radius",
        "current_angle",
        "current_time",
    )
    READ_ONLY_STANDARD_VARIABLES: tuple[str, ...] = (
        "gantry_angle",
        "start_time",
        "stop_time",
        "step_time",
        "current_angle",
        "current_time",
    )

    @property
    def protocol_standard_variables(self) -> List[str]:
        """Список стандартных переменных протокола."""
        return list(self.PROTOCOL_STANDARD_VARIABLES)

    def is_standard_variable(self, variable_name: str) -> bool:
        """Проверяет, является ли переменная стандартной переменной протокола."""
        return str(variable_name) in self.PROTOCOL_STANDARD_VARIABLES

    def is_read_only_variable(self, variable_name: str) -> bool:
        """Проверяет, защищена ли стандартная переменная от ручного редактирования в таблице."""
        return str(variable_name) in self.READ_ONLY_STANDARD_VARIABLES

    def __init__(
        self,
        steps: int = 32,
        gamma_cameras: int = 2,
        radius: Any = 250.0,
        time_per_step: float = 1.0,
        initial_time: float = 0.0,
        sequential_time: bool = True,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        head_mode: str = "Симметричный (360°/N)",
        head_angles: Optional[List[float]] = None,
        endpoint: bool = False,
        grid_variables: Optional[Dict[str, List[float]]] = None,
        zipped_variables: Optional[Dict[str, List[float]]] = None,
        parent: Optional[QObject] = None,
        **kwargs: Any,
    ) -> None:
        proc_type = str(kwargs.get("procedure_type", "StepAndShoot"))
        proc_name = str(kwargs.get("name", "Протокол пошагового сканирования (Step-and-Shoot)"))
        super().__init__(
            grid_variables=grid_variables,
            zipped_variables=zipped_variables,
            name=proc_name,
            procedure_type=proc_type,
            parent=parent,
        )
        self._steps: int = max(1, int(steps))
        self._gamma_cameras: int = max(1, int(gamma_cameras))
        if isinstance(radius, (list, tuple)):
            self._radius: Any = [float(radius_element) for radius_element in radius]
        elif radius is not None:
            self._radius = max(10.0, float(radius))
        else:
            self._radius = 250.0

        self._time_per_step: float = max(0.001, float(time_per_step))
        self._initial_time: float = max(0.0, float(initial_time))
        self._sequential_time: bool = bool(sequential_time)
        self._start_angle: float = float(start_angle)
        self._end_angle: float = float(end_angle)
        self._head_mode: str = head_mode
        self._head_angles: List[float] = list(head_angles) if head_angles is not None else []
        self._endpoint: bool = bool(endpoint)
        self._observed_gantry: Optional[Any] = None
        self._observed_cameras: List[Any] = []
        self._observed_scene: Optional[Any] = None
        self._is_syncing_with_scene: bool = False

        if not self._head_angles:
            self._recalculate_head_angles()

        self._update_standard_variables()

    def _update_standard_variables(self, regenerate_angles: bool = True) -> None:
        """
        Автоматическое наполнение и синхронизация стандартных переменных протокола:
        gantry_angle, current_angle, radius, start_time, stop_time, step_time, current_time.
        """
        steps = max(1, int(self._steps))
        endpoint = bool(self._endpoint)
        existing_angles = self._zipped_variables.get("gantry_angle")
        if regenerate_angles or existing_angles is None or len(existing_angles) != steps:
            angles_list = np.linspace(
                float(self._start_angle),
                float(self._end_angle),
                steps,
                endpoint=endpoint,
            ).tolist()
        else:
            angles_list = list(existing_angles)

        step_duration = float(self._time_per_step)
        initial_offset = float(self._initial_time)
        time_list = [step_duration] * steps
        if self._sequential_time:
            start_time_list = [initial_offset + step_idx * step_duration for step_idx in range(steps)]
            stop_time_list = [initial_offset + (step_idx + 1) * step_duration for step_idx in range(steps)]
        else:
            start_time_list = [initial_offset] * steps
            stop_time_list = [initial_offset + step_duration] * steps
        self._start_time = start_time_list[0]
        self._stop_time = stop_time_list[0]

        if isinstance(self._radius, list):
            fallback_radius = float(self._radius[0]) if self._radius else 250.0
            radius_list = [float(radius_element) for radius_element in self._radius[:steps]]
            radius_list.extend([fallback_radius] * max(0, steps - len(radius_list)))
            self._radius = radius_list
        else:
            radius_value = float(self._radius) if self._radius is not None else 250.0
            radius_list = [radius_value] * steps

        standard_updates = {
            "gantry_angle": list(angles_list),
            "current_angle": list(angles_list),
            "start_time": start_time_list,
            "stop_time": stop_time_list,
            "step_time": time_list,
            "current_time": time_list,
            "radius": list(radius_list),
        }
        self._zipped_variables.update(standard_updates)

        for variable_name, value_list in list(self._zipped_variables.items()):
            if variable_name in standard_updates:
                continue
            if len(value_list) != steps:
                padding_value = value_list[-1] if value_list else 0.0
                trimmed_list = value_list[:steps]
                trimmed_list.extend([padding_value] * max(0, steps - len(trimmed_list)))
                self._zipped_variables[variable_name] = trimmed_list

    def _disconnect_scene_observers(self) -> None:
        """Отсоединение слушателей от предыдущей сцены и станины."""
        if self._observed_scene is not None:
            try:
                self._observed_scene.node_removed.disconnect(self._on_scene_node_removed)
            except (RuntimeError, TypeError):
                pass
            self._observed_scene = None

        if self._observed_gantry is not None:
            try:
                self._observed_gantry.property_changed.disconnect(self._on_gantry_property_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self._observed_gantry.child_added.disconnect(self._on_gantry_child_added)
            except (RuntimeError, TypeError):
                pass
            try:
                self._observed_gantry.child_removed.disconnect(self._on_gantry_child_removed)
            except (RuntimeError, TypeError):
                pass
            self._observed_gantry = None

        for camera_view_model in self._observed_cameras:
            try:
                camera_view_model.property_changed.disconnect(self._on_camera_property_changed)
            except (RuntimeError, TypeError):
                pass
        self._observed_cameras.clear()

    def _on_scene_node_removed(self, removed_node_vm: Any) -> None:
        """Слушатель удаления узлов из дерева сцены."""
        if removed_node_vm is self._observed_gantry:
            self._disconnect_scene_observers()

    def _on_gantry_property_changed(self, property_name: str, property_value: Any) -> None:
        """Слушатель (Observer) изменений свойств узла станины GantryViewModel."""
        if self._is_syncing_with_scene:
            return
        if property_name == "gantry_angle":
            angle_value = float(property_value)
            if abs(self._start_angle - angle_value) > 1e-4:
                self._is_syncing_with_scene = True
                try:
                    self._start_angle = angle_value
                    self.changed.emit()
                    self.parameter_changed.emit("start_angle", angle_value)
                finally:
                    self._is_syncing_with_scene = False

    def _on_camera_property_changed(self, property_name: str, property_value: Any) -> None:
        """Слушатель (Observer) изменений свойств дочерних гамма-камер GammaCameraViewModel."""
        if self._is_syncing_with_scene:
            return
        if property_name == "local_matrix" and isinstance(property_value, np.ndarray):
            sender_obj = self.sender()
            sender_half_thickness = sender_obj.half_thickness if isinstance(sender_obj, GammaCameraViewModel) else 0.0
            position_vector = property_value[0:3, 3]
            center_distance = float(np.hypot(position_vector[0], position_vector[1]))
            new_radius = max(10.0, center_distance - sender_half_thickness)
            if abs(self._radius - new_radius) > 1e-4:
                self._is_syncing_with_scene = True
                try:
                    self._radius = new_radius
                    for other_camera_vm in self._observed_cameras:
                        if other_camera_vm is not sender_obj:
                            other_position = other_camera_vm.local_matrix[0:3, 3]
                            other_angle = float(np.degrees(np.arctan2(other_position[1], other_position[0])) % 360.0)
                            other_camera_vm.local_matrix = compute_orbit_matrix(
                                radius=new_radius,
                                angle=other_angle,
                                z=float(other_position[2]),
                                half_thickness=other_camera_vm.half_thickness,
                            )
                    self.changed.emit()
                    self.parameter_changed.emit("radius", new_radius)
                finally:
                    self._is_syncing_with_scene = False

    def _on_gantry_child_added(self, child_node_vm: Any) -> None:
        """Слушатель добавления узлов на станину."""
        if isinstance(child_node_vm.core_node, GammaCameraNode) and child_node_vm not in self._observed_cameras:
            child_node_vm.property_changed.connect(self._on_camera_property_changed)
            self._observed_cameras.append(child_node_vm)

    def _on_gantry_child_removed(self, child_node_vm: Any) -> None:
        """Слушатель удаления узлов со станины."""
        if child_node_vm in self._observed_cameras:
            try:
                child_node_vm.property_changed.disconnect(self._on_camera_property_changed)
            except (RuntimeError, TypeError):
                pass
            self._observed_cameras.remove(child_node_vm)

    def _recalculate_head_angles(self) -> None:
        """Автоматический пересчет смещений головок в зависимости от выбранного режима."""
        if "Симметричный" in self._head_mode or "symmetric" in self._head_mode.lower():
            step = 360.0 / self._gamma_cameras if self._gamma_cameras > 0 else 0.0
            self._head_angles = [step * idx for idx in range(self._gamma_cameras)]
        elif "90" in self._head_mode or "L-режим" in self._head_mode or "l-mode" in self._head_mode.lower():
            self._gamma_cameras = 2
            self._head_angles = [0.0, 90.0]

    @property
    def steps(self) -> int:
        """Число дискретных шагов вращения гантри ОФЭКТ."""
        return self._steps

    @steps.setter
    def steps(self, new_steps: int) -> None:
        validated_steps = max(1, int(new_steps))
        if self._steps != validated_steps:
            self._steps = validated_steps
            self._update_standard_variables()
            self.changed.emit()
            self.parameter_changed.emit("steps", validated_steps)

    @property
    def total_projections(self) -> int:
        """Общее число получаемых 2D-проекций: steps * gamma_cameras."""
        return self._steps * self._gamma_cameras

    @property
    def gamma_cameras(self) -> int:
        """Количество детекторных головок гамма-камер."""
        return self._gamma_cameras

    @gamma_cameras.setter
    def gamma_cameras(self, new_cameras: int) -> None:
        camera_count = max(1, int(new_cameras))
        if self._gamma_cameras != camera_count:
            self._gamma_cameras = camera_count
            self._recalculate_head_angles()
            self.changed.emit()
            self.parameter_changed.emit("gamma_cameras", camera_count)

    @property
    def radius(self) -> float:
        """Радиус орбиты вращения детекторов вокруг изоцентра (мм). Для массива возвращает первый элемент."""
        if isinstance(self._radius, list):
            return float(self._radius[0]) if self._radius else 250.0
        return float(self._radius)

    @radius.setter
    def radius(self, new_radius: Any) -> None:
        if isinstance(new_radius, (list, tuple)):
            self._radius = [float(rad_val) for rad_val in new_radius]
            eff_radius = float(self._radius[0]) if self._radius else 250.0
        else:
            self._radius = max(10.0, float(new_radius))
            eff_radius = self._radius

        if self._observed_cameras and not self._is_syncing_with_scene:
            self._is_syncing_with_scene = True
            try:
                for camera_view_model in self._observed_cameras:
                    camera_position = camera_view_model.local_matrix[0:3, 3]
                    camera_angle = float(np.degrees(np.arctan2(camera_position[1], camera_position[0])) % 360.0)
                    camera_view_model.local_matrix = compute_orbit_matrix(
                        radius=eff_radius,
                        angle=camera_angle,
                        z=float(camera_position[2]),
                        half_thickness=camera_view_model.half_thickness,
                    )
            finally:
                self._is_syncing_with_scene = False
        self._update_standard_variables(regenerate_angles=False)
        self.changed.emit()
        self.parameter_changed.emit("radius", self._radius)

    @property
    def radius_raw(self) -> Any:
        """Исходное значение радиуса (число или массив чисел)."""
        return self._radius

    @property
    def time_per_step(self) -> float:
        """Время экспозиции одного шага (секунды)."""
        return self._time_per_step

    @time_per_step.setter
    def time_per_step(self, new_time: float) -> None:
        validated_time = max(0.001, float(new_time))
        if self._time_per_step != validated_time:
            self._time_per_step = validated_time
            self._update_standard_variables(regenerate_angles=False)
            self.changed.emit()
            self.parameter_changed.emit("time_per_step", validated_time)

    @property
    def initial_time(self) -> float:
        """Начальное время смещения сканирования (секунды)."""
        return self._initial_time

    @initial_time.setter
    def initial_time(self, new_initial_time: float) -> None:
        validated_initial = max(0.0, float(new_initial_time))
        if self._initial_time != validated_initial:
            self._initial_time = validated_initial
            self._update_standard_variables(regenerate_angles=False)
            self.changed.emit()
            self.parameter_changed.emit("initial_time", validated_initial)

    @property
    def sequential_time(self) -> bool:
        """Флаг последовательного хронологического отсчета времени по шагам."""
        return self._sequential_time

    @sequential_time.setter
    def sequential_time(self, new_sequential_flag: bool) -> None:
        validated_flag = bool(new_sequential_flag)
        if self._sequential_time != validated_flag:
            self._sequential_time = validated_flag
            self._update_standard_variables(regenerate_angles=False)
            self.changed.emit()
            self.parameter_changed.emit("sequential_time", validated_flag)

    @property
    def start_angle(self) -> float:
        """Начальный угол поворота гантри (градусы)."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, new_angle: float) -> None:
        angle_value = float(new_angle)
        if self._start_angle != angle_value:
            self._start_angle = angle_value
            self._update_standard_variables()
            if self._observed_gantry is not None and not self._is_syncing_with_scene:
                self._is_syncing_with_scene = True
                try:
                    self._observed_gantry.gantry_angle = angle_value
                finally:
                    self._is_syncing_with_scene = False
            elif self._observed_cameras and not self._is_syncing_with_scene:
                self._is_syncing_with_scene = True
                try:
                    self.sync_cameras(self._observed_cameras)
                finally:
                    self._is_syncing_with_scene = False
            self.changed.emit()
            self.parameter_changed.emit("start_angle", angle_value)


    @property
    def end_angle(self) -> float:
        """Конечный угол поворота гантри (градусы)."""
        return self._end_angle

    @end_angle.setter
    def end_angle(self, new_angle: float) -> None:
        angle_value = float(new_angle)
        if self._end_angle != angle_value:
            self._end_angle = angle_value
            self._update_standard_variables()
            self.changed.emit()
            self.parameter_changed.emit("end_angle", angle_value)

    @property
    def head_mode(self) -> str:
        """Режим угловой конфигурации детекторных головок."""
        return self._head_mode

    @head_mode.setter
    def head_mode(self, new_mode: str) -> None:
        if self._head_mode != new_mode:
            self._head_mode = str(new_mode)
            self._recalculate_head_angles()
            if self._observed_cameras and not self._is_syncing_with_scene and len(self._observed_cameras) == len(self._head_angles):
                self._is_syncing_with_scene = True
                try:
                    self.sync_cameras(self._observed_cameras)
                finally:
                    self._is_syncing_with_scene = False
            self.changed.emit()
            self.parameter_changed.emit("head_mode", self._head_mode)

    @property
    def head_angles(self) -> List[float]:
        """Список угловых смещений для каждой детекторной головки (градусы)."""
        return list(self._head_angles)

    @head_angles.setter
    def head_angles(self, new_head_angles: List[float]) -> None:
        self._head_angles = [float(angle_item) for angle_item in new_head_angles]
        self._gamma_cameras = max(1, len(self._head_angles))
        if self._observed_cameras and not self._is_syncing_with_scene and len(self._observed_cameras) == len(self._head_angles):
            self._is_syncing_with_scene = True
            try:
                self.sync_cameras(self._observed_cameras)
            finally:
                self._is_syncing_with_scene = False
            self.changed.emit()
        self.parameter_changed.emit("head_angles", self._head_angles)

    @property
    def endpoint(self) -> bool:
        """Включать ли конечный угол end_angle в траекторию."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, new_endpoint: bool) -> None:
        endpoint_flag = bool(new_endpoint)
        if self._endpoint != endpoint_flag:
            self._endpoint = endpoint_flag
            self._update_standard_variables()
            self.changed.emit()
            self.parameter_changed.emit("endpoint", endpoint_flag)

    @property
    def angular_range(self) -> float:
        """Диапазон углов сканирования в градусах."""
        return self._end_angle - self._start_angle

    @angular_range.setter
    def angular_range(self, angular_span: float) -> None:
        self._end_angle = self._start_angle + float(angular_span)
        self.changed.emit()
        self.parameter_changed.emit("angular_range", angular_span)

    def sync_cameras(self, camera_vms: List[Any]) -> None:
        """
        Прямая синхронизация списка детекторных камер со свойствами процедуры.
        """
        for camera_index, camera_view_model in enumerate(camera_vms):
            angle_offset = self._head_angles[camera_index] if camera_index < len(self._head_angles) else (360.0 / max(1, len(camera_vms)) * camera_index)
            axial_position_z = float(camera_view_model.local_matrix[2, 3]) if camera_view_model.local_matrix is not None else 0.0
            orbit_angle = (angle_offset % 360.0) if isinstance(camera_view_model.parent_vm, GantryViewModel) else ((self._start_angle + angle_offset) % 360.0)

            roll_angle = 0.0
            if camera_view_model.local_matrix is not None:
                current_dir_z = camera_view_model.local_matrix[0:3, 2]
                current_dir_y = camera_view_model.local_matrix[0:3, 1]
                cos_roll = float(np.clip(np.dot(current_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
                sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), current_dir_y), current_dir_z))
                roll_angle = float(np.degrees(np.arctan2(sin_roll, cos_roll)))

            camera_view_model.local_matrix = compute_orbit_matrix(
                radius=self._radius,
                angle=orbit_angle,
                z=axial_position_z,
                half_thickness=camera_view_model.half_thickness,
                roll=roll_angle,
            )

    def remove_variable(self, var_name: str) -> None:
        """Удаляет переменную, запрещая удаление стандартных переменных протокола Step-and-Shoot."""
        clean_name = str(var_name).strip()
        if self.is_standard_variable(clean_name):
            return
        super().remove_variable(clean_name)

    def set_variable(
        self,
        var_name: str,
        values: List[float],
        sweep_type: str = "grid",
    ) -> None:
        """
        Добавляет или обновляет переменную. Стандартные переменные протокола Step-and-Shoot
        (gantry_angle, radius, step_time) всегда остаются zipped и синхронизируют свойства.
        """
        clean_name = str(var_name).strip().replace(" ", "_")
        float_values = [float(val) for val in values]
        if self.is_standard_variable(clean_name):
            sweep_type = "zipped"

        super().set_variable(clean_name, float_values, sweep_type=sweep_type)

        if clean_name == "gantry_angle":
            self._zipped_variables["current_angle"] = list(float_values)
            self._steps = len(float_values)
            self.parameter_changed.emit("steps", self._steps)
        elif clean_name == "current_angle":
            self._zipped_variables["gantry_angle"] = list(float_values)
            self._steps = len(float_values)
            self.parameter_changed.emit("steps", self._steps)
        elif clean_name == "step_time":
            self._zipped_variables["current_time"] = list(float_values)
            if float_values:
                self._time_per_step = float_values[0]
        elif clean_name == "current_time":
            self._zipped_variables["step_time"] = list(float_values)
            if float_values:
                self._time_per_step = float_values[0]
        elif clean_name == "radius":
            self._radius = list(float_values)

    def add_zipped_step(self, step_values: Optional[Dict[str, float]] = None) -> None:
        values_dict = dict(step_values) if step_values is not None else {}
        if "gantry_angle" in values_dict and "current_angle" not in values_dict:
            values_dict["current_angle"] = values_dict["gantry_angle"]
        elif "current_angle" in values_dict and "gantry_angle" not in values_dict:
            values_dict["gantry_angle"] = values_dict["current_angle"]

        if "step_time" in values_dict and "current_time" not in values_dict:
            values_dict["current_time"] = values_dict["step_time"]
        elif "current_time" in values_dict and "step_time" not in values_dict:
            values_dict["step_time"] = values_dict["current_time"]

        super().add_zipped_step(values_dict)
        if self._zipped_variables:
            self._steps = len(next(iter(self._zipped_variables.values())))
            if isinstance(self._radius, list):
                self._radius = list(self._zipped_variables.get("radius", self._radius))
            self.parameter_changed.emit("steps", self._steps)

    def remove_zipped_step(self, step_index: int = -1) -> None:
        super().remove_zipped_step(step_index)
        if self._zipped_variables:
            self._steps = len(next(iter(self._zipped_variables.values())))
            if isinstance(self._radius, list):
                self._radius = list(self._zipped_variables.get("radius", self._radius))
            self.parameter_changed.emit("steps", self._steps)

    def to_config(self) -> StepAndShootProtocolConfig:
        """Конвертация в модель конфигурации протокола пошагового сканирования StepAndShoot ядра."""
        radius_cfg: Optional[Union[float, List[float]]] = None
        if self._radius is not None:
            if isinstance(self._radius, list):
                radius_cfg = [float(rad_val) * units.mm for rad_val in self._radius]
            else:
                radius_cfg = float(self._radius) * units.mm

        extra_zipped = {
            var_name: var_vals for var_name, var_vals in self._zipped_variables.items()
            if var_name not in (
                "gantry_angle",
                "current_angle",
                "start_time",
                "stop_time",
                "step_time",
                "current_time",
                "radius",
            )
        }

        return StepAndShootProtocolConfig(
            type="StepAndShoot",
            steps=self._steps,
            start_angle=float(np.radians(self._start_angle)) * units.rad,
            end_angle=float(np.radians(self._end_angle)) * units.rad,
            endpoint=self._endpoint,
            time_per_step=float(self._time_per_step) * units.s,
            initial_time=float(self._initial_time) * units.s,
            sequential_time=self._sequential_time,
            radius=radius_cfg,
            grid_variables=dict(self._grid_variables),
            zipped_variables=extra_zipped,
        )

    @classmethod
    def from_config(cls, config: StepAndShootProtocolConfig, parent: Optional[QObject] = None) -> 'StepAndShootProcedureViewModel':
        """Восстановление модели представления из конфигурации StepAndShootProtocolConfig."""
        radius_val: Any = None
        if config.radius is not None:
            if isinstance(config.radius, list):
                radius_val = [float(rad_val) / float(units.mm) for rad_val in config.radius]
            else:
                radius_val = float(config.radius) / float(units.mm)

        return cls(
            steps=config.steps,
            radius=radius_val if radius_val is not None else 250.0,
            time_per_step=float(config.time_per_step) / float(units.s),
            initial_time=float(config.initial_time) / float(units.s),
            sequential_time=bool(config.sequential_time),
            start_angle=float(np.degrees(config.start_angle)),
            end_angle=float(np.degrees(config.end_angle)),
            endpoint=config.endpoint,
            grid_variables=dict(config.grid_variables or {}),
            zipped_variables=dict(config.zipped_variables or {}),
            parent=parent,
        )

    def sync_with_scene(self, scene_view_model: Any) -> None:
        """
        Реактивная синхронизация количества и пространственного расположения гамма-камер в сцене.
        Процедура выступает ведущей: монтирует детекторы внутрь GantryViewModel,
        создает недостающие или удаляет избыточные камеры, задает радиус и начальный угол поворота.
        Также устанавливает двустороннее наблюдение (Observer) за перемещением станины и кареток камер.
        """
        self._disconnect_scene_observers()

        if scene_view_model is None or scene_view_model.root_vm is None:
            return

        all_nodes = scene_view_model.all_nodes()
        camera_view_models: List[GammaCameraViewModel] = [
            camera_node for camera_node in all_nodes
            if isinstance(camera_node, GammaCameraViewModel)
        ]

        # Если в сцене отсутствуют гамма-камеры, процедура не создает их самовольно
        if not camera_view_models:
            return

        # Поиск или создание узла станины GantryViewModel
        gantry_view_models: List[GantryViewModel] = [
            node for node in all_nodes
            if isinstance(node, GantryViewModel)
        ]
        if gantry_view_models:
            gantry_view_model = gantry_view_models[0]
        else:
            gantry_core = GantryNode(name="Gantry")
            gantry_view_model = GantryViewModel(gantry_core)
            scene_view_model.add_node(scene_view_model.root_vm, gantry_view_model)

        # Привязка ссылки на процедуру в узле станины для кинематических ограничений
        self._observed_scene = scene_view_model
        scene_view_model.node_removed.connect(self._on_scene_node_removed)
        gantry_view_model.procedure_vm = self
        self._observed_gantry = gantry_view_model
        gantry_view_model.property_changed.connect(self._on_gantry_property_changed)
        gantry_view_model.child_added.connect(self._on_gantry_child_added)
        gantry_view_model.child_removed.connect(self._on_gantry_child_removed)

        # Монтирование всех существующих камер внутрь gantry_view_model
        for existing_camera_view_model in camera_view_models:
            if existing_camera_view_model.parent_vm is not gantry_view_model:
                gantry_view_model.add_child(existing_camera_view_model)

        had_created_cameras = False
        needed_camera_count = self._gamma_cameras
        if len(camera_view_models) < needed_camera_count:
            had_created_cameras = True
            for new_camera_index in range(len(camera_view_models), needed_camera_count):
                camera_name = f"GammaCamera_{new_camera_index + 1}"
                camera_core, slots_cfg = create_default_gamma_camera(name=camera_name)
                created_camera_view_model = GammaCameraViewModel(camera_core, slots=slots_cfg)
                scene_view_model.slots_registry[camera_core] = slots_cfg
                scene_view_model.add_node(gantry_view_model, created_camera_view_model)
                camera_view_models.append(created_camera_view_model)
        elif len(camera_view_models) > needed_camera_count:
            for extra_camera_view_model in camera_view_models[needed_camera_count:]:
                scene_view_model.remove_node(extra_camera_view_model)
            camera_view_models = camera_view_models[:needed_camera_count]

        # Подключение слушателей изменений параметров для всех активных камер
        for camera_view_model in camera_view_models:
            if camera_view_model not in self._observed_cameras:
                camera_view_model.property_changed.connect(self._on_camera_property_changed)
                self._observed_cameras.append(camera_view_model)

        # 2. Обновление угла станины и локальных параметров камер с подавлением эхо-сигналов
        self._is_syncing_with_scene = True
        try:
            if had_created_cameras:
                gantry_view_model.gantry_angle = self._start_angle
                self.sync_cameras(camera_view_models)
            elif camera_view_models:
                # Камеры уже присутствуют в сцене: считываем радиус и локальные углы из сцены без мутации матриц
                first_camera_vm = camera_view_models[0]
                camera_position = first_camera_vm.local_matrix[0:3, 3]
                distance_to_isocenter = float(np.hypot(camera_position[0], camera_position[1]))
                if distance_to_isocenter > 1.0:
                    self._radius = max(10.0, distance_to_isocenter - first_camera_vm.half_thickness)

                extracted_head_angles: List[float] = []
                for cam_vm in camera_view_models:
                    cam_pos = cam_vm.local_matrix[0:3, 3]
                    if float(np.hypot(cam_pos[0], cam_pos[1])) > 1e-3:
                        cam_angle = float(np.degrees(np.arctan2(cam_pos[1], cam_pos[0])) % 360.0)
                    else:
                        cam_angle = 0.0
                    extracted_head_angles.append(cam_angle)
                self._head_angles = extracted_head_angles
                self._gamma_cameras = len(camera_view_models)

            # 3. Zero-Config авто-привязка станины в единый реестр привязок процедуры
            if gantry_view_model is not None and gantry_view_model.core_node is not None:
                self.bind_node_property(gantry_view_model.core_node, "rotation.alpha", "gantry_angle")
                self.bind_node_property(gantry_view_model.core_node, "gantry_angle", "gantry_angle")
                gantry_view_model.bindings.setdefault("gantry_angle", "gantry_angle")
                gantry_view_model.bindings.setdefault("rotation.alpha", "gantry_angle")
                if scene_view_model is not None:
                    gantry_bindings = scene_view_model.bindings_registry.setdefault(gantry_view_model.core_node, {})
                    gantry_bindings.setdefault("rotation.alpha", "gantry_angle")
                    gantry_bindings.setdefault("gantry_angle", "gantry_angle")
        finally:
            self._is_syncing_with_scene = False


    def get_kinematic_constraint_for_node(
        self,
        node_view_model: NodeViewModel
    ) -> Optional[IKinematicConstraint]:
        """
        Возвращает специализированное кинематическое ограничение:
        - опрашивает граф сцены на наличие эффективного ограничения узла
        - для GantryViewModel -> GantryKinematicConstraint (fallback)
        - для GammaCameraViewModel -> CameraMountKinematicConstraint (fallback)
        - для фантомов, стола и источников -> None (свободный 6-DOF)
        """
        effective_constraint = node_view_model.get_effective_kinematic_constraint()
        if effective_constraint is not None:
            return effective_constraint

        if isinstance(node_view_model, GantryViewModel):
            return GantryKinematicConstraint(procedure_vm=self, gantry_vm=node_view_model)
        if isinstance(node_view_model, GammaCameraViewModel):
            return CameraMountKinematicConstraint(procedure_vm=self, camera_vm=node_view_model)
        return None


# Алиас для обратной совместимости существующих контроллеров и вьюпортов
class SpectProcedureViewModel(StepAndShootProcedureViewModel):
    """
    Модель представления протокола ОФЭКТ (SPECT).
    Является специализацией StepAndShootProcedureViewModel для протоколов ОФЭКТ.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("procedure_type", "SPECT")
        kwargs.setdefault("name", "Протокол ОФЭКТ (SPECT)")
        super().__init__(*args, **kwargs)


class PetProcedureViewModel(BaseProcedureViewModel):
    """
    Модель представления протокола ПЭТ (позитронно-эмиссионная томография).
    Управляет радиусом кольца детекторов, числом модулей и временными кадрами.
    """

    def __init__(
        self,
        ring_radius: float = 400.0,
        detector_heads: int = 16,
        time_per_frame: float = 60.0,
        parent: Optional[QObject] = None
    ) -> None:
        super().__init__(name="Протокол ПЭТ (PET)", procedure_type="PET", parent=parent)
        self._ring_radius: float = max(50.0, float(ring_radius))
        self._detector_heads: int = max(2, int(detector_heads))
        self._time_per_frame: float = max(0.1, float(time_per_frame))
        self._start_time = 0.0
        self._stop_time = self._time_per_frame

    @property
    def ring_radius(self) -> float:
        return self._ring_radius

    @ring_radius.setter
    def ring_radius(self, radius_value: float) -> None:
        new_ring_radius = max(50.0, float(radius_value))
        if self._ring_radius != new_ring_radius:
            self._ring_radius = new_ring_radius
            self.changed.emit()
            self.parameter_changed.emit("ring_radius", new_ring_radius)

    @property
    def detector_heads(self) -> int:
        return self._detector_heads

    @detector_heads.setter
    def detector_heads(self, new_heads: int) -> None:
        heads_count = max(2, int(new_heads))
        if self._detector_heads != heads_count:
            self._detector_heads = heads_count
            self.changed.emit()
            self.parameter_changed.emit("detector_heads", heads_count)

    @property
    def time_per_frame(self) -> float:
        return self._time_per_frame

    @time_per_frame.setter
    def time_per_frame(self, new_time: float) -> None:
        frame_time_val = max(0.1, float(new_time))
        if self._time_per_frame != frame_time_val:
            self._time_per_frame = frame_time_val
            self._stop_time = frame_time_val
            self.changed.emit()
            self.parameter_changed.emit("time_per_frame", frame_time_val)

    def to_config(self) -> CustomSweepProtocolConfig:
        return CustomSweepProtocolConfig(
            type="CustomSweep",
            grid_variables={},
            zipped_variables={"frame_time": [float(self._time_per_frame)]}
        )





def create_procedure_viewmodel(procedure_type: str = "StepAndShoot") -> BaseProcedureViewModel:
    """
    Фабрика создания моделей представления процедур по их идентификатору типа.
    """
    procedure_type_normalized = procedure_type.upper().replace(" ", "").replace("_", "").strip()
    if procedure_type_normalized in ("SPECT", "ОФЭКТ"):
        return SpectProcedureViewModel()
    elif procedure_type_normalized in ("STEPANDSHOOT", "STEP", "STEPANDSHOOTPROTOCOL"):
        return StepAndShootProcedureViewModel()
    elif procedure_type_normalized in ("PET", "ПЭТ"):
        return PetProcedureViewModel()
    elif procedure_type_normalized in ("CUSTOMSWEEP", "CUSTOM", "ПОЛЬЗОВАТЕЛЬСКИЙ", "ПОЛЬЗОВАТЕЛЬСКАЯ"):
        return CustomSweepProcedureViewModel()
    else:
        raise ValueError(f"Неизвестный тип процедуры: {procedure_type}")


def procedure_from_config(config: AnyProtocolConfig, parent: Optional[QObject] = None) -> BaseProcedureViewModel:
    """
    Восстановление ViewModel процедуры из конфигурационного объекта ядра.
    """
    if isinstance(config, StepAndShootProtocolConfig):
        return SpectProcedureViewModel.from_config(config, parent=parent)
    elif isinstance(config, CustomSweepProtocolConfig):
        return CustomSweepProcedureViewModel(
            grid_variables=config.grid_variables,
            zipped_variables=config.zipped_variables,
            parent=parent,
        )
    raise ValueError(f"Неподдерживаемый тип конфигурации протокола: {type(config)}")


__all__ = [
    "BaseProcedureViewModel",
    "CustomSweepProcedureViewModel",
    "StepAndShootProcedureViewModel",
    "SpectProcedureViewModel",
    "PetProcedureViewModel",
    "create_procedure_viewmodel",
    "procedure_from_config",
]
