import logging
from typing import Any, Dict, List, Optional
import numpy as np
import hepunits as units
from PySide6.QtCore import QObject, Signal

from core.config.models import (
    AnyProtocolConfig,
    SpectProtocolConfig,
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

    def __init__(self, name: str = "Procedure", procedure_type: str = "Base", parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._name: str = name
        self._procedure_type: str = procedure_type

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


class SpectProcedureViewModel(BaseProcedureViewModel):
    """
    Модель представления протокола ОФЭКТ (SPECT).
    Управляет числом ракурсов, количеством детекторных головок, радиусом вращения,
    временем экспозиции и геометрической расстановкой гамма-камер в графе сцены.
    """

    def __init__(
        self,
        steps: int = 32,
        gamma_cameras: int = 2,
        radius: float = 250.0,
        time_per_view: float = 1.0,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        head_mode: str = "Симметричный (360°/N)",
        head_angles: Optional[List[float]] = None,
        endpoint: bool = False,
        parent: Optional[QObject] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="Протокол ОФЭКТ (SPECT)", procedure_type="SPECT", parent=parent)
        self._steps: int = max(1, int(steps))
        self._gamma_cameras: int = max(1, int(gamma_cameras))
        self._radius: float = max(10.0, float(radius))
        self._time_per_view: float = max(0.001, float(time_per_view))
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
        if property_name == "gantry_angle_deg":
            angle_degrees = float(property_value)
            if abs(self._start_angle - angle_degrees) > 1e-4:
                self._is_syncing_with_scene = True
                try:
                    self._start_angle = angle_degrees
                    self.changed.emit()
                    self.parameter_changed.emit("start_angle", angle_degrees)
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
                                angle_deg=other_angle,
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
        """Радиус орбиты вращения детекторов вокруг изоцентра (мм)."""
        return self._radius

    @radius.setter
    def radius(self, new_radius: float) -> None:
        validated_radius = max(10.0, float(new_radius))
        if self._radius != validated_radius:
            self._radius = validated_radius
            if self._observed_cameras and not self._is_syncing_with_scene:
                self._is_syncing_with_scene = True
                try:
                    for camera_view_model in self._observed_cameras:
                        pos = camera_view_model.local_matrix[0:3, 3]
                        angle = float(np.degrees(np.arctan2(pos[1], pos[0])) % 360.0)
                        camera_view_model.local_matrix = compute_orbit_matrix(
                            radius=validated_radius,
                            angle_deg=angle,
                            z=float(pos[2]),
                            half_thickness=camera_view_model.half_thickness,
                        )
                finally:
                    self._is_syncing_with_scene = False
            self.changed.emit()
            self.parameter_changed.emit("radius", validated_radius)

    @property
    def time_per_view(self) -> float:
        """Время экспозиции одного ракурса (секунды)."""
        return self._time_per_view

    @time_per_view.setter
    def time_per_view(self, new_time: float) -> None:
        validated_time = max(0.001, float(new_time))
        if self._time_per_view != validated_time:
            self._time_per_view = validated_time
            self.changed.emit()
            self.parameter_changed.emit("time_per_view", validated_time)

    @property
    def start_angle(self) -> float:
        """Начальный угол поворота гантри (градусы)."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, new_angle: float) -> None:
        angle_value = float(new_angle)
        if self._start_angle != angle_value:
            self._start_angle = angle_value
            if self._observed_gantry is not None and not self._is_syncing_with_scene:
                self._is_syncing_with_scene = True
                try:
                    self._observed_gantry.gantry_angle_deg = angle_value
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
        self._head_angles = [float(angle_deg) for angle_deg in new_head_angles]
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
            self.changed.emit()
            self.parameter_changed.emit("endpoint", endpoint_flag)

    @property
    def angular_range(self) -> float:
        """Диапазон углов сканирования в градусах."""
        return self._end_angle - self._start_angle

    @angular_range.setter
    def angular_range(self, range_degrees: float) -> None:
        self._end_angle = self._start_angle + float(range_degrees)
        self.changed.emit()
        self.parameter_changed.emit("angular_range", range_degrees)

    def sync_cameras(self, camera_vms: List[Any]) -> None:
        """
        Прямая синхронизация списка детекторных камер со свойствами процедуры.
        """
        for camera_index, camera_view_model in enumerate(camera_vms):
            angle_offset = self._head_angles[camera_index] if camera_index < len(self._head_angles) else (360.0 / max(1, len(camera_vms)) * camera_index)
            axial_position_z = float(camera_view_model.local_matrix[2, 3]) if camera_view_model.local_matrix is not None else 0.0
            orbit_angle = (angle_offset % 360.0) if isinstance(camera_view_model.parent_vm, GantryViewModel) else ((self._start_angle + angle_offset) % 360.0)

            roll_angle_deg = 0.0
            if camera_view_model.local_matrix is not None:
                current_dir_z = camera_view_model.local_matrix[0:3, 2]
                current_dir_y = camera_view_model.local_matrix[0:3, 1]
                cos_roll = float(np.clip(np.dot(current_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
                sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), current_dir_y), current_dir_z))
                roll_angle_deg = float(np.degrees(np.arctan2(sin_roll, cos_roll)))

            camera_view_model.local_matrix = compute_orbit_matrix(
                radius=self._radius,
                angle_deg=orbit_angle,
                z=axial_position_z,
                half_thickness=camera_view_model.half_thickness,
                roll_deg=roll_angle_deg,
            )


    def to_config(self) -> SpectProtocolConfig:
        """Конвертация в модель конфигурации протокола ОФЭКТ ядра."""
        head_angles_rad = [float(np.radians(angle_item)) * units.rad for angle_item in self._head_angles] if self._head_angles else None
        return SpectProtocolConfig(
            type="SPECT",
            views=self.total_projections,
            gamma_cameras=self._gamma_cameras,
            start_angle=float(np.radians(self._start_angle)) * units.rad,
            end_angle=float(np.radians(self._end_angle)) * units.rad,
            time_per_view=float(self._time_per_view) * units.s,
            radius=float(self._radius) * units.mm,
            head_angles=head_angles_rad,
            endpoint=self._endpoint,
        )

    @classmethod
    def from_config(cls, config: SpectProtocolConfig, parent: Optional[QObject] = None) -> 'SpectProcedureViewModel':
        """Восстановление модели представления из конфигурации SpectProtocolConfig."""
        head_angles = [float(np.degrees(angle_item)) for angle_item in config.head_angles] if config.head_angles else None
        radius = float(config.radius) if config.radius is not None else 250.0
        steps = max(1, config.views // config.gamma_cameras) if config.gamma_cameras > 0 else config.views
        return cls(
            steps=steps,
            gamma_cameras=config.gamma_cameras,
            radius=radius,
            time_per_view=float(config.time_per_view) / float(units.s),
            start_angle=float(np.degrees(config.start_angle)),
            end_angle=float(np.degrees(config.end_angle)),
            head_angles=head_angles,
            endpoint=config.endpoint,
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
                gantry_view_model.gantry_angle_deg = self._start_angle
                self.sync_cameras(camera_view_models)
            elif camera_view_models:
                # Камеры уже присутствуют в сцене: считываем радиус из сцены без мутации матриц
                first_cam = camera_view_models[0]
                pos = first_cam.local_matrix[0:3, 3]
                dist = float(np.hypot(pos[0], pos[1]))
                if dist > 1.0:
                    self._radius = max(10.0, dist - first_cam.half_thickness)
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
            self.changed.emit()
            self.parameter_changed.emit("time_per_frame", frame_time_val)

    def to_config(self) -> CustomSweepProtocolConfig:
        return CustomSweepProtocolConfig(
            type="CustomSweep",
            grid_variables={},
            zipped_variables={"frame_time": [float(self._time_per_frame)]}
        )


class CustomSweepProcedureViewModel(BaseProcedureViewModel):
    """
    Модель представления пользовательского протокола сканирования (параметрический скан).
    """

    def __init__(
        self,
        grid_variables: Optional[Dict[str, List[float]]] = None,
        zipped_variables: Optional[Dict[str, List[float]]] = None,
        parent: Optional[QObject] = None
    ) -> None:
        super().__init__(name="Пользовательский скан (Custom Sweep)", procedure_type="CustomSweep", parent=parent)
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

    def to_config(self) -> CustomSweepProtocolConfig:
        return CustomSweepProtocolConfig(
            type="CustomSweep",
            grid_variables=self._grid_variables,
            zipped_variables=self._zipped_variables
        )


def create_procedure_viewmodel(procedure_type: str = "SPECT") -> BaseProcedureViewModel:
    """
    Фабрика создания моделей представления процедур по их идентификатору типа.
    """
    procedure_type_normalized = procedure_type.upper().replace(" ", "").replace("_", "").strip()
    if procedure_type_normalized in ("SPECT", "ОФЭКТ"):
        return SpectProcedureViewModel()
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
    if isinstance(config, SpectProtocolConfig):
        return SpectProcedureViewModel.from_config(config, parent=parent)
    elif isinstance(config, CustomSweepProtocolConfig):
        return CustomSweepProcedureViewModel(
            grid_variables=config.grid_variables,
            zipped_variables=config.zipped_variables,
            parent=parent,
        )
    raise ValueError(f"Неподдерживаемый тип конфигурации протокола: {type(config)}")
