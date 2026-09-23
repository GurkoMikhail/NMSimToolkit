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
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
import settings.database_setting as database_setting
from gui.viewmodels.node_viewmodel import GammaCameraViewModel, NodeViewModel

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


class SpectProcedureViewModel(BaseProcedureViewModel):
    """
    Модель представления протокола ОФЭКТ (SPECT).
    Управляет числом ракурсов, количеством детекторных головок, радиусом вращения,
    временем экспозиции и геометрической расстановкой гамма-камер в графе сцены.
    """

    def __init__(
        self,
        views: int = 32,
        gamma_cameras: int = 2,
        radius: float = 250.0,
        time_per_view: float = 1.0,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        head_mode: str = "Симметричный (360°/N)",
        head_angles: Optional[List[float]] = None,
        endpoint: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(name="Протокол ОФЭКТ (SPECT)", procedure_type="SPECT", parent=parent)
        self._views: int = max(1, int(views))
        self._gamma_cameras: int = max(1, int(gamma_cameras))
        self._radius: float = max(10.0, float(radius))
        self._time_per_view: float = max(0.001, float(time_per_view))
        self._start_angle: float = float(start_angle)
        self._end_angle: float = float(end_angle)
        self._head_mode: str = head_mode
        self._head_angles: List[float] = list(head_angles) if head_angles is not None else []
        self._endpoint: bool = bool(endpoint)

        if not self._head_angles:
            self._recalculate_head_angles()

    def _recalculate_head_angles(self) -> None:
        """Автоматический пересчет смещений головок в зависимости от выбранного режима."""
        if "Симметричный" in self._head_mode or "symmetric" in self._head_mode.lower():
            step = 360.0 / self._gamma_cameras if self._gamma_cameras > 0 else 0.0
            self._head_angles = [step * i for i in range(self._gamma_cameras)]
        elif "90" in self._head_mode or "L-режим" in self._head_mode or "l-mode" in self._head_mode.lower():
            self._gamma_cameras = 2
            self._head_angles = [0.0, 90.0]

    @property
    def views(self) -> int:
        """Общее количество ракурсов сканирования."""
        return self._views

    @views.setter
    def views(self, val: int) -> None:
        v = max(1, int(val))
        if self._views != v:
            self._views = v
            self.changed.emit()
            self.parameter_changed.emit("views", v)

    @property
    def gamma_cameras(self) -> int:
        """Количество детекторных головок гамма-камер."""
        return self._gamma_cameras

    @gamma_cameras.setter
    def gamma_cameras(self, val: int) -> None:
        c = max(1, int(val))
        if self._gamma_cameras != c:
            self._gamma_cameras = c
            self._recalculate_head_angles()
            self.changed.emit()
            self.parameter_changed.emit("gamma_cameras", c)

    @property
    def radius(self) -> float:
        """Радиус орбиты вращения детекторов вокруг изоцентра (мм)."""
        return self._radius

    @radius.setter
    def radius(self, val: float) -> None:
        r = max(10.0, float(val))
        if self._radius != r:
            self._radius = r
            self.changed.emit()
            self.parameter_changed.emit("radius", r)

    @property
    def time_per_view(self) -> float:
        """Время экспозиции одного ракурса (секунды)."""
        return self._time_per_view

    @time_per_view.setter
    def time_per_view(self, val: float) -> None:
        t = max(0.001, float(val))
        if self._time_per_view != t:
            self._time_per_view = t
            self.changed.emit()
            self.parameter_changed.emit("time_per_view", t)

    @property
    def start_angle(self) -> float:
        """Начальный угол поворота гантри (градусы)."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, val: float) -> None:
        a = float(val)
        if self._start_angle != a:
            self._start_angle = a
            self.changed.emit()
            self.parameter_changed.emit("start_angle", a)

    @property
    def end_angle(self) -> float:
        """Конечный угол поворота гантри (градусы)."""
        return self._end_angle

    @end_angle.setter
    def end_angle(self, val: float) -> None:
        a = float(val)
        if self._end_angle != a:
            self._end_angle = a
            self.changed.emit()
            self.parameter_changed.emit("end_angle", a)

    @property
    def head_mode(self) -> str:
        """Режим угловой конфигурации детекторных головок."""
        return self._head_mode

    @head_mode.setter
    def head_mode(self, val: str) -> None:
        if self._head_mode != val:
            self._head_mode = str(val)
            self._recalculate_head_angles()
            self.changed.emit()
            self.parameter_changed.emit("head_mode", self._head_mode)

    @property
    def head_angles(self) -> List[float]:
        """Список угловых смещений для каждой детекторной головки (градусы)."""
        return list(self._head_angles)

    @head_angles.setter
    def head_angles(self, val: List[float]) -> None:
        self._head_angles = [float(x) for x in val]
        self._gamma_cameras = max(1, len(self._head_angles))
        self.changed.emit()
        self.parameter_changed.emit("head_angles", self._head_angles)

    @property
    def endpoint(self) -> bool:
        """Включать ли конечный угол end_angle в траекторию."""
        return self._endpoint

    @endpoint.setter
    def endpoint(self, val: bool) -> None:
        b = bool(val)
        if self._endpoint != b:
            self._endpoint = b
            self.changed.emit()
            self.parameter_changed.emit("endpoint", b)

    @property
    def views_number(self) -> int:
        """Общее число ракурсов (псевдоним views)."""
        return self._views

    @views_number.setter
    def views_number(self, val: int) -> None:
        self.views = val

    @property
    def orbit_radius(self) -> float:
        """Радиус орбиты (псевдоним radius)."""
        return self._radius

    @orbit_radius.setter
    def orbit_radius(self, val: float) -> None:
        self.radius = val

    @property
    def angular_range(self) -> float:
        """Диапазон углов сканирования в градусах."""
        return self._end_angle - self._start_angle

    @angular_range.setter
    def angular_range(self, val: float) -> None:
        self._end_angle = self._start_angle + float(val)
        self.changed.emit()
        self.parameter_changed.emit("angular_range", val)

    def sync_cameras(self, camera_vms: List[Any]) -> None:
        """
        Прямая синхронизация списка детекторных камер со свойствами процедуры.
        """
        for i, cam_vm in enumerate(camera_vms):
            offset = self._head_angles[i] if i < len(self._head_angles) else (360.0 / max(1, len(camera_vms)) * i)
            cam_vm.set_orbit_position(self._radius, (self._start_angle + offset) % 360.0, cam_vm.orbit_z)


    def to_config(self) -> SpectProtocolConfig:
        """Конвертация в модель конфигурации протокола ОФЭКТ ядра."""
        head_angles_rad = [float(np.radians(a)) * units.rad for a in self._head_angles] if self._head_angles else None
        return SpectProtocolConfig(
            type="SPECT",
            views=self._views,
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
        head_angles = [float(np.degrees(a)) for a in config.head_angles] if config.head_angles else None
        radius = float(config.radius) if config.radius is not None else 250.0
        return cls(
            views=config.views,
            gamma_cameras=config.gamma_cameras,
            radius=radius,
            time_per_view=float(config.time_per_view) / float(units.s),
            start_angle=float(np.degrees(config.start_angle)),
            end_angle=float(np.degrees(config.end_angle)),
            head_angles=head_angles,
            endpoint=config.endpoint,
            parent=parent,
        )

    def sync_with_scene(self, scene_vm: Any) -> None:
        """
        Реактивная синхронизация количества и пространственного расположения гамма-камер в сцене.
        Процедура выступает ведущей: создает недостающие или удаляет избыточные камеры,
        а также задает радиус орбиты и угловые смещения.
        """
        if scene_vm is None or scene_vm.root_vm is None:
            return

        all_nodes = scene_vm.all_nodes()
        camera_vms: List[GammaCameraViewModel] = [n for n in all_nodes if isinstance(n, GammaCameraViewModel)]

        # Если в сцене отсутствуют гамма-камеры, процедура не создает их самовольно
        if not camera_vms:
            return

        # 1. Приведение количества камер к требуемому
        needed = self._gamma_cameras
        if len(camera_vms) < needed:
            lead_mat = database_setting.material_database.get('Pb', Material(name='Lead'))
            nai_mat = database_setting.material_database.get('Sodium Iodide', Material(name='NaI'))
            for i in range(len(camera_vms), needed):
                cam_name = f"GammaCamera_{i + 1}"
                col = Volume(geometry=Box(400.0, 400.0, 30.0), material=lead_mat, name=f"Collimator_{cam_name}")
                det = Volume(geometry=Box(400.0, 400.0, 10.0), material=nai_mat, name=f"Detector_{cam_name}")
                cam = GammaCamera(collimator=col, detector=det, name=cam_name)
                cam_vm = GammaCameraViewModel(cam)
                scene_vm.add_node(scene_vm.root_vm, cam_vm)
                camera_vms.append(cam_vm)
        elif len(camera_vms) > needed:
            for extra_vm in camera_vms[needed:]:
                scene_vm.remove_node(extra_vm)
            camera_vms = camera_vms[:needed]

        # 2. Обновление параметров орбит и позиционирование каждой камеры вокруг оси Z
        for idx, cam_vm in enumerate(camera_vms):
            offset_deg = self._head_angles[idx] if idx < len(self._head_angles) else (360.0 / needed) * idx
            angle = (self._start_angle + offset_deg) % 360.0
            cam_vm.set_orbit_position(self._radius, angle, cam_vm.orbit_z)


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
    def ring_radius(self, val: float) -> None:
        r = max(50.0, float(val))
        if self._ring_radius != r:
            self._ring_radius = r
            self.changed.emit()
            self.parameter_changed.emit("ring_radius", r)

    @property
    def detector_heads(self) -> int:
        return self._detector_heads

    @detector_heads.setter
    def detector_heads(self, val: int) -> None:
        h = max(2, int(val))
        if self._detector_heads != h:
            self._detector_heads = h
            self.changed.emit()
            self.parameter_changed.emit("detector_heads", h)

    @property
    def time_per_frame(self) -> float:
        return self._time_per_frame

    @time_per_frame.setter
    def time_per_frame(self, val: float) -> None:
        t = max(0.1, float(val))
        if self._time_per_frame != t:
            self._time_per_frame = t
            self.changed.emit()
            self.parameter_changed.emit("time_per_frame", t)

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
    def grid_variables(self, val: Dict[str, List[float]]) -> None:
        self._grid_variables = dict(val)
        self.changed.emit()
        self.parameter_changed.emit("grid_variables", self._grid_variables)

    @property
    def zipped_variables(self) -> Dict[str, List[float]]:
        return dict(self._zipped_variables)

    @zipped_variables.setter
    def zipped_variables(self, val: Dict[str, List[float]]) -> None:
        self._zipped_variables = dict(val)
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
    pt = procedure_type.upper().replace(" ", "").replace("_", "").strip()
    if pt in ("SPECT", "ОФЭКТ"):
        return SpectProcedureViewModel()
    elif pt in ("PET", "ПЭТ"):
        return PetProcedureViewModel()
    elif pt in ("CUSTOMSWEEP", "CUSTOM", "ПОЛЬЗОВАТЕЛЬСКИЙ", "ПОЛЬЗОВАТЕЛЬСКАЯ"):
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
