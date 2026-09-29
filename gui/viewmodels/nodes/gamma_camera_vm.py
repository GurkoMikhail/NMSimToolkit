import logging
import math
from typing import Optional, Sequence
import numpy as np

from core.geometry.gamma_cameras import GammaCamera
from core.other.typing_definitions import Float
from gui.viewmodels.decorators import gui_field
from gui.viewmodels.nodes.base_node_vm import NodeViewModel
from gui.viewmodels.nodes.volume_vm import VolumeViewModel

_logger = logging.getLogger(__name__)


class GammaCameraViewModel(VolumeViewModel):
    """
    ViewModel для ОФЭКТ гамма-камеры с поддержкой параметров круговой орбиты.
    """
    orbit_radius = gui_field(default=250.0)
    orbit_angle = gui_field(default=0.0)
    orbit_z = gui_field(default=0.0)

    def __init__(self, core_node: GammaCamera, parent_vm: Optional[NodeViewModel] = None) -> None:
        super().__init__(core_node, parent_vm)
        self._sync_orbit_params_from_matrix()
        detector_viewmodel = self.detector_vm
        if detector_viewmodel is not None:
            detector_viewmodel.is_sensitive_detector = True

    @property
    def detector_vm(self) -> Optional[VolumeViewModel]:
        """
        ViewModel чувствительного объема детектора гамма-камеры.
        """
        if isinstance(self.core_node, GammaCamera):
            detector_core = self.core_node.detector
            stack = list(self.children)
            while stack:
                current_vm = stack.pop()
                if current_vm.core_node is detector_core and isinstance(current_vm, VolumeViewModel):
                    return current_vm
                stack.extend(current_vm.children)
        return None

    @property
    def collimator_vm(self) -> Optional[VolumeViewModel]:
        """
        ViewModel коллиматора гамма-камеры.
        """
        if isinstance(self.core_node, GammaCamera):
            collimator_core = self.core_node.collimator
            stack = list(self.children)
            while stack:
                current_vm = stack.pop()
                if current_vm.core_node is collimator_core and isinstance(current_vm, VolumeViewModel):
                    return current_vm
                stack.extend(current_vm.children)
        return None

    @property
    def gap(self) -> float:
        """Внутренний зазор между коллиматором и детектором (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.gap)
        return 1.0

    @gap.setter
    def gap(self, gap_value: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            numeric_gap = float(gap_value)
            self.core_node.gap = Float(numeric_gap)
            self.property_changed.emit('gap', numeric_gap)
            self.property_changed.emit('size', self.size)

    @property
    def shielding_thickness(self) -> float:
        """Толщина свинцовой защиты корпуса гамма-камеры (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.shielding_thickness)
        return 20.0

    @shielding_thickness.setter
    def shielding_thickness(self, thickness_value: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            numeric_thickness = float(thickness_value)
            self.core_node.shielding_thickness = Float(numeric_thickness)
            self.property_changed.emit('shielding_thickness', numeric_thickness)
            self.property_changed.emit('size', self.size)

    @property
    def glass_backend_thickness(self) -> float:
        """Толщина подложки оптического стекла (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.glass_backend_thickness)
        return 50.0

    @glass_backend_thickness.setter
    def glass_backend_thickness(self, thickness_value: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            numeric_thickness = float(thickness_value)
            self.core_node.glass_backend_thickness = Float(numeric_thickness)
            self.property_changed.emit('glass_backend_thickness', numeric_thickness)
            self.property_changed.emit('size', self.size)

    def _sync_orbit_params_from_matrix(self) -> None:
        """
        Синхронизирует параметры орбиты (orbit_radius, orbit_angle, orbit_z)
        из текущей матрицы local_matrix ядра.
        Радиус орбиты отсчитывается до лицевой поверхности гамма-камеры.
        """
        if self.core_node.local_matrix is not None:
            pos_x = float(self.core_node.local_matrix[0, 3])
            pos_y = float(self.core_node.local_matrix[1, 3])
            pos_z = float(self.core_node.local_matrix[2, 3])
            radius_xy = float(np.hypot(pos_x, pos_y))
            self.orbit_z = pos_z
            if radius_xy > 1e-4:
                self.orbit_radius = max(0.0, radius_xy - self.half_thickness)
                calculated_angle = float(np.degrees(np.arctan2(pos_y, pos_x)) % 360.0)
                if np.isclose(calculated_angle, 360.0) or np.isclose(calculated_angle, 0.0):
                    calculated_angle = 0.0
                elif np.isclose(calculated_angle, round(calculated_angle), atol=1e-5):
                    calculated_angle = float(round(calculated_angle, 5))
                self.orbit_angle = calculated_angle

    @NodeViewModel.local_matrix.setter
    def local_matrix(self, matrix: np.ndarray) -> None:
        NodeViewModel.local_matrix.fset(self, matrix)
        self._sync_orbit_params_from_matrix()

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, in_local: bool = False) -> None:
        super().translate(x=x, y=y, z=z, in_local=in_local)
        self._sync_orbit_params_from_matrix()

    def rotate(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        rotation_center: Sequence[float] = (0.0, 0.0, 0.0),
        in_local: bool = False
    ) -> None:
        super().rotate(alpha=alpha, beta=beta, gamma=gamma, rotation_center=rotation_center, in_local=in_local)
        self._sync_orbit_params_from_matrix()

    @property
    def half_thickness(self) -> float:
        """
        Половина толщины гамма-камеры вдоль оси Z (мм).
        Лицевая поверхность коллиматора/камеры находится на расстоянии half_thickness
        от геометрического центра камеры в направлении нормали (+Z, к центру орбиты).
        """
        camera_size = self.size
        return float(camera_size[2]) / 2.0 if len(camera_size) >= 3 and camera_size[2] > 0 else 0.0

    @staticmethod
    def compute_orbit_matrix(radius: float, angle_deg: float, z: float = 0.0, half_thickness: float = 0.0) -> np.ndarray:
        """
        Вычисляет кинематическую матрицу трансформации 4x4 для круговой орбиты ОФЭКТ,
        ориентирующую гамма-камеру к центру орбиты (0, 0, z).
        Делегирует физико-математический расчет классу ядра GammaCamera.
        """
        return GammaCamera.compute_orbit_matrix(radius=radius, angle_deg=angle_deg, z=z, half_thickness=half_thickness)

    def set_orbit_position(self, radius: float, angle_deg: float, z: float = 0.0, preserve_roll: bool = True) -> None:
        """
        Установка положения гамма-камеры на круговой орбите (ОФЭКТ манипулятор).
        Радиус орбиты задается до лицевой поверхности гамма-камеры.
        При preserve_roll=True сохраняет текущую ориентацию детектора в собственной плоскости (Landscape/Portrait).
        """
        self.orbit_radius = float(radius)
        self.orbit_angle = float(angle_deg % 360.0)
        self.orbit_z = float(z)
        new_matrix = self.compute_orbit_matrix(radius, angle_deg, z, half_thickness=self.half_thickness)
        if preserve_roll and self.local_matrix is not None:
            initial_dir_z = self.local_matrix[0:3, 2]
            initial_dir_y = self.local_matrix[0:3, 1]
            cos_roll = float(np.clip(np.dot(initial_dir_y, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
            sin_roll = float(np.dot(np.cross(np.array([0.0, 0.0, 1.0]), initial_dir_y), initial_dir_z))
            roll_angle_deg = math.degrees(math.atan2(sin_roll, cos_roll))
            if abs(roll_angle_deg) > 1e-3:
                roll_rad = math.radians(roll_angle_deg)
                roll_mat = np.array([
                    [math.cos(roll_rad), -math.sin(roll_rad), 0.0],
                    [math.sin(roll_rad), math.cos(roll_rad), 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                new_matrix = new_matrix.copy()
                new_matrix[0:3, 0:3] = new_matrix[0:3, 0:3] @ roll_mat
        self.local_matrix = new_matrix
