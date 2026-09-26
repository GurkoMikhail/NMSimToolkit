import logging
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
    def gap(self, val: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            gap_val = float(val)
            self.core_node.gap = Float(gap_val)
            self.property_changed.emit('gap', gap_val)
            self.property_changed.emit('size', self.size)

    @property
    def shielding_thickness(self) -> float:
        """Толщина свинцовой защиты корпуса гамма-камеры (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.shielding_thickness)
        return 20.0

    @shielding_thickness.setter
    def shielding_thickness(self, val: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            shielding_val = float(val)
            self.core_node.shielding_thickness = Float(shielding_val)
            self.property_changed.emit('shielding_thickness', shielding_val)
            self.property_changed.emit('size', self.size)

    @property
    def glass_backend_thickness(self) -> float:
        """Толщина подложки оптического стекла (мм)."""
        if isinstance(self.core_node, GammaCamera):
            return float(self.core_node.glass_backend_thickness)
        return 50.0

    @glass_backend_thickness.setter
    def glass_backend_thickness(self, val: float) -> None:
        if isinstance(self.core_node, GammaCamera):
            glass_val = float(val)
            self.core_node.glass_backend_thickness = Float(glass_val)
            self.property_changed.emit('glass_backend_thickness', glass_val)
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

    def set_orbit_position(self, radius: float, angle_deg: float, z: float = 0.0) -> None:
        """
        Установка положения гамма-камеры на круговой орбите (ОФЭКТ манипулятор).
        Радиус орбиты задается до лицевой поверхности гамма-камеры.
        """
        self.orbit_radius = float(radius)
        self.orbit_angle = float(angle_deg % 360.0)
        self.orbit_z = float(z)
        self.local_matrix = self.compute_orbit_matrix(radius, angle_deg, z, half_thickness=self.half_thickness)
