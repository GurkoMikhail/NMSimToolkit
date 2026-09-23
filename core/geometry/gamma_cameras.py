from typing import Optional

import numpy as np
from core.other.typing_definitions import Float
import hepunits as units

import settings.database_setting as database_setting
from core.geometry.geometries import Box
from core.geometry.volumes import Volume


class GammaCamera(Volume):

    def __init__(self, collimator: Volume, detector: Volume, gap: Float = Float(1 * units.mm), shielding_thickness: Float = Float(2 * units.cm), glass_backend_thickness: Float = Float(5 * units.cm), name: Optional[str] = None) -> None:
        detector_box_size = np.where(collimator.size > detector.size, collimator.size, detector.size)
        detector_box_size[2] = collimator.size[2] + gap + detector.size[2] + glass_backend_thickness
        material_database = database_setting.material_database
        detector_box = Volume(
            geometry=Box(*detector_box_size),
            material=material_database['Air, Dry (near sea level)'],
            name='Detector_box'
        )
        glass_backend_size = detector_box_size.copy()
        glass_backend_size[2] = glass_backend_thickness
        glass_backend = Volume(
            geometry=Box(*glass_backend_size),
            material=material_database['Glass, Borosilicate (Pyrex)'],
            name='Glass_backend'
        )
        super().__init__(
            geometry=Box(detector_box_size[0] + 2*shielding_thickness, detector_box_size[1] + 2*shielding_thickness, detector_box_size[2] + shielding_thickness),
            material=material_database['Pb'],
            name=name
        )
        detector_box.translate(z=shielding_thickness/2)
        detector_box.set_parent(self)
        collimator.translate(z=(detector_box_size[2]/2 - collimator.size[2]/2))
        detector_box.add_child(collimator)
        detector.translate(z=(detector_box_size[2]/2 - collimator.size[2] - detector.size[2]/2 - gap))
        detector_box.add_child(detector)
        glass_backend.translate(z=(glass_backend.size[2]/2 - detector_box_size[2]/2))
        detector_box.add_child(glass_backend)

        self._gap = Float(gap)
        self._shielding_thickness = Float(shielding_thickness)
        self._glass_backend_thickness = Float(glass_backend_thickness)

    @property
    def gap(self) -> Float:
        """Внутренний зазор между коллиматором и детектором (мм)."""
        return self._gap

    @gap.setter
    def gap(self, value: Float) -> None:
        self._gap = Float(value)
        self.rebuild_camera()

    @property
    def shielding_thickness(self) -> Float:
        """Толщина свинцовой защиты корпуса гамма-камеры (мм)."""
        return self._shielding_thickness

    @shielding_thickness.setter
    def shielding_thickness(self, value: Float) -> None:
        self._shielding_thickness = Float(value)
        self.rebuild_camera()

    @property
    def glass_backend_thickness(self) -> Float:
        """Толщина подложки оптического стекла (мм)."""
        return self._glass_backend_thickness

    @glass_backend_thickness.setter
    def glass_backend_thickness(self, value: Float) -> None:
        self._glass_backend_thickness = Float(value)
        self.rebuild_camera()

    @property
    def detector_box(self) -> Volume:
        return self.childs[0]

    @property
    def collimator(self) -> Volume:
        return self.detector_box.childs[0]

    @property
    def detector(self) -> Volume:
        return self.detector_box.childs[1]

    @property
    def glass_backend(self) -> Volume:
        return self.detector_box.childs[2]

    def rebuild_camera(
        self,
        collimator: Optional[Volume] = None,
        detector: Optional[Volume] = None,
        gap: Optional[Float] = None,
        shielding_thickness: Optional[Float] = None,
        glass_backend_thickness: Optional[Float] = None
    ) -> None:
        """
        Пересчитывает геометрические размеры корпуса и взаимное расположение компонентов гамма-камеры.
        """
        collimator_vol = collimator if collimator is not None else self.collimator
        detector_vol = detector if detector is not None else self.detector
        if gap is not None:
            self._gap = Float(gap)
        if shielding_thickness is not None:
            self._shielding_thickness = Float(shielding_thickness)
        if glass_backend_thickness is not None:
            self._glass_backend_thickness = Float(glass_backend_thickness)

        gap_val = self._gap
        shielding_val = self._shielding_thickness
        glass_backend_val = self._glass_backend_thickness

        det_box_size = np.where(collimator_vol.size > detector_vol.size, collimator_vol.size, detector_vol.size)
        det_box_size[2] = collimator_vol.size[2] + gap_val + detector_vol.size[2] + glass_backend_val

        self.size = np.array([det_box_size[0] + 2 * shielding_val, det_box_size[1] + 2 * shielding_val, det_box_size[2] + shielding_val], dtype=Float)

        det_box = self.detector_box
        det_box.size = det_box_size.copy()
        det_box.local_matrix = np.eye(4, dtype=Float)
        det_box.translate(z=shielding_val / 2.0)

        collimator_vol.local_matrix = np.eye(4, dtype=Float)
        collimator_vol.translate(z=(det_box_size[2] / 2.0 - collimator_vol.size[2] / 2.0))

        detector_vol.local_matrix = np.eye(4, dtype=Float)
        detector_vol.translate(z=(det_box_size[2] / 2.0 - collimator_vol.size[2] - detector_vol.size[2] / 2.0 - gap_val))

        glass = self.glass_backend
        glass_size = det_box_size.copy()
        glass_size[2] = glass_backend_val
        glass.size = glass_size.copy()
        glass.local_matrix = np.eye(4, dtype=Float)
        glass.translate(z=(glass.size[2] / 2.0 - det_box_size[2] / 2.0))

        self.invalidate_geometry()

    @property
    def half_thickness(self) -> Float:
        """
        Половина толщины гамма-камеры вдоль оси Z (мм).
        """
        return Float(self.size[2] / 2.0 if len(self.size) >= 3 and self.size[2] > 0 else 0.0)

    @staticmethod
    def compute_orbit_matrix(radius: float, angle_deg: float, z: float = 0.0, half_thickness: float = 0.0) -> np.ndarray:
        """
        Вычисляет кинематическую матрицу трансформации 4x4 для круговой орбиты ОФЭКТ,
        ориентирующую гамма-камеру к центру орбиты (0, 0, z).
        Лицевая поверхность гамма-камеры находится на заданном расстоянии radius от центра орбиты.
        При half_thickness > 0 геометрический центр камеры смещается на radius + half_thickness,
        что обеспечивает точный отсчет радиуса орбиты по лицевой поверхности гамма-камеры.
        Лицевая нормаль детектора (+Z) направлена к центру орбиты.
        Ось стола (+Y детектора) направлена вдоль глобальной оси Z.
        Поперечная ось (+X детектора) направлена тангенциально к орбите.
        """
        rad = np.radians(angle_deg)
        center_radius = float(radius) + float(half_thickness)
        x = center_radius * np.cos(rad)
        y = center_radius * np.sin(rad)
        mat = np.eye(4, dtype=Float)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z

        # Лицевая нормаль детектора (локальная ось +Z, столбец 2) смотрит в центр орбиты (0, 0, z)
        mat[0, 2] = -np.cos(rad)
        mat[1, 2] = -np.sin(rad)
        mat[2, 2] = 0.0

        # Осевая ось детектора (локальная ось +Y, столбец 1) направлена вдоль глобальной оси +Z (ось стола)
        mat[0, 1] = 0.0
        mat[1, 1] = 0.0
        mat[2, 1] = 1.0

        # Трансверсионная ось детектора (локальная ось +X, столбец 0) образует правую тройку: X = Y x Z
        mat[0, 0] = np.sin(rad)
        mat[1, 0] = -np.cos(rad)
        mat[2, 0] = 0.0

        return mat

    def set_orbit_position(self, radius: float, angle_deg: float, z: float = 0.0) -> None:
        """
        Установка положения гамма-камеры на круговой орбите.
        Радиус орбиты задается до лицевой поверхности гамма-камеры.
        """
        self.local_matrix = self.compute_orbit_matrix(radius, angle_deg, z, half_thickness=float(self.half_thickness))
        self.invalidate_geometry()

