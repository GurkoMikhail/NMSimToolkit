"""
Фабрика декларативного создания узлов гамма-камеры и их моделей представления.
Формирует стандартную 5-узловую иерархию сборки со слотами:
GammaCameraNode -> Casing -> Detector_box -> (Collimator, Crystal, Glass_backend).
"""

from typing import Optional, Sequence, Tuple
import numpy as np

import settings.database_setting as database_setting
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.materials.materials import Material
from core.other.typing_definitions import Float
from core.scene.gamma_camera_node import GammaCameraNode
from core.config.models import GammaCameraSlotsConfig


def create_default_gamma_camera(
    name: Optional[str] = None,
    detector_size: Sequence[float] = (400.0, 400.0),
    detector_thickness: float = 10.0,
    collimator_thickness: float = 30.0,
    gap: float = 1.0,
    shielding_thickness: float = 20.0,
    glass_backend_thickness: float = 50.0,
    collimator_material_name: str = "Pb",
    detector_material_name: str = "Sodium Iodide",
    casing_material_name: str = "Pb",
    internal_material_name: str = "Air, Dry (near sea level)",
    glass_material_name: str = "Glass, Borosilicate (Pyrex)",
    collimator: Optional[Volume] = None,
    detector: Optional[Volume] = None,
    shielding_material: Optional[Material] = None,
    internal_medium: Optional[Material] = None,
    glass_material: Optional[Material] = None,
) -> Tuple[GammaCameraNode, GammaCameraSlotsConfig]:
    """
    Создает декларативный граф узлов стандартной гамма-камеры со слотами.
    Возвращает корневой узел GammaCameraNode и конфигурацию привязки слотов GammaCameraSlotsConfig.
    """
    effective_name = name or "GammaCamera"

    # Разрешение размеров детектора и коллиматора
    if detector is not None:
        det_size_x = float(detector.size[0])
        det_size_y = float(detector.size[1])
        eff_detector_thickness = float(detector.size[2])
    else:
        if len(detector_size) != 2:
            raise ValueError(f"detector_size должен содержать ровно 2 элемента [Lx, Ly], получено {len(detector_size)}")
        det_size_x = float(detector_size[0])
        det_size_y = float(detector_size[1])
        eff_detector_thickness = float(detector_thickness)

    if collimator is not None:
        eff_collimator_thickness = float(collimator.size[2])
    else:
        eff_collimator_thickness = float(collimator_thickness)

    # Валидация инвариантов (DbC / LBYL)
    if det_size_x <= 0 or det_size_y <= 0:
        raise ValueError(f"Размеры детектора должны быть строго положительными: ({det_size_x}, {det_size_y})")
    if eff_detector_thickness <= 0:
        raise ValueError(f"Толщина детектора должна быть строго положительной: {eff_detector_thickness}")
    if eff_collimator_thickness <= 0:
        raise ValueError(f"Толщина коллиматора должна быть строго положительной: {eff_collimator_thickness}")
    if gap < 0:
        raise ValueError(f"Зазор gap не может быть отрицательным: {gap}")
    if shielding_thickness <= 0:
        raise ValueError(f"Толщина защиты корпуса должна быть строго положительной: {shielding_thickness}")
    if glass_backend_thickness <= 0:
        raise ValueError(f"Толщина оптической подложки должна быть строго положительной: {glass_backend_thickness}")

    # Разрешение материалов из базы данных или аргументов
    material_db = database_setting.material_database
    collimator_mat = collimator.material if collimator is not None else material_db.get(collimator_material_name, Material(name=collimator_material_name))
    detector_mat = detector.material if detector is not None else material_db.get(detector_material_name, Material(name=detector_material_name))
    casing_mat = shielding_material if shielding_material is not None else material_db.get(casing_material_name, Material(name=casing_material_name))
    internal_mat = internal_medium if internal_medium is not None else material_db.get(internal_material_name, Material(name=internal_material_name))
    glass_mat = glass_material if glass_material is not None else material_db.get(glass_material_name, Material(name=glass_material_name))

    # Расчет габаритов полостей и корпусов
    box_size_x = det_size_x
    box_size_y = det_size_y
    box_size_z = eff_collimator_thickness + gap + eff_detector_thickness + glass_backend_thickness

    casing_size_x = box_size_x + 2.0 * shielding_thickness
    casing_size_y = box_size_y + 2.0 * shielding_thickness
    casing_size_z = box_size_z + shielding_thickness

    # Создание физических объемов
    camera_node = GammaCameraNode(name=effective_name)

    casing_volume = Volume(
        geometry=Box(casing_size_x, casing_size_y, casing_size_z),
        material=casing_mat,
        name=f"Casing_{effective_name}"
    )

    detector_box_volume = Volume(
        geometry=Box(box_size_x, box_size_y, box_size_z),
        material=internal_mat,
        name=f"Detector_box_{effective_name}"
    )

    collimator_volume = collimator if collimator is not None else Volume(
        geometry=Box(det_size_x, det_size_y, eff_collimator_thickness),
        material=collimator_mat,
        name=f"Collimator_{effective_name}"
    )

    crystal_volume = detector if detector is not None else Volume(
        geometry=Box(det_size_x, det_size_y, eff_detector_thickness),
        material=detector_mat,
        name=f"Detector_{effective_name}"
    )

    glass_backend_volume = Volume(
        geometry=Box(det_size_x, det_size_y, glass_backend_thickness),
        material=glass_mat,
        name=f"Glass_backend_{effective_name}"
    )

    # Построение иерархии связей
    camera_node.add_child(casing_volume)
    casing_volume.add_child(detector_box_volume)
    detector_box_volume.add_child(collimator_volume)
    detector_box_volume.add_child(crystal_volume)
    detector_box_volume.add_child(glass_backend_volume)

    # Установка локальных смещений (позиционирование слоев)
    detector_box_volume.local_matrix = np.eye(4, dtype=Float)
    detector_box_volume.translate(z=shielding_thickness / 2.0)

    collimator_volume.local_matrix = np.eye(4, dtype=Float)
    collimator_volume.translate(z=(box_size_z / 2.0 - eff_collimator_thickness / 2.0))

    crystal_volume.local_matrix = np.eye(4, dtype=Float)
    crystal_volume.translate(z=(box_size_z / 2.0 - eff_collimator_thickness - eff_detector_thickness / 2.0 - gap))

    glass_backend_volume.local_matrix = np.eye(4, dtype=Float)
    glass_backend_volume.translate(z=(glass_backend_thickness / 2.0 - box_size_z / 2.0))

    slots_dict = {
        "casing": casing_volume.name,
        "detector_box": detector_box_volume.name,
        "collimator": collimator_volume.name,
        "crystal": crystal_volume.name,
        "glass_backend": glass_backend_volume.name,
    }
    camera_node.slots = dict(slots_dict)
    slots_config = GammaCameraSlotsConfig(**slots_dict)

    return camera_node, slots_config


__all__ = [
    "create_default_gamma_camera",
]
