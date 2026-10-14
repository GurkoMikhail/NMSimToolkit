"""
Сценарий моделирования ОФЭКТ головного мозга (Brain SPECT) с фантомом Хоффмана,
четырехголовочной системой гамма-камер и радиофармпрепаратом 99mTc-HMPAO / 99mTc-MIBI.
"""

import logging
from multiprocessing import Manager, Pool
import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import hepunits as units
import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

from core.data.data_handlers import HistoryAssemblerHandler
from core.data.data_manager import DataManager
from core.geometry.direct_collimators import CollimatorHoleShape
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Float
from core.physics.processes import (
    CoherentScattering,
    ComptonScattering,
    PhotoelectricEffect,
)
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.nodes import SpatialNode
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from infra.monitoring.telegram import TeleBotHandler
from settings.database_setting import attenuation_database, material_database

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _build_camera_head(
    head_angle: float,
    radius: float,
) -> Tuple[GammaCameraNode, Volume]:
    """
    Собирает детекторную головку гамма-камеры со слотами (корпус, коллиматор, кристалл, световод)
    и позиционирует её на заданный угол и радиус орбиты.
    """
    angle_label = f"{round(head_angle / units.degree, 1)} deg"
    detector_size_x = 54.0 * units.cm
    detector_size_y = 40.0 * units.cm
    detector_thickness = 0.95 * units.cm
    collimator_thickness = 3.5 * units.cm
    gap = 0.1 * units.cm
    shielding_thickness = 2.0 * units.cm
    glass_backend_thickness = 7.6 * units.cm

    box_size_z = collimator_thickness + gap + detector_thickness + glass_backend_thickness
    casing_size_x = detector_size_x + 2.0 * shielding_thickness
    casing_size_y = detector_size_y + 2.0 * shielding_thickness
    casing_size_z = box_size_z + shielding_thickness

    camera_node = GammaCameraNode(name=f"Gamma_camera at {angle_label}")

    casing = Volume(
        geometry=Box(Float(casing_size_x), Float(casing_size_y), Float(casing_size_z)),
        material=material_database["Pb"],
        name=f"Casing at {angle_label}",
    )
    detector_box = Volume(
        geometry=Box(Float(detector_size_x), Float(detector_size_y), Float(box_size_z)),
        material=material_database["Air, Dry (near sea level)"],
        name=f"Detector_box at {angle_label}",
    )
    detector_box.translate(z=Float(shielding_thickness / 2.0))

    collimator = ParametricParallelCollimator(
        size=(Float(detector_size_x), Float(detector_size_y), Float(collimator_thickness)),
        hole_diameter=Float(1.5 * units.mm),
        septa=Float(0.2 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
        material=material_database["Pb"],
        name=f"Collimator at {angle_label}",
    )
    collimator.translate(z=Float(box_size_z / 2.0 - collimator_thickness / 2.0))

    detector = Volume(
        geometry=Box(Float(detector_size_x), Float(detector_size_y), Float(detector_thickness)),
        material=material_database["Sodium Iodide"],
        name=f"Detector at {angle_label}",
    )
    detector.translate(
        z=Float(box_size_z / 2.0 - collimator_thickness - detector_thickness / 2.0 - gap)
    )

    glass_backend = Volume(
        geometry=Box(Float(detector_size_x), Float(detector_size_y), Float(glass_backend_thickness)),
        material=material_database["Glass, Borosilicate (Pyrex)"],
        name=f"Glass_backend at {angle_label}",
    )
    glass_backend.translate(z=Float(glass_backend_thickness / 2.0 - box_size_z / 2.0))

    detector_box.add_child(collimator)
    detector_box.add_child(detector)
    detector_box.add_child(glass_backend)
    casing.add_child(detector_box)
    camera_node.add_child(casing)

    camera_node.slots = {
        "casing": casing.name or "",
        "detector_box": detector_box.name or "",
        "collimator": collimator.name or "",
        "crystal": detector.name or "",
        "glass_backend": glass_backend.name or "",
    }

    camera_node.rotate(gamma=Float(np.pi / 2.0))
    camera_node.translate(y=Float(radius + casing_size_z / 2.0))
    camera_node.rotate(alpha=Float(head_angle))

    return camera_node, detector


def modeling(
    angle: float,
    radius: float,
    gamma_cameras: int,
    delta_angle: float,
    time_interval: NDArray[np.float64] | Sequence[float],
    seed: SeedSequence | int,
    lock: Any,
) -> None:
    """
    Выполняет моделирование одного углового положения и временного интервала ОФЭКТ головного мозга.
    """
    log_path = PROJECT_ROOT / "logs" / "brain_healthy" / f"{round(angle / units.degree, 1)} deg.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s: %(levelname)s] %(message)s"))
    telebot_handler = TeleBotHandler()
    telebot_handler.setLevel(logging.WARNING)
    telebot_handler.setFormatter(logging.Formatter("[%(asctime)s: %(levelname)s]\n%(message)s"))

    logging.basicConfig(handlers=[file_handler, telebot_handler])

    rng = np.random.default_rng(seed)
    start_time, stop_time = float(time_interval[0]), float(time_interval[1])

    world = Volume(
        geometry=Box(Float(120.0 * units.cm), Float(120.0 * units.cm), Float(80.0 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name="Simulation_volume",
    )

    material_id_distribution = np.load(PROJECT_ROOT / "phantoms" / "hoffman_attenuation.npy")
    material_distribution = MaterialArray(material_id_distribution.shape)
    material_distribution[material_id_distribution == 0] = material_database["Air, Dry (near sea level)"]
    material_distribution[material_id_distribution == 3] = material_database["Tissue, Soft (ICRU-44)"]

    phantom = WoodcockVoxelVolume(
        voxel_size=Float(4.0 * units.mm),
        material_distribution=material_distribution,
        name="Phantom",
    )
    world.add_child(phantom)

    detector_list: List[SpatialNode] = []
    for camera_index in range(gamma_cameras):
        head_angle = angle + delta_angle * camera_index
        spect_head, detector = _build_camera_head(head_angle=head_angle, radius=radius)
        world.add_child(spect_head)
        detector_list.append(detector)

    activity_distribution = np.load(PROJECT_ROOT / "phantoms" / "hoffman_activity.npy")
    source = Source(
        distribution=activity_distribution,
        activity=Float(200.0 * units.MBq),
        voxel_size=Float(4.0 * units.mm),
        radiation_type="Gamma",
        energy=[[Float(140.511 * units.keV), 0.89]],
        half_life=Float(6.0067 * units.hour),
        rng=rng,
    )
    phantom.add_child(source)

    processes = [PhotoelectricEffect, ComptonScattering, CoherentScattering]
    propagator = ParticlePropagator(
        processes_list=processes,
        attenuation_database=attenuation_database,
        rng=rng,
    )
    manager = SimulationManager(
        scene=world,
        propagator=propagator,
        start_time=Float(start_time),
        stop_time=Float(stop_time),
        particles_number=10**6,
        buffer_capacity=100_000,
        name=f"{round(angle / units.degree, 1)} deg",
    )
    manager.start()

    handler = HistoryAssemblerHandler(sensitive_volumes=detector_list)
    data_manager = DataManager(
        filename=PROJECT_ROOT / "output data" / "brain_healthy" / f"{manager.name}.hdf",
        handlers=[handler],
        queue=manager.queue,
        lock=lock,
    )
    data_manager.start()

    manager.join()
    data_manager.join()


def main() -> None:
    """Точка входа для параллельного запуска томографического сканирования мозга."""
    views = 120
    gamma_cameras = 4
    steps = 5
    radius = float(200.0 * units.mm)
    time_start = float(0.0 * units.second)
    time_stop = float(5.0 * units.second)
    pool_size = 30

    angles = np.linspace(0.0, 2.0 * np.pi, views, endpoint=False)[: views // gamma_cameras]
    delta_angle = float(np.pi / 2.0)

    time_points = np.linspace(time_start, time_stop, steps + 1)
    time_intervals = np.column_stack([time_points[:-1], time_points[1:]])

    seed_sequence = SeedSequence()

    mp_manager = Manager()
    locks = [mp_manager.Lock() for _ in range(angles.size)]

    with Pool(pool_size) as pool:
        for time_interval in time_intervals:
            for angle_val, lock_obj in zip(angles, locks):
                child_seed = seed_sequence.spawn(1)[0]
                pool.apply_async(
                    modeling,
                    (float(angle_val), radius, gamma_cameras, delta_angle, time_interval, child_seed, lock_obj),
                )
        pool.close()
        pool.join()


__all__ = [
    "PROJECT_ROOT",
    "modeling",
    "main",
]


if __name__ == "__main__":
    main()


