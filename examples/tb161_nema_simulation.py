"""
Скрипт моделирования ОФЭКТ/КТ 161Tb с двухкамерной системой (GE Discovery LEHR)
и фантомом NEMA IEQ. Основан на клиническом исследовании Marin et al. (EJNMMI Phys 2020)
и прецизионных спектрометрических данных Juget et al. (Appl. Radiat. Isot. 174, 2021 + Corrigendum 200, 2023).

Физическая точность:
- Моделирование учитывает выверенный 21-линейный спектр излучения 161Tb (Juget et al., 2021/2023; ENSDF 2020).
- Используется аналитический коллиматор DirectParallelCollimator (RayCasting гексагональной решетки без холостых шагов Вудкока в каналах).
- Энергетические окна не ограничивают расчет Монте-Карло: в HDF5 сохраняются все реальные
  взаимодействия и энерговыделения (min_energy = 1.0 кэВ) для последующей спектрометрической
  постобработки и выделения окон (фотопикового EM2 и рассеянного SC2).
"""

from multiprocessing import Pool
import os
from pathlib import Path
import queue
import time
from typing import Any, List, Tuple

# Ограничение сторонней многопоточности для корректной работы пула multiprocessing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import hepunits as units
import numpy as np
from numpy.typing import NDArray

from core.data.data_handlers import HistoryAssemblerHandler
from core.data.data_manager import DataManager
from core.data.distribution_loader import DistributionLoader
from core.geometry.direct_collimators import CollimatorHoleShape, DirectParallelCollimator
from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Float
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode
from core.source.sources import Source
from core.transport.propagator import ParticlePropagator
from core.transport.simulation_managers import SimulationManager
from settings.database_setting import attenuation_database, material_database
from settings.processes_settings import processes_list

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------------------------------------------------
# 1. Спецификация радионуклида 161Tb (Juget et al., 2021 + Corrigendum 2023; Durán et al., 2020)
# ------------------------------------------------------------------------------------------------------------------------
TB161_HALF_LIFE: float = float(6.953 * units.day)

# Многолинейный спектр фотонного излучения 161Tb (гамма-кванты и характеристический X-ray Dy, выход > 0.003%):
# Каждая запись: [энергия квантов в HepUnits, интенсивность эмиссии на 100 распадов (%)]
TB161_ENERGY_SPECTRUM: List[List[float]] = [
    [float(25.65 * units.keV), 23.154],    # Низкоэнергетическая гамма-линия (ENSDF 2020)
    [float(45.21 * units.keV), 7.747],     # Характеристический рентген Dy K_alpha2
    [float(46.00 * units.keV), 12.779],    # Характеристический рентген Dy K_alpha1
    [float(48.92 * units.keV), 17.73],     # Гамма-линия 48.915 кэВ (вторичное окно ОФЭКТ)
    [float(52.19 * units.keV), 3.741],     # Характеристический рентген Dy K_beta1'
    [float(53.64 * units.keV), 1.0185],    # Характеристический рентген Dy K_beta2'
    [float(57.19 * units.keV), 2.066],     # Гамма-линия 57.192 кэВ
    [float(59.24 * units.keV), 0.2002],    # Гамма-линия 59.243 кэВ
    [float(74.57 * units.keV), 10.28],     # Главный гамма-пик ОФЭКТ (74.567 кэВ)
    [float(77.42 * units.keV), 0.0459],    # Гамма-линия 77.422 кэВ (внутри окна 74.6 кэВ)
    [float(87.94 * units.keV), 0.1996],    # Гамма-линия 87.941 кэВ (комптоновский down-scatter)
    [float(103.07 * units.keV), 0.1077],   # Гамма-линия 103.065 кэВ (комптоновский down-scatter)
    [float(106.11 * units.keV), 0.0816],   # Гамма-линия 106.113 кэВ (комптоновский down-scatter)
    [float(286.48 * units.keV), 0.01574],  # Гамма-линия 286.481 кэВ (септальное проникновение)
    [float(292.40 * units.keV), 0.07102],  # Гамма-линия 292.401 кэВ (септальное проникновение)
    [float(319.66 * units.keV), 0.00367],  # Гамма-линия 319.660 кэВ
    [float(341.40 * units.keV), 0.00375],  # Гамма-линия 341.400 кэВ
    [float(343.63 * units.keV), 0.01473],  # Гамма-линия 343.630 кэВ
    [float(418.47 * units.keV), 0.00925],  # Гамма-линия 418.470 кэВ
    [float(475.66 * units.keV), 0.01660],  # Гамма-линия 475.658 кэВ
    [float(550.25 * units.keV), 0.04258],  # Гамма-линия 550.249 кэВ (главная линия пробоя септ)
]


# ------------------------------------------------------------------------------------------------------------------------
# 2. Построитель одной детекторной головки гамма-камеры GE Discovery LEHR
# ------------------------------------------------------------------------------------------------------------------------
def build_camera_head(
    head_index: int,
    relative_angle: float,
    center_of_rotation_radius: float,
) -> Tuple[GammaCameraNode, Volume]:
    """
    Создает одну детекторную головку GE Discovery с коллиматором LEHR
    и монтирует её на заданный относительный угол (в радианах) внутри станины.
    Возвращает кортеж (узел камеры, объём кристалла-детектора).
    """
    camera_node = GammaCameraNode(name=f"GammaCamera_{head_index}")

    # Внешний свинцовый корпус
    casing = Volume(
        geometry=Box(Float(58.0 * units.cm), Float(44.0 * units.cm), Float(14.06 * units.cm)),
        material=material_database["Pb"],
        name=f"Casing_{head_index}",
    )

    # Внутренняя воздушная полость
    detector_box = Volume(
        geometry=Box(Float(54.0 * units.cm), Float(40.0 * units.cm), Float(12.06 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name=f"Detector_box_{head_index}",
    )
    detector_box.translate(z=Float(1.0 * units.cm))

    # Коллиматор LEHR (GE Discovery): толщина 3.5 см, гексагональные отверстия 1.5 мм, септа 0.2 мм
    collimator = DirectParallelCollimator(
        size=[Float(54.0 * units.cm), Float(40.0 * units.cm), Float(3.5 * units.cm)],
        hole_diameter=Float(1.5 * units.mm),
        septa=Float(0.2 * units.mm),
        hole_shape=CollimatorHoleShape.HEXAGONAL,
        material=material_database["Pb"],
        name=f"Collimator_{head_index}",
    )
    collimator.translate(z=Float(4.28 * units.cm))

    # Сцинтилляционный кристалл NaI(Tl) толщиной 3/8 дюйма (9.5 мм)
    crystal = Volume(
        geometry=Box(Float(54.0 * units.cm), Float(40.0 * units.cm), Float(9.5 * units.mm)),
        material=material_database["Sodium Iodide"],
        name=f"Detector_{head_index}",
    )
    crystal.translate(z=Float(2.045 * units.cm))

    # Подложка из боросиликатного стекла (световод)
    glass_backend = Volume(
        geometry=Box(Float(54.0 * units.cm), Float(40.0 * units.cm), Float(7.6 * units.cm)),
        material=material_database["Glass, Borosilicate (Pyrex)"],
        name=f"Glass_backend_{head_index}",
    )
    glass_backend.translate(z=Float(-2.23 * units.cm))

    # Сборка иерархии головки
    detector_box.add_child(collimator)
    detector_box.add_child(crystal)
    detector_box.add_child(glass_backend)
    casing.add_child(detector_box)
    camera_node.add_child(casing)

    # Позиционирование головки относительно центра гантри
    camera_node.rotate(Float(0.0), Float(0.0), Float(90.0 * units.deg))
    camera_node.translate(y=Float(center_of_rotation_radius))
    if relative_angle != 0.0:
        camera_node.rotate(Float(relative_angle), Float(0.0), Float(0.0))

    camera_node.slots = {
        "casing": casing.name or "",
        "detector_box": detector_box.name or "",
        "collimator": collimator.name or "",
        "crystal": crystal.name or "",
        "glass_backend": glass_backend.name or "",
    }

    return camera_node, crystal


# ------------------------------------------------------------------------------------------------------------------------
# 3. Построитель двухкамерного гентри (Gantry)
# ------------------------------------------------------------------------------------------------------------------------
def build_dual_head_gantry(
    gantry_angle: float,
    center_of_rotation_radius: float = float(329.05 * units.mm),
    head_relative_angles: Tuple[float, float] = (0.0, float(180.0 * units.deg)),
) -> Tuple[GantryNode, Volume, Volume]:
    """
    Создает поворотную станину (Gantry) с 2 противоположными головками под 180°.
    Возвращает кортеж (gantry, crystal_1, crystal_2).
    """
    gantry = GantryNode(name="Gantry")

    head_1, crystal_1 = build_camera_head(
        1,
        relative_angle=head_relative_angles[0],
        center_of_rotation_radius=center_of_rotation_radius,
    )
    head_2, crystal_2 = build_camera_head(
        2,
        relative_angle=head_relative_angles[1],
        center_of_rotation_radius=center_of_rotation_radius,
    )

    gantry.add_child(head_1)
    gantry.add_child(head_2)

    gantry.rotate(Float(gantry_angle), Float(0.0), Float(0.0))
    return gantry, crystal_1, crystal_2


# ------------------------------------------------------------------------------------------------------------------------
# 4. Построитель фантома NEMA IEQ
# ------------------------------------------------------------------------------------------------------------------------
def _load_voxel_grid(
    file_path: str | Path,
    shape: Tuple[int, int, int] = (128, 128, 92),
) -> NDArray[Any]:
    """Загружает трехмерную воксельную матрицу из файла относительно корня проекта."""
    path = Path(file_path)
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    return DistributionLoader.load(resolved_path, target_shape=shape, order="F", dtype=np.float32)


def build_nema_phantom(activity: float = float(100.0 * units.MBq)) -> WoodcockVoxelVolume:
    """Создает воксельный фантом NEMA IEC с источником 161Tb."""
    raw_attenuation = _load_voxel_grid(PROJECT_ROOT / "phantoms" / "nema" / "anema_voxel_size_4.2_mm.dat")

    material_distribution = MaterialArray((128, 128, 92))
    material_distribution[np.isclose(raw_attenuation, 0.04)] = material_database["Air, Dry (near sea level)"]
    material_distribution[np.isclose(raw_attenuation, 0.15)] = material_database["Water, Liquid"]

    phantom = WoodcockVoxelVolume(
        voxel_size=Float(4.2 * units.mm),
        material_distribution=material_distribution,
        name="Phantom",
    )
    phantom.rotate(Float(90.0 * units.deg), Float(0.0), Float(0.0))

    raw_activity = _load_voxel_grid(PROJECT_ROOT / "phantoms" / "nema" / "nema_voxel_size_4.2_mm.dat")

    source = Source(
        distribution=raw_activity,
        activity=Float(activity),
        voxel_size=Float(4.2 * units.mm),
        radiation_type="Gamma",
        energy=TB161_ENERGY_SPECTRUM,
        half_life=Float(TB161_HALF_LIFE),
    )
    source.name = "Source"
    phantom.add_child(source)

    return phantom


# ------------------------------------------------------------------------------------------------------------------------
# 5. Функция симуляции одного шага гантри
# ------------------------------------------------------------------------------------------------------------------------
def simulate_dual_head_step(
    gantry_angle: float,
    particles_number: int,
    output_directory: Path,
    seed: int = 42,
) -> None:
    """
    Моделирует сбор данных обеими головками (0° и 180°) при текущем угле поворота станины (в радианах).
    Все взаимодействия выше min_energy = 1.0 кэВ записываются в HDF5.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    angle_in_degrees = gantry_angle / units.deg
    h5_filename = str(output_directory / f"gantry_{angle_in_degrees:.1f}_deg.hdf")

    root_scene = Volume(
        geometry=Box(Float(120.0 * units.cm), Float(120.0 * units.cm), Float(80.0 * units.cm)),
        material=material_database["Air, Dry (near sea level)"],
        name="Simulation_volume",
    )

    phantom = build_nema_phantom()
    root_scene.add_child(phantom)

    # Двухкамерная гантри (COR = 329.05 мм)
    gantry, crystal_1, crystal_2 = build_dual_head_gantry(
        gantry_angle=gantry_angle,
        center_of_rotation_radius=float(329.05 * units.mm),
        head_relative_angles=(0.0, float(180.0 * units.deg)),
    )
    root_scene.add_child(gantry)

    # Межпоточная очередь между генератором Монте-Карло и обработчиком HDF5
    sim_queue: queue.Queue[Any] = queue.Queue()

    history_handler = HistoryAssemblerHandler(
        sensitive_volumes=[crystal_1, crystal_2],
        save_initial_states=True,
    )

    data_manager = DataManager(
        filename=h5_filename,
        handlers=[history_handler],
        queue=sim_queue,
    )

    rng = np.random.default_rng(seed)
    propagator = ParticlePropagator(
        processes_list=processes_list,
        attenuation_database=attenuation_database,
        rng=rng,
    )

    sim_manager = SimulationManager(
        scene=root_scene,
        propagator=propagator,
        start_time=Float(0.0 * units.s),
        stop_time=Float(30.0 * units.s),
        particles_number=particles_number,
        min_energy=Float(1.0 * units.keV),
        queue=sim_queue,
        buffer_capacity=100_000,
        seed=seed,
    )

    # Запуск фонового сборщика данных, расчет шага и корректный сброс буферов в файл
    data_manager.start()
    try:
        sim_manager.run()
    finally:
        sim_manager.flush_all()
        data_manager.stop()


def _worker_task(task_params: Tuple[float, int, Path, int]) -> float:
    """Рабочая функция процесса пула для расчета одного углового положения гантри."""
    gantry_angle, particles, out_dir, seed = task_params
    time_start = time.perf_counter()
    simulate_dual_head_step(
        gantry_angle=gantry_angle,
        particles_number=particles,
        output_directory=out_dir,
        seed=seed,
    )
    elapsed = time.perf_counter() - time_start
    print(f"[Готово] Угол {gantry_angle / units.deg:.1f}° завершен за {elapsed:.1f} с (seed: {seed})")
    return gantry_angle


# ------------------------------------------------------------------------------------------------------------------------
# 6. Точка входа: параллельный запуск
# ------------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """Точка входа для параллельного запуска 60-шагового протокола ОФЭКТ 161Tb."""
    gantry_positions = np.linspace(0.0, float(180.0 * units.deg), 60, endpoint=False)
    particles_per_step = int(np.ceil(2_000_000 / len(gantry_positions)))
    output_dir = PROJECT_ROOT / "results" / "tb161_nema_120projections"

    num_processes = max(1, (os.cpu_count() or 1) - 1)
    base_seed = 100_000

    print("=== Двухкамерный протокол ОФЭКТ 161Tb (GE Discovery NM/CT 670) ===")
    print(
        f"Шагов гантри: {len(gantry_positions)} "
        f"(0.0° - {gantry_positions[-1] / units.deg:.1f}°)"
    )
    print(f"Частиц на один шаг: {particles_per_step:,}")
    print(f"Суммарно частиц: {particles_per_step * len(gantry_positions):,}")
    print(f"Директория сохранения: {output_dir.resolve()}")
    print(f"Выделено процессов: {num_processes}")
    print("------------------------------------------------------------------")

    tasks: List[Tuple[float, int, Path, int]] = [
        (float(angle), particles_per_step, output_dir, base_seed + idx)
        for idx, angle in enumerate(gantry_positions)
    ]

    start_wall_time = time.perf_counter()

    with Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(_worker_task, tasks):
            pass

    total_time = time.perf_counter() - start_wall_time
    print("------------------------------------------------------------------")
    print(f"Моделирование полностью завершено за {total_time / 60.0:.2f} мин.")


__all__ = [
    "PROJECT_ROOT",
    "TB161_HALF_LIFE",
    "TB161_ENERGY_SPECTRUM",
    "build_camera_head",
    "build_dual_head_gantry",
    "build_nema_phantom",
    "simulate_dual_head_step",
    "main",
]


if __name__ == "__main__":
    main()