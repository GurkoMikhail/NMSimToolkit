"""
Скрипт параметрического моделирования ОФЭКТ/КТ 161Tb с двухкамерной системой (GE Discovery LEHR)
и фантомом NEMA IEQ. Основан на клиническом исследовании Marin et al. (EJNMMI Phys 2020).

Физическая точность:
- Моделирование учитывает полный спектр линий излучения 161Tb (ICRP 107 / Marin et al., 2020).
- Энергетические окна не ограничивают расчет Монте-Карло: в HDF5 сохраняются все реальные
  взаимодействия и энерговыделения (min_energy = 1.0 кэВ) для последующей спектрометрической
  постобработки и выделения окон (фотопикового EM2 и рассеянного SC2).
"""

import os
import queue
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
import numpy as np
import hepunits as units

# Ограничение сторонней многопоточности для корректной работы пула multiprocessing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from core.geometry.geometries import Box
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.materials.materials import MaterialArray
from core.scene.gamma_camera_node import GammaCameraNode
from core.scene.gantry_node import GantryNode
from core.source.sources import Source
from core.data.data_manager import DataManager
from core.data.data_handlers import HistoryAssemblerHandler
from core.data.distribution_loader import DistributionLoader
from core.transport.simulation_managers import SimulationManager
from core.transport.propagator import ParticlePropagator

# Импорт баз данных и физических процессов
from settings.database_setting import material_database, attenuation_database
from settings.processes_settings import processes_list


# ------------------------------------------------------------------------------------------------------------------------
# 1. Спецификация радионуклида 161Tb (ICRP 107 / Marin et al., 2020)
# ------------------------------------------------------------------------------------------------------------------------
TB161_HALF_LIFE = 6.89 * units.day

# Полный многолинейный спектр фотонного излучения 161Tb (гамма-кванты и характеристический X-ray Dy):
# Каждая запись: [энергия квантов в HepUnits, относительный выход %]
TB161_ENERGY_SPECTRUM = [
    [74.57 * units.keV, 10.3],   # Основной фотопик ОФЭКТ
    [48.9 * units.keV, 17.0],    # Рентгеновские K-бета и сопутствующие гамма-линии
    [46.0 * units.keV, 11.2],    # Характеристический рентген K-альфа Dy
    [25.65 * units.keV, 23.2],   # Низкоэнергетическая гамма-линия
    [52.0 * units.keV, 1.4],     # Характеристический X-ray Dy
    [57.4 * units.keV, 0.4],     # Высокие рентгеновские переходы
    [106.1 * units.keV, 0.05],   # Септальное проникновение в LEHR
    [131.8 * units.keV, 0.03],
    [160.0 * units.keV, 0.004],
    [292.0 * units.keV, 0.002],
]


# ------------------------------------------------------------------------------------------------------------------------
# 2. Построитель одной детекторной головки гамма-камеры GE Discovery LEHR
# ------------------------------------------------------------------------------------------------------------------------
def build_camera_head(head_index: int, relative_angle_deg: float, center_of_rotation_radius: float) -> Tuple[GammaCameraNode, Volume]:
    """
    Создает одну детекторную головку GE Discovery с коллиматором LEHR
    и монтирует её на заданный относительный угол внутри станины.
    Возвращает кортеж (узел камеры, объём кристалла-детектора).
    """
    camera_node = GammaCameraNode(name=f"GammaCamera_{head_index}")

    # Внешний свинцовый корпус
    casing = Volume(
        geometry=Box(58.0 * units.cm, 44.0 * units.cm, 14.06 * units.cm),
        material=material_database["Pb"],
        name=f"Casing_{head_index}"
    )

    # Внутренняя воздушная полость
    detector_box = Volume(
        geometry=Box(54.0 * units.cm, 40.0 * units.cm, 12.06 * units.cm),
        material=material_database["Air, Dry (near sea level)"],
        name=f"Detector_box_{head_index}"
    )
    detector_box.translate(z=1.0 * units.cm)

    # Коллиматор LEHR (GE Discovery): толщина 3.5 см, гексагональные отверстия 1.5 мм, септа 0.2 мм
    collimator = ParametricParallelCollimator(
        size=[54.0 * units.cm, 40.0 * units.cm, 3.5 * units.cm],
        hole_diameter=1.5 * units.mm,
        septa=0.2 * units.mm,
        hole_shape="hexagonal",
        material=material_database["Pb"],
        name=f"Collimator_{head_index}"
    )
    collimator.translate(z=4.28 * units.cm)

    # Сцинтилляционный кристалл NaI(Tl) толщиной 3/8 дюйма (9.5 мм)
    crystal = Volume(
        geometry=Box(54.0 * units.cm, 40.0 * units.cm, 9.5 * units.mm),
        material=material_database["Sodium Iodide"],
        name=f"Detector_{head_index}"
    )
    crystal.translate(z=2.045 * units.cm)

    # Подложка из боросиликатного стекла (световод)
    glass_backend = Volume(
        geometry=Box(54.0 * units.cm, 40.0 * units.cm, 7.6 * units.cm),
        material=material_database["Glass, Borosilicate (Pyrex)"],
        name=f"Glass_backend_{head_index}"
    )
    glass_backend.translate(z=-2.23 * units.cm)

    # Сборка иерархии головки
    detector_box.add_child(collimator)
    detector_box.add_child(crystal)
    detector_box.add_child(glass_backend)
    casing.add_child(detector_box)
    camera_node.add_child(casing)

    # Позиционирование головки относительно центра гантри
    camera_node.rotate(0.0, 0.0, 90.0 * units.deg)
    camera_node.translate(y=center_of_rotation_radius)
    if relative_angle_deg != 0.0:
        camera_node.rotate(relative_angle_deg * units.deg, 0.0, 0.0)

    camera_node.slots = {
        "casing": casing.name,
        "detector_box": detector_box.name,
        "collimator": collimator.name,
        "crystal": crystal.name,
        "glass_backend": glass_backend.name,
    }

    return camera_node, crystal


# ------------------------------------------------------------------------------------------------------------------------
# 3. Построитель двухкамерного гентри (Gantry)
# ------------------------------------------------------------------------------------------------------------------------
def build_dual_head_gantry(
    gantry_angle_deg: float,
    center_of_rotation_radius: float = 329.05 * units.mm,
    head_relative_angles: Tuple[float, float] = (0.0, 180.0)
) -> Tuple[GantryNode, Volume, Volume]:
    """
    Создает поворотную станину (Gantry) с 2 противоположными головками под 180°.
    Возвращает кортеж (gantry, crystal_1, crystal_2).
    """
    gantry = GantryNode(name="Gantry")

    head_1, crystal_1 = build_camera_head(1, relative_angle_deg=head_relative_angles[0], center_of_rotation_radius=center_of_rotation_radius)
    head_2, crystal_2 = build_camera_head(2, relative_angle_deg=head_relative_angles[1], center_of_rotation_radius=center_of_rotation_radius)

    gantry.add_child(head_1)
    gantry.add_child(head_2)

    gantry.rotate(gantry_angle_deg * units.deg, 0.0, 0.0)
    return gantry, crystal_1, crystal_2


# ------------------------------------------------------------------------------------------------------------------------
# 4. Построитель фантома NEMA IEQ
# ------------------------------------------------------------------------------------------------------------------------
def _load_voxel_grid(file_path: str, shape: Tuple[int, int, int] = (128, 128, 92)) -> np.ndarray:
    path = Path(file_path)
    try:
        return DistributionLoader.load(path, target_shape=shape, order="F", dtype=np.float32)
    except Exception:
        pass
    try:
        return np.fromfile(path, dtype=np.float32).reshape(shape, order="F")
    except Exception:
        return np.loadtxt(path, dtype=np.float32).reshape(shape, order="F")


def build_nema_phantom(activity_bq: float = 100.0 * units.MBq) -> WoodcockVoxelVolume:
    raw_attenuation = _load_voxel_grid("phantoms/nema/anema_voxel_size_4.2_mm.dat")

    material_distribution = MaterialArray((128, 128, 92))
    material_distribution[np.isclose(raw_attenuation, 0.04)] = material_database["Air, Dry (near sea level)"]
    material_distribution[np.isclose(raw_attenuation, 0.15)] = material_database["Water, Liquid"]

    phantom = WoodcockVoxelVolume(
        voxel_size=4.2 * units.mm,
        material_distribution=material_distribution,
        name="Phantom"
    )

    raw_activity = _load_voxel_grid("phantoms/nema/nema_voxel_size_4.2_mm.dat")

    source = Source(
        distribution=raw_activity,
        activity=activity_bq,
        voxel_size=4.2 * units.mm,
        radiation_type="Gamma",
        energy=TB161_ENERGY_SPECTRUM,
        half_life=TB161_HALF_LIFE
    )
    source.name = "Source"
    phantom.add_child(source)

    return phantom


# ------------------------------------------------------------------------------------------------------------------------
# 5. Функция симуляции одного шага гантри
# ------------------------------------------------------------------------------------------------------------------------
def simulate_dual_head_step(
    gantry_angle_deg: float,
    particles_number: int,
    output_directory: Path,
    seed: int = 42
) -> None:
    """
    Моделирует сбор данных обеими головками (0° и 180°) при текущем угле поворота станины.
    Все взаимодействия выше min_energy = 1.0 кэВ записываются в HDF5.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    h5_filename = str(output_directory / f"gantry_{gantry_angle_deg:.1f}_deg.hdf")

    root_scene = Volume(
        geometry=Box(120.0 * units.cm, 120.0 * units.cm, 80.0 * units.cm),
        material=material_database["Air, Dry (near sea level)"],
        name="Simulation_volume"
    )

    phantom = build_nema_phantom()
    root_scene.add_child(phantom)

    # Двухкамерная гантри (COR = 329.05 мм)
    gantry, crystal_1, crystal_2 = build_dual_head_gantry(
        gantry_angle_deg=gantry_angle_deg,
        center_of_rotation_radius=329.05 * units.mm,
        head_relative_angles=(0.0, 180.0)
    )
    root_scene.add_child(gantry)

    # Межпоточная очередь между генератором Монте-Карло и обработчиком HDF5
    sim_queue: queue.Queue = queue.Queue()

    history_handler = HistoryAssemblerHandler(
        sensitive_volumes=[crystal_1, crystal_2],
        save_initial_states=True
    )

    data_manager = DataManager(
        filename=h5_filename,
        handlers=[history_handler],
        queue=sim_queue
    )

    rng = np.random.default_rng(seed)
    propagator = ParticlePropagator(
        processes_list=processes_list,
        attenuation_database=attenuation_database,
        rng=rng
    )

    sim_manager = SimulationManager(
        scene=root_scene,
        propagator=propagator,
        start_time=0.0 * units.s,
        stop_time=30.0 * units.s,
        particles_number=particles_number,
        min_energy=1.0 * units.keV,
        queue=sim_queue,
        buffer_capacity=100_000,
        seed=seed
    )

    # Запуск фонового сборщика данных, расчет шага и корректный сброс буферов в файл
    data_manager.start()
    try:
        sim_manager.run()
    finally:
        sim_manager.flush_all()
        data_manager.stop()


def _worker_task(task_params: Tuple[float, int, Path, int]) -> float:
    angle_deg, particles, out_dir, seed = task_params
    t0 = time.perf_counter()
    simulate_dual_head_step(
        gantry_angle_deg=angle_deg,
        particles_number=particles,
        output_directory=out_dir,
        seed=seed
    )
    elapsed = time.perf_counter() - t0
    print(f"[Готово] Угол {angle_deg:.1f}° завершен за {elapsed:.1f} с (seed: {seed})")
    return angle_deg


# ------------------------------------------------------------------------------------------------------------------------
# 6. Точка входа: параллельный запуск
# ------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    gantry_positions = np.linspace(0.0, 180.0, 60, endpoint=False)
    particles_per_step = int(np.ceil(2_000_000 / len(gantry_positions)))
    output_dir = Path("results/tb161_nema_120projections")

    num_processes = max(1, (os.cpu_count() or 1) - 1)
    base_seed = 100_000

    print("=== Двухкамерный протокол ОФЭКТ 161Tb (GE Discovery NM/CT 670) ===")
    print(f"Шагов гантри: {len(gantry_positions)} (0.0° - {gantry_positions[-1]:.1f}°)")
    print(f"Частиц на один шаг: {particles_per_step:,}")
    print(f"Суммарно частиц: {particles_per_step * len(gantry_positions):,}")
    print(f"Директория сохранения: {output_dir.resolve()}")
    print(f"Выделено процессов: {num_processes}")
    print("------------------------------------------------------------------")

    tasks: List[Tuple[float, int, Path, int]] = [
        (angle, particles_per_step, output_dir, base_seed + idx)
        for idx, angle in enumerate(gantry_positions)
    ]

    start_wall_time = time.perf_counter()

    with Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(_worker_task, tasks):
            pass

    total_time = time.perf_counter() - start_wall_time
    print("------------------------------------------------------------------")
    print(f"Моделирование полностью завершено за {total_time / 60:.2f} мин.")