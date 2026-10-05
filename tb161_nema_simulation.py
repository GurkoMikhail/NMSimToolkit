"""
Скрипт параметрического моделирования ОФЭКТ/КТ 161Tb с двухкамерной системой (GE Discovery LEHR)
и фантомом NEMA IEQ. Основан на клиническом исследовании Marin et al. (EJNMMI Phys 2020).

Физическая точность:
- Моделирование учитывает полный спектр линий излучения 161Tb (ICRP 107 / Marin et al., 2020).
- Энергетические окна не ограничивают расчет Монте-Карло: в HDF5 сохраняются все реальные
  взаимодействия и энерговыделения (с порогом отсечки min_energy = 1.0 кэВ) для последующей
  спектрометрической постобработки и выделения окон (фотопикового EM2 и рассеянного SC2).
"""

import os
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
from core.transport.simulation_managers import SimulationManager
from core.transport.propagator import ParticlePropagator
from core.physics.physics_compiler import PhysicsCompiler
from settings.database_setting import material_database


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
    [106.1 * units.keV, 0.05],   # Слабые гамма-переходы (важны для расчета септального проникновения в LEHR)
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
def build_nema_phantom(activity_bq: float = 100.0 * units.MBq) -> WoodcockVoxelVolume:
    """
    Загружает и собирает воксельный фантом NEMA IEQ с полным спектром линий 161Tb.
    """
    raw_attenuation = np.fromfile(
        "phantoms/nema/anema_voxel_size_4.2_mm.dat", dtype=np.float32
    ).reshape((128, 128, 92), order="F")

    material_distribution = MaterialArray((128, 128, 92))
    material_distribution[np.isclose(raw_attenuation, 0.04)] = material_database["Air, Dry (near sea level)"]
    material_distribution[np.isclose(raw_attenuation, 0.15)] = material_database["Water, Liquid"]

    phantom = WoodcockVoxelVolume(
        voxel_size=4.2 * units.mm,
        material_distribution=material_distribution,
        name="Phantom"
    )

    raw_activity = np.fromfile(
        "phantoms/nema/nema_voxel_size_4.2_mm.dat", dtype=np.float32
    ).reshape((128, 128, 92), order="F")

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
# 5. Функция симуляции одного шага гантри с двумя головками
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

    data_manager = DataManager(filename=h5_filename, buffer_capacity=100_000)
    data_manager.handlers.append(
        HistoryAssemblerHandler(sensitive_volumes=[crystal_1, crystal_2], save_initial_states=True)
    )

    sim_manager = SimulationManager(
        scene=root_scene,
        data_manager=data_manager,
        start_time=0.0 * units.s,
        stop_time=30.0 * units.s,
        particles_number=particles_number,
        min_energy=1.0 * units.keV
    )

    rng = np.random.default_rng(seed)
    physics = PhysicsCompiler().compile(root_scene)
    propagator = ParticlePropagator(scene=root_scene, physics=physics, rng=rng)

    sim_manager.run(propagator)


# ------------------------------------------------------------------------------------------------------------------------
# 6. Главная точка входа: расчет сеток сканирования для двух головок
# ------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Сетки поворота станины:
    # 120 ракурсов при 2 детекторах: 60 шагов по 3.0° в диапазоне [0°, 180°)
    # 60 ракурсов при 2 детекторах: 30 шагов по 6.0° в диапазоне [0°, 180°)
    gantry_positions_120 = np.linspace(0.0, 180.0, 60, endpoint=False)
    gantry_positions_60 = np.linspace(0.0, 180.0, 30, endpoint=False)

    print("=== Двухкамерный протокол ОФЭКТ 161Tb (GE Discovery NM/CT 670) ===")
    print(f"Радионуклид: 161Tb, T1/2 = {TB161_HALF_LIFE / units.day:.2f} дней")
    print(f"Число спектральных линий: {len(TB161_ENERGY_SPECTRUM)}")
    for en_val, yield_val in TB161_ENERGY_SPECTRUM:
        print(f"  - E = {en_val / units.keV:.2f} кэВ, выход = {yield_val:.3f}%")
    print("\nЭнергетические окна исключены из симуляции (сохраняется полный энергетический спектр событий в HDF5).")
    print("Конфигурация детекторов: 2 противоположные головки под 180° (Detector_1, Detector_2)")
    print(f"Сетка 120 ракурсов: {len(gantry_positions_120)} шагов гантри с шагом {gantry_positions_120[1] - gantry_positions_120[0]:.1f}° (0.0° - 177.0°)")
    print(f"Сетка 60 ракурсов:  {len(gantry_positions_60)} шагов гантри с шагом {gantry_positions_60[1] - gantry_positions_60[0]:.1f}° (0.0° - 174.0°)")
    print("\nОриентировочное число частиц (суммарно по обеим головкам):")
    print("  - 120 ракурсов: 2 млн (~33 400 / шаг гантри) и 4 млн (~66 700 / шаг гантри)")
    print("  - 60 ракурсов:  1 млн (~33 400 / шаг гантри) и 2 млн (~66 700 / шаг гантри)")
