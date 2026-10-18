"""
Тесты компилятора геометрии, трассировки лучей (SoA vs OOP) и воксельных объемов Вудкока.
"""

import os
import sys
import time
import tracemalloc

import numpy as np
import settings.database_setting as settings
from core.geometry.geometries import Box
from core.geometry.geometry_kernels import cast_path_kernel
from core.geometry.volumes import Volume
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import Material, MaterialArray
from core.other.typing_definitions import Energy, Float, Length, Species, Time
from core.other.vectors import Vector3DSoA
from core.particles.particles import ParticleBank


def create_complex_scene() -> Volume:
    """Создает трехуровневую иерархическую сцену для проверки компилятора геометрии."""
    mat_root = Material(name="RootMat", ID=1)
    mat_phantom = Material(name="PhantomMat", ID=2)
    mat_spine = Material(name="SpineMat", ID=3)
    mat_marrow = Material(name="MarrowMat", ID=4)
    mat_detector = Material(name="DetectorMat", ID=5)
    mat_pixel = Material(name="PixelMat", ID=6)

    root_vol = Volume(Box(100.0, 100.0, 100.0), mat_root, name="Root")

    phantom_vol = Volume(Box(40.0, 40.0, 40.0), mat_phantom, name="Phantom")
    phantom_vol.translate(-20.0, 0.0, 0.0)
    root_vol.add_child(phantom_vol)

    spine_vol = Volume(Box(10.0, 30.0, 10.0), mat_spine, name="Spine")
    spine_vol.translate(10.0, 0.0, 0.0)
    spine_vol.rotate(alpha=0.1, beta=0.2, gamma=0.0)
    phantom_vol.add_child(spine_vol)

    marrow_vol = Volume(Box(4.0, 20.0, 4.0), mat_marrow, name="Marrow")
    spine_vol.add_child(marrow_vol)

    detector_vol = Volume(Box(20.0, 40.0, 40.0), mat_detector, name="Detector")
    detector_vol.translate(30.0, 0.0, 0.0)
    detector_vol.rotate(alpha=0.0, beta=-0.1, gamma=0.0)
    root_vol.add_child(detector_vol)

    for i in range(3):
        for j in range(3):
            pixel_vol = Volume(Box(18.0, 10.0, 10.0), mat_pixel, name=f"Pixel_{i}_{j}")
            pixel_vol.translate(0.0, -15.0 + i * 15.0, -15.0 + j * 15.0)
            detector_vol.add_child(pixel_vol)

    return root_vol


def generate_rays(ray_count: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Генерирует изотропные лучи с использованием изолированного генератора случайных чисел."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-20.0, 20.0, size=(ray_count, 3)).astype(Float)

    phi = rng.uniform(0.0, 2.0 * np.pi, size=ray_count)
    costheta = rng.uniform(-1.0, 1.0, size=ray_count)
    theta = np.arccos(costheta)

    directions = np.zeros((ray_count, 3), dtype=Float)
    directions[:, 0] = np.sin(theta) * np.cos(phi)
    directions[:, 1] = np.sin(theta) * np.sin(phi)
    directions[:, 2] = np.cos(theta)

    return positions, directions


def test_geometry_compiler_fuzzing() -> None:
    """Проверка компиляции сложной иерархической сцены и трассировки случайных лучей через SoA-ядро."""
    scene = create_complex_scene()
    buffer = scene.geometry_buffer
    assert buffer.shape[0] == 14

    ray_count = 1000
    positions, directions = generate_rays(ray_count, seed=42)

    bank = ParticleBank.allocate(ray_count)
    species = np.ones(ray_count, dtype=Species)
    pos_soa = Vector3DSoA(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2])
    dir_soa = Vector3DSoA(x=directions[:, 0], y=directions[:, 1], z=directions[:, 2])
    energy = np.ones(ray_count, dtype=Energy)
    time_arr = np.zeros(ray_count, dtype=Time)
    dist = np.zeros(ray_count, dtype=Length)

    target_indices = bank.inject_particles(
        species=species,
        position=pos_soa,
        direction=dir_soa,
        energy=energy,
        emission_time=time_arr,
        distance_traveled=dist,
    )

    bank.navigation_state.boundary_distance[:] = 0.0
    cast_path_kernel(pos_soa, dir_soa, target_indices, buffer, bank.navigation_state)

    soa_distances = bank.navigation_state.boundary_distance
    assert np.all(soa_distances > 0.0)
    assert np.all(np.isfinite(soa_distances))


def test_woodcock_voxel_volume_centering_and_cache_invalidation() -> None:
    """
    Регрессионный тест WoodcockVoxelVolume:
    1. Эквивалентность индексации вокселей между _compile_cfunc (SoA) и _parametric_function (OOP)
       на границах вокселей и в центре координат (0, 0, 0) благодаря формуле floor((pos + 0.5 * size) / voxel_size).
    2. Автоматический сброс кэша мажорантного материала _cached_majorant_material и функции _cfunc
       при обновлении распределения материалов и размера вокселя.
    """
    water_mat = settings.material_database["Water, Liquid"]
    bone_mat = settings.material_database["Sodium Iodide"]
    lead_mat = settings.material_database["Pb"]

    # Сетка 4x4x4 с размером вокселя 2.0 мм -> габариты 8.0 x 8.0 x 8.0 мм, интервал [-4.0, 4.0]
    mat_dist = MaterialArray((4, 4, 4))
    mat_dist[:, :, :] = water_mat
    # В вокселе (2, 2, 2), покрывающем [0.0, 2.0) по всем осям, размещаем костную ткань
    mat_dist[2, 2, 2] = bone_mat
    # В вокселе (1, 1, 1), покрывающем [-2.0, 0.0) по всем осям, оставляем воду

    voxel_vol = WoodcockVoxelVolume(voxel_size=Float(2.0), material_distribution=mat_dist, name="TestVoxelVol")

    # Проверяем мажорантный материал до модификации (Cortical Bone тяжелее Water)
    assert voxel_vol.material == bone_mat
    assert voxel_vol._cached_majorant_material == bone_mat

    cfunc_lookup = voxel_vol._compile_cfunc()

    # Тестовые координаты вокруг центра (0, 0, 0) и границ вокселей
    test_points = np.array(
        [
            [-3.99, -3.99, -3.99],  # воксель (0, 0, 0) -> Water
            [-0.01, -0.01, -0.01],  # воксель (1, 1, 1) -> Water (сразу перед центром)
            [0.01, 0.01, 0.01],     # воксель (2, 2, 2) -> Cortical Bone (сразу за центром)
            [1.99, 1.99, 1.99],     # воксель (2, 2, 2) -> Cortical Bone
            [2.01, 2.01, 2.01],     # воксель (3, 3, 3) -> Water
        ],
        dtype=Float,
    )

    _, oop_materials = voxel_vol._parametric_function(test_points)
    for idx in range(test_points.shape[0]):
        pt_x, pt_y, pt_z = test_points[idx]
        soa_mat_id = cfunc_lookup(Float(pt_x), Float(pt_y), Float(pt_z))
        oop_mat_id = oop_materials.ID[idx]
        assert int(soa_mat_id) == int(oop_mat_id), (
            f"Расхождение ID материала в точке {test_points[idx]}: SoA={soa_mat_id}, OOP={oop_mat_id}"
        )

    assert int(cfunc_lookup(Float(-0.01), Float(-0.01), Float(-0.01))) == water_mat.ID
    assert int(cfunc_lookup(Float(0.01), Float(0.01), Float(0.01))) == bone_mat.ID

    # Проверка инвалидации кэша мажорантного материала и _cfunc при замене material_distribution
    _ = voxel_vol.material_cfunc
    assert voxel_vol._cfunc is not None
    new_dist = MaterialArray((4, 4, 4))
    new_dist[:, :, :] = water_mat
    new_dist[0, 0, 0] = lead_mat
    voxel_vol.material_distribution = new_dist

    assert voxel_vol._cached_majorant_material is None
    assert voxel_vol._cfunc is None
    assert voxel_vol.material == lead_mat
    assert voxel_vol._cached_majorant_material == lead_mat

    # Проверка сброса кэша _cfunc при изменении voxel_size
    _ = voxel_vol.material_cfunc
    assert voxel_vol._cfunc is not None
    voxel_vol.voxel_size = Float(3.0)
    assert voxel_vol._cfunc is None


def main() -> None:
    """Точка входа для ручного профилирования трассировки лучей."""
    scene = create_complex_scene()
    buffer = scene.geometry_buffer
    ray_count = 10_000
    positions, directions = generate_rays(ray_count, seed=42)

    bank = ParticleBank.allocate(ray_count)
    species = np.ones(ray_count, dtype=Species)
    pos_soa = Vector3DSoA(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2])
    dir_soa = Vector3DSoA(x=directions[:, 0], y=directions[:, 1], z=directions[:, 2])
    energy = np.ones(ray_count, dtype=Energy)
    time_arr = np.zeros(ray_count, dtype=Time)
    dist = np.zeros(ray_count, dtype=Length)

    target_indices = bank.inject_particles(
        species=species,
        position=pos_soa,
        direction=dir_soa,
        energy=energy,
        emission_time=time_arr,
        distance_traveled=dist,
    )

    tracemalloc.start()
    t0_soa = time.perf_counter()
    cast_path_kernel(pos_soa, dir_soa, target_indices, buffer, bank.navigation_state)
    t1_soa = time.perf_counter()
    tracemalloc.stop()
    print(f"SoA: {t1_soa - t0_soa:.4f}s, Rays: {ray_count}")


if __name__ == "__main__":
    main()


