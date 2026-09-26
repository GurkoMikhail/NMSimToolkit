import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.particles.particles import ParticleBank
from core.other.typing_definitions import Species, Float, Length, Energy, Time
from core.other.vectors import Vector3D


def generate_test_data(n: int):
    """
    Генерация тестовых данных для инициализации банка частиц ParticleBank (SoA).
    """
    rng = np.random.default_rng(42)

    species = np.ones(n, dtype=Species)
    position = rng.random((n, 3), dtype=Length)
    direction = rng.random((n, 3), dtype=Float)
    norms = np.linalg.norm(direction, axis=1)
    direction = direction / norms[:, np.newaxis]

    energy = rng.random(n, dtype=Energy)
    emission_time = rng.random(n, dtype=Time)
    emission_position = rng.random((n, 3), dtype=Length)
    emission_direction = rng.random((n, 3), dtype=Float)
    norms_emit = np.linalg.norm(emission_direction, axis=1)
    emission_direction = emission_direction / norms_emit[:, np.newaxis]
    distance_traveled = rng.random(n, dtype=Length)

    return (
        species, position, direction, energy,
        emission_time, emission_position, emission_direction, distance_traveled
    )


def test_particle_bank_lifecycle_and_operations():
    """
    Тестирование жизненного цикла пула частиц SoA: инжекция, перемещение (move) и вращение (rotate).
    """
    n = 1000
    (species, position, direction, energy,
     emission_time, emission_position, emission_direction, distance_traveled) = generate_test_data(n)

    bank = ParticleBank.allocate(n)
    assert bank.capacity == n
    assert bank.count == 0

    pos_soa = Vector3D(position[:, 0], position[:, 1], position[:, 2])
    dir_soa = Vector3D(direction[:, 0], direction[:, 1], direction[:, 2])
    em_pos_soa = Vector3D(emission_position[:, 0], emission_position[:, 1], emission_position[:, 2])
    em_dir_soa = Vector3D(emission_direction[:, 0], emission_direction[:, 1], emission_direction[:, 2])

    target_indices = bank.inject_particles(
        species=species,
        position=pos_soa,
        direction=dir_soa,
        energy=energy,
        emission_time=emission_time,
        distance_traveled=distance_traveled
    )

    np.testing.assert_array_equal(target_indices, np.arange(n))
    assert bank.count == n

    # Проверка исходного состояния
    np.testing.assert_allclose(bank.state.position.x[target_indices], position[:, 0])
    np.testing.assert_allclose(bank.state.position.y[target_indices], position[:, 1])
    np.testing.assert_allclose(bank.state.position.z[target_indices], position[:, 2])

    # Проверка прямолинейного переноса (move)
    rng = np.random.default_rng(123)
    step_distances = rng.random(n, dtype=Length) * 10.0
    expected_x = position[:, 0] + direction[:, 0] * step_distances
    expected_y = position[:, 1] + direction[:, 1] * step_distances
    expected_z = position[:, 2] + direction[:, 2] * step_distances
    expected_dist = distance_traveled + step_distances

    bank.move(target_indices, step_distances)

    np.testing.assert_allclose(bank.state.position.x[target_indices], expected_x, rtol=1e-5)
    np.testing.assert_allclose(bank.state.position.y[target_indices], expected_y, rtol=1e-5)
    np.testing.assert_allclose(bank.state.position.z[target_indices], expected_z, rtol=1e-5)
    np.testing.assert_allclose(bank.state.distance_traveled[target_indices], expected_dist, rtol=1e-5)

    # Проверка вращения направляющего вектора (rotate)
    thetas = rng.random(n, dtype=Float) * np.pi
    phis = rng.random(n, dtype=Float) * 2 * np.pi

    bank.rotate(target_indices, thetas, phis)

    # Норма вектора направления должна оставаться единичной
    new_norms = np.sqrt(
        bank.state.direction.x[target_indices] ** 2 +
        bank.state.direction.y[target_indices] ** 2 +
        bank.state.direction.z[target_indices] ** 2
    )
    np.testing.assert_allclose(new_norms, 1.0, atol=1e-6)
