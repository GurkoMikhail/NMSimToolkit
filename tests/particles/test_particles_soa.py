import numpy as np
import time

from core.particles.particles import ParticleArray
from core.particles.particles_soa import ParticleBank
from core.other.typing_definitions import Species, Float, Length, Energy, Time


def generate_test_data(n: int):
    """
    Generates test data for initializing both AoS and SoA models.
    """
    rng = np.random.default_rng(42)

    species = np.ones(n, dtype=Species)
    position = rng.random((n, 3), dtype=Length)
    direction = rng.random((n, 3), dtype=Float)
    # Normalize directions
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


def test_particles_soa_equivalence():
    n = 1000
    (species, position, direction, energy,
     emission_time, emission_position, emission_direction, distance_traveled) = generate_test_data(n)

    # --- 1. Initialize AoS (Old) ---
    old_array = ParticleArray.create(
        species=species,
        position=position,
        direction=direction,
        energy=energy,
        emission_time=emission_time,
        emission_position=emission_position,
        emission_direction=emission_direction,
        distance_traveled=distance_traveled
    )

    # --- 2. Initialize SoA (New) ---
    bank = ParticleBank(capacity=n)

    from core.other.vectors_soa import Vector3DSoA
    # Convert vectors for SoA injection
    pos_soa = Vector3DSoA(position[:, 0], position[:, 1], position[:, 2])
    dir_soa = Vector3DSoA(direction[:, 0], direction[:, 1], direction[:, 2])
    em_pos_soa = Vector3DSoA(emission_position[:, 0], emission_position[:, 1], emission_position[:, 2])
    em_dir_soa = Vector3DSoA(emission_direction[:, 0], emission_direction[:, 1], emission_direction[:, 2])

    target_indices = bank.inject_particles(
        species=species,
        position=pos_soa,
        direction=dir_soa,
        energy=energy,
        emission_time=emission_time,
        emission_position=em_pos_soa,
        emission_direction=em_dir_soa,
        distance_traveled=distance_traveled
    )

    # Assert successful injection
    np.testing.assert_array_equal(target_indices, np.arange(n))

    # --- 3. Verify Equality at Init ---
    np.testing.assert_allclose(old_array.distance_traveled, bank.state.distance_traveled[target_indices])

    np.testing.assert_allclose(old_array.position[:, 0], bank.state.position.x[target_indices])
    np.testing.assert_allclose(old_array.position[:, 1], bank.state.position.y[target_indices])
    np.testing.assert_allclose(old_array.position[:, 2], bank.state.position.z[target_indices])

    np.testing.assert_allclose(old_array.direction[:, 0], bank.state.direction.x[target_indices])
    np.testing.assert_allclose(old_array.direction[:, 1], bank.state.direction.y[target_indices])
    np.testing.assert_allclose(old_array.direction[:, 2], bank.state.direction.z[target_indices])

    # --- 4. Test Move ---
    rng = np.random.default_rng(123)
    distances = rng.random(n, dtype=Length)

    old_array.move(distances)
    bank.move(target_indices, distances)

    np.testing.assert_allclose(old_array.distance_traveled, bank.state.distance_traveled[target_indices])
    np.testing.assert_allclose(old_array.position[:, 0], bank.state.position.x[target_indices])
    np.testing.assert_allclose(old_array.position[:, 1], bank.state.position.y[target_indices])
    np.testing.assert_allclose(old_array.position[:, 2], bank.state.position.z[target_indices])

    # --- 5. Test Rotate ---
    thetas = rng.random(n, dtype=Float) * np.pi
    phis = rng.random(n, dtype=Float) * 2 * np.pi

    old_array.rotate(thetas, phis)
    bank.rotate(target_indices, thetas, phis)

    np.testing.assert_allclose(old_array.direction[:, 0], bank.state.direction.x[target_indices])
    np.testing.assert_allclose(old_array.direction[:, 1], bank.state.direction.y[target_indices])
    np.testing.assert_allclose(old_array.direction[:, 2], bank.state.direction.z[target_indices])


def test_particles_soa_benchmark():
    n = 10**6
    (species, position, direction, energy,
     emission_time, emission_position, emission_direction, distance_traveled) = generate_test_data(n)

    # Create test vectors
    rng = np.random.default_rng(777)
    distances = rng.random(n, dtype=Length)
    thetas = rng.random(n, dtype=Float) * np.pi
    phis = rng.random(n, dtype=Float) * 2 * np.pi

    # ---------------------------------------------
    # 1. Benchmark AoS
    # ---------------------------------------------
    old_array = ParticleArray.create(
        species=species,
        position=position,
        direction=direction,
        energy=energy,
        emission_time=emission_time,
        emission_position=emission_position,
        emission_direction=emission_direction,
        distance_traveled=distance_traveled
    )

    start_time = time.perf_counter()
    old_array.move(distances)
    old_array.rotate(thetas, phis)
    aos_time = time.perf_counter() - start_time

    # ---------------------------------------------
    # 2. Benchmark SoA
    # ---------------------------------------------
    bank = ParticleBank(capacity=n)

    from core.other.vectors_soa import Vector3DSoA
    pos_soa = Vector3DSoA(position[:, 0], position[:, 1], position[:, 2])
    dir_soa = Vector3DSoA(direction[:, 0], direction[:, 1], direction[:, 2])
    em_pos_soa = Vector3DSoA(emission_position[:, 0], emission_position[:, 1], emission_position[:, 2])
    em_dir_soa = Vector3DSoA(emission_direction[:, 0], emission_direction[:, 1], emission_direction[:, 2])

    target_indices = bank.inject_particles(
        species=species,
        position=pos_soa,
        direction=dir_soa,
        energy=energy,
        emission_time=emission_time,
        emission_position=em_pos_soa,
        emission_direction=em_dir_soa,
        distance_traveled=distance_traveled
    )

    # Compile first (Warm-up JIT)
    bank.move(target_indices[:1], distances[:1])
    bank.rotate(target_indices[:1], thetas[:1], phis[:1])

    start_time = time.perf_counter()
    bank.move(target_indices, distances)
    bank.rotate(target_indices, thetas, phis)
    soa_time = time.perf_counter() - start_time

    print(f"\n[Benchmark] AoS Time: {aos_time:.4f} s | SoA Time: {soa_time:.4f} s")
    print(f"[Benchmark] Speedup: {aos_time / soa_time:.2f}x")

    # Require at least some positive speedup (SoA should be better or equal at scale)
    assert soa_time < aos_time * 1.5, "SoA implementation is unexpectedly slow."
