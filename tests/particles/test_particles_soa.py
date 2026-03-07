import time
import numpy as np

from core.particles.particles import ParticleArray
from core.particles.particles_soa import ParticleBank

def test_soa_equivalence_and_benchmark():
    n_particles = 1_000_000

    # 1. Prepare data
    species = np.ones(n_particles, dtype=np.uint8)
    pos_x = np.random.uniform(-10, 10, n_particles)
    pos_y = np.random.uniform(-10, 10, n_particles)
    pos_z = np.random.uniform(-10, 10, n_particles)

    dir_x = np.random.uniform(-1, 1, n_particles)
    dir_y = np.random.uniform(-1, 1, n_particles)
    dir_z = np.random.uniform(-1, 1, n_particles)
    # normalize directions
    norms = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    dir_x /= norms
    dir_y /= norms
    dir_z /= norms

    energy = np.ones(n_particles, dtype=np.float64) * 140.0
    time_arr = np.zeros(n_particles, dtype=np.float64)

    # Random distances, thetas, phis for move and rotate
    distances = np.random.uniform(0.1, 5.0, n_particles)
    thetas = np.random.uniform(0, np.pi, n_particles)
    phis = np.random.uniform(0, 2 * np.pi, n_particles)

    # ------------------ AOS (Old) ------------------
    pos_matrix = np.column_stack((pos_x, pos_y, pos_z))
    dir_matrix = np.column_stack((dir_x, dir_y, dir_z))

    aos_array = ParticleArray.create(
        species=species,
        position=pos_matrix.copy(),
        direction=dir_matrix.copy(),
        energy=energy.copy(),
        emission_time=time_arr.copy()
    )

    t0_aos_move = time.perf_counter()
    aos_array.move(distances)
    t1_aos_move = time.perf_counter()

    t0_aos_rotate = time.perf_counter()
    aos_array.rotate(thetas, phis)
    t1_aos_rotate = time.perf_counter()

    aos_move_time = t1_aos_move - t0_aos_move
    aos_rotate_time = t1_aos_rotate - t0_aos_rotate

    # ------------------ SOA (New) ------------------
    from core.particles.vector3d_soa import Vector3DSoA

    pos_soa = Vector3DSoA(pos_x.copy(), pos_y.copy(), pos_z.copy())
    pos_soa._validate()

    dir_soa = Vector3DSoA(dir_x.copy(), dir_y.copy(), dir_z.copy())
    dir_soa._validate()

    bank = ParticleBank(capacity=n_particles)
    injected_indices = bank.inject(
        species=species,
        position=pos_soa,
        direction=dir_soa,
        energy=energy.copy(),
        emission_time=time_arr.copy()
    )

    # Numba compilation overhead: run once with 1 particle
    bank.move(injected_indices[:1], distances[:1])
    bank.rotate(injected_indices[:1], thetas[:1], phis[:1])

    # Reset state for correct benchmark
    bank.state.position.x[:] = pos_x.copy()
    bank.state.position.y[:] = pos_y.copy()
    bank.state.position.z[:] = pos_z.copy()
    bank.state.direction.x[:] = dir_x.copy()
    bank.state.direction.y[:] = dir_y.copy()
    bank.state.direction.z[:] = dir_z.copy()
    bank.state.distance_traveled[:] = 0.0

    t0_soa_move = time.perf_counter()
    bank.move(injected_indices, distances)
    t1_soa_move = time.perf_counter()

    t0_soa_rotate = time.perf_counter()
    bank.rotate(injected_indices, thetas, phis)
    t1_soa_rotate = time.perf_counter()

    soa_move_time = t1_soa_move - t0_soa_move
    soa_rotate_time = t1_soa_rotate - t0_soa_rotate

    # ------------------ EQUIVALENCE CHECK ------------------
    np.testing.assert_allclose(aos_array.position[:, 0], bank.state.position.x, atol=1e-12)
    np.testing.assert_allclose(aos_array.position[:, 1], bank.state.position.y, atol=1e-12)
    np.testing.assert_allclose(aos_array.position[:, 2], bank.state.position.z, atol=1e-12)

    np.testing.assert_allclose(aos_array.direction[:, 0], bank.state.direction.x, atol=1e-12)
    np.testing.assert_allclose(aos_array.direction[:, 1], bank.state.direction.y, atol=1e-12)
    np.testing.assert_allclose(aos_array.direction[:, 2], bank.state.direction.z, atol=1e-12)

    np.testing.assert_allclose(aos_array.distance_traveled, bank.state.distance_traveled, atol=1e-12)

    # Print results
    print(f"\\n--- BENCHMARK ({n_particles} particles) ---")
    print(f"AoS Move:   {aos_move_time:.5f} s")
    print(f"SoA Move:   {soa_move_time:.5f} s (Speedup: {aos_move_time / soa_move_time:.2f}x)")
    print(f"AoS Rotate: {aos_rotate_time:.5f} s")
    print(f"SoA Rotate: {soa_rotate_time:.5f} s (Speedup: {aos_rotate_time / soa_rotate_time:.2f}x)")

if __name__ == "__main__":
    test_soa_equivalence_and_benchmark()
