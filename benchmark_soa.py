import time
import numpy as np
import hepunits as units
from numpy.typing import NDArray

from core.particles.particles import ParticleArray, Species
from core.particles.soa_particles import ParticleArraySoA
from core.other.typing_definitions import Float, Energy

from numba import njit

import core.physics.g4compton as g4compton


# --- Extracting calculation functions to test ---

@njit(cache=True)
def rotate_soa(particles: ParticleArraySoA, theta: NDArray[Float], phi: NDArray[Float]) -> None:
    """
    Повернуть направления частиц (SoA/1D arrays version)
    direction shape (N, 3), theta/phi shape (N,)
    Accepts ParticleArraySoA directly!
    """
    direction = particles.direction
    n = direction.shape[0]
    for i in range(n):
        cos_theta = np.cos(theta[i])
        sin_theta = np.sin(theta[i])
        delta1 = sin_theta * np.cos(phi[i])
        delta2 = sin_theta * np.sin(phi[i])

        dir_z = direction[i, 2]
        delta = 1.0 - 2.0 * (dir_z < 0)

        b = direction[i, 0] * delta1 + direction[i, 1] * delta2
        tmp = cos_theta - b / (1.0 + np.abs(dir_z))

        direction[i, 0] = direction[i, 0] * tmp + delta1
        direction[i, 1] = direction[i, 1] * tmp + delta2
        direction[i, 2] = dir_z * cos_theta - delta * b


# We will benchmark the existing AoS rotate vs SoA rotate, and AoS theta_generation vs SoA theta_generation.

def run_benchmark(n_particles: int = 1_000_000):
    print(f"=== Starting Memory Refactoring Benchmark (AoS vs SoA) ===")
    print(f"Number of particles: {n_particles}\n")

    # 1. Initialize data
    np.random.seed(42)
    energies = np.random.uniform(10 * units.keV, 500 * units.keV, size=n_particles).astype(Energy)

    # Generate random directions normalized
    directions = np.random.uniform(-1, 1, size=(n_particles, 3)).astype(Float)
    norms = np.linalg.norm(directions, axis=1)
    directions = directions / norms[:, np.newaxis]

    positions = np.zeros((n_particles, 3), dtype=Float)
    species = np.zeros(n_particles, dtype=Species)

    theta_angles = np.random.uniform(0, np.pi, size=n_particles).astype(Float)
    phi_angles = np.random.uniform(-np.pi, np.pi, size=n_particles).astype(Float)

    # 2. Setup AoS (Array of Structures)
    aos_particles = ParticleArray.create(
        species=species,
        position=positions,
        direction=np.copy(directions),
        energy=np.copy(energies)
    )

    # 3. Setup SoA (Structure of Arrays)
    soa_particles = ParticleArraySoA.create(
        species=species,
        position=positions,
        direction=np.copy(directions),
        energy=np.copy(energies)
    )

    # Compile Numba functions and other things ahead of timing to ensure fairness

    # Warmup AoS
    aos_warmup = ParticleArray.create(
        species=species[:10],
        position=positions[:10],
        direction=np.copy(directions[:10]),
        energy=np.copy(energies[:10])
    )
    aos_warmup.rotate(theta_angles[:10], phi_angles[:10])

    # Warmup SoA
    soa_warmup = ParticleArraySoA.create(
        species=species[:10],
        position=positions[:10],
        direction=np.copy(directions[:10]),
        energy=np.copy(energies[:10])
    )
    rotate_soa(soa_warmup, theta_angles[:10], phi_angles[:10])

    # === Benchmark: Rotate ===
    print("--- Benchmark: rotation (Mathematical crunching) ---")

    # AoS Rotate
    start_time = time.perf_counter()
    aos_particles.rotate(theta_angles, phi_angles)
    aos_time = time.perf_counter() - start_time

    # SoA Rotate (using Numba)
    start_time = time.perf_counter()
    rotate_soa(soa_particles, theta_angles, phi_angles)
    soa_time = time.perf_counter() - start_time

    print(f"AoS rotate time: {aos_time:.5f} s")
    print(f"SoA rotate time: {soa_time:.5f} s")
    print(f"Speedup (Rotate): {aos_time / soa_time:.2f}x\n")

    # Verify identical results
    np.testing.assert_allclose(aos_particles.direction, soa_particles.direction, rtol=1e-6, atol=1e-8)
    print("[PASS] Rotate Physics exact match proven!\n")


    # === Benchmark: Vectorized generator (Theta Generation) ===
    print("--- Benchmark: theta_generator (Numba Ufunc with memory scan) ---")
    Z = 6  # Carbon, arbitrary

    # Warmup Theta Generator
    rng = np.random.default_rng(42)
    theta_generator = g4compton.initialize(rng)
    _ = theta_generator(aos_warmup.energy, Z)

    rng = np.random.default_rng(42)
    theta_generator = g4compton.initialize(rng)
    _ = theta_generator(soa_warmup.energy, Z)

    # AoS Generator
    rng = np.random.default_rng(42)
    theta_generator = g4compton.initialize(rng)

    start_time = time.perf_counter()
    theta_aos = theta_generator(aos_particles.energy, Z)
    aos_theta_time = time.perf_counter() - start_time

    # SoA Generator
    # Reset RNG for identical outputs
    rng = np.random.default_rng(42)
    theta_generator = g4compton.initialize(rng)

    start_time = time.perf_counter()
    theta_soa = theta_generator(soa_particles.energy, Z)
    soa_theta_time = time.perf_counter() - start_time

    print(f"AoS theta_generator time: {aos_theta_time:.5f} s")
    print(f"SoA theta_generator time: {soa_theta_time:.5f} s")
    print(f"Speedup (Theta Gen): {aos_theta_time / soa_theta_time:.2f}x\n")

    np.testing.assert_allclose(theta_aos, theta_soa, rtol=1e-6, atol=1e-8)
    print("[PASS] Theta Gen exact match proven!\n")


if __name__ == "__main__":
    run_benchmark(1_000_000)
