import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tracemalloc
import unittest
import numpy as np
import hepunits as units
import time
from numba import njit
import ctypes

from core.particles.particles_soa import ParticleBank
from core.physics.interaction_soa import InteractionBuffer, RNGContext
from core.physics.processes_soa_kernels import make_photoelectric_kernel, make_compton_kernel, make_coherent_kernel

# Old implementations for testing math correctness
import core.physics.g4compton as old_compton
import core.physics.g4coherent as old_coherent
from core.other.vectors_soa import _rotate_direction_scalar


class TestProcessesSoA(unittest.TestCase):
    def setUp(self):
        self.capacity = 1000
        self.bank = ParticleBank(self.capacity)

        # Inject some particles
        species = np.full(self.capacity, 1, dtype=np.uint8)
        energy = np.full(self.capacity, 0.5 * units.MeV)

        pos = self.bank.state.position
        pos.x[:] = np.random.rand(self.capacity)
        pos.y[:] = np.random.rand(self.capacity)
        pos.z[:] = np.random.rand(self.capacity)

        dir_vec = self.bank.state.direction
        dir_vec.x[:] = 0.0
        dir_vec.y[:] = 0.0
        dir_vec.z[:] = 1.0

        # Simulate active
        self.bank._state.is_active[:] = True
        self.bank._state.ID[:] = np.arange(self.capacity, dtype=np.uint64)
        self.bank._state.energy[:] = energy
        self.target_indices = np.arange(self.capacity, dtype=np.int64)

        self.buffer = InteractionBuffer.allocate(self.capacity)

        self.rng = np.random.default_rng(42)

        cffi_next_double = self.rng.bit_generator.cffi.next_double
        state_addr = self.rng.bit_generator.cffi.state_address

        self.rng_ctx = RNGContext(
            next_double=cffi_next_double,
            state_addr=state_addr
        )

    def test_photoelectric_kernel(self):
        kernel = make_photoelectric_kernel(process_id=1)
        Z = np.int8(13)

        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)

        self.assertEqual(self.buffer.cursor[0], self.capacity)

        # Verify energy in particle state is 0
        np.testing.assert_array_equal(self.bank.state.energy, 0.0)

        # Verify energy deposit
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.5 * units.MeV)

        # Verify process_id
        np.testing.assert_array_equal(self.buffer.process_id, 1)

    def test_compton_kernel_equivalence_and_performance(self):
        kernel = make_compton_kernel(process_id=2)
        Z = np.int8(13)  # Aluminum

        # 1. Physics Equivalence
        old_rng = np.random.default_rng(1234)
        old_theta_generator = old_compton.initialize(old_rng)

        # Old implementation uses scalar values and `@vectorize`, so we compute N thetas:
        # Note: since the new logic uses multiple RNG calls per particle (theta, then phi),
        # we can't easily reproduce the exact sequence using `old_theta_generator` on an array
        # unless `old_theta_generator` does precisely the same RNG calls per element.
        # Actually `old_theta_generator` also generates theta using the same rejection sampling!
        # Let's verify equivalence:
        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        Z_arr = np.full(self.capacity, 13)
        theta_old = old_theta_generator(energy_arr, Z_arr)

        # New implementation with identically seeded RNG
        new_rng = np.random.default_rng(1234)
        cffi_next_double = new_rng.bit_generator.cffi.next_double
        state_addr = new_rng.bit_generator.cffi.state_address
        new_rng_ctx = RNGContext(next_double=cffi_next_double, state_addr=state_addr)

        # Reset bank states for equivalence check
        self.bank.state.energy[:] = energy_arr

        # To perfectly match the old stream, the old logic ONLY calculates theta and consumes RNG state.
        # The NEW logic calculates theta AND THEN phi! This means particle 2's theta in the new stream
        # will use RNG states shifted by phi calculations of particle 1!
        # So we can't do a straightforward sequence comparison on arrays of N > 1 unless we test N=1
        # or we test the theta generation independently. Let's test a single particle for exact equivalence.
        old_rng_1 = np.random.default_rng(777)
        old_theta_generator_1 = old_compton.initialize(old_rng_1)
        theta_old_single = old_theta_generator_1(np.float64(0.5 * units.MeV), np.int64(13))

        new_rng_1 = np.random.default_rng(777)
        cffi_next_double_1 = new_rng_1.bit_generator.cffi.next_double
        state_addr_1 = new_rng_1.bit_generator.cffi.state_address
        new_rng_ctx_1 = RNGContext(next_double=cffi_next_double_1, state_addr=state_addr_1)

        from core.physics.g4compton_soa import _generate_compton_theta_scalar
        theta_new_single = _generate_compton_theta_scalar(0.5 * units.MeV, np.int8(13), new_rng_ctx_1)

        self.assertAlmostEqual(theta_old_single, theta_new_single, places=5)

    def test_coherent_kernel_equivalence(self):
        old_rng_1 = np.random.default_rng(777)
        old_theta_generator_1 = old_coherent.initialize(old_rng_1)
        theta_old_single = old_theta_generator_1(np.float64(0.5 * units.MeV), np.int64(82))

        new_rng_1 = np.random.default_rng(777)
        cffi_next_double_1 = new_rng_1.bit_generator.cffi.next_double
        state_addr_1 = new_rng_1.bit_generator.cffi.state_address
        new_rng_ctx_1 = RNGContext(next_double=cffi_next_double_1, state_addr=state_addr_1)

        from core.physics.g4coherent_soa import _generate_coherent_theta_scalar
        theta_new_single = _generate_coherent_theta_scalar(0.5 * units.MeV, np.int8(82), new_rng_ctx_1)

        self.assertAlmostEqual(theta_old_single, theta_new_single, places=5)

    def test_compton_kernel(self):
        kernel = make_compton_kernel(process_id=2)
        Z = np.int8(13)  # Aluminum

        # 1. Memory Consumption Benchmark
        # First run to compile the kernel and allocate Numba overhead
        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)

        tracemalloc.start()
        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Since we're in Numba njit, memory should be zero (or near-zero from python wrapper overhead)
        print(f"\n[RAM] SoA Compton Scattering Peak Memory: {peak / 10**6:.6f} MB")
        # Assert minimal memory overhead (under 100 KB is safe for wrapper/dispatcher overhead)
        self.assertTrue(peak < 100_000)

        # 2. Speed Benchmark
        start = time.perf_counter()
        for _ in range(100):
            kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)
        end = time.perf_counter()
        print(f"[BENCHMARK] NEW SoA Compton Scattering 100x (N={self.capacity}): {end - start:.5f}s")

        old_rng = np.random.default_rng(1234)
        old_theta_generator = old_compton.initialize(old_rng)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        Z_arr = np.full(self.capacity, 13)
        # compile old
        old_theta_generator(energy_arr, Z_arr)

        tracemalloc.start()
        start = time.perf_counter()
        for _ in range(100):
            theta_old = old_theta_generator(energy_arr, Z_arr)
            phi = np.pi * (old_rng.random(self.capacity) * 2 - 1)
        end = time.perf_counter()
        current, old_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[BENCHMARK] OLD vectorize Compton Scattering 100x (N={self.capacity}): {end - start:.5f}s")
        print(f"[RAM] OLD vectorize Compton Scattering Peak Memory: {old_peak / 10**6:.6f} MB")

        # 3. Correctness
        self.assertEqual(self.buffer.cursor[0], self.capacity * 102)
        self.assertTrue(np.all(self.buffer.energy_deposit > 0.0))
        self.assertTrue(np.all(self.bank.state.energy < 0.5 * units.MeV))
        np.testing.assert_array_equal(self.buffer.process_id, 2)
        dir_norms = np.sqrt(self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2)
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_coherent_kernel(self):
        kernel = make_coherent_kernel(process_id=3)
        Z = np.int8(82)  # Lead

        # Warmup
        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)

        tracemalloc.start()
        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n[RAM] SoA Coherent Scattering Peak Memory: {peak / 10**6:.6f} MB")
        self.assertTrue(peak < 100_000)

        start = time.perf_counter()
        for _ in range(100):
            kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)
        end = time.perf_counter()
        print(f"[BENCHMARK] NEW SoA Coherent Scattering 100x (N={self.capacity}): {end - start:.5f}s")

        old_rng = np.random.default_rng(1234)
        old_theta_generator = old_coherent.initialize(old_rng)
        energy_arr = np.full(self.capacity, 0.5 * units.MeV)
        Z_arr = np.full(self.capacity, 82)
        # compile old
        old_theta_generator(energy_arr, Z_arr)

        tracemalloc.start()
        start = time.perf_counter()
        for _ in range(100):
            theta_old = old_theta_generator(energy_arr, Z_arr)
            phi = np.pi * (old_rng.random(self.capacity) * 2 - 1)
        end = time.perf_counter()
        current, old_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[BENCHMARK] OLD vectorize Coherent Scattering 100x (N={self.capacity}): {end - start:.5f}s")
        print(f"[RAM] OLD vectorize Coherent Scattering Peak Memory: {old_peak / 10**6:.6f} MB")

        self.assertEqual(self.buffer.cursor[0], self.capacity * 102)
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.0)
        np.testing.assert_array_equal(self.bank.state.energy, 0.5 * units.MeV)
        np.testing.assert_array_equal(self.buffer.process_id, 3)

        dir_norms = np.sqrt(self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2)
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_rotate_direction_scalar(self):
        # Base vector
        dir_x, dir_y, dir_z = 0.0, 0.0, 1.0
        theta, phi = np.pi/4, np.pi/2

        nx, ny, nz = _rotate_direction_scalar(dir_x, dir_y, dir_z, theta, phi)

        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        # For an initial (0, 0, 1) and rotating theta=pi/4, phi=pi/2
        # x' ~ 0
        # y' = sin(pi/4) ~ 0.707
        # z' = cos(pi/4) ~ 0.707
        self.assertAlmostEqual(nx, 0.0, places=5)
        self.assertAlmostEqual(ny, np.sin(theta), places=5)
        self.assertAlmostEqual(nz, np.cos(theta), places=5)


if __name__ == '__main__':
    unittest.main()
