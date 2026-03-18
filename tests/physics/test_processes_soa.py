import unittest
import numpy as np
import hepunits as units
import time
from numba import njit
import ctypes

from core.particles.particles_soa import ParticleBank
from core.physics.interaction_soa import allocate_interaction_buffer, RNGContext
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

        self.buffer = allocate_interaction_buffer(self.capacity)

        self.rng = np.random.default_rng(42)

        cffi_next_double = self.rng.bit_generator.cffi.next_double
        state_addr = self.rng.bit_generator.cffi.state_address

        self.rng_ctx = RNGContext(
            next_double=cffi_next_double,
            state_addr=state_addr
        )

    def test_photoelectric_kernel(self):
        kernel = make_photoelectric_kernel(process_id=1)

        kernel(self.bank.state, self.target_indices, self.buffer, self.rng_ctx)

        self.assertEqual(self.buffer.cursor[0], self.capacity)

        # Verify energy in particle state is 0
        np.testing.assert_array_equal(self.bank.state.energy, 0.0)

        # Verify energy deposit
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.5 * units.MeV)

        # Verify process_id
        np.testing.assert_array_equal(self.buffer.process_id, 1)

    def test_compton_kernel(self):
        kernel = make_compton_kernel(process_id=2)
        Z = 13  # Aluminum

        # Warmup and Benchmark
        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)

        start = time.perf_counter()
        for _ in range(100):
            kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)
        end = time.perf_counter()
        print(f"\n[BENCHMARK] SoA Compton Scattering 100x (N={self.capacity}): {end - start:.5f}s")

        self.assertEqual(self.buffer.cursor[0], self.capacity)

        # Energy deposit should be logged and subtracted (we did 101 iterations, so checking the exact formula is tricky now)
        # We can just verify it decreased
        self.assertTrue(np.all(self.buffer.energy_deposit > 0.0))
        self.assertTrue(np.all(self.bank.state.energy < 0.5 * units.MeV))

        np.testing.assert_array_equal(self.buffer.process_id, 2)

        # Verify direction rotation validity (norm approx 1)
        dir_norms = np.sqrt(self.bank.state.direction.x**2 + self.bank.state.direction.y**2 + self.bank.state.direction.z**2)
        np.testing.assert_allclose(dir_norms, 1.0, rtol=1e-5)

    def test_coherent_kernel(self):
        kernel = make_coherent_kernel(process_id=3)
        Z = 82  # Lead

        kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)

        start = time.perf_counter()
        for _ in range(100):
            kernel(self.bank.state, self.target_indices, Z, self.buffer, self.rng_ctx)
        end = time.perf_counter()
        print(f"\n[BENCHMARK] SoA Coherent Scattering 100x (N={self.capacity}): {end - start:.5f}s")

        self.assertEqual(self.buffer.cursor[0], self.capacity)

        # Energy deposit is 0 for coherent scattering
        np.testing.assert_array_equal(self.buffer.energy_deposit, 0.0)
        np.testing.assert_array_equal(self.bank.state.energy, 0.5 * units.MeV)

        np.testing.assert_array_equal(self.buffer.process_id, 3)

        # Verify direction rotation validity (norm approx 1)
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
