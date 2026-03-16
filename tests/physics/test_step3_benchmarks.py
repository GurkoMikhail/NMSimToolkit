import os
import sys
import time
import unittest

import numpy as np
import hepunits as units

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set dummy environment variables to bypass telegram init errors
os.environ['TELEGRAM_BOT_TOKEN'] = 'dummy'
os.environ['TELEGRAM_USER_ID'] = 'dummy'

import settings.database_setting as settings
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Float
from core.physics.physics_compiler import PhysicsCompiler
from core.physics.physics_kernels import _get_macroscopic_cross_sections
from core.physics.processes import PhotoelectricEffect, ComptonScattering, CoherentScattering


class TestStep3Benchmarks(unittest.TestCase):

    def test_macroscopic_cross_sections_benchmark(self):
        """
        Verify that the Numba in-place interpolation exactly matches
        the OOP Process.get_LAC() implementation and measure the speedup.
        """
        material = settings.material_database['Pb']
        mat_id = material.ID

        # Initialize processes
        processes = [
            PhotoelectricEffect(settings.attenuation_database),
            ComptonScattering(settings.attenuation_database),
            CoherentScattering(settings.attenuation_database)
        ]

        # Compile physics buffer
        compiler = PhysicsCompiler()
        # Create a dummy volume for compilation just to pass the unique materials
        from core.geometry.geometries import Box
        from core.geometry.volumes import Volume
        dummy_vol = Volume(Box(10.0, 10.0, 10.0), material)

        physics_buffer = compiler.compile_scene(dummy_vol, processes)
        material_bank = physics_buffer.material_bank

        # Test data
        num_particles = 100_000
        energies = np.random.uniform(0.01 * units.MeV, 1.0 * units.MeV, num_particles).astype(Float)

        # 1. Benchmark Original OOP Approach
        start_time = time.perf_counter()
        oop_lacs = np.zeros((num_particles, len(processes)), dtype=Float)

        # We need a dummy particle array to pass to Process.get_LAC()
        class DummyParticleArray:
            def __init__(self, energy_val):
                self.energy = energy_val

        # To simulate OOP, we'd normally call process.get_LAC on the whole array
        dummy_particles = DummyParticleArray(energies)
        for p_idx, process in enumerate(processes):
            oop_lacs[:, p_idx] = process.get_LAC(dummy_particles, material)

        oop_time = time.perf_counter() - start_time

        # 2. Benchmark Numba C-Array Approach
        start_time = time.perf_counter()
        numba_lacs = np.zeros((num_particles, len(processes)), dtype=Float)

        # Compile the Numba kernel (run once to trigger JIT compilation)
        _get_macroscopic_cross_sections(Float(0.1), mat_id, material_bank, numba_lacs[0])

        from numba import njit

        @njit
        def numba_loop(energies_arr, mat_id, bank, out_lacs):
            for i in range(energies_arr.shape[0]):
                _get_macroscopic_cross_sections(energies_arr[i], mat_id, bank, out_lacs[i])

        # Compile run
        numba_loop(energies[:1], mat_id, material_bank, numba_lacs[:1])

        start_numba = time.perf_counter()
        numba_loop(energies, mat_id, material_bank, numba_lacs)
        numba_time = time.perf_counter() - start_numba

        print(f"\n--- Interpolation Benchmark ({num_particles} samples) ---")
        print(f"OOP Time:   {oop_time:.4f} s")
        print(f"Numba Time: {numba_time:.4f} s")
        print(f"Speedup:    {oop_time / numba_time:.2f}x")

        # Verify exact correctness
        np.testing.assert_allclose(oop_lacs, numba_lacs, rtol=1e-5, atol=1e-8, err_msg="Numba LAC interpolation does not match OOP implementation")


    def test_parametric_collimator_benchmark(self):
        """
        Verify that the compiled @cfunc hexagon logic matches the vectorized NumPy logic
        and measure execution speed.
        """
        size = np.array([100.0, 100.0, 20.0])
        hole_diameter = 1.0
        septa = 0.2
        material = settings.material_database['Pb']

        collimator = ParametricParallelCollimator(size, hole_diameter, septa, material)

        cfunc_callable = collimator.material_cfunc

        num_points = 100_000
        # Random positions inside the collimator XY plane
        positions = np.random.uniform(-50.0, 50.0, (num_points, 3)).astype(Float)

        # 1. Benchmark Vectorized OOP NumPy
        start_time = time.perf_counter()
        is_vacuum_mask, _ = collimator._parametric_function(positions)
        oop_time = time.perf_counter() - start_time

        # 2. Benchmark Numba @cfunc Loop
        from numba import njit
        import ctypes

        # Numba can call CFUNCTYPE natively if passed in, we just need to wrap it.
        # Wait, you can just call it from Python! Oh, the user wants to benchmark
        # Numba's speed calling it. We can just use the memory address inside njit
        # but let's try just letting Numba parse the CFUNCTYPE.

        @njit
        def cfunc_loop(c_ptr, pos, out_ids):
            for i in range(pos.shape[0]):
                out_ids[i] = c_ptr(pos[i, 0], pos[i, 1], pos[i, 2])

        numba_mat_ids = np.zeros(num_points, dtype=np.int64)

        # Compile and run
        addr = ctypes.cast(cfunc_callable, ctypes.c_void_p).value
        # Actually passing CFUNCTYPE directly to njit works in recent Numba versions
        cfunc_loop(cfunc_callable, positions[:1], numba_mat_ids[:1])

        start_time = time.perf_counter()
        cfunc_loop(cfunc_callable, positions, numba_mat_ids)
        numba_time = time.perf_counter() - start_time

        print(f"\n--- Parametric Collimator Benchmark ({num_points} samples) ---")
        print(f"OOP Numpy Time: {oop_time:.4f} s")
        print(f"Numba CFunc Loop Time:  {numba_time:.4f} s")
        print(f"Speedup:        {oop_time / numba_time:.2f}x")

        # Verify correctness
        # The Numba cfunc returns vac_id if true, else mat_id
        vac_id = collimator._vacuum.ID
        numba_vacuum_mask = (numba_mat_ids == vac_id)

        np.testing.assert_array_equal(is_vacuum_mask, numba_vacuum_mask, err_msg="Numba CFunc logic differs from OOP Numpy logic in ParametricParallelCollimator")


    def test_voxel_volume_benchmark(self):
        """
        Verify the WoodcockVoxelVolume flat lookup logic matches the NumPy indexing.
        """
        voxel_size = Float(2.0)
        shape = (10, 10, 10)
        mat_dist = MaterialArray(shape)

        # Fill with random materials
        pb = settings.material_database['Pb']
        vacuum = settings.material_database['Vacuum']

        # Alternate materials
        mat_dist[0:5, :, :] = pb
        mat_dist[5:10, :, :] = vacuum

        volume = WoodcockVoxelVolume(voxel_size, mat_dist)

        cfunc_callable = volume.material_cfunc

        num_points = 100_000
        # Generate points strictly inside the bounds (size = shape * voxel_size)
        positions = np.random.uniform(-9.0, 9.0, (num_points, 3)).astype(Float)

        # 1. OOP NumPy Logic
        start_time = time.perf_counter()
        _, oop_materials = volume._parametric_function(positions)
        oop_mat_ids = oop_materials.ID
        oop_time = time.perf_counter() - start_time

        # 2. Numba @cfunc Logic
        from numba import njit

        @njit
        def cfunc_loop_voxel(c_ptr, pos, out_ids):
            for i in range(pos.shape[0]):
                out_ids[i] = c_ptr(pos[i, 0], pos[i, 1], pos[i, 2])

        numba_mat_ids = np.zeros(num_points, dtype=np.int64)
        cfunc_loop_voxel(cfunc_callable, positions[:1], numba_mat_ids[:1])

        start_time = time.perf_counter()
        cfunc_loop_voxel(cfunc_callable, positions, numba_mat_ids)
        numba_time = time.perf_counter() - start_time

        print(f"\n--- Voxel Volume Benchmark ({num_points} samples) ---")
        print(f"OOP Numpy Time: {oop_time:.4f} s")
        print(f"Numba CFunc Loop Time:  {numba_time:.4f} s")
        print(f"Speedup:        {oop_time / numba_time:.2f}x")

        np.testing.assert_array_equal(oop_mat_ids, numba_mat_ids, err_msg="Numba Voxel flat lookup differs from OOP NumPy logic")


if __name__ == '__main__':
    unittest.main()
