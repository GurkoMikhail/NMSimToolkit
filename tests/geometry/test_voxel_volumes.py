import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import settings.database_setting as settings
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.materials.materials import MaterialArray
from core.other.typing_definitions import Float

class TestWoodcockVoxelVolume(unittest.TestCase):
    def test_parametric_func_memory_lookup(self):
        """
        Verify that parametric_func correctly reads materials from memory using the intrinsic,
        preventing garbage collection and reading correct indices even for negative coordinates.
        """
        voxel_size = Float(1.0)
        shape = (4, 4, 4)
        mat_dist = MaterialArray(shape)

        # Materials
        vacuum = settings.material_database['Vacuum']
        cu = settings.material_database['Cu']
        pb = settings.material_database['Pb']

        # Fill different regions
        mat_dist[0:2, 0:2, 0:2] = cu
        mat_dist[2:4, 2:4, 2:4] = pb
        mat_dist[0:2, 2:4, 0:2] = vacuum

        volume = WoodcockVoxelVolume(voxel_size, mat_dist)

        import ctypes
        cfunc_callable = volume.material_cfunc

        from numba import njit
        from ctypes import cast, CFUNCTYPE, c_int64, c_double
        cfunc_type = CFUNCTYPE(c_int64, c_double, c_double, c_double)
        cfunc_address = ctypes.cast(cfunc_callable, ctypes.c_void_p).value
        cfunc_instance = cast(cfunc_address, cfunc_type)

        @njit
        def cfunc_caller(func_ptr, x, y, z):
            return func_ptr(x, y, z)

        # Test within bounds
        # The volume size is 4x4x4. The center is at (0, 0, 0).
        # x ranges from -2 to 2.
        # x=-1.5 corresponds to ix=0, x=1.5 corresponds to ix=3

        # Coordinate for region [0:2, 0:2, 0:2] (Cu)
        # Should be x=-1.5, y=-1.5, z=-1.5
        id_cu_actual = cfunc_caller(cfunc_instance, -1.5, -1.5, -1.5)
        # In MaterialArray, the internal IDs might not correspond to the physical material IDs directly
        # The true test is whether cfunc caller matches what oop logic gives

        expected_cu_id = mat_dist.ID[0,0,0]
        self.assertEqual(id_cu_actual, expected_cu_id)

        # Coordinate for region [2:4, 2:4, 2:4] (Pb)
        # Should be x=1.5, y=1.5, z=1.5
        id_pb_actual = cfunc_caller(cfunc_instance, 1.5, 1.5, 1.5)
        expected_pb_id = mat_dist.ID[3,3,3]
        self.assertEqual(id_pb_actual, expected_pb_id)

        # Coordinate for region [0:2, 2:4, 0:2] (Vacuum)
        # Should be x=-1.5, y=1.5, z=-1.5
        id_vacuum_actual = cfunc_caller(cfunc_instance, -1.5, 1.5, -1.5)
        expected_vacuum_id = mat_dist.ID[0,3,0]
        self.assertEqual(id_vacuum_actual, expected_vacuum_id)

        # Test out of bounds (should clamp to the nearest valid index)
        id_oob_negative = cfunc_caller(cfunc_instance, -100.0, -100.0, -100.0)
        self.assertEqual(id_oob_negative, expected_cu_id) # Clamps to 0,0,0 which is Cu

        id_oob_positive = cfunc_caller(cfunc_instance, 100.0, 100.0, 100.0)
        self.assertEqual(id_oob_positive, expected_pb_id) # Clamps to 3,3,3 which is Pb

if __name__ == '__main__':
    unittest.main()
