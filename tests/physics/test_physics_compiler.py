import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.physics.physics_compiler import PhysicsCompiler

def test_build_element_csr_basic():
    compiler = PhysicsCompiler()

    # Create mock materials
    # ID=1, 2 elements
    # ID=3, 1 element
    # ID=5, 2 elements with unnormalized weights

    class MockMaterial:
        def __init__(self, ID, composition_dict):
            self.ID = ID
            self._comp_dict = composition_dict

        @property
        def composition_dict(self):
            return self._comp_dict

    mat1 = MockMaterial(ID=1, composition_dict={'H': 0.2, 'O': 0.8})
    mat3 = MockMaterial(ID=3, composition_dict={'C': 1.0})
    mat5 = MockMaterial(ID=5, composition_dict={'N': 0.4, 'O': 0.4}) # Sum is 0.8

    materials_list = [mat1, mat3, mat5]

    capacity = 11
    csr = compiler._build_element_csr(materials_list, capacity) # type: ignore

    # 1. Correctness of offsets
    assert len(csr.element_offsets) == capacity + 1 # 12
    # expected counts:
    # 0: 0
    # 1: 2
    # 2: 0
    # 3: 1
    # 4: 0
    # 5: 2
    # rest 0
    # cumsum: [0, 0, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5]
    expected_offsets = np.array([0, 0, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5])
    np.testing.assert_array_equal(csr.element_offsets, expected_offsets)

    # 2. Z sampling
    # ID=1 (H, O) -> idx 0, 1 -> Z=1, 8
    # ID=3 (C) -> idx 2 -> Z=6
    # ID=5 (N, O) -> idx 3, 4 -> Z=7, 8
    expected_Z = np.array([1, 8, 6, 7, 8])
    np.testing.assert_array_equal(csr.element_Z, expected_Z)

    # 3. Normalization
    # ID=1 -> 0.2/1.0, 0.8/1.0 -> 0.2, 0.8
    # ID=3 -> 1.0/1.0 -> 1.0
    # ID=5 -> 0.4/0.8, 0.4/0.8 -> 0.5, 0.5
    expected_fractions = np.array([0.2, 0.8, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(csr.element_fraction, expected_fractions)

if __name__ == '__main__':
    test_build_element_csr_basic()
    print("All Pytest-style assertions passed!")
