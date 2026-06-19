import unittest
import numpy as np

from core.other.nonunique_array import NonuniqueArray


class TestNonuniqueArray(unittest.TestCase):
    def test_inverse_indices_1d(self):
        arr = NonuniqueArray((3,))
        arr[0] = 'a'
        arr[1] = 'b'
        arr[2] = 'a'

        indices = arr.inverse_indices
        self.assertIn('a', indices)
        self.assertIn('b', indices)
        self.assertTrue(np.array_equal(indices['a'][0], [0, 2]))
        self.assertTrue(np.array_equal(indices['b'][0], [1]))

    def test_inverse_indices_2d(self):
        arr = NonuniqueArray((2, 2))
        arr[0, 0] = 'a'
        arr[1, 1] = 'b'

        indices = arr.inverse_indices
        self.assertIn('a', indices)
        self.assertIn('b', indices)
        self.assertTrue(np.array_equal(indices['a'][0], [0]))
        self.assertTrue(np.array_equal(indices['a'][1], [0]))
        self.assertTrue(np.array_equal(indices['b'][0], [1]))
        self.assertTrue(np.array_equal(indices['b'][1], [1]))

        # Test None elements
        self.assertTrue(np.array_equal(indices[None][0], [0, 1]))
        self.assertTrue(np.array_equal(indices[None][1], [1, 0]))

    def test_inverse_indices_3d(self):
        arr = NonuniqueArray((2, 2, 2))
        arr[0, 0, 0] = 'a'
        arr[1, 1, 1] = 'b'

        indices = arr.inverse_indices
        self.assertIn('a', indices)
        self.assertIn('b', indices)
        self.assertTrue(np.array_equal(indices['a'][0], [0]))
        self.assertTrue(np.array_equal(indices['a'][1], [0]))
        self.assertTrue(np.array_equal(indices['a'][2], [0]))
        self.assertTrue(np.array_equal(indices['b'][0], [1]))
        self.assertTrue(np.array_equal(indices['b'][1], [1]))
        self.assertTrue(np.array_equal(indices['b'][2], [1]))

    def test_setitem_with_slice(self):
        arr1 = NonuniqueArray((2, 2, 2))
        arr2 = NonuniqueArray((2, 2))
        arr2[0, 0] = 'a'
        arr2[1, 1] = 'b'

        # Assign 2D array to a 2D slice of 3D array
        arr1[:, :, 0] = arr2

        restored = arr1.restore()
        self.assertEqual(restored[0, 0, 0], 'a')
        self.assertEqual(restored[1, 1, 0], 'b')
        self.assertEqual(restored[0, 1, 0], None)
        self.assertEqual(restored[1, 0, 0], None)
        self.assertEqual(restored[0, 0, 1], None)

    def test_setitem_whole_array(self):
        arr1 = NonuniqueArray((2, 2))
        arr1[0, 0] = 'a'
        arr1[1, 1] = 'b'

        arr2 = NonuniqueArray((2, 2))
        arr2[:] = arr1

        restored = arr2.restore()
        self.assertEqual(restored[0, 0], 'a')
        self.assertEqual(restored[1, 1], 'b')

if __name__ == '__main__':
    unittest.main()
