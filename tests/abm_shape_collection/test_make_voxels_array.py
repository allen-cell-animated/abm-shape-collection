import unittest

import numpy as np

from abm_shape_collection.make_voxels_array import make_voxels_array


class TestMakeVoxelsArray(unittest.TestCase):
    def test_make_voxels_array_no_reference_no_scale(self):
        voxels = [
            (3, 1, 1),
            (2, 0, 2),
            (1, 0, 3),
            (2, 3, 6),
        ]

        expected_voxels = [
            (1, 2, 3),
            (2, 1, 2),
            (3, 1, 1),
            (6, 4, 2),
        ]

        expected_array = np.zeros((8, 6, 5), dtype=np.uint8)
        expected_array[tuple(np.transpose(expected_voxels))] = [1] * len(expected_voxels)

        array = make_voxels_array(voxels, reference=None, scale=1)

        self.assertTrue((expected_array == array).all())

    def test_make_voxels_array_with_reference_no_scale(self):
        reference = [
            (10, 10, 10),
            (0, 0, 0),
        ]

        voxels = [
            (3, 1, 1),
            (2, 0, 2),
            (1, 0, 3),
            (2, 3, 6),
        ]

        expected_voxels = [
            (2, 2, 4),
            (3, 1, 3),
            (4, 1, 2),
            (7, 4, 3),
        ]

        expected_array = np.zeros((13, 13, 13), dtype=np.uint8)
        expected_array[tuple(np.transpose(expected_voxels))] = [1] * len(expected_voxels)

        array = make_voxels_array(voxels, reference=reference, scale=1)

        self.assertTrue((expected_array == array).all())

    def test_make_voxels_array_no_reference_with_scale(self):
        scale = 2

        voxels = [
            (3, 1, 1),
            (2, 0, 2),
            (1, 0, 3),
            (2, 3, 6),
        ]

        base_voxels = [
            (1, 2, 3),
            (2, 1, 2),
            (3, 1, 1),
            (6, 4, 2),
        ]

        expected_voxels = [
            (zi, yi, xi)
            for z, y, x in base_voxels
            for zi in [2 * z, 2 * z + 1]
            for yi in [2 * y, 2 * y + 1]
            for xi in [2 * x, 2 * x + 1]
        ]
        expected_array = np.zeros((16, 12, 10), dtype=np.uint8)
        expected_array[tuple(np.transpose(expected_voxels))] = [1] * len(expected_voxels)

        array = make_voxels_array(voxels, reference=None, scale=scale)

        self.assertTrue((expected_array == array).all())

    def test_make_voxels_array_with_reference_with_scale(self):
        scale = 2

        reference = [
            (10, 10, 10),
            (0, 0, 0),
        ]

        voxels = [
            (3, 1, 1),
            (2, 0, 2),
            (1, 0, 3),
            (2, 3, 6),
        ]

        base_voxels = [
            (2, 2, 4),
            (3, 1, 3),
            (4, 1, 2),
            (7, 4, 3),
        ]

        expected_voxels = [
            (zi, yi, xi)
            for z, y, x in base_voxels
            for zi in [2 * z, 2 * z + 1]
            for yi in [2 * y, 2 * y + 1]
            for xi in [2 * x, 2 * x + 1]
        ]
        expected_array = np.zeros((26, 26, 26), dtype=np.uint8)
        expected_array[tuple(np.transpose(expected_voxels))] = [1] * len(expected_voxels)

        array = make_voxels_array(voxels, reference=reference, scale=scale)

        self.assertTrue((expected_array == array).all())


if __name__ == "__main__":
    unittest.main()
