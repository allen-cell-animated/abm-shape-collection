import unittest

import numpy as np
from vtk.util import numpy_support

from abm_shape_collection.construct_mesh_from_array import construct_mesh_from_array


class TestConstructMeshFromArray(unittest.TestCase):
    def test_construct_mesh_from_array_same_reference(self):
        array = np.zeros((10, 10, 10))
        array[4:6, 4:6, 2:8] = 1
        array[4, :, :]

        expected_coords = [
            (x, y, z)
            for x in [-3, 3]
            for y, z in [
                (-0.5, -0.5),
                (-0.5, 0.5),
                (0.5, -0.5),
                (0.5, 0.5),
            ]
        ] + [
            (x, y, z)
            for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
            for y, z in [
                (-1.0, -0.5),
                (-1.0, 0.5),
                (-0.5, -1.0),
                (-0.5, 1.0),
                (0.5, -1.0),
                (0.5, 1.0),
                (1.0, -0.5),
                (1.0, 0.5),
            ]
        ]

        mesh = construct_mesh_from_array(array, array)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        coords = list(map(tuple, points))

        self.assertCountEqual(expected_coords, coords)

    def test_construct_mesh_from_array_different_reference(self):
        reference = np.zeros((10, 6, 6))
        reference[4:6, 1:3, 3:5] = 1
        reference[4:6, 2:4, 2:4] = 1
        reference[4:6, 3:5, 1:3] = 1

        array = np.zeros((10, 6, 6))
        array[4:6, 1:3, 1:3] = 1
        array[4:6, 2:4, 2:4] = 1
        array[4:6, 3:5, 3:5] = 1
        array[4:6, 4, 4] = 0

        expected_coords = [
            (x, y, z)
            for z in [-1, 1]
            for x, y in [
                (-0.5, -1.5),
                (0.5, -1.5),
                (-0.5, -0.5),
                (0.5, -0.5),
                (-0.5, 0.5),
                (0.5, 0.5),
                (-0.5, 1.5),
                (0.5, 1.5),
            ]
        ] + [
            (x, y, z)
            for z in [-0.5, 0.5]
            for x, y in [
                (-0.5, -2.0),
                (0.5, -2.0),
                (-1.0, -1.5),
                (1.0, -1.5),
                (-1.0, -0.5),
                (1.0, -0.5),
                (-1.0, 0.5),
                (1.0, 0.5),
                (-1.0, 1.5),
                (-0.5, 2.0),
                (1.0, 1.5),
                (0.5, 2.0),
            ]
        ]

        mesh = construct_mesh_from_array(array, reference)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        coords = list(map(tuple, points))

        self.assertCountEqual(expected_coords, coords)


if __name__ == "__main__":
    unittest.main()
