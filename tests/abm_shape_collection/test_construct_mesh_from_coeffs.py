import unittest

import pandas as pd
from vtk.util import numpy_support

from abm_shape_collection.construct_mesh_from_coeffs import construct_mesh_from_coeffs


class TestConstructMeshFromCoeffs(unittest.TestCase):
    def test_construct_mesh_from_coeffs_no_scale(self):
        order = 1
        prefix = "prefix"
        suffix = "suffix"
        radius = 100

        coeffs = pd.DataFrame(
            {
                f"{prefix}shcoeffs_L0M0C{suffix}": [radius],
                f"{prefix}shcoeffs_L0M1C{suffix}": [0],
                f"{prefix}shcoeffs_L1M0C{suffix}": [0],
                f"{prefix}shcoeffs_L1M1C{suffix}": [0],
                f"{prefix}shcoeffs_L0M0S{suffix}": [0],
                f"{prefix}shcoeffs_L0M1S{suffix}": [0],
                f"{prefix}shcoeffs_L1M0S{suffix}": [0],
                f"{prefix}shcoeffs_L1M1S{suffix}": [0],
            }
        )

        mesh = construct_mesh_from_coeffs(coeffs, order, prefix, suffix)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        coords = list(map(tuple, points))

        # Not checking the exact mesh, but instead that there exist specific points.
        self.assertTrue((0, 0, radius) in coords)
        self.assertTrue((0, 0, -radius) in coords)

    def test_construct_mesh_from_coeffs_with_scale(self):
        order = 1
        prefix = "prefix"
        suffix = "suffix"
        radius = 100
        scale = 2

        coeffs = pd.DataFrame(
            {
                f"{prefix}shcoeffs_L0M0C{suffix}": [radius],
                f"{prefix}shcoeffs_L0M1C{suffix}": [0],
                f"{prefix}shcoeffs_L1M0C{suffix}": [0],
                f"{prefix}shcoeffs_L1M1C{suffix}": [0],
                f"{prefix}shcoeffs_L0M0S{suffix}": [0],
                f"{prefix}shcoeffs_L0M1S{suffix}": [0],
                f"{prefix}shcoeffs_L1M0S{suffix}": [0],
                f"{prefix}shcoeffs_L1M1S{suffix}": [0],
            }
        )

        mesh = construct_mesh_from_coeffs(coeffs, order, prefix, suffix, scale=scale)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        coords = list(map(tuple, points))

        # Not checking the exact mesh, but instead that there exist specific points.
        self.assertTrue((0, 0, scale * radius) in coords)
        self.assertTrue((0, 0, -scale * radius) in coords)


if __name__ == "__main__":
    unittest.main()
