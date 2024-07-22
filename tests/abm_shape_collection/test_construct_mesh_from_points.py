import unittest
from unittest import mock

import numpy as np
from sklearn.decomposition import PCA
from vtk.util import numpy_support

from abm_shape_collection.construct_mesh_from_points import construct_mesh_from_points


class TestConstructMeshFromPoints(unittest.TestCase):
    def test_construct_mesh_from_points(self):
        order = 1
        prefix = "prefix"
        suffix = "suffix"
        radius = 100

        feature_names = [
            f"{prefix}shcoeffs_L0M0C{suffix}",
            f"{prefix}shcoeffs_L0M1C{suffix}",
            f"{prefix}shcoeffs_L1M0C{suffix}",
            f"{prefix}shcoeffs_L1M1C{suffix}",
            f"{prefix}shcoeffs_L0M0S{suffix}",
            f"{prefix}shcoeffs_L0M1S{suffix}",
            f"{prefix}shcoeffs_L1M0S{suffix}",
            f"{prefix}shcoeffs_L1M1S{suffix}",
        ]
        points = np.array([1, 2, 3])
        coeffs = np.array([radius, 0, 0, 0, 0, 0, 0, 0])

        mock_transform = {str(points): coeffs}
        pca = mock.Mock(spec=PCA)
        pca.inverse_transform.side_effect = lambda values: mock_transform[str(values)]

        mesh = construct_mesh_from_points(pca, points, feature_names, order, prefix, suffix)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
        coords = list(map(tuple, points))

        # Not checking the exact mesh, but instead that there exist specific points.
        self.assertTrue((0, 0, radius) in coords)
        self.assertTrue((0, 0, -radius) in coords)


if __name__ == "__main__":
    unittest.main()
