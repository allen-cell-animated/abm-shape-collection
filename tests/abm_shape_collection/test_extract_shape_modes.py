import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import abm_shape_collection.construct_mesh_from_points
import abm_shape_collection.extract_mesh_projections
from abm_shape_collection.extract_shape_modes import calculate_region_offsets, extract_shape_modes


class TestExtractShapeModes(unittest.TestCase):
    def setUp(self):
        self.mocks = SimpleNamespace(
            construct=mock.MagicMock(spec_set=abm_shape_collection.construct_mesh_from_points.fn),
            extract=mock.MagicMock(spec_set=abm_shape_collection.extract_mesh_projections.fn),
        )

    def test_extract_shape_modes(self):
        order = 1
        regions = ["DEFAULT", "REGION"]
        suffixes = ["", ".REGION"]

        coeffs = [
            f"shcoeffs_L{lix}M{mix}{letter}{suffix}"
            for lix in range(order + 1)
            for mix in range(order + 1)
            for letter in ["C", "S"]
            for suffix in suffixes
        ]
        positions = [f"CENTER_{axis}{suffix}" for axis in ["X", "Y", "Z"] for suffix in suffixes]
        columns = coeffs + positions + ["angle"]

        data = pd.DataFrame(np.random.random_sample((100, len(columns))), columns=columns)

        components = 3
        pca = PCA(n_components=components)
        pca = pca.fit(data[coeffs].values)

        delta = 2
        expected_keys = [
            (mode, point)
            for mode in range(1, components + 1)
            for point in np.arange(-2, 2.5, delta)
        ]

        mesh = "mesh"
        projection = "projection"
        self.mocks.construct.side_effect = lambda *args, **kwargs: mesh
        self.mocks.extract.side_effect = lambda *args, **kwargs: f"{args[0]}{projection}"

        shape_modes = extract_shape_modes(
            pca,
            data,
            components,
            regions,
            order,
            delta,
            _construct_mesh_from_points=self.mocks.construct,
            _extract_mesh_projections=self.mocks.extract,
        )

        self.assertEqual(regions, list(shape_modes.keys()))
        for region in regions:
            keys = []

            for shape_mode in shape_modes[region]:
                keys.append((shape_mode["mode"], shape_mode["point"]))
                self.assertEqual(f"{mesh}{projection}", shape_mode["projections"])

            self.assertCountEqual(expected_keys, keys)

    def test_calculate_region_offset_default(self):
        region = "DEFAULT"
        data = pd.DataFrame()

        offsets = calculate_region_offsets(data, region)

        self.assertDictEqual({}, offsets)

    def test_calculate_region_offset_region(self):
        region = "REGION"
        angles = [0, 90, 180]

        center_x = np.random.random_sample((3,))
        center_y = np.random.random_sample((3,))
        center_z = np.random.random_sample((3,))

        delta_x = np.random.random_sample((3,))
        delta_y = np.random.random_sample((3,))
        delta_z = np.random.random_sample((3,))

        data = pd.DataFrame(
            {
                "CENTER_X": center_x,
                "CENTER_Y": center_y,
                "CENTER_Z": center_z,
                f"CENTER_X.{region}": center_x + delta_x,
                f"CENTER_Y.{region}": center_y + delta_y,
                f"CENTER_Z.{region}": center_z + delta_z,
                "angle": angles,
            }
        )

        offsets_x = [delta_x[0], -delta_y[1], -delta_x[2]]
        offsets_y = [delta_y[0], delta_x[1], -delta_y[2]]

        offsets = calculate_region_offsets(data, region)

        self.assertTrue(np.allclose(offsets_x, offsets["x"]))
        self.assertTrue(np.allclose(offsets_y, offsets["y"]))
        self.assertTrue(np.allclose(delta_z, offsets["z"]))


if __name__ == "__main__":
    unittest.main()
