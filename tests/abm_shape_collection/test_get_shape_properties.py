import unittest

import numpy as np

from abm_shape_collection.get_shape_properties import get_shape_properties


class TestGetShapeProperties(unittest.TestCase):
    def test_get_shape_properties(self):
        array = np.zeros((10, 10, 10), dtype=np.uint8)
        array[4:6, 4:6, 2:8] = 1

        expected_values = {
            "area": 12,  # number of pixels of the region
            "perimeter": 12,  # line through the centers of border pixels
            "extent": 1,  # ratio of pixels in region to pixels in total bounding box
            "solidity": 1,  # ratio of pixels in region to pixels of convex hull image
        }

        properties = get_shape_properties(array, list(expected_values.keys()))

        for property_name, expected_value in expected_values.items():
            with self.subTest(property=property_name):
                self.assertEqual(expected_value, properties[property_name])


if __name__ == "__main__":
    unittest.main()
