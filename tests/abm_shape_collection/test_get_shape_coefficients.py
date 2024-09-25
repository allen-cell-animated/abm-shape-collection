import unittest

import numpy as np

from abm_shape_collection.get_shape_coefficients import get_shape_coefficients


class TestGetShapeCoefficients(unittest.TestCase):
    def setUp(self):
        self.order = 2

        self.coeff_names = [
            f"shcoeffs_L{lix}M{mix}{letter}"
            for lix in range(self.order + 1)
            for mix in range(self.order + 1)
            for letter in ["C", "S"]
        ]

    def test_get_shape_coefficients_same_reference(self):
        array = np.zeros((10, 10, 10))
        array[4:6, 4:6, 2:8] = 1

        coeffs = get_shape_coefficients(array, array, self.order)

        self.assertEqual(0.0, coeffs["angle"])
        self.assertTrue("mse" in coeffs)
        self.assertTrue(all(name in coeffs for name in self.coeff_names))

    def test_get_shape_coefficients_different_reference(self):
        reference = np.zeros((10, 6, 6))
        reference[4:6, 1:3, 3:5] = 1
        reference[4:6, 2:4, 2:4] = 1
        reference[4:6, 3:5, 1:3] = 1

        array = np.zeros((10, 6, 6))
        array[4:6, 1:3, 1:3] = 1
        array[4:6, 2:4, 2:4] = 1
        array[4:6, 3:5, 3:5] = 1
        array[4:6, 4, 4] = 0

        coeffs = get_shape_coefficients(array, reference, self.order)

        self.assertEqual(-45.0, coeffs["angle"])
        self.assertTrue("mse" in coeffs)
        self.assertTrue(all(name in coeffs for name in self.coeff_names))


if __name__ == "__main__":
    unittest.main()
