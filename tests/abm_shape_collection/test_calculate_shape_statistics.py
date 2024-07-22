import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from abm_shape_collection.calculate_shape_statistics import calculate_shape_statistics


class TestCalculateShapeStatistics(unittest.TestCase):
    def test_calculate_shape_statistics(self):
        label = "label"
        components = 2
        data_length = 6
        ref_length = 10

        data = pd.DataFrame(
            {
                f"{label}_a": np.random.random_sample((data_length,)),
                f"{label}_b": np.random.random_sample((data_length,)),
                f"{label}_c": np.random.random_sample((data_length,)),
            }
        )
        ref_data = pd.DataFrame(
            {
                f"{label}_a": np.random.random_sample((ref_length,)),
                f"{label}_b": np.random.random_sample((ref_length,)),
                f"{label}_c": np.random.random_sample((ref_length,)),
            }
        )

        pc1_data = np.random.random_sample((data_length,))
        pc1_ref = np.random.random_sample((ref_length,))

        pc2_data = np.random.random_sample((data_length,))
        pc2_ref = np.random.random_sample((ref_length,))

        mock_transform = {
            data_length: np.array([pc1_data, pc2_data]).T,
            ref_length: np.array([pc1_ref, pc2_ref]).T,
        }

        pc1_ks = ks_2samp(pc1_data, pc1_ref, mode="asymp")
        pc2_ks = ks_2samp(pc2_data, pc2_ref, mode="asymp")

        pca = mock.Mock(spec=PCA)
        pca.transform.side_effect = lambda values: mock_transform[values.shape[0]]

        expected_statistics = pd.DataFrame(
            [
                {
                    "FEATURE": "PC1",
                    "SIZE": len(pc1_data),
                    "KS_STATISTIC": pc1_ks.statistic,
                    "KS_PVALUE": pc1_ks.pvalue,
                },
                {
                    "FEATURE": "PC2",
                    "SIZE": len(pc2_data),
                    "KS_STATISTIC": pc2_ks.statistic,
                    "KS_PVALUE": pc2_ks.pvalue,
                },
            ]
        )

        statistics = calculate_shape_statistics(pca, data, ref_data, components, label)

        self.assertTrue(expected_statistics.equals(statistics))


if __name__ == "__main__":
    unittest.main()
