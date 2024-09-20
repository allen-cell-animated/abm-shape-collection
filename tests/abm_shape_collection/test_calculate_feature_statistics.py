import unittest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from abm_shape_collection.calculate_feature_statistics import calculate_feature_statistics


class TestCalculateFeatureStatistics(unittest.TestCase):
    def test_calculate_feature_statistics(self):
        data_length = 5
        ref_length = 10

        rng = np.random.default_rng()

        feature_a = "feature_a"
        feature_a_data = rng.random((data_length,))
        feature_a_ref = rng.random((ref_length,))
        feature_a_len = len(feature_a_data)
        feature_a_ks = ks_2samp(feature_a_data, feature_a_ref, mode="asymp")

        feature_b = "feature_b"
        feature_b_data = rng.random((data_length,))
        feature_b_ref = rng.random((ref_length,))
        feature_b_len = len(feature_b_data)
        feature_b_ks = ks_2samp(feature_b_data, feature_b_ref, mode="asymp")

        features = [feature_a, feature_b]
        data = pd.DataFrame(
            {
                feature_a: feature_a_data,
                feature_b: feature_b_data,
            }
        )
        ref_data = pd.DataFrame(
            {
                feature_a: feature_a_ref,
                feature_b: feature_b_ref,
            }
        )

        expected_statistics = pd.DataFrame(
            {
                "FEATURE": [feature_a.upper(), feature_b.upper()],
                "SIZE": [feature_a_len, feature_b_len],
                "KS_STATISTIC": [feature_a_ks.statistic, feature_b_ks.statistic],
                "KS_PVALUE": [feature_a_ks.pvalue, feature_b_ks.pvalue],
            }
        )

        statistics = calculate_feature_statistics(features, data, ref_data)

        self.assertTrue(expected_statistics.equals(statistics))


if __name__ == "__main__":
    unittest.main()
