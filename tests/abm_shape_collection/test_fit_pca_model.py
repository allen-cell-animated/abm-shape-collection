import unittest

import numpy as np

from abm_shape_collection.fit_pca_model import fit_pca_model


class TestFitPCAModel(unittest.TestCase):
    def test_fit_pca_model(self):
        components = 2
        num_features = 5
        num_samples = 100

        rng = np.random.default_rng()
        features = rng.random((num_samples, num_features))
        ordering = rng.random((num_samples,))

        # Ensure that at least one of the features needs to be reoriented.
        features[:, 0] = ordering * -10

        pca = fit_pca_model(features, components, ordering)

        self.assertEqual(components, pca.n_components_)
        self.assertEqual(num_samples, pca.n_samples_)


if __name__ == "__main__":
    unittest.main()
