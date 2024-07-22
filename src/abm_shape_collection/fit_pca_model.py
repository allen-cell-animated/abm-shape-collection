import numpy as np
from sklearn.decomposition import PCA


def fit_pca_model(features: np.ndarray, components: int, ordering: np.ndarray) -> PCA:
    """
    Fit PCA model to given data.

    Parameters
    ----------
    features
        Feature data (shape = num_samples x num_feature).
    components
        Number of components to keep.
    ordering
        Data used to reorient components.

    Returns
    -------
    :
        Fit PCA object.
    """

    # Fit data.
    pca = PCA(n_components=components)
    pca = pca.fit(features)

    # Reorient features by ordering data.
    transform = pca.transform(features)
    for i in range(components):
        pearson = np.corrcoef(ordering, transform[:, i])
        if pearson[0, 1] < 0:
            pca.components_[i] = pca.components_[i] * -1

    return pca
