import numpy as np
from prefect import task
from sklearn.decomposition import PCA


@task
def fit_pca_model(features: np.ndarray, components: int, ordering: np.ndarray) -> PCA:
    # Fit data.
    pca = PCA(n_components=components)
    pca = pca.fit(features)

    # Reorient features by ordering data.
    transformed = pca.transform(features)
    for i in range(components):
        pearson = np.corrcoef(ordering, transformed[:, i])
        if pearson[0, 1] < 0:
            pca.components_[i] = pca.components_[i] * -1

    return pca
