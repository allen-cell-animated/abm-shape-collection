import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from vtk import vtkPolyData

from abm_shape_collection.construct_mesh_from_coeffs import construct_mesh_from_coeffs


def construct_mesh_from_points(
    pca: PCA,
    points: np.ndarray,
    feature_names: list[str],
    order: int,
    prefix: str = "",
    suffix: str = "",
) -> vtkPolyData:
    """Constructs mesh given PCA transformation points."""
    coeffs = pd.Series(pca.inverse_transform(points), index=feature_names)
    return construct_mesh_from_coeffs(coeffs, order, prefix, suffix)
