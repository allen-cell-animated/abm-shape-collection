import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from vtk import vtkPolyData  # pylint: disable=no-name-in-module

from abm_shape_collection.construct_mesh_from_coeffs import construct_mesh_from_coeffs


def construct_mesh_from_points(
    pca: PCA,
    points: np.ndarray,
    feature_names: list[str],
    order: int,
    prefix: str = "",
    suffix: str = "",
) -> vtkPolyData:
    """
    Construct mesh given PCA transformation points.

    Parameters
    ----------
    pca
        Fit PCA object.
    points
        Select point in PC space.
    feature_names
        Spherical harmonics coefficient names.
    order
        Order of the spherical harmonics coefficient parametrization.
    prefix
        Prefix string for all coefficient columns.
    suffix
        Suffix string for all coefficient columns.

    Returns
    -------
    :
        Mesh object.
    """

    coeffs = pd.Series(pca.inverse_transform(points), index=feature_names)
    return construct_mesh_from_coeffs(coeffs, order, prefix, suffix)
