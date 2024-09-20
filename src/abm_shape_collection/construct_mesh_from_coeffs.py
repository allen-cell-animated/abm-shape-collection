import numpy as np
import pandas as pd
from aicsshparam import shtools
from vtk import (  # pylint: disable=no-name-in-module
    vtkPolyData,
    vtkTransform,
    vtkTransformPolyDataFilter,
)


def construct_mesh_from_coeffs(
    coeffs: pd.DataFrame,
    order: int,
    prefix: str = "",
    suffix: str = "",
    scale: float = 1.0,
) -> vtkPolyData:
    """
    Construct a mesh from spherical harmonic coefficients.

    Parameters
    ----------
    coeffs
        Spherical harmonic coefficients.
    order
        Order of the spherical harmonics coefficient parametrization.
    prefix
        Prefix string for all coefficient columns.
    suffix
        Suffix string for all coefficient columns.
    scale
        Scale factor for mesh points.

    Returns
    -------
    :
        Mesh object.
    """

    coeffs_map = np.zeros((2, order + 1, order + 1), dtype=np.float32)

    for lix in range(order + 1):
        for mix in range(order + 1):
            coeffs_map[0, lix, mix] = coeffs[f"{prefix}shcoeffs_L{lix}M{mix}C{suffix}"]
            coeffs_map[1, lix, mix] = coeffs[f"{prefix}shcoeffs_L{lix}M{mix}S{suffix}"]

    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs_map)

    if scale != 1.0:
        transform = vtkTransform()
        transform.Scale((scale, scale, scale))

        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetInputData(mesh)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mesh = transform_filter.GetOutput(0)

    return mesh
