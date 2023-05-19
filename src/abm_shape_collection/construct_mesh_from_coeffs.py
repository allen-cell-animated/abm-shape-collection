import numpy as np
import pandas as pd
from aicsshparam import shtools
from vtk import vtkPolyData, vtkTransform, vtkTransformPolyDataFilter


def construct_mesh_from_coeffs(
    coeffs: pd.DataFrame,
    order: int,
    prefix: str = "",
    suffix: str = "",
    scale: float = 1.0,
) -> vtkPolyData:
    coeffs_map = np.zeros((2, order + 1, order + 1), dtype=np.float32)

    for l in range(order + 1):
        for m in range(order + 1):
            coeffs_map[0, l, m] = coeffs[f"{prefix}shcoeffs_L{l}M{m}C{suffix}"]
            coeffs_map[1, l, m] = coeffs[f"{prefix}shcoeffs_L{l}M{m}S{suffix}"]

    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs_map)

    if scale != 1:
        transform = vtkTransform()
        transform.Scale((scale, scale, scale))

        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetInputData(mesh)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mesh = transform_filter.GetOutput(0)

    return mesh
