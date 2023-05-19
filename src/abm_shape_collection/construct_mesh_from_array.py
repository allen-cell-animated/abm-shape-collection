import numpy as np
from aicsshparam import shtools
from vtk import vtkPolyData


def construct_mesh_from_array(array: np.ndarray, reference: np.ndarray) -> vtkPolyData:
    _, angle = shtools.align_image_2d(image=reference)
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()
    mesh, _, _ = shtools.get_mesh_from_image(image=aligned_array)
    return mesh
