import numpy as np
from aicsshparam import shtools
from vtk import vtkPolyData  # pylint: disable=no-name-in-module
from vtk.util import numpy_support  # pylint: disable=no-name-in-module, import-error


def construct_mesh_from_array(array: np.ndarray, reference: np.ndarray) -> vtkPolyData:
    """
    Construct a mesh from binary image array.

    Parameters
    ----------
    array
        Binary image array.
    reference
        Binary reference array that determines alignment angle and centroid.

    Returns
    -------
    :
        Mesh object.
    """

    _, angle = shtools.align_image_2d(image=reference)
    aligned_reference = shtools.apply_image_alignment_2d(reference, angle).squeeze()
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()

    mesh, _, _ = shtools.get_mesh_from_image(
        image=aligned_array, lcc=False, translate_to_origin=False
    )

    centroid = np.argwhere(aligned_reference == 1).mean(axis=0)[[2, 1, 0]]
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()) - centroid
    mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(coords))

    return mesh
