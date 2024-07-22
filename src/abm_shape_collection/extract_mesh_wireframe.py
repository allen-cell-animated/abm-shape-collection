from typing import Optional, Union

import trimesh
from vtk import vtkPolyData  # pylint: disable=no-name-in-module

from abm_shape_collection.extract_mesh_projections import convert_vtk_to_trimesh


def extract_mesh_wireframe(
    mesh: Union[vtkPolyData, trimesh.Trimesh], offset: Optional[tuple[float, float, float]] = None
) -> list[list[tuple[float, float, float]]]:
    """
    Extract wireframe edges from mesh.

    Parameters
    ----------
    mesh
        Mesh object.
    offset
        Mesh translation applied before extracting slices and/or meshes.

    Returns
    -------
    :
        List of wireframe edges.
    """

    if isinstance(mesh, vtkPolyData):
        mesh = convert_vtk_to_trimesh(mesh)

    if offset is not None:
        mesh.apply_translation(offset)

    all_edges = [[tuple(mesh.vertices[a]), tuple(mesh.vertices[b])] for a, b in mesh.edges]
    wireframe = [
        [(x1, y1, z1), (x2, y2, z2)]
        for (x1, y1, z1), (x2, y2, z2) in {frozenset(edge) for edge in all_edges}
    ]

    return wireframe
