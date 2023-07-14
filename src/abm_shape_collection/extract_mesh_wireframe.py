from typing import Optional, Union

import trimesh
from vtk import vtkPolyData

from abm_shape_collection.extract_mesh_projections import convert_vtk_to_trimesh


def extract_mesh_wireframe(
    mesh: Union[vtkPolyData, trimesh.Trimesh], offset: Optional[tuple[float, float, float]] = None
) -> list[list[list[float]]]:
    if isinstance(mesh, vtkPolyData):
        mesh = convert_vtk_to_trimesh(mesh)

    if offset is not None:
        mesh.apply_translation(offset)

    wireframe = [[list(mesh.vertices[a]), list(mesh.vertices[b])] for a, b in mesh.edges]

    return wireframe
