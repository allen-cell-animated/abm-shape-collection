from __future__ import annotations

import tempfile
from enum import Enum

import numpy as np
import trimesh
from vtk import vtkPLYWriter, vtkPolyData  # pylint: disable=no-name-in-module

PROJECTIONS: list[tuple[str, tuple[int, int, int], int]] = [
    ("side1", (0, 1, 0), 1),
    ("side2", (1, 0, 0), 0),
    ("top", (0, 0, 1), 2),
]
"""Mesh projection names, normals, and extent axes."""


class ProjectionType(Enum):
    """Projection slice types."""

    SLICE = 1
    """Slice projection type."""

    EXTENT = 2
    """Extent projection type."""


def extract_mesh_projections(
    mesh: vtkPolyData | trimesh.Trimesh,
    projection_types: list[ProjectionType] | None = None,
    offset: tuple[float, float, float] | None = None,
) -> dict:
    """
    Extract slices and/or extents from mesh.

    Slice projections are taken as the cross section of the mesh with planes in
    the x, y, and z directions with origin at (0,0,0). Extent projections are
    taken as cross section of the mesh with planes in the x, y, and z directions
    at increments of 0.5.

    Parameters
    ----------
    mesh
        Mesh object.
    projection_types
        Mesh projection types.
    offset
        Mesh translation applied before extracting slices and/or meshes.

    Returns
    -------
    :
        Map of mesh projection path points.
    """

    if isinstance(mesh, vtkPolyData):
        mesh = convert_vtk_to_trimesh(mesh)

    if projection_types is None:
        projection_types = [ProjectionType.SLICE, ProjectionType.EXTENT]

    if offset is not None:
        mesh.apply_translation(offset)

    projections: dict[str, list[list[list[float]]] | dict[float, list[list[list[float]]]]] = {}

    if ProjectionType.SLICE in projection_types:
        for projection, normal, _ in PROJECTIONS:
            projections[f"{projection}_slice"] = get_mesh_slice(mesh, normal)

    if ProjectionType.EXTENT in projection_types:
        for projection, normal, index in PROJECTIONS:
            projections[f"{projection}_extent"] = get_mesh_extent(mesh, normal, index)

    return projections


def convert_vtk_to_trimesh(mesh: vtkPolyData) -> trimesh.Trimesh:
    """
    Convert VTK polydata to trimesh object.

    Parameters
    ----------
    mesh
        VTK mesh object.

    Returns
    -------
    :
        Trimesh mesh object.
    """
    with tempfile.NamedTemporaryFile() as temp:
        writer = vtkPLYWriter()
        writer.SetInputData(mesh)
        writer.SetFileTypeToASCII()
        writer.SetFileName(f"{temp.name}.ply")
        _ = writer.Write()
        return trimesh.load(f"{temp.name}.ply")


def get_mesh_slice(mesh: trimesh.Trimesh, normal: tuple[int, int, int]) -> list[list[list[float]]]:
    """
    Get slice of mesh along plane for given normal as path points.

    Parameters
    ----------
    mesh
        Mesh object.
    normal
        Vector normal to slice plane.

    Returns
    -------
    :
        List of connected vertices in space specifying the slice.
    """

    mesh_slice = mesh.section_multiplane((0, 0, 0), normal, [0])
    return [[list(point) for point in entity] for entity in mesh_slice[0].discrete]


def get_mesh_extent(
    mesh: trimesh.Trimesh, normal: tuple[int, int, int], index: int
) -> dict[float, list[list[list[float]]]]:
    """
    Get extent of mesh along plane for given normal as path points.

    Parameters
    ----------
    mesh
        Mesh object.
    normal
        Vector normal to slice plane.
    index
        Index of normal axis.

    Returns
    -------
    :
        Map to list of connected vertices in space specifying the extent.
    """

    layers = int(mesh.extents[index] + 2)
    plane_indices = list(np.arange(-layers, layers + 1, 0.5))
    mesh_extents = mesh.section_multiplane((0, 0, 0), normal, plane_indices)
    return {
        index: [[list(point) for point in entity] for entity in mesh_extent.discrete]
        for mesh_extent, index in zip(mesh_extents, plane_indices)
        if mesh_extent is not None
    }
