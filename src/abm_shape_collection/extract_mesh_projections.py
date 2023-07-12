import tempfile
import xml.etree.ElementTree as ET
from typing import Optional, Union

import numpy as np
import trimesh
from vtk import vtkPLYWriter, vtkPolyData

PROJECTIONS: list[tuple[str, tuple[int, int, int], int]] = [
    ("side1", (0, 1, 0), 1),
    ("side2", (1, 0, 0), 0),
    ("top", (0, 0, 1), 2),
]


def extract_mesh_projections(
    mesh: Union[vtkPolyData, trimesh.Trimesh],
    slices: bool = True,
    extents: bool = True,
    offset: Optional[tuple[float, float, float]] = None,
) -> dict:
    if isinstance(mesh, vtkPolyData):
        mesh = convert_vtk_to_trimesh(mesh)

    if offset is not None:
        mesh.apply_translation(offset)

    projections = {}

    if slices:
        for projection, normal, _ in PROJECTIONS:
            projections[f"{projection}_slice"] = get_mesh_slice(mesh, normal)

    if extents:
        for projection, normal, index in PROJECTIONS:
            projections[f"{projection}_extent"] = get_mesh_extent(mesh, normal, index)

    return projections


def convert_vtk_to_trimesh(mesh: vtkPolyData) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile() as temp:
        writer = vtkPLYWriter()
        writer.SetInputData(mesh)
        writer.SetFileTypeToASCII()
        writer.SetFileName(f"{temp.name}.ply")
        _ = writer.Write()
        mesh = trimesh.load(f"{temp.name}.ply")

    return mesh


def get_mesh_slice(mesh: trimesh.Trimesh, normal: tuple[int, int, int]) -> list[str]:
    """Get slice of mesh along plane for given normal as svg path."""
    mesh_slice = mesh.section_multiplane((0, 0, 0), normal, [0])
    svg = trimesh.path.exchange.svg_io.export_svg(mesh_slice[0])
    paths = [path.attrib["d"] for path in ET.fromstring(svg).findall(".//")]
    return paths


def get_mesh_extent(mesh: trimesh.Trimesh, normal: tuple[int, int, int], index: int) -> list[str]:
    """Get extent of mesh along plane for given normal as svg path."""
    layers = int(mesh.extents[index] + 2)
    plane_indices = list(np.arange(-layers, layers + 1, 0.5))
    mesh_extents = mesh.section_multiplane((0, 0, 0), normal, plane_indices)
    mesh_extents = [mesh_extent for mesh_extent in mesh_extents if mesh_extent is not None]
    mesh_extents_concat = trimesh.path.util.concatenate(mesh_extents)
    svg = trimesh.path.exchange.svg_io.export_svg(mesh_extents_concat)
    paths = [path.attrib["d"] for path in ET.fromstring(svg).findall(".//")]
    return paths
