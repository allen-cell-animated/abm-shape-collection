import tempfile

import numpy as np
import pandas as pd
import trimesh
from aicsshparam import shtools
from prefect import task
from sklearn.decomposition import PCA
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOPLY import vtkPLYWriter


@task
def extract_shape_modes(
    pca: PCA, data: pd.DataFrame, components: int, regions: list[str], order: int, delta: float
) -> dict:
    features = data.filter(like="shcoeffs").columns.values
    data_transform = pca.transform(data[features].values)
    means = data_transform.mean(axis=0)
    stds = data_transform.std(axis=0, ddof=1)

    shape_svgs = {}

    for component in range(components):
        point_vector = np.zeros((components))
        component_shape_modes = {}

        for point in np.arange(-2, 2.5, delta):
            point_vector[component] = point
            vector = means + np.multiply(stds, point_vector)
            point_shape_modes = {}

            for region in regions:
                shape_mode_slices = extract_shape_mode_slices(pca, vector, features, order, region)
                point_shape_modes[region] = shape_mode_slices

            component_shape_modes[point] = point_shape_modes

        shape_svgs[component + 1] = component_shape_modes

    return shape_svgs


def extract_shape_mode_slices(
    pca: PCA, vector: np.ndarray, feature_names: list[str], order: int, region: str
) -> dict:
    prefix = ""
    suffix = f".{region}" if region != "DEFAULT" else ""
    mesh = construct_mesh_from_points(pca, vector, feature_names, order, prefix, suffix)
    mesh = convert_vtk_to_trimesh(mesh)
    slices = get_mesh_slices(mesh)
    return slices


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
    coeffs_map = np.zeros((2, order + 1, order + 1), dtype=np.float32)

    for l in range(order + 1):
        for m in range(order + 1):
            coeffs_map[0, l, m] = coeffs[f"{prefix}shcoeffs_L{l}M{m}C{suffix}"]
            coeffs_map[1, l, m] = coeffs[f"{prefix}shcoeffs_L{l}M{m}S{suffix}"]

    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs_map)
    return mesh


def convert_vtk_to_trimesh(mesh: vtkPolyData) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile() as temp:
        writer = vtkPLYWriter()
        writer.SetInputData(mesh)
        writer.SetFileTypeToASCII()
        writer.SetFileName(f"{temp.name}.ply")
        _ = writer.Write()
        mesh = trimesh.load(f"{temp.name}.ply")

    return mesh


def get_mesh_slices(mesh: trimesh.Trimesh) -> dict:
    return {
        "side_1": get_mesh_slice(mesh, (0, 1, 0)),
        "side_2": get_mesh_slice(mesh, (1, 0, 0)),
        "top": get_mesh_slice(mesh, (0, 0, 1)),
    }


def get_mesh_slice(mesh: trimesh.Trimesh, normal: tuple[int, int, int]) -> str:
    """Get svg slice of mesh along plane for given normal."""
    mesh_slice = mesh.section_multiplane(mesh.centroid, normal, [0])
    svg = trimesh.path.exchange.svg_io.export_svg(mesh_slice[0])
    return svg
