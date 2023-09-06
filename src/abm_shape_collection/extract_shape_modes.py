import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from abm_shape_collection.construct_mesh_from_points import construct_mesh_from_points
from abm_shape_collection.extract_mesh_projections import extract_mesh_projections


def extract_shape_modes(
    pca: PCA, data: pd.DataFrame, components: int, regions: list[str], order: int, delta: float
) -> dict:
    # Transform data into shape mode space.
    columns = data.filter(like="shcoeffs").columns
    transform = pca.transform(data[columns].values)

    # Calculate transformed means and standard deviations.
    means = transform.mean(axis=0)
    stds = transform.std(axis=0, ddof=1)

    # Create bins.
    map_points = np.arange(-2, 2.5, delta)
    bin_edges = [-np.inf] + [point + delta / 2 for point in map_points[:-1]] + [np.inf]
    transform_binned = np.digitize(transform / stds, bin_edges)

    # Initialize output dictionary.
    shape_modes: dict[str, list] = {}

    for region in regions:
        region_shape_modes = []

        suffix = f".{region}" if region != "DEFAULT" else ""
        offsets = calculate_region_offsets(data, region)

        for component in range(components):
            point_vector = np.zeros((components))

            for point in map_points:
                point_bin = np.digitize(point, bin_edges)
                point_vector[component] = point

                vector = means + np.multiply(stds, point_vector)
                indices = transform_binned[:, component] == point_bin

                mesh = construct_mesh_from_points(pca, vector, columns, order, suffix=suffix)

                if region == "DEFAULT" or not any(indices):
                    offset = None
                else:
                    offset = (
                        offsets["x"][indices].mean(),
                        offsets["y"][indices].mean(),
                        offsets["z"][indices].mean(),
                    )

                region_shape_modes.append(
                    {
                        "mode": component + 1,
                        "point": point,
                        "projections": extract_mesh_projections(mesh, extents=False, offset=offset),
                    }
                )

        shape_modes[region] = region_shape_modes

    return shape_modes


def calculate_region_offsets(data: pd.DataFrame, region: str) -> dict:
    if region == "DEFAULT":
        return {}

    x_deltas = data[f"CENTER_X.{region}"].values - data["CENTER_X"].values
    y_deltas = data[f"CENTER_Y.{region}"].values - data["CENTER_Y"].values
    z_deltas = data[f"CENTER_Z.{region}"].values - data["CENTER_Z"].values
    angles = data["angle"].values * np.pi / 180.0

    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)

    return {
        "x": x_deltas * cos_angles - y_deltas * sin_angles,
        "y": x_deltas * sin_angles + y_deltas * cos_angles,
        "z": z_deltas,
    }
