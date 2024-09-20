from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from abm_shape_collection.construct_mesh_from_points import construct_mesh_from_points
from abm_shape_collection.extract_mesh_projections import extract_mesh_projections


def extract_shape_modes(
    pca: PCA,
    data: pd.DataFrame,
    components: int,
    regions: list[str],
    order: int,
    delta: float,
    _construct_mesh_from_points: Callable = construct_mesh_from_points,
    _extract_mesh_projections: Callable = extract_mesh_projections,
) -> dict:
    """
    Extract shape modes (latent walks in PC space) at the specified intervals.

    Parameters
    ----------
    pca
        Fit PCA object.
    data
        Sample data, with shape coefficients as columns.
    components
        Number of shape coefficients components.
    regions
        List of regions.
    order
        Order of the spherical harmonics coefficient parametrization.
    delta
        Interval for latent walk, bounded by -2 and +2 standard deviations.

    Returns
    -------
    :
        Map of regions to lists of shape modes at select points.
    """

    # pylint: disable=too-many-locals

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
            point_vector = np.zeros(components)

            for point in map_points:
                point_bin = np.digitize(point, bin_edges)
                point_vector[component] = point

                vector = means + np.multiply(stds, point_vector)
                indices = transform_binned[:, component] == point_bin

                mesh = _construct_mesh_from_points(pca, vector, columns, order, suffix=suffix)

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
                        "projections": _extract_mesh_projections(
                            mesh, extents=False, offset=offset
                        ),
                    }
                )

        shape_modes[region] = region_shape_modes

    return shape_modes


def calculate_region_offsets(data: pd.DataFrame, region: str) -> dict:
    """
    Calculate offsets for non-default regions.

    Parameters
    ----------
    data
        Centroid location data.
    region
        Name of region (skipped if region is DEFAULT).

    Returns
    -------
    :
        Map of offsets in the x, y, and z directions.
    """

    if region == "DEFAULT":
        return {}

    x_deltas = data[f"CENTER_X.{region}"].to_numpy() - data["CENTER_X"].to_numpy()
    y_deltas = data[f"CENTER_Y.{region}"].to_numpy() - data["CENTER_Y"].to_numpy()
    z_deltas = data[f"CENTER_Z.{region}"].to_numpy() - data["CENTER_Z"].to_numpy()
    angles = data["angle"].to_numpy() * np.pi / 180.0

    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)

    return {
        "x": x_deltas * cos_angles - y_deltas * sin_angles,
        "y": x_deltas * sin_angles + y_deltas * cos_angles,
        "z": z_deltas,
    }
