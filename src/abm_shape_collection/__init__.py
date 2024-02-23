"""Tasks for analyzing cell shapes including spherical harmonics and shape modes."""

from prefect import task

from .calculate_feature_statistics import calculate_feature_statistics
from .calculate_shape_statistics import calculate_shape_statistics
from .construct_mesh_from_array import construct_mesh_from_array
from .construct_mesh_from_coeffs import construct_mesh_from_coeffs
from .construct_mesh_from_points import construct_mesh_from_points
from .extract_mesh_projections import extract_mesh_projections
from .extract_mesh_wireframe import extract_mesh_wireframe
from .extract_shape_modes import extract_shape_modes
from .extract_voxel_contours import extract_voxel_contours
from .fit_pca_model import fit_pca_model
from .get_shape_coefficients import get_shape_coefficients
from .get_shape_properties import get_shape_properties
from .make_voxels_array import make_voxels_array

calculate_shape_statistics = task(calculate_shape_statistics)
calculate_feature_statistics = task(calculate_feature_statistics)
construct_mesh_from_array = task(construct_mesh_from_array)
construct_mesh_from_coeffs = task(construct_mesh_from_coeffs)
construct_mesh_from_points = task(construct_mesh_from_points)
extract_mesh_projections = task(extract_mesh_projections)
extract_mesh_wireframe = task(extract_mesh_wireframe)
extract_shape_modes = task(extract_shape_modes)
extract_voxel_contours = task(extract_voxel_contours)
fit_pca_model = task(fit_pca_model)
get_shape_coefficients = task(get_shape_coefficients)
get_shape_properties = task(get_shape_properties)
make_voxels_array = task(make_voxels_array)
