from prefect import task

from .calculate_shape_stats import calculate_shape_stats
from .calculate_size_stats import calculate_size_stats
from .construct_mesh_from_array import construct_mesh_from_array
from .construct_mesh_from_coeffs import construct_mesh_from_coeffs
from .construct_mesh_from_points import construct_mesh_from_points
from .extract_mesh_projections import extract_mesh_projections
from .extract_mesh_wireframe import extract_mesh_wireframe
from .extract_shape_modes import extract_shape_modes
from .fit_pca_model import fit_pca_model
from .get_shape_coefficients import get_shape_coefficients
from .get_shape_properties import get_shape_properties
from .make_voxels_array import make_voxels_array

calculate_shape_stats = task(calculate_shape_stats)
calculate_size_stats = task(calculate_size_stats)
construct_mesh_from_array = task(construct_mesh_from_array)
construct_mesh_from_coeffs = task(construct_mesh_from_coeffs)
construct_mesh_from_points = task(construct_mesh_from_points)
extract_mesh_projections = task(extract_mesh_projections)
extract_mesh_wireframe = task(extract_mesh_wireframe)
extract_shape_modes = task(extract_shape_modes)
fit_pca_model = task(fit_pca_model)
get_shape_coefficients = task(get_shape_coefficients)
get_shape_properties = task(get_shape_properties)
make_voxels_array = task(make_voxels_array)
