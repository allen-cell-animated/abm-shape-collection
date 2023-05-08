from prefect import task

from .calculate_shape_stats import calculate_shape_stats
from .calculate_size_stats import calculate_size_stats
from .compile_shape_modes import compile_shape_modes
from .extract_shape_modes import extract_shape_modes
from .fit_pca_model import fit_pca_model
from .get_shape_coefficients import get_shape_coefficients
from .get_shape_properties import get_shape_properties
from .make_voxels_array import make_voxels_array
from .merge_shape_modes import merge_shape_modes

calculate_shape_stats = task(calculate_shape_stats)
calculate_size_stats = task(calculate_size_stats)
compile_shape_modes = task(compile_shape_modes)
extract_shape_modes = task(extract_shape_modes)
fit_pca_model = task(fit_pca_model)
get_shape_coefficients = task(get_shape_coefficients)
get_shape_properties = task(get_shape_properties)
make_voxels_array = task(make_voxels_array)
merge_shape_modes = task(merge_shape_modes)
