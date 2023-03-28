import importlib
import sys

from prefect import task

from .calculate_shape_stats import calculate_shape_stats
from .calculate_size_stats import calculate_size_stats
from .compile_shape_modes import compile_shape_modes
from .extract_shape_modes import extract_shape_modes
from .fit_pca_model import fit_pca_model
from .get_shape_coefficients import get_shape_coefficients
from .make_voxels_array import make_voxels_array
from .merge_shape_modes import merge_shape_modes

TASK_MODULES = [
    calculate_shape_stats,
    calculate_size_stats,
    compile_shape_modes,
    extract_shape_modes,
    fit_pca_model,
    get_shape_coefficients,
    make_voxels_array,
    merge_shape_modes,
]

for task_module in TASK_MODULES:
    MODULE_NAME = task_module.__name__
    module = importlib.import_module(f".{MODULE_NAME}", package=__name__)
    setattr(sys.modules[__name__], MODULE_NAME, task(getattr(module, MODULE_NAME)))
