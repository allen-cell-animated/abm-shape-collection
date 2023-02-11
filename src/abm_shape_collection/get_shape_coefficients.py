import numpy as np
from aicsshparam import shparam, shtools
from prefect import task


@task
def get_shape_coefficients(array: np.ndarray, reference: np.ndarray, lmax: int) -> dict:
    _, angle = shtools.align_image_2d(image=reference)
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()

    (coeffs, _), _ = shparam.get_shcoeffs(
        image=aligned_array, lmax=lmax, compute_lcc=False, alignment_2d=False
    )

    return coeffs
