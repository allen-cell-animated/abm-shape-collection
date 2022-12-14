from prefect import task
import numpy as np
from aicsshparam import shparam, shtools


@task
def calculate_sh_coefficients(array: np.ndarray, reference: np.ndarray, lmax: int) -> dict:
    _, angle = shtools.align_image_2d(image=reference)
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()

    (coeffs, _), _ = shparam.get_shcoeffs(
        image=aligned_array, lmax=lmax, compute_lcc=False, alignment_2d=False
    )

    return coeffs
