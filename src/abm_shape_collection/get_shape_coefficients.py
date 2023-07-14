import numpy as np
from aicsshparam import shparam, shtools


def get_shape_coefficients(array: np.ndarray, reference: np.ndarray, lmax: int) -> dict:
    _, angle = shtools.align_image_2d(image=reference)
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()

    (coeffs, reconstructed_array), (_, _, downsampled_array, _) = shparam.get_shcoeffs(
        image=aligned_array, lmax=lmax, compute_lcc=False, alignment_2d=False
    )

    mse = shtools.get_reconstruction_error(downsampled_array, reconstructed_array)
    coeffs["angle"] = angle
    coeffs["mse"] = mse

    return coeffs
