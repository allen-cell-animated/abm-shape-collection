import numpy as np
from aicsshparam import shparam, shtools


def get_shape_coefficients(array: np.ndarray, reference: np.ndarray, lmax: int) -> dict:
    _, angle = shtools.align_image_2d(image=reference)
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()

    (coeffs, reconstructed_array), (_, _, downsampled_array, _) = shparam.get_shcoeffs(
        image=aligned_array, lmax=lmax, compute_lcc=False, alignment_2d=False
    )

    mse = shtools.get_reconstruction_error(downsampled_array, reconstructed_array)
    center = np.argwhere(aligned_array == 1).mean(axis=0)

    coeffs["angle"] = angle
    coeffs["mse"] = mse
    coeffs["center_z"] = center[0]
    coeffs["center_y"] = center[1]
    coeffs["center_x"] = center[2]

    return coeffs
