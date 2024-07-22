import numpy as np
from aicsshparam import shparam, shtools


def get_shape_coefficients(array: np.ndarray, reference: np.ndarray, order: int) -> dict:
    """
    Calculate spherical harmonic coefficients for binary image array.

    Parameters
    ----------
    array
        Binary image array.
    reference
        Binary reference array that determines alignment angle.
    order
        Order of the spherical harmonics coefficient parametrization.

    Returns
    -------
    :
        Dictionary of spherical harmonics, angle, and MSE.
    """

    _, angle = shtools.align_image_2d(image=reference)
    aligned_array = shtools.apply_image_alignment_2d(array, angle).squeeze()

    (coeffs, reconstructed_array), (_, _, downsampled_array, _) = shparam.get_shcoeffs(
        image=aligned_array, lmax=order, compute_lcc=False, alignment_2d=False
    )

    mse = shtools.get_reconstruction_error(downsampled_array, reconstructed_array)
    coeffs["angle"] = angle
    coeffs["mse"] = mse

    return coeffs
