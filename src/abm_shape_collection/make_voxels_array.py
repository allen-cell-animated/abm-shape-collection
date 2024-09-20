from __future__ import annotations

import numpy as np


def make_voxels_array(
    voxels: list[tuple[int, int, int]],
    reference: list[tuple[int, int, int]] | None = None,
    scale: int = 1,
) -> np.ndarray:
    """
    Convert list of voxels to a binary image array.

    Parameters
    ----------
    voxels
        List of (x, y, z) voxels.
    reference
        List of (x, y, z) reference voxels used to define center and image size.
    scale
        Scaling factor for image array.

    Returns
    -------
    :
        Binary image array.
    """

    # Set reference if not specified.
    if reference is None:
        reference = voxels

    # Center voxels around (0,0,0).
    center_x, center_y, center_z = [round(x) for x in np.array(reference).mean(axis=0)]
    voxels_centered = [(z - center_z, y - center_y, x - center_x) for x, y, z in voxels]
    reference_centered = [(z - center_z, y - center_y, x - center_x) for x, y, z in reference]

    # Create empty array.
    mins = np.min(reference_centered, axis=0)
    maxs = np.max(reference_centered, axis=0)
    height, width, length = np.subtract(maxs, mins) + 3
    array = np.zeros((height, width, length), dtype=np.uint8)

    # Fill in voxel array.
    array_offset = [
        (z - mins[0] + 1, y - mins[1] + 1, x - mins[2] + 1) for z, y, x in voxels_centered
    ]
    array[tuple(np.transpose(array_offset))] = [1] * len(array_offset)

    # Scale the array.
    if scale > 1:
        array = array.repeat(scale, axis=0).repeat(scale, axis=1).repeat(scale, axis=2)

    return array
