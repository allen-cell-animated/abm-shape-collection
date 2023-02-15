import numpy as np
from prefect import task


@task
def make_voxels_array(voxels: list[tuple[int, int, int]], scale: int = 1) -> np.ndarray:
    # Center voxels around (0,0,0).
    center_x, center_y, center_z = [round(x) for x in np.array(voxels).mean(axis=0)]
    all_xyz_centered = [(z - center_z, y - center_y, x - center_x) for x, y, z in voxels]

    # Create empty array.
    mins = np.min(all_xyz_centered, axis=0)
    maxs = np.max(all_xyz_centered, axis=0)
    height, width, length = np.subtract(maxs, mins) + 3
    array = np.zeros((height, width, length), dtype=np.uint8)

    # Fill in voxel array.
    all_xyz_offset = [
        (z - mins[0] + 1, y - mins[1] + 1, x - mins[2] + 1) for z, y, x in all_xyz_centered
    ]
    vals = [1] * len(all_xyz_offset)
    array[tuple(np.transpose(all_xyz_offset))] = vals

    # Scale the array.
    if scale > 1:
        array = array.repeat(scale, axis=0).repeat(scale, axis=1).repeat(scale, axis=2)

    return array
