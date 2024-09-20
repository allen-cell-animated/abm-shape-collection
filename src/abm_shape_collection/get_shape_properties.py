import numpy as np
from skimage.measure import regionprops_table


def get_shape_properties(array: np.ndarray, properties: list[str]) -> dict:
    """
    Calculate shape properties for binary image array.

    Shape properties must be value properties from skimage.measure.regionprops.
    Image is projected along the first axis.

    Parameters
    ----------
    array
        Binary image array.
    properties
        List of shape properties to calculate.

    Returns
    -------
    :
        Dictionary of calculated shape properties.
    """

    flattened_array = array.max(axis=0)

    properties_dict = regionprops_table(flattened_array, properties=properties)
    return {key: value[0] for key, value in properties_dict.items()}
