import numpy as np
from skimage.measure import regionprops_table


def get_shape_properties(array: np.ndarray, properties: list[str]) -> dict:
    flattened_array = array.max(axis=0)

    properties_dict = regionprops_table(flattened_array, properties=properties)
    props = {key: value[0] for key, value in properties_dict.items()}

    return props
