import numpy as np


def select(array: np.ndarray, z: np.ndarray, selectBy: str) -> np.ndarray:
    """
    Selects a subset from array according to indicator z

    Args:
        array: The array to select from.
        z: The indicator vector indicating which elements to select.
        selectBy: The method to select the subset (row or matrix).

    Returns: The selected subset from the array.

    Raises: ValueError: If an unknown selection method is specified.
    """
    if selectBy == "row":
        return array[z == 1]
    elif selectBy == "matrix":
        return array[z == 1][:, z == 1]
    else:
        raise ValueError("Unknown selection method specified.")
