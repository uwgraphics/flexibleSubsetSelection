# --- Imports ------------------------------------------------------------------

# Third party libraries
import ot
import numpy as np
from scipy.ndimage import convolve
from sklearn.neighbors import LocalOutlierFactor


# --- Objective Functions ------------------------------------------------------

def preserveMetric(subset, metric, datasetMetric, p=1) -> float:
    """
    An objective function for preserving a metric between a dataset and a subset

    Args:
        subset (array): A subset of the full dataset
        metric (function): The metric function defining the distance
        p (optional): Specify the ord of the norm from Numpy options

    Returns: The absolute difference between the dataset and the subset 
        according to the given metric function.
    """
    subsetMetric = metric(subset)
    
    # If the metric results are scalars, use the absolute difference
    if np.isscalar(datasetMetric):
        return np.abs(datasetMetric - subsetMetric)

    # Otherwise, use np.linalg.norm for array-like metric results
    return np.linalg.norm(datasetMetric - subsetMetric, ord=1)

def distinctiveness(distances) -> float:
    """
    An objective function for maximizing the distance to the nearest neighbor
    for each point in space

    Args:
        distances (array): Distance matrix of points

    Returns: The negative of the sum of the distance to the nearest neighbor
    """
    return -np.sum(np.min(distances, axis=1))

def outlierness(subset, neighbors=20) -> float:
    """
    Compute the outlierness of a subset using the Local Outlier Factor (LOF).

    Args:
        subset (np.ndarray): The subset of the dataset.
        neighbors (int): Number of neighbors to use for LOF calculation.

    Returns:
        float: The mean outlierness score of the subset.
    """
    lof = LocalOutlierFactor(n_neighbors=neighbors)
    lof.fit(subset)
    return lof.negative_outlier_factor_

def discreteCoverage(array) -> float:
    """
    Computes the discrete coverage of the one hot encoded array

    Arg: array: One hot encoded subset array to compute discrete coverage
    """
    return -np.sum(np.minimum(np.ones(array.shape[1]), np.sum(array, axis=0)))

def earthMoversDistance(subset, dataset) -> float:
    """
    Computes the earth movers distance using the POT library

    Args:
        subset (array): subset of dataset
        dataset (array): the full dataset
    """
    return ot.emd2([], [], ot.dist(subset, dataset))

def pcpLineCrossings(array):
    """returns the total number of line crosses"""
    sum = 0
    w = np.array([[1, 1]])
    for i in range(array.shape[0]):
        convolution = convolve(np.sign(array[i] - array[i+1:]), w)[:, :-1]
        sum += np.ceil(np.sum(np.abs((np.abs(convolution)-2)/2)))
    return sum
