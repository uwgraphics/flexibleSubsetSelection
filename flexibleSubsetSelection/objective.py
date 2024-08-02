# --- Imports ------------------------------------------------------------------

# Standard library
from collections import Counter
from typing import Callable

# Third party
import ot
from ott.geometry import costs, pointcloud

import numpy as np
from numpy.typing import ArrayLike

from scipy.ndimage import convolve
from sklearn.neighbors import LocalOutlierFactor


# --- Objective Functions ------------------------------------------------------

def preserveMetric(subset: np.ndarray, metric: Callable, 
                   datasetMetric: ArrayLike, p: int | str = 1) -> float:
    """
    An objective function for preserving a metric between a dataset and a subset

    Args:
        subset: A subset of the full dataset
        metric: The metric function to apply to the data
        datasetMetric: The metric results on the dataset
        p: Specify the ord of the norm from Numpy options

    Returns: The absolute difference between the dataset and the subset 
        according to the given metric function.
    """
    subsetMetric = metric(subset)
    
    # If the metric results are scalars, use the absolute difference
    if np.isscalar(datasetMetric):
        return np.abs(datasetMetric - subsetMetric)

    # Otherwise, use np.linalg.norm for array-like metric results
    return np.linalg.norm(datasetMetric - subsetMetric, ord=p)

def distinctness(distances: np.ndarray) -> float:
    """
    An objective function for maximizing the distance to the nearest neighbor
    for each point in space

    Args:
        distances: Distance matrix of points

    Returns: The negative of the sum of the distance to the nearest neighbor
    """
    return -np.sum(np.min(distances, axis=1))

def outlierness(subset: np.ndarray, neighbors: int = 20) -> float:
    """
    Compute the outlierness of a subset using the Local Outlier Factor (LOF).

    Args:
        subset: The subset of the dataset.
        neighbors: Number of neighbors to use for LOF calculation.

    Returns:
        float: The mean outlierness score of the subset.
    """
    lof = LocalOutlierFactor(n_neighbors=neighbors)
    lof.fit(subset)
    return lof.negative_outlier_factor_

def discreteDistribution(array: np.ndarray) -> float:
    return 0

def discreteCoverage(array: np.ndarray) -> float:
    """
    Computes the discrete coverage of the one hot encoded array

    Args: 
        array: One hot encoded subset array to compute discrete coverage
    """
    return -np.sum(np.minimum(np.ones(array.shape[1]), np.sum(array, axis=0)))

def earthMoversDistance(subset: np.ndarray, dataset: np.ndarray) -> float:
    """
    Computes the earth movers distance using the POT library

    Args:
        subset: subset of dataset
        dataset: the full dataset
    """
    return ot.emd2([], [], ot.dist(subset, dataset))

def sinkhornDistance(distances: np.ndarray, datasetLength, subsetLength, 
                     reg: float = 0.1, verbose: bool = False) -> float:
    """
    Computes the Sinkhorn distance using the POT library.

    Args:
        distances (np.ndarray): distance matrix .
        reg (float, optional): Regularization parameter. Defaults to 0.1.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        float: Sinkhorn distance.
    """
    print(distances.shape)
    return ot.sinkhorn2(np.ones(datasetLength) / datasetLength, 
                        np.ones(subsetLength) / subsetLength, 
                        distances, 
                        reg,
                        stopThr=1e-05,
                        verbose=verbose)

def pcpLineCrossings(array: np.ndarray) -> int:
    """Returns the total number of line crosses"""
    sum = 0
    w = np.array([[1, 1]])
    for i in range(array.shape[0]):
        convolution = convolve(np.sign(array[i] - array[i+1:]), w)[:, :-1]
        sum += np.ceil(np.sum(np.abs((np.abs(convolution)-2)/2)))
    return sum

def spread(distances: np.ndarray) -> float:
    """Computes the total distance between all points in distances matrix"""
    np.fill_diagonal(distances, 0)
    return -np.sum(distances)

def clusterCenters(array: np.ndarray, clusterCenters: np.ndarray) -> float:
    """
    An objective function for minimizing the distance to the nearest point for 
    each cluster center.

    Args:
        array (np.ndarray): Array of datapoints in the set.
        clusterCenters (np.ndarray): Array of cluster centers.

    Returns: The sum of distances to the nearest point for each cluster center.
    """
    # Calculate pairwise distance between data points and cluster centers
    difference = array[:, np.newaxis, :] - clusterCenters[np.newaxis, :, :]
    distances = np.linalg.norm(difference, axis=2)

    # Find the minimum distance to a data point for each cluster center
    minDistances = np.min(distances, axis=0)
    
    # Return the sum of these minimum distances
    return np.sum(minDistances)

def emdCategorical(subset, dataset, features, categorical, categories):
    emd_losses = []
    for category in categories:
        subset_data = subset.loc[subset[categorical] == category, features].values
        dataset_data = dataset.loc[dataset[categorical] == category, features].values
        emd_loss = ot.emd2([], [], ot.dist(subset_data, dataset_data))
        emd_losses.append(emd_loss)
    return emd_losses

def entropy(array: np.ndarray) -> float:
    counts = Counter(map(tuple, array))
    total = sum(counts.values())
    probabilities = np.array(list(counts.values()))/total
    return np.sum(probabilities * np.log(probabilities))

def sinkhorn(subset, fullData, solveFunction):
    geometry = pointcloud.PointCloud(fullData, subset)
    sinkhornOutput = solveFunction(geometry)
    return sinkhornOutput.reg_ot_cost