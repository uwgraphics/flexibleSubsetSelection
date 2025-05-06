# --- Imports ------------------------------------------------------------------

# Third party
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans


# --- Metric Functions ---------------------------------------------------------

def max(array: np.ndarray)-> np.ndarray:
    """Returns the maximum of each feature of array"""
    return np.max(array, axis=0)

def min(array: np.ndarray) -> np.ndarray:
    """Returns the minimum of each feature of array"""
    return np.min(array, axis=0)

def mean(array: np.ndarray) -> np.ndarray:
    """Returns the means of each column feature of array."""
    return np.mean(array, axis=0)

def range(array: np.ndarray) -> np.ndarray:
    """Returns the ranges of each column feature of array."""
    return np.ptp(array, axis=0)

def variance(array: np.ndarray) -> np.ndarray:
    """Returns the variance of each column feature of array."""
    return np.var(array, axis=0)

def positiveVariance(array: np.ndarray) -> np.ndarray:
    """Returns the positive variance of each column feature of array."""
    return mean(array) + variance(array)

def negativeVariance(array: np.ndarray) -> np.ndarray:
    """Returns the negative variance of each column feature of array."""
    return mean(array) - variance(array)

def hull(array: np.ndarray) -> np.ndarray:
    """Returns the convex hull area or volume of array."""
    hull = ConvexHull(array)
    return hull.volume if array.shape[1] > 2 else hull.area

def distanceMatrix(array: np.ndarray) -> np.ndarray:
    """Returns the distance matrix of an array or DataFrame."""
    if isinstance(array, pd.DataFrame):
        array = array.values
    distances = np.linalg.norm(array[:, np.newaxis] - array, axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances

def discreteDistribution(array: np.ndarray) -> float:
    """
    Returns the discrete distribution of the one hot encoded array
    """
    return np.mean(array, axis=0)

def clusterCenters(array: np.ndarray, k: int) -> np.ndarray:
    """
    Returns the cluster centers of the k-means clustering for k clusters of the 
    data in array.

    Args:
        array (np.ndarray): Array of datapoints in the set.
        k (int): Number of clusters.

    Returns:
        np.ndarray: Array of cluster centers.
    """
    kmeans = KMeans(n_clusters=k).fit(array)
    return kmeans.cluster_centers_