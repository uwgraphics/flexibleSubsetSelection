# --- Imports ------------------------------------------------------------------

# Third party libraries
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


# --- Metric Functions ---------------------------------------------------------

def max(data, *_)-> np.array:
    """Returns the maximum of each feature of array"""
    return np.max(data, axis=0)

def min(data, *_) -> np.array:
    """Returns the minimum of each feature of array"""
    return np.min(data, axis=0)

def mean(array) -> np.array:
    """Returns the means of each column feature of array."""
    return np.mean(array, axis=0)

def range(array) -> np.array:
    """Returns the ranges of each column feature of array."""
    return np.ptp(array, axis=0)

def variance(array) -> np.array:
    """Returns the variance of each column feature of array."""
    return np.var(array, axis=0)

def positiveVariance(array) -> np.array:
    """Returns the positive variance of each column feature of array."""
    return mean(array) + variance(array)

def negativeVariance(array) -> np.array:
    """Returns the negative variance of each column feature of array."""
    return mean(array) - variance(array)

def hull(array) -> np.array:
    """Returns the convex hull area or volume of array."""
    hull = ConvexHull(array)
    return hull.volume if array.shape[1] > 2 else hull.area

def distanceMatrix(array) -> np.array:
    """Returns the distance matrix of an array or DataFrame."""
    if isinstance(array, pd.DataFrame):
        array = array.values
    distances = np.linalg.norm(array[:, np.newaxis] - array, axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances

def discreteDistribution(array) -> float:
    """
    Returns the discrete distribution of the one hot encoded array
    """
    return np.mean(array, axis=0)
