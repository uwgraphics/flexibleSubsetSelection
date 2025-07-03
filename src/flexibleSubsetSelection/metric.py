# --- Imports ------------------------------------------------------------------

# Standard library
from typing import Any, Callable

# Third party
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

# Local files
from .dataset import Dataset
from . import logger

# Setup logger
log = logger.setup(name=__name__)

# --- Metric Functions ---------------------------------------------------------

class Metric:
    """
    A class for queuing, computing, and caching metrics of datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        function: Callable,
        array: str,
        params: dict[str, Any] | None = None,
        indices: list[int]| None = None
    ):
        """
        Initialize a metric with a metric function and additional parameters
        """
        self.function = function
        self.name = name
        self.params = params
        self.array = array
        self._indices = indices
        self._dataset = dataset 
        self._value = None

        log.info("Queued metric '%s' on '%s'.", self.name, self.array)

    def __call__(self):
        if self._value is None:
            array = getattr(self._dataset, self.array)
            if self._indices is not None:
                array = array[:, self._indices]
            if self.params is None:
                self._value = self.function(array)
            else:
                self._value = self.function(array, **self.params)
            log.info("Evaluated metric '%s' on '%s'.", self.name, self.array)
        return self._value

def max(array: np.ndarray) -> np.ndarray:
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


def discreteDistribution(array: np.ndarray) -> np.ndarray:
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
