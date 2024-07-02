# --- Imports ------------------------------------------------------------------

# Standard library
from functools import partial

# Third party libraries
import ot

import numpy as np
import pandas as pd

from scipy.ndimage import convolve
from scipy.spatial import ConvexHull
from sklearn.neighbors import LocalOutlierFactor


# --- Loss Function ------------------------------------------------------------

class MultiCriterion():
    def __init__(self, objectives, parameters, weights=None):
        """
        Define a multi-criterion loss function with a set of objectives, 
        weights, and parameters

        Args:
            objectives (array): The objective functions to define the loss
            parameters (list of dict): The parameters for each objective
            weights (array, optional): The weights to assign to each objective

        Raises:
            ValueError: If weights or parameters has incorrect length.
        """
        
        # Initialize weights
        if weights is None:
            weights = np.ones(len(objectives))
        if len(weights) != len(objectives):
            raise ValueError("weights does not match length of objectives")
        if len(parameters) != len(objectives):
            raise ValueError("parameters does not match length of objectives")
        
        self.objectives = objectives
        self.parameters = parameters
        self.weights = weights

        # Generate the combined objective function
        self.calculate = partial(self._loss)

    def _loss(self, dataset, z) -> float:
        """
        Compute the overall loss function by evaluating each objective function
        with its corresponding parameters and combining them with weights.

        Args:
            dataset (object): The dataset object containing the arrays.
            z (array): The subset indices or other selector.

        Returns:
            float: The computed value of the overall loss function.
        """
        loss = 0.0
        zipped = zip(self.objectives, self.parameters, self.weights)
        for objective, params, weight in zipped:
            array = getattr(dataset, params.get('solveArray', 'dataArray'))
            subset = select(array, z, selectBy=params.get('selectBy', 'row'))

            objectiveParameters = {
                key: value for key, value in params.items() 
                if key not in ['solveArray', 'selectBy']
            }
            objectiveLoss = weight * objective(subset, **objectiveParameters)
            loss += objectiveLoss
        return loss
    
class UniCriterion():
    def __init__(self, objective, solveArray="dataArray", selectBy="row", 
                 **parameters):
        """
        Define a loss function with an objective and optional parameters for 
        subset selection.

        Args:
            objective (function): The objective function to define the loss.
            solveArray (str, optional): The name of the array in dataset to use 
                for subset selection. Default is "dataArray".
            selectBy (str, optional): The method to select subset from array. 
                Default is "row".
            **parameters: Additional parameters of the objective function.
        """
        self.objective = objective
        self.solveArray = solveArray
        self.selectBy = selectBy
        self.parameters = parameters

    def calculate(self, dataset, z) -> float:
        """
        Compute the loss by evaluating the objective with its parameters on the 
        selected subset.

        Args:
            dataset (object): The dataset object containing the arrays.
            z (array): The subset indices or other selector.

        Returns:
            float: The computed value of the loss function.
        """
        array = getattr(dataset, self.solveArray)
        subset = select(array, z, selectBy=self.selectBy)
        return self.objective(subset, **self.parameters)

def select(array, z, selectBy) -> np.array:
    """
    Selects a subset from  array according to indicator z

    Args:
        array: to select from
        z: indicator of what to select
        selectBy: name of selection method to select by
    
        Returns: The selected subset from array
    """
    if selectBy == "row":
        return array[z == 1]
    elif selectBy == "matrix":
        return array[z == 1][:, z == 1]
    else:
        raise ValueError("unknown selection method specified")      

# --- Metric Functions ---------------------------------------------------------

def maxMetric(data, *_)-> np.array:
    """Returns the maximum of each feature of array"""
    return np.max(data, axis=0)

def minMetric(data, *_) -> np.array:
    """Returns the minimum of each feature of array"""
    return np.min(data, axis=0)

def meanMetric(array) -> np.array:
    """Returns the means of each column feature of array."""
    return np.mean(array, axis=0)

def rangeMetric(array) -> np.array:
    """Returns the ranges of each column feature of array."""
    return np.ptp(array, axis=0)

def varianceMetric(array) -> np.array:
    """Returns the variance of each column feature of array."""
    return np.var(array, axis=0)

def positiveVariance(array) -> np.array:
    """Returns the positive variance of each column feature of array."""
    return meanMetric(array) + varianceMetric(array)

def negativeVariance(array) -> np.array:
    """Returns the negative variance of each column feature of array."""
    return meanMetric(array) - varianceMetric(array)

def hullMetric(array) -> np.array:
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
