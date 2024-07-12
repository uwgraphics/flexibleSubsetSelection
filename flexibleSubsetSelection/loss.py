# --- Imports ------------------------------------------------------------------

# Standard library
from functools import partial
from typing import Callable

# Third party
import numpy as np
from numpy.typing import ArrayLike

# Local
from . import sets

# --- Loss Function ------------------------------------------------------------

class Base():
    """
    Base class for MultiCriterion and UniCriterion loss function classes 
    providing shared select function.
    """

    def select(array: np.ndarray, z: ArrayLike, selectBy: str) -> np.array:
        """
        Selects a subset from array according to indicator z

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
            raise ValueError("Unknown selection method specified.")

class MultiCriterion(Base):
    """
    Create and apply multicriterion loss functions from a set of objectives and 
    corresponding weights for subset selection.
    """

    def __init__(self, objectives: ArrayLike, parameters: ArrayLike, 
                 weights: ArrayLike = None) -> None:
        """
        Define a multi-criterion loss function with a set of objectives, 
        weights, and parameters

        Args:
            objectives: The objective functions to define the loss
            parameters: The set of dictionaries of parameters for each objective
            weights: The weights to assign to each objective

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

    def _loss(self, dataset: sets.Dataset, z: ArrayLike) -> float:
        """
        Compute the overall loss function by evaluating each objective function
        with its corresponding parameters and combining them with weights.

        Args:
            dataset: The dataset object containing the data.
            z: The indicator vector indicating which samples from the dataset 
                are included in the subset.
        Returns:
            float: The computed value of the overall loss function.
        """
        print(dataset)
        loss = 0.0
        zipped = zip(self.objectives, self.parameters, self.weights)
        for objective, params, weight in zipped:
            array = getattr(dataset, params.get('solveArray', 'dataArray'))
            seletBy = params.get('selectBy', 'row')
            subset = self.select(array, z, selectBy=selectBy)

            objectiveParameters = {
                key: value for key, value in params.items() 
                if key not in ['solveArray', 'selectBy']
            }
            objectiveLoss = weight * objective(subset, **objectiveParameters)
            loss += objectiveLoss
        return loss
    
class UniCriterion(Base):
    """
    Create and apply a unicriterion loss function from an objective, apply to a 
    particular data array for subset selection.
    """

    def __init__(self, objective: Callable, solveArray: str = "dataArray", 
                 selectBy: str = "row", **parameters):
        """
        Define a loss function with an objective and optional parameters for 
        subset selection.

        Args:
            objective: The objective function to define the loss.
            solveArray: The name of the array in dataset to use 
                for subset selection. Default is "dataArray".
            selectBy: The method to select subset from array. 
            **parameters: Additional parameters of the objective function.
        """
        self.objectives = objective
        self.solveArray = solveArray
        self.selectBy = selectBy
        self.parameters = parameters

    def calculate(self, dataset: sets.Dataset, z: ArrayLike) -> float:
        """
        Compute the loss by evaluating the objective with its parameters on the 
        selected subset.

        Args:
            dataset: The dataset object containing the data.
            z: The indicator vector indicating which samples from the dataset 
                are included in the subset.

        Returns:
            float: The computed value of the loss function.
        """
        array = getattr(dataset, self.solveArray)
        subset = self.select(array, z, selectBy=self.selectBy)
        return self.objectives(subset, **self.parameters)