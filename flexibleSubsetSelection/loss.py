# --- Imports ------------------------------------------------------------------

# Standard library
from functools import partial
import logging
from typing import Any, Callable, Dict, List

# Third party
import numpy as np
from numpy.typing import ArrayLike

# Local files
from .sets import Dataset, Subset

# Setup logger
logger = logging.getLogger(__name__)


# --- Loss Function ------------------------------------------------------------

class MultiCriterion():
    """
    Create and apply multi-criterion loss functions from a set of objectives and
    corresponding weights for subset selection.
    """

    def __init__(self, objectives: List[Callable], 
                 parameters: List[Dict[str, Any]], 
                 weights: (np.ndarray | None) = None) -> None:
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
            raise ValueError("Weights length must match objectives length.")
        if len(parameters) != len(objectives):
            raise ValueError("Parameters length must match objectives length.")

        self.objectives = objectives
        self.parameters = parameters
        self.weights = weights

        # Generate the combined objective function
        self.calculate = partial(self._loss)

        logger.debug("Initialized a multi-criterion loss function with "
                     "objectives: %s, parameters: %s, and weights: %s", 
                     objectives, parameters, weights)

    def _loss(self, dataset: Dataset, z: ArrayLike) -> float:
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
        loss = 0.0
        zipped = zip(self.objectives, self.parameters, self.weights)
        for objective, params, weight in zipped:
            # retrieve solve array from attributes or default to dataArray
            array = getattr(dataset, params.get("solveArray", "dataArray"))

            # retrieve selectBy from attributes or default to row
            selectBy = params.get("selectBy", "row")
            subset = select(array, z, selectBy=selectBy)

            # retrieve any remaining parameters as objective parameters
            objectiveParameters = {
                key: value for key, value in params.items() 
                if key not in ["solveArray", "selectBy"]
            }
            objectiveLoss = weight * objective(subset, **objectiveParameters)
            loss += objectiveLoss
        return loss

    def __str__(self) -> str:
        """
        Return a string representation of the MultiCriterion loss function.
        """
        zipped = zip(self.objectives, self.parameters, self.weights)
        objectives = []
        for objective, parameter, weight in zipped:
            parameters = []
            for key, value in parameter.items():
                if callable(value):
                    parameters.append(f"{key}: {value.__name__}")
                if key == "solveArray" and value != "dataArray":
                    parameters.append(value)
            parameters = ", ".join(parameters)

            if len(parameters) > 0:
                objectives.append((f"{weight}*({objective.__name__}, "
                                   f"{parameters})"))
            else:
                objectives.append(f"{weight}*({objective.__name__})")
        
        objectives = " + ".join(objectives)
        return f"Multi-criterion: {objectives}"


class UniCriterion():
    """
    Create and apply a unicriterion loss function from an objective, apply to a 
    particular data array for subset selection.
    """

    def __init__(self, objective: Callable, solveArray: str = "dataArray", 
                 selectBy: str = "row", **parameters: Any):
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
        self.objective = objective
        self.solveArray = solveArray
        self.selectBy = selectBy
        self.parameters = parameters

        logger.info("Initialized a uni-criterion loss function with "
                    "objective: %s, solve array: %s, selection method: %s, "
                    "and parameters: %s", 
                    objective.__name__, solveArray, selectBy, parameters)


    def calculate(self, dataset: Dataset, z: ArrayLike) -> float:
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
        subset = select(array, z, selectBy=self.selectBy)
        return self.objective(subset, **self.parameters)
    
    def __str__(self) -> str:
        """
        Return a string representation of the UniCriterion loss function.
        """
        parameters = []
        for key, value in self.parameters.items():
            if callable(value):
                parameters.append(value.__name__)
        if self.solveArray != "dataArray":
                parameters.append(self.solveArray)
        parameters = ", ".join(parameters)

        if parameters:
            return f"Uni-criterion: {self.objective.__name__}, {parameters}"
        return f"Uni-criterion: {self.objective.__name__}"

def select(array: np.ndarray, z: ArrayLike, selectBy: str) -> np.ndarray:
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