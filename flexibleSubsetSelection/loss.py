# --- Imports ------------------------------------------------------------------

# Standard library
from functools import partial

# Third party libraries
import numpy as np


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