# --- Imports ------------------------------------------------------------------

# Standard library
from typing import Callable

# Third party
import gurobipy as gp

# Local
from . import loss
from . import sets

# --- Solver -------------------------------------------------------------------

class Solver():
    """
    A general optimization solver class for subset selection defined by a 
    solving algorithm and loss function, applied to calculate a subset.
    """
    def __init__(self, algorithm: Callable, 
                 loss: loss.UniCriterion | loss.MultiCriterion = None) -> None:
        """
        Initialize a subset selection solver with a solve algorithm and, 
        optionally, a loss function.

        Args:
            algorithm: The algorithm function to find the subset.
            loss: The loss function class object.
        """
        self.algorithm = algorithm
        self.loss = loss

    def solve(self, dataset: sets.Dataset, **parameters) -> sets.Subset:
        """
        Solve for the optimal subset with the algorithm and loss function for 
        the specified dataset.

        Args:
            dataset: The dataset to solve by selecting a subset from.
            **parameters: Additional parameters for the algorithm function.

        Returns: The resulting subset of the selection solved for.
        """
        if self.loss is None:
            z, time, loss = self.algorithm(dataset = dataset, 
                                           environment = self.environment, 
                                           **parameters)
        else:
            z, time, loss = self.algorithm(dataset = dataset, 
                                           lossFunction = self.loss,
                                           **parameters)
        
        return sets.Subset(dataset, z, time, loss)
    
    def createEnvironment(self, outputFlag: int = 0):
        """
        Create and set up an environment required by the Gurobi solver

        Arg: outputFlag: Flag for Gurobi output.
        """
        self.environment = gp.Env(empty=True)
        self.environment.setParam("OutputFlag", outputFlag)
        self.environment.start()