# --- Imports ------------------------------------------------------------------

import gurobipy as gp


# --- Solver -------------------------------------------------------------------

class Solver():
    """
    A general optimization solver class for subset selection defined by a 
    solving algorithm and loss function, applied to calculate a subset.
    """
    def __init__(self, algorithm, loss=None):
        """
        Initialize a subset selection solver with a solve algorithm and, 
        optionally, a loss function.

        Args:
            algorithm (function): The algorithm function to find the subset.
            loss (Loss object, optional): The loss function class object.
        """
        self.algorithm = algorithm
        self.loss = loss

    def solve(self, dataset, **parameters):
        """
        Solve for the optimal subset with the algorithm and loss function for the specified dataset.

        Args:
            dataset (Dataset object): The dataset class object.
            **parameters: Additional parameters for the algorithm function.

        Returns:
            tuple: The result of the algorithm function specified.
        """
        if self.loss is None:
            return self.algorithm(dataset = dataset, 
                                  environment = self.environment, 
                                  **parameters)
        else:
            return self.algorithm(dataset = dataset, 
                                  lossFunction = self.loss,
                                  **parameters)
    
    def createEnvironment(self, outputFlag=0):
        """
        Create and set up an environment required by the Gurobi solver

        Arg: outputFlag (int): Flag for Gurobi output.
        """
        self.environment = gp.Env(empty=True)
        self.environment.setParam("OutputFlag", outputFlag)
        self.environment.start()