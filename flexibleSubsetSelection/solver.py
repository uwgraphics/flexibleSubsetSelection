# --- Imports ------------------------------------------------------------------

import gurobipy as gp


# --- Solver -------------------------------------------------------------------

class Solver():
    def __init__(self, algorithm, loss=None):
        """
        Initialize a solver with a solve algorithm and a loss function.

        Args:
            algorithm (function): The solve algorithm           
            loss (object, optional): The loss function class object
        """
        self.algorithm = algorithm
        self.loss = loss

    def solve(self, dataset, **parameters):
        """
        Solve for optimal with algorithm and loss function for specified dataset

        Args:
            dataset (object): The dataset class object  
            **parameters: Additional parameters of the algorithm function          
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
        Create and set up a gp environment for integer optimization solvers.

        Arg: outputFlag (int): Flag.
        """
        self.environment = gp.Env(empty=True)
        self.environment.setParam("OutputFlag", outputFlag)
        self.environment.start()