# --- Imports ------------------------------------------------------------------

# Standard library
from typing import Callable
import csv

# Third party
import gurobipy as gp

# Local files
from .loss import UniCriterion, MultiCriterion
from .sets import Dataset, Subset
from .timer import Timer


# --- Solver -------------------------------------------------------------------

class Solver():
    """
    A general optimization solver class for subset selection defined by a 
    solving algorithm and loss function, applied to calculate a subset.
    """
    def __init__(self, algorithm: Callable, 
                 lossFunction: (UniCriterion | MultiCriterion | None) = None,
                 logPath: str = "../data/solverLog.csv") -> None:
        """
        Initialize a subset selection solver with a solve algorithm and, 
        optionally, a loss function.

        Args:
            algorithm: The algorithm function to find the subset.
            loss: The loss function class object.
            logPath: The path to the solver log file.
        """
        self.algorithm = algorithm
        self.lossFunction = lossFunction
        self.logPath = logPath

        # Initialize the log file with headers if it doesn't exist
        try:
            with open(self.logPath, 'x', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow(["Loss Function", "Algorithm", "Dataset Length", 
                                 "Dataset Width", "Subset Length", 
                                 "Computation Time", "Loss"])
        except FileExistsError:
            pass

    def solve(self, dataset: Dataset, **parameters) -> Subset:
        """
        Solve for the optimal subset with the algorithm and loss function for 
        the specified dataset.

        Args:
            dataset: The dataset to solve by selecting a subset from.
            **parameters: Additional parameters for the algorithm function.

        Returns: The resulting subset of the selection solved for.
        """
        with Timer() as timer:
            z, loss = self.algorithm(dataset, self.lossFunction, **parameters)
        
        subset = Subset(dataset, z, timer.elapsedTime, loss)

        self.log(dataset.size, subset.size, self.lossFunction, 
                 self.algorithm.__name__, timer.elapsedTime, loss)

        print(f"Solved for {subset}.")

        return subset

    def log(self, datasetSize: tuple, subsetSize: tuple, 
            lossFunction: (UniCriterion | MultiCriterion), 
            algorithm: str, computationTime: float, loss: float):

        # Write log entry to the log file
        with open(self.logPath, 'a', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([str(lossFunction), algorithm, datasetSize[0], 
                             datasetSize[1], subsetSize[0], computationTime, 
                             loss])