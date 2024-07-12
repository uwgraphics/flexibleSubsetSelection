# --- Imports ------------------------------------------------------------------

# Standard library
from typing import Callable
import csv

# Third party
import gurobipy as gp

# Local
from . import loss
from . import sets
from timer import Timer

# --- Solver -------------------------------------------------------------------

class Solver():
    """
    A general optimization solver class for subset selection defined by a 
    solving algorithm and loss function, applied to calculate a subset.
    """
    def __init__(self, algorithm: Callable, 
                 loss: loss.UniCriterion | loss.MultiCriterion = None,
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
        self.loss = loss
        self.logPath = logPath

        # Initialize the log file with headers if it doesn't exist
        try:
            with open(self.logPath, 'x', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow(["Objective", "Algorithm", "Dataset Length", 
                                 "Dataset Width", "Subset Length", 
                                 "Computation Time", "Loss"])
        except FileExistsError:
            pass

    def solve(self, dataset: sets.Dataset, **parameters) -> sets.Subset:
        """
        Solve for the optimal subset with the algorithm and loss function for 
        the specified dataset.

        Args:
            dataset: The dataset to solve by selecting a subset from.
            **parameters: Additional parameters for the algorithm function.

        Returns: The resulting subset of the selection solved for.
        """
        with Timer() as timer:
            z, loss = self.algorithm(dataset, self.loss, **parameters)
        
        subset = sets.Subset(dataset, z, timer.elapsedTime, loss)
        self.log(dataset.size, subset.size, self.loss.objectives,
                 self.algorithm.__name__, timer.elapsedTime, loss)

        return subset

    def log(self, datasetSize: tuple, subsetSize: tuple, objectives, 
            algorithm: str, computationTime: float, loss: float):
        # Ensure objectives is a list or iterable of objectives
        if not isinstance(objectives, list):
            objectives = [objectives]

        # Convert objectives list to a single string
        objectivesStrList = [
            obj.__name__ if callable(obj) else str(obj) for obj in objectives
        ]
        objectivesStr = '_'.join(objectivesStrList)

        # Write log entry to the file
        with open(self.logPath, 'a', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([objectivesStr, algorithm, datasetSize[0], 
                             datasetSize[1], subsetSize[0], computationTime, 
                             loss])