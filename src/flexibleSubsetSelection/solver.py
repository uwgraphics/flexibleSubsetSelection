# --- Imports ------------------------------------------------------------------

# Standard library
import csv
import logging
from typing import Callable

# Third party
import numpy as np

# Local files
from .loss import UniCriterion, MultiCriterion
from .subset import Subset
from .dataset import Dataset
from .timer import Timer
from . import logger

# Setup logger
log = logger.setup(__name__)


# --- Solver -------------------------------------------------------------------

class Solver():
    """
    A general optimization solver class for subset selection defined by a 
    solving algorithm and loss function, applied to calculate a subset.
    """
    def __init__(self, algorithm: Callable, 
                 lossFunction: (UniCriterion | MultiCriterion | None) = None,
                 savePath: str = "../data/solverData.csv") -> None:
        """
        Initialize a subset selection solver with a solve algorithm and, 
        optionally, a loss function.

        Args:
            algorithm: The algorithm function to find the subset.
            loss: The loss function class object.
            savePath: The path to the solver save file.
        """
        log.debug((f"Initializing Solver with algorithm: %s, "
                  f"lossFunction: %s, savePath: %s"), 
                  algorithm.__name__, lossFunction, savePath)
    
        self.algorithm = algorithm
        self.lossFunction = lossFunction
        self.savePath = savePath

        # Initialize the data file with headers if it doesn't exist
        try:
            with open(self.savePath, 'x', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow(["Loss Function", "Algorithm", "Dataset Length", 
                                 "Dataset Width", "Subset Length", 
                                 "Computation Time", "Loss"])
        except FileExistsError:
            log.debug("Log file already exists at %s", self.savePath)

        log.info("Initialized a '%s' solver.", algorithm.__name__)

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
        
        log.info(f"Selected subset with '%s' and '%s' in %ss with %s loss.", 
                    self.algorithm.__name__,
                    self.lossFunction,
                    np.round(timer.elapsedTime, 2),
                    loss)

        subset = Subset(dataset, z, timer.elapsedTime, loss)

        self.save(dataset.size, subset.size, self.lossFunction, 
                 self.algorithm.__name__, timer.elapsedTime, loss)

        return subset

    def save(self, datasetSize: tuple, subsetSize: tuple, 
            lossFunction: (UniCriterion | MultiCriterion), 
            algorithm: str, computationTime: float, loss: float):

        # Write performance data to the save file
        with open(self.savePath, 'a', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([str(lossFunction), algorithm, datasetSize[0], 
                             datasetSize[1], subsetSize[0], computationTime, 
                             loss])

        log.info(f"Saved solver performance data to %s.", self.savePath)