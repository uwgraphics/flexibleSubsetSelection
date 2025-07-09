# --- Imports and Setup --------------------------------------------------------

# Standard library
from typing import Any, Callable, Dict, List

# Third party
import numpy as np
from numpy.typing import ArrayLike

# Local files
from . import logger
from .dataset import Dataset
from .subset import Subset

# Setup logger
log = logger.setup(__name__)


# --- Loss Function ------------------------------------------------------------


class MultiCriterion:
    """
    Create and apply multi-criterion loss functions from a set of objectives and
    corresponding weights for subset selection.
    """

    def __init__(
        self,
        objectives: List[Callable],
        parameters: List[Dict[str, Any]],
        weights: np.ndarray | None = None,
    ) -> None:
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

        log.info("Initialized multi-criterion loss function: %s", self)


    def __call__(self, dataset: Dataset, z: ArrayLike) -> float:
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
            # retrieve solve array from attributes or default to array
            array = getattr(dataset, params.get("solveArray", "original"))

            # retrieve selectBy from attributes or default to row
            selectBy = params.get("selectBy", "row")
            subset = Subset.select(array, z, selectBy=selectBy)

            # retrieve any remaining parameters as objective parameters
            objectiveParameters = {
                key: value
                for key, value in params.items()
                if key not in ["solveArray", "selectBy"]
            }
            objectiveLoss = weight * objective(subset, **objectiveParameters)
            loss += objectiveLoss
        return loss

    def __repr__(self) -> str:
        """
        A detailed string representation of the loss function.
        """
        objectives_str = [obj.__name__ for obj in self.objectives]
        return (
            f"MultiCriterion(objectives={objectives_str}, "
            f"parameters={self.parameters}, weights={self.weights.tolist()})"
        )

    def __str__(self) -> str:
        """
        A user-friendly string representation of the loss function.
        """
        parts = []
        objs = zip(self.objectives, self.parameters, self.weights)
        for objective, params, weight in objs:
            subparts = []

            if params.get("solveArray", "original") != "original":
                subparts.append(f"{params['solveArray']} array")
            if params.get("selectBy", "row") != "row":
                subparts.append(f"select by {params['selectBy']}")

            for k, v in params.items():
                if k in ["solveArray", "selectBy"]:
                    continue
                if isinstance(v, np.ndarray):
                    continue
                if callable(v):
                    subparts.append(f"{v.__name__} {k}")
                elif isinstance(v, str):
                    subparts.append(f"{v} {k}")
                else:
                    subparts.append(f"{k}={v}")

            joined = ", ".join(subparts)
            parts.append(f"{weight}*{objective.__name__}({joined})")

        return " + ".join(parts)


class UniCriterion:
    """
    Create and apply a unicriterion loss function from an objective, apply to a
    particular data array for subset selection.
    """

    def __init__(
        self,
        objective: Callable,
        solveArray: str = "original",
        selectBy: str = "row",
        **parameters: Any,
    ) -> None:
        """
        Define a loss function with an objective and optional parameters for
        subset selection.

        Args:
            objective: The objective function to define the loss.
            solveArray: The name of the array in dataset to use
                for subset selection. Default is "array".
            selectBy: The method to select subset from array.
            **parameters: Additional parameters of the objective function.
        """
        self.objective = objective
        self.solveArray = solveArray
        self.selectBy = selectBy
        self.parameters = parameters

        log.info("Initialized uni-criterion loss function: %s", self)

    def __call__(self, dataset: Dataset, z: ArrayLike) -> float:
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
        subset = Subset.select(array, z, selectBy=self.selectBy)
        return self.objective(subset, **self.parameters)
    
    def __repr__(self) -> str:
        """
        A detailed string representation of the loss function.
        """
        return (
            f"UniCriterion(objective={self.objective.__name__}, "
            f"solveArray={self.solveArray}, selectBy={self.selectBy}, "
            f"parameters={self.parameters})"
        )

    def __str__(self) -> str:
        """
        A user-friendly string representation of the loss function.
        """
        parts = []

        if self.solveArray != "original":
            parts.append(f"{self.solveArray} array")
        if self.selectBy != "row":
            parts.append(f"select by {self.selectBy}")

        for k, v in self.parameters.items():
            if isinstance(v, np.ndarray):
                continue 
            if callable(v):
                parts.append(f"{v.__name__} {k}")
            elif isinstance(v, str):
                parts.append(f"{v} {k}")
            else:
                parts.append(f"{k}={v}")

        return f"{self.objective.__name__}({', '.join(parts)})"