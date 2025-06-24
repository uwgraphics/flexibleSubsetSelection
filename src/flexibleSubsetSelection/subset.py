# --- Imports and Setup --------------------------------------------------------

# Standard library
from pathlib import Path

# Third party
import numpy as np
import pandas as pd
import pickle

# Local files
from . import logger
from .dataset import Dataset

# Setup logger
log = logger.setup(name=__name__)


# --- Subset Class -------------------------------------------------------------


class Subset:
    """
    A class for creating, storing, and handling subsets of datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        z: np.ndarray,
        name: str = None,
        solveTime: (float | None) = None,
        loss: (float | None) = None,
        selectBy: str = "row",
    ) -> None:
        """
        Initialize a subset with a Dataset object and the indicator vector z.

        Args:
            dataset: The dataset from which to take the subset.
            z: The indicator vector indicating which samples from the dataset
                are included in the subset.
            solveTime: The computation time to solve for the subset in seconds.
            loss: The calculated loss of the subset.
            selectBy: The mode of selection to use for selecting the subset

        Raises:
            ValueError: If length of z does not match the length of dataset.
        """
        self.dataset = dataset
        self.z = z.astype(bool)
        self.selectBy = selectBy
        self.name = name or f"{dataset.name}Subset"

        if len(z) != dataset.size[0]:
            raise ValueError("Length of z must match the length of dataset.")

        length = int(np.sum(z))
        self.size = (length, dataset.size[1])

        self.solveTime = solveTime
        self.loss = loss
        log.info("Created %s.", self)

    @property
    def transforms(self) -> list[str]:
        """
        Expose available dataset transformations (mirrored from dataset).
        """
        return self.dataset.transforms

    @classmethod
    def load(
        cls,
        fileName: str,
        dataset: Dataset,
        fileType: str = "pickle",
        directory: (str | Path) = "../data",
    ) -> "Subset":
        """
        Class method to load a saved subset from file.

        Args:
            fileName: The filename (without extension).
            dataset: The Dataset object the subset belongs to.
            fileType: 'pickle' or 'csv'.
            directory: Path to the folder.

        Returns:
            A new Subset instance.
        """
        filePath = Path(directory) / f"{fileName}.{fileType}"

        try:
            if fileType == "pickle":
                with open(filePath, "rb") as f:
                    return pickle.load(f)
            elif fileType == "csv":
                z = pd.read_csv(filePath).squeeze().to_numpy()
                return cls(dataset=dataset, z=z)
            else:
                raise ValueError(f"Unsupported file type: {fileType}.")
        except Exception as e:
            errorMessage = "Error loading subset from '%s'" % filePath
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

    def save(
        self,
        fileType: str = "pickle",
        directory: (str | Path) = "../data",
        name: str = None,
    ) -> None:
        """
        Save the subset to pickle for convenience or to csv for portability.

        Args:
            fileType: The file format to save the data: 'pickle' or 'csv'.
            directory: Directory to save the file in.

        Raises:
            ValueError: If an unsupported file type is specified.
        """
        if name is None:
            name = self.name
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filePath = path / f"{name}.{fileType}"

        try:
            if fileType == "pickle":
                with open(filePath, "wb") as f:
                    pickle.dump(self, f)
            elif fileType == "csv":
                pd.DataFrame(
                    data = self.original, 
                    columns = self.dataset.features
                ).to_csv(filePath, index=False)
            else:
                raise ValueError(f"Unsupported file type: {fileType}")
            log.info(f"Subset saved to '{filePath}'.")
        except Exception as e:
            errorMessage = "Error saving subset to '%s'" % filePath
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Subset object.
        """
        parts = [f"Subset(name={self.name}, size={self.size})"]
        if self.solveTime is not None:
            parts.append(f"time={self.solveTime:.4f}s")
        if self.loss is not None:
            parts.append(f"loss={self.loss:.4f}")
        return ", ".join(parts)

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Subset object.
        """
        if len(self.size) == 1:
            size = str(self.size[0])
        else:
            size = f"{self.size[0]}x{self.size[1]}"

        string = f"Subset of {self.name} of size {size}"
        if self.solveTime is not None:
            string += f" in {round(self.solveTime, 2)}s "
        if self.loss is not None:
            string += f"with {round(self.loss, 2)} loss"
        return string

    def __getattr__(self, attr: str) -> np.ndarray:
        """
        Returns the specified transformed view of the dataset, subsetted by z.
        """
        if hasattr(self.dataset, attr):
            data = getattr(self.dataset, attr)
            return self.select(data, self.z, self.selectBy)
        raise AttributeError(f"'Subset' object has no attribute '{attr}'")

    def __len__(self) -> int:
        return self.size[0]

    @staticmethod
    def select(array: np.ndarray, z: np.ndarray, selectBy: str) -> np.ndarray:
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
