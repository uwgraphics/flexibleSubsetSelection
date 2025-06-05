# --- Imports and Setup --------------------------------------------------------

# Standard library
from pathlib import Path
from typing import Literal

# Third party
import numpy as np
import pandas as pd
import pickle

# Local files
from . import logger
from .dataset import Dataset
from .util import select

# Setup logger
log = logger.setup(name=__name__)


# --- Subset Class -------------------------------------------------------------

class Subset:
    """
    A class for creating, storing, and handling subsets of datasets.
    """

    def __init__(self, 
        dataset: Dataset, 
        z: np.ndarray, 
        solveTime: (float | None) = None,
        loss: (float | None) = None,
        selectBy: str = "row"
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
        
        if len(z) != dataset.size[0]:
            raise ValueError("Length of z must match the length of dataset.")

        length = int(np.sum(z))
        if dataset.size[0] == 1:  # one-dimensional dataset
            self.size = (length,)
        else:
            self.size = (length, dataset.size[1])
        
        self.solveTime = solveTime
        self.loss = loss
        log.info("Created %s.", self)
    
    @property
    def array(self) -> np.ndarray:
        """
        Returns the subset of the dataset array.
        """
        if not hasattr(self, "_array"):
            self._array = select(self.dataset.array, self.z, self.selectBy)
        return self._array

    @classmethod
    def load(cls, 
        name: str, 
        dataset: Dataset, 
        fileType: str = "pickle",  
        directory: (str | Path) = "../data"
    ) -> "Subset":
        """
        Class method to load a saved subset from file.

        Args:
            name: The filename (without extension).
            dataset: The Dataset object the subset belongs to.
            fileType: 'pickle' or 'csv'.
            directory: Path to the folder.

        Returns:
            A new Subset instance.
        """
        filePath = Path(directory) / f"{name}.{fileType}"

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
            log.exception(f"Error loading subset from '{filePath}': {e}")
            raise

    def save(self, 
        fileType: str = "pickle", 
        directory: (str | Path) = "../data"
    ) -> None:
        """
        Save the subset data to a file.

        Args:
            fileType: The file format to save the data: 'pickle' or 'csv'.
            directory: Directory to save the file in.
        
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filePath = path / f"{self.dataset.name}.{fileType}"

        try:
            if fileType == "pickle":
                with open(filePath, "wb") as f:
                    pickle.dump(self, f)
            elif fileType == "csv":
                pd.DataFrame(self.z).to_csv(filePath, index=False)
            else:
                raise ValueError(f"Unsupported file type: {fileType}")
            log.info(f"Subset saved to '{filePath}'.")
        except Exception as e:
            log.exception(f"Error saving subset to '{filePath}'.")

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Subset object.
        """
        parts = [f"Subset(size={self.size})"]
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

        string = f"subset of size {size}"
        if self.solveTime is not None:
            string += f" in {round(self.solveTime, 2)}s "
        if self.loss is not None:
            string += f"with {round(self.loss, 2)} loss"
        return string