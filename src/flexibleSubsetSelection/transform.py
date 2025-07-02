# --- Imports and Setup --------------------------------------------------------

# Standard library
from typing import Literal, Callable

# Third party
import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# Local files
from . import logger

# Setup logger
log = logger.setup(name=__name__)


# --- Data Transform Pipeline --------------------------------------------------


class Transforms:
    """
    A class for creating, storing, and evaluating data transforms on a Dataset.
    """

    def __init__(self, array: Callable):
        """
        Initialize the transforms class with a pipeline, cache and functions.

        Args:
            array: The base array property method of the corresponding Dataset
        """
        self._array = array
        self._pipeline = {"original": {"func": None, "params": {}}}
        self._cache = {}

    @property
    def cached(self) -> list:
        """
        Return the names of the cached transforms.
        """
        return list(self._cache.keys())

    @property
    def queued(self) -> list:
        """
        Return the names of the queued transforms.
        """
        return list(self._pipeline.keys())

    @staticmethod
    def scale(
        data: np.ndarray, interval: tuple[float, float], indices: list[int]
    ) -> np.ndarray:
        """
        Returns data at indices scaled to the specified interval.

        Args:
            interval: The interval to scale the data to.
            indices: The indices of the features to use for the binning.
        """
        selected = data[:, indices]
        minVals = selected.min(axis=0)
        maxVals = selected.max(axis=0)
        rangeVals = np.where(maxVals - minVals == 0, 1, maxVals - minVals)
        scaled = (selected - minVals) / rangeVals
        scaled = scaled * (interval[1] - interval[0]) + interval[0]

        result = data.copy()
        result[:, indices] = scaled
        return result

    @staticmethod
    def discretize(
        data: np.ndarray,
        bins: int | ArrayLike,
        indices: list,
        strategy: Literal["uniform", "quantile", "kmeans"] = "uniform",
    ) -> np.ndarray:
        """
        Returns discretized data at indices according to the specified strategy.

        Args:
            bins: Number of bins to use, bins in each feature, or bin edges.
            indices: The indices of the features to use for the binning.
            strategy: sklearn KBinsDiscretizer strategy to use.
        """
        selected = data[:, indices]
        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
        return discretizer.fit_transform(selected)

    @staticmethod
    def encode(
        data: np.ndarray, 
        indices: list[int], 
        dimensions: int = 1
    ) -> np.ndarray:
        """
        Returns one hot encoding of discrete data at specified indices.

        Args:
            indices: The indices of the features to use for the binning.
            dimensions: The number of dimensions to take the encoding in
        """
        selected = data[:, indices]
        if dimensions > 1:
            dims = [int(data[:, i].max()) + 1 for i in indices]
            selected = np.ravel_multi_index(selected.T.astype(int), dims=dims)
            selected = selected.reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(selected)

        mask = np.ones(data.shape[1], dtype=bool)
        mask[indices] = False
        return np.hstack((data[:, mask], encoded))

    def __setitem__(self, key: str, value: tuple) -> None:
        """
        Add a transform to the pipeline by specifying name, func, and params.

        Args:
            key: The name of the transform to add
            value: The transform function and any parameters of the transform.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("Value must be a tuple of (function, params)")

        func, params = value
        self._pipeline[key] = {"func": func, "params": params}
        log.info("Queued '%s' with parameters '%s'.", key, params)

    def __getitem__(self, name: str) -> np.ndarray:
        """
        Retrieve a transform from the cache or evaluate and cache

        Args:
            name: The name of the transform to get
        """
        if name in self._cache:
            return self._cache[name]
        if name not in self._pipeline:
            raise KeyError(f"No transform named '{name}' in the pipeline.")

        names = list(self._pipeline.keys())
        targetIndex = names.index(name)
        index = -1
        array = None
        for i in range(targetIndex, -1, -1):
            currentName = names[i]
            if currentName in self._cache:
                array = self._cache[currentName]
                index = i
                break
        if index == -1:
            array = self._array()
            self._cache["original"] = array
            index = 0

        for i in range(index + 1, targetIndex + 1):
            currentName = names[i]
            transform = self._pipeline[currentName]
            array = transform["func"](array, **transform["params"])

        self._cache[name] = array
        return array

    def __len__(self):
        """Returns the number of steps in the pipeline."""
        return len(self._pipeline)

    def __iter__(self):
        """Iterates over the names of the transforms in order."""
        return iter(self._pipeline.keys())
