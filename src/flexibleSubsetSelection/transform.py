# --- Imports and Setup --------------------------------------------------------

# Standard library
from typing import Literal

# Third party
import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from . import logger

# Setup logger
log = logger.setup(name=__name__)


# --- Transform Functions ------------------------------------------------------


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


def discretize(
    data: np.ndarray,
    bins: int | ArrayLike,
    indices: list,
    strategy: Literal["uniform", "quantile", "kmeans"] = "uniform",
) -> np.ndarray:
    """
    Returns the discretized data at indices according to the specified strategy.

    Args:
        bins: Number of bins to use, bins in each feature, or bin edges.
        indices: The indices of the features to use for the binning.
        strategy: sklearn KBinsDiscretizer strategy to use.
    """
    selected = data[:, indices]
    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    return discretizer.fit_transform(selected)


def encode(data: np.ndarray, indices: list[int], dimensions: int = 1) -> np.ndarray:
    """
    Returns one hot encoding of the data at indices, assuming they are discrete.

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
