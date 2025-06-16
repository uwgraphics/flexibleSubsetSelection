# --- Imports and Setup --------------------------------------------------------

# Standard library
from pathlib import Path
from typing import Any, Literal, Self

# Third party
import ibis
from ibis.expr.types import Table

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

# Local files
from . import generate, logger
from .transform import Transforms

# Setup logger
log = logger.setup(name=__name__)


# --- Dataset Class ------------------------------------------------------------


class Dataset:
    """
    A class for creating, storing, and processing datasets for subset selection.
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame | np.ndarray | str | Path | Table | None = None,
        randTypes: str | list[str] | None = None,
        size: tuple[int, int] | None = None,
        interval: tuple[float, float] = (1.0, 5.0),
        features: list[str] | None = None,
        seed: int | np.random.Generator | None = None,
        backend: str = "duckdb",
    ) -> None:
        """
        Initialize a dataset by providing data or by random data generation.

        Args:
            name: The name of the dataset and the Ibis table storing it
            data: Specify the dataset with a dataframe, array, filepath, or Ibis
                table, or use random data generation.
            randTypes: The method(s) for random data generation. Supported
                methods: "uniform", "binary", "categories", "normal",
                "multimodal", "skew", "blobs".
            size: The size of the dataset to generate (rows, columns).
            interval: The interval range for generating or scaling data.
            features: The list of column features to assign to the dataset.
            seed: The random seed or NumPy generator for reproducibility.
            backend: The Ibis backend for execution. (e.g., "duckdb", "pandas",
                "polars", "pyspark", etc.).

        Raises:
            ValueError: If neither data, nor a random generation method and size
            of data are specified.
        """
        self.name = name
        self.backend = backend
        self.interval = interval
        self._metrics = []

        # Create Ibis connection with backend
        if isinstance(data, Table):  # Data specified as an Ibis table
            self._table = data
            self._conn = data._client
        else:
            self._connect(backend)

        # Load data if specified as a DataFrame, ndarray, or string
        if data is not None:  # initialize with provided data
            if isinstance(data, pd.DataFrame):
                if features is not None:
                    missingFeatures = set(features) - set(data.columns)
                    if missingFeatures:
                        raise ValueError(
                            f"Features not found in data: {missingFeatures}"
                        )
                    data = data[features]
                self._table = self._conn.create_table(name, data)
                self._features = list(data.columns)
                self._size = data.shape
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
                if features is not None:
                    df.columns = features
                self._table = self._conn.create_table(name, df)
                self._size = df.shape
            elif isinstance(data, (str, Path)):
                if not Path(data).exists():
                    raise ValueError(f"File '{data}' does not exist.")
                try:
                    self._table = self._conn.read_csv(data)
                except Exception as e:
                    raise ValueError(f"Failed to read '{data}': {e}.")

        # Initialize dataset with random data generation
        else:
            if randTypes is None:
                raise ValueError("No data or random generation type specified.")
            elif isinstance(randTypes, str):
                randTypes = [randTypes]
            if size is None:
                raise ValueError("No size of data to generate specified.")

            df = pd.concat(
                [generate.random(i, size, interval, seed) for i in randTypes], axis=1
            )

            if features is not None:
                if len(features) != df.shape[1]:
                    raise ValueError("Number of features != number of columns.")
                df.columns = features

            self._table = self._conn.create_table(name, df)
            self._size = size
            self._features = list(df.columns)

        self._transforms = Transforms(self._array)
        log.info("Dataset '%s' created with backend '%s'.", name, backend)

    @property
    def rows(self) -> int:
        """
        Lazy evaluation of the number of rows of the dataset when required
        """
        if hasattr(self, "_rows"):
            return self._rows
        try:
            self._rows = self._conn.execute(self._table.count())
            return self._rows
        except Exception as e:
            errorMessage = "Error evaluating the number of rows."
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

    @property
    def cols(self) -> int:
        """
        Lazy evaluation of the number of columns of the dataset when required
        """
        if hasattr(self, "_cols"):
            return self._cols
        try:
            self._cols = len(self._table.schema().names)
            return self._cols
        except Exception as e:
            errorMessage = "Error evaluating the number of columns."
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

    @property
    def size(self) -> tuple[int, int]:
        """
        Lazy evaluation of the size of the dataset when required
        """
        if not hasattr(self, "_size"):
            self._size = (self.rows, self.cols)
        return self._size

    @property
    def features(self) -> list[str]:
        """
        Lazy evaluation of feature columns in dataset when required
        """
        if hasattr(self, "_features"):
            return self._features
        try:
            self._features = list(self._table.schema().names)
            return self._features
        except Exception as e:
            errorMessage = "Error evaluating the features in the dataset."
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

    def _array(self) -> np.ndarray:
        """
        Materialization of the data array from the Ibis table
        """
        return self._table.to_pandas().to_numpy()

    @property
    def metrics(self) -> list[str]:
        """
        Publicly expose the list of computed and cached metrics.

        Returns:
            A list of metric names.
        """
        return [t["name"] for t in self._metrics]

    def compute(
        self, array: str = None, features: list[str] | None = None, **metric: Any
    ) -> None:
        """
        Compute and cache a named metric on the dataset.

        Args:
            array: The array to compute the metric on
            features: The features to compute the metric on
            metric: Keyword arguments where each key is the name to assign the
            result, and each value is either:
                - a function (applied to the dataset array), or
                - a tuple of (function, parameter dictionary).

        Raises:
            RuntimeError: If any metric function fails.
        """
        array = getattr(self, array) if array else self.original

        if features is not None:
            try:
                indices = [self.features.index(f) for f in features]
                array = array[:, indices]
            except ValueError as e:
                errorMessage = "Feature not found in 'compute()'."
                log.exception(errorMessage)
                raise RuntimeError(errorMessage) from e
        for name, function in metric.items():
            try:
                if isinstance(function, tuple):  # with parameters
                    func, params = function
                    setattr(self, name, func(array, **params))
                else:
                    func = function
                    params = {}
                    setattr(self, name, function(array))
                self._metrics.append({"name": name, "func": func, "params": params})
                log.info("Data preprocessed with function '%s'.", name)
            except Exception as e:
                errorMessage = "Error applying function '%s'." % name
                log.exception(errorMessage)
                raise RuntimeError(errorMessage) from e

    def scale(
        self,
        interval: tuple[float, float] | None = None,
        features: list[str] | None = None,
    ) -> Self:
        """
        Scale dataset features to a specified interval.

        Args:
            interval: The interval to scale the data to.
            features: The features to scale. All features will be scaled if
                unspecified.

        Raises:
            ValueError: if features are not found
        """
        if interval is None:
            interval = self.interval
        if features is None:
            features = self.features
        try:
            indices = [self.features.index(f) for f in features]
        except ValueError as e:
            errorMessage = "Feature not found in 'scale()'."
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

        self._transforms["scaled"] = (
            Transforms.scale,
            {"interval": interval, "indices": indices},
        )
        return self

    def discretize(
        self,
        bins: int | ArrayLike,
        features: list[str] | None = None,
        strategy: Literal["uniform", "quantile", "kmeans"] = "uniform",
    ) -> Self:
        """
        Modifies the specified features of the dataset by discretizing them into
        bins according to the given strategy.

        Args:
            bins: Number of bins to use, bins in each feature, or bin edges.
            features: List of features to use for the binning. All features will
                be encoded if unspecified.
            strategy: sklearn KBinsDiscretizer strategy to use.

        Raises:
            ValueError: if dimensions or features do not match dimensionality
        """
        if features is None:
            features = self.features
        try:
            indices = [self.features.index(f) for f in features]
        except ValueError as e:
            errorMessage = "Feature not found in 'discretize()'."
            log.exception(errorMessage)
            raise RuntimeError(errorMessage) from e

        self._transforms["discretized"] = (
            Transforms.discretize,
            {"bins": bins, "indices": indices, "strategy": strategy},
        )
        return self

    def encode(self, features: list[str] | None = None, dimensions: int = 1) -> Self:
        """
        Modifies the specified features of the dataset by one hot encoding them,
        assuming they are discrete.

        Args:
            features: The features to use for the binning. All features will be
                encoded if unspecified.
            dimensions: The number of dimensions to take the encoding in

        Raises:
            ValueError: if features are not found
        """
        if features is None:
            features = self.features
        try:
            indices = [self.features.index(f) for f in features]
        except ValueError as e:
            errorMessage = "Feature not found in 'encode()'."
            log.exception(errorMessage)
            raise RuntimeWarning(errorMessage) from e

        self._transforms["encoded"] = (
            Transforms.encode,
            {"indices": indices, "dimensions": dimensions},
        )
        return self

    def save(
        self,
        fileType: Literal["csv", "parquet"] = "csv",
        directory: str | Path | None = None,
    ) -> None:
        """
        Saves the dataset eagerly to disk.

        Args:
            fileType: The file format to save the data: csv or parquet
            directory: The target directory in which to save the file.

        Raises:
            ValueError: If an unsupported file type is specified.
        """
        if directory is None:
            path = Path(__file__).resolve().parent.parent / "data"
        else:
            path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filePath = path / f"{self.name}.{fileType}"

        try:
            if fileType == "csv":
                self._table.to_csv(filePath)
            elif fileType == "parquet":
                self._table.to_parquet(filePath)
            else:
                raise ValueError(f"Unsupported file type: {fileType}.")
        except Exception as e:
            log.exception("Error saving file to '%s': %s", filePath, e)
            raise

        log.info("Data successfully saved at '%s'.", filePath)

    def __repr__(self) -> str:
        transformString = (
            f", transforms=original→{'→'.join(self._transforms.queued[1:])}"
            if len(self._transforms) > 1
            else ""
        )
        return (
            f"Dataset(name='{self.name}', size={self.size}, "
            f"features={self.features} {transformString})"
        )

    def __str__(self) -> str:
        transforms = self._transforms.queued[1:]
        if len(transforms) == 0:
            transformString = ""
        elif len(transforms) == 1:
            transformString = f"{transforms[0]}"
        elif len(transforms) == 2:
            transformString = f"{transforms[0]} and {transforms[1]}"
        else:
            transformString = f"{', '.join(transforms[:-1])} and {transforms[-1]}"

        featureString = (
            f"[{', '.join(map(str, self.features[:3]))}"
            f"{', ...' if len(self.features) > 3 else ''}]"
        )
        return (
            f"Dataset {self.name}: {self.size[0]} rows x "
            f"{self.size[1]} features {featureString} {transformString}"
        )

    def __len__(self) -> int:
        """
        Returns the number of rows in the dataset
        """
        return self.size[0]

    def __getattr__(self, attr: str) -> np.ndarray:
        """
        Returns the specified transformed version of the dataset if specified.

        Args:
            attr: Specify the name of a transform function
        """
        if attr in self._transforms.queued:
            return self._transforms[attr]
        else:
            raise AttributeError(f"'Dataset' object has no attribute '{attr}'")

    def _connect(self, backend: str) -> None:
        """
        Connect dataset to specified Ibis backend

        Args:
            backend: The name of the Ibis backend: "duckdb", "pandas", etc.

        Raises:
            ValueError: If the backend is not available or the connection fails
        """
        try:
            connectionFunction = getattr(getattr(ibis, backend), "connect")
        except AttributeError:
            raise ValueError(
                f"Ibis backend '{backend}' is not available. "
                f"Install backends with: pip install 'ibis-framework[backend]'."
            )
        try:
            self._conn = connectionFunction()
        except Exception as e:
            raise ValueError(f"Failed to connect to backend '{backend}': {e}")
