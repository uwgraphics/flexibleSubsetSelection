# --- Imports and Setup --------------------------------------------------------

# Standard library
from pathlib import Path
from typing import Literal

# Third party
import ibis
from ibis.expr.types import Table

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

# Local files
from . import generate
from . import logger
from . import transform

# Setup logger
log = logger.setup(name=__name__)


# --- Dataset Class ------------------------------------------------------------

class Dataset:
    """
    A class for creating, storing, and processing datasets for subset selection.
    """

    def __init__(self, 
        name: str,
        data: pd.DataFrame | np.ndarray | str | Table | None = None,
        randTypes: str | list[str] | None = None, 
        size: tuple[int, int] | None = None, 
        interval: tuple[float, float] = (1.0, 5.0), 
        features: list[str] | None = None, 
        seed: int | np.random.Generator | None = None,  
        backend: str = "duckdb"
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
        self._transforms = [{"name": "original"}]
        self._metrics = []

        # Create Ibis connection with backend
        if isinstance(data, Table): # Data specified as an Ibis table
            self._table = data
            self._conn = data._client
        else:
            self._connect(backend)

        # Load data if specified as a DataFrame, ndarray, or string
        if data is not None: # initialize with provided data
            if isinstance(data, pd.DataFrame):
                if features is not None:
                    missingFeatures = set(features) - set(data.columns)
                    if missingFeatures:
                        raise ValueError(f"Features not found in data: " 
                                         f"{missingFeatures}")
                    data = data[features]
                self._table = self._conn.create_table(name, data)
                self._array = data.to_numpy()
                self._features = list(data.columns)
                self._size = data.shape
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
                if features is not None:
                    df.columns = features
                self._table = self._conn.create_table(name, df)
                self._array = data
                self._size = df.shape
            elif isinstance(data, (str, Path)):
                if not Path(data).exists():
                    raise ValueError(f"File '{data}' does not exist.")
                try:
                    self._table = self._conn.read_uri(data)
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
                [generate.random(i, size, interval, seed) for i in randTypes], 
                axis=1
            )

            if features is not None:
                if len(features) != df.shape[1]:
                    raise ValueError("Number of features != number of columns.")
                df.columns = features

            self._table = self._conn.create_table(name, df)
            self._array = df.to_numpy()
            self._size = size
            self._features = list(df.columns)

        log.info("Dataset '%s' created with backend '%s'.", name, backend)

    @property
    def size(self) -> tuple[int, int]:
        """
        Lazy evaluation of the size of the dataset when required
        """
        if not hasattr(self, "_size"):
            rows = self._conn.execute(self._table.count()).scalar()
            cols = len(self._table.schema().names)
            self._size = (rows, cols)
        return self._size

    @property
    def array(self) -> np.ndarray:
        """
        Lazy materialization of the data array from the Ibis table when required
        """
        if not hasattr(self, "_array") or self._array is None:
            log.info("Materializing the dataset array.")
            self._array = self._transform(self._table.to_pandas().to_numpy())
        return self._array
    
    @property
    def features(self) -> list[str]:
        """
        Lazy evaluation of feature columns in dataset when required
        """
        if not hasattr(self, "_features"):
            self._features = list(self._table.schema().names)
        return self._features

    @property
    def transforms(self) -> list[str]:
        """
        Publicly expose the list of available transformation stages.
        
        Returns:
            A list of transform names, e.g., ['original', 'scale', 'discretize'].
        """
        return [t["name"] for t in self._transforms]
    
    @property
    def metrics(self) -> list[str]:
        """
        Publicly expose the list of computed and cached metrics
        
        Returns:
            A list of metric names, e.g., [].
        """
        return [t["name"] for t in self._metrics]
    
    def compute(self, **parameters) -> None:
        """
        Perform custom preprocessing of a preprocess function on the dataset
        and assign it to the specified name.

        Args:
            parameters: Keyword arguments where the key is the name of the 
                preprocessing function and the value is either the function (for
                functions that don't require parameters), or a tuple where the 
                first element is the function and the second element is a 
                dictionary of additional parameters.
        """
        for name, function in parameters.items():
            try:
                if isinstance(function, tuple): # with parameters
                    func, params = function
                    setattr(self, name, func(self.array, **params))
                else:
                    setattr(self, name, function(self.array))
                self._metrics.append({"name": name, 
                                      "func": function, 
                                      "params": parameters})
                log.info(f"Data preprocessed with function '%s'.", name)
            except Exception as e:
                log.exception(f"Error applying function '%s'.", name)

    def scale(self, 
        interval: tuple | None = None, 
        features: list | None = None
    ) -> None:
        """
        Modifies the specified features of the dataset by scaling them to the 
        specified interval.

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
            log.exception("Feature not found in scale.")
            raise

        self._transforms.append({
            "name": "scaled",
            "func": transform.scale,
            "params": {"interval": interval, "indices": indices},
        })
        self._array = None
        log.info("Queued scale for %s to interval %s.", features, interval)

    def discretize(self, 
        bins: int | ArrayLike, 
        features: list | None = None, 
        strategy: Literal["uniform", "quantile", "kmeans"] = "uniform"
    ) -> None:
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
            log.exception("Feature not found in discretize.")
            raise
        
        self._transforms.append({
            "name": "discretized",
            "func": transform.discretize,
            "params": {"bins": bins, "indices": indices, "strategy": strategy},
        })
        self._array = None
        log.info("Dataset discretized by %s with %s bins.", strategy, bins)

    def encode(self, 
        features: list | None = None, 
        dimensions: int = 1
    ) -> None:
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
            log.exception("Feature not found in encode.")
            raise
        
        self._transforms.append({
            "name": "encoded",
            "func": transform.encode,
            "params": {"indices": indices, "dimensions": dimensions},
        })
        self._array = None
        log.info("Data one-hot encoded with %d dimensions.", dimensions)

    def save(self, 
        fileType: str = "csv", 
        directory: str | Path | None = None
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

        log.info(f"Data successfully saved at '%s'.", filePath)

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

    def _transform(self, array: np.ndarray, name: str = None) -> np.ndarray:
        """
        Evaluate transformations up to and including the specified transform.

        Args: 
            name: The name of the transform to evaluate to if specified. If 
                None, the full transformation pipeline is evaluated.
        """
        if name == "original" or len(self._transforms) == 1:
            return array
        for transform in self._transforms[1:]:
            try:
                array = transform["func"](array, **transform["params"])
            except Exception as e:
                log.exception(f"Failed to apply transform {transform['name']}")
                raise
            if name and transform["name"] == name:
                break
        return array

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Dataset object.
        """
        return (f"Dataset(name='{self.name}', size={self.size}, "
                f"features={self.features})")

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Dataset object.
        """
        return (f"Dataset {self.name}: {self.size[0]} rows x "
                f"{self.size[1]} features "
                f"{self.features[:3]}{'...' if len(self.features) > 3 else ''}")
    
    def __len__(self) -> int:
        """
        Returns the number of rows in the dataset
        """
        return self.size[0]
    
    def __getattr__(self, attr):
        """
        Returns the specified transformed version of the dataset if specified.

        Args:
            attr: Specify the name of a transform function
        """
        transforms = [t["name"] for t in self._transforms]
        if attr in transforms:
            array = self._table.to_pandas().to_numpy()
            return self._transform(array, name=attr)
        raise AttributeError(f"'Dataset' object has no attribute '{attr}'")