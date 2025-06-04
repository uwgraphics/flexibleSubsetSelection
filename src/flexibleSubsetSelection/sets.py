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
import pickle
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# Local files
from . import generate
from . import logger

# Setup logger
log = logger.setup(name=__name__)


# --- Dataset and Subset Classes -----------------------------------------------

class Set:
    """
    Base class for Dataset and Subset providing shared save and load functions.
    """
    
    def save(self, 
        fileType: str = "csv", 
        directory: (str | Path) = "../data"
    ) -> None:
        """
        Saves the dataset eagerly to disk.

        Args:
            fileType: The file format to save the data: csv, parquet, or pickle
            directory: The target directory in which to save the file.
        
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filePath = path / f"{self.name}.{fileType}"

        try:
            if fileType == "pickle":
                with open(filePath, "wb") as f:
                    pickle.dump(self, f)
                log.info(f"Data successfully saved at '%s'.", filePath)
            elif fileType == "csv":
                self.table.to_csv(filePath)
                log.info(f"Data successfully saved at '%s'.", filePath)
            elif fileType == "parquet":
                self.table.to_parquet(filePath)
                log.info(f"Data successfully saved at '%s'.", filePath)
            else:
                raise ValueError(f"Unsupported file type: {fileType}.")
        except Exception as e:
            log.exception("Error saving file to '%s': %s", filePath, e)
            raise

    def load(self, name: str, fileType: str = "pickle",  
             directory: (str | Path) = "../data") -> None:
        """
        Loads from a file into self.data.

        Args:
            name: The name of the file.
            fileType: The type of file (pickle or csv).
            directory: Directory to load the file from.

        Raises:
            ValueError: If an unsupported file type is specified.
        """
        path = Path(directory)
        filePath = path / f"{name}.{fileType}"

        try:
            if fileType == "pickle":
                with open(filePath, "rb") as f:
                    self.data = pickle.load(f)
                log.info(f"Data successfully loaded from '%s'.", filePath)
            elif fileType == "csv":
                self.data = pd.read_csv(filePath)
                log.info(f"Data successfully loaded from '%s'.", filePath)
            else:
                raise ValueError(f"Unsupported file type: {fileType}.")
        except Exception as e:
            log.exception("Error loading file")

class Dataset(Set):
    """
    A class for creating, storing, and processing datasets for subset selection.
    """

    def __init__(self, 
        name: str,
        data: pd.DataFrame | np.ndarray | str | Table | None = None,
        randTypes: (str | list[str] | None) = None, 
        size: (tuple[int, int] | None) = None, 
        interval: tuple[float, float] = (1.0, 5.0), 
        features: (list[str] | None) = None, 
        seed: (int | np.random.Generator | None) = None,  
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

        # Create Ibis connection with backend
        if isinstance(data, Table): # Data specified as an Ibis table
            self.table = data
            self.conn = data._client
        else:
            self._connect(backend)   

        # Load data if specified as a DataFrame, ndarray, or string
        if data is not None: # initialize with provided data
            if isinstance(data, (pd.DataFrame, np.ndarray)):
                df = pd.DataFrame(data)
                self.table = self.conn.create_table(name, df)
            elif isinstance(data, (str, Path)):
                if not Path(data).exists():
                    raise ValueError(f"File '{data}' does not exist.")
                try:
                    self.table = self.conn.read_uri(data)
                except Exception as e:
                    raise ValueError(f"Failed to read '{data}': {e}.")
                
        # Initialize dataset with random data generation
        else:
            if randTypes is None: 
                raise ValueError("No data or random generation method specified.")

            if size is None:
                raise ValueError("No size of data to generate specified.")
            
            df = (pd.concat(
                [generate.random(i, size, interval, seed) for i in randTypes],
                axis=1
            ) if isinstance(randTypes, list) else
                generate.random(randTypes, size, interval, seed)
            )
            if features is not None:
                if len(features) != df.shape[1]:
                    raise ValueError("Number of features != number of columns.")
                df.columns = features
            self.table = self.conn.create_table(name, df)

        log.info("Dataset '%s' created with backend '%s'.", name, backend)

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
            self.conn = connectionFunction()
        except Exception as e:
            raise ValueError(f"Failed to connect to backend '{backend}': {e}")

    def preprocess(self, **parameters) -> None:
        """
        Perform custom preprocessing of a preprocess function on self.dataArray
        and assign it to the specified name.

        Args:
            parameters: Keyword arguments where the key is the name of the 
                preprocessing function and the value is either the function (for
                functions that don't require parameters), or a tuple where the 
                first element is the function and the second element is a 
                dictionary of additional parameters.
        """
        for name, preprocessor in parameters.items():
            try:
                if isinstance(preprocessor, tuple): # with parameters
                    func, params = preprocessor
                    setattr(self, name, func(self.dataArray, **params))
                else:
                    setattr(self, name, preprocessor(self.dataArray))
                log.info(f"Data preprocessed with function '%s'.", name)
            except Exception as e:
                log.exception("Error applying function '%s'.", name)

    def scale(self, interval: (tuple | None) = None, 
        features: (list | None) = None) -> None:
        """
        Scales self.dataArray numpy array based on self.interval tuple

        Args:
            interval: The interval to scale the data to. The class attribute is 
            used if none is specified.
            features: List of features to scale. Defaults to self.features.
        """
        if features is None:
            features = self.features
        if interval is None:
            interval = self.interval
        
        # Gets specified features
        try:
            indices = [self.indices[feature] for feature in features]
        except KeyError as e:
            log.exception("Feature not found in indices.")
        
        selected = self.dataArray[:, indices]
        minVals = selected.min(axis=0)
        maxVals = selected.max(axis=0)
        
        # Avoid division by zero
        rangeVals = maxVals - minVals
        rangeVals[rangeVals == 0] = 1

        # Update self.data to reflect scaled values
        scaled = (selected - minVals) / rangeVals
        scaled = scaled * (interval[1] - interval[0]) + interval[0]
        self.dataArray[:, indices] = scaled
        self.data.loc[:, features] = self.dataArray[:, indices]

        log.info("Features %s scaled to %s.", features, interval)
        
    def discretize(self, bins: (int | ArrayLike), 
                   features: (list | None) = None, 
                   strategy: Literal["uniform","quantile","kmeans"] = "uniform", 
                   array: (str | None) = None) -> None:
        """
        Discretize self.dataArray into bins.

        Arg: 
            bins: Number of bins to use, bins in each feature, or bin edges.
            features: List of features to use for the binning
            strategy: sklearn KBinsDiscretizer strategy to use.
            array: The array to assignt he result to.
        
        Raises:
            ValueError: if dimensions or features do not match dimensionality
        """
        if features is None:
            features = self.features
        if array is None:
            array = "dataArray"

        # Gets specified features
        try:
            indices = [self.indices[feature] for feature in features]
        except KeyError as e:
            log.exception("Feature not found in indices.")
        
        selected = self.dataArray[:, indices]
        discretizer = KBinsDiscretizer(n_bins = bins, 
                                       encode = "ordinal", 
                                       strategy = strategy)

        setattr(self, array, discretizer.fit_transform(selected))
        self.bins = bins
        log.info("%s discretized by %s with %s bins.", array, strategy, bins)
        
    def encode(self, features: (list | None) = None, dimensions: int = 1, 
               array: (str | None) = None) -> None:
        """
        One hot encodes self.dataArray with sklearn OneHotEncoder assuming data 
        is discretized.

        Arg: 
            features: The features to use for the binning
            dimensions: The number of dimensions to take the encoding in
            array: The array to assignt he result to.
        """
        if features is None:
            features = self.features
        if array is None:
            array = "dataArray"

        # Get specified features
        indices = [self.indices[feature] for feature in features]
        selected = self.dataArray[:, indices]

        if dimensions > 1:
            dims = [int(self.dataArray[:, i].max()) + 1 for i in indices]
            selected = np.ravel_multi_index(selected.T.astype(int), dims=dims)
            selected = selected.reshape(-1, 1)

        # Apply OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(selected)
        
        # Remove the original columns and insert the one-hot encoded columns
        mask = np.ones(self.dataArray.shape[1], dtype=bool)
        mask[indices] = False
        setattr(self, array, np.hstack((self.dataArray[:, mask], encoded)))
        log.info("Data one-hot encoded in '%s'", array)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Dataset object.
        """
        return (f"Dataset(size={self.size}, "
                f"features={self.features}, "
                f"interval={self.interval})")

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Dataset object.
        """
        return (f"Dataset with {self.size[0]} rows and {len(self.features)} "
                f"features: {self.features}")

class Subset(Set):
    """
    A class for creating, storing, and handling subsets of datasets.
    """

    def __init__(self, dataset: Dataset, z: ArrayLike, 
                 solveTime: (float | None) = None,
                 loss: (float | None) = None) -> None:
        """
        Initialize a subset with a Dataset object and the indicator vector z.

        Args:
            dataset: The dataset from which to take the subset.
            z: The indicator vector indicating which samples from the dataset 
                are included in the subset.
            solveTime: The computation time to solve for the subset in seconds.
            loss: The calculated loss of the subset.

        Raises:
            ValueError: If length of z does not match the length of dataset.
        """
        if len(z) != dataset.size[0]:
            raise ValueError("Length of z must match the length of dataset.")

        length = int(np.sum(z))
        if dataset.data.ndim == 1:  # one-dimensional dataset
            self.size = (length,)
        else:
            self.size = (length, dataset.size[1])
        
        self.data = dataset.data[z == 1].copy()  # subset of the full data
        self.solveTime = solveTime
        self.loss = loss
        log.info("Created %s.", self)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Subset object.
        """
        string = f"Subset(size={self.size}"
        if self.solveTime is not None:
            string = ", ".join(string, f"time={self.solveTime:.4f}s")
        if self.loss is not None:
            string = ", ".join(string, f"loss={self.loss:.4f})")
        return string

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