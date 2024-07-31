# --- Imports ------------------------------------------------------------------

# Standard library
import os

# Third party
import numpy as np
from numpy.typing import ArrayLike

import pandas as pd
import pickle

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# Local files
from . import generate


# --- Dataset and Subset Classes -----------------------------------------------

class Base:
    """
    Base class for Dataset and Subset providing shared save and load functions.
    """
    
    def save(self, name: str, fileType: str = 'pickle', 
             directory: str = '../data', index: bool = False) -> None:
        """
        Saves self.data as a file.

        Args:
            name: The name of the file.
            fileType: The type of file (pickle or csv).
            directory: Directory to save the file in.
            index: Whether to save the data index or not.
        
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        filePath = os.path.join(directory, f"{name}.{fileType}")
        
        try:
            if fileType == "pickle":
                with open(filePath, 'wb') as f:
                    pickle.dump(self.data, f)
            elif fileType == "csv":
                self.data.to_csv(filePath, index=index)
            else:
                raise ValueError(f"Unsupported file type: {fileType}.")
        except Exception as e:
            print(f"Error saving file: {e}")

    def load(self, name: str, fileType: str = 'pickle', 
             directory: str = '../data') -> None:
        """
        Loads from a file into self.data.

        Args:
            name: The name of the file.
            fileType: The type of file (pickle or csv).
            directory: Directory to load the file from.
        
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        filePath = os.path.join(directory, f"{name}.{fileType}")
        
        try:
            if fileType == "pickle":
                with open(filePath, 'rb') as f:
                    self.data = pickle.load(f)
            elif fileType == "csv":
                self.data = pd.read_csv(filePath)
            else:
                raise ValueError(f"Unsupported file type: {fileType}.")
        except Exception as e:
            print(f"Error loading file: {e}")


class Dataset(Base):
    """
    A class for creating, storing, and processing of datasets for subsetting
    """

    def __init__(self, data: ArrayLike = None, randTypes: str | list = None, 
                 size: tuple = None, interval: tuple = (1, 5), 
                 features: list = None, 
                 seed: int | np.random.Generator = None) -> None:
        """
        Initialize a dataset with data or by random data generation.

        Args:
            data: The DataFrame or array of the data. 
            randTypes: The method or methods for random data generation. 
                Supported methods: "uniform", "binary", "categorical", "normal",
                "multimodal", "skew", "blobs"
            size: The size of the dataset to create for random dataset 
                generation or the size of the data (num rows, num columns).
            interval: The interval for scaling data. 
            features: The list of column features to consider.
            seed: The random seed or generator for reproducibility. 

        Raises:
            ValueError: If no data or random generation method is specified.
        """
        # Initialize data
        if data is not None:  # initialize with provided data
            self.size = data.shape
            if isinstance(data, pd.DataFrame):
                self.data = data 
            else:
                self.data = pd.DataFrame(data)
        elif randTypes is not None:  # initialize with random data generation
            if isinstance(randTypes, list):
                self.data = pd.DataFrame({
                    i: generate.randomData(randType, size, interval, seed) 
                    for i, randType in enumerate(randTypes)
                })
            else:
                self.data = generate.randomData(randTypes, size, interval, seed)
            if features is not None:
                self.data.columns = features
            self.size = size
        else:
            raise ValueError("No data or random generation method specified.")

        # Initialize features
        if features is None:
            self.features = list(self.data.columns)
        else:
            self.features = features

        # Initialize dataArray
        self.dataArray = self.data[self.features].to_numpy()
        self.indices = {feature: i for i, feature in enumerate(self.features)}
        self.interval = interval

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
            if isinstance(preprocessor, tuple): # with preprocessor parameters
                preprocessor, parameters = preprocessor
                setattr(self, name, preprocessor(self.dataArray, **parameters))
            else:
                setattr(self, name, preprocessor(self.dataArray))

    def scale(self, interval: tuple = None) -> None:
        """
        Scales self.dataArray numpy array based on self.interval tuple

        Args:
            interval: The interval to scale the data to. The class attribute is 
            used if none is specified.
        """
        if interval is None:
            interval = self.interval

        minVals = self.dataArray.min(axis=0)
        maxVals = self.dataArray.max(axis=0)
        
        # Avoid division by zero
        rangeVals = maxVals - minVals
        rangeVals[rangeVals == 0] = 1

        self.dataArray = (self.dataArray - minVals) / rangeVals
        self.dataArray = self.dataArray * (interval[1] - interval[0])
        self.dataArray += interval[0]
        
    def discretize(self, bins: int | ArrayLike, features: list = None, 
                   strategy: str = 'uniform', array: str = None) -> None:
        """
        Discretize self.dataArray into bins.

        Arg: 
            bins: Number of bins to use, bins in each feature, or bin edges.
            features: The features to use for the binning
            strategy: sklearn KBinsDiscretizer strategy to use from 'uniform', 
                'quantile', or 'kmeans'.
            array: The array to assignt he result to.
        
        Raises:
            ValueError: if dimensions or features do not match dimensionality
        """
        if features is None:
            features = self.features
        if array is None:
            array = "dataArray"

        # Gets specified features
        indices = [self.indices[feature] for feature in features]
        selected = self.dataArray[:, indices]
        discretizer = KBinsDiscretizer(n_bins = bins, 
                                       encode = 'ordinal', 
                                       strategy = strategy)

        setattr(self, array, discretizer.fit_transform(selected))
        self.bins = bins
        
    def encode(self, features: list = None, dimensions: int = 1, 
               array: str = None) -> None:
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

class Subset(Base):
    """
    A class for creating, storing, and handling subsets of datasets.
    """

    def __init__(self, dataset: Dataset, z: ArrayLike, solveTime: float = None,
                 loss: float = None) -> None:
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
            TypeError: If dataset is not an instance of Dataset.
        """
        # if not isinstance(dataset, Dataset):
        #     raise TypeError("Dataset must be an instance of Dataset class.")
        
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

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Subset object.
        """
        return (f"Subset(size={self.size}, "
                f"time={round(self.solveTime, 4)}s, "
                f"loss={round(self.loss, 4)})")
    
    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Subset object.
        """
        if len(self.size) == 1:
            size = str(self.size[0])
        else:
            size = f"{self.size[0]}x{self.size[1]}"

        return (f"subset of size {size} in {round(self.solveTime, 2)}s "
                f"with {round(self.loss, 2)} loss")