# --- Imports ------------------------------------------------------------------

# Standard libraries
import os

# Third party libraries
import numpy as np
import pandas as pd
import pickle 

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# Local imports
from . import generate


# --- Sets ---------------------------------------------------------------------

class Base:
    """
    Base class for Dataset and Subset providing shared save and load functions.
    """
    
    def save(self, name, fileType="pickle", directory="../data", index=False):
        """
        Saves self.data as a file.

        Args:
            name (str): The name of the file.
            fileType (str, optional): The type of file (pickle or csv).
            directory (str, optional): Directory to save the file in.
            index (bool, optional): Whether to save the data index or not.
        
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        file_path = os.path.join(directory, f"{name}.{fileType}")
        
        if fileType == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(self.data, f)
        elif fileType == "csv":
            self.data.to_csv(file_path, index=index)
        else:
            raise ValueError(f"Unsupported file type: {fileType}.")

    def load(self, name, fileType="pickle", directory="../data"):
        """
        Loads self.data from a file.

        Args:
            name (str): The name of the file.
            fileType (str, optional): The type of file (pickle or csv).
            directory (str, optional): Directory to load the file from.
        
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        file_path = os.path.join(directory, f"{name}.{fileType}")
        
        if fileType == "pickle":
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        elif fileType == "csv":
            self.data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {fileType}.")


class Dataset(Base):
    """
    A class for creating, storing, and processing of datasets for subsetting
    """

    def __init__(self, data=None, randTypes=None, size=None, interval=(1, 5), 
                 features=None, seed=None):
        """
        Initialize a dataset with data or by random data generation.

        Args:
            data (array-like, optional): The DataFrame or ndarray of the data. 
            randTypes (str or list, optional): The method or methods for random 
                data generation. Supported methods: "uniform", "binary", 
                "categorical", "normal", "multimodal", "skew", "blobs"
            size (tuple): The size of the dataset to create for random dataset 
                generation or the size of the data (num_rows, num_columns).
            interval (tuple, optional): The interval for scaling data. 
            features (list, optional): The list of column features to consider. 
            seed (int, rng, optional): The random seed or Numpy rng for random 
                generation and reproducibility. 

        Raises:
            ValueError: If no data or random generation method is specified.
        """
        # Initialize data
        if data is not None:  # initialize with data
            self.size = data.shape
            if isinstance(data, pd.DataFrame):
                self.data = data 
            else:
                pd.DataFrame(data)
        elif randTypes is not None:  # initialize with random data generation
            if isinstance(randTypes, list):
                self.data = pd.DataFrame({
                    i: generate.randomData(randType, size, interval, seed) 
                    for i, randType in enumerate(randTypes)
                })
            else:
                self.data = generate.randomData(randTypes, size, interval, seed)
            self.size = size
        else:
            raise ValueError("No data or random generation method specified.")

        # Initialize features
        if features is None:
            self.features = list(self.data.columns)
        else:
            self.features = features

        self.dataArray = self.data[self.features].to_numpy()
        self.interval = interval

    def preprocess(self, **parameters):
        """
        Perform custom preprocessing of preprocessFunction on self.dataArray and
        assign it to the specified name.

        Args:
            parameters: Keyword arguments where the key is the name of the 
                preprocessing function and the value is either the function (for
                functions that don't require parameters), or a tuple where the 
                first element is the function and the second element is a 
                dictionary of additional parameters.
        """
        for name, preprocessTuple in parameters.items():
            if isinstance(preprocessTuple, tuple):
                function, funcParams = preprocessTuple
                setattr(self, name, function(self.dataArray, **funcParams))
            else:
                preprocessFunc = preprocessTuple
                setattr(self, name, preprocessFunc(self.dataArray))

    def scale(self, interval=None):
        """
        Scales self.dataArray numpy array based on self.interval tuple
        """
        if interval is None:
            interval = self.interval

        minVals = self.dataArray.min(axis=0)
        maxVals = self.dataArray.max(axis=0)
        self.dataArray = (self.dataArray - minVals) / (maxVals - minVals)
        self.dataArray = self.dataArray * (interval[1] - interval[0])
        self.dataArray += interval[0]
        
    def discretize(self, bins, dimensions=1, features=None, strategy='uniform'):
        """
        Discretize self.dataArray into bins.

        Arg: 
            bins (int or array-like): Number of bins to use, bins in each 
                feature, or bin edges.
            dimensions (int, optional): Dimensionality of bins. 1 dimension bins
                each feature separately. >1 uses bins in >=2D space
            features (List, optional): The features to use for the binning
            strategy (String, optional): sklearn KBinsDiscretizer strategy to 
                use from 'uniform', 'quantile', or 'kmeans'.
        
        Raises:
            ValueError: if unknown strategy or dimensions given or features do 
                not match necessary dimensionality
        """
        if strategy not in ['uniform', 'quantile', 'kmeans']:
            raise ValueError("unknown strategy specified")
        if dimensions < 1:
            raise ValueError("unknown dimension specified")
        if features is None:
            features = self.features

        # Get indices of the specified features
        indices = [self.features.index(feature) for feature in features]
        selected = self.dataArray[:, indices]
        discretizer = KBinsDiscretizer(n_bins = bins, 
                                       encode = 'ordinal', 
                                       strategy = strategy)

        if dimensions == 1:
            self.dataArray[:, indices] = discretizer.fit_transform(selected)
        else:
            numFeatures = len(features)
            if dimensions > numFeatures:
                raise ValueError(f"Cannot bin {features} in {dimensions}D")
            
            newData = []
            for i in range(0, numFeatures, dimensions):
                subset = selected[:, i:i+dimensions]
                binnedSubset = discretizer.fit_transform(subset)
                newData.append(binnedSubset)

            binnedData = np.hstack(newData)
            self.dataArray[:, indices] = binnedData

        self.bins = bins
        
    def encode(self):
        """
        One hot encodes self.dataArray numpy array. Assumes continuous variables
        have been binned if necessary.
        """
        encoder = OneHotEncoder()
        self.dataArray = encoder.fit_transform(self.dataArray).toarray()


class Subset(Base):
    """
    A class for creating, storing, and handling subsets of datasets.
    """

    def __init__(self, dataset, z):
        """
        Initialize a subset with a Dataset object and the indicator vector z.

        Args:
            dataset (Dataset): The dataset from which to take the subset.
            z (array-like): The indicator vector indicating which samples from 
                the dataset are included in the subset.

        Raises:
            ValueError: If length of z does not match the length of dataset.
            TypeError: If dataset is not an instance of Dataset.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be an instance of Dataset class.")
        
        if len(z) != dataset.data.shape[0]:
            raise ValueError("Length of z must match the length of dataset.")

        if dataset.data.ndim == 1:  # one-dimensional dataset
            self.size = (np.sum(z),)
        else:
            self.size = (np.sum(z), dataset.size[1])
        
        self.data = dataset.data[z == 1].copy()  # subset of the full data
        self.z = z

    def __repr__(self):
        """Return a string representation of the Subset object."""
        return f"Subset(size={self.size})"