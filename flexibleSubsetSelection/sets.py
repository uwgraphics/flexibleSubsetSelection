# --- Imports ------------------------------------------------------------------

# Third party libraries
import numpy as np
import pandas as pd
import pickle 

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# Local imports
import generate


# --- Sets ---------------------------------------------------------------------

class Base:
    """
    Base class of Dataset and Subset providing shared save and load functions
    """
    def save(self, name, fileType="pickle", directory="../data", index=False):
        """
        Saves self.data as a file.

        Args:
            name (str): The name of the file.
            fileType (str, optional): The type of file (pickle or csv).
            directory (str, optional): Directory to save file in.
            index (bool, optional): Whether to save the data index or not.
        
        Raises: ValueError if unsupported filetype is specified
        """
        if fileType == "pickle":
            with open(f"{directory}/{name}.pkl", 'wb') as f:
                pickle.dump(self.data, f)
        elif fileType == "csv":
            self.data.to_csv(f"{directory}/{name}.csv", index=index)
        else:
            raise ValueError(f"Unsupported file type: {fileType}")

    def load(self, name, fileType="pickle", directory="data"):
        """
        Loads self.data from a file.

        Args:
            name (str): The name of the file.
            fileType (str, optional): The type of file (pickle or csv).
            directory (str, optional): Directory to load file from.
        
        Raises: ValueError if unsupported filetype is specified
        """
        if fileType == "pickle":
            with open(f"{directory}/{name}.pkl", 'rb') as f:
                self.data = pickle.load(f)
        elif fileType == "csv":
            self.data = pd.read_csv(f"{directory}/{name}.csv")
        else:
            raise ValueError(f"Unsupported file type: {fileType}")
        
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

        Raises: ValueError: If no data or random generation method is specified.
        """

        # Initialize data
        if data is not None: # initialize with data
            self.size = data.shape
            self.data = data
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
            raise ValueError("no data or random generation method specified")
        
        # Initialize features
        if features is None:
            if isinstance(self.data, pd.DataFrame):
                self.features = tuple(self.data.columns)
            else:
                self.features = tuple(range(self.data.shape[1]))
        else:
            self.features = features

        self.dataArray = self.data.to_numpy() # numpy array for calculations
        self.interval = interval              # interval of dataset
        self.scale()                          # scale the data to interval 
        
    def preprocess(self, **parameters):
        """
        Perform preprocessing tasks.

        Args:
            parameters: Keyword arguments where the key is the name of the 
                preprocessing function and the value is the function itself.
        """
        for name, preprocessFunction in parameters.items():
            setattr(self, name, preprocessFunction(self.dataArray))
    
    def scale(self):
        """
        Scales self.dataArray numpy array based on self.interval tuple
        """
        min = self.dataArray.min(axis=0)
        max = self.dataArray.max(axis=0)
        self.dataArray = (self.dataArray - min) / (max - min)
        self.dataArray = self.dataArray * (self.interval[1] - self.interval[0])
        self.dataArray += self.interval[0]
        
    def bin(self, numBins=6):
        """
        Bins self.dataArray numpy array into bins based on self.numBins.

        Arg: 
            numBins (int or array-like, optional): The number of bins to use
        """
        self.numBins = numBins
        est = KBinsDiscretizer(n_bins = self.numBins, 
                               encode = 'ordinal', 
                               strategy = 'uniform')
        self.dataArray = est.fit_transform(self.dataArray)
        
    def oneHotEncode(self):
        """
        One hot encodes self.dataArray numpy array. Assumes continuous variables
        have been binned if necessary.
        """
        encoder = OneHotEncoder()
        self.dataArray = encoder.fit_transform(self.dataArray).toarray()


class Subset(Base):
    """
    A class for creating, storing, and handling subsets of datasets
    """
    def __init__(self, dataset, z):
        """
        Initialize a subset with a Dataset object and the indicator vector z.

        Args:
            dataset (Dataset): The Dataset object from which the subset is taken
            z (array-like): The indicator vector indicating which samples from 
                the dataset are included in the subset.
        """

        self.size = (np.sum(z), dataset.size[1])
        self.z = z
        self.data = dataset.data[z == 1].copy() # subset of the full data