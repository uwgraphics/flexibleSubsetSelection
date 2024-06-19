# --- Imports ------------------------------------------------------------------

# Third party libraries
import numpy as np
import pandas as pd
import pickle 

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

# Local imports
import generate


# --- Sets ---------------------------------------------------------------------

class Dataset():
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
        self.interval = interval
        self.scale()
        
    def preprocess(self, **parameters):
        """
        Perform preprocessing tasks.

        Args:
            kwargs: Keyword arguments where the key is the name of the 
                preprocessing function and the value is the function itself.
        """
        for name, preprocessFunction in parameters.items():
            setattr(self, name, preprocessFunction(self.data))
    
    def scale(self):
        """
        Scales self.dataArray numpy array based on self.interval tuple
        """
        min = self.dataArray.min(axis=0)
        max = self.dataArray.max(axis=0)
        self.dataArray = (self.dataArray - min) / (max - min)
        self.dataArray = self.dataArray * (self.interval[1] - self.interval[0])
        self.dataArray += self.interval[0]
        
    def bin(self, numBins=8):
        """
        Bins self.dataArray numpy array into bins based on self.numBins.

        Arg: numBins (int, array-like): The number of bins for data binning. 
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

    def save(self, filename):
        """
        Saves self.data as a pickled file.
        """
        with open(f"data/{filename}Data.pkl", 'wb') as f:
            pickle.dump(self.data, f)

class Subset():
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
        self.data = dataset.data[z == 1].copy()  # subset of the full data

    def save(self, filename):
        """
        Saves self.data as a pickled file.
        """
        with open(f"data/{filename}Subset.pkl", 'wb') as f:
            pickle.dump(self.data, f)
