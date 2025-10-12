# --- Imports ------------------------------------------------------------------

# Third party
import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import make_blobs


# --- Random Dataset Generation ------------------------------------------------

def randomDataset(
    random: str | tuple[str, dict] | list[tuple[str, dict]],
    size: tuple, 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random dataset based on the specified random generation method.

    Args:
        random: The method(s) for random data generation. Supported methods: 
            "uniform", "binary", "categorical", "normal", "multimodal", "skew", 
            "blobs" and optional parameters.
        size: The size of the dataset to create for random dataset generation or
            the size of the data (num rows, num columns).
        seed: The random seed or generator for reproducibility. 

    Returns: A DataFrame containing the generated random data.

    Raises:
        ValueError: If the specified random generation method or size is not 
        supported or specified incorrectly.
    """
    if not (isinstance(size, tuple) and len(size) == 2 and
            all(isinstance(x, int) and x > 0 for x in size)):
        raise ValueError("Size must be a tuple of positive ints: (rows, cols)")
    
    if isinstance(random, str):
        specifications: list[tuple[str, dict]] = [(random, {})]
    elif isinstance(random, tuple):
        name, parameters = random
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError("Tuple form must be (name:str, parameters:dict)")
        specifications = [(name, parameters or {})]
    elif isinstance(random, list):
        if not random:
            raise ValueError("Random list must not be empty.")
        specifications = []
        for i, item in enumerate(random):
            if not (isinstance(item, tuple) and len(item) == 2 and
                    isinstance(item[0], str) and isinstance(item[1], dict)):
                raise ValueError(f"Random[{i}] must be a tuple[str, dict].")
            specifications.append((item[0], item[1] or {}))
    else:
        raise ValueError("Random must be specified by string, tuple, or list.")

    randomFunctions = {
        "uniform": uniform,
        "binary": binary,
        "categorical": categorical,
        "normal": normal,
        "multimodal": multimodal,
        "skew": skew,
        "blobs": blobs,
    }

    parts: list[pd.DataFrame] = []
    for i, (name, parameters) in enumerate(specifications):
        randomFunction = randomFunctions.get(name)
        if randomFunction is None:
            raise ValueError(f"Unknown random generation method: {name}")
        block = randomFunction(size=size, seed=seed, **parameters)
        parts.append(block)

    return pd.concat(parts, axis=1)

def uniform(
    size: tuple, 
    rangeInterval: tuple = (0, 1), 
    seed: (int | np.random.Generator| None) = None
) -> pd.DataFrame:
    """
    Generate random data from a uniform distribution using numpy.

    Args:
        size: Number of data points and features to generate.
        rangeInterval: Range of values for the uniform distribution.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated uniformly distributed data.
    """
    rng = np.random.default_rng(seed)
    data = rng.uniform(rangeInterval[0], rangeInterval[1], size=size)
    return pd.DataFrame(data)

def binary(
    size: tuple, 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random binary data points of bernoulli trials using numpy where 
    each feature has a random probability p.

    Args:
        size: Number of data points and features to generate.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated binary data.
    """
    rng = np.random.default_rng(seed)
    probabilities = rng.random(size=size[1]) # probabilities in features
    data = rng.binomial(1, probabilities, size=size)
    return pd.DataFrame(data)

def categorical(
    size: tuple, 
    numberInterval: tuple = (2, 5), 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random categorical data points using numpy with a random number of 
    categories and a random probability p in numberInterval.

    Args:
        size: Number of data points and features to generate.
        numberInterval: Range of possible number of categories.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated categorical data.
    """
    rng = np.random.default_rng(seed)
    categories = rng.integers(numberInterval[0], 
                              numberInterval[1], 
                              size=size[1])
    data = np.zeros(size, dtype=int)
    for i in range(size[1]):
        probabilities = rng.dirichlet(np.ones(categories[i]), size=1)[0]
        data[:, i] = rng.choice(categories[i], size=size[0], p=probabilities)
    return pd.DataFrame(data)

def normal(
    size: tuple,
    sigmaInterval: tuple = (0.1, 3), 
    muInterval: tuple =  (-0.5, 0.5), 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random data from a normal distribution using numpy centered on 
    random mean and with random standard deviation.

    Args:
        size: Number of data points and features to generate.
        muInterval: Range of values for the mean.
        sigmaInterval: Range of values for the standard deviation.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated normally distributed data.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(muInterval[0], muInterval[1], size=size[1])
    sigma = rng.uniform(sigmaInterval[0], sigmaInterval[1], size=size[1])

    data = sigma * rng.standard_normal(size) + mu
    return pd.DataFrame(data)

def multimodal(
    size: tuple, 
    modesInterval: tuple = (1, 5), 
    muInterval: tuple =  (-0.5, 0.5), 
    sigmaInterval: tuple = (0.1, 3), 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random data from multimodal distributions using numpy with a random
    number of normal distributions centered on random means and standard 
    deviations.

    Args:
        size: Number of data points and features to generate.
        interval: Range of possible number of modes for each feature.
        sigmaInterval: Range of possible standard deviations.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated normally distributed multimodal data.
    """

    rng = np.random.default_rng(seed)
    modes = rng.integers(modesInterval[0], modesInterval[1], size=size[1])

    data = np.zeros(size)
    for i in range(size[1]):
        mu = rng.uniform(muInterval[0], muInterval[1], size=modes[i]) # random μ's
        sigma = rng.uniform(sigmaInterval[0], sigmaInterval[1], size=modes[i])

        splitsRange = np.arange(1, size[0])
        splits = rng.choice(splitsRange, size=modes[i] - 1, replace=False)
        splits = np.sort(np.concatenate(([0], splits, [size[0]])))
        for j in range(modes[i]):
            start = splits[j]
            end = splits[j + 1]
            distribution = rng.normal(loc=mu[j], scale=sigma[j], size=end-start)
            data[start:end, i] = distribution
            start = end

    return pd.DataFrame(data)

def skew(
    size: tuple, 
    skewnessInterval: tuple = (-5, 5), 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random data from skewed distributions using scipy with random 
    skewness parameter.

    Args:
        size: Number of data points and features to generate.
        skewnessInterval: Range of possible skewness parameters.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated normally distributed skewed data.
    """
    rng = np.random.default_rng(seed)

    data = np.zeros((size[1], size[0]))
    for i in range(size[1]):
        a = rng.uniform(skewnessInterval[0], skewnessInterval[1], size=1)
        data[i, :] = scipy.stats.skewnorm.rvs(a, 
                                              size=size[0], 
                                              random_state=seed)

    return pd.DataFrame(data.T)

def blobs(
    size: tuple, 
    numClusters: int = 2, 
    rangeInterval: tuple = (-5, 5),
    sigmaInterval: tuple = (0.1, 3), 
    seed: (int | np.random.Generator | None) = None
) -> pd.DataFrame:
    """
    Generate random data points using sklearn with numClusters blobs and with 
    random means and standard deviations.

    Args:
        size: Number of data points and features to generate.
        numClusters: Number of clusters to generate.
        rangeInterval: Range of values for the cluster centers.
        sigmaInterval: Range of possible standard deviations.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated blobs data.
    """
    rng = np.random.default_rng(seed)
    rs = seed if isinstance(seed, int) else int(rng.integers(0, 2**31-1))
    sigma = rng.uniform(sigmaInterval[0], sigmaInterval[1], size=numClusters)

    data = make_blobs(n_samples = size[0], 
                      n_features = size[1], 
                      centers = numClusters, 
                      cluster_std = sigma, 
                      center_box = rangeInterval,
                      random_state=rs)
    
    return pd.DataFrame(data[0])
