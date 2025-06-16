# --- Imports ------------------------------------------------------------------

# Third party
import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import make_blobs


# --- Random Dataset Generation ------------------------------------------------


def random(
    randType: str,
    size: tuple,
    interval: tuple,
    seed: (int | np.random.Generator | None) = None,
) -> pd.DataFrame:
    """
    Generate random data based on the specified random generation method.

    Args:
        randType: The method or methods for random data generation. Supported
            methods: "uniform", "binary", "categorical", "normal", "multimodal",
            "skew", and "blobs".
        size: The size of the dataset to create for random dataset generation or
            the size of the data (num rows, num columns).
        interval: The interval specifying the range of random values.
        seed: The random seed or generator for reproducibility.

    Returns: A DataFrame containing the generated random data.

    Raises:
        ValueError: If the specified random generation method is not supported.
    """
    generators = {
        "uniform": uniform,
        "binary": binary,
        "categories": categories,
        "normal": normal,
        "multimodal": multimodal,
        "skew": skew,
        "blobs": blobs,
    }

    generator = generators.get(randType)
    if generator:
        if generator == binary:
            return generator(size=size, seed=seed)
        else:
            return generator(size=size, interval=interval, seed=seed)
    else:
        raise ValueError(f"unknown random generation method: {randType}")


def uniform(
    size: tuple, interval: tuple, seed: int | np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Generate random data from a uniform distribution using numpy.

    Args:
        size: Number of data points and features to generate.
        interval: Range of values for the uniform distribution.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated uniformly distributed data.
    """
    rng = np.random.default_rng(seed)
    data = rng.uniform(interval[0], interval[1], size=size)
    return pd.DataFrame(data)


def binary(size: tuple, seed: int | np.random.Generator | None = None) -> pd.DataFrame:
    """
    Generate random binary data points of bernoulli trials using numpy where
    each feature has a random probability p.

    Args:
        size: Number of data points and features to generate.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated binary data.
    """
    rng = np.random.default_rng(seed)
    probabilities = rng.random(size=size[1])  # probabilities in features
    data = rng.binomial(1, probabilities, size=size)
    return pd.DataFrame(data)


def categories(
    size: tuple, interval: tuple, seed: int | np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Generate random categorical data points using numpy with a random number of
    categories and a random probability p in interval.

    Args:
        size: Number of data points and features to generate.
        interval: Range of possible number of categories.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated categorical data.
    """
    rng = np.random.default_rng(seed)
    categories = rng.integers(interval[0], interval[1], size=size[1])
    data = np.zeros(size, dtype=int)
    for i in range(size[1]):
        probabilities = rng.dirichlet(np.ones(categories[i]), size=1)[0]
        data[:, i] = rng.choice(categories[i], size=size[0], p=probabilities)
    return pd.DataFrame(data)


def normal(
    size: tuple, interval: tuple, seed: int | np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Generate random data from a normal distribution using numpy centered on
    random mean and with random standard deviation.

    Args:
        size: Number of data points and features to generate.
        interval: Range of values for the mean and standard deviation.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated normally distributed data.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(interval[0], interval[1], size=size[1])  # random μ's
    sigma = rng.uniform(interval[0], interval[1], size=size[1])  # random σ's

    data = sigma * rng.standard_normal(size) + mu
    return pd.DataFrame(data)


def multimodal(
    size: tuple,
    interval: tuple,
    sigmaInterval: tuple = (0.1, 3),
    seed: int | np.random.Generator | None = None,
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
    modes = rng.integers(interval[0], interval[1], size=size[1])

    data = np.zeros(size)
    for i in range(size[1]):
        mu = rng.uniform(interval[0], interval[1], size=modes[i])  # random μ's
        sigma = rng.uniform(sigmaInterval[0], sigmaInterval[1], size=modes[i])

        splitsRange = np.arange(1, size[0])
        splits = rng.choice(splitsRange, size=modes[i] - 1, replace=False)
        splits = np.sort(np.concatenate(([0], splits, [size[0]])))
        for j in range(modes[i]):
            start = splits[j]
            end = splits[j + 1]
            distribution = rng.normal(loc=mu[j], scale=sigma[j], size=end - start)
            data[start:end, i] = distribution
            start = end

    return pd.DataFrame(data)


def skew(
    size: tuple,
    interval: tuple = (-5, 5),
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate random data from skewed distributions using scipy with random
    skewness parameter.

    Args:
        size: Number of data points and features to generate.
        interval: Range of possible skewness parameters.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated normally distributed skewed data.
    """
    rng = np.random.default_rng(seed)

    data = np.zeros((size[1], size[0]))
    for i in range(size[1]):
        a = rng.uniform(interval[0], interval[1], size=1)
        data[i, :] = scipy.stats.skewnorm.rvs(a, size=size[0], random_state=seed)

    return pd.DataFrame(data.T)


def blobs(
    size: tuple,
    interval: tuple,
    numClusters: int = 6,
    sigmaInterval: tuple = (0.1, 3),
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate random data points using sklearn with numClusters blobs and with
    random means and standard deviations.

    Args:
        size: Number of data points and features to generate.
        numClusters: Number of clusters to generate.
        interval: Range of values for the cluster centers.
        sigmaInterval: Range of possible standard deviations.
        seed: Seed or rng for random number generation.

    Returns: The randomly generated blobs data.
    """
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(sigmaInterval[0], sigmaInterval[1], size=numClusters)

    data = make_blobs(
        n_samples=size[0],
        n_features=size[1],
        centers=numClusters,
        cluster_std=sigma,
        center_box=interval,
    )

    return pd.DataFrame(data[0])
