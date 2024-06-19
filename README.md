# Flexible Subset Selection for Data Visualization

## Abstract
Subset selection has many uses in data visualization, but each use often has a specialized strategy or implementation. We propose a general strategy for organizing, designing, and prototyping subset selection for visualization by casting the problem in terms of multi-criterion optimization. This strategy allows us to select subsets for typical uses in data visualization. We can incorporate both standard and novel selection approaches using this strategy. It also provides several advantages. Objective functions encode criteria, representing the desired properties of a subset. The objectives can be selected, blended and tuned to meet the needs of particular visualizations. General purpose solvers can be used as an effective prototyping strategy. We applied the proposed strategy to example situations of designing visualizations using realistic example datasets, objectives, and charts. These demonstrate the effectiveness and flexibility of the strategy in common visualization uses such as decluttering scatterplots, summarizing datasets, or highlighting exemplars.

# Code
## Sets

### Overview
The `sets.py` module provides classes for managing datasets and subsets within a data science project. It includes functionality for dataset initialization, random data generation, preprocessing, scaling, binning, one-hot encoding, and saving datasets. Additionally, it supports creating subsets of datasets based on indicator vectors.

### Dataset Class
The `Dataset` class initializes and manages a dataset. It can either be initialized with existing data or generate random data based on specified methods.

- **Initialization**: Can initialize with provided data in tabular array-like form or generate random data from `generate.py`
  
- **Preprocessing**: Supports custom preprocessing functions provided as keyword arguments.
  
- **Scaling**: Scales the dataset based on specified intervals.
  
- **Binning**: Bins continuous data into discrete intervals.
  
- **One-Hot Encoding**: Encode categorical variables into binary vectors.
  
- **Saving**: Save the dataset as a pickled file for later use.

### Subset Class
The `Subset` class represents a subset of a `Dataset` based on an indicator vector `z`. It allows creating and managing subsets of data efficiently.

- **Initialization**: Initializes a subset from a parent `Dataset` based on an indicator vector `z`.
  
- **Saving**: Save the subset as a pickled file for later use.

## Generate

The `generate.py` module provides functions for generating random datasets based on various distribution types using numpy, scipy, and sklearn. It supports generating datasets for tasks like machine learning experiments, data analysis, and statistical simulations. Distribution types are: `uniform`, `binary`, `categorical`, `normal`, `multimodal`, `skew`, `blobs`

## Loss Functions and Metrics

The `loss.py` module provides classes and functions for defining multi-criterion and single criterion loss functions, as well as various metric functions for evaluating datasets and subsets.

### MultiCriterion Class

The `MultiCriterion` class defines a multi-criterion loss function using a set of objective functions, parameters, and optional weights. It allows combining multiple objectives into a single loss function for subset selection.

- **Initialization**: Initializes with a list of objective functions, parameters for each objective, and optional weights.
  
- **Calculate Method**: Computes the overall loss function by evaluating each objective function with its corresponding parameters and combining them with weights.

### UniCriterion Class

The `UniCriterion` class defines a single criterion loss function with an objective function and optional parameters for subset selection.

- **Initialization**: Initializes with an objective function, solve array name, selection method, and additional parameters.
  
- **Calculate Method**: Computes the loss by evaluating the objective function on the selected subset for given parameters.


## Solver Class

The `Solver` class encapsulates a solver with an algorithm and loss function for subset selection.

- **Initialization**: Initializes with a solve algorithm and a loss function.
  
- **solve Method**: Executes the algorithm on a specified dataset using optional parameters.



## Plot 

The `plot` module provides functions for automatic configuring and creating plots of dataseta dn subsets using Matplotlib and Seaborn.

### Color Class

- **Color Class**: Defines color palette and color bars used throughout the module for consistent visualization.

### Figure Operations

- **moveFigure**: Moves the upper-left corner of a figure to a specified pixel position.
  
- **clearAxes**: Clears all axes in a given figure.
  
- **removeAxes**: Removes all axes from a given figure.
  
- **setPickEvent**: Sets a pick event on a figure, invoking a specified function upon selection.

### Plot functions

Plot functions that take datasets and/or subsets and generate the corresponding plot containing one or both sets.

# Data

Data can be found in the data folder which stores saved pickle files for pandas dataframes of the randomly generated data and calculated subsets presented in our work. 