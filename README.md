# Flexible Subset Selection for Data Visualization

## Abstract
Subset selection is crucial in data visualization for decluttering, summarizing, and emphasizing key insights. This project proposes a unified approach using multi-criterion optimization for flexible subset selection strategies. By defining objective functions, we tailor subsets to meet specific visualization needs, leveraging both traditional and innovative selection methods. General-purpose solvers facilitate rapid prototyping, demonstrated through realistic examples of scatterplot decluttering, dataset summarization, and exemplar highlighting.

# Code

## Sets

### Overview
The `sets.py` module manages datasets and their subsets within data science projects. It supports dataset initialization, random data generation, preprocessing, scaling, binning, one-hot encoding, and dataset saving. Subset creation based on indicator vectors is also provided.

#### Dataset Class

- **Initialization**: Initialize with tabular data or generate random data using methods in `generate.py`.
- **Preprocessing**: Apply custom preprocessing functions.
- **Scaling**: Scale dataset values within specified intervals.
- **Binning**: Convert continuous data into discrete intervals.
- **One-Hot Encoding**: Encode categorical variables into binary vectors.
- **Saving**: Save datasets as pickled files.

#### Subset Class

The `Subset` class represents subsets of datasets based on indicator vectors (`z`). It allows efficient creation and management of data subsets.

- **Initialization**: Initialize a subset from a parent `Dataset` based on an indicator vector `z`.
- **Saving**: Save the subset as a pickled file.

## Generate

The `generate.py` module provides functions for generating random datasets based on various distribution types using numpy, scipy, and sklearn. It supports tasks like machine learning experiments, data analysis, and statistical simulations. Distribution types include `uniform`, `binary`, `categorical`, `normal`, `multimodal`, `skew`, and `blobs`.

## Loss Functions and Metrics

The `loss.py` module provides classes and functions for defining multi-criterion and single-criterion loss functions, as well as various metric functions for evaluating datasets and subsets.

### MultiCriterion Class

The `MultiCriterion` class defines a multi-criterion loss function using a set of objective functions, parameters, and optional weights. It allows combining multiple objectives into a single loss function for subset selection.

- **Initialization**: Initialize with a list of objective functions, parameters for each objective, and optional weights.
- **Calculate Method**: Compute the overall loss function by evaluating each objective function with its corresponding parameters and combining them with weights.

### UniCriterion Class

The `UniCriterion` class defines a single-criterion loss function with an objective function and optional parameters for subset selection.

- **Initialization**: Initialize with an objective function, solve array name, selection method, and additional parameters.
- **Calculate Method**: Compute the loss by evaluating the objective function on the selected subset for given parameters.

## Solver Class

The `Solver` class encapsulates a solver with an algorithm and loss function for subset selection.

- **Initialization**: Initialize with a solve algorithm and a loss function.
- **solve Method**: Execute the algorithm on a specified dataset using optional parameters.

## Plot

The `plot` module provides functions for configuring and creating plots of datasets and subsets using Matplotlib and Seaborn.

### Color Class

- **Color Class**: Define color palettes and color bars for consistent visualization.

### Figure Operations

- **moveFigure**: Move the upper-left corner of a figure to a specified pixel position.
- **clearAxes**: Clear all axes in a given figure.
- **removeAxes**: Remove all axes from a given figure.
- **setPickEvent**: Set a pick event on a figure, invoking a specified function upon selection.

### Plot functions

Functions that take datasets and/or subsets and generate corresponding plots.

# Data

Data is stored in the `data` folder, which contains saved pickle files of pandas dataframes representing randomly generated data and calculated subsets used in this project.
