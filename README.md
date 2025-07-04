# Flexible Subset Selection for Data Visualization

Subset selection has many uses in data visualization, but each use often has a specialized strategy or implementation.
We propose a general strategy for organizing, designing, and prototyping subset selection for visualization by casting the problem in terms of multi-criterion optimization. 
This strategy allows us to select subsets for typical uses in data visualization.
We can incorporate both standard and novel selection approaches using this strategy. 
It also provides several advantages. 
Objective functions encode criteria, representing the desired properties of a subset. 
The objectives can be selected, blended and tuned to meet the needs of particular visualizations. 
General purpose solvers can be used as an effective prototyping strategy. 
We applied the proposed strategy to example situations of designing visualizations using realistic example datasets, objectives, and charts. 
These demonstrate the effectiveness and flexibility of the strategy in common visualization uses such as decluttering scatterplots, summarizing datasets, or highlighting exemplars.

## Usage
Once downloaded, installation is recommended to be done using  using `pip` and an environment manager. Dependencies can be installed from the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

The flexibleSubsetSelection package can be installed locally: 
```sh
pip install .
```

Alternatively, if you want to modify the source code of the package, you can install it in editable mode:
```sh
pip install -e .
```

Once installed, the package can be imported, such as by:
```py
import flexibleSubsetSelection as fss
```

## Sets

### Overview
Dataset objects can be initialized from an existing source (DataFrame, array, database, or file) or by random dataset generation. Data transformations can be applied to the dataset prior to subsetting via scaling, discretizing, one-hot encoding, and other custom preprocessing functions. Metrics of the dataset can be procomputed. Subsets can be created from a Dataset and an indicator vector indicating which items in the dataset to include in the subset. Datasets and Subsets can be saved and reloaded.

### Dataset Class

- **Initialization**: Initialize with tabular data or generate random data using methods in `generate.py`.
- **Preprocessing**: Apply custom preprocessing functions.
- **Scaling**: Scale dataset values within specified intervals.
- **Binning**: Convert continuous data into discrete intervals.
- **One-Hot Encoding**: Encode categorical variables into binary vectors.
- **Saving**: Save datasets as pickled or csv files.
- **Loading**: Load datasets from pickled or csv files.

#### Example Usage

```py
dataset = fss.Dataset(randTypes="multimodal", size=(500, 10), seed=123)
dataset.save("../data/myNewDataset", fileType="csv")
```

### Subset Class

The `Subset` class represents subsets of datasets specified by indicator vectors.

- **Initialization**: Initialize a subset from a `Dataset` based on an indicator vector `z`.
- **Saving**: Save datasets as pickled or csv files.
- **Loading**: Load datasets from pickled or csv files.

#### Example Usage

```py
subset = fss.Subset(dataset, z)
subset.save("../data/myNewSubset", fileType="csv")
```

### Generate

The `generate.py` module provides functions for generating random datasets based on various distribution types using numpy, scipy, and sklearn. Distribution types include `uniform`, `binary`, `categorical`, `normal`, `multimodal`, `skew`, and `blobs`. Specify the random generation type when creating datasets using the `randTypes` parameter of `Dataset` as a string name or a list of string names per column of the new Dataset.

#### Example Usage

```py
skewDataset = fss.Dataset(randTypes="skew", size=(1000, 2))
variedDataset = fss.Dataset(randTypes=["skew", "normal", "multimodal"], size=(1000, 3))
```

## Loss Functions and Metrics

The `loss.py` module provides classes and functions for defining multi-criterion and single-criterion loss functions, as well as various metric functions for evaluating datasets and subsets.

### MultiCriterion Class

The `MultiCriterion` class defines a multi-criterion loss function using a set of objective functions, parameters, and optional weights. It allows combining multiple objectives into a single loss function for subset selection.

- **Initialization**: Initialize with a list of objective functions, parameters for each objective, and optional weights.
- **Calculate**: Compute the overall loss function by evaluating each objective function with its corresponding parameters and combining them with weights.

#### Example Usage

```py
objectives = [fss.loss.earthMoversDistance, fss.loss.distinctiveness]
parameters = [{"dataset": dataset.dataArray}, 
              {"solveArray": "distances", "selectBy": "matrix"}]
weights = np.array([1000, 0.1])
solveMethod.loss = fss.loss.MultiCriterion(objectives, parameters, weights=weights)
```

### UniCriterion Class

The `UniCriterion` class defines a single-criterion loss function with an objective function and optional parameters for subset selection.

- **Initialization**: Initialize with an objective function, solve array name, selection method, and additional parameters.
- **Calculate**: Compute the loss by evaluating the objective function on the selected subset for given parameters.


#### Example Usage

```py
dataset.preprocess(distances = fss.loss.distanceMatrix)
solveMethod.loss = fss.UniCriterion(objective = fss.loss.distinctiveness,
                                    solveArray = "distances",
                                    selectBy = "matrix")
```

## Solvers

The `Solver` class encapsulates a solver with an algorithm and loss function for subset selection.

- **Initialization**: Initialize with a solve algorithm and a loss function.
- **Solve**: Execute the algorithm on a specified dataset using optional parameters.

#### Example Usage

```py
solveMethod = fss.Solver(algorithm=fss.solver.greedyMinSubset, loss=lossFunction)
```

## Plot

The `plot` module provides functions for configuring and creating plots of datasets and subsets using the Matplotlib and Seaborn libraries.

### Color Class

- **Color Class**: Define color palettes and color bars for consistent visualization.

### Figure Operations

- **moveFigure**: Move the upper-left corner of a figure to a specified pixel position.
- **clearAxes**: Clear all axes in a given figure.
- **removeAxes**: Remove all axes from a given figure.
- **setPickEvent**: Set a pick event on a figure, invoking a specified function upon selection.

### Plot functions

Functions that take datasets and/or subsets and generate corresponding plots. Including `scatter`, `parallelCoordinates`, and `histogram`.

#### Example Usage

```py
# Initialize color and plot settings
color = fss.plot.Color()
fss.plot.initialize(color)

# Create a scatterplot
fss.plot.scatter(ax, color, dataset, subset, alpha=0.5)
```

# Data

Data is stored in the `data` folder, which contains subdirectories containing saved pickle files of pandas dataframes representing randomly generated data and calculated subsets used in this project. Example datasets used are in the `data/exampleDatasets` subdirectory. 


# Figures

Figures are stored in the `figures` directory as PDFs. 


# [Notebooks](https://pages.graphics.cs.wisc.edu/flexibleSubsetSelection/)

A series of example demonstration Jupyter Notebooks can be found in the `jupyter` directory. 
