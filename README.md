# Flexible Subset Selection for Data Visualization

Subset selection has many uses in data visualization, but each use often has a specialized strategy or implementation. We propose a general strategy for organizing, designing, and prototyping subset selection for visualization by casting the problem in terms of multi-criterion optimization. This strategy allows us to select subsets for typical uses in data visualization. We can incorporate both standard and novel selection approaches using this strategy. It also provides several advantages. Objective functions encode criteria, representing the desired properties of a subset. The objectives can be selected, blended and tuned to meet the needs of particular visualizations. General purpose solvers can be used as an effective prototyping strategy. We applied the proposed strategy to example situations of designing visualizations using realistic example datasets, objectives, and charts. These demonstrate the effectiveness and flexibility of the strategy in common visualization uses such as decluttering scatterplots, summarizing datasets, or highlighting exemplars.


## Sets

Contains Dataset and Subset classes for working with data and selecting subsets

### Dataset

Load datasets using sets.Dataset() or generate random ones by specifying randTypes. These use generation methods found in generate.py to create random datasets.

### Subset

Create a subset from a dataset and an indicator vector z.


## Generate

Generate random distributions for randomized datasets


## Loss

### Multi-criterion Loss

Create a multicriterion loss function from a set of objective functions and optional weight parameters. Specify objective parameters with the loss function.

### Uni-criterion Loss

Create a loss function from a single objective function.


## Solver

Specify a solver from approximation or optimization implementations and solve for an indicator vector z. Create a subset with the resulting indicator vector.


## Plot 

Plot results as in plot.scatter(subset, dataset).
