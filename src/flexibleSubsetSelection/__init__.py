"""
A Python package for flexible subset selection via multi-criterion objective 
optimization. This package allows subsets to be selected from datasets according
to loss functions formed by selecting, blending, and tuning objectives using 
general optimization solving methods such as greedy algorithms or MILP solvers.
"""

# Import top level classes
from .sets import Dataset, Subset              # Data management classes
from .loss import UniCriterion, MultiCriterion # Loss function classes
from .solver import Solver                     # Solver class
from .color import Color                       # Color class

# Import sub-level component functions
from . import (
    plot,      # Plotting functions for datasets and subsets
    algorithm, # Algorithms for subset selection
    objective, # Objective functions for defining criteria
    metric,    # Data metric functions
    logger     # Logging information to console or files
)