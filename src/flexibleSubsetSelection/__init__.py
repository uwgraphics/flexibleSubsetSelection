"""
A Python package for flexible subset selection for visualization via a 
multi-criterion optimization strategy.

This toolkit enables selection of subsets from arbitrary datasets selecting, 
blending, and tuning objectives using general optimization solving methods such 
as greedy, exact (MILP), and custom solving strategies.

Modules:
- Dataset and Subset structures
- Objective and loss functions
- Optimization solvers
- Visualization and analysis utilities
"""
__version__ = "0.2"

# Import top level classes
from .dataset import Dataset                   # Data management classes
from .subset import Subset                     
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