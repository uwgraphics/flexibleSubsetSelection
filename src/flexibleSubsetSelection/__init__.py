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

# Import top-level classes
from .dataset import Dataset
from .subset import Subset                     
from .loss import UniCriterion, MultiCriterion
from .solver import Solver
from .color import Color

# Import sub-modules
from . import (
    plot,      # Plotting functions for datasets and subsets
    algorithm, # Algorithms for subset selection
    objective, # Objective functions for defining criteria
    metric,    # Data metric functions
    logger     # Logging information to console or files
)

__all__ = [
    "Dataset",
    "Subset",
    "UniCriterion",
    "MultiCriterion",
    "Solver",
    "Color",
    "plot",
    "algorithm",
    "objective",
    "metric",
    "logger",
]