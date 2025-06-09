# --- Imports ------------------------------------------------------------------

# Standard library
from typing import Callable

# Third party
from IPython.display import display, clear_output

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb, to_hex
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

# Local files
from .subset import Dataset, Subset
from .color import Color


# --- Figures ------------------------------------------------------------------

def moveFigure(fig, x: int, y: int):
    """Move figure's upper left corner to pixel (x, y)."""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        fig.canvas.manager.window.SetPosition((x, y))
    elif backend == "Qt5Agg" or backend == "Qt4Agg":
        fig.canvas.manager.window.move(x, y)

def clearAxes(fig):
    """Clear all axes in the figure."""
    for ax in fig.get_axes():
        ax.cla()

def removeAxes(fig):
    """Remove all axes in the figure."""
    for ax in fig.get_axes():
        ax.remove()

def setPickEvent(fig, pickFunction: Callable):
    """Set pickFunction as a pick event on the figure."""
    fig.canvas.mpl_connect("pick_event", pickFunction)

def onPick(event, color: Color):
    """Toggle color of the selected item between green and yellow on event."""
    line = event.artist
    if line._color == color.palette["yellow"]:
        line._color = color.palette["green"]
        line.zorder = 1
    else:
        line._color = color.palette["yellow"]
        line.zorder = 3
    line._axes.figure.canvas.draw_idle()

def initializePane3D(ax: Axes, color: str):
    """Initialize the color of the background panes with hex color for 3D."""
    rgb = to_rgb(color)
    ax.zaxis.set_pane_color(to_hex([min(1, c + (1 - c)*0.03) for c in rgb]))
    ax.xaxis.set_pane_color(to_hex([max(0, c*0.97) for c in rgb]))
    ax.yaxis.set_pane_color(rgb)    


# --- Error Indicators ---------------------------------------------------------

def errorBars(ax: Axes, x: float, vals1: float, vals2: float, color: str) -> None:
    """
    Plot a series of error bars on ax along x between vals1 and vals2 of a 
    given color.

    Args:
        ax: The axes on which to plot.
        x: The x-coordinate for the error bars.
        vals1: The first set of values.
        vals2: The second set of values.
        color: The color of the error bars.
    """
    ax.errorbar(x = x,
                y = (vals1 + vals2) / 2,
                yerr = abs(vals1 - vals2) / 2,
                ecolor = color,
                ls = "none",
                elinewidth = 3,
                capsize = 5,
                capthick = 1.5,
                zorder = 4)

def errorMarkers(ax: Axes, x: list, vals1: list | None, color1: str | None,
                 marker1: str, vals2: list | None = None, 
                 color2: str | None = None, marker2: str | None = None) -> None:
    """
    Plot a series of error markers on ax along x at vals1 and optionally vals2 
    of given colors.

    Args:
        ax: The axes on which to plot.
        x: The x-coordinates for the error markers.
        vals1: The first set of values.
        color1: The color for the first set of markers.
        marker1: The marker style for the first set of markers.
        vals2: The second set of values.
        color2: The color for the second set of markers.
        marker2: The marker style for the second set of markers.
    """
    for i in range(len(x)):
        if vals1 is not None and len(vals1[i]) > 0:
            for val in vals1[i]:
                ax.plot(x[i], 
                        val, 
                        color = color1, 
                        markersize = 4, 
                        marker = marker1, 
                        zorder = 4)

        if vals2 is not None and len(vals2[i]) > 0:
            for val in vals2[i]:
                ax.plot(x[i], 
                        val, 
                        color = color2, 
                        markersize = 3.5, 
                        markerfacecolor = None, 
                        marker = marker2, 
                        zorder = 4)

def errorLines(ax: Axes, vals1: np.ndarray, vals2: np.ndarray, color: str, 
               weights: (np.ndarray | None) = None) -> None:
    """
    Plot a series of error lines on ax at vals1 and vals2 of given color.

    Args:
        ax: The axes on which to plot.
        vals1: The first set of values.
        color: The color of the error lines.
        vals2: The second set of values.
        weights: The weights of the error lines.
    """
    # Create grid of points
    datasetX, subsetX = np.meshgrid(vals1[:, 0], vals2[:, 0], indexing='ij')
    datasetY, subsetY = np.meshgrid(vals1[:, 1], vals2[:, 1], indexing='ij')

    # Create line segments
    linesX = np.stack([datasetX, subsetX], axis=-1)
    linesY = np.stack([datasetY, subsetY], axis=-1)
    lines = np.stack([linesX, linesY], axis=-1).reshape(-1, 2, 2)

    # Assign weights if not provided
    if weights is None: 
        weights = np.ones(len(lines))
                           
    # Create LineCollection
    lines = LineCollection(lines, 
                           colors=color, 
                           linewidths=weights.flatten(), 
                           alpha=0.2)
    lines.set_capstyle("round")
    ax.add_collection(lines)


# --- Plots --------------------------------------------------------------------

def initialize(color, font: str = "Times New Roman", size: int = 42, 
               faceColorAx: (str | None) = None, 
               faceColorFig: (str | None) = None) -> None:
    """
    Initialize matplotlib settings global parameters for text and background
    
    Args:
        color (Color object): A color object with the color palette to use
        font (str, optional): Font name to set for text
        family (str, optional): Font family to use
        size (int, optional): Font size to use

    Raises: ValueError: If no correct color object is provided
    """
    if not isinstance(color, Color):
        raise ValueError("color must be an instance of Color object")
    
    plt.rcParams["font.family"] = font
    plt.rcParams["pdf.fonttype"] = size
    plt.rcParams["ps.fonttype"] = size
    plt.rcParams["figure.autolayout"] = True

    if faceColorFig is None:
        plt.rcParams["figure.facecolor"] = "white"
    else:
        plt.rcParams["axes.facecolor"] = faceColorFig

    if faceColorAx is None:
        plt.rcParams["axes.facecolor"] = color["grey"]
    else:
        plt.rcParams["axes.facecolor"] = faceColorAx

def scatter(
    ax: Axes, 
    color: Color, 
    dataset: Dataset | None = None, 
    subset: Subset | None = None, 
    features: list | None = None, 
    transform: str | None = None,
    **parameters
) -> None:
    """
    Plot a scatterplot of data features on ax

    Args:
        ax: The axis to plot the scatterplot on.
        color: A color object with the color palette to use.
        dataset: The dataset to plot.
        subset: The subset to plot.
        features: The features to plot on x and y axes.
        transform: The transformed dataset to plot
        **parameters: Additional parameters to pass to the plotting functions.
    
    Raises: 
        ValueError: If neither a dataset or subset are provided or 3D data 
        is specified without a 3D axis.
    """
    if dataset is None and subset is None:
        raise ValueError("No dataset or subset specified.")
    if features is None:
        features = (dataset or subset.dataset).features[:2]

    if len(features) == 3:
        if not hasattr(ax, "zaxis"):
            raise ValueError("3D data is specified but axis is not 3D.")
        initializePane3D(ax, color["grey"])
        data = []
        colors = []
        if dataset is not None:
            if transform is None:
                transformed = dataset
            else:
                transformed = getattr(dataset, transform)
            data.append(transformed)
            colors.extend([color["green"]] * dataset.size[0])
        if subset is not None:
            if transform is None:
                transformed = subset.array
            else:
                transformed = getattr(subset, transform)
            data.append(transformed)
            colors.extend([color["darkGreen"]] * subset.size[0])

        data = np.vstack(data)
        names = (dataset or subset.dataset).features
        x, y, z = [names.index(f) for f in features]
        ax.scatter(data[:, x], data[:, y], data[:, z], c=colors, **parameters)

    else:
        if dataset is not None:
            if transform is None:
                transformed = dataset.array
            else:
                transformed = getattr(dataset, transform)
            df = pd.DataFrame(transformed, columns=dataset.features)
            sns.scatterplot(data = df, 
                            x = features[0], 
                            y = features[1], 
                            color = color["green"], 
                            ax = ax,
                            zorder = 3,
                            **parameters)
        if subset is not None:
            if transform is None:
                transformed = subset.array
            else:
                transformed = getattr(subset, transform)
            df = pd.DataFrame(transformed, columns=subset.dataset.features)
            sns.scatterplot(data = df, 
                            x = features[0], 
                            y = features[1], 
                            color = color["darkGreen"], 
                            ax = ax,
                            zorder = 4,
                            **parameters)

def parallelCoordinates(ax: Axes, 
    color: Color, 
    dataset: (Dataset | None) = None, 
    subset: (Subset | None) = None, 
    transform: str | None = None,
    dataLinewidth: float = 0.5, 
    subsetLinewidth: float = 1.5, 
    **parameters
) -> None:
    """
    Plot a parallel coordinates chart of dataset on ax

    Args:
        ax: The axis to plot the parallel coordinates on
        dataset: The dataset to plot
        color: A color object with the color palette to use
        subset: The subset to plot
        transform: The transformed dataset to plot
        dataLinewidth: Linewidth for the main dataset
        subsetLinewidth: Linewidth for the subset
        **parameters: Additional parameters to pass to 
            pd.plotting.parallel_coordinates

    Raises: ValueError: If neither a dataset or subset are provided
    """
    if dataset is None and subset is None:
        raise ValueError("At least one of dataset or subset must be provided.")

    if dataset is not None:
        df_dataset = pd.DataFrame(getattr(dataset, transform), 
                                  columns=dataset.features)
        df_dataset["set"] = "dataset"
        pd.plotting.parallel_coordinates(df_dataset,
                                         "set",
                                         ax=ax,
                                         color=color.palette["green"],
                                         axvlines_kwds={'c': "white", "lw": 1},
                                         linewidth=dataLinewidth,
                                         **parameters)

    if subset is not None:
        df_subset = pd.DataFrame(getattr(subset, transform), 
                                 columns=subset.dataset.features)
        df_subset["set"] = "subset"
        pd.plotting.parallel_coordinates(df_subset,
                                         "set",
                                         ax=ax,
                                         color=color.palette["darkGreen"],
                                         axvlines_kwds={'c': "white", "lw": 1},
                                         linewidth=subsetLinewidth,
                                         alpha=1,
                                         **parameters)

def histogram(ax: Axes, 
    color: Color, 
    dataset: (Dataset | None) = None, 
    subset: (Subset | None) = None, 
    numBins: int = 6, 
    transform: str | None = None,
    **parameters
) -> None:
    """
    Plot histograms of each feature side by side on ax with normalized subset 
    and dataset overlapping on common bins

    Args:
        ax: The axis to plot the histogram on
        color: A color object with the color palette to use
        dataset: The dataset to plot
        subset: The subset to plot
        numBins: The number of bins to bin the dataset
        transform: The transformed dataset to plot
    
    Raises: ValueError: If neither a dataset or subset are provided
    """
    if dataset is None and subset is None:
        raise ValueError("No dataset or subset specified")

    if dataset is not None:        
        # Get the positions of each bar group
        barPositions = range(0, numBins * dataset.size[1], numBins)
        
        for i, feature in enumerate(dataset.features):
            # Plot the dataset histogram
            datasetHist = np.histogram(getattr(dataset, transform)[feature], 
                                       bins=numBins)
            datasetHeights = datasetHist[0]
            
            # Adjust bar positions
            positions = barPositions[i] + np.arange(numBins)
            
            ax.bar(positions, datasetHeights, width=1, 
                   color=color.palette["green"], alpha=0.5)
    if subset is not None:
        features = subset.dataset.features
        numFeatures = len(features)
        
        # Get the positions of each bar group
        barPositions = range(0, numBins * numFeatures, numBins)
        
        for i, feature in enumerate(features):
            # Calculate histogram of subset normalized by subset size
            subsetHist = np.histogram(getattr(subset, transform)[feature], 
                                      bins=numBins)
            subsetHeights = subsetHist[0] / subset.size[0] * dataset.size[0]
            
            # Adjust bar positions
            positions = barPositions[i] + np.arange(numBins)
            
            ax.bar(positions, subsetHeights, width=1, 
                   color=color.palette["darkGreen"], alpha=0.5)
            
class RealTimePlotter:
    def __init__(self, color):
        # Initialize the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.iterations = []
        self.losses = []
        self.subsetSizes = []
        self.color = color

        # Set up the initial plot
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Real-Time Loss During Solver')
        self.ax.set_ylim(bottom=0)  # Set the lower y-axis limit to 0

        # Display the figure initially
        display(self.fig)
    
    def update(self, iteration, loss, subsetSize=None):
        # Append the current iteration and loss to lists
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.subsetSizes.append(subsetSize)

        # Clear the previous output (but don't clear the figure)
        clear_output(wait=True)
        self.ax.clear()
        
        self.ax.set_xlabel('Iteration')
        # self.ax.set_ylabel('Loss')
        self.ax.set_title('Minimum Loss')
        
        # Plot the updated loss values
        self.ax.plot(self.iterations, 
                     self.losses, 
                     c=self.color["orange"], 
                     label="Loss", 
                     lw=2)
        if subsetSize is not None:
            self.ax.plot(self.iterations, 
                        self.subsetSizes, 
                        c=self.color["green"], 
                        label="Subset Size", 
                        lw=2)
        self.ax.legend()
        
        # Display the updated plot
        display(self.fig)
        plt.close(self.fig)

    def close(self):
        # Close the plot when done
        plt.close(self.fig)