# --- Imports ------------------------------------------------------------------

# Third party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# --- Color --------------------------------------------------------------------

class Color:
    """
    Create and store color palettes and color bars for use in visualizations
    """
    def __init__(self, palette=None):
        if palette is None:
            self.palette = {
                "green": "#8dd3c7",
                "darkGreen": "#338477",
                "orange": "#fb8072",
                "yellow": "#fdb462",
                "blue": "#8dadd3",
                "grey": "#eff0f2"
            }
        else:
            self.palette = palette         


# --- Figures ------------------------------------------------------------------

def moveFigure(fig, x, y):
    """move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        fig.canvas.manager.window.SetPosition((x, y))
    elif backend == "Qt5Agg" or backend == "Qt4Agg":
        fig.canvas.manager.window.move(x, y)

def clearAxes(fig):
    """clear all axes in the figure"""
    for ax in fig.get_axes():
        ax.cla()

def removeAxes(fig):
    """remove all axes in the figure"""
    for ax in fig.get_axes():
        ax.remove()

def setPickEvent(fig, pickFunction):
    """Set pickFunction as a pick event on the figure"""
    fig.canvas.mpl_connect("pick_event", pickFunction)

def onPick(event, color):
    """Toggle color of the selected item between green and yellow on event"""
    line = event.artist
    if line._color == color.palette["yellow"]:
        line._color = color.palette["green"]
        line.zorder = 1
    else:
        line._color = color.palette["yellow"]
        line.zorder = 3
    line._axes.figure.canvas.draw_idle()

# --- Error Markers ------------------------------------------------------------

def errorBar(ax, x, vals1, vals2, color):
    """
    plot a series of error bars on ax along x between vals1 and vals2 of
    given color
    """
    ax.errorbar(x=x,
                y=(vals1 + vals2)/2,
                yerr=abs(vals1 - vals2)/2,
                ecolor=color,
                ls="none",
                elinewidth=3,
                capsize=5,
                capthick=1.5,
                zorder=4)

def errorMarkers(ax, x, vals1, color1, marker1, vals2=None, color2=None, 
                 marker2=None):
    """
    plot a series of error markers on ax along x at vals1 and optionally vals2 
    of given colors
    """

    for i in x:
        if vals1[i].size > 0:
            for j in range(vals1[i].size):
                ax.plot(x[i], vals1[i][j], 
                        color = color1, 
                        markersize = 4, 
                        marker = marker1, 
                        zorder = 4)
        if vals2[i].size > 0:
            for j in range(vals2[i].size):
                ax.plot(x[i], vals2[i][j], 
                        color = color2, 
                        markersize = 3.5, 
                        markerfacecolor = None, 
                        marker = marker2, 
                        zorder = 4)


# --- Plots --------------------------------------------------------------------

def initialize(color, font="Times New Roman", family="sans-serif", size=42):
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
    
    plt.rcParams["font.sans-serif"] = font
    plt.rcParams["font.family"] = family
    plt.rcParams["pdf.fonttype"] = size
    plt.rcParams["ps.fonttype"] = size
    plt.rcParams["axes.facecolor"] = color.palette["grey"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["figure.autolayout"] = True

def scatter(ax, color, dataset=None, subset=None, features=(0, 1), 
            **parameters):
    """
    Plot a scatterplot of data features on ax

    Args:
        ax (matplotlib ax): The axis to plot the scatterplot on
        color (Color object): A color object with the color palette to use
        dataset (sets.Dataset object, optional): The dataset to plot
        subset (sets.Subset object, optional): The subset to plot
        features (tuple, optional): The features to plot on x and y axes
    
    Raises: ValueError: If neither a dataset or subset are provided
    """

    if dataset is None and subset is None:
        raise ValueError("no dataset or subset specified")
    if dataset is not None:
        sns.scatterplot(data = dataset.data, 
                        x = features[0], 
                        y = features[1], 
                        color = color.palette["green"],
                        ax = ax,
                        **parameters)
        
    if subset is not None:
        sns.scatterplot(data = subset.data, 
                        x = features[0], 
                        y = features[1], 
                        color = color.palette["darkGreen"], 
                        ax = ax,
                        zorder=4,
                        **parameters)
        
def parallelCoordinates(ax, dataset, color, subset=None, dataLinewidth=0.5, 
                        subsetLinewidth=1.5, **parameters):
    """
    Plot a parallel coordinates chart of dataset on ax

    Args:
        ax (matplotlib ax): The axis to plot the parallel coordinates on
        dataset (pandas DataFrame): The dataset to plot
        color (Color object): A color object with the color palette to use
        subset (pandas DataFrame or None, optional): The subset to plot
        dataLinewidth (float, optional): Linewidth for the main dataset
        subsetLinewidth (float, optional): Linewidth for the subset
        **parameters: Additional parameters to pass to 
            pd.plotting.parallel_coordinates

    Raises: ValueError: If neither a dataset or subset are provided
    """
    if dataset is None and subset is None:
        raise ValueError("At least one of dataset or subset must be provided.")
    
    if dataset is not None:
        pd.plotting.parallel_coordinates(dataset.data.assign(set="dataset"),
                                         "set",
                                         ax=ax,
                                         color=color.palette["green"],
                                         axvlines_kwds={'c': "white", "lw": 1},
                                         linewidth=dataLinewidth,
                                         **parameters)
    if subset is not None:
        pd.plotting.parallel_coordinates(subset.data.assign(set="subset"),
                                         "set",
                                         ax=ax,
                                         color=color.palette["darkGreen"],
                                         axvlines_kwds={'c': "white", "lw": 1},
                                         linewidth=subsetLinewidth,
                                         alpha=1,
                                         **parameters)


def histogram(ax, color, dataset=None, subset=None, numBins=6, **parameters):
    """
    Plot histograms of each feature side by side on ax with normalized subset 
    and dataset overlapping on common bins

    Args:
        ax (matplotlib ax): The axis to plot the histogram on
        color (Color object): A color object with the color palette to use
        dataset (sets.Dataset object, optional): The dataset to plot
        subset (sets.Subset object, optional): The subset to plot
        numBins (float): The number of bins to bin the dataset
    
    Raises: ValueError: If neither a dataset or subset are provided
    """
    
    if dataset is None and subset is None:
        raise ValueError("No dataset or subset specified")
    
    # Check if dataset is provided
    if dataset is not None:
        features = dataset.data.columns
        num_features = len(features)
        
        # Get the positions of each bar group
        bar_positions = np.arange(numBins * num_features, step=numBins)
        
        for i, feature in enumerate(features):
            # Plot the dataset histogram
            dataset_hist = np.histogram(dataset.data[feature], bins=numBins)
            dataset_heights = dataset_hist[0]
            
            # Adjust bar positions
            positions = bar_positions[i] + np.arange(numBins)
            
            ax.bar(positions, dataset_heights, width=1, 
                   color=color.palette["green"], alpha=0.5)
    
    # Check if subset is provided
    if subset is not None:
        features = subset.data.columns
        num_features = len(features)
        
        # Get the positions of each bar group
        bar_positions = np.arange(numBins * num_features, step=numBins)
        
        for i, feature in enumerate(features):
            # Calculate histogram of subset normalized by subset size
            subset_hist = np.histogram(subset.data[feature], bins=numBins)
            subset_heights = subset_hist[0] / len(subset.data) * len(dataset.data)
            
            # Adjust bar positions
            positions = bar_positions[i] + np.arange(numBins)
            
            ax.bar(positions, subset_heights, width=1, 
                   color=color.palette["darkGreen"], alpha=0.5)  # Increase alpha for better visibility