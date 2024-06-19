# --- Imports ------------------------------------------------------------------

# Standard libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# --- Color --------------------------------------------------------------------

class Color:
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


# --- Plots --------------------------------------------------------------------

def initialize(color, font="Times New Roman", family="sans-serif", size=42):
    """
    Initialize matplotlib settings global parameters for text and background
    """
    plt.rcParams["font.sans-serif"] = font
    plt.rcParams["font.family"] = family
    plt.rcParams["pdf.fonttype"] = size
    plt.rcParams["ps.fonttype"] = size
    plt.rcParams["axes.facecolor"] = color.palette["grey"]
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
                        **parameters)
