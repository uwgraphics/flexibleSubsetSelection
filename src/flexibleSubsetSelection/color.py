# --- Imports ------------------------------------------------------------------

# Third party
import seaborn as sns


# --- Color --------------------------------------------------------------------


class Color:
    """
    Create and store color palettes and color bars for use in visualizations
    """

    def __init__(self, palette: dict | None = None):
        """
        Initialize the class with a custom or default palette

        Args:
            palette: dictionary of color names and color values
        """
        if palette is None:
            self.palette = {
                "green": "#8dd3c7",
                "darkGreen": "#338477",
                "orange": "#fb8072",
                "yellow": "#fdb462",
                "blue": "#8dadd3",
                "grey": "#eff0f2",
            }
        else:
            self.palette = palette

    def __getitem__(self, color):
        """Returns a color value from the palette directly."""
        return self.palette[color]

    def getPalette(self, names: list, colors: list) -> dict:
        """
        Create a custom palette for a categorical set by assigning colors from
        the default set to a category name.

        Args:
            names: List of category names to assign a color to
            colors: corresponding colors to assign to the names

        Returns: dictionary of names and colors

        Raises: ValueError if the names and color lists do not match
        """

        if len(names) != len(colors):
            raise ValueError("Names and colors lists must be the same length.")

        return {name: self.palette[color] for name, color in zip(names, colors)}

    def getGradientPalette(
        self, color: str, number: int = 6, type: str = "light"
    ) -> list:
        """
        Create a gradient palette based on a base color.

        Args:
            color: The base color to create a gradient from.
            number: Number of colors in the gradient palette.

        Returns: A list of colors in the gradient palette.

        Raises: ValueError if type is not light or dark.
        """
        if type == "light":
            return sns.light_palette(color=self.palette[color], n_colors=number)
        elif type == "dark":
            return sns.dark_palette(color=self.palette[color], n_colors=number)
        else:
            raise ValueError("Palette type unrecognized.")
