"""

This module provides additional functions to the kml-builder of simplekml. These functions allow the addition of
spatial bar plots or markers with a table connected to them.

Jurjen de Jong, 20-11-2020

"""

from simplekml import Kml
import simplekml
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from typing import Collection, Tuple


class KmlCharts(Kml):
    cmap = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set default colormap
        self.set_cmap()

        self.folders = {}

    def set_cmap(self, vmin=10, vmax=20, cmap='Reds', nan_color='ff000000'):
        """
        Set a colormap for the kml file which is used for coloring the bar charts

        :param vmin: start of color range
        :param vmax: end of color range
        :param cmap: colormap
        :param nan_color: color to choose when value is Nan
        """
        self.cmap = KmlColorMap(vmin, vmax, cmap, nan_color)

    def get_folder(self, name):
        """
        Get folder, or create if it does not yet exist

        :param name: name of folder
        :return: (new) folder handle
        """
        if name is None:
            return self
        elif name in self.folders:
            return self.folders[name]
        else:
            folder = self.newfolder(name=name)
            self.folders[name] = folder

            return folder

    def chart_stacked_bar(self, xy: Tuple[float, float], heights: Collection, values: Collection,
                          colors: Collection = None, shape='circle', radius=0.01, folder=None, name=None):
        """
        Build stacked bar chart

        :param xy: centerpoint (x, y)
        :param heights: list of heights (cumulative!)
        :param values: list of values (or names)
        :param colors: None, or list of colors. If none, uses cmap
        :param shape: 'circle', or 'square'
        :param radius: radius of bar
        :param folder: None, or name of folder
        :param name: name of feature, will be appended with value
        :return: list of kml-objects
        """

        # If colors is empty, use the colormap
        if not colors:
            colors = [self.cmap.get_color(v) for v in values]

        # Every added polygon is a bit smaller than the previous
        d_radius = 0.00001
        radius_range = radius - np.arange(0, d_radius * len(values), d_radius)

        pols = []
        for radius, color, height, value in zip(radius_range, colors, heights, values):
            name_bar = f'{name}+ {value}'
            pol = self.chart_bar(xy, height, color, radius, shape, folder=folder, name=name_bar)
            pols.append(pol)
        return pols

    def chart_bar(self, xy, height, color, radius=0.01, shape='circle', name=None, folder=None):
        """
        Create simple bar


        :param xy: centerpoint (x, y)
        :param height:
        :param color:
        :param radius: radius of bar
        :param shape: 'circle', or 'square'
        :param name: name of feature
        :param folder: None, or name of folder
        :return: kml-object
        """
        if shape == 'circle':
            points = build_shape_circle(xy, height, radius=radius)
        elif shape == 'square':
            points = build_shape_square(xy, height, radius=radius)
        else:  # Just make a point..
            points = [(xy[0], xy[1], height)]

        folder = self.get_folder(folder)
        pol = folder.newpolygon()

        pol.name = name
        pol.outerboundaryis = points
        pol.extrude = 1  # Floating polygon to earth
        pol.altitudemode = 'absolute'

        # Apply styling
        pol.style.linestyle.width = 0
        pol.style.polystyle.color = color
        return pol

    def chart_marker_table(self, xy, table, markerstyle=None, name=None):
        """
        Create marker with balloon with table (or other html).
        Also allows pandas (will do .to_html())

        :param xy: markerlocation (x, y)
        :param table: html, pandas dataframe or pandas series
        :param markerstyle: if given, applying this style to the marker
        :param name: name of feature and header in balloon
        :return: kml-object
        """
        p = self.newpoint(coords=[xy], name=name)

        if markerstyle is not None:
            p.style = markerstyle

        if isinstance(table, pd.DataFrame) or isinstance(table, pd.Series):
            table = table.to_html()

        if name is not None:
            table = f"<h2>{name}</h2>" + table

        p.style.balloonstyle.text = table

        return p

    def chart_marker_figure(self, xy, figurepath, markerstyle=None, name=None):
        """
        Create marker with balloon with figure.

        :param xy: marker location (x, y)
        :param figurepath: path to figure, will be added to kmz
        :param markerstyle: if given, applying this style to the marker
        :param name: name of feature and header in balloon
        :return: kml-object
        """
        p = self.newpoint(coords=[xy], name=name)

        if markerstyle is not None:
            p.style = markerstyle

        internal_figurepath = self.addfile(figurepath)
        html = f'<img src="{internal_figurepath}" alt="picture" height="300" align="left" />'

        if name is not None:
            html = f"<h2>{name}</h2>" + html

        p.style.balloonstyle.text = html
        return p

    def chart_legend(self, legend_figure):
        """
        Add overlay legend in top left corner

        :param legend_figure: path to figure
        :return: kml-object
        """
        # NOT YET TESTED
        screen = self.newscreenoverlay(name='Legend')

        # TODO: automatically make figure of legend

        screen.icon.href = legend_figure
        screen.overlayxy = simplekml.OverlayXY(x=0, y=1, xunits=simplekml.Units.fraction,
                                               yunits=simplekml.Units.fraction)
        screen.screenxy = simplekml.ScreenXY(x=15, y=15, xunits=simplekml.Units.insetpixels,
                                             yunits=simplekml.Units.insetpixels)
        screen.size.x = -1
        screen.size.y = -1
        screen.size.xunits = simplekml.Units.fraction
        screen.size.yunits = simplekml.Units.fraction
        return screen


class KmlColorMap:

    def __init__(self, vmin, vmax, cmap, nan_color):
        cmap = get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)

        self.norm_cmap = lambda value: cmap(norm(value))
        self.nan_color = nan_color

    def get_color(self, value):
        if np.isnan(value):
            return self.nan_color

        rgb_color = self.norm_cmap(value)
        rgb_color = [np.int(c * 255) for c in rgb_color]

        kml_color = simplekml.Color.rgb(*rgb_color)
        return kml_color


# These functions are not computing the circle too nice...
def pol2cart(rho, phi):
    """
    Convert polar to cartesian

    rho: radius (degree)
    phi: angle (rad)
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def build_shape_circle(centerpoint, height, radius=0.01, n_coords=20):
    """

    :param centerpoint:
    :param height:
    :param radius:
    :param n_coords:
    :return:
    """

    phi = np.linspace(0, 2 * np.pi, n_coords + 1)
    x, y = pol2cart(radius, phi)

    x = [xp + centerpoint[0] for xp in x]
    y = [yp + centerpoint[1] for yp in y]

    # Combine points to array, round values, add height
    points = [(xp, yp, height) for xp, yp in zip(x, y)]

    return points


def build_shape_square(centerpoint, height, radius):
    """

    :param centerpoint:
    :param height:
    :param radius:
    :return:
    """

    xmin = centerpoint[0] - radius
    xmax = centerpoint[0] + radius

    ymin = centerpoint[1] - radius
    ymax = centerpoint[1] + radius

    points = [(xmin, ymin, height),
              (xmin, ymax, height),
              (xmax, ymax, height),
              (xmax, ymin, height),
              (xmin, ymin, height)]

    return points


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    kml = KmlCharts()

    # Create stacked bar based on cmap
    kml.set_cmap(vmin=10, vmax=20, cmap='Reds', nan_color='ff000000')
    kml.chart_stacked_bar(xy=(5, 52), heights=[500, 1000, 1500, 2000], values=[5, 25, 6, 14], folder='stacked_bars',
                          name=f'Stacked bar at location X1')
    kml.chart_stacked_bar(xy=(5.2, 52), heights=[200, 1400, 1500, 5000], values=[21, 12, 44, 14], folder='stacked_bars',
                          name=f'Stacked bar at location X2')

    # Create stacked bar with given colors
    kml.chart_stacked_bar(xy=(5, 52.5), heights=[500, 1000, 1500], values=['bins1', 'bin2', 'bin3'],
                          colors=['ff00ffff', 'ffff00ff', 'ffffff00'], folder='stacked_bars',
                          name=f'Stacked bar with labels')

    # Create simple bar
    kml.chart_bar(xy=(5, 52.4), height=1000, color='ff00ffff', folder='bars', name=f'Bar at location X3')

    # Example of changing folder styling
    kml.folders['bars'].style.liststyle.listitemtype = 'checkHideChildren'

    # Add figure
    kml.chart_marker_figure(xy=(6, 51), figurepath='Capture.PNG', name='Figure')

    # Create dummy table and add to kml
    df = pd.DataFrame({'A': 1.,
                       'B': pd.Timestamp('20130102'),
                       'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                       'D': np.array([3] * 4, dtype='int32'),
                       'E': pd.Categorical(["test", "train", "test", "train"]),
                       'F': 'foo'})

    kml.chart_marker_table(xy=(6.3, 51), table=df, name='Table')

    # Save to file
    outputfile = Path('testfile.kmz')
    kml.savekmz(outputfile)
