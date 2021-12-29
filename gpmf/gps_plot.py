import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import numpy
import pandas


from .gps import extract_gps_blocks, parse_gps_block


LATLON = "EPSG:4326"
LAMBERT93 = "EPSG:2154"


def to_dataframe(gps_data_blocks):
    """Convert a sequence of GPSData into pandas dataframe.

    Parameters
    ----------
    gps_data_blocks: seq of GPSData
        A sequence of GPSData objects
    Returns
    -------
    df_gps: pandas.DataFrame
        The output dataframe
    """
    df_blocks = []
    for i, block in enumerate(gps_data_blocks):
        df_block = pandas.DataFrame()
        df_block["latitude"] = block.latitude
        df_block["longitude"] = block.longitude
        df_block["altitude"] = block.altitude
        df_block["time"] = block.timestamp
        df_block["speed_2d"] = block.speed_2d
        df_block["speed_3d"] = block.speed_3d
        df_block["precision"] = block.precision
        df_block["fix"] = block.fix
        df_block["block_id"] = i
        df_blocks.append(df_block)

    return pandas.concat(df_blocks)


class GPSPlotter():

    def __init__(self, stream) -> None:
        self.stream = stream
        self.geodataframe = None

    def filter_outliers(self, x):
        """Filter outliers based on 0.01 and 0.99 quantiles"""
        q01, q50, q99 = numpy.quantile(x, q=[0.01, 0.5, 0.99])
        return (q50 - (1.1 * (q50 - q01)) < x) & (x < q50 + (1.1 * (q99 - q50)))

    def get_bounding_box_geojson(self):
        """Return bounding box for whole route as a GeoJson (WGS 84)"""
        df = self.build_geodataframe()
        minx, miny, maxx, maxy = df.total_bounds
        return {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [minx, maxy],
                    [maxx, maxy],
                    [maxx, miny],
                    [minx, miny],
                ]]
            }
        }

    def get_bounding_box(self):
        """
        Return bounding box for whole route as array (WGS 84)
        (minx, miny, maxx, maxy)
        """
        df = self.build_geodataframe()
        return df.total_bounds

    def build_geodataframe(self, first_only=False, precision_max=3.0) -> gpd.GeoDataFrame:
        """Build a GeoDataFrame from stream"""
        if self.geodataframe is not None:
            return self.geodataframe

        gps_data_blocks = map(parse_gps_block, extract_gps_blocks(self.stream))

        if first_only:
            latlon = numpy.array([
                [b.latitude[0], b.longitude[0]]
                for b in gps_data_blocks
                if b.precision < precision_max
            ])
        else:
            latlon = numpy.vstack([
                numpy.vstack([b.latitude, b.longitude]).T
                for b in gps_data_blocks
                if b.precision < precision_max
            ])

        y, x = latlon.T

        mask = self.filter_outliers(x) & self.filter_outliers(y)

        df = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x[mask], y[mask], crs=LATLON)
        )

        # FIXME:
        # /mnt/lvm/backup/share/GoPro-pytube/2021-02-26/gh010039.mp4 already exists, skipping
        # /home/alexandre/pytube-server/pyenv/lib/python3.8/site-packages/geopandas/plotting.py:681: UserWarning: The GeoDataFrame you are attempting to plot is empty. Nothing has been displayed.
        if df.empty():
            raise ValueError("GeoDataFrame is empty")

        self.geodataframe = df
        return self.geodataframe

    def plot_gps_trace(
        self,
        min_tile_size=10,
        map_provider=None,
        zoom=12,
        figsize=(10, 10),
        proj_crs=LAMBERT93,
        color="tab:red"
    ):
        """ Plot a (lat, lon) coordinates on a Map

        Parameters
        ----------
        latlon: numpy.ndarray
            Array of (latitude, longitude) coordinates
        min_tile_size: int, optional (default=10)
            Minimum size of the map in km
        map_provider: dict
            Dictionnary describing a map provider as given by `contextly.providers`. If None
            `contextily.providers.GeoportailFrance["maps"]` is used.
        zoom: int, optional (default=12)
            The zoom level used.
        figsize: tuple of int, optional (default=(10, 10))
            The matplotlib figure size
        proj_crs: str or geopandas.CRS object, optional (default="EPSG:2154")
            The projection system used to compute distances on the map. The default value
            corresponds to the Lambert 93 system.
        color: str, optional (default="tab:red")
            The color used to plot the track.
        """

        # FIXME: close plot
        # /home/alexandre/pytube-server/pyenv/src/gpmf/gpmf/gps_plot.py:149: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
        #   plt.figure(figsize=figsize)

        if map_provider is None:
            map_provider = ctx.providers.OpenTopoMap

        min_tile_size *= 1000

        df = self.geodataframe

        plt.figure(figsize=figsize)
        ax = plt.gca()

        df.to_crs(proj_crs).plot(ax=ax, color=color)

        xmin, xmax = plt.xlim()
        dx = xmax - xmin

        if dx < min_tile_size:
            xc = 0.5 * (xmin + xmax)
            xmin = xc - min_tile_size / 2
            xmax = xc + min_tile_size / 2
            plt.xlim(xmin, xmax)

        ymin, ymax = plt.ylim()
        dy = ymax - ymin

        if dy < min_tile_size:
            yc = 0.5 * (ymin + ymax)
            ymin = yc - min_tile_size / 2
            ymax = yc + min_tile_size / 2
            plt.ylim(ymin, ymax)

        ctx.add_basemap(ax, source=map_provider, zoom=zoom, crs=proj_crs)
        ax.set_axis_off()

    def plot(
        self,
        first_only=False,
        min_tile_size=10,
        map_provider=None,
        zoom=12,
        figsize=(10, 10),
        proj_crs=LAMBERT93,
        output_path=None,
        precision_max=3.0,
        color="tab:red"
    ):
        """ Plot GPS data from a string on a map.

            Parameters
            ----------
            min_tile_size: int, optional (default=10)
                Minimum size of the map in km
            map_provider: dict
                Dictionnary describing a map provider as given by `contextly.providers`. If None
                `contextily.providers.GeoportailFrance["maps"]` is used.
            zoom: int, optional (default=12)
                The zoom level used.
            figsize: tuple of int, optional (default=(10, 10))
                The matplotlib figure size
            proj_crs: str or geopandas.CRS object, optional (default="EPSG:2154")
                The projection system used to compute distances on the map. The default value
                corresponds to the Lambert 93 system.
            color: str, optional (default="tab:red")
                The color used to plot the track.
        """
        self.build_geodataframe(first_only, precision_max)

        self.plot_gps_trace(
            min_tile_size=min_tile_size,
            map_provider=map_provider,
            zoom=zoom, figsize=figsize,
            proj_crs=proj_crs, color=color
        )
        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path)
