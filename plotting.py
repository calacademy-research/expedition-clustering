import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from cartopy.io.shapereader import Reader
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors

def plot_time_histogram(df, datetime_col='datetime', bins='auto'):
    """
    Plots a histogram of records over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a datetime column.
    - datetime_col (str): Name of the datetime column.
    - bins (str or int): Number of bins ('auto' for automatic binning or int for manual).
    """
    # Ensure the datetime column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram
    counts, bin_edges, _ = ax.hist(
        df[datetime_col], bins=bins, edgecolor='black', alpha=0.7
    )

    # Format x-axis for date readability
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Records')
    ax.set_title('Records Over Time')

    # Rotate x-ticks for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_geographical_positions(df, lat_col='lat', lon_col='lon', datetime_col='datetime', zoom='auto', cluster_line=False):
    """
    Plots geographical positions from a DataFrame overlaid on a detailed world map with configurable zoom, lat/lon labels, and filtered town names.

    Parameters:
    - df (pd.DataFrame): DataFrame containing latitude, longitude, and datetime columns.
    - lat_col (str): Name of the latitude column.
    - lon_col (str): Name of the longitude column.
    - datetime_col (str): Name of the datetime column.
    - zoom (str or float): Zoom level ('auto', 'california', 'us', 'world') or a numeric value to control lat/lon buffers inversely.
    """
    # Create the figure and axis using PlateCarree projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    df = df.sort_values(by=datetime_col)

    # Add map features (coastlines, countries, etc.)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)

    # Add rivers and roads
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.7, linewidth=0.5, label='Rivers')
    roads_shp = shpreader.natural_earth(category='cultural', name='roads', resolution='10m')
    roads = Reader(roads_shp).geometries()
    ax.add_geometries(roads, crs=ccrs.PlateCarree(),
                      edgecolor='brown', facecolor='none', linewidth=0.5, alpha=0.7, label='Roads')

    # Plot each point on the map
    scatter = ax.scatter(
        df[lon_col], df[lat_col],
        c=df[datetime_col].apply(lambda x: x.timestamp()),  # Color by datetime
        cmap='viridis', s=50, alpha=0.7,
        edgecolor='k', transform=ccrs.PlateCarree()
    )

    # Normalize datetime for color mapping
    norm = mcolors.Normalize(vmin=pd.to_datetime(df[datetime_col]).min().timestamp(),
                              vmax=pd.to_datetime(df[datetime_col]).max().timestamp())
    cmap = plt.cm.viridis

    # Plot each point on the map
    scatter = ax.scatter(
        df[lon_col], df[lat_col],
        c=pd.to_datetime(df[datetime_col]).astype(int) // 10**9,  # Color by datetime
        cmap='viridis', s=50, alpha=0.7,
        edgecolor='k', transform=ccrs.PlateCarree()
    )

    # Plot lines connecting consecutive points if cluster_line is True
    if cluster_line:
        for i in range(len(df) - 1):
            lon1, lat1 = df.iloc[i][lon_col], df.iloc[i][lat_col]
            lon2, lat2 = df.iloc[i + 1][lon_col], df.iloc[i + 1][lat_col]

            time1 = pd.to_datetime(df.iloc[i][datetime_col]).timestamp()
            time2 = pd.to_datetime(df.iloc[i + 1][datetime_col]).timestamp()

            # Get the average time and corresponding color
            avg_time = (time1 + time2) / 2
            avg_color = cmap(norm(avg_time))

            # Plot the line segment with the average color
            ax.plot(
                [lon1, lon2], [lat1, lat2],
                color=avg_color, linewidth=2, alpha=0.8, transform=ccrs.PlateCarree()
            )



    # Set the extent based on zoom level
    if zoom == 'auto':
        # Calculate bounds for zoom
        min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
        min_lon, max_lon = df[lon_col].min(), df[lon_col].max()

        # Buffer for visualization
        lat_buffer = (max_lat - min_lat) * 0.1
        lon_buffer = (max_lon - min_lon) * 0.1

        ax.set_extent([min_lon - lon_buffer, max_lon + lon_buffer,
                       min_lat - lat_buffer, max_lat + lat_buffer],
                      crs=ccrs.PlateCarree())

    elif isinstance(zoom, (int, float)):
        # Calculate bounds for zoom based on numeric value
        min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
        min_lon, max_lon = df[lon_col].min(), df[lon_col].max()

        lat_buffer = (max_lat - min_lat) / zoom
        lon_buffer = (max_lon - min_lon) / zoom

        ax.set_extent([min_lon - lon_buffer, max_lon + lon_buffer,
                       min_lat - lat_buffer, max_lat + lat_buffer],
                      crs=ccrs.PlateCarree())

    elif zoom == 'california':
        ax.set_extent([-125, -113, 32, 42], crs=ccrs.PlateCarree())  # California region

    elif zoom == 'us':
        ax.set_extent([-130, -60, 24, 50], crs=ccrs.PlateCarree())  # Continental US

    elif zoom == 'world':
        ax.set_global()

    else:
        raise ValueError("Invalid zoom option. Choose from 'auto', 'california', 'us', or 'world', or provide a numeric value.")


    # Add lat/lon gridlines and labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add town names (filtered by the current extent)
    if extent:
        towns_shp = shpreader.natural_earth(category='cultural', name='populated_places', resolution='10m')
        towns = Reader(towns_shp).records()
        for town in towns:
            town_name = town.attributes['NAME']  # Town name
            town_coords = town.geometry.centroid.coords[0]  # Get the centroid coordinates (lon, lat)
            lon, lat = town_coords

            # Only plot towns within the zoomed extent
            if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
                # Plot town marker
                ax.plot(lon, lat, marker='o', color='red', markersize=3, transform=ccrs.PlateCarree())

                # Annotate with the town name
                ax.text(lon + 0.2, lat, town_name,
                        fontsize=8, color='darkred', transform=ccrs.PlateCarree())

    # Add color bar and labels
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Time (Epoch)')

    ax.set_title('Geographical Positions Over Time')
    plt.legend(loc='lower left', fontsize=10)
    plt.show()
