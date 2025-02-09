#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import scipy.ndimage as ndimage
import pandas as pd


# In[2]:


from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from metpy.plots import USCOUNTIES
import matplotlib.patheffects as pe


# In[3]:


import cmocean
import xarray as xr
import numpy as np
import pathlib
import sys
import os


# In[4]:


from dateparser import parse
from matplotlib import font_manager


# ### parses dates and run time

# In[5]:


date = parse('today GMT')

date = date.strftime("%Y%m%d")


# ### opens dset

# In[6]:


ds = xr.open_zarr("../data/ecmwf.zarr")


# ### precip

# In[7]:


precip = ds['tp'] * 39.3701


# ### timestamp selection

# In[8]:


# Extract the time values for the specified indices
timestamp_1 = precip.valid_time.isel(valid_time=0).dt.strftime("%b. %d").item()
timestamp_2 = precip.valid_time.isel(valid_time=40).dt.strftime("%b. %d").item()
timestamp_3 = precip.valid_time.isel(valid_time=52).dt.strftime("%b. %d").item()
timestamp_4 = precip.valid_time.isel(valid_time=64).dt.strftime("%b. %d").item()
timestamp_5 = precip.valid_time.isel(valid_time=-1).dt.strftime("%b. %d").item()

# Print or use the formatted timestamps
print(f"Timestamp for precip[0]: {timestamp_1}") # First time step
print(f"Timestamp for precip[40]: {timestamp_2}") # Day 5
print(f"Timestamp for precip[52]: {timestamp_3}") # Day 7
print(f"Timestamp for precip[64]: {timestamp_4}") # Day 10
print(f"Timestamp for precip[-1]: {timestamp_5}") # Day 15


# ### sets time

# In[9]:


conv = ds['valid_time'].dt.strftime('%Y-%m-%d %H')


# In[10]:


conv = conv.values


# In[11]:


step = ds['step']


# In[12]:


step = step.values


# In[13]:


step = step.astype('timedelta64[h]')


# In[14]:


valid_time = ds['valid_time'].dt.round('H')


# In[15]:


utc = valid_time.to_index()


# In[16]:


local = utc.tz_localize('GMT').tz_convert('America/New_York')


# In[17]:


local_time = local.strftime("%Y-%m-%d")


# In[18]:


formatted_dates = pd.to_datetime(local_time).strftime("%b. %d")


# ### plots

# In[19]:


lats = ds.variables['latitude'][:]  
lons = ds.variables['longitude'][:]


# ### wapo styling

# In[20]:


font_path = '../fonts/Franklin/FranklinITCStd-Black.otf'
font_properties = font_manager.FontProperties(fname=font_path)

font_path2 = '../fonts/Franklin/FranklinITCStd-Bold.otf'
font_properties2 = font_manager.FontProperties(fname=font_path2)

font_path3 = '../fonts/Franklin/FranklinITCStd-Light.otf'
font_properties3 = font_manager.FontProperties(fname=font_path3, size=24)

font_path4 = '../fonts/Franklin/FranklinITCStd-Light.otf'
font_properties4 = font_manager.FontProperties(fname=font_path4, size=20)


# In[21]:


state_centers = {
    "AK": (-152.0, 65.3), "AL": (-86.9, 32.8), "AZ": (-111.7, 34.1), "AR": (-92.3, 34.8),
    "CA": (-119.4, 36.7), "CO": (-105.5, 39.5), "CT": (-72.7, 41.6), "DC": (-77.0, 38.9),
    "DE": (-75.5, 38.9), "FL": (-81.6, 27.9), "GA": (-83.7, 33.3), "HI": (-157.8, 20.8),
    "IA": (-93.3, 41.8), "ID": (-114.5, 44.3), "IL": (-89.2, 40.0), "IN": (-86.3, 39.8),
    "KS": (-98.5, 38.5), "KY": (-84.3, 37.6), "LA": (-92.5, 31.1), "MA": (-71.9, 42.3),
    "MD": (-76.6, 39.4), "ME": (-69.3, 45.3), "MI": (-84.8, 43.3), "MN": (-94.4, 45.7),
    "MO": (-92.3, 38.5), "MS": (-89.7, 32.7), "MT": (-110.3, 46.8), "NC": (-79.0, 35.6),
    "ND": (-99.9, 47.5), "NE": (-98.9, 41.3), "NH": (-71.5, 43.4), "NJ": (-74.5, 40.5),
    "NM": (-106.2, 34.5), "NY": (-75.4, 42.5), "NV": (-116.4, 39.1), "OH": (-82.8, 40.4),
    "OK": (-97.5, 35.6), "OR": (-120.5, 44.0), "PA": (-77.3, 40.9), "RI": (-71.5, 41.7),
    "SC": (-81.0, 33.8), "SD": (-99.5, 44.3), "TN": (-86.5, 35.9), "TX": (-99.0, 31.0),
    "UT": (-111.7, 39.8), "VA": (-78.3, 37.5), "VT": (-72.6, 44.0), "WA": (-120.6, 47.4),
    "WI": (-89.6, 44.5), "WV": (-80.6, 38.6), "WY": (-107.3, 43.0)
}

# Eastern region
eastern_states = [
    "AL", "AR", "CT", "DC", "DE", "FL", "GA", "IL", "IN", "IA", "KY", "LA", "ME", "MA",
    "MD", "MI", "MN", "MS", "MO", "NC", "NH", "NJ", "NY", "OH", "PA", "RI", "SC", "TN",
    "VA", "VT", "WI", "WV"
]

# Western region
western_states = [
    "AZ", "CA", "CO", "ID", "KS", "MT", "NV", "NM", "ND", "NE", "OK", "OR", "SD", "TX",
    "UT", "WA", "WY"
]

# Central region
central_states = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MA", "MD", "MI", "MN", "MS", "MO", "MT", "NC", "ND", "NE",
    "NV", "NH", "NJ", "NM", "NY", "OK", "OR", "OH", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VA", "VT", "WA", "WI", "WV", "WY"
]

# North America region
north_america_states = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MA", "MD", "MI", "MN", "MS", "MO", "MT", "NC", "ND",
    "NE", "NV", "NH", "NJ", "NM", "NY", "OK", "OR", "OH", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
]

# Great Lakes region
great_lakes_states = [
    "CT", "DC", "DE", "IA", "IL", "IN", "KY", "MA", "MD", "ME", "MI", "MO", "NH", "NJ", 
    "NY", "OH", "PA", "RI", "VT", "VA", "WI", "WV"
]

# Deep South region
deep_south_states = [
    "AL", "AR", "DC", "DE", "FL", "GA", "IL", "IN", "KS", "KY", "LA", "MD", "MS", "MO",
    "NC", "OH", "OK", "SC", "TN", "TX", "VA", "WV"
]


# ### 15 day precip

# In[22]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from metpy.plots import USCOUNTIES
import shapefile
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

# Set projection type and region
projection_type = "albers"  # Albers Equal Area Projection
region = "central"  # Change to 'eastern' or 'western' as needed

# Define projection parameters
usa_projections = {
    "central": {"central_longitude": 265, "central_latitude": 42},
    "eastern": {"central_longitude": 283, "central_latitude": 42},
    "western": {"central_longitude": 240, "central_latitude": 42},
    "great_lakes": {"central_longitude": 275, "central_latitude": 43},
    "deep_south": {"central_longitude": 270, "central_latitude": 37},
}

# Define extent options
zoom_options = {
    "central": [-125, -65, 24, 49],
    "western": [-125, -95, 25, 49],
    "eastern": [-95, -65, 25, 49],
    "great_lakes": [-95, -68, 37, 49],
    "deep_south": [-105, -77, 25, 40],
}

def create_plot(region, output_file, title_offset):
    # Set the projection and extent for the given region
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5))
    fig, ax = plt.subplots(figsize=(25, 15), subplot_kw={"projection": proj})
    ax.set_extent(zoom_options[region], crs=ccrs.PlateCarree())

    # Add map features
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none'
    )
    # Add map features with improved settings
    ax.add_feature(cfeature.LAND, color='#F5F5F5', edgecolor='k', alpha=0.8)
    ax.coastlines(resolution='50m', color='dimgray', linewidth=1, zorder=104)
    ax.add_feature(cfeature.BORDERS, edgecolor='dimgray')
    lakes = cfeature.NaturalEarthFeature(
        'physical', 'lakes', '50m',
        edgecolor='dimgray', facecolor='white'
    )
    ax.add_feature(lakes, alpha=1, linewidth=0.5, zorder=100)
    ax.add_feature(states_provinces, edgecolor='dimgray')
    ax.add_feature(cfeature.OCEAN, color='#FFFFFF', alpha=1, zorder=103)

    # Titles
    plt.suptitle(
        'Total precipitation (in)', 
        fontsize=36, 
        color='k', 
        fontproperties=font_properties2, 
        y=0.94 + title_offset  
    )
    
    x_value = 0.47 if region in ["eastern", "western"] else 0.48
    
    plt.title(
        f'For the 15 days ending on {timestamp_5}', 
        loc='center', 
        fontsize=30, 
        color='k', 
        fontproperties=font_properties3,
        x=x_value,
        pad=60
    )

    # Define levels and colors
    levels = [0.1, 0.5, 1, 2, 4, 8, 100]  # One more level to match number of colors
    colors = ['#d4edc9', '#b2d6a0', '#91c078', '#6ea951', '#499327', '#38711e', '#285115']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=levels, ncolors=len(colors), extend='neither')
    
    # Plot the data
    data = ax.pcolormesh(lons, lats, precip[-1], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    # Create legend
    legend_labels = ['0.1-0.50', '0.50-1', '1-2', '2-4', '4-8', '8+']
    patches = [mpatches.Patch(color=cmap(i / (len(levels) - 1)), label=legend_labels[i]) for i in range(len(legend_labels))]
    
    # Adjust legend position for "great_lakes" region
    legend_y_anchor = 0.88 if region == "great_lakes" else 0.83 + title_offset  

    fig.legend(
        handles=patches,
        loc='lower center',
        ncol=6,
        bbox_to_anchor=(0.51, legend_y_anchor),
        frameon=False,
        prop=font_properties4
    )

    # Remove the black border surrounding the map
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add state labels
    for state, (lon, lat) in state_centers.items():
        if (
            (region == "central" and state in central_states) or
            (region == "eastern" and state in eastern_states) or
            (region == "western" and state in western_states) or
            (region == "deep_south" and state in deep_south_states) or
            (region == "great_lakes" and state in great_lakes_states) or
            (region == "california" and state in california_locations)
        ):
            ax.text(lon, lat, state, transform=ccrs.PlateCarree(), fontproperties=font_properties3,
                    fontsize=20, color='k', ha='center', va='center', zorder=105, alpha=0.5)

    # Save the plot
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(fig)

# Define parameters for each region
regions = {
    "central": {"file": f"../imagery/{date}/precip_central-15day.png", "title_offset": 0},
    "eastern": {"file": f"../imagery/{date}/precip_eastern-15day.png", "title_offset": 0.05},
    "western": {"file": f"../imagery/{date}/precip_western-15day.png", "title_offset": 0.05},
    "deep_south": {"file": f"../imagery/{date}/precip_deep_south-15day.png", "title_offset": 0.05},
    "great_lakes": {"file": f"../imagery/{date}/precip_great_lakes-15day.png", "title_offset": 0.05},
}

# Loop through the regions and create the plots
for region, params in regions.items():
    create_plot(region, params["file"], params["title_offset"])


# ### 5-day precip

# In[23]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from metpy.plots import USCOUNTIES
import shapefile
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

# Set projection type and region
projection_type = "albers"  # Albers Equal Area Projection
region = "central"  # Change to 'eastern' or 'western' as needed

# Define projection parameters
usa_projections = {
    "central": {"central_longitude": 265, "central_latitude": 42},
    "eastern": {"central_longitude": 283, "central_latitude": 42},
    "western": {"central_longitude": 240, "central_latitude": 42},
    "great_lakes": {"central_longitude": 275, "central_latitude": 43},
    "deep_south": {"central_longitude": 270, "central_latitude": 37},
}

# Define extent options
zoom_options = {
    "central": [-125, -65, 24, 49],
    "western": [-125, -95, 25, 49],
    "eastern": [-95, -65, 25, 49],
    "great_lakes": [-95, -68, 37, 49],
    "deep_south": [-105, -77, 25, 40],
}

def create_plot(region, output_file, title_offset):
    # Set the projection and extent for the given region
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5))
    fig, ax = plt.subplots(figsize=(25, 15), subplot_kw={"projection": proj})
    ax.set_extent(zoom_options[region], crs=ccrs.PlateCarree())

    # Add map features
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none'
    )
    # Add map features with improved settings
    ax.add_feature(cfeature.LAND, color='#F5F5F5', edgecolor='k', alpha=0.8)
    ax.coastlines(resolution='50m', color='dimgray', linewidth=1, zorder=104)
    ax.add_feature(cfeature.BORDERS, edgecolor='dimgray')
    lakes = cfeature.NaturalEarthFeature(
        'physical', 'lakes', '50m',
        edgecolor='dimgray', facecolor='white'
    )
    ax.add_feature(lakes, alpha=1, linewidth=0.5, zorder=100)
    ax.add_feature(states_provinces, edgecolor='dimgray')
    ax.add_feature(cfeature.OCEAN, color='#FFFFFF', alpha=1, zorder=103)

    # Titles
    plt.suptitle(
        'Total precipitation (in)', 
        fontsize=36, 
        color='k', 
        fontproperties=font_properties2, 
        y=0.94 + title_offset  
    )
    
    x_value = 0.47 if region in ["eastern", "western"] else 0.48
    
    plt.title(
        f'For the 5 days ending on {timestamp_2}', 
        loc='center', 
        fontsize=30, 
        color='k', 
        fontproperties=font_properties3,
        x=x_value,
        pad=60
    )

    # Define levels and colors
    levels = [0.1, 0.5, 1, 2, 4, 8, 100]  # One more level to match number of colors
    colors = ['#d4edc9', '#b2d6a0', '#91c078', '#6ea951', '#499327', '#38711e', '#285115']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=levels, ncolors=len(colors), extend='neither')
    
    # Plot the data
    data = ax.pcolormesh(lons, lats, precip[40], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    # Create legend
    legend_labels = ['0.1-0.50', '0.50-1', '1-2', '2-4', '4-8', '8+']
    patches = [mpatches.Patch(color=cmap(i / (len(levels) - 1)), label=legend_labels[i]) for i in range(len(legend_labels))]
    
    # Adjust legend position for "great_lakes" region
    legend_y_anchor = 0.88 if region == "great_lakes" else 0.83 + title_offset  

    fig.legend(
        handles=patches,
        loc='lower center',
        ncol=6,
        bbox_to_anchor=(0.51, legend_y_anchor),
        frameon=False,
        prop=font_properties4
    )

    # Remove the black border surrounding the map
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add state labels
    for state, (lon, lat) in state_centers.items():
        if (
            (region == "central" and state in central_states) or
            (region == "eastern" and state in eastern_states) or
            (region == "western" and state in western_states) or
            (region == "deep_south" and state in deep_south_states) or
            (region == "great_lakes" and state in great_lakes_states) or
            (region == "california" and state in california_locations)
        ):
            ax.text(lon, lat, state, transform=ccrs.PlateCarree(), fontproperties=font_properties3,
                    fontsize=20, color='k', ha='center', va='center', zorder=105, alpha=0.5)

    # Save the plot
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(fig)

# Define parameters for each region
regions = {
    "central": {"file": f"../imagery/{date}/precip_central-5day.png", "title_offset": 0},
    "eastern": {"file": f"../imagery/{date}/precip_eastern-5day.png", "title_offset": 0.05},
    "western": {"file": f"../imagery/{date}/precip_western-5day.png", "title_offset": 0.05},
    "deep_south": {"file": f"../imagery/{date}/precip_deep_south-5day.png", "title_offset": 0.05},
    "great_lakes": {"file": f"../imagery/{date}/precip_great_lakes-5day.png", "title_offset": 0.05},
}

# Loop through the regions and create the plots
for region, params in regions.items():
    create_plot(region, params["file"], params["title_offset"])


# In[ ]:




