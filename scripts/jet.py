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


import cmocean
import xarray as xr
import pathlib
import wget


# In[32]:


from dateparser import parse
from matplotlib import font_manager


# ### parses dates and run time

# In[7]:


date = parse('today GMT')

date = date.strftime("%Y%m%d")


# ### defines fig path

# In[8]:


fig_path = "../imagery/{}/jet".format(date)


# In[9]:


fig_path = pathlib.Path(fig_path)


# In[10]:


if fig_path.exists() == False:
    fig_path.mkdir(parents = True)


# ### opens dset, computes jet stream wind speed

# In[15]:


forecast = xr.open_dataset('../data/ecmwf.nc')


# In[16]:


forecast['v'] = forecast['v'].sel(isobaricInhPa=250)


# In[17]:


forecast['u'] = forecast['u'].sel(isobaricInhPa=250)


# In[18]:


forecast['gh'] = forecast['gh'].sel(isobaricInhPa=250)


# In[19]:


ws = (forecast['u']*forecast['u'] + forecast['v']*forecast['v'])**(1/2)
ws = ws*2.237 # Converts to mph


# In[20]:


ws = ws.sortby('valid_time')


# ### drops wind speeds less than 50 mph

# In[21]:


new_ws = ws.where(ws > 50, drop=True)


# In[22]:


gh = forecast['gh']
gh = gh.sortby('valid_time')


# In[23]:


lats = forecast.variables['latitude'][:]  
lons = forecast.variables['longitude'][:]


# ### sets time

# In[24]:


conv = ws['valid_time'].dt.strftime('%Y-%m-%d %H')


# In[25]:


conv = conv.values


# In[26]:


valid_time = ws['valid_time'].dt.round('H')


# In[27]:


utc = valid_time.to_index()


# In[28]:


local = utc.tz_localize('GMT').tz_convert('America/New_York')


# In[29]:


local_time = local.strftime("%Y-%m-%d")


# In[30]:


formatted_dates = pd.to_datetime(local_time).strftime("%b. %d")
formatted_dates = formatted_dates.str.replace(r"(?<=\.\s)0", "", regex=True)


# ### plots

# In[31]:


lon2 = new_ws['longitude']
lat2 = new_ws['latitude']
lons2, lats2 = np.meshgrid(lon2, lat2)


# ### jet

# ### wapo styling

# In[33]:


font_path = '../fonts/Franklin/FranklinITCStd-Black.otf'
font_properties = font_manager.FontProperties(fname=font_path)

font_path2 = '../fonts/Franklin/FranklinITCStd-Bold.otf'
font_properties2 = font_manager.FontProperties(fname=font_path2)

font_path3 = '../fonts/Franklin/FranklinITCStd-Light.otf'
font_properties3 = font_manager.FontProperties(fname=font_path3, size=24)


# In[34]:


for i in range(len(forecast.valid_time)):

    # Set projection type and region
    projection_type = "near_persp"  # Options: "ortho", "albers", "near_persp"
    region = "central"  # Change to 'eastern' or 'western' as needed

    # Define the projection parameters for different U.S. regions
    usa_projections = {
        "central": {"central_longitude": 265, "central_latitude": 42}, # For ortho
        "eastern": {"central_longitude": 283, "central_latitude": 42}, # For ortho
        "western": {"central_longitude": 240, "central_latitude": 42}, # For ortho
        "southern_patagonia": {"central_longitude": 293, "central_latitude": -50}, # For ortho
        "central": {"central_longitude": 265, "central_latitude": 42, "satellite_height": 4000000}, # For near_persp
        "eastern": {"central_longitude": 283, "central_latitude": 42, "satellite_height": 4000000}, # For near_persp
        "western": {"central_longitude": 240, "central_latitude": 42, "satellite_height": 4000000}, # For near_persp
    }

    # Define zoom options (extent for different regions)
    zoom_options = {
        "central": [-125, -65, 24, 50],  # Central U.S.
        "western": [-125, -95, 25, 50],  # Western U.S.
        "eastern": [-95, -65, 25, 50],   # Eastern U.S.
    }

    # Set the projection based on the selected type
    if projection_type == "ortho":
        projection_params = usa_projections[region]
        proj = ccrs.Orthographic(
            central_longitude=projection_params["central_longitude"],
            central_latitude=projection_params["central_latitude"]
        )
    elif projection_type == "albers":
        proj = ccrs.AlbersEqualArea(
            central_longitude=-96,
            central_latitude=23,
            standard_parallels=(29.5, 45.5)
        )
    elif projection_type == "near_persp":
        projection_params = usa_projections[region]
        proj = ccrs.NearsidePerspective(
            central_longitude=projection_params["central_longitude"],
            central_latitude=projection_params["central_latitude"],
            satellite_height=projection_params["satellite_height"]
        )

    # Create the figure and axes with the chosen projection
    fig, ax = plt.subplots(figsize=(25, 15), subplot_kw={"projection": proj})
    if projection_type == "albers":
        ax.set_extent(zoom_options[region], crs=ccrs.PlateCarree())
        
    # Create the figure and axes with the selected projection
    if projection_type == "ortho":
        fig, ax = plt.subplots(figsize=(25, 15), subplot_kw={"projection": proj})

    # Add map features
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    # Add map features with improved settings
    ax.add_feature(cfeature.LAND, color='#F5F5F5', edgecolor='k', alpha=0.8)
    ax.coastlines(resolution='50m', color='dimgray', linewidth=1)
    ax.add_feature(cfeature.BORDERS, edgecolor='dimgray')
    lakes = cfeature.NaturalEarthFeature(
        'physical', 'lakes', '50m',
        edgecolor='dimgray', facecolor='white'
    )
    ax.add_feature(lakes, alpha=0.5, linewidth=0.5, zorder=100)
    ax.add_feature(states_provinces, edgecolor='dimgray')

    # Plot jet stream with 'viridis' colormap
    data = ax.pcolormesh(
        lons2, lats2, new_ws[i],
        vmin=50, vmax=200,
        cmap='CMRmap_r',
        transform=ccrs.PlateCarree()
    )

    # Add contour lines for atmospheric height
    cs = ax.contour(
        lons, lats, forecast['gh'][i],
        levels=np.arange(9000, 12000, 120),
        colors='k',
        linewidths=0.5,
        transform=ccrs.PlateCarree()
    )
    
    # Add contour labels with font properties
    contour_labels = ax.clabel(cs, fontsize=10, inline=True, fmt='%i', colors='k')

    # Apply font properties to each label
    for label in contour_labels:
        label.set_fontproperties(font_properties3)

    # Add titles
    plt.suptitle(
        f'Jet stream wind speed (mph) & atmospheric heights (m), {formatted_dates[i]}',
        fontsize=48, color='k', fontproperties=font_properties2, y=0.31, x=0.51
    )

    # Add a color bar
    cbar = plt.colorbar(data, ax=ax, orientation="horizontal", ticks=[50,70,90,110,130,150,170,190], pad=-1.82, aspect=20, shrink=0.5)
    cbar.ax.tick_params(labelsize=30, labelcolor="k")
    cbar.ax.set_facecolor([1, 1, 1, 0])
    
    # Apply font properties (e.g., weight and family) to the tick labels
    for label in cbar.ax.get_xticklabels():
        label.set_fontproperties(font_properties3)  # Apply font properties to each label
        label.set_size(30)  # Set the font size separately

    # Remove the black border surrounding the map
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save the figure
    plt.savefig(fig_path.joinpath(f"jet_{i}_{region}.png"), facecolor='white', bbox_inches='tight', dpi=100)

    # Close the figure to free up memory
    plt.close(fig)


# In[ ]:




