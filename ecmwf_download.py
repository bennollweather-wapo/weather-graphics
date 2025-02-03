#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dateparser import parse
import os
import wget
import glob


# In[2]:


import os
import shutil

# Define the folder name
folder_name = "data"

# Get the current working directory
cwd = os.getcwd()

# Create the full path
folder_path = os.path.join(cwd, folder_name)

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Change the working directory to the new folder
os.chdir(folder_path)

# Print the new working directory to confirm
print("Current working directory:", os.getcwd())

# Clear the folder
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

print("Data folder cleared.")


# ### parses dates and run time

# In[3]:


current_month = parse('0 months ago')

current_month = current_month.strftime("%b")


# In[4]:


date = parse('today GMT')

date = date.strftime("%Y%m%d")


# ### model run

# In[5]:


run = '00'


# ### download data

# In[7]:


import concurrent.futures
import requests
import os

# Generate the list of forecast hours
# First, forecast every 3 hours from 0 to 144 hours, then every 6 hours from 150 to 360 hours
forecast_hours = [str(i) for i in range(0, 145, 3)] + [str(i) for i in range(150, 361, 6)]

# Generate the URLs directly into the list
ec_data_list = [
    f'https://data.ecmwf.int/forecasts/{date}/{run}z/ifs/0p25/oper/{date}{run}0000-{step}h-oper-fc.grib2'
    for step in forecast_hours
]

# Function to download a single URL
def download_data(index, url):
    # Extract the original file name from the URL
    file_name = url.split('/')[-1]  # This gets the file name from the URL
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a file directly into the current working directory
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {url} as {file_name}")
        else:
            print(f"Failed to download {url} (HTTP {response.status_code})")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Example of using the list for concurrent downloads
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(download_data, idx, url) for idx, url in enumerate(ec_data_list)]

    # Wait for all downloads to complete
    for future in concurrent.futures.as_completed(futures):
        future.result()  # Block until all downloads are complete


# ### process & create netcdf

# In[8]:


import xarray as xr

# Variable names
var_names = ['u', 'v', 'r', 'gh', 't', 'tp', 'skt', 't2m', 'sp', 'st', 'msl', 'tcwv', 
             'q', 'vo', 'd', 'ro', 'u10', 'v10', 'cape']

# Path to GRIB2 files
grib_files = glob.glob("*.grib2")

ds_list = []

for v in var_names:
    print(f"Processing variable: {v}")
    try:
        # Open datasets with filtering by variable name
        ds = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"cfVarName": v}},
            concat_dim="valid_time",
            combine="nested",
        )
        print(f"Successfully loaded dataset for variable: {v}")
        ds_list.append(ds)

    except Exception as e:
        print(f"Error for variable {v}: {e}")

# Merge all datasets
try:
    ecmwf = xr.merge(ds_list, compat="override")
    ecmwf = ecmwf.sortby('valid_time')  # Sort the merged dataset by valid_time
    print("Merge and sorting successful!")
except Exception as e:
    print(f"Error during merge or sorting: {e}")

ecmwf.to_netcdf('ecmwf.nc')


# In[ ]:




