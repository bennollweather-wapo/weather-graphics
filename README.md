# weather-graphics

The weather graphics suite at the Washington Post downloads ECMWF weather model data, post-processes it and creates maps for use in stories.

1. ecmwf_download.py: this script downloads the latest ECMWF weather model data, comprising 3 hourly forecasts from hour 0 to 144 and six hourly forecasts from hour 144 to 360, and merges it into a netcdf (~40 GB) and zarr (~15 GB) file. ECMWF runs four times daily (00, 06, 12 and 18 UTC). This script downloads 00 UTC data by default, but could be modified to download 12 UTC data. The data arrives at 08:34 UTC daily. This script can be triggered to run at 4:00 a.m. eastern time.