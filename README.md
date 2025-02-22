# weather-graphics

The weather graphics suite at the Washington Post downloads ECMWF weather model data, post-processes it and creates beautiful graphics for use in stories.

To start generating graphics, you should:

1. Clone this repository
2. Create a Python enviroment using the yaml in /environment
3. Run the following download script between 4:00 a.m. and noon eastern time
4. Run the image-generating scripts as needed

**ecmwf_download**: this script downloads the latest ECMWF weather model data, comprising 3 hourly forecasts from hour 0 to 144 and six hourly forecasts from hour 144 to 360, and merges it into a netcdf (~40 GB) and zarr (~15 GB) file. ECMWF runs four times daily (00, 06, 12 and 18 UTC). This script downloads 00 UTC data by default, but could be modified to download 12 UTC data. The data arrives at 08:34 UTC daily. This script can be triggered to run at 4:00 a.m. eastern time.

Within the ecmwf.nc and ecmwf.zarr files exist these meteorological variables:

var_names = ['u', 'v', 'r', 'gh', 't', 'tp', 'skt', 't2m', 'sp', 'st', 'msl', 'tcwv', 
             'q', 'vo', 'd', 'ro', 'u10', 'v10', 'cape']

## Variable descriptions

**u**: Zonal wind component (m/s), representing wind speed in the east-west direction.  

**v**: Meridional wind component (m/s), representing wind speed in the north-south direction.  

**r**: Relative humidity (%), indicating the amount of moisture in the air relative to its maximum capacity.  

**gh**: Geopotential height (m), representing the height of a pressure surface above sea level.  

**t**: Temperature (K), the atmospheric temperature at various pressure levels.  

**tp**: Total precipitation (m), the accumulated rainfall or snowfall over a given period.  

**skt**: Skin temperature (K), the temperature of the Earth's surface.  

**t2m**: 2-meter temperature (K), the air temperature at 2 meters above the ground.  

**sp**: Surface pressure (Pa), the atmospheric pressure at the Earth's surface.  

**st**: Soil temperature (K), the temperature within the soil at various depths.  

**msl**: Mean sea level pressure (Pa), the atmospheric pressure adjusted to sea level.  

**tcwv**: Total column water vapor (kg/m²), the total amount of water vapor in a vertical column of the atmosphere.  

**q**: Specific humidity (kg/kg), the mass of water vapor per unit mass of air.  

**vo**: Vorticity (1/s), a measure of the rotation of air in the atmosphere.  

**d**: Divergence (1/s), representing the rate at which air spreads apart or converges.  

**ro**: Runoff (m), the amount of water that flows over land surfaces after precipitation.  

**u10**: 10-meter zonal wind (m/s), the east-west wind component at 10 meters above ground.  

**v10**: 10-meter meridional wind (m/s), the north-south wind component at 10 meters above ground.  

**cape**: Convective available potential energy (J/kg), indicating atmospheric instability and the potential for convection or storms.  

## Script descriptions

**temp_extremes**: Computes the maximum and minimum temperatures across the United States over the next 15 days and plots them. Useful for visualizing temperature extremes (e.g., polar vortex and heat waves).

**precip**: Plots five and fifteen day rainfall. Useful for visualizing how much rain or snow may fall.

**jet**: Computes jet stream winds and plots them. Useful for visualizing winds in the upper atmosphere, which fuels storms.

**pwat_mslp**: Plots atmospheric moisture and air pressure. Useful for visualizing the level of moisture in the atmosphere and the strength of storms, which determines how much precipitation may fall.