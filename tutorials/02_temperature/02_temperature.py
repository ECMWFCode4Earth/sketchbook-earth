# %%
# Python Standard Libraries
import os
import datetime as dt
import zipfile
import urllib.request
from string import ascii_lowercase as ABC

# Data Manipulation Libraries
import numpy as np
import pandas as pd
import xarray as xr
import regionmask as rm
import dask

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from dask.diagnostics.progress import ProgressBar

plt.style.use(
    "../copernicus.mplstyle"
)  # Set the visual style of the plots; not necessary for the tutorial

# Climate Data Store API for retrieving climate data
import cdsapi

dask.config.set(**{"array.slicing.split_large_chunks": True})

# Boolean land-sea mask
lsm = rm.defined_regions.natural_earth_v5_0_0.land_110

# %%
file_name = {}  # dictionary containing [data source : file name]

# Add the data sources and file names
file_name.update(
    {"berkeley": "temperature_berkeley.nc"}
)  # is not available as netCDF, only as zip file
file_name.update(
    {"gistemp": "temperature_gistemp.gz"}
)  # is not available as netCDF, only as zip file
file_name.update({"hadcrut": "temperature_hadcrut.nc"})
file_name.update({"era5": "temperature_era5.nc"})
file_name.update({"eobs": "temperature_eobs.tar.gz"})

# Create the paths to the files
path_to = {
    source: os.path.join(f"data/{source}/", file) for source, file in file_name.items()
}

# Create necessary directories if they do not exist
for path in path_to.values():
    os.makedirs(
        os.path.dirname(path), exist_ok=True
    )  # create the folder if not available

path_to


# %%
def coordinate_is_monthly(ds, coord: str = "time"):
    """Return True if the coordinates are months"""
    time_diffs = np.diff(ds.coords[coord].values)
    time_diffs = pd.to_timedelta(time_diffs).days

    # If all differences are between 28 and 31 days
    if np.all((28 <= time_diffs) & (time_diffs <= 31)):
        return True
    else:
        return False


def streamline_coords(da):
    """Streamline the coordinates of a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to streamline.
    """

    # Ensure that time coordinate is fixed to the first day of the month
    if "time" in da.coords:
        if coordinate_is_monthly(da, "time"):
            da.coords["time"] = da["time"].to_index().to_period("M").to_timestamp()

    # Ensure that spatial coordinates are called 'lon' and 'lat'
    if "longitude" in da.coords:
        da = da.rename({"longitude": "lon"})
    if "latitude" in da.coords:
        da = da.rename({"latitude": "lat"})

    # Ensure that lon/lat are sorted in ascending order
    da = da.sortby("lat")
    da = da.sortby("lon")

    # Ensure that lon is in the range [-180, 180]
    lon_min = da["lon"].min()
    lon_max = da["lon"].max()
    if lon_min < -180 or lon_max > 180:
        da.coords["lon"] = (da.coords["lon"] + 180) % 360 - 180
        da = da.sortby(da.lon)

    return da


# %%
# Define regions of interest
# =============================================================================
# Some regions are defined here: https://climate.copernicus.eu/esotc/2022/about-data#Regiondefinitions
region_of_interest = {
    "Global": {"lon": slice(-180, 180), "lat": slice(-90, 90)},
    "Northern Hemisphere": {"lon": slice(-180, 180), "lat": slice(0, 90)},
    "Southern Hemisphere": {"lon": slice(-180, 180), "lat": slice(-90, 0)},
    "Europe": {"lon": slice(-25, 40), "lat": slice(34, 72)},
    "Arctic": {"lon": slice(-180, 180), "lat": slice(66.6, 90)},
}

# Define climatology period
# =============================================================================
clim_period = {"time": slice("1991", "2020")}

# %%
# Long term evolution of global mean temperature
# =============================================================================
# %%
# NOAA
url_to_noaa = "https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5.1/NOAAGlobalTemp_v5.1.0_gridded_s185001_e202306_c20230708T112624.nc"
noaa = xr.open_dataset(url_to_noaa)
noaa = noaa.isel(z=0, drop=True)
noaa = streamline_coords(noaa)
noaa["lsm"] = lsm.mask(noaa).notnull()

# %%
# Berkeley
berkeley = xr.open_dataset(path_to["berkeley"])
new_time_coords = xr.cftime_range(
    start="1850-01-01", periods=berkeley.time.size, freq="MS"
).to_datetimeindex()
berkeley.coords.update({"time": new_time_coords})
berkeley = streamline_coords(berkeley)
# %%
# GISTEMP
with xr.open_dataset("data/gistemp/temperature_gistemp_1200km.gz") as gistemp_1200:
    gistemp_1200 = gistemp_1200["tempanomaly"]
with xr.open_dataset("data/gistemp/temperature_gistemp_250km.gz") as gistemp_250:
    gistemp_250 = gistemp_250["tempanomaly"]
gistemp = gistemp_250.where(gistemp_250.notnull(), other=gistemp_1200)
gistemp_lsm = pd.read_csv(
    "data/gistemp/temperature_gistemp_land_sea_mask.txt",
    sep="\s+",
    header=1,
    names=["lon", "lat", "mask"],
)
gistemp_lsm = gistemp_lsm.set_index(["lat", "lon"])
gistemp_lsm = gistemp_lsm.to_xarray()
gistemp = xr.merge([gistemp, gistemp_lsm["mask"]])
gistemp = streamline_coords(gistemp)


# %%
# HadCRUT
with xr.open_dataset(path_to["hadcrut"]) as hadcrut:
    pass
with xr.open_dataset("data/hadcrut/temperature_weights.nc") as hadcrut_weights:
    pass
hadcrut_members = xr.open_mfdataset(
    "data/hadcrut/*analysis*.nc", combine="nested", concat_dim="realization"
)
hadcrut_members = hadcrut_members.compute()
hadcrut = xr.Dataset(
    {
        "mean": hadcrut["tas_mean"],
        "weights": hadcrut_weights["weights"],
        "ensemble": hadcrut_members["tas"],
    }
)
hadcrut = streamline_coords(hadcrut)

# %%
# ERA5
with xr.open_mfdataset(path_to["era5"]) as era5:
    # convert from Kelvin to Celsius
    era5["t2m"] = era5["t2m"] - 273.15
era5 = streamline_coords(era5)
era5_monthly_climatology = era5["t2m"].sel(clim_period).groupby("time.month").mean()
era5["anom"] = era5["t2m"].groupby("time.month") - era5_monthly_climatology


# %%


def weighted_spatial_average(da, region_of_interest, land_mask=None):
    """Calculate the weighted spatial average of a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to average.
    weights : xr.DataArray, optional
        A DataArray with the same dimensions as `da` containing the weights.
    """

    # Area weighting: calculate the area of each grid cell
    weights = np.cos(np.deg2rad(da.lat))

    # Additional user-specified weights, e.g. land-sea mask
    if land_mask is not None:
        weights = weights * land_mask.fillna(0)

    return da.sel(**region_of_interest).weighted(weights).mean(("lat", "lon"))


# %%
# Spatial average of global temperature anomalies
# -----------------------------------------------------------------------------

region = "Arctic"

temps = {
    "HadCRUT": hadcrut["mean"],
    "HadCRUT_ensemble": hadcrut["ensemble"],
    "Berkeley": berkeley["temperature"],
    "GISTEMP": gistemp["tempanomaly"],
    "NOAA": noaa["anom"],
    "ERA5": era5["t2m"],
}
land_masks = {
    "HadCRUT": hadcrut["weights"],
    "HadCRUT_ensemble": hadcrut["weights"],
    "Berkeley": berkeley["land_mask"],
    "GISTEMP": gistemp["mask"],
    "NOAA": noaa["lsm"],
    "ERA5": era5["lsm"],
}

temp_evolution = {}
for source in temps:
    spatial_average = weighted_spatial_average(
        temps[source], region_of_interest[region], land_masks[source]
    )
    with ProgressBar():
        temp_evolution[source] = spatial_average.compute()
temp_evolution = xr.Dataset(temp_evolution)
# Show anomalies with respect to the 1991-2020 climatology
temp_evolution = temp_evolution - temp_evolution.sel(clim_period).mean("time")

# %%
temp_evolution_smooth = temp_evolution.rolling(time=60, center=True).mean()

anom_1850_1900 = temp_evolution_smooth.drop_vars("HadCRUT_ensemble").sel(
    time=slice("1850", "1900")
)
mean_1850_1900 = anom_1850_1900.to_array().mean()

# %%
ci = temp_evolution_smooth["HadCRUT_ensemble"].quantile([0.0, 1.0], dim="realization")
# %%
fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
temp_evolution_smooth["HadCRUT"].plot(ax=ax, label="HadCRUT")
temp_evolution_smooth["Berkeley"].plot(ax=ax, label="Berkeley")
temp_evolution_smooth["GISTEMP"].plot(ax=ax, label="GISTEMP")
temp_evolution_smooth["NOAA"].plot(ax=ax, label="NOAA")
temp_evolution_smooth["ERA5"].plot(ax=ax, label="ERA5")
ax.fill_between(
    ci.time,
    ci.sel(quantile=0.0),
    ci.sel(quantile=1.0),
    color=".7",
    alpha=0.5,
    lw=0,
    zorder=-1,
)
ax.legend(ncols=6, frameon=False, loc="upper center")


ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(True)
ax.xaxis.set_major_locator(mdates.YearLocator(20))

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title(f"{region} Mean Temperature Anomaly (in ºC) since 1850")
ax.text(
    -0.02,
    1,
    "Relative to \n1991-2020",
    rotation=0,
    ha="right",
    va="top",
    transform=ax.transAxes,
)
ax.text(
    1.02,
    1,
    "Increase above \n1850-1900\n reference level",
    rotation=0,
    ha="left",
    va="top",
    transform=ax.transAxes,
)
ax.axhline(mean_1850_1900, color=".5", lw=0.5, ls="--")

yticks = np.arange(-10, 10, 1)
ax2_yticks = yticks + mean_1850_1900.item()
ax2.set_yticks(ax2_yticks)
ax2.set_yticklabels(yticks)
ax.set_ylim(-4.1, 1.8)
ax2.set_ylim(-4.1, 1.8)

plt.show()

# %%
# Taking Earth's mean temperature
region = "Europe"
land_mask = era5["lsm"]
clim_temp_era5 = era5["t2m"].sel(clim_period)
clim_temp_era5 = weighted_spatial_average(
    clim_temp_era5, region_of_interest[region], land_mask=land_mask
)

with ProgressBar():
    print(clim_temp_era5.mean().compute())
# %%
# ERA5 vs EOBS (observational data)
# =============================================================================
# %%
# EOBS
eobs_daily = xr.open_mfdataset("data/eobs/tg_ens_mean_0.25deg_reg_v27.0e.nc")
eobs_daily = streamline_coords(eobs_daily)
# %%
# Europe only
eobs_europe_daily = eobs_daily.sel(region_of_interest["Europe"])
era5_europe = era5.sel(region_of_interest["Europe"])

# %%
# Figure 1. Annual European land surface air temperature anomalies for
# 1950 to 2022, relative to the 1991–2020 reference period.
# =============================================================================
# Convert EOBS to monthly
eobs_europe = eobs_europe_daily.resample(time="MS").mean("time")

with ProgressBar():
    eobs_europe = eobs_europe.compute()
    era5_europe = era5_europe.compute()


# %%
# Calculate the monthly climatology
def annual_anomalies(da, land_mask=None):
    da = weighted_spatial_average(da, region_of_interest["Europe"], land_mask)
    climatology = da.sel(clim_period).groupby("time.month").mean("time")
    anomalies = da.groupby("time.month") - climatology
    return anomalies.groupby("time.year").mean("time")


eobs_europe_anom = annual_anomalies(eobs_europe["tg"], land_mask=None)
era5_europe_anom = annual_anomalies(era5_europe["t2m"], land_mask=era5_europe["lsm"])

# %%
# make red and blue colors for above and below zero
clrs_eobs = np.where(eobs_europe_anom > 0, "tab:red", "tab:blue")
clrs_era5 = np.where(era5_europe_anom > 0, "darkred", "darkblue")

fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(111)
ax.bar(
    eobs_europe_anom.year,
    eobs_europe_anom,
    color=clrs_eobs,
    label="EOBS",
    alpha=0.5,
)
ax.bar(
    era5_europe_anom.year,
    era5_europe_anom,
    0.5,
    color=clrs_era5,
    label="ERA5",
    alpha=0.5,
)
ax.axhline(0, color=".5", lw=0.5, ls="--")
ax.legend(ncols=2, frameon=False, loc="upper center")
ax.set_title("European annual temperature anomalies (in ºC)")

# %%
# Correct climatolgoy!
my_clim = era5_europe["t2m"].sel(clim_period).groupby("time.month").mean()

# %%
