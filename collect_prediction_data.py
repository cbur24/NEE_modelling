import os
import odc.geo.xr
import xarray as xr
import numpy as np
import pandas as pd
from odc.algo import xr_reproject
from odc.geo.xr import assign_crs


def round_coords(ds):
    """
    Due to precision of float64 on coordinates, the lai/lst/fpar coordinates
    don't quite match after reprojection, resulting in adding spurious
    pixels after merge. Converting to float32 rounds coords so they match.
    """
    ds['latitude'] = ds.latitude.astype('float32')
    ds['longitude'] = ds.longitude.astype('float32')
    return ds
    

def collect_prediction_data(time_start, time_end, verbose=True):
    
    # Leaf Area Index from MODIS
    if verbose:
        print('   Extracting MODIS LAI')
    lai = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/LAI_5km_monthly_2002_2021.nc'))
    lai = lai.sel(time=slice(time_start, time_end))
    
    # LST from MODIS
    if verbose:
        print('   Extracting MODIS LST')
    lst = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/LST_5km_monthly_2002_2021.nc'))
    lst = lst.sel(time=slice(time_start, time_end))
    
    # fPAR from MODIS
    if verbose:
        print('   Extracting MODIS fPAR')
    fpar = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/FPAR_5km_monthly_2002_2021.nc'))
    fpar = fpar.sel(time=slice(time_start, time_end))
    
    # Delta temp
    if verbose:
        print('   Extracting dT')
    dT = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/LST_Tair_2002_2021.nc'))
    dT = dT.sel(time=slice(time_start, time_end))
    
    # SPEI
    if verbose:
        print('   Extracting SPEI')
    spei = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/SPEI/chirps_spei_gamma_06.nc')
    spei = spei.rename({'spei_gamma_06':'spei'})
    spei = round_coords(spei.rename({'lat':'latitude', 'lon':'longitude'}))
    spei = spei.sel(time=slice(time_start, time_end))

    # Climate
    if verbose:
        print('   Extracting Climate')
    solar = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/solar_monthly_wm2_2000_2021.nc'))
    solar = solar.rename({'solar_exposure_day':'solar'})
    solar = solar.sel(time=slice(time_start, time_end))
    
    tavg = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/tavg_monthly_1991_2021.nc'))
    tavg = tavg.rename({'temp_avg_month':'Ta'})
    tavg = tavg.sel(time=slice(time_start, time_end))
    
    vpd = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/vpd_monthly_2000_2021.nc'))
    vpd = vpd.sel(time=slice(time_start, time_end))

    rain = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/chirps_aus_monthly_1991_2021.nc'))
    rain = rain.sel(time=slice(time_start, time_end))

    #add lags to rainfall
    if verbose:
        print('   Adding Rainfall lags')
    rain_l1 = rain.shift(time=1).rename({'precip':'precip_L1'})
    rain_l2 = rain.shift(time=2).rename({'precip':'precip_L2'})
    rain_l3 = rain.shift(time=3).rename({'precip':'precip_L3'})
    rain = xr.merge([rain,rain_l1,rain_l2,rain_l3])
    
    # landcover
    if verbose:
        print('   Adding Landcover class')
    lc = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/IGBP_Landcover_MODIS_5km.nc'))
    lc = lc.sel(time=slice(time_start, time_end))
    
    #merge all datasets together
    if verbose:
        print('   Merge and create valid data mask')
    data = xr.merge([lai,lst,fpar,dT,spei,solar,tavg,vpd,rain,lc], compat='override')
    
    #create mask where data is valid (spurios values from reproject)
    mask = ~np.isnan(data.precip.isel(time=0))
    data = data.where(mask)
    
    if verbose:
        print('   Exporting netcdf')
    # export data
    data = data.rename({'latitude':'y', 'longitude':'x'}) #this helps with predict_xr
    data = data.astype('float32') #make sure all data is in float32
    
    data.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/results/prediction_data/prediction_data_'+time_start+'_'+time_end+'.nc')
    
    return data
