import os
import xarray as xr
import numpy as np
from odc.geo.xr import assign_crs


def round_coords(ds):
    """
    Due to precision of float64 on coordinates, the lai/lst/fpar coordinates
    don't quite match after reprojection, resulting in adding spurious
    pixels after merge. Converting to float32 rounds coords so they match.
    """
    ds['latitude'] = ds.latitude.astype('float32')
    ds['longitude'] = ds.longitude.astype('float32')
    try:
        ds = ds.drop('spatial_ref')
    except:
        pass
    return ds
    

def collect_prediction_data(time_start, time_end, verbose=True):
    
    if verbose:
        print('   Extracting MODIS LAI')
    lai = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/LAI_5km_monthly_2002_2021.nc'))
    lai = lai.sel(time=slice(time_start, time_end))
    
    if verbose:
        print('   Extracting MODIS EVI')
    evi = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/EVI_5km_monthly_2002_2021.nc'))
    evi = evi.sel(time=slice(time_start, time_end))
    
    if verbose:
        print('   Extracting MODIS LST')
    lst = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/LST_5km_monthly_2002_2021.nc'))
    lst = lst.sel(time=slice(time_start, time_end))
    
    if verbose:
        print('   Extracting MODIS fPAR')
    fpar = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/FPAR_5km_monthly_2002_2021.nc'))
    fpar = fpar.sel(time=slice(time_start, time_end))
    
    if verbose:
        print('   Extracting dT')
    dT = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/LST_Tair_5km_2002_2021.nc'))
    dT = dT.sel(time=slice(time_start, time_end))
    
    if verbose:
        print('   Extracting Moisture Index')
    mi = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/Moisture_index_5km_monthly_2002_2021.nc'))
    mi = mi.sel(time=slice(time_start, time_end))

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

    rain = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/chirps_aus_monthly_1991_2021.nc'))
    rain = rain.sel(time=slice(time_start, time_end))

    # Three-monthly cumulative rainfall
    if verbose:
        print('   Cumulative rainfall')
    rain_cml_3 = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/chirps_cml3_1991_2021.nc'))
    rain_cml_3 = rain_cml_3.sel(time=slice(time_start, time_end))
    
    rain_cml_6 = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/chirps_cml6_1991_2021.nc'))
    rain_cml_6 = rain_cml_6.sel(time=slice(time_start, time_end))
    
    # VCF
    if verbose:
        print('   Adding Vegetation fractions')
    tree = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/Tree_cover_5km_monthly_2002_2021.nc'))
    tree = tree.sel(time=slice(time_start, time_end))
    
    nontree = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/NonTree_cover_5km_monthly_2002_2021.nc'))
    nontree = nontree.sel(time=slice(time_start, time_end))
    
    nonveg = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/NonVeg_cover_5km_monthly_2002_2021.nc'))
    nonveg = nonveg.sel(time=slice(time_start, time_end))
    
    twi = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/TWI_5km_monthly_2002_2021.nc'))
    twi = twi.sel(time=slice(time_start, time_end))
    
    lc = round_coords(xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/5km/Landcover_merged_5km.nc'))
    lc = lc.sel(time=slice(time_start, time_end))
    
    #merge all datasets together
    if verbose:
        print('   Merge and create valid data mask')
    data = xr.merge([lai,evi,lst,fpar,tree,nontree,nonveg,dT,mi,solar,tavg,vpd,rain,rain_cml_3,rain_cml_6,twi,lc], compat='override')
                         
    #create mask where data is valid (excludes urban, water)
    mask = ~np.isnan(data['PFT'].isel(time=0))
    data = data.where(mask)
    
    if verbose:
        print('   Exporting netcdf')
    # export data
    data = data.rename({'latitude':'y', 'longitude':'x'}) #this helps with predict_xr
    data = data.astype('float32') #make sure all data is in float32
    data = assign_crs(data, crs='epsg:4326')
    
    data.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/results/prediction_data/prediction_data_'+time_start+'_'+time_end+'.nc')
    
    return data
