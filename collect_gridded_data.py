import os
import xarray as xr
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from datacube.utils.geometry import assign_crs
from datacube.utils.dask import start_local_dask

def collect_gridded_data(time, chunks, verbose=True):
    
    # Retrieving ANU climate data (1 km resolution)
    if verbose:
        print('Collecting data for year: '+time)
        print('   Extracting ANU Climate data')
    base='/g/data/gh70/ANUClimate/v2-0/stable/month/'
    
    Ta = xr.open_mfdataset([base+'tavg/'+time+'/'+i for i in os.listdir(base+'tavg/'+time+'/')]).compute()
    precip = xr.open_mfdataset([base+'rain/'+time+'/'+i for i in os.listdir(base+'rain/'+time+'/')]).compute()
    fn = xr.open_mfdataset([base+'srad/'+time+'/'+i for i in os.listdir(base+'srad/'+time+'/')]).compute()
    vpd = xr.open_mfdataset([base+'vpd/'+time+'/'+i for i in os.listdir(base+'vpd/'+time+'/')]).compute()
    
    Ta=assign_crs(Ta, crs=Ta.crs.attrs['spatial_ref'])
    precip = assign_crs(precip, crs=precip.crs.attrs['spatial_ref'])   
    fn = assign_crs(fn, crs=fn.crs.attrs['spatial_ref'])
    vpd = assign_crs(vpd, crs=vpd.crs.attrs['spatial_ref'])

    clim = xr.merge([Ta, precip, fn, vpd], compat='override')
    clim = clim.rename({"lon": "x", "lat": "y"})
    
    #Leaf Area Index from MODIS
    if verbose:
        print('   Extracting MODIS LAI')
    lai = xr.open_dataset('/g/data/ub8/au/MODIS/mosaic/MOD15A2H.006/MOD15A2H.006.b02.500m_lai.'+time+'.nc',
                          chunks=chunks).rename({"500m_lai": "lai"})

    lai = assign_crs(lai, crs='epsg:4326')
    lai = lai.lai.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().compute()
    lai = lai.rename({"longitude": "x", "latitude": "y"})
    
#     ## Soil moisture from GRAFS
    # if verbose:
    #     print('Extracting soil moisture')
#     sws = xr.open_dataset('/g/data/ub8/global/GRAFS/GRAFS_RootzoneSoilWaterIndex_'+time+'.nc',
#                           chunks=chunks)

#     sws = assign_crs(sws.soil_water_index, crs='epsg:4326')
#     sws = sws.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().compute()
#     sws = sws.rename({"lon": "x", "lat": "y"})
    
    # Reproject to match climate data 
    if verbose:
        print('   Reprojecting datasets')
    lai = lai.rio.reproject_match(clim, resampling=Resampling.average)
    #sws = sws.rio.reproject_match(clim, resampling=Resampling.bilinear)
    
    #merge all datasets together
    data = xr.merge([clim, lai], compat='override').drop('crs')
    
    #create mask where data is valid
    mask = ~np.isnan(data.tavg.isel(time=0))
    data = data.where(mask)
    
    #add a 1-month lag
    if verbose:
        print('   Adding a lag')
    data_lag1 = data.shift(time=1)
    for i in data.data_vars:
        data_lag1 = data_lag1.rename({i:i+'_L1'})
    data = data.merge(data_lag1)
    
    if verbose:
        print('   Exporting netcdf')
    #export data
    data.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/results/input_data/input_data_'+time+'.nc')