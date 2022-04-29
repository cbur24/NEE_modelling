import os
import odc.geo.xr
import xarray as xr
import numpy as np
import pandas as pd
from odc.algo import xr_reproject
from odc.geo.xr import assign_crs

def collect_gridded_data(time, verbose=True):
    
    # Retrieving ANU climate data (1 km resolution)
    if verbose:
        print('Collecting data for year: '+time)
        print('   Extracting ANU Climate data')
        
    base='/g/data/gh70/ANUClimate/v2-0/stable/month/'
    Ta = xr.open_mfdataset([base+'tavg/'+time+'/'+i for i in os.listdir(base+'tavg/'+time+'/')],
                          chunks=dict(lat=1000, lon=1000)).compute()
    Ta = assign_crs(Ta, crs='epsg:4283') #GDA94
    Ta = Ta.drop('crs').tavg
    Ta = Ta.rename({'lat':'latitude', 'lon':'longitude'})
    
    precip = xr.open_mfdataset([base+'rain/'+time+'/'+i for i in os.listdir(base+'rain/'+time+'/')],
                              chunks=dict(lat=1000, lon=1000)).compute()
    precip = assign_crs(precip, crs='epsg:4283') #GDA94
    precip = precip.drop('crs').rain
    precip = precip.rename({'lat':'latitude', 'lon':'longitude'})
    
    # srad = xr.open_mfdataset([base+'srad/'+time+'/'+i for i in os.listdir(base+'srad/'+time+'/')]).compute()
    # srad = assign_crs(srad, crs='epsg:4283') #GDA94
    # srad = srad.drop('crs').srad
    # srad = srad.rename({'lat':'latitude', 'lon':'longitude'})
    
    vpd = xr.open_mfdataset([base+'vpd/'+time+'/'+i for i in os.listdir(base+'vpd/'+time+'/')],
                           chunks=dict(lat=1000, lon=1000)).compute()
    vpd = assign_crs(vpd, crs='epsg:4283') #GDA94
    vpd = vpd.drop('crs').vpd
    vpd = vpd.rename({'lat':'latitude', 'lon':'longitude'})

    clim = xr.merge([Ta, precip, vpd], compat='override')
    
    # Leaf Area Index from MODIS
    if verbose:
        print('   Loading MODIS LAI')
    lai = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/LAI/LAI_500m_monthly_'+time+'.nc').compute()
    
    ## Soil moisture from GRAFS
    if verbose:
        print('   Loading soil moisture')
    sws = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/GRAFS/GRAFS_1km_monthly_'+time+'.nc').compute()

    # Reproject to match ANU Clim 1km grid
    if verbose:
        print('   Reprojecting datasets')
    sws = xr_reproject(sws, geobox=clim.geobox, resampling='bilinear')
    lai = xr_reproject(lai, geobox=clim.geobox, resampling='average')
    
    # Due to precision of Float64 on coordinates, the lai/sws/clim coordinates
    # don't quite match after reprojection, resulting in adding spurious pixels after merge.
    # converting to float32 rounds coords so they match
    clim['latitude'] = clim.latitude.astype('float32')
    clim['longitude'] = clim.longitude.astype('float32')

    sws['latitude'] = sws.latitude.astype('float32')
    sws['longitude'] = sws.longitude.astype('float32')

    lai['latitude'] = lai.latitude.astype('float32')
    lai['longitude'] = lai.longitude.astype('float32')
    
    #merge all datasets together
    data = xr.merge([clim, lai, sws], compat='override')
    data = data.unify_chunks()
    
    #create mask where data is valid (spurios values from reproject)
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
    # export data
    data = data.rename({'latitude':'y', 'longitude':'x'})
    data.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/results/input_data/input_data_'+time+'.nc')