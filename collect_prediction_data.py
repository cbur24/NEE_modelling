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
    

def collect_prediction_data(time_start,
                            time_end,
                            scale='1km',
                            covariables=[
                               #'LAI',
                                 'LAI_anom',
                                 #'kNDVI',
                                 'kNDVI_anom',
                                 'FPAR',
                                 'LST',
                                 'Tree',
                                 'NonTree',
                                 'NonVeg',
                                 'LST_Tair',
                                 'TWI',
                                 'NDWI',
                                 #'NDWI_anom',
                                 #'rain'
                                 'rain_anom',
                                 'rain_cml3_anom',
                                 #'rain_cml6_anom',
                                 'rain_cml12_anom',
                                 'CWD',
                                 'srad',
                                 'vpd',
                                 #'tavg',
                                 'tavg_anom',
                                 'SOC',
                                 'CO2'
                                 #'FireDisturbance'
                            ],
                            chunks=dict(latitude=1200, longitude=1200),
                            export=False,
                            verbose=True
                           ):
  
    dss=[]
    base='/g/data/os22/chad_tmp/NEE_modelling/data/' 
    for var in covariables:
        if verbose:
            print(f'   Extracting {var}')
            
        ds = xr.open_dataset(f'{base}{scale}/{var}_{scale}_monthly_2002_2021.nc',
                             chunks=chunks
                            )
        ds = ds.sel(time=slice(time_start, time_end))
        
        #makse sure coords match (trailing zeros)
        ds['latitude'] = ds.latitude.astype('float32')
        ds['latitude'] = np.array([round(i,4) for i in ds.latitude.values])
        ds['longitude'] = ds.longitude.astype('float32')
        ds['longitude'] = np.array([round(i,4) for i in ds.longitude.values])
        
        dss.append(ds)
    
    #merge all datasets together
    if verbose:
        print('   Merge and create valid data mask')
    data = xr.merge(dss, compat='override')
                         
    # #create mask where data is valid (excludes urban, water)
    # mask = ~np.isnan(data['PFT'].isel(time=0))
    # data = data.where(mask)
    
    #remove landcover
    #data = data.drop('PFT')
    
    if verbose:
        print('   Exporting netcdf')
    
    # export data
    data = data.rename({'latitude':'y', 'longitude':'x'}) #this helps with predict_xr
    data = data.astype('float32') #make sure all data is in float32
    data = assign_crs(data, crs='epsg:4326')
    
    if export:
        data.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/results/prediction_data/prediction_data_'+time_start+'_'+time_end+'.nc')
    
    return data
