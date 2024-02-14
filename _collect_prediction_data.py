import os
import xarray as xr
import numpy as np
from odc.geo.xr import assign_crs


def allNaN_arg(da, dim, stat, idx=True):
    """
    Calculate da.argmax() or da.argmin() while handling
    all-NaN slices. Fills all-NaN locations with an
    float and then masks the offending cells.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str
        Dimension over which to calculate argmax, argmin e.g. 'time'
    stat : str
        The statistic to calculte, either 'min' for argmin()
        or 'max' for .argmax()
    idx : bool
        If True then use da.idxmax() or da.idxmin(), otherwise
        use ds.argmax() or ds.argmin()

    Returns
    -------
    xarray.DataArray
    """
    # generate a mask where entire axis along dimension is NaN
    mask = da.isnull().all(dim)

    if stat == "max":
        y = da.fillna(float(da.min() - 1))
        if idx==True:
            y = y.idxmax(dim=dim, skipna=True).where(~mask)
        else:
            y = y.argmax(dim=dim, skipna=True).where(~mask)
        return y

    if stat == "min":
        y = da.fillna(float(da.max() + 1))
        if idx==True:
            y = y.idxmin(dim=dim, skipna=True).where(~mask)
        else:
            y = y.argmin(dim=dim, skipna=True).where(~mask)
        return y

def round_coords(ds):
    """
    Due to precision of float64 on coordinates, coordinates
    don't quite match after reprojection, resulting in adding spurious
    pixels after merge. Converting to float32 rounds coords so they match.
    """
    try:
        ds['latitude'] = ds.latitude.astype('float32')
        ds['longitude'] = ds.longitude.astype('float32')
        ds['latitude'] = np.array([round(i,4) for i in ds.latitude.values])
        ds['longitude'] = np.array([round(i,4) for i in ds.longitude.values])
    except:
        ds['x'] = ds.x.astype('float32')
        ds['y'] = ds.y.astype('float32')
        ds['x'] = np.array([round(i,4) for i in ds.x.values])
        ds['y'] = np.array([round(i,4) for i in ds.y.values])
    
    return ds
    

def collect_prediction_data(time_start,
                            time_end,
                            scale='1km',
                            covariables=[
                                 #'LAI',
                                 #'LAI_anom',
                                 'kNDVI',
                                 'kNDVI_anom',
                                 #'FPAR',
                                 #'FPAR-NDVI',
                                 'LST',
                                 'trees',
                                 'grass',
                                 'bare',
                                 'C4_grass',
                                 #'Tree',
                                 #'NonTree',
                                 #'NonVeg',
                                 'LST_Tair',
                                 'TWI',
                                 'NDWI',
                                 #'NDWI_anom',
                                 'rain',
                                 'rain_cml3',
                                 'rain_cml6',
                                 'rain_cml12',
                                 'rain_anom',
                                 'rain_cml3_anom',
                                 'rain_cml6_anom',
                                 'rain_cml12_anom',
                                 'srad',
                                 'srad_anom',
                                 'vpd',
                                 'tavg',
                                 'tavg_anom',
                                 #'SOC',
                                 #'CO2',
                                 #'C4percent',
                                 #'Elevation',
                                 #'MOY',
                                 'VegH',
                                 #'MI'
                            ],
                            chunks=dict(latitude=1150, longitude=1100, time=1),
                            export=False,
                            verbose=True
                           ):
  
    dss=[]
    base='/g/data/os22/chad_tmp/NEE_modelling/data/' 
    for var in covariables:
        if verbose:
            print(f'   Extracting {var}')
            
        ds = xr.open_dataset(f'{base}{scale}/{var}_{scale}_monthly_2002_2022.nc',
                             chunks=chunks
                            )
        ds = ds.sel(time=slice(time_start, time_end))
        
        #makse sure coords match (remove trailing zeros)
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
