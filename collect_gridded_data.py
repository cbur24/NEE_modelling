import os
import odc.geo.xr
import xarray as xr
import numpy as np
import pandas as pd
from odc.algo import xr_reproject
from odc.geo.xr import assign_crs

def collect_gridded_data(time, chunks, verbose=True):
    
    # Retrieving ANU climate data (1 km resolution)
    if verbose:
        print('Collecting data for year: '+time)
        print('   Extracting ANU Climate data')
    base='/g/data/gh70/ANUClimate/v2-0/stable/month/'
    
    Ta = xr.open_mfdataset([base+'tavg/'+time+'/'+i for i in os.listdir(base+'tavg/'+time+'/')]).compute()
    Ta = assign_crs(Ta, crs='epsg:4283') #GDA94
    Ta = Ta.drop('crs').tavg
    Ta = Ta.rename({'lat':'latitude', 'lon':'longitude'})
    
    precip = xr.open_mfdataset([base+'rain/'+time+'/'+i for i in os.listdir(base+'rain/'+time+'/')]).compute()
    precip = assign_crs(precip, crs='epsg:4283') #GDA94
    precip = precip.drop('crs').rain
    precip = precip.rename({'lat':'latitude', 'lon':'longitude'})
    
    srad = xr.open_mfdataset([base+'srad/'+time+'/'+i for i in os.listdir(base+'srad/'+time+'/')]).compute()
    srad = assign_crs(srad, crs='epsg:4283') #GDA94
    srad = srad.drop('crs').srad
    srad = srad.rename({'lat':'latitude', 'lon':'longitude'})
    
    vpd = xr.open_mfdataset([base+'vpd/'+time+'/'+i for i in os.listdir(base+'vpd/'+time+'/')]).compute()
    vpd = assign_crs(vpd, crs='epsg:4283') #GDA94
    vpd = vpd.drop('crs').vpd
    vpd = vpd.rename({'lat':'latitude', 'lon':'longitude'})

    clim = xr.merge([Ta, precip, srad, vpd], compat='override')

    #Leaf Area Index from MODIS
    if verbose:
        print('   Extracting MODIS LAI')
    lai = xr.open_dataset('/g/data/fj4/MODIS_LAI/AU/nc/MOD15A2H.'+time+'_AU_AWRAgrd.nc', chunks=chunks)
    lai = assign_crs(lai, crs=lai.crs.spatial_ref)
    lai = lai.Band1.rename('LAI') #tidy up the dataset
    lai = lai.where((lai <= 10) & (lai >=0)) #remove artefacts and 'no-data'
    lai = lai.rename({'lat':'latitude', 'lon':'longitude'})
    lai = lai.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().compute()

    ## Soil moisture from GRAFS
    if verbose:
        print('   Extracting soil moisture')
    sws = xr.open_dataset('/g/data/fj4/SatelliteSoilMoistureProducts/S-GRAFS/ANNUAL_NC/surface_soil_moisture_vol_1km_'+time+'.nc',
                          chunks=chunks)
    sws = assign_crs(sws, crs=sws.attrs['crs'][-9:])
    sws = sws.soil_moisture.where(sws >=0)
    sws = sws.rename({'lat':'latitude', 'lon':'longitude'})
    sws = sws.soil_moisture.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().compute()

    # Reproject to match AWRA 5km grid
    if verbose:
        print('   Reprojecting datasets')
    sws = xr_reproject(sws, geobox=lai.geobox).compute()
    clim = xr_reproject(clim, geobox=lai.geobox).compute()
    
    # Due to precision of Float64 on coordinates, the clim/sws coordinates
    # didn't quite match the lai coords, resulting in adding spurious pixels after merge.
    # converting to float32 rounds coords so they match
    clim['latitude'] = clim.latitude.astype('float32')
    clim['longitude'] = clim.longitude.astype('float32')

    sws['latitude'] = sws.latitude.astype('float32')
    sws['longitude'] = sws.longitude.astype('float32')

    lai['latitude'] = lai.latitude.astype('float32')
    lai['longitude'] = lai.longitude.astype('float32')
    
    #merge all datasets together
    data = xr.merge([clim, lai, sws], compat='override')
    
    #create mask where data is valid
    mask = ~np.isnan(data.LAI.isel(time=0))
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