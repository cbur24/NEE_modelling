import os
import sys
import xarray as xr
import numpy as np
import pandas as pd

def VPD(rh, ta):
    sat_vp = (6.11 * np.exp((2500000/461) * (1/273 - 1/(273 + ta))))
    vpd = (((100 - rh)/100) * sat_vp)
    return vpd

def ec_vars(flux, var):
    df = flux[var].to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    return df

def climate_vars(path, flux_time, rename, time_start, time_end, idx):
    ds = xr.open_dataset(path)
    ds = ds.rename(rename)
    if "spei" in path:
        ds = ds .rename({'lat':'latitude', 'lon':'longitude'})
    ds = ds.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    ds = ds.reindex(time=flux_time, method='nearest', tolerance='1D').compute() 
    try:
        ds = ds[list(rename.values())[0]].to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    except:
        ds = ds[list(rename.values())[0]].to_dataframe().drop(['latitude', 'longitude'], axis=1)
    return ds

def rs_vars(path, var, flux_time, time_start, time_end, idx):
    ds = xr.open_dataset(path)
    ds = ds.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    ds = ds.reindex(time=flux_time, method='nearest', tolerance='1D').compute() 
    try:
        ds = ds[var].to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    except:
        ds = ds[var].to_dataframe().drop(['latitude', 'longitude'], axis=1)
    return ds

def extract_ec_gridded_data(suffix, scale='1km', save_ec_data=False,
                            return_coords=True, verbose=False):
    """
    Extract variables from EC tower data, and environmental
    data from remote sensing/climate datasets over pixels at EC
    tower location.
    
    Params:
    ------
    suffix : str. The string path on the THREDDS server to the netcdf eddy covariance file,
            this is appended to: 'https://dap.tern.org.au/thredds/dodsC/ecosystem_process/ozflux/'
    scale : str. One of either '1km' or '5km' denoting the spatial resolution of the
            dataset to use.
    return_coords : bool. If True returns the x,y coordinates of the EC tower as columns on the
            pandas dataframe
    verbose : bool. If true progress statements are printed
    
    
    Returns:
    -------
        Pandas.Dataframe containing coincident observations between
        EC data and gridded data.
        
    """
    
    base = 'https://dap.tern.org.au/thredds/dodsC/ecosystem_process/ozflux/'
    
    # load flux data from site
    flux = xr.open_dataset(base+suffix)
    if save_ec_data:
        flux.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/data/ec_netcdfs/'+suffix[0:5]+'_EC_site.nc')
    
    # Set negative GPP, ER, and ET measurements as zero
    flux['GPP_SOLO'] = xr.where(flux.GPP_SOLO < 0, 0, flux.GPP_SOLO)
    flux['ET'] = xr.where(flux.ET < 0, 0, flux.ET)
    flux['ER_SOLO'] = xr.where(flux.ER_SOLO < 0, 0, flux.ET)
    
    # offset time to better match gridded data
    flux['time'] = flux.time + np.timedelta64(14,'D') 
    
    #indexing spatiotemporal values at EC site
    lat = flux.latitude.values[0]
    lon = flux.longitude.values[0]
    time_start = str(np.datetime_as_string(flux.time.values[0], unit='D'))
    time_end = str(np.datetime_as_string(flux.time.values[-1], unit='D'))
    idx=dict(latitude=lat,  longitude=lon)

    # extract co2 fluxes and environ data from EC data
    if verbose:
        print('   Extracting EC data')
    
    variables = ['GPP_SOLO','ER_SOLO','ET','Ta','Sws','RH','VP','Precip','Fn','Fe','Fh','Fsd','Fld','CO2']
    nee = ec_vars(flux, 'NEE_SOLO') #extract first variable
    df_ec=[]
    for var in variables: #loop through other vars
        df = ec_vars(flux, var)
        df_ec.append(df)
    
    df_ec = nee.join(df_ec) #join other vars to NEE
    df_ec = df_ec.add_suffix('_EC')
 
    # calculate VPD on ec data
    df_ec['VPD_EC'] = VPD(df_ec.RH_EC, df_ec.Ta_EC)
    df_ec = df_ec.drop(['VP_EC'], axis=1) # drop VP
    
    #extract the first remote sensing variable
    if verbose:
        print('   Extracting LAI')
    lai = rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/{scale}/LAI_{scale}_monthly_2002_2021.nc',
                  'LAI', flux.time, time_start, time_end, idx)
    
    #extract the rest of the RS variables in loop
    rs_variables=['EVI','LST','FPAR','Tree','NonTree','NonVeg','LST_Tair',
               'AridityIndex','TWI', 'NDWI','FireDisturbance','Landcover']
    names = ['EVI','LST','FPAR','tree_cover','nontree_cover','nonveg_cover','LST-Tair',
               'AI','TWI','NDWI','Months_since_burn','PFT']
    dffs = []
    for var, name in zip(rs_variables,names):
        if verbose:
            print(f'   Extracting {var}')
        
        if var == 'LST_Tair':    
            df = rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/5km/{var}_5km_monthly_2002_2021.nc',
                  name, flux.time, time_start, time_end, idx)
        
        else:
            df = rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/{scale}/{var}_{scale}_monthly_2002_2021.nc',
                  name, flux.time, time_start, time_end, idx)
        
        dffs.append(df)
    
    #handle MI differently
    if verbose:
        print('   Extracting MoistureIndex')
    #coastal locations sometimes grab NaN over ocean for Moisture Index so shifting location slightly
    mi_path = '/g/data/os22/chad_tmp/NEE_modelling/data/5km/MoistureIndex_5km_monthly_2002_2021.nc'
    if 'CowBay' in suffix:
        mi = rs_vars(mi_path,
                  'MI', flux.time, time_start, time_end, dict(latitude=idx['latitude'], longitude=142.35))
    
    elif 'CapeTribulation' in suffix:
        mi = rs_vars(mi_path,
                  'MI', flux.time, time_start, time_end, dict(latitude=idx['latitude'], longitude=142.35))
    
    elif 'Otway' in suffix:
        mi = rs_vars(mi_path,
                  'MI', flux.time, time_start, time_end, dict(latitude=-38.45, longitude=idx['longitude']))
    
    else:
        mi = rs_vars(mi_path,
              'MI', flux.time, time_start, time_end, idx)

    if verbose:
        print('   Extracting Climate')
    solar = climate_vars('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/solar_monthly_wm2_2000_2021.nc',
                        flux.time, {'solar_exposure_day':'solar'}, time_start, time_end, idx)
    
    tavg = climate_vars('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/tavg_monthly_1991_2021.nc',
                        flux.time, {'temp_avg_month':'Ta'}, time_start, time_end, idx)
    
    vpd = climate_vars('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/vpd_monthly_2000_2021.nc',
                        flux.time, {'VPD':'VPD'}, time_start, time_end, idx)
    
    rain =  climate_vars('/g/data/os22/chad_tmp/NEE_modelling/data/5km/chirps_5km_monthly_1991_2021.nc',
                        flux.time, {'precip':'precip'}, time_start, time_end, idx)
    
    rain_cml_3 =  climate_vars('/g/data/os22/chad_tmp/NEE_modelling/data/5km/chirps_cml3_5km_monthly_1991_2021.nc',
                        flux.time, {'precip_cml_3':'precip_cml_3'}, time_start, time_end, idx)

    rain_cml_6 =  climate_vars('/g/data/os22/chad_tmp/NEE_modelling/data/5km/chirps_cml6_5km_monthly_1991_2021.nc',
                    flux.time, {'precip_cml_6':'precip_cml_6'}, time_start, time_end, idx)
    
    # join all the datasets
    all_dfs = dffs+[solar,tavg,vpd,rain,rain_cml_3,rain_cml_6]
    df_rs = lai.join(all_dfs)
                      
    df_rs = df_rs.add_suffix('_RS') 
    df = df_ec.join(df_rs)
    
    if return_coords:
        df['x_coord'] = lon
        df['y_coord'] = lat
    
    # add a LST-Tair using EC air temp instead of RS air temp
    df['LST-Tair_EC'] = (df['LST_RS']- 273.15) - df['Ta_EC']
    
    df.to_csv('/g/data/os22/chad_tmp/NEE_modelling/results/training_data/'+suffix[0:5]+'_training_data.csv')

    return df
    
