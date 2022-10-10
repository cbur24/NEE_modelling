import os
import sys
import xarray as xr
import numpy as np
import pandas as pd

def VPD(rh, ta):
    sat_vp = (6.11 * np.exp((2500000/461) * (1/273 - 1/(273 + ta))))
    vpd = (((100 - rh)/100) * sat_vp)
    return vpd

def extract_ec_vars(flux, var):
    df = flux[var].to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    return df

def extract_rs_vars(path, flux_time, time_start, time_end, idx, add_comparisons=False):
    if add_comparisons:
        ds = xr.open_dataset(path)
    else:
        ds = xr.open_dataarray(path)

    ds = ds.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    ds = ds.reindex(time=flux_time, method='nearest', tolerance='1D').compute() 

    try:
        ds = ds.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    except:
        ds = ds.to_dataframe().drop(['latitude', 'longitude'], axis=1)
    
    return ds

def extract_ec_gridded_data(suffix,
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
                                 'rain_cml6_anom',
                                 'rain_cml12_anom',
                                 #'CWD',
                                 'srad_anom',
                                 'vpd',
                                 #'tavg',
                                 'tavg_anom',
                                 'SOC',
                                 'CO2'
                                 #'FireDisturbance'
                            ],
                            add_comparisons=False,
                            save_ec_data=False,
                            return_coords=True,
                            verbose=False
                           ):
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
    #-----Eddy covaraince data--------------------------------------------
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

    # extract carbon fluxes and environ data from EC data
    if verbose:
        print('   Extracting EC data')
    
    variables = ['GPP_SOLO','ER_SOLO','ET','Ta','Sws','RH','VP','Precip','Fn','Fe','Fh','Fsd','Fld','CO2']
    nee = extract_ec_vars(flux, 'NEE_SOLO') #extract first variable
    df_ec=[]
    for var in variables: #loop through other vars
        df = extract_ec_vars(flux, var)
        df_ec.append(df)
    
    df_ec = nee.join(df_ec) #join other vars to NEE
    df_ec = df_ec.add_suffix('_EC')
 
    # calculate VPD on ec data
    df_ec['VPD_EC'] = VPD(df_ec.RH_EC, df_ec.Ta_EC)
    df_ec = df_ec.drop(['VP_EC'], axis=1) # drop VP
    
    #--------Remote sensing data--------------------------------------
    
    # extract the first remote sensing variable
    first_var = covariables[0]
    if verbose:
        print('   Extracting '+first_var)
    first = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/{scale}/{first_var}_{scale}_monthly_2002_2021.nc',
                  flux.time, time_start, time_end, idx)
    
    #extract the rest of the RS variables in loop    
    dffs = []
    for var in covariables[1:]:
        if verbose:
            print(f'   Extracting {var}')
            
        df = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/{scale}/{var}_{scale}_monthly_2002_2021.nc',
                   flux.time, time_start, time_end, idx)
        
        dffs.append(df)
    
    # join all the datasets
    df_rs = first.join(dffs)
                      
    df_rs = df_rs.add_suffix('_RS') 
    df = df_ec.join(df_rs)
    
    if return_coords:
        df['x_coord'] = lon
        df['y_coord'] = lat
    
    time = df.reset_index()['time'].dt.normalize()
    df = df.set_index(time)
    
    if add_comparisons:
        others_gpp = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/harmonized_gpp.nc',
                   time, time_start, time_end, idx, add_comparisons=add_comparisons)
        
        others_nee = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/harmonized_nee.nc',
                   time, time_start, time_end, idx, add_comparisons=add_comparisons)
        
        df = pd.merge(df, others_nee, left_index=True, right_index=True)
        df = pd.merge(df, others_gpp, left_index=True, right_index=True)
        
    # add a LST-Tair using EC air temp instead of RS air temp
    #df['LST-Tair_EC'] = (df['LST_RS']- 273.15) - df['Ta_EC']
    
    df.to_csv('/g/data/os22/chad_tmp/NEE_modelling/results/training_data/'+suffix[0:5]+'_training_data.csv')

    return df
    
