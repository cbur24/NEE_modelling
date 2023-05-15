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
        if 'quantiles' in path:
            ds = xr.open_dataset(path).sel(quantile=0.5).drop('quantile')
        else:
            ds = xr.open_dataset(path)
        
        if 'FLUXCOM' in path:
            ds = ds*30
            
        if 'meteo_era5' in path:
            ds = ds.rename({'lat':'latitude', 'lon':'longitude'})
            ds=ds[path[-20:-17]]

        try:
            ds = ds.rename({'y':'latitude', 'x':'longitude'})
        
        except:
            pass
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
                                 #'rain_cml3',
                                 #'rain_cml6',
                                 #'rain_cml12',
                                 'rain_anom',
                                 'rain_cml3_anom',
                                 'rain_cml6_anom',
                                 'rain_cml12_anom',
                                 'srad',
                                 'srad_anom',
                                 'vpd',
                                 'tavg',
                                 'tavg_anom',
                                 'SOC',
                                 #'CO2',
                                 #'C4percent',
                                 #'Elevation',
                                 #'MOY',
                                 'VegH',
                                 #'MI'
                            ],
                            add_comparisons=False,
                            save_ec_data=False,
                            return_coords=True,
                            export_path=None,
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
    # base = 'https://dap.tern.org.au/thredds/dodsC/ecosystem_process/ozflux/'
    # base = 'https://dap.tern.org.au/thredds/fileServer/ecosystem_process/ozflux/'
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/ec_netcdfs/'
    
    # load flux data from site
    flux = xr.open_dataset(base+suffix)
    if save_ec_data:
        flux.to_netcdf('/g/data/os22/chad_tmp/NEE_modelling/data/ec_netcdfs/'+suffix)
    
    # Set negative GPP, ER, and ET measurements as zero
    flux['GPP_SOLO'] = xr.where(flux.GPP_SOLO < 0, 0, flux.GPP_SOLO)
    flux['ET'] = xr.where(flux.ET < 0, 0, flux.ET)
    flux['ER_SOLO'] = xr.where(flux.ER_SOLO < 0, 0, flux.ER_SOLO)
    
    # offset time to better match gridded data
    flux['time'] = flux.time + np.timedelta64(14,'D') 
    
    #indexing spatiotemporal values at EC site
    lat = flux.latitude.values[0]
    lon = flux.longitude.values[0]
    time_start = str(np.datetime_as_string(flux.time.values[0], unit='D'))
    time_end = str(np.datetime_as_string(flux.time.values[-1], unit='D'))
    
    if "TiTreeEast" in suffix: #metadata on nc file is wrong
        idx=dict(latitude=-22.287,  longitude=133.640)
    
    if "DalyPasture" in suffix: #metadata on nc file is wrong
        idx=dict(latitude=-14.0633,  longitude=131.3181)
    
    else:
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
    first = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/{scale}/{first_var}_{scale}_monthly_2002_2022.nc',
                  flux.time, time_start, time_end, idx)
    
    #extract the rest of the RS variables in loop    
    dffs = []
    for var in covariables[1:]:
        if verbose:
            print(f'   Extracting {var}')
         
        if var=='MI': #temporary just for testing
            df = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/5km/MI_5km_monthly_2002_2022.nc',
                   flux.time, time_start, time_end, idx)
            
        else:
            df = extract_rs_vars(f'/g/data/os22/chad_tmp/NEE_modelling/data/{scale}/{var}_{scale}_monthly_2002_2022.nc',
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
        others = {
            'MODIS_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/1km/MODIS_GPP_1km_monthly_2002_2021.nc',
            'GOSIF_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/5km/GOSIF_GPP_5km_monthly_2002_2021.nc',
            'DIFFUSE_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/1km/DIFFUSE_GPP_1km_monthly_2003_2021.nc',
            'CABLE_BIOS_NEE':'/g/data/os22/chad_tmp/NEE_modelling/data/CABLE/CABLE-BIOS/CABLE_BIOS_nbp_25km_monthly_2003_2019.nc',
            'CABLE_BIOS_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/CABLE/CABLE-BIOS/CABLE_BIOS_gpp_25km_monthly_2003_2019.nc',
            'CABLE_BIOS_ER':'/g/data/os22/chad_tmp/NEE_modelling/data/CABLE/CABLE-BIOS/CABLE_BIOS_er_25km_monthly_2003_2019.nc',
            'CABLE_POP_NEE':'/g/data/os22/chad_tmp/NEE_modelling/data/CABLE/CABLE-POP_v10/CABLE-POP_nbp_100km_monthly_2003_2020.nc',
            'CABLE_POP_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/CABLE/CABLE-POP_v10/CABLE-POP_gpp_100km_monthly_2003_2020.nc',
            'CABLE_POP_ER':'/g/data/os22/chad_tmp/NEE_modelling/data/CABLE/CABLE-POP_v10/CABLE-POP_er_100km_monthly_2003_2020.nc',
            'This_Study_NEE':'/g/data/os22/chad_tmp/NEE_modelling/results/predictions/AusEFlux_NEE_2003_2022_1km_quantiles_v1.0',
            'This_Study_GPP':'/g/data/os22/chad_tmp/NEE_modelling/results/predictions/AusEFlux_GPP_2003_2022_1km_quantiles_v1.0',
            'This_Study_ER':'/g/data/os22/chad_tmp/NEE_modelling/results/predictions/AusEFlux_ER_2003_2022_1km_quantiles_v1.0',
            'FLUXCOM_RS_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/FLUXCOM/GPP_rs.nc',
            'FLUXCOM_RS_NEE':'/g/data/os22/chad_tmp/NEE_modelling/data/FLUXCOM/NEE_rs.nc',
            'FLUXCOM_RS_ER':'/g/data/os22/chad_tmp/NEE_modelling/data/FLUXCOM/TER_rs.nc',
            'FLUXCOM_MET_GPP':'/g/data/os22/chad_tmp/NEE_modelling/data/FLUXCOM/GPP_rs_meteo_era5.nc',
            'FLUXCOM_MET_NEE':'/g/data/os22/chad_tmp/NEE_modelling/data/FLUXCOM/NEE_rs_meteo_era5.nc',
            'FLUXCOM_MET_ER':'/g/data/os22/chad_tmp/NEE_modelling/data/FLUXCOM/TER_rs_meteo_era5.nc'
        }
        other_dffs = []
        for prod in others.items():
            
            other = extract_rs_vars(prod[1],
                   time, time_start, time_end, idx, add_comparisons=add_comparisons)
            # print(other)
            other = other.rename({other.columns[0] : prod[0]}, axis=1)
            
            if prod[0]=='MODIS_GPP':
                other['MODIS_GPP'] = other['MODIS_GPP']*1000
            
            if prod[0]=='DIFFUSE_GPP':
                other['DIFFUSE_GPP'] = other['DIFFUSE_GPP']*30
            
            other_dffs.append(other)

        df = df.join(other_dffs)
        df = df.drop(['NEE_mad', 'GPP_mad', 'TER_mad'], axis=1)
    
    if export_path:
        df.to_csv(export_path+suffix[0:5]+'_training_data.csv')

    return df
    
