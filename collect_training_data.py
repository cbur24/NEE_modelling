import os
import sys
import xarray as xr
import numpy as np
import pandas as pd

def VPD(rh, ta):
    sat_vp = (6.11 * np.exp((2500000/461) * (1/273 - 1/(273 + ta))))
    vpd = (((100 - rh)/100) * sat_vp)
    return vpd

def extract_var(flux, var):
    df = flux[var].to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    return df

def extract_ec_gridded_data(suffix, verbose=False):
    """
    Extract variables from EC tower data, and environmental
    data from remote sensing/climate datasets over pixels at EC
    tower location.
    
    Returns:
        Pandas.Dataframe containing coincident observations between
        EC data and gridded data
    """
    
    base = 'https://dap.tern.org.au/thredds/dodsC/ecosystem_process/ozflux/'
    
    # load flux data from site
    flux = xr.open_dataset(base+suffix)
    
    # offset time to better match gridded data
    flux['time'] = flux.time + np.timedelta64(14,'D') 
    
    # if "WallabyCreek" in suffix:
    #     #clip to before the fire
    #     flux = flux.sel(time=slice('2005', '2009'))
    
    #indexing spatiotemporal values at EC site
    lat = flux.latitude.values[0]
    lon = flux.longitude.values[0]
    time_start = str(np.datetime_as_string(flux.time.values[0], unit='D'))
    time_end = str(np.datetime_as_string(flux.time.values[-1], unit='D'))
    idx=dict(latitude=lat,  longitude=lon)
    
    # extract co2 fluxes and environ data from EC data
    if verbose:
        print('   Extracting EC data')
    nee = extract_var(flux, 'NEE_SOLO')
    gpp = extract_var(flux, 'GPP_SOLO')
    er = extract_var(flux, 'ER_SOLO')
    ta = extract_var(flux, 'Ta')
    sws = extract_var(flux, 'Sws')
    rh = extract_var(flux, 'RH')
    vp = extract_var(flux, 'VP')
    prec = extract_var(flux, 'Precip')
    fn = extract_var(flux, 'Fn')
    fe = extract_var(flux, 'Fe')
    fh = extract_var(flux, 'Fh')
    fsd = extract_var(flux, 'Fsd')
    fld = extract_var(flux, 'Fld')
    
    df_ec = nee.join([gpp,er,ta,sws,rh,vp,prec,fe,fh,fsd,fn,fld])
    df_ec = df_ec.add_suffix('_EC')
    
    # calculate VPD
    df_ec['VPD_EC'] = VPD(df_ec.RH_EC, df_ec.Ta_EC)
    df_ec = df_ec.drop(['VP_EC'], axis=1) # drop VP
    
    # Leaf Area Index from MODIS
    if verbose:
        print('   Extracting MODIS LAI')
        # Leaf Area Index from MODIS
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/LAI/'
    lai = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    lai = lai.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    lai = lai.reindex(time=flux.time, method='nearest', tolerance='1D').compute()
    lai = lai.lai.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # LST from MODIS
    if verbose:
        print('   Extracting MODIS LST')
        # Leaf Area Index from MODIS
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/LST/'
    lst = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    lst = lst.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    lst = lst.reindex(time=flux.time, method='nearest', tolerance='1D').compute() 
    lst = lst.LST.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # fPAR from MODIS
    if verbose:
        print('   Extracting MODIS fPAR')
        # Leaf Area Index from MODIS
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/FPAR/'
    fpar = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    fpar = fpar.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    fpar = fpar.reindex(time=flux.time, method='nearest', tolerance='1D').compute() 
    fpar = fpar.Fpar.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # soil moisture from GRAFS
    if verbose:
        print('   Extracting GRAFS SM')
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/GRAFS/'
    sm = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    sm = sm.sel(idx, method='nearest').sel(time=slice(time_start, time_end)).compute() # grab pixel
    sm = sm.reindex(time=flux.time, method='nearest', tolerance='1D').compute()
    sm = sm.soil_moisture.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # Delta temp
    if verbose:
        print('   Extracting dT')
    dT = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/LST_Tair_2002_2021.nc')
    dT = dT.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    dT = dT.reindex(time=flux.time, method='nearest', tolerance='1D').compute()
    dT = dT['LST-Tair'].to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # SPEI
    if verbose:
        print('   Extracting SPEI')
    spei = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/SPEI/AWRA_spei_5km_gamma_06.nc')
    spei = spei.rename({'lat':'latitude', 'lon':'longitude'})
    spei = spei.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    spei = spei.reindex(time=flux.time, method='nearest', tolerance='1D').compute() # ensure lai matches flux
    spei = spei.spei_gamma_06.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # Climate
    if verbose:
        print('   Extracting AWRA Climate')
    solar = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/solar_monthly_wm2_2000_2021.nc')
    solar = solar.rename({'solar_exposure_day':'solar_exposure_month'})
    solar = solar.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    solar = solar.reindex(time=flux.time, method='nearest', tolerance='1D').compute() 
    solar = solar.solar_exposure_month.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    tavg = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/tavg_monthly_1991_2021.nc')
    tavg = tavg.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    tavg = tavg.reindex(time=flux.time, method='nearest', tolerance='1D').compute() 
    tavg = tavg.temp_avg_month.to_dataframe().drop(['latitude', 'longitude'], axis=1)
    
    vpd = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/vpd_monthly_2000_2021.nc')
    vpd = vpd.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    vpd = vpd.reindex(time=flux.time, method='nearest', tolerance='1D').compute() 
    vpd = vpd.VPD.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)

    rain = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/rain_monthly_1991_2021.nc')
    rain = rain.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    rain = rain.reindex(time=flux.time, method='nearest', tolerance='1D').compute() 
    rain = rain.rain_month.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    #add lags to rainfall
    rain_l1 = rain.shift(1).rename({'rain_month':'rain_month_L1'},axis=1)
    rain_l2 = rain.shift(2).rename({'rain_month':'rain_month_L2'},axis=1)
    rain_l3 = rain.shift(3).rename({'rain_month':'rain_month_L3'},axis=1)
    rain = rain.join([rain_l1,rain_l2,rain_l3])
    
    # join all the datasets
    df_rs = lai.join([lst,fpar,sm,dT,spei,solar,tavg,vpd,rain])
    df_rs = df_rs.add_suffix('_RS') 
    df = df_ec.join(df_rs)
    
    df.to_csv('/g/data/os22/chad_tmp/NEE_modelling/results/training_data/'+suffix[0:3]+'_training_data.csv')

    return df
