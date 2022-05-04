import xarray as xr
import numpy as np
import pandas as pd
import datacube
import os
import sys
sys.path.append('/g/data/os22/chad_tmp/dea-notebooks/Tools/dea_tools/')
from datahandling import load_ard, mostcommon_crs
from bandindices import calculate_indices
from classification import HiddenPrints

def VPD(rh, ta):
    sat_vp = (6.11 * xr.ufuncs.exp((2500000/461) * (1/273 - 1/(273 + ta))))
    vpd = (((100 - rh)/100) * sat_vp)
    return vpd / 10  # go from mb to kPA


def preprocess_data_insitu(base, suffix):
    # load flux data from site
    flux = xr.open_dataset(base+suffix)
    
    if "WallabyCreek" in suffix:
        #clip to before the fire
        flux = flux.sel(time=slice('2005', '2009'))
    
    # Leaf Area Index from MODIS
    base = '/g/data/ub8/au/MODIS/mosaic/MOD15A2H.006/'
    lai = xr.open_mfdataset([base+i for i in os.listdir(base) if not 'quality' in i])
    
    # indexing values
    lat, lon = flux.latitude.values[0], flux.longitude.values[0]
    idx=dict(latitude=lat,  longitude=lon)
    time_start = np.datetime_as_string(flux.time.values[0], unit='D')
    time_end = np.datetime_as_string(flux.time.values[-1], unit='D')
    
    lai = lai['500m_lai'].rename('lai') #tidy up the dataset
    lai = lai.sel(idx, method='nearest').sel(time=slice(str(time_start), str(time_end))) # grab pixel
    lai = lai.where((lai <= 10) & (lai >=0)) #remove artefacts and 'no-data'
    lai = lai.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean() # resample to monthly
    lai = lai.reindex(time=flux.time, method='nearest')# ensure lai matches flux

    # stack all variables
    nee = flux.NEE_LT.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    ta = flux.Ta.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    sws = flux.Sws.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    rh = flux.RH.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    vp = flux.VP.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    # fn = flux.Fn.to_dataframe().reset_index(
    #     level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    prec = flux.Precip.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    lai = lai.to_dataframe().drop(['latitude', 'longitude'], axis=1)

    df = nee.join([ta, sws, rh, vp, prec, lai])
    df = df.dropna()

    # calculate VPD
    df['VPD'] = VPD(df.RH, df.Ta)
    df = df.drop(['VP', 'RH'], axis=1)  # drop VP

    # corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # fig,ax=plt.subplots(1,1, figsize=(8,8))
    # sb.heatmap(corr, cmap="bwr_r", annot=True, ax=ax, cbar=False, mask=mask);

    # add lags
    df_lag1 = df.drop('NEE_LT', axis=1).shift(1)
    df = df.join(df_lag1, rsuffix='_L1')
    df = df.dropna()

    return df

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def preprocess_data_gridded(suffix, verbose=False):
    """
    Extract pixels from gridded RS/Climate data over the location
    of EC flux towers.
    
    return EC flux tower estimates of GPP, RE, NEE and climate/rs
    co-variables
     - AWRA climate
         - solar, rain, tavg, vpd
     - fPAR
     - GRAFS soil moisture
         - add 1-month lag
     - LAI
     - LST
     - SPEI
     - lst-tair
     - add lags on
     
    """
    base = 'https://dap.tern.org.au/thredds/dodsC/ecosystem_process/ozflux/'
    # load flux data from site
    flux = xr.open_dataset(base+suffix)
    
    # if "WallabyCreek" in suffix:
    #     #clip to before the fire
    #     flux = flux.sel(time=slice('2005', '2009'))
    
    #indexing spatiotemporal values at EC site
    lat = flux.latitude.values[0]
    lon = flux.longitude.values[0]
    time_start = str(np.datetime_as_string(flux.time.values[0], unit='D'))
    time_end = str(np.datetime_as_string(flux.time.values[-1], unit='D'))
    idx=dict(latitude=lat,  longitude=lon)
    
    # extract co2 fluxes from EC data
    nee = flux.NEE_SOLO.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    gpp = flux.GPP_SOLO.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    er = flux.ER_SOLO.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    
    # Leaf Area Index from MODIS
    if verbose:
        print('   Extracting MODIS LAI')
        # Leaf Area Index from MODIS
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/LAI/'
    lai = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    lai = lai.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    lai = lai.reindex(time=flux.time, method='nearest').compute()
    lai = lai.lai.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # LST from MODIS
    if verbose:
        print('   Extracting MODIS LST')
        # Leaf Area Index from MODIS
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/LST/'
    lst = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    lst = lst.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    lst = lst.reindex(time=flux.time, method='nearest').compute() 
    lst = lst.LST.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)

    # fPAR from MODIS
    if verbose:
        print('   Extracting MODIS fPAR')
        # Leaf Area Index from MODIS
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/FPAR/'
    fpar = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    fpar = fpar.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    fpar = fpar.reindex(time=flux.time, method='nearest').compute() 
    fpar = fpar.Fpar.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)

    # soil moisture from GRAFS
    if verbose:
        print('   Extracting GRAFS SM')
    base = '/g/data/os22/chad_tmp/NEE_modelling/data/GRAFS/'
    sm = xr.open_mfdataset([base+i for i in os.listdir(base)], chunks=dict(time=1))
    sm = sm.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    sm = sm.reindex(time=flux.time, method='nearest').compute() # ensure lai matches flux
    sm = sm.soil_moisture.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # Delta temp
    if verbose:
        print('   Extracting dT')
    dT = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/LST_Tair_2002_2021.nc')
    dT = dT.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    dT = dT.reindex(time=flux.time, method='nearest').compute() # ensure lai matches flux
    dT = dT['LST-Tair'].to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # SPEI
    if verbose:
        print('   Extracting SPEI')
    spei = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/SPEI/AWRA_spei_5km_gamma_06.nc')
    spei = spei.rename({'lat':'latitude', 'lon':'longitude'})
    spei = spei.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    spei = spei.reindex(time=flux.time, method='nearest').compute() # ensure lai matches flux
    spei = spei.spei_gamma_06.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # Climate
    if verbose:
        print('   Extracting AWRA Climate')
    solar = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/solar_monthly_wm2_2000_2021.nc')
    solar = solar.rename({'solar_exposure_day':'solar_exposure_month'})
    solar = solar.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    solar = solar.reindex(time=flux.time, method='nearest').compute() 
    solar = solar.solar_exposure_month.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    tavg = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/tavg_monthly_1991_2021.nc')
    tavg = tavg.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    tavg = tavg.reindex(time=flux.time, method='nearest').compute() 
    tavg = tavg.temp_avg_month.to_dataframe().drop(['latitude', 'longitude'], axis=1)
    
    vpd = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/vpd_monthly_2000_2021.nc')
    vpd = vpd.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    vpd = vpd.reindex(time=flux.time, method='nearest').compute() 
    vpd = vpd.VPD.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    rain = xr.open_dataset('/g/data/os22/chad_tmp/NEE_modelling/data/AWRA/rain_monthly_1991_2021.nc')
    rain = rain.sel(idx, method='nearest').sel(time=slice(time_start, time_end))
    rain = rain.reindex(time=flux.time, method='nearest').compute() 
    rain = rain.rain_month.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    
    # join all the datasets
    df = nee.join([gpp,er,lai,lst,fpar,sm,dT,spei,solar,tavg,vpd,rain])
    df = df.dropna()
    
    df.to_csv('/g/data/os22/chad_tmp/NEE_modelling/results/training_data/'+suffix[0:5]+'_training_data.csv')
    
    # add lags
    # df_lag1 = df.drop('NEE_LT', axis=1).shift(1)
    # df = df.join(df_lag1, rsuffix='_L1')
    # df = df.dropna()

    return df
