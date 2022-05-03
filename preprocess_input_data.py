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

def preprocess_data_gridded(base, suffix, verbose=False):
    # load flux data from site
    flux = xr.open_dataset(base+suffix)
    
    if "WallabyCreek" in suffix:
        #clip to before the fire
        flux = flux.sel(time=slice('2005', '2009'))
    
    #indexing values
    lat = flux.latitude.values[0]
    lon = flux.longitude.values[0]
    time_start = np.datetime_as_string(flux.time.values[0], unit='D')
    time_end = np.datetime_as_string(flux.time.values[-1], unit='D')
    idx=dict(latitude=lat,  longitude=lon)
    
    # Leaf Area Index from MODIS
    if verbose:
        print('   Extracting MODIS LAI')
    lai = lai['500m_lai'].rename('lai') #tidy up the dataset
    lai = lai.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    lai = lai.where((lai <= 10) & (lai >=0)) #remove artefacts and 'no-data'
    lai = lai.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean() # resample to monthly
    lai = lai.reindex(time=flux.time, method='nearest')# ensure lai matches flux

    # stack all variables
    nee = flux.NEE_LT.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
#     ts = flux.Ts.to_dataframe().reset_index(
#         level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    ta = flux.Ta.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    sws = flux.Sws.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    rh = flux.RH.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    vp = flux.VP.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    fn = flux.Fn.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    prec = flux.Precip.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
    lai = lai.to_dataframe().reset_index(
        level=[1, 2]).drop(['latitude', 'longitude'], axis=1)

    df = nee.join([ta, sws, rh, vp, fn, prec, lai])
    df = df.dropna()

    # calculate VPD
    df['VPD'] = VPD(df.RH, df.Ta)
    df = df.drop(['VP', 'RH'], axis=1)  # drop VP

    # corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # fig,ax=plt.subplots(1,1, figsize=(8,8))
    # sb.heatmap(corr, cmap="bwr_r", annot=True, ax=ax, cbar=False, mask=mask);

    #df = df.resample('Q-DEC').mean()

    # add lags
    df_lag1 = df.drop('NEE_LT', axis=1).shift(1)
    df = df.join(df_lag1, rsuffix='_L1')
    df = df.dropna()

    return df
