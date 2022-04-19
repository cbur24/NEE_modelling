import xarray as xr
import numpy as np
import pandas as pd
import datacube

from dea_tools.datahandling import load_ard, mostcommon_crs
from dea_tools.bandindices import calculate_indices
from dea_tools.classification import HiddenPrints


def VPD(rh, ta):
    sat_vp = (6.11 * xr.ufuncs.exp((2500000/461) * (1/273 - 1/(273 + ta))))
    vpd = (((100 - rh)/100) * sat_vp)
    return vpd / 10  # go from mb to kPA


def preprocess_data(base, suffix):
    # load flux data from site
    flux = xr.open_dataset(base+suffix)

    # load satellite data from datacuve
    dc = datacube.Datacube(app='NEE flux model')

    query = {
        'y': flux.latitude.values[0],
        'x': flux.longitude.values[0],
        'time': (np.datetime_as_string(flux.time.values[0], unit='D'),
                 np.datetime_as_string(flux.time.values[-1], unit='D')),
        'measurements': ['nbart_red', 'nbart_nir', 'nbart_blue'],
        'resolution': (-100, 100),
    }

    # Identify the most common projection system in the input query
    output_crs = mostcommon_crs(dc=dc, product='ga_ls8c_ard_3', query=query)

    # Load available data from all three Landsat satellites
    with HiddenPrints():
        ds = load_ard(dc=dc,
                      products=['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'],
                      #mask_filters = [("opening", 4), ("dilation", 2)],
                      output_crs=output_crs,
                      group_by='solar_day',
                      **query
                      )

    lai = calculate_indices(ds, 'LAI', collection='ga_ls_3', drop=True).LAI
    lai = lai.resample(time='MS').median()
    lai = lai.reindex(time=flux.time, method='nearest')

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
        level=[1, 2]).drop(['y', 'x', 'spatial_ref'], axis=1)

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
    df = df.join(df_lag1, rsuffix='_L2')
    df = df.dropna()

    return df
