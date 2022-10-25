from typing import List, Sequence
import dask.array as da
import xarray as xr
import numpy as np
from dask.base import tokenize
import dask
from functools import partial

def keep_good_np(xx, where, nodata, out=None):
    if out is None:
        out = np.full_like(xx, nodata)
    else:
        assert out.shape == xx.shape
        assert out.dtype == xx.dtype
        assert out is not xx
        out[:] = nodata
    np.copyto(out, xx, where=where)
    return out

def np_percentile(xx, percentile, nodata):

    if np.isnan(nodata):
        high = True
        mask = ~np.isnan(xx)
    else:
        high = nodata >= xx.max()
        mask = xx != nodata

    valid_counts = mask.sum(axis=0)

    xx = np.sort(xx, axis=0)

    indices = np.round(percentile * (valid_counts - 1))
    if not high:
        indices += xx.shape[0] - valid_counts
        indices[valid_counts == 0] = 0

    indices = indices.astype(np.int64).flatten()
    step = xx.size // xx.shape[0]
    indices = step * indices + np.arange(len(indices))

    xx = xx.take(indices).reshape(xx.shape[1:])

    return keep_good_np(xx, (valid_counts >= 3), nodata)


def xr_quantile(
    src: xr.DataArray,
    quantiles: Sequence,
    nodata,
) -> xr.DataArray:

    """
    Calculates the percentiles of the input data along the time dimension.

    This approach is approximately 700x faster than the `numpy` and `xarray` nanpercentile functions.

    :param src: xr.Dataset, bands can be either
        float or integer with `nodata` values to indicate gaps in data.
        `nodata` must be the largest or smallest values in the dataset or NaN.

    :param percentiles: A sequence of quantiles in the [0.0, 1.0] range

    :param nodata: The `nodata` value
    """
    data_vars={}
    xx_data = src.data
    out_dims = ("quantile",) + src.dims[1:]

    # if dask.is_dask_collection(xx_data):
    #     xx_data = xx_data.rechunk({'time': -1})

    tk = tokenize(xx_data, quantiles, nodata)
    data = []
    for quantile in quantiles:
        name = f"pc_{int(100 * quantile)}"
        if dask.is_dask_collection(xx_data):
            yy = da.map_blocks(
                partial(np_percentile, percentile=quantile, nodata=nodata),
                xx_data,
                drop_axis=0,
                meta=np.array([], dtype=src.dtype),
                name=f"{name}-{tk}",
            )
        else:
            yy = np_percentile(xx_data, percentile=quantile, nodata=nodata)
        data.append(yy)

    if dask.is_dask_collection(yy):
        data_vars['band'] = (out_dims, da.stack(data, axis=0))
    else:
        data_vars['band'] = (out_dims, np.stack(data, axis=0))

    coords = dict((dim, src.coords[dim]) for dim in src.dims[1:])
    coords["quantile"] = np.array(quantiles)
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=src.attrs)
