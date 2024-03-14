#!/usr/bin/env python3

import math
import os
import os.path as op
from typing import Union

import click
import numpy as np
import pynapple as nap
import requests
import tqdm.auto as tqdm

TsdType = Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]

import fsspec
import h5py
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem

# Dandi stuffs
from pynwb import NWBHDF5IO


def download_data(filename, url, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    filename = op.join(data_dir, filename)
    if not os.path.exists(filename):
        r = requests.get(url, stream=True)
        block_size = 1024*1024
        with open(filename, "wb") as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit="MB", unit_scale=True,
                                  total=math.ceil(int(r.headers.get("content-length", 0))//block_size)):
                f.write(data)
    return filename


def download_dandi_data(dandiset_id, filepath):
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    
    # first, create a virtual filesystem based on the http protocol
    fs = fsspec.filesystem("http")

    # create a cache to save downloaded data to disk (optional)
    fs = CachingFileSystem(
        fs=fs,
        cache_storage="nwb-cache",  # Local folder for the cache
    )

    # next, open the file
    file = h5py.File(fs.open(s3_url, "rb"))
    io = NWBHDF5IO(file=file, load_namespaces=True)

    return io



def fill_forward(time_series, data, ep=None, out_of_range=np.nan):
    """
    Fill a time series forward in time with data.

    Parameters
    ----------
    time_series:
        The time series to match.
    data: Tsd, TsdFrame, or TsdTensor
        The time series with data to be extend.

    Returns
    -------
    : Tsd, TsdFrame, or TsdTensor
        The data time series filled forward.

    """
    assert isinstance(data, TsdType)

    if ep is None:
        ep = time_series.time_support
    else:
        assert isinstance(ep, nap.IntervalSet)
        time_series.restrict(ep)

    data = data.restrict(ep)
    starts = ep.start
    ends = ep.end

    filled_d = np.full((time_series.t.shape[0], *data.shape[1:]), out_of_range, dtype=data.dtype)
    fill_idx = 0
    for start, end in zip(starts, ends):
        data_ep = data.get(start, end)
        ts_ep = time_series.get(start, end)
        idxs = np.searchsorted(data_ep.t, ts_ep.t, side="right") - 1
        filled_d[fill_idx:fill_idx + ts_ep.t.shape[0]][idxs >= 0] = data_ep.d[idxs[idxs>=0]]
        fill_idx += ts_ep.t.shape[0]
    return type(data)(t=time_series.t, d=filled_d, time_support=ep)


@click.command()
@click.argument('data_dir')
def main(data_dir):
    download_data("allen_478498617.nwb", "https://osf.io/vf2nj/download",
                  data_dir)
    download_data("m691l1.nwb", "https://osf.io/xesdm/download",
                  data_dir)
    download_data("Mouse32-140822.nwb", "https://osf.io/jb2gd/download",
                  data_dir)


if __name__ == '__main__':
    main()
