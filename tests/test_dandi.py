import pynwb

from pynwb import NWBHDF5IO, TimeSeries

from dandi.dandiapi import DandiAPIClient
import pynapple as nap
import numpy as np
import jax.numpy as jnp
import fsspec
from fsspec.implementations.cached import CachingFileSystem

import pynwb
import h5py

from matplotlib.pylab import *

#####################################
# Dandi
#####################################

# ecephys, Buzsaki Lab (15.2 GB)
dandiset_id, filepath = "000582", "sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb"


with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)




# first, create a virtual filesystem based on the http protocol
fs=fsspec.filesystem("http")

# create a cache to save downloaded data to disk (optional)
fs = CachingFileSystem(
    fs=fs,
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
file = h5py.File(fs.open(s3_url, "rb"))
io = pynwb.NWBHDF5IO(file=file, load_namespaces=True)


#####################################
# Pynapple
#####################################

nwb = nap.NWBFile(io.read())

units = nwb["units"]

position = nwb["SpatialSeriesLED1"]

tc, binsxy = nap.compute_2d_tuning_curves(units, position, 15)


figure()
for i in tc.keys():
    subplot(3,3,i+1)
    imshow(tc[i])
#show()

figure()
for i in units.keys():
    subplot(3,3,i+1)
    plot(position['x'], position['y'])
    spk_pos = units[i].value_from(position)
    plot(spk_pos["x"], spk_pos["y"], 'o', color = 'red', markersize = 1, alpha = 0.5)

show()


#####################################
# GLM
#####################################
# create the binning
t0 = position.time_support.start[0]
tend = position.time_support.end[0]
ts = np.arange(t0-0.01, tend+0.01, 0.02)
binning = nap.IntervalSet(start=ts[:-1], end=ts[1:], time_units='s')

# bin and convert to jax array
counts = jnp.asarray(units.count(ep=binning))
position_binned = jnp.asarray(position.restrict(binning))

