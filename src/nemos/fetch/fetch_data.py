"""
Module for fetching data and utilities using the pooch library.

This module allows you to download datasets and utility scripts
from specified URLs and manage their versions and integrity using
SHA256 hash verification. It also provides helper functions to
calculate hashes for local files and manage file paths.
"""

import os
import pathlib
from typing import Optional, Union

try:
    import pooch
    from pooch import Pooch
    from tqdm.auto import tqdm
except ImportError:
    pooch = None
    Pooch = None
    tqdm = None

try:
    import dandi
    import fsspec
    import h5py
    from dandi.dandiapi import DandiAPIClient
    from fsspec.implementations.cached import CachingFileSystem
    from pynwb import NWBHDF5IO
except ImportError:
    dandi = None
    NWBHDF5IO = None


# Registry of dataset filenames and their corresponding SHA256 hashes.
REGISTRY_DATA = {
    "A0670-221213.nwb": "8587dd6dde107504bd4a17a68ce8fb934fcbcccc337e653f31484611ee51f50a",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "Achilles_10252013.nwb": "42857015aad4c2f7f6f3d4022611a69bc86d714cf465183ce30955731e614990",
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "m691l1.nwb": "1990d8d95a70a29af95dade51e60ffae7a176f6207e80dbf9ccefaf418fe22b6",
    "A2929-200711.nwb": "f698d7319efa5dfeb18fb5fe718ec1a84fdf96b85a158177849a759cd5e396fe",
}
DOWNLOADABLE_FILES = list(REGISTRY_DATA.keys())

# URL templates for downloading datasets and utility scripts.
OSF_TEMPLATE = "https://osf.io/{}/download"

# Mapping of dataset filenames to their download URLs.
REGISTRY_URLS_DATA = {
    "A0670-221213.nwb": OSF_TEMPLATE.format("sbnaw"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm"),
    "A2929-200711.nwb": OSF_TEMPLATE.format("y7zwd"),
}

_NEMOS_ENV = "NEMOS_DATA_DIR"


def _create_retriever(path: Optional[pathlib.Path] = None) -> Pooch:
    """Create a pooch retriever for fetching datasets.

    This function sets up the pooch retriever, which manages the
    downloading and caching of files, including handling retries
    and checking file integrity using SHA256 hashes.

    Parameters
    ----------
    path :
        The directory where datasets will be stored. If not provided,
        defaults to pooch's cache (check ``pooch.os_cache('nemos')`` for that path)

    Returns
    -------
    :
        A configured pooch retriever object.

    """
    if path is None:
        # Use the default data directory if none is provided.
        path = pooch.os_cache("nemos")

    return pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS_DATA,
        registry=REGISTRY_DATA,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
        env=_NEMOS_ENV,
    )


def fetch_data(
    dataset_name: str, path: Optional[Union[pathlib.Path, str]] = None
) -> str:
    """
    Download a dataset using pooch.

    This function downloads a dataset, checking if it already exists
    and is unchanged. If the dataset is an archive (ends in .tar.gz),
    it decompresses the archive and returns the path to the resulting
    directory. Otherwise, it returns the path to the downloaded file.

    Parameters
    ----------
    dataset_name :
        The name of the dataset to download. Must match an entry in
        REGISTRY_DATA.
    path :
        The directory where the dataset will be stored. If not provided,
        defaults to _DEFAULT_DATA_DIR.

    Returns
    -------
    :
        The path to the downloaded file or directory.
    """
    if pooch is None:
        raise ImportError(
            "Missing optional dependency 'pooch'."
            " Please use pip or "
            "conda to install 'pooch'."
        )
    retriever = _create_retriever(path)
    # Fetch the dataset using pooch.
    return retriever.fetch(dataset_name)


def download_dandi_data(dandiset_id: str, filepath: str) -> NWBHDF5IO:
    """Download a dataset from the DANDI Archive (https://dandiarchive.org/)

    Parameters
    ----------
    dandiset_id :
        6-character string of numbers giving the ID of the dandiset.
    filepath :
        filepath to the specific .nwb file within the dandiset we wish to return.

    Returns
    -------
    io :
        NWB file containing specified data.

    Examples
    --------
    >>> import nemos as nmo
    >>> import pynapple as nap
    >>> io = nmo.fetch.download_dandi_data("000582",
    ...                                    "sub-11265/sub-11265_ses-07020602_behavior+ecephys.nwb")
    >>> nwb = nap.NWBFile(io.read(), lazy_loading=False)
    >>> print(nwb)
    07020602
    ┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┑
    │ Keys                │ Type     │
    ┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━┥
    │ units               │ TsGroup  │
    │ ElectricalSeriesLFP │ Tsd      │
    │ SpatialSeriesLED2   │ TsdFrame │
    │ SpatialSeriesLED1   │ TsdFrame │
    │ ElectricalSeries    │ Tsd      │
    ┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙

    """
    if dandi is None:
        raise ImportError(
            "Missing optional dependency 'dandi'."
            " Please use pip or "
            "conda to install 'dandi'."
        )
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # first, create a virtual filesystem based on the http protocol
    fs = fsspec.filesystem("http")

    # create a cache to save downloaded data to disk (optional)
    # mimicking caching behavior of pooch create
    if _NEMOS_ENV in os.environ:
        cache_dir = pathlib.Path(os.environ[_NEMOS_ENV])
    else:
        cache_dir = pooch.os_cache("nemos") / "nwb-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fs = CachingFileSystem(
        fs=fs,
        cache_storage=cache_dir.as_posix(),  # Local folder for the cache
    )

    # next, open the file
    file = h5py.File(fs.open(s3_url, "rb"))
    io = NWBHDF5IO(file=file, load_namespaces=True)

    return io
