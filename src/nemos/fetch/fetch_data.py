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
except ImportError:
    pooch = None
    Pooch = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import dandi
    import fsspec
    import h5py
    from dandi.dandiapi import DandiAPIClient
    from pynwb import NWBHDF5IO
except ImportError:
    dandi = None
    NWBHDF5IO = None

import hashlib

# Registry of dataset filenames and their corresponding SHA256 hashes.
REGISTRY_DATA = {
    "A0670-221213.nwb": "8587dd6dde107504bd4a17a68ce8fb934fcbcccc337e653f31484611ee51f50a",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "Achilles_10252013.nwb": "42857015aad4c2f7f6f3d4022611a69bc86d714cf465183ce30955731e614990",
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "m691l1.nwb": "1990d8d95a70a29af95dade51e60ffae7a176f6207e80dbf9ccefaf418fe22b6",
    "A2929-200711.nwb": "f698d7319efa5dfeb18fb5fe718ec1a84fdf96b85a158177849a759cd5e396fe",
    "Achilles_10252013_EEG.nwb": "a97a69d231e7e91c07e24890225f8fe4636bac054de50345551f32fc46b9efdd",
    "em_three_states.npz": "92e9fe7990e98f3d23536a40658e258acbc83c26d773f78336431abc12e01951",
    "julia_regression_mstep_no_prior.npz": "412430a4e0d0bbc0abb224a3a8516f660266e3e5bc0d840c4ee6fd80b5501faa",
    "julia_regression_mstep_good_prior.npz": "e5bf87ca2f48b4904b0ce1c0e258f91c44033b05163e039f79b9b48c2c3e484d",
    "julia_regression_mstep_flat_prior.npz": "fdd1117c521945f25595f78f220256d37cb7bf37834403bd1cc04941e3203b22",
}
DOWNLOADABLE_FILES = list(REGISTRY_DATA.keys())

# URL templates for downloading datasets and utility scripts.
OSF_TEMPLATE = "https://osf.io/download/{}/"

# Mapping of dataset filenames to their download URLs.
REGISTRY_URLS_DATA = {
    "A0670-221213.nwb": OSF_TEMPLATE.format("sbnaw"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm"),
    "A2929-200711.nwb": OSF_TEMPLATE.format("y7zwd"),
    "Achilles_10252013_EEG.nwb": OSF_TEMPLATE.format("2dfvp"),
    "em_three_states.npz": OSF_TEMPLATE.format("wdz7j"),
    "julia_regression_mstep_no_prior.npz": OSF_TEMPLATE.format("3ky6m"),
    "julia_regression_mstep_good_prior.npz": OSF_TEMPLATE.format("w586p"),
    "julia_regression_mstep_flat_prior.npz": OSF_TEMPLATE.format("umz4d"),
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


def _check_dependencies(needs_dandi: bool = False):
    """Check optional dependencies."""
    if needs_dandi and dandi is None:
        raise ImportError(
            "Missing optional dependency 'dandi'."
            " Please use pip or "
            "conda to install 'dandi'."
        )

    if pooch is None and tqdm is None:
        raise ImportError(
            "Missing optional dependencies 'pooch' and 'tqdm'."
            " Please use pip or "
            "conda to install 'pooch' and 'tqdm'."
        )
    elif pooch is None:
        raise ImportError(
            "Missing optional dependency 'pooch'."
            " Please use pip or "
            "conda to install 'pooch'."
        )
    elif tqdm is None:
        raise ImportError(
            "Missing optional dependency 'tqdm'."
            " Please use pip or "
            "conda to install 'tqdm'."
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
    _check_dependencies()
    retriever = _create_retriever(path)
    # Fetch the dataset using pooch.
    return retriever.fetch(
        dataset_name,
        progressbar=True if tqdm else False,
    )


def download_dandi_data(
    dandiset_id: str, file_path: str, force_download: bool = False
) -> NWBHDF5IO:
    """Download a dataset from the [DANDI Archive](https://dandiarchive.org/).

    Parameters
    ----------
    dandiset_id :
        6-character string of numbers giving the ID of the dandiset.
    file_path :
        filepath to the specific .nwb file within the dandiset we wish to return.
    force_download :
        True if you want to download the dataset even if it already exists, False - default - otherwise.

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
    _check_dependencies(needs_dandi=True)

    # Set up cache directory
    if _NEMOS_ENV in os.environ:
        cache_dir = pathlib.Path(os.environ[_NEMOS_ENV])
    else:
        cache_dir = pooch.os_cache("nemos") / "nwb-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a deterministic filename based on dandiset_id and file_path
    # Hash to make sure that there are no problematic characters for filename.
    dandiset_hash = hashlib.md5(str(dandiset_id).encode()).hexdigest()
    filepath_hash = hashlib.md5(str(file_path).encode()).hexdigest()
    cache_filename = dandiset_hash + filepath_hash + ".nwb"
    cached_file_path = cache_dir / cache_filename

    # Check if file already exists in cache
    if cached_file_path.exists() and not force_download:
        # File exists, open it directly
        file = h5py.File(cached_file_path, "r")
        io = NWBHDF5IO(file=file, load_namespaces=True)
        return io

    # File doesn't exist, download it
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(file_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # Download file using fsspec
    fs = fsspec.filesystem("http")

    # Download to temporary location first, then move to final location
    temp_file_path = cached_file_path.with_suffix(".tmp")

    try:
        # Get file size for progress bar
        with fs.open(s3_url, "rb") as remote_file:
            # Try to get content length from headers
            file_size = getattr(remote_file, "size", None)

        with fs.open(s3_url, "rb") as remote_file:
            # Initialize progress bar

            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {file_path.split('/')[-1]}",
            ) as pbar:
                with open(temp_file_path, "wb") as local_file:
                    chunk_size = 8192 * 1024  # 8MB chunks
                    while True:
                        chunk = remote_file.read(chunk_size)
                        if not chunk:
                            break
                        local_file.write(chunk)
                        pbar.update(len(chunk))

        # Move completed download to final location
        temp_file_path.rename(cached_file_path)

    except Exception:
        # Clean up temp file if download failed
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise

    # Open the downloaded file
    file = h5py.File(cached_file_path, "r")
    io = NWBHDF5IO(file=file, load_namespaces=True)

    return io
