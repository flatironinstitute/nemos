"""
Module for fetching data and utilities using the pooch library.

This module allows you to download datasets and utility scripts
from specified URLs and manage their versions and integrity using
SHA256 hash verification. It also provides helper functions to
calculate hashes for local files and manage file paths.
"""

import hashlib
import pathlib
import shutil
from typing import List, Optional, Union

import pooch
import requests

from .. import __version__

# Registry of dataset filenames and their corresponding SHA256 hashes.
REGISTRY_DATA = {
    "A0670-221213.nwb": "8587dd6dde107504bd4a17a68ce8fb934fcbcccc337e653f31484611ee51f50a",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "Achilles_10252013.nwb": "42857015aad4c2f7f6f3d4022611a69bc86d714cf465183ce30955731e614990",
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "m691l1.nwb": "1990d8d95a70a29af95dade51e60ffae7a176f6207e80dbf9ccefaf418fe22b6",
}

# Registry of utility script versions and their corresponding SHA256 hashes.
REGISTRY_UTILS = {
    "0.1.1": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.2": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.3": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.4": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.5": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.6": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
}

# URL templates for downloading datasets and utility scripts.
OSF_TEMPLATE = "https://osf.io/{}/download"
GITHUB_TEMPLATE_PLOTTING = (
    "https://raw.githubusercontent.com/flatironinstitute/nemos"
    "/{}/docs/neural_modeling/examples_utils/plotting.py"
)

# Mapping of dataset filenames to their download URLs.
REGISTRY_URLS_DATA = {
    "A0670-221213.nwb": OSF_TEMPLATE.format("sbnaw"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm"),
}

# Mapping of utility script versions to their download URLs.
REGISTRY_URLS_UTILS = {
    version: GITHUB_TEMPLATE_PLOTTING.format(version)
    for version in REGISTRY_UTILS.keys()
}

# Default directories for storing downloaded data and utility scripts.
_DEFAULT_DATA_DIR = pathlib.Path.home() / "nemos-fetch-cache"
_DEFAULT_UTILS_DIR = pathlib.Path(".") / "examples_utils"


def _calculate_sha256(data_dir: Union[str, pathlib.Path]):
    """
    Calculate the SHA256 hash for each file in a directory.

    This function iterates through files in the specified directory
    and computes their SHA256 hash. This is useful for verifying
    file integrity or generating new registry entries.

    Parameters
    ----------
    data_dir :
        The path to the directory containing the files to hash.

    Returns
    -------
    :
        A dictionary where the keys are filenames and the values
        are their corresponding SHA256 hashes.
    """
    data_dir = pathlib.Path(data_dir)

    # Initialize the registry dictionary to store file hashes.
    registry_hash = dict()
    for file_path in data_dir.iterdir():
        if file_path.is_dir():
            continue
        # Open the file in binary mode to read it.
        with open(file_path, "rb") as f:
            sha256_hash = hashlib.sha256()  # Initialize the SHA256 hash object.
            # Read the file in chunks to avoid loading it all into memory.
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)  # Update the hash with the chunk.
            registry_hash[file_path.name] = sha256_hash.hexdigest()  # Store the hash.
    return registry_hash


def _create_retriever(path: Optional[pathlib.Path] = None) -> pooch.Pooch:
    """
    Create a pooch retriever for fetching datasets.

    This function sets up the pooch retriever, which manages the
    downloading and caching of files, including handling retries
    and checking file integrity using SHA256 hashes.

    Parameters
    ----------
    path :
        The directory where datasets will be stored. If not provided,
        defaults to _DEFAULT_DATA_DIR.

    Returns
    -------
    :
        A configured pooch retriever object.
    """
    if path is None:
        # Use the default data directory if none is provided.
        _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _DEFAULT_DATA_DIR

    return pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS_DATA,
        registry=REGISTRY_DATA,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
        env="NEMOS_DATA_DIR",
    )


def _find_shared_directory(paths: List[pathlib.Path]) -> pathlib.Path:
    """
    Find the common parent directory shared by all given paths.

    This function takes a list of file paths and determines the
    highest-level directory that all paths share.

    Parameters
    ----------
    paths :
        A list of file paths.

    Returns
    -------
    :
        The shared parent directory.

    Raises
    ------
    ValueError
        If no paths are provided or if the paths do not share a common directory.
    """
    # Iterate through the parents of the first path to find a common directory.
    if len(paths) == 0:
        raise ValueError("Must provide at least one path. The input list of paths is empty.")

    if len(paths[0].parents) == 0:
        raise ValueError("The provided path does not have any parent directories.")

    for directory in paths[0].parents:
        if all([directory in p.parents for p in paths]):
            return directory

    raise ValueError("The provided paths do not share a common parent directory.")


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
    retriever = _create_retriever(path)
    return _retrieve_data(dataset_name, retriever).as_posix()


def _retrieve_data(dataset_name: str, retriever: pooch.Pooch) -> pathlib.Path:
    """
    Helper function to fetch and process a dataset.

    This function is used internally to download a dataset and, if
    necessary, decompress it.

    Parameters
    ----------
    dataset_name :
        The name of the dataset to download.
    retriever :
        The pooch retriever object used to fetch the dataset.

    Returns
    -------
    :
        The path to the downloaded file or directory.
    """
    # Determine if the dataset is an archive and set the appropriate processor.
    if dataset_name.endswith(".tar.gz"):
        processor = pooch.Untar()
    else:
        processor = None

    # Fetch the dataset using pooch.
    file_name = retriever.fetch(dataset_name, progressbar=True, processor=processor)

    # If the dataset was an archive, find the shared directory; otherwise, return the file path.
    if dataset_name.endswith(".tar.gz"):
        file_name = _find_shared_directory([pathlib.Path(f) for f in file_name])
    else:
        file_name = pathlib.Path(file_name)

    return file_name

def _get_all_tags() -> List[str]:
    """
    Retrieve all available tags.

    Returns
    -------
    :
        List of all available tags.
    """
    api_url = "https://api.github.com/repos/flatironinstitute/nemos/tags"

    # Request the list of tags (versions) from the GitHub API.
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error for bad status codes.
    return [tag["name"] for tag in response.json()]


def _get_hash_for_plotting_utils_registry(version: Optional[str] = None) -> dict:
    """
    Fetch and hash the plotting.py utility script from GitHub for a given package version.

    This function computes the SHA256 hash for the downloaded script.
    This function should help updating the hash registry when a new
    version of NeMoS is released.

    Parameters
    ----------
    version :
        The version tag string.

    Returns
    -------
    :
        The hash of the script in a dict with key the version.
    """
    try:
        if version is None:
            version = __version__

        url = (
            f"https://raw.githubusercontent.com/flatironinstitute/nemos/{version}/docs/"
            f"neural_modeling/examples_utils/plotting.py"
        )
        _ = pooch.retrieve(
            url=url,
            known_hash=None,  # Add a hash here if needed for verification.
            fname="plotting_" + version + ".py",  # Local filename.
            path=pooch.os_cache("nemos-cache"),
        )

        # Calculate and return SHA256 hashes for all downloaded scripts.
        tmp = _calculate_sha256(pooch.os_cache("nemos-cache"))
        registry_hash = {
            key.split("plotting_")[1].split(".py")[0]: tmp[key]
            for key in sorted(tmp.keys())
        }
    finally:
        # clear cache
        if pooch.os_cache("nemos-cache").exists():
            shutil.rmtree(pooch.os_cache("nemos-cache"))
    return registry_hash


def fetch_utils(path=None, version: Optional[str] = None):
    """
    Fetch the utility script corresponding to the specified NeMoS version.

    This function downloads the utility script (`plotting.py`) corresponding
    to the current version of the library. The script is renamed to a fixed
    filename and stored in the specified directory.

    Parameters
    ----------
    path :
        The directory where the utility script will be stored. If not provided,
        defaults to _DEFAULT_UTILS_DIR.
    version :
        The package version. Default is the current version.

    Returns
    -------
    :
        The path to the downloaded and renamed utility script.

    Raises
    ------
    ValueError
        If the version is not in the regisrtry.
    """
    if path is None:
        path = _DEFAULT_UTILS_DIR

    if version is None:
        version = __version__

    if version not in REGISTRY_UTILS:
        raise ValueError(f"Version {version} is not available in the registry.")

    # Create the pooch retriever for fetching the utility script.
    retriever = pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS_UTILS,
        registry=REGISTRY_UTILS,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
    )

    # Retrieve the utility script.
    file_name = _retrieve_data(version, retriever)

    # Define the fixed name for the utility script.
    fixed_file_name = pathlib.Path(file_name).parent / "plotting.py"

    # Rename the script to the fixed filename.
    pathlib.Path(file_name).rename(fixed_file_name)

    return fixed_file_name.as_posix()
