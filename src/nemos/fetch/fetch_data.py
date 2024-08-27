"""Fetch data using pooch."""

import hashlib
import pathlib
from typing import List, Optional, Union

import pooch
import requests

from .. import __version__

REGISTRY_DATA = {
    "A0670-221213.nwb": "8587dd6dde107504bd4a17a68ce8fb934fcbcccc337e653f31484611ee51f50a",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "Achilles_10252013.nwb": "42857015aad4c2f7f6f3d4022611a69bc86d714cf465183ce30955731e614990",
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "m691l1.nwb": "1990d8d95a70a29af95dade51e60ffae7a176f6207e80dbf9ccefaf418fe22b6",
}

REGISTRY_UTILS = {
    "0.1.1": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.2": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.3": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.4": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.5": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
    "0.1.6": "369b5d0db98172856363072e48e51f16a2c41f20c4c7d5d988e29987b391c291",
}


OSF_TEMPLATE = "https://osf.io/{}/download"
GITHUB_TEMPLATE_PLOTTING = (
    "https://raw.githubusercontent.com/flatironinstitute/nemos"
    "/{}/docs/neural_modeling/examples_utils/plotting.py"
)

# these are all from the OSF project at https://osf.io/ts37w/.
REGISTRY_URLS_DATA = {
    "A0670-221213.nwb": OSF_TEMPLATE.format("sbnaw"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm"),
}

REGISTRY_URLS_UTILS = {
    version: GITHUB_TEMPLATE_PLOTTING.format(version)
    for version in REGISTRY_UTILS.keys()
}


# default to "~/nemos-fetch-cache" for downloads when no env var is set.
_DEFAULT_DATA_DIR = pathlib.Path.home() / "nemos-fetch-cache"
_DEFAULT_UTILS_DIR = pathlib.Path(".") / "examples_utils"


def _calculate_sha256(data_dir: Union[str, pathlib.Path]):
    """
    Calculate hash signature for each fetch in a directory.

    Helper function to generate hash keys for files in a specified directory.
    Useful when one wants to add new fetch to the registry.

    Parameters
    ----------
    data_dir:
        The path to the directory containing the fetch.

    Returns
    -------
    :
        Dictionary, keys are file names, values are hash keys.
    """
    data_dir = pathlib.Path(data_dir)

    # Initialize the registry dict
    registry_hash = dict()
    for file_path in data_dir.iterdir():
        if file_path.is_dir():
            continue
        # Open the file in binary mode
        with open(file_path, "rb") as f:
            # Initialize the hash
            sha256_hash = hashlib.sha256()
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            registry_hash[file_path.name] = sha256_hash.hexdigest()
    # Return the hexadecimal digest of the hash
    return registry_hash


def _create_retriever(path: Optional[pathlib.Path] = None):

    if path is None:
        # create the directory if it doesn't exist and save.
        _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _DEFAULT_DATA_DIR

    return pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS_DATA,
        registry=REGISTRY_DATA,
        retry_if_failed=2,
        # this defaults to true, unless the env variable with same name is set
        allow_updates="POOCH_ALLOW_UPDATES",
        env="NEMOS_DATA_DIR",
    )


def _find_shared_directory(paths: List[pathlib.Path]) -> pathlib.Path:
    """Find directory shared by all paths."""
    for dir in paths[0].parents:
        if all([dir in p.parents for p in paths]):
            break
    return dir


def fetch_data(
    dataset_name: str, path: Optional[Union[pathlib.Path, str]] = None
) -> str:
    """Download fetch, using pooch. These are largely used for testing.

    To view list of downloadable files, look at `DOWNLOADABLE_FILES`.

    This checks whether the fetch already exists and is unchanged and downloads
    again, if necessary. If dataset_name ends in .tar.gz, this also
    decompresses and extracts the archive, returning the Path to the resulting
    directory. Else, it just returns the Path to the downloaded file.

    """
    retriever = _create_retriever(path)
    return _retrieve_data(dataset_name, retriever).as_posix()


def _retrieve_data(dataset_name: str, retriever):

    if dataset_name.endswith(".tar.gz"):
        processor = pooch.Untar()
    else:
        processor = None
    file_name = retriever.fetch(dataset_name, progressbar=True, processor=processor)
    if dataset_name.endswith(".tar.gz"):
        file_name = _find_shared_directory([pathlib.Path(f) for f in file_name])
    else:
        file_name = pathlib.Path(file_name)
    return file_name


def _get_github_utils_registry():
    """Fetch all tags from a GitHub repository."""

    fname = "plotting_{}.py"
    api_url = "https://api.github.com/repos/flatironinstitute/nemos/tags"
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error for bad status codes
    tags = [tag["name"] for tag in response.json()]

    for version in tags:
        url = (
            f"https://raw.githubusercontent.com/flatironinstitute/nemos/{version}/docs/neural_modeling"
            f"/examples_utils/plotting.py"
        )
        try:
            actual_path = pooch.retrieve(
                url=url,
                known_hash=None,  # Add a hash here if needed for verification
                fname=fname.format(version),  # Local filename
                path=pooch.os_cache(
                    "nemos-cache"
                ),  # Cache the file in the package-specific cache directory
            )
            print(actual_path)
        except Exception as e:
            print(e)
            continue

    tmp = _calculate_sha256(pooch.os_cache("nemos-cache"))
    registry_hash = {
        key.split("plotting_")[1].split(".py")[0]: tmp[key]
        for key in sorted(tmp.keys())
    }

    return registry_hash


def fetch_utils(path=None):
    if path is None:
        path = _DEFAULT_UTILS_DIR

    version = __version__

    retriever = pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS_UTILS,
        registry=REGISTRY_UTILS,
        retry_if_failed=2,
        # this defaults to true, unless the env variable with same name is set
        allow_updates="POOCH_ALLOW_UPDATES",
    )

    file_name = _retrieve_data(version, retriever)

    # Define the fixed name you want to use (this does not cash)
    fixed_file_name = pathlib.Path(file_name).parent / "plotting.py"

    # Rename the file using pathlib
    pathlib.Path(file_name).rename(fixed_file_name)

    # Return the path to the renamed file
    return fixed_file_name.as_posix()
