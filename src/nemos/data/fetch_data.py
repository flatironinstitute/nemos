"""Fetch data using pooch."""
import hashlib
import pathlib
from typing import List, Optional, Union

import pooch

REGISTRY = {
    "A0670-221213.nwb": "8587dd6dde107504bd4a17a68ce8fb934fcbcccc337e653f31484611ee51f50a",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "Achilles_10252013.nwb": "42857015aad4c2f7f6f3d4022611a69bc86d714cf465183ce30955731e614990",
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "m691l1.nwb": "1990d8d95a70a29af95dade51e60ffae7a176f6207e80dbf9ccefaf418fe22b6",
}

OSF_TEMPLATE = "https://osf.io/{}/download"
# these are all from the OSF project at https://osf.io/ts37w/.
REGISTRY_URLS = {
    "A0670-221213.nwb": OSF_TEMPLATE.format("sbnaw"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm"),
}
DOWNLOADABLE_FILES = list(REGISTRY_URLS.keys())


# default to "./data" for downloads when no env var is set.
_DEFAULT_DATA_DIR = pathlib.Path("data")


def _calculate_sha256(data_dir: Union[str, pathlib.Path]):
    """
    Calculate hash signature for each data in a directory.

    Helper function to generate hash keys for files in a specified directory.
    Useful when one wants to add new data to the registry.

    Parameters
    ----------
    data_dir:
        The path to the directory containing the data.

    Returns
    -------
    :
        Dictionary, keys are file names, values are hash keys.
    """
    data_dir = pathlib.Path(data_dir)

    # Initialize the registry dict
    registry_hash = dict()
    for file_path in data_dir.iterdir():
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
        path = _DEFAULT_DATA_DIR

    return pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS,
        registry=REGISTRY,
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
) -> pathlib.Path:
    """Download data, using pooch. These are largely used for testing.

    To view list of downloadable files, look at `DOWNLOADABLE_FILES`.

    This checks whether the data already exists and is unchanged and downloads
    again, if necessary. If dataset_name ends in .tar.gz, this also
    decompresses and extracts the archive, returning the Path to the resulting
    directory. Else, it just returns the Path to the downloaded file.

    """
    retriever = _create_retriever(path)
    if retriever is None:
        raise ImportError(
            "Missing optional dependency 'pooch'."
            " Please use pip or "
            "conda to install 'pooch'."
        )
    if dataset_name.endswith(".tar.gz"):
        processor = pooch.Untar()
    else:
        processor = None
    file_name = retriever.fetch(dataset_name, progressbar=True, processor=processor)
    if dataset_name.endswith(".tar.gz"):
        file_name = _find_shared_directory([pathlib.Path(f) for f in file_name])
    else:
        file_name = pathlib.Path(file_name)
    return file_name.as_posix()
