"""Fetch data using pooch."""

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
    "A0634-210617.nwb": OSF_TEMPLATE.format("28ths"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm")
}
DOWNLOADABLE_FILES = list(REGISTRY_URLS.keys())

import pathlib
from typing import List, Optional, Union
import pooch
import click
import requests
from tqdm import tqdm
import hashlib

# default to "./data" for downloads
DEFAULT_DATA_DIR = pathlib.Path("data")


def _calculate_sha256(data_dir: Union[str, pathlib.Path]):
    """Calculate hash signature for each data in a directory."""
    sha256_hash = hashlib.sha256()

    data_dir = pathlib.Path(data_dir)
    registry_hash = dict()
    for file_path in data_dir.iterdir():
        # Open the file in binary mode
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

            registry_hash[file_path.name] = sha256_hash.hexdigest()

    # Return the hexadecimal digest of the hash
    return registry_hash


def create_retriever(path: Optional[pathlib.Path] = None):
    return pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS,
        registry=REGISTRY,
        retry_if_failed=2,
        # this defaults to true, unless the env variable with same name is set
        allow_updates="POOCH_ALLOW_UPDATES",
        env="NEMOS_DATA_DIR"
    )
