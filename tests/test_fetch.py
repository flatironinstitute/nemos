import pathlib

import pytest

import nemos as nmo
from nemos.fetch.fetch_data import REGISTRY_UTILS


def test_registry():
    """Check that the current tag is in the registry."""
    assert nmo.__version__ in REGISTRY_UTILS


@pytest.mark.parametrize("version", list(REGISTRY_UTILS.keys())[-3:])
def test_download_utils(version, tmpdir):
    """Test that utils can be downloaded.

    This use the pytest fixture tmpdir which is a test specific temporary directory.
    """
    fhname = nmo.fetch.fetch_utils(tmpdir, version=version)
    assert pathlib.Path(fhname).exists()
