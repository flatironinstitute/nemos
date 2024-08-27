import os.path

import pytest
import nemos as nmo
from nemos.fetch.fetch_data import REGISTRY_UTILS


def test_registry():
    """Check that the current tag is in registry"""
    assert nmo.__version__ in REGISTRY_UTILS


# use the pytest fixture tmpdir which is a test specific temporary directory
def test_download_utils(tmpdir):
    fhname = nmo.fetch.fetch_utils(tmpdir)
    assert fhname.exists()
