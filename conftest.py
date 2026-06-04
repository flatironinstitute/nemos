"""Register a custom option in root."""

import matplotlib.pyplot as plt
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--timeit",
        action="store_true",
        default=False,
        help="Show aggregated parametrized test durations",
    )


# following https://github.com/scverse/scanpy/issues/1662
@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    """
    Close figures when exiting from doctests.

    Note that this won't close between tests in a given docstring, but it will close
    between docstrings.
    """  # numpydoc ignore=YD01
    yield
    plt.close("all")
