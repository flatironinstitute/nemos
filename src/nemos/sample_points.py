"""Helper functions for generating arrays of sample points, for basis functions."""

import numpy as np
from numpy.typing import NDArray


def raised_cosine_log(n_basis_funcs: int, window_size: int) -> NDArray:
    """Generate log-spaced sample points for RaisedCosineBasis.

    These are the sample points used in Pillow et al. [1]_ and, when used with
    the RaisedCosineBasis, result in log-spaced cosine bumps.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.

    Returns
    -------
    sample_pts : (window_size,)
        log-spaced points running from ``n_basis_funcs * np.pi`` to 0.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013. http://dx.doi.org/10.1523/jneurosci.3305-05.2005

    """
    # in the config of ubuntu-latest on the github runner with python 3.10 logspace at -1 does not return the exactly
    # 0.1, but something slightly smaller; for this reason I am subtracting the last element of the logspace array
    # instead of 0.1 to make sure that the array starts from 0.
    log_space_samp = np.logspace(np.log10(np.pi * (n_basis_funcs - 1)), -1, window_size)
    return log_space_samp - log_space_samp[-1]


def raised_cosine_linear(n_basis_funcs: int, window_size: int) -> NDArray:
    """Generate linear-spaced sample points for RaisedCosineBasis.

    When used with the RaisedCosineBasis, results in evenly (linear) spaced
    cosine bumps.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.

    Returns
    -------
    sample_pts : (window_size,)
        linearly-spaced points running from 0 to ``n_basis_funcs * np.pi``.

    """
    return np.linspace(0, np.pi * (n_basis_funcs - 1), window_size)
