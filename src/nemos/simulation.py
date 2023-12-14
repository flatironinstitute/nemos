"""Utility functions for coupling filter definition."""


import numpy as np
import scipy.stats as sts
from numpy.typing import NDArray


def difference_of_gammas(
    ws: int,
    upper_percentile: float = 0.99,
    inhib_a: float = 1.0,
    excit_a: float = 2.0,
    inhib_b: float = 1.0,
    excit_b: float = 2.0,
) -> NDArray:
    r"""Generate coupling filter as a Gamma pdf difference.

    Parameters
    ----------
    ws:
        The window size of the filter.
    upper_percentile:
        Upper bound of the gamma range as a percentile. The gamma function
        will be evaluated over the range [0, ppf(upper_percentile)].
    inhib_a:
        The `a` constant for the gamma pdf of the inhibitory part of the filter.
    excit_a:
        The `a` constant for the gamma pdf of the excitatory part of the filter.
    inhib_b:
        The `b` constant for the gamma pdf of the inhibitory part of the filter.
    excit_b:
        The `a` constant for the gamma pdf of the excitatory part of the filter.

    Notes
    -----
    The probability density function of a gamma distribution is parametrized as
    follows$^1$,
    $$
        p(x;\; a, b) = \frac{b^a x^{a-1} e^{-x}}{\Gamma(a)},
    $$
    where $\Gamma(a)$ refers to the gamma function, see$^1$.

    Returns
    -------
    filter:
        The coupling filter.

    Raises
    ------
    ValueError:
        - If any of the Gamma parameters is lesser or equal to 0.
        - If the upper_percentile is not in [0, 1).

    References
    ----------
    1. [SciPy Docs - "scipy.stats.gamma"](https://docs.scipy.org/doc/
    scipy/reference/generated/scipy.stats.gamma.html)
    """
    # check that the gamma parameters are positive (scipy returns
    # nans otherwise but no exception is raised)
    variables = {
        "excit_a": excit_a,
        "inhib_a": inhib_a,
        "excit_b": excit_b,
        "inhib_b": inhib_b,
    }
    for name, value in variables.items():
        if value <= 0:
            raise ValueError(f"Gamma parameter {name} must be >0.")
    # check for valid pecentile
    if upper_percentile < 0 or upper_percentile >= 1:
        raise ValueError(
            f"upper_percentile should lie in the [0, 1) interval. {upper_percentile} provided instead!"
        )

    gm_inhibition = sts.gamma(a=inhib_a, scale=1 / inhib_b)
    gm_excitation = sts.gamma(a=excit_a, scale=1 / excit_b)

    # calculate upper bound for the evaluation
    xmax = max(gm_inhibition.ppf(upper_percentile), gm_excitation.ppf(upper_percentile))
    # equi-spaced sample covering the range
    x = np.linspace(0, xmax, ws)

    # compute difference of gammas & normalize
    gamma_diff = gm_excitation.pdf(x) - gm_inhibition.pdf(x)
    gamma_diff = gamma_diff / np.linalg.norm(gamma_diff, ord=2)

    return gamma_diff


def regress_filter(coupling_filters: NDArray, eval_basis: NDArray) -> NDArray:
    """Approximate scipy.stats.gamma based filters with basis function.

    Find the Ordinary Least Squares weights for representing the filters in terms of basis functions.

    Parameters
    ----------
    coupling_filters:
        The coupling filters. Shape (window_size, n_neurons_receiver, n_neurons_sender)
    eval_basis:
        The evaluated basis function, shape (window_size, n_basis_funcs)

    Returns
    -------
    weights:
        The weights for each neuron. Shape (n_neurons_receiver, n_neurons_sender, n_basis_funcs)

    Raises
    ------
    ValueError
        - If eval_basis is not two-dimensional
        - If coupling_filters is not three-dimensional
        - If window_size differs between eval_basis and coupling_filters
    """
    # check shapes
    if eval_basis.ndim != 2:
        raise ValueError(
            "eval_basis must be a 2 dimensional array, "
            "shape (window_size, n_basis_funcs). "
            f"{eval_basis.ndim} dimensional array provided instead!"
        )
    if coupling_filters.ndim != 3:
        raise ValueError(
            "coupling_filters must be a 3 dimensional array, "
            "shape (window_size, n_neurons, n_neurons). "
            f"{coupling_filters.ndim} dimensional array provided instead!"
        )

    ws, n_neurons_receiver, n_neurons_sender = coupling_filters.shape

    # check that window size matches
    if eval_basis.shape[0] != ws:
        raise ValueError(
            "window_size mismatch. The window size of coupling_filters and eval_basis "
            f"does not match. coupling_filters has a window size of {ws}; "
            f"eval_basis has a window size of {eval_basis.shape[0]}."
        )

    # Reshape the coupling_filters for vectorized least-squares
    filters_reshaped = coupling_filters.reshape(ws, -1)

    # Solve the least squares problem for all filters at once
    # (vecotrizing the features)
    weights = np.linalg.lstsq(eval_basis, filters_reshaped, rcond=None)[0]

    # Reshape back to the original dimensions
    weights = np.transpose(
        weights.reshape(-1, n_neurons_receiver, n_neurons_sender), axes=(1, 2, 0)
    )

    return weights
