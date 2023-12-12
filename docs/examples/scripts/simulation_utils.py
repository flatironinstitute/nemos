"""Utility functions for coupling filter definition."""

from typing import Tuple

import numpy as np
import scipy.stats as sts
from numpy.typing import NDArray

import nemos as nmo


def temporal_fitler(ws: int, inhib_a: float = 1., excit_a: float = 2., inhib_b: float = 2., excit_b: float = 2.)\
        -> NDArray:
    """Generate coupling filter as Gamma pdf difference.

    Parameters
    ----------
    ws:
        The window size of the filter.
    inhib_a:
        The `a` constant for the gamma pdf of the inhibitory part of the filer.
    excit_a:
        The `a` constant for the gamma pdf of the excitatory part of the filer.
    inhib_b:
        The `b` constant for the gamma pdf of the inhibitory part of the filer.
    excit_b:
        The `a` constant for the gamma pdf of the excitatory part of the filer.

    Returns
    -------
    filter:
        The coupling filter.
    """
    x = np.linspace(0, 5, ws)
    gm_inhibition = sts.gamma(a=inhib_a, scale=1/inhib_b)
    gm_excitation = sts.gamma(a=excit_a, scale=1/excit_b)
    filter = gm_excitation.pdf(x) - gm_inhibition.pdf(x)
    # impose a norm < 1 for the filter
    filter = 0.8 * filter / np.linalg.norm(filter)
    return filter


def regress_filter(coupling_filter_bank: NDArray, basis: nmo.basis.Basis) -> Tuple[NDArray, NDArray]:
    """Approximate scipy.stats.gamma based filters with basis function.

    Find the ols weights for representing the filters in terms of basis functions.
    This is done to re-use the nsl.glm.simulate method.

    Parameters
    ----------
    coupling_filter_bank:
        The coupling filters. Shape (n_neurons, n_neurons, window_size)
    basis:
        The basis function to instantiate.

    Returns
    -------
    eval_basis:
        The basis matrix, shape (window_size, n_basis_funcs)
    weights:
        The weights for each neuron. Shape (n_neurons, n_neurons, n_basis_funcs)
    """
    n_neurons, _, ws = coupling_filter_bank.shape
    eval_basis = basis.evaluate(np.linspace(0, 1, ws))

    # Reshape the coupling_filter_bank for vectorized least squares
    filters_reshaped = coupling_filter_bank.reshape(-1, ws)

    # Solve the least squares problem for all filters at once
    # (vecotrizing the features)
    weights = np.linalg.lstsq(eval_basis, filters_reshaped.T, rcond=None)[0]

    # Reshape back to the original dimensions
    weights = weights.T.reshape(n_neurons, n_neurons, -1)

    return eval_basis, weights


def define_coupling_filters(n_neurons: int, window_size: int, n_basis_funcs: int = 20):

    np.random.seed(101)
    # inhibition params
    a_inhib = 1
    b_inhib = 1
    a_excit = np.random.uniform(1.1, 5, size=(n_neurons, n_neurons))
    b_excit = np.random.uniform(1.1, 5, size=(n_neurons, n_neurons))

    # define 2x2 coupling filters of the specific width
    coupling_filter_bank = np.zeros((n_neurons, n_neurons, window_size))
    for neu_i in range(n_neurons):
        for neu_j in range(n_neurons):
            coupling_filter_bank[neu_i, neu_j, :] = temporal_fitler(window_size,
                                                                    inhib_a=a_inhib,
                                                                    excit_a=a_excit[neu_i, neu_j],
                                                                    inhib_b=b_inhib,
                                                                    excit_b=b_excit[neu_i, neu_j]
                                                                    )
    basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs)
    coupling_basis, weights = regress_filter(coupling_filter_bank, basis)
    weights = weights.reshape(n_neurons, -1)
    intercept = -4 * np.ones(n_neurons)
    return coupling_basis,  weights, intercept
