"""Utility functions for coupling filter definition."""

from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as sts
from numpy.typing import NDArray

from . import convolve
from .pytrees import FeaturePytree


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


def simulate_recurrent(
    model,
    random_key: jax.Array,
    feedforward_input: Union[NDArray, jnp.ndarray],
    coupling_basis_matrix: Union[NDArray, jnp.ndarray],
    init_y: Union[NDArray, jnp.ndarray],
):
    """
    Simulate neural activity using the GLM as a recurrent network.

    This function projects neural activity into the future, employing the fitted
    parameters of the GLM. It is capable of simulating activity based on a combination
    of historical activity and external feedforward inputs like convolved currents, light
    intensities, etc.

    Parameters
    ----------
    random_key :
        jax.random.key for seeding the simulation.
    feedforward_input :
        External input matrix to the model, representing factors like convolved currents,
        light intensities, etc. When not provided, the simulation is done with coupling-only.
        Expected shape: (n_time_bins, n_neurons, n_basis_input).
    init_y :
        Initial observation (spike counts for PoissonGLM) matrix that kickstarts the simulation.
        Expected shape: (window_size, n_neurons).
    coupling_basis_matrix :
        Basis matrix for coupling, representing between-neuron couplings
        and auto-correlations. Expected shape: (window_size, n_basis_coupling).

    Returns
    -------
    simulated_activity :
        Simulated activity (spike counts for PoissonGLMs) for each neuron over time.
        Shape, (n_time_bins, n_neurons).
    firing_rates :
        Simulated rates for each neuron over time. Shape, (n_time_bins, n_neurons,).

    Raises
    ------
    NotFittedError
        If the model hasn't been fitted prior to calling this method.
    ValueError
        - If the instance has not been previously fitted.
        - If there's an inconsistency between the number of neurons in model parameters.
        - If the number of neurons in input arguments doesn't match with model parameters.


    See Also
    --------
    [predict](./#nemos.glm.GLM.predict) :
    Method to predict rates based on the model's parameters.

    Notes
    -----
    The model coefficients (`self.coef_`) are structured such that the first set of coefficients
    (of size `n_basis_coupling * n_neurons`) are interpreted as the weights for the recurrent couplings.
    The remaining coefficients correspond to the weights for the feed-forward input.


    The sum of `n_basis_input` and `n_basis_coupling * n_neurons` should equal `self.coef_.shape[1]`
    to ensure consistency in the model's input feature dimensionality.
    """
    if isinstance(feedforward_input, FeaturePytree):
        raise ValueError(
            "simulate_recurrent works only with arrays. "
            "FeaturePytree provided instead!"
        )
    # check if the model is fit
    model._check_is_fit()

    # convert to jnp.ndarray
    coupling_basis_matrix = jnp.asarray(coupling_basis_matrix, dtype=float)

    n_basis_coupling = coupling_basis_matrix.shape[1]
    n_neurons = model.intercept_.shape[0]

    w_feedforward = model.coef_[:, n_basis_coupling * n_neurons:]
    w_recurrent = model.coef_[:, : n_basis_coupling * n_neurons]
    bs = model.intercept_

    feedforward_input, init_y = model._preprocess_simulate(
        feedforward_input,
        params_feedforward=(w_feedforward, bs),
        init_y=init_y,
        params_recurrent=(w_recurrent, bs),
    )

    if init_y.shape[0] != coupling_basis_matrix.shape[0]:
        raise ValueError(
            "`init_y` and `coupling_basis_matrix`"
            " should have the same window size! "
            f"`init_y` window size: {init_y.shape[1]}, "
            f"`coupling_basis_matrix` window size: {coupling_basis_matrix.shape[1]}"
        )

    subkeys = jax.random.split(random_key, num=feedforward_input.shape[0])
    # (n_samples, n_neurons)
    feed_forward_contrib = jnp.einsum(
        "ik,tik->ti", w_feedforward, feedforward_input
    )

    def scan_fn(
        data: Tuple[jnp.ndarray, int], key: jax.Array
    ) -> Tuple[Tuple[jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Scan over time steps and simulate activity and rates.

        This function simulates the neural activity and firing rates for each time step
        based on the previous activity, feedforward input, and model coefficients.
        """
        activity, t_sample = data

        # Convolve the neural activity with the coupling basis matrix
        # Output of shape (1, n_neuron, n_basis_coupling)
        # 1. The first dimension is time, and 1 is by construction since we are simulating 1
        #    sample
        # 2. Flatten to shape (n_neuron * n_basis_coupling, )
        conv_act = convolve.reshape_convolve(
            activity, coupling_basis_matrix
        ).reshape(
            -1,
        )

        # Extract the slice of the feedforward input for the current time step
        input_slice = jax.lax.dynamic_slice(
            feed_forward_contrib,
            (t_sample, 0),
            (1, feed_forward_contrib.shape[1]),
        ).squeeze(axis=0)

        # Predict the firing rate using the model coefficients
        # Doesn't use predict because the non-linearity needs
        # to be applied after we add the feed forward input
        firing_rate = model._observation_model.inverse_link_function(
            w_recurrent.dot(conv_act) + input_slice + bs
        )

        # Simulate activity based on the predicted firing rate
        new_act = model._observation_model.sample_generator(key, firing_rate)

        # Shift of one sample the spike count window
        # for the next iteration (i.e. remove the first counts, and
        # stack the newly generated sample)
        # Increase the t_sample by one
        carry = jnp.vstack((activity[1:], new_act)), t_sample + 1
        return carry, (new_act, firing_rate)

    _, outputs = jax.lax.scan(scan_fn, (init_y, 0), subkeys)
    simulated_activity, firing_rates = outputs
    return simulated_activity, firing_rates
