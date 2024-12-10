"""Utility functions for coupling filter definition."""

from __future__ import annotations

from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as sts
from numpy.typing import NDArray

from . import convolve, validation
from .pytrees import FeaturePytree


def difference_of_gammas(
    ws: int,
    upper_percentile: float = 0.99,
    inhib_a: float = 1.0,
    excit_a: float = 2.0,
    inhib_b: float = 1.0,
    excit_b: float = 2.0,
) -> NDArray:
    r"""
    Generate coupling filter as a Gamma pdf difference.

    Parameters
    ----------
    ws:
        The window size of the filter.
    upper_percentile:
        Upper bound of the gamma range as a percentile. The gamma function
        will be evaluated over the range [0, ppf(upper_percentile)].
    inhib_a:
        The ``a`` constant for the gamma pdf of the inhibitory part of the filter.
    excit_a:
        The ``a`` constant for the gamma pdf of the excitatory part of the filter.
    inhib_b:
        The ``b`` constant for the gamma pdf of the inhibitory part of the filter.
    excit_b:
        The ``a`` constant for the gamma pdf of the excitatory part of the filter.

    Notes
    -----
    The probability density function of a gamma distribution is parametrized as
    follows [1]_ :,

    .. math::

        p(x;\; a, b) = \frac{b^a x^{a-1} e^{-x}}{\Gamma(a)},

    where :math:`\Gamma(a)` refers to the gamma function, see [1]_.

    Returns
    -------
    filter:
        The coupling filter.

    Raises
    ------
    ValueError:
        If any of the Gamma parameters is lesser or equal to 0.
    ValueError:
        If the upper_percentile is not in [0, 1).

    References
    ----------
    .. [1] SciPy Docs -
       :meth:`scipy.stats.gamma <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>`

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from nemos.simulation import difference_of_gammas
    >>> coupling_duration = 100
    >>> inhib_a, inhib_b = 1.0, 1.0
    >>> excit_a, excit_b = 2.0, 2.0
    >>> coupling_filter = difference_of_gammas(
    ...     ws=coupling_duration,
    ...     inhib_a=inhib_a,
    ...     inhib_b=inhib_b,
    ...     excit_a=excit_a,
    ...     excit_b=excit_b
    ... )
    >>> _ = plt.plot(coupling_filter)
    >>> _ = plt.title("Coupling filter from difference of gammas")
    >>> _ = plt.show()

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
    # check for valid percentile
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
        The coupling filters. Shape ``(window_size, n_neurons_receiver, n_neurons_sender)``
    eval_basis:
        The evaluated basis function, shape ``(window_size, n_basis_funcs)``

    Returns
    -------
    weights:
        The weights for each neuron. Shape ``(n_basis_funcs, n_neurons_receiver, n_neurons_sender)``

    Raises
    ------
    ValueError
        If eval_basis is not two-dimensional.
    ValueError
        If coupling_filters is not three-dimensional.
    ValueError
        If window_size differs between eval_basis and coupling_filters.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from nemos.simulation import regress_filter, difference_of_gammas
    >>> from nemos.basis import RaisedCosineLogEval
    >>> filter_duration = 100
    >>> n_basis_funcs = 20
    >>> filter_bank = difference_of_gammas(filter_duration).reshape(filter_duration, 1, 1)
    >>> _, basis = RaisedCosineLogEval(10).evaluate_on_grid(filter_duration)
    >>> weights = regress_filter(filter_bank, basis)[0, 0]
    >>> print("Weights shape:", weights.shape)
    Weights shape: (10,)
    >>> _ = plt.plot(filter_bank[:, 0, 0], label=f"True filter")
    >>> _ = plt.plot(basis.dot(weights), "--", label=f"Approx. filter")
    >>> _ = plt.legend()
    >>> _ = plt.title("True vs. Approximated Filters")
    >>> _ = plt.show()

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
    coupling_coef: NDArray,
    feedforward_coef: NDArray,
    intercepts: NDArray,
    random_key: jax.Array,
    feedforward_input: Union[NDArray, jnp.ndarray],
    coupling_basis_matrix: Union[NDArray, jnp.ndarray],
    init_y: Union[NDArray, jnp.ndarray],
    inverse_link_function: Callable = jax.nn.softplus,
):
    """
    Simulate neural activity using the GLM as a recurrent network.

    This function projects neural activity into the future, employing the fitted
    parameters of the GLM. It is capable of simulating activity based on a combination
    of historical activity and external feedforward inputs like convolved currents, light
    intensities, etc.

    Parameters
    ----------
    coupling_coef :
        Coefficients for the coupling (recurrent connections) between neurons.
        Expected shape: (n_neurons (receiver), n_neurons (sender), n_basis_coupling).
    feedforward_coef :
        Coefficients for the feedforward inputs to each neuron.
        Expected shape: ``(n_neurons, n_basis_input)``.
    intercepts :
        Bias term for each neuron. Expected shape: ``(n_neurons,)``.
    random_key :
        jax.random.key for seeding the simulation.
    feedforward_input :
        External input matrix to the model, representing factors like convolved currents,
        light intensities, etc. When not provided, the simulation is done with coupling-only.
        Expected shape: ``(n_time_bins, n_neurons, n_basis_input)``.
    init_y :
        Initial observation (spike counts for PoissonGLM) matrix that kickstarts the simulation.
        Expected shape: ``(window_size, n_neurons)``.
    coupling_basis_matrix :
        Basis matrix for coupling, representing between-neuron couplings
        and auto-correlations. Expected shape: ``(window_size, n_basis_coupling)``.
    inverse_link_function :
        The inverse link function for the observation model.

    Returns
    -------
    simulated_activity :
        Simulated activity (spike counts for PoissonGLMs) for each neuron over time.
        Shape, ``(n_time_bins, n_neurons)``.
    firing_rates :
        Simulated rates for each neuron over time. Shape, ``(n_time_bins, n_neurons,)``.

    Raises
    ------
    ValueError
        If there's an inconsistency between the number of neurons in model parameters.
    ValueError
        If the number of neurons in input arguments doesn't match with model parameters.

    Examples
    --------
    >>> import numpy as np
    >>> import jax
    >>> import matplotlib.pyplot as plt
    >>> from nemos.simulation import simulate_recurrent
    >>>
    >>> n_neurons = 2
    >>> coupling_duration = 100
    >>> feedforward_input = np.random.normal(size=(1000, n_neurons, 1))
    >>> coupling_basis = np.random.normal(size=(coupling_duration, 10))
    >>> coupling_coef = np.random.normal(size=(n_neurons, n_neurons, 10))
    >>> intercept = -9 * np.ones(n_neurons)
    >>> init_spikes = np.zeros((coupling_duration, n_neurons))
    >>> random_key = jax.random.key(123)
    >>> spikes, rates = simulate_recurrent(
    ...     coupling_coef=coupling_coef,
    ...     feedforward_coef=np.ones((n_neurons, 1)),
    ...     intercepts=intercept,
    ...     random_key=random_key,
    ...     feedforward_input=feedforward_input,
    ...     coupling_basis_matrix=coupling_basis,
    ...     init_y=init_spikes
    ... )
    >>> _ = plt.figure()
    >>> _ = plt.plot(rates[:, 0], label="Neuron 0 rate")
    >>> _ = plt.plot(rates[:, 1], label="Neuron 1 rate")
    >>> _ = plt.legend()
    >>> _ = plt.title("Simulated firing rates")
    >>> _ = plt.show()
    """
    if isinstance(feedforward_input, FeaturePytree):
        raise ValueError(
            "simulate_recurrent works only with arrays. "
            "FeaturePytree provided instead!"
        )
    # convert to jnp.ndarray of floats
    coupling_basis_matrix = jnp.asarray(coupling_basis_matrix, dtype=float)
    coupling_coef = jnp.asarray(coupling_coef, dtype=float)
    feedforward_coef = jnp.asarray(feedforward_coef, dtype=float)
    intercepts = jnp.asarray(intercepts, dtype=float)
    feedforward_input = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=float), feedforward_input
    )
    init_y = jnp.asarray(init_y, dtype=float)

    # check that n_neurons is consistent
    n_neurons = intercepts.shape[0]
    if (
        feedforward_input.shape[1] != n_neurons
        or feedforward_coef.shape[0] != n_neurons
        or init_y.shape[1] != n_neurons
        or coupling_coef.shape[0] != n_neurons
        or coupling_coef.shape[1] != n_neurons
    ):
        raise ValueError(
            "The number of neurons provided in the inputs is inconsistent!"
        )

    # checks the input size
    validation.check_tree_leaves_dimensionality(
        feedforward_input,
        expected_dim=3,
        err_message="`feedforward_input` must be three-dimensional, with shape "
        "(n_timebins, n_neurons, n_features) or pytree of the same shape.",
    )
    validation.check_tree_axis_consistency(
        feedforward_coef,
        feedforward_input,
        axis_1=1,
        axis_2=2,
        err_message="Inconsistent number of features. "
        f"spike basis coefficients has {jax.tree_util.tree_map(lambda p: p.shape[0], feedforward_coef)} features, "
        f"X has {jax.tree_util.tree_map(lambda x: x.shape[2], feedforward_input)} features instead!",
    )

    validation.error_invalid_entry(feedforward_input)

    # validate y
    validation.check_tree_leaves_dimensionality(
        init_y,
        expected_dim=2,
        err_message="`init_y` must be two-dimensional, with shape (n_timebins, ).",
    )
    n_basis = coupling_coef.shape[-1]
    coupling_coef = coupling_coef.reshape(n_neurons, -1)

    if coupling_basis_matrix.shape[1] * n_neurons != coupling_coef.shape[1]:
        raise ValueError(
            f"Inconsistent number of features. `coupling_basis_matrix` assumes "
            f"{coupling_basis_matrix.shape[1]} basis functions for the coupling filters, "
            f"`coupling_coef` assumes {n_basis} basis functions instead."
        )

    if init_y.shape[0] != coupling_basis_matrix.shape[0]:
        raise ValueError(
            "`init_y` and `coupling_basis_matrix`"
            " should have the same window size! "
            f"`init_y` window size: {init_y.shape[0]}, "
            f"`coupling_basis_matrix` window size: {coupling_basis_matrix.shape[0]}"
        )

    subkeys = jax.random.split(random_key, num=feedforward_input.shape[0])
    # (n_samples, n_neurons)
    feed_forward_contrib = jnp.einsum("ik,tik->ti", feedforward_coef, feedforward_input)

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
        conv_act = convolve.tensor_convolve(activity, coupling_basis_matrix).reshape(
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
        firing_rate = inverse_link_function(
            coupling_coef.dot(conv_act) + input_slice + intercepts
        )

        # Simulate activity based on the predicted firing rate
        new_act = jax.random.poisson(key, firing_rate)

        # Shift of one sample the spike count window
        # for the next iteration (i.e. remove the first counts, and
        # stack the newly generated sample)
        # Increase the t_sample by one
        carry = jnp.vstack((activity[1:], new_act)), t_sample + 1
        return carry, (new_act, firing_rate)

    _, outputs = jax.lax.scan(scan_fn, (init_y, 0), subkeys)
    simulated_activity, firing_rates = outputs
    return simulated_activity, firing_rates
