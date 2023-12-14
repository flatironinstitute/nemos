import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import nemos.basis as basis
import nemos.simulation as simulation


@pytest.mark.parametrize(
        "inhib_a, expectation",
        [
            (-1, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (0, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (1, does_not_raise()),
        ],
    )
def test_difference_of_gammas_inhib_a(inhib_a, expectation):
    with expectation:
        simulation.difference_of_gammas(10, inhib_a=inhib_a)


@pytest.mark.parametrize(
        "excit_a, expectation",
        [
            (-1, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (0, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (1, does_not_raise()),
        ],
    )
def test_difference_of_gammas_excit_a(excit_a, expectation):
    with expectation:
        simulation.difference_of_gammas(10, excit_a=excit_a)


@pytest.mark.parametrize(
        "inhib_b, expectation",
        [
            (-1, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (0, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (1, does_not_raise()),
        ],
    )
def test_difference_of_gammas_excit_a(inhib_b, expectation):
    with expectation:
        simulation.difference_of_gammas(10, inhib_b=inhib_b)


@pytest.mark.parametrize(
        "excit_b, expectation",
        [
            (-1, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (0, pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0.")),
            (1, does_not_raise()),
        ],
    )
def test_difference_of_gammas_excit_a(excit_b, expectation):
    with expectation:
        simulation.difference_of_gammas(10, excit_b=excit_b)


@pytest.mark.parametrize(
        "upper_percentile, expectation",
        [
            (-0.1, pytest.raises(ValueError, match=r"upper_percentile should lie in the \[0, 1\) interval.")),
            (0, does_not_raise()),
            (0.1, does_not_raise()),
            (1, pytest.raises(ValueError, match=r"upper_percentile should lie in the \[0, 1\) interval.")),
            (10, pytest.raises(ValueError, match=r"upper_percentile should lie in the \[0, 1\) interval.")),
        ],
    )
def test_difference_of_gammas_percentile_params(upper_percentile, expectation):
    with expectation:
        simulation.difference_of_gammas(10, upper_percentile)


@pytest.mark.parametrize("window_size", [0, 1, 2])
def test_difference_of_gammas_output_shape(window_size):
    result_size = simulation.difference_of_gammas(window_size).size
    assert result_size == window_size, f"Expected output size {window_size}, but got {result_size}"


@pytest.mark.parametrize("window_size", [1, 2, 10])
def test_difference_of_gammas_output_norm(window_size):
    result = simulation.difference_of_gammas(window_size)
    assert np.allclose(np.linalg.norm(result, ord=2),1), "The output of difference_of_gammas is not unit norm."


@pytest.mark.parametrize(
        "coupling_filters, expectation",
        [
            (np.zeros((10, )), pytest.raises(ValueError, match=r"coupling_filters must be a 3 dimensional array")),
            (np.zeros((10, 2)), pytest.raises(ValueError, match=r"coupling_filters must be a 3 dimensional array")),
            (np.zeros((10, 2, 2)), does_not_raise()),
            (np.zeros((10, 2, 2, 2)), pytest.raises(ValueError, match=r"coupling_filters must be a 3 dimensional array"))
        ],
    )
def test_regress_filter_coupling_filters_dim(coupling_filters, expectation):
    ws = coupling_filters.shape[0]
    with expectation:
        simulation.regress_filter(coupling_filters, np.zeros((ws, 3)))


@pytest.mark.parametrize(
        "eval_basis, expectation",
        [
            (np.zeros((10, )), pytest.raises(ValueError, match=r"eval_basis must be a 2 dimensional array")),
            (np.zeros((10, 2)), does_not_raise()),
            (np.zeros((10, 2, 2)), pytest.raises(ValueError, match=r"eval_basis must be a 2 dimensional array")),
            (np.zeros((10, 2, 2, 2)), pytest.raises(ValueError, match=r"eval_basis must be a 2 dimensional array"))
        ],
    )
def test_regress_filter_eval_basis_dim(eval_basis, expectation):
    ws = eval_basis.shape[0]
    with expectation:
        simulation.regress_filter(np.zeros((ws, 1, 1)), eval_basis)


@pytest.mark.parametrize(
        "delta_ws, expectation",
        [
            (-1, pytest.raises(ValueError, match=r"window_size mismatch\. The window size of ")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match=r"window_size mismatch\. The window size of ")),
        ],
    )
def test_regress_filter_window_size_matching(delta_ws, expectation):
    ws = 2
    with expectation:
        simulation.regress_filter(np.zeros((ws, 1, 1)), np.zeros((ws + delta_ws, 1)))


@pytest.mark.parametrize(
        "window_size, n_neurons_sender, n_neurons_receiver, n_basis_funcs",
        [x for x in itertools.product([1, 2], [1, 2], [1, 2], [1, 2])],
    )
def test_regress_filter_weights_size(window_size, n_neurons_sender, n_neurons_receiver, n_basis_funcs):
    weights = simulation.regress_filter(
        np.zeros((window_size, n_neurons_sender, n_neurons_receiver)),
        np.zeros((window_size, n_basis_funcs))
    )
    assert weights.shape[0] == n_neurons_sender, (f"First dimension of weights (n_neurons_receiver) does not "
                                                  f"match the second dimension of coupling_filters.")
    assert weights.shape[1] == n_neurons_receiver, (f"Second dimension of weights (n_neuron_sender) does not "
                                                    f"match the third dimension of coupling_filters.")
    assert weights.shape[2] == n_basis_funcs, (f"Third dimension of weights (n_basis_funcs) does not "
                                               f"match the second dimension of eval_basis.")


def test_least_square_correctness():
    """
    Test the correctness of the least square estimate by enforcing an invertible map,
    i.e. a map for which the least-square estimator matches the original weights.
    """
    # set up problem dimensionality
    ws, n_neurons_receiver, n_neurons_sender, n_basis_funcs = 100, 1, 2, 10
    # evaluate a basis
    _, eval_basis = basis.RaisedCosineBasisLog(n_basis_funcs).evaluate_on_grid(ws)
    # generate random weights to define filters
    weights = np.random.normal(size=(n_neurons_receiver, n_neurons_sender, n_basis_funcs))
    # define filters as linear combination of basis elements
    coupling_filt = np.einsum("ijk, tk -> tij", weights, eval_basis)
    # recover weights by means of linear regression
    weights_lsq = simulation.regress_filter(coupling_filt, eval_basis)
    # check the exact matching of the filters up to numerical error
    assert np.allclose(weights_lsq, weights)


