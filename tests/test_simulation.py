import itertools
from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import numpy as np
import pytest

import nemos.simulation as simulation
from nemos import basis


@pytest.mark.parametrize(
    "inhib_a, expectation",
    [
        (
            -1,
            pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0."),
        ),
        (
            0,
            pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0."),
        ),
        (1, does_not_raise()),
    ],
)
def test_difference_of_gammas_inhib_a(inhib_a, expectation):
    with expectation:
        simulation.difference_of_gammas(10, inhib_a=inhib_a)


@pytest.mark.parametrize(
    "excit_a, expectation",
    [
        (
            -1,
            pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0."),
        ),
        (
            0,
            pytest.raises(ValueError, match="Gamma parameter [a-z]+_[a,b] must be >0."),
        ),
        (1, does_not_raise()),
    ],
)
def test_difference_of_gammas_excit_a(excit_a, expectation):
    with expectation:
        simulation.difference_of_gammas(10, excit_a=excit_a)


@pytest.mark.parametrize(
    "upper_percentile, expectation",
    [
        (
            -0.1,
            pytest.raises(
                ValueError,
                match=r"upper_percentile should lie in the \[0, 1\) interval.",
            ),
        ),
        (0, does_not_raise()),
        (0.1, does_not_raise()),
        (
            1,
            pytest.raises(
                ValueError,
                match=r"upper_percentile should lie in the \[0, 1\) interval.",
            ),
        ),
        (
            10,
            pytest.raises(
                ValueError,
                match=r"upper_percentile should lie in the \[0, 1\) interval.",
            ),
        ),
    ],
)
def test_difference_of_gammas_percentile_params(upper_percentile, expectation):
    with expectation:
        simulation.difference_of_gammas(10, upper_percentile)


@pytest.mark.parametrize("window_size", [0, 1, 2])
def test_difference_of_gammas_output_shape(window_size):
    result_size = simulation.difference_of_gammas(window_size).size
    assert (
        result_size == window_size
    ), f"Expected output size {window_size}, but got {result_size}"


@pytest.mark.parametrize("window_size", [1, 2, 10])
def test_difference_of_gammas_output_norm(window_size):
    result = simulation.difference_of_gammas(window_size)
    assert np.allclose(
        np.linalg.norm(result, ord=2), 1
    ), "The output of difference_of_gammas is not unit norm."


@pytest.mark.parametrize(
    "coupling_filters, expectation",
    [
        (
            np.zeros((10,)),
            pytest.raises(
                ValueError, match=r"coupling_filters must be a 3 dimensional array"
            ),
        ),
        (
            np.zeros((10, 2)),
            pytest.raises(
                ValueError, match=r"coupling_filters must be a 3 dimensional array"
            ),
        ),
        (np.zeros((10, 2, 2)), does_not_raise()),
        (
            np.zeros((10, 2, 2, 2)),
            pytest.raises(
                ValueError, match=r"coupling_filters must be a 3 dimensional array"
            ),
        ),
    ],
)
def test_regress_filter_coupling_filters_dim(coupling_filters, expectation):
    ws = coupling_filters.shape[0]
    with expectation:
        simulation.regress_filter(coupling_filters, np.zeros((ws, 3)))


@pytest.mark.parametrize(
    "eval_basis, expectation",
    [
        (
            np.zeros((10,)),
            pytest.raises(
                ValueError, match=r"eval_basis must be a 2 dimensional array"
            ),
        ),
        (np.zeros((10, 2)), does_not_raise()),
        (
            np.zeros((10, 2, 2)),
            pytest.raises(
                ValueError, match=r"eval_basis must be a 2 dimensional array"
            ),
        ),
        (
            np.zeros((10, 2, 2, 2)),
            pytest.raises(
                ValueError, match=r"eval_basis must be a 2 dimensional array"
            ),
        ),
    ],
)
def test_regress_filter_eval_basis_dim(eval_basis, expectation):
    ws = eval_basis.shape[0]
    with expectation:
        simulation.regress_filter(np.zeros((ws, 1, 1)), eval_basis)


@pytest.mark.parametrize(
    "delta_ws, expectation",
    [
        (
            -1,
            pytest.raises(
                ValueError, match=r"window_size mismatch\. The window size of "
            ),
        ),
        (0, does_not_raise()),
        (
            1,
            pytest.raises(
                ValueError, match=r"window_size mismatch\. The window size of "
            ),
        ),
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
def test_regress_filter_weights_size(
    window_size, n_neurons_sender, n_neurons_receiver, n_basis_funcs
):
    weights = simulation.regress_filter(
        np.zeros((window_size, n_neurons_sender, n_neurons_receiver)),
        np.zeros((window_size, n_basis_funcs)),
    )
    assert weights.shape[0] == n_neurons_sender, (
        "First dimension of weights (n_neurons_receiver) does not "
        "match the second dimension of coupling_filters."
    )
    assert weights.shape[1] == n_neurons_receiver, (
        "Second dimension of weights (n_neuron_sender) does not "
        "match the third dimension of coupling_filters."
    )
    assert weights.shape[2] == n_basis_funcs, (
        "Third dimension of weights (n_basis_funcs) does not "
        "match the second dimension of eval_basis."
    )


def test_least_square_correctness():
    """
    Test the correctness of the least square estimate by enforcing an invertible map,
    i.e. a map for which the least-square estimator matches the original weights.
    """
    # set up problem dimensionality
    ws, n_neurons_receiver, n_neurons_sender, n_basis_funcs = 100, 1, 2, 10
    # evaluate a basis
    _, eval_basis = basis.RaisedCosineLinearEval(n_basis_funcs).evaluate_on_grid(ws)
    # generate random weights to define filters
    weights = np.random.normal(
        size=(n_neurons_receiver, n_neurons_sender, n_basis_funcs)
    )
    # define filters as linear combination of basis elements
    coupling_filt = np.einsum("ijk, tk -> tij", weights, eval_basis)
    # recover weights by means of linear regression
    weights_lsq = simulation.regress_filter(coupling_filt, eval_basis)
    # check the exact matching of the filters up to numerical error
    assert np.allclose(weights_lsq, weights)


class TestSimulateRecurrent:
    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of neurons"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of neurons"),
            ),
        ],
    )
    def test_simulate_n_neuron_match_input(
        self, delta_n_neuron, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method to ensure that The number of neurons in the input
        matches the model's parameters.
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        if delta_n_neuron != 0:
            feedforward_input = np.zeros(
                (
                    feedforward_input.shape[0],
                    feedforward_input.shape[1] + delta_n_neuron,
                    feedforward_input.shape[2],
                )
            )
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="`feedforward_input` must be three-dimensional"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="`feedforward_input` must be three-dimensional"
                ),
            ),
        ],
    )
    def test_simulate_input_dimensionality(
        self, delta_dim, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method with input data of different dimensionalities.
        Ensure correct dimensionality for input.
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        if delta_dim == -1:
            feedforward_input = np.zeros(feedforward_input.shape[:2])
        elif delta_dim == 1:
            feedforward_input = np.zeros(feedforward_input.shape + (1,))
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="`init_y` must be two-dimensional")),
        ],
    )
    def test_simulate_y_dimensionality(
        self, delta_dim, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method with init_spikes of different dimensionalities.
        Ensure correct dimensionality for init_spikes.
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        if delta_dim == -1:
            init_spikes = np.zeros((feedforward_input.shape[0],))
        elif delta_dim == 1:
            init_spikes = np.zeros(
                (feedforward_input.shape[0], feedforward_input.shape[1], 1)
            )
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of neurons"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of neurons"),
            ),
        ],
    )
    def test_simulate_n_neuron_match_y(
        self, delta_n_neuron, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method to ensure that The number of neurons in init_spikes
        matches the model's parameters.
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        init_spikes = jnp.zeros(
            (init_spikes.shape[0], feedforward_input.shape[1] + delta_n_neuron)
        )
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
        ],
    )
    def test_simulate_time_point_match_y(
        self, delta_tp, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method to ensure that the time points in init_y
        are consistent with the coupling_basis window size (they must be equal).
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        init_spikes = jnp.zeros((init_spikes.shape[0] + delta_tp, init_spikes.shape[1]))
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
        ],
    )
    def test_simulate_time_point_match_coupling_basis(
        self, delta_tp, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method to ensure that the window size in coupling_basis
        is consistent with the time-points in init_spikes (they must be equal).
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        coupling_basis = jnp.zeros(
            (coupling_basis.shape[0] + delta_tp,) + coupling_basis.shape[1:]
        )
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_features, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError,
                    match="Inconsistent number of features. spike basis coefficients has",
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError,
                    match="Inconsistent number of features. spike basis coefficients has",
                ),
            ),
        ],
    )
    def test_simulate_feature_consistency_input(
        self, delta_features, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method ensuring the number of features in `feedforward_input` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        feedforward_input = jnp.zeros(
            (
                feedforward_input.shape[0],
                feedforward_input.shape[1],
                feedforward_input.shape[2] + delta_features,
            )
        )
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )

    @pytest.mark.parametrize(
        "delta_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_simulate_feature_consistency_coupling_basis(
        self, delta_features, expectation, coupled_model_simulate
    ):
        """
        Test the `simulate` method ensuring the number of features in `coupling_basis` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        (
            coupling_coeff,
            feedforward_coeff,
            intercepts,
            random_key,
            feedforward_input,
            coupling_basis,
            init_spikes,
            inv_link_func,
        ) = coupled_model_simulate
        coupling_basis = jnp.zeros(
            (coupling_basis.shape[0], coupling_basis.shape[1] + delta_features)
        )
        with expectation:
            simulation.simulate_recurrent(
                coupling_coeff,
                feedforward_coeff,
                intercepts,
                random_key,
                feedforward_input,
                coupling_basis,
                init_spikes,
                inv_link_func,
            )
