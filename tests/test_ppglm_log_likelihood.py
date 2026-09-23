from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from pynapple import IntervalSet

from nemos.basis import RaisedCosineLogEval
from nemos.glm.validation import to_glm_params
from nemos.pp_glm import log_likelihood, utils
from nemos.pp_glm.data import MCSamplePPGLM, PredictorsPPGLM, SpikesPPGLM

NLL_KWARG_NAMES = (
    "inverse_link_function",
    "M_samples",
    "M_grid",
    "recording_time",
    "n_basis_funcs",
    "scan_size",
    "max_window",
    "eval_function",
)


def nll_kwargs(dataset, **overrides):
    """Collect the arguments of the nll that will become model attributes."""
    kwargs = {key: dataset[key] for key in NLL_KWARG_NAMES}
    kwargs.update(overrides)
    return kwargs


def create_dataset_constant_rate(recording_time, n_neurons=3, **kwargs):
    """Dataset with zero coefficients, so that the firing rates are constant in time."""
    dataset = create_dataset(
        n_neurons=n_neurons, sim_time=float(recording_time.end[-1]), **kwargs
    )
    n_basis_funcs = dataset["n_basis_funcs"]
    # intercepts of one sign, so that the spike term does not cancel itself out
    if kwargs.get("all_to_one", False):
        coef, intercept = jnp.zeros(n_neurons * n_basis_funcs), jnp.atleast_1d(1.5)
    else:
        coef, intercept = (
            jnp.zeros((n_neurons * n_basis_funcs, n_neurons)),
            jnp.linspace(0.5, 1.5, n_neurons),
        )
    in_epoch = np.any(
        [
            (np.asarray(dataset["y"].times) >= s)
            & (np.asarray(dataset["y"].times) <= e)
            for s, e in zip(recording_time.start, recording_time.end)
        ],
        axis=0,
    )
    dataset["y"] = jax.tree_util.tree_map(lambda arr: arr[in_epoch], dataset["y"])
    dataset["params"] = to_glm_params((coef, intercept))
    dataset["recording_time"] = recording_time
    dataset["M_grid"] = utils.build_mc_sampling_grid(
        recording_time, dataset["M_samples"]
    )
    return dataset


def create_basis(n_basis_funcs=4, history_window=0.01):
    """Use nemos RC Eval basis and return the evaluate method"""
    basis = RaisedCosineLogEval(n_basis_funcs, bounds=(0, history_window), fill_value=0)
    return lambda pts: basis.evaluate(pts)


def create_params(n_neurons, n_basis_funcs, all_to_one=False):
    """Use PP-GLM params structures"""
    if all_to_one:
        return to_glm_params(
            (
                jnp.ones(n_neurons * n_basis_funcs),
                jnp.atleast_1d(jnp.zeros(1)),
            )
        )
    return to_glm_params(
        (
            jnp.ones((n_neurons * n_basis_funcs, n_neurons)),
            jnp.atleast_1d(jnp.zeros(n_neurons)),
        )
    )


def unpack_params(params, n_basis_funcs):
    """Unpack and reshape params, extract n_predictors"""
    weights = utils._reshape_2d_coef(params.coef)
    bias = params.intercept
    n_predictors = weights.shape[0] // n_basis_funcs
    return weights, bias, n_predictors


def create_dataset(
    n_neurons=5,
    n_spikes=400,
    sim_time=5.0,
    M_samples=100,
    n_basis_funcs=4,
    history_window=0.01,
    scan_size=3,
    seed=0,
    all_to_one=False,
) -> dict[str, Any]:
    """Create a fake dataset (without running an actual simulation) for fitting an all-to-all or
    all-to-one coupled model. Returns preprocessed inputs, model hyperparams and arbitrary PP-GLM params
    """
    recording_time = IntervalSet(0, sim_time)
    M_grid = utils.build_mc_sampling_grid(recording_time, M_samples)
    eval_function = create_basis(n_basis_funcs, history_window)

    np.random.seed(seed)
    spike_times = np.sort(np.random.uniform(0, sim_time, n_spikes))
    spike_ids = np.random.choice(np.arange(n_neurons), n_spikes)

    X = PredictorsPPGLM(
        times=jnp.asarray(spike_times), predictor_ids=jnp.asarray(spike_ids, dtype=int)
    )

    y = SpikesPPGLM(
        times=jnp.asarray(spike_times),
        neuron_ids=jnp.asarray(spike_ids, dtype=int),
        timestamp_idx=jnp.arange(spike_times.size, dtype=int),
    )

    if all_to_one:
        n_target = 0
        mask = y.neuron_ids == n_target
        y = SpikesPPGLM(
            times=y.times[mask],
            neuron_ids=y.neuron_ids[mask],
            timestamp_idx=y.timestamp_idx[mask],
        )

    max_window = int(
        utils.compute_max_window_size(jnp.array([-history_window, 0]), X.times, X.times)
    )
    X, y = utils.adjust_indices_and_spike_times(X, history_window, max_window, y)

    return dict(
        params=create_params(n_neurons, n_basis_funcs, all_to_one),
        random_key=jax.random.PRNGKey(seed),
        X=X,
        y=y,
        recording_time=recording_time,
        M_samples=M_samples,
        M_grid=M_grid,
        n_basis_funcs=n_basis_funcs,
        scan_size=scan_size,
        max_window=max_window,
        eval_function=eval_function,
        history_window=history_window,
        inverse_link_function=jnp.exp,
        n_neurons=n_neurons,
    )


def create_dataset_single_spike(
    spike_time, history_window=0.01, n_basis_funcs=4, M_samples=100, seed=0
) -> dict[str, Any]:
    """
    Create the minimal dataset for a single-neuron, single-spike scenario.
    """
    n_neurons = 1
    sim_time = spike_time + 1.0
    recording_time = IntervalSet(0, sim_time)
    M_grid = utils.build_mc_sampling_grid(recording_time, M_samples)
    eval_function = create_basis(n_basis_funcs, history_window)

    spike_times = jnp.array([spike_time])
    spike_ids = jnp.array([0]).astype(int)

    X = PredictorsPPGLM(
        times=jnp.asarray(spike_times), predictor_ids=jnp.asarray(spike_ids, dtype=int)
    )

    y = SpikesPPGLM(
        times=jnp.asarray(spike_times),
        neuron_ids=jnp.asarray(spike_ids, dtype=int),
        timestamp_idx=jnp.arange(1, dtype=int),
    )

    max_window = int(
        utils.compute_max_window_size(
            jnp.array([-history_window, 0.0]), X.times, X.times
        )
    )
    X, y = utils.adjust_indices_and_spike_times(X, history_window, max_window, y)

    params = to_glm_params(
        (
            jnp.ones((n_neurons * n_basis_funcs, n_neurons)),
            jnp.atleast_1d(0.0),
        )
    )
    return dict(
        params=params,
        random_key=jax.random.PRNGKey(seed),
        X=X,
        y=y,
        recording_time=recording_time,
        M_samples=M_samples,
        M_grid=M_grid,
        n_basis_funcs=n_basis_funcs,
        scan_size=1,
        max_window=max_window,
        eval_function=eval_function,
        inverse_link_function=jnp.exp,
        history_window=history_window,
        n_neurons=n_neurons,
    )


class TestUtils:
    def test_reshape_2d_coef(self):
        """Test that reshaping works correctly for 1d and 2d scenarios"""
        # 1d (single postsynaptic neuron) gets a trailing dimension
        n_predictors, n_bases = 5, 4
        w = jnp.ones(n_predictors * n_bases)
        out = utils._reshape_2d_coef(w)

        assert out.shape == (n_predictors * n_bases, 1)

        # 2d (population) kept unchanged
        n_predictors, n_bases, n_target = 5, 4, 3
        w = jnp.ones((n_predictors * n_bases, n_target))
        out = utils._reshape_2d_coef(w)

        assert out.shape == (n_predictors * n_bases, n_target)

        # anything else is rejected
        with pytest.raises(ValueError):
            utils._reshape_2d_coef(jnp.ones((n_predictors, n_bases, n_target)))

    def test_reshape_and_pad_eval_points(self):
        """Test that reshaping works properly and that the validity mask marks the padding"""
        # when divisible, padding length is 0, all valid
        times = MCSamplePPGLM(
            times=jnp.ones(8), timestamp_idx=jnp.arange(8).astype(int)
        )

        reshaped, valid = utils._reshape_and_pad_eval_points(times, chunk_size=2)
        jax.tree_util.tree_map(
            lambda arr: np.testing.assert_array_equal(
                arr.shape, (4, 2)
            ),  # (n_chunks, chunk_size)
            reshaped,
        )
        assert valid.shape == (4, 2)
        assert np.all(valid)

        # when not divisible, padding fills to next multiple
        times = MCSamplePPGLM(
            times=jnp.ones(9), timestamp_idx=jnp.arange(9).astype(int)
        )
        reshaped, valid = utils._reshape_and_pad_eval_points(times, chunk_size=2)
        jax.tree_util.tree_map(
            lambda arr: np.testing.assert_array_equal(
                arr.shape, (5, 2)
            ),  # (n_chunks, chunk_size)
            reshaped,
        )
        assert valid.shape == (5, 2)
        assert valid.sum() == 9
        # the padded entry is the final one and is the only invalid entry
        assert not valid[-1, -1]

        # test that padding is the last value and that the mask lines up with it
        times = MCSamplePPGLM(
            times=jnp.ones(4), timestamp_idx=jnp.arange(4).astype(int)
        )
        reshaped, valid = utils._reshape_and_pad_eval_points(times, chunk_size=3)
        pad_len = -4 % 3
        assert valid.sum() == 4

        # check padding values match last element of original
        jax.tree_util.tree_map(
            lambda orig, resh: np.testing.assert_array_equal(
                resh.reshape(-1)[-pad_len:], orig[-1]
            ),
            times,
            reshaped,
        )

        # check the mask marks exactly the padded positions
        np.testing.assert_array_equal(valid.reshape(-1)[-pad_len:], False)
        np.testing.assert_array_equal(valid.reshape(-1)[:-pad_len], True)

    @pytest.mark.requires_x64
    def test_build_mc_sampling_grid(self):
        """test that the grid is built correctly with multiple epochs"""
        recording_time = IntervalSet(start=[0.0, 6.0], end=[4.0, 10.0])
        grid = utils.build_mc_sampling_grid(recording_time, M_samples=100)

        # assert grid size is exactly M_samples
        assert grid.shape[0] == 100

        # assert that all grid points are within epochs
        in_epoch = np.any(
            [
                (grid >= s) & (grid <= e)
                for s, e in zip(recording_time.start, recording_time.end)
            ],
            axis=0,
        )

        assert np.all(in_epoch)

        # test that edge case when the number of samples is less than the number of epochs
        # raises an error
        starts = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ends = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        recording_time = IntervalSet(start=starts, end=ends)

        with pytest.raises(ValueError):
            utils.build_mc_sampling_grid(recording_time, M_samples=3)

    @pytest.mark.requires_x64
    def test_adjust_indices_and_spike_times(self):
        """Test that shapes and indices are shifted correctly and that padding
        is outside the history window"""

        # returns already preprocessed X and y
        dataset = create_dataset()

        # test X shape increase by history window
        n_spk_original = dataset["y"].times.shape[0]  # unchanged
        jax.tree_util.tree_map(
            lambda arr: np.testing.assert_array_equal(
                arr.shape[0], n_spk_original + dataset["max_window"]
            ),
            dataset["X"],
        )

        # test y index is shifted by max_window
        assert dataset["y"].timestamp_idx[0] == dataset["max_window"]

        # test padding values are out of bound and basis evals to 0
        bound = dataset["recording_time"].start[0] - dataset["history_window"]
        padding = dataset["X"].times[: dataset["max_window"]]
        assert np.all(padding < bound)

        first_spike = dataset["y"].times[0]
        dts = first_spike - padding
        basis_at_dts = dataset["eval_function"](dts)

        np.testing.assert_array_equal(basis_at_dts, 0)

    def test_compute_max_window_size(self):
        """Test max window is selected correctly with any non-empty dataset"""

        # test the edge case with a single spike dataset
        dataset = create_dataset_single_spike(1.0)

        max_window = int(
            utils.compute_max_window_size(
                jnp.array([-dataset["history_window"], 0]),
                dataset["X"].times,
                dataset["X"].times,
            )
        )

        assert max_window == 0

        # one reference spike, multiple events
        ref = jnp.array([1.0])
        events = jnp.array(
            [0.1, 0.5, 0.990, 0.993, 0.995]
        )  # the last 3 fall within 0.01 s
        max_window = int(
            utils.compute_max_window_size(
                jnp.array([-dataset["history_window"], 0]), ref, events
            )
        )

        assert max_window == 3

        # multiple spikes, multiple events
        ref = jnp.array([1.0, 3.0])
        events = jnp.array(
            [0.990, 0.993, 0.995, 2.990, 2.993, 2.995, 2.999]  # 3 events within ref 1
        )  # 4 events within ref 2

        max_window = int(
            utils.compute_max_window_size(
                jnp.array([-dataset["history_window"], 0]), ref, events
            )
        )

        assert max_window == 4

    def test_slice_array(self):
        """slice_array returns the window_size entries ending at i, exclusive."""
        array = jnp.arange(10.0)

        np.testing.assert_array_equal(utils.slice_array(array, 7, 3), [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(utils.slice_array(array, 3, 3), [0.0, 1.0, 2.0])

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "dataset_kwargs",
        [
            dict(),
            dict(n_spikes=60, history_window=0.1),
            dict(all_to_one=True),
            dict(n_spikes=1000, history_window=0.2),
        ],
    )
    def test_no_window_reaches_past_the_first_event(self, dataset_kwargs):
        """Every history window starts at a non-negative index, for spikes and for MC samples."""
        # for MC samples the index comes from searchsorted, which returns 0 for a
        # sample landing before the first event unless the padding is there
        dataset = create_dataset(**dataset_kwargs)
        max_window = dataset["max_window"]

        assert int(dataset["y"].timestamp_idx.min()) >= max_window

        draw = jax.vmap(
            lambda key: (
                log_likelihood._draw_mc_sample(
                    dataset["X"],
                    key,
                    dataset["M_samples"],
                    dataset["recording_time"].tot_length(),
                    dataset["M_grid"],
                ).timestamp_idx
            )
        )
        mc_idx = draw(jax.random.split(jax.random.PRNGKey(0), 50))

        assert int(mc_idx.min()) >= max_window

    @pytest.mark.requires_x64
    def test_padding_prevents_windows_reaching_past_the_first_event(self):
        """Without the prepended events the first spike's feature vector is nan instead of zero."""
        dataset = create_dataset()
        max_window, X = dataset["max_window"], dataset["X"]
        unpadded = PredictorsPPGLM(
            times=X.times[max_window:], predictor_ids=X.predictor_ids[max_window:]
        )
        first_event = unpadded.times[0]

        def design_row(predictors, timestamp_idx):
            return log_likelihood._compute_design_matrix(
                jnp.array([first_event]),
                jnp.array([timestamp_idx]),
                predictors,
                dataset["eval_function"],
                max_window,
                dataset["n_neurons"],
            )

        # the padded window holds only the dummy events, whose lags are far above the
        # history window; the unpadded one wraps to the end of the recording, whose
        # lags are negative, and the basis is nan below zero even with fill_value=0
        np.testing.assert_array_equal(design_row(X, max_window), 0.0)
        assert not jnp.all(design_row(unpadded, 0) == 0)

    @pytest.mark.requires_x64
    def test_reshape_and_pad_spikes(self):
        """Padding replicates the last entry of every field, so padded rows stay in range."""
        n_points, chunk_size = 5, 2
        eval_pts = SpikesPPGLM(
            times=jnp.arange(n_points, dtype=float),
            neuron_ids=jnp.arange(n_points, dtype=int),
            timestamp_idx=jnp.arange(n_points, dtype=int),
        )

        chunked, valid = utils._reshape_and_pad_eval_points(eval_pts, chunk_size)

        assert valid.sum() == n_points
        jax.tree_util.tree_map(
            lambda orig, resh: np.testing.assert_array_equal(
                resh.reshape(-1)[n_points:], orig[-1]
            ),
            eval_pts,
            chunked,
        )
        np.testing.assert_array_equal(chunked.neuron_ids.max(), n_points - 1)


@pytest.mark.requires_x64
class TestLogLikelihood:
    def test_single_spike_dataset(self):
        """
        Test the edge case with a single spike in the dataset.

        The negative log-likelihood should still be valid but only include the bias contribution.
        """
        dataset = create_dataset_single_spike(5.0)

        params = dataset["params"]
        X = dataset["X"]
        y = dataset["y"]
        inverse_link_function = dataset["inverse_link_function"]
        M_samples = dataset["M_samples"]
        M_grid = dataset["M_grid"]
        recording_time = dataset["recording_time"]
        n_basis_funcs = dataset["n_basis_funcs"]
        scan_size = dataset["scan_size"]
        max_window = dataset["max_window"]
        eval_function = dataset["eval_function"]

        # test that nll returns a finite number
        loss = log_likelihood._negative_log_likelihood(
            params,
            X,
            y,
            dataset["random_key"],
            inverse_link_function=inverse_link_function,
            M_samples=M_samples,
            M_grid=M_grid,
            recording_time=recording_time,
            n_basis_funcs=n_basis_funcs,
            scan_size=scan_size,
            max_window=max_window,
            eval_function=eval_function,
        )

        assert jnp.isfinite(loss)

        # the first term is log(exp(lambda_tilde)) = bias
        bias_contrib = params.intercept
        weights, bias, n_predictors = unpack_params(params, n_basis_funcs)
        log_lam_y = log_likelihood._compute_log_lambda_y(
            X,
            y,
            weights,
            bias,
            inverse_link_function,
            eval_function,
            max_window,
            n_predictors,
            scan_size,
        )

        np.testing.assert_almost_equal(log_lam_y, bias_contrib)

    @pytest.mark.parametrize(
        "all_to_one", [True, False], ids=["single_neuron", "population"]
    )
    def test_for_loop_ppglm_ll(self, all_to_one):
        """
        Test the model nll computation against a numpy loop implementation.

        Validates that the log-likelihood computed with a chunked lax.scan over the
        design matrix matches a naive loop implementation. Also validates that the
        padding added to keep the scan chunks a fixed size is masked out correctly.
        """
        dataset = create_dataset(all_to_one=all_to_one)

        params = dataset["params"]
        X = dataset["X"]
        y = dataset["y"]
        inverse_link_function = dataset["inverse_link_function"]
        M_samples = dataset["M_samples"]
        M_grid = dataset["M_grid"]
        recording_time = dataset["recording_time"]
        n_basis_funcs = dataset["n_basis_funcs"]
        scan_size = dataset["scan_size"]
        max_window = dataset["max_window"]
        eval_function = dataset["eval_function"]

        weights, bias, n_predictors = unpack_params(params, n_basis_funcs)

        # first ll term
        # chunked lax.scan over the design matrix
        log_lam_y_scan = log_likelihood._compute_log_lambda_y(
            X,
            y,
            weights,
            bias,
            inverse_link_function,
            eval_function,
            max_window,
            n_predictors,
            scan_size,
        )

        # numpy loop
        n_spikes = y.times.shape[0]
        n_targets = weights.shape[1]
        # (n_predictors, n_basis_funcs, n_targets); must match the row order that
        # _eval_point produces when it flattens the segment sum
        w = np.asarray(weights).reshape(n_predictors, n_basis_funcs, n_targets)

        log_lam_y_loop = 0
        for sp in range(n_spikes):
            t, id, slice_end = y.times[sp], y.neuron_ids[sp], y.timestamp_idx[sp]
            slice_start = slice_end - max_window
            spk_in_window = X.times[slice_start:slice_end]
            ids_in_window = X.predictor_ids[slice_start:slice_end]
            dts = t - spk_in_window
            basis_at_dts = eval_function(dts)
            selected_w = w[ids_in_window, :, id]
            lam_tilde = np.sum(basis_at_dts * selected_w) + bias[id]
            log_lam_y_loop += np.log(inverse_link_function(lam_tilde))

        np.testing.assert_almost_equal(log_lam_y_scan, log_lam_y_loop)

        # second ll term
        # uses the same random key as _negative_log_likelihood below
        mc_samples = log_likelihood._draw_mc_sample(
            X,
            dataset["random_key"],
            M_samples,
            recording_time.tot_length(),
            M_grid,
        )

        # chunked lax.scan for all postsynaptic neurons
        mc_est_scan = log_likelihood._compute_mc_estimate(
            X,
            mc_samples,
            weights,
            bias,
            inverse_link_function,
            eval_function,
            max_window,
            n_predictors,
            scan_size,
        )

        # numpy loop
        mc_est_loop = 0
        for sp in range(M_samples):
            t, slice_end = mc_samples.times[sp], mc_samples.timestamp_idx[sp]
            slice_start = slice_end - max_window
            spk_in_window = X.times[slice_start:slice_end]
            ids_in_window = X.predictor_ids[slice_start:slice_end]
            dts = t - spk_in_window
            basis_at_dts = eval_function(dts)
            selected_w = w[ids_in_window]
            lam_tilde = (
                np.sum(basis_at_dts[:, :, None] * selected_w, axis=(0, 1)) + bias
            )
            mc_est_loop += inverse_link_function(lam_tilde).sum()

        np.testing.assert_almost_equal(mc_est_scan, mc_est_loop)

        # full nll computation
        loss_scan = log_likelihood._negative_log_likelihood(
            params,
            X,
            y,
            dataset["random_key"],
            inverse_link_function=inverse_link_function,
            M_samples=M_samples,
            M_grid=M_grid,
            recording_time=recording_time,
            n_basis_funcs=n_basis_funcs,
            scan_size=scan_size,
            max_window=max_window,
            eval_function=eval_function,
        )

        # nll from loop results
        loss_loop = (
            (recording_time.tot_length() / M_samples) * mc_est_loop
        ) - log_lam_y_loop
        loss_loop /= n_spikes

        np.testing.assert_almost_equal(loss_loop, loss_scan)

    @pytest.mark.parametrize("scan_size", [1, 2, 7, 500])
    def test_scan_size_invariance(self, scan_size):
        """
        Test that the result does not depend on how the evaluation points are chunked.

        Covers chunk sizes that divide the number of points exactly, that leave a
        partial final chunk, and that exceed the number of points entirely.
        """
        # n_spikes = 400
        dataset = create_dataset(scan_size=scan_size)

        loss = log_likelihood._negative_log_likelihood(
            dataset["params"],
            dataset["X"],
            dataset["y"],
            dataset["random_key"],
            inverse_link_function=dataset["inverse_link_function"],
            M_samples=dataset["M_samples"],
            M_grid=dataset["M_grid"],
            recording_time=dataset["recording_time"],
            n_basis_funcs=dataset["n_basis_funcs"],
            scan_size=scan_size,
            max_window=dataset["max_window"],
            eval_function=dataset["eval_function"],
        )

        reference = create_dataset(scan_size=3)
        loss_reference = log_likelihood._negative_log_likelihood(
            reference["params"],
            reference["X"],
            reference["y"],
            reference["random_key"],
            inverse_link_function=reference["inverse_link_function"],
            M_samples=reference["M_samples"],
            M_grid=reference["M_grid"],
            recording_time=reference["recording_time"],
            n_basis_funcs=reference["n_basis_funcs"],
            scan_size=3,
            max_window=reference["max_window"],
            eval_function=reference["eval_function"],
        )

        np.testing.assert_almost_equal(loss, loss_reference)

    @pytest.mark.parametrize(
        "all_to_one", [True, False], ids=["single_neuron", "population"]
    )
    @pytest.mark.parametrize(
        "recording_time",
        [IntervalSet(0, 5.0), IntervalSet(start=[0.0, 6.0], end=[4.0, 10.0])],
        ids=["one_epoch", "two_epochs"],
    )
    @pytest.mark.requires_x64
    def test_constant_intensity_matches_analytic_nll(self, all_to_one, recording_time):
        """With zero coefficients the rate is constant, so the MC term carries no sampling error."""
        dataset = create_dataset_constant_rate(recording_time, all_to_one=all_to_one)
        params = dataset["params"]
        weights, bias, n_predictors = unpack_params(params, dataset["n_basis_funcs"])

        # both terms are asserted separately: summed, they partly cancel, and the
        # spike term is the smaller one
        log_lambda_y = log_likelihood._compute_log_lambda_y(
            dataset["X"],
            dataset["y"],
            weights,
            bias,
            dataset["inverse_link_function"],
            dataset["eval_function"],
            dataset["max_window"],
            n_predictors,
            dataset["scan_size"],
        )
        np.testing.assert_allclose(
            log_lambda_y, jnp.sum(bias[dataset["y"].neuron_ids]), rtol=1e-10
        )

        mc_samples = log_likelihood._draw_mc_sample(
            dataset["X"],
            dataset["random_key"],
            dataset["M_samples"],
            recording_time.tot_length(),
            dataset["M_grid"],
        )
        mc_estimate = log_likelihood._compute_mc_estimate(
            dataset["X"],
            mc_samples,
            weights,
            bias,
            dataset["inverse_link_function"],
            dataset["eval_function"],
            dataset["max_window"],
            n_predictors,
            dataset["scan_size"],
        )
        mc_term = (recording_time.tot_length() / dataset["M_samples"]) * mc_estimate
        np.testing.assert_allclose(
            mc_term, recording_time.tot_length() * jnp.sum(jnp.exp(bias)), rtol=1e-10
        )

        loss = log_likelihood._negative_log_likelihood(
            params,
            dataset["X"],
            dataset["y"],
            dataset["random_key"],
            **nll_kwargs(dataset),
        )

        n_spikes = dataset["y"].times.shape[0]
        np.testing.assert_allclose(
            loss, (mc_term - log_lambda_y) / n_spikes, rtol=1e-10
        )

    @pytest.mark.parametrize(
        "all_to_one", [True, False], ids=["single_neuron", "population"]
    )
    @pytest.mark.requires_x64
    def test_grad_matches_finite_differences(self, all_to_one):
        """Autodiff gradient of the loss against central differences on every parameter entry."""
        # the history window must be long enough for every coefficient to enter some
        # window, otherwise most entries of the gradient are trivially zero
        dataset = create_dataset(
            n_neurons=3,
            n_spikes=200,
            M_samples=40,
            scan_size=7,
            history_window=0.1,
            all_to_one=all_to_one,
        )
        kwargs = nll_kwargs(dataset)
        flat_params, unravel = ravel_pytree(dataset["params"])

        def loss(flat):
            return log_likelihood._compute_loss(
                unravel(flat),
                dataset["X"],
                dataset["y"],
                dataset["random_key"],
                **kwargs,
            )

        grad = jax.grad(loss)(flat_params)
        assert jnp.all(grad != 0)

        # measured floor over all entries: 6.1e-10 at eps=1e-5, 2.8e-9 at 1e-6, 3.2e-8 at 1e-4
        eps = 1e-5
        finite_differences = jnp.array(
            [
                (loss(flat_params.at[i].add(eps)) - loss(flat_params.at[i].add(-eps)))
                / (2 * eps)
                for i in range(flat_params.size)
            ]
        )

        np.testing.assert_allclose(grad, finite_differences, rtol=1e-6)

    def test_random_key_controls_the_mc_draw(self):
        """The key argument alone decides the MC sample, so it changes the loss and repeats it exactly."""
        dataset = create_dataset()
        args = (dataset["params"], dataset["X"], dataset["y"])
        kwargs = nll_kwargs(dataset)
        # a split key, whose two words are both large, unlike jax.random.PRNGKey(n)
        rolled_key, _ = jax.random.split(dataset["random_key"])

        loss = log_likelihood._compute_loss(*args, dataset["random_key"], **kwargs)
        same_key = log_likelihood._compute_loss(*args, dataset["random_key"], **kwargs)
        other_key = log_likelihood._compute_loss(*args, rolled_key, **kwargs)

        np.testing.assert_array_equal(loss, same_key)
        assert not np.isclose(loss, other_key)

    def test_loss_is_jittable(self):
        """The loss compiles with params, X, y and the key as the only traced arguments."""
        dataset = create_dataset()
        kwargs = nll_kwargs(dataset)

        def loss(params, X, y, random_key):
            return log_likelihood._compute_loss(params, X, y, random_key, **kwargs)

        args = (dataset["params"], dataset["X"], dataset["y"], dataset["random_key"])
        np.testing.assert_allclose(jax.jit(loss)(*args), loss(*args))

        grad = jax.jit(jax.grad(loss))(*args)
        assert jnp.all(jnp.isfinite(grad.coef))

    @pytest.mark.parametrize("scan_size", [1, 7, 100, 500])
    def test_scan_size_invariance_per_term(self, scan_size):
        """Chunking changes neither term of the log-likelihood."""
        dataset = create_dataset(scan_size=scan_size)
        reference = create_dataset(scan_size=3)
        weights = utils._reshape_2d_coef(dataset["params"].coef)
        bias = dataset["params"].intercept
        n_predictors = weights.shape[0] // dataset["n_basis_funcs"]
        mc_samples = log_likelihood._draw_mc_sample(
            dataset["X"],
            dataset["random_key"],
            dataset["M_samples"],
            dataset["recording_time"].tot_length(),
            dataset["M_grid"],
        )
        args = (
            weights,
            bias,
            dataset["inverse_link_function"],
            dataset["eval_function"],
            dataset["max_window"],
            n_predictors,
        )

        for term, eval_pts in (
            (log_likelihood._compute_log_lambda_y, dataset["y"]),
            (log_likelihood._compute_mc_estimate, mc_samples),
        ):
            np.testing.assert_allclose(
                term(dataset["X"], eval_pts, *args, scan_size),
                term(reference["X"], eval_pts, *args, 3),
                rtol=1e-10,
            )


@pytest.mark.requires_x64
class TestDesignMatrix:
    def test_row_layout_is_predictor_major(self):
        """Column p * n_basis_funcs + j of the design matrix holds basis j summed over predictor p."""
        n_predictors, n_basis_funcs, max_window = 2, 3, 4
        eval_function = create_basis(n_basis_funcs, history_window=1.0)
        # lags spread over the whole window, so that no expected entry is zero
        times = jnp.array([0.1, 0.4, 0.95, 0.98])
        predictor_ids = jnp.array([0, 1, 1, 0])
        X = PredictorsPPGLM(times=times, predictor_ids=predictor_ids)

        A = log_likelihood._compute_design_matrix(
            jnp.array([1.0]),
            jnp.array([max_window]),
            X,
            eval_function,
            max_window,
            n_predictors,
        )

        basis_at_dts = eval_function(1.0 - times)
        assert jnp.all(A != 0)
        for p in range(n_predictors):
            for j in range(n_basis_funcs):
                np.testing.assert_allclose(
                    A[0, p * n_basis_funcs + j],
                    basis_at_dts[predictor_ids == p, j].sum(),
                )

    def test_empty_window_gives_a_zero_row(self):
        """An evaluation point with no event in its history window has an all-zero feature vector."""
        dataset = create_dataset()
        max_window = dataset["max_window"]
        assert max_window > 0

        A = log_likelihood._compute_design_matrix(
            jnp.array([dataset["recording_time"].start[0]]),
            jnp.array([max_window]),
            dataset["X"],
            dataset["eval_function"],
            max_window,
            dataset["n_neurons"],
        )

        np.testing.assert_array_equal(A, 0.0)

    def test_padding_events_do_not_enter_the_design_matrix(self):
        """Prepending more out-of-window events leaves every feature vector unchanged."""
        dataset = create_dataset()
        X, y, max_window = dataset["X"], dataset["y"], dataset["max_window"]
        # the check is vacuous unless some window actually reaches into the padding
        assert jnp.any(y.timestamp_idx - max_window < max_window)

        design_matrix = log_likelihood._compute_design_matrix(
            y.times,
            y.timestamp_idx,
            X,
            dataset["eval_function"],
            max_window,
            dataset["n_neurons"],
        )
        np.testing.assert_array_equal(design_matrix[0], 0.0)

        padded_X, padded_y = utils.adjust_indices_and_spike_times(
            X, dataset["history_window"], max_window, y
        )
        np.testing.assert_array_equal(
            design_matrix,
            log_likelihood._compute_design_matrix(
                padded_y.times,
                padded_y.timestamp_idx,
                padded_X,
                dataset["eval_function"],
                max_window,
                dataset["n_neurons"],
            ),
        )


@pytest.mark.requires_x64
class TestLinks:
    @pytest.mark.parametrize(
        "inverse_link_function, intercept",
        [(jnp.exp, 0.0), (jax.nn.softplus, 0.0), (jax.nn.softplus, -3.0)],
        ids=["exp", "softplus", "softplus_negative_rate"],
    )
    @pytest.mark.parametrize(
        "all_to_one", [True, False], ids=["single_neuron", "population"]
    )
    def test_log_lambda_y_matches_loop(
        self, all_to_one, inverse_link_function, intercept
    ):
        """The spike term matches a numpy loop for every inverse link."""
        dataset = create_dataset(all_to_one=all_to_one)
        X, y = dataset["X"], dataset["y"]
        weights, _, n_predictors = unpack_params(
            dataset["params"], dataset["n_basis_funcs"]
        )
        bias = jnp.full(weights.shape[1], intercept)
        eval_function = dataset["eval_function"]
        max_window = dataset["max_window"]

        log_lam_y_scan = log_likelihood._compute_log_lambda_y(
            X,
            y,
            weights,
            bias,
            inverse_link_function,
            eval_function,
            max_window,
            n_predictors,
            dataset["scan_size"],
        )

        w = np.asarray(weights).reshape(
            n_predictors, dataset["n_basis_funcs"], weights.shape[1]
        )
        log_lam_y_loop = 0
        for sp in range(y.times.shape[0]):
            t, id, slice_end = y.times[sp], y.neuron_ids[sp], y.timestamp_idx[sp]
            basis_at_dts = eval_function(
                t - X.times[slice_end - max_window : slice_end]
            )
            selected_w = w[X.predictor_ids[slice_end - max_window : slice_end], :, id]
            log_lam_y_loop += np.log(
                inverse_link_function(np.sum(basis_at_dts * selected_w) + bias[id])
            )

        np.testing.assert_allclose(log_lam_y_scan, log_lam_y_loop, rtol=1e-10)


@pytest.mark.requires_x64
class TestMCSampling:
    @pytest.mark.parametrize(
        "recording_time",
        [
            IntervalSet(0, 1.0),
            IntervalSet(start=[0.0, 10.0], end=[1.0, 11.0]),
            IntervalSet(start=[0.0, 50.0], end=[49.0, 50.2]),
        ],
        ids=["one_epoch", "two_epochs", "unequal_epochs"],
    )
    def test_grid_points_lie_inside_the_epochs(self, recording_time):
        """Every stratification point belongs to a recording epoch."""
        grid = np.asarray(utils.build_mc_sampling_grid(recording_time, M_samples=10))

        inside = np.any(
            [
                (grid >= s) & (grid <= e)
                for s, e in zip(recording_time.start, recording_time.end)
            ],
            axis=0,
        )

        np.testing.assert_array_equal(inside, True)

    def test_every_epoch_receives_a_sample(self):
        """No recording epoch is left without a stratification point."""
        recording_time = IntervalSet(start=[0.0, 50.0], end=[49.0, 50.2])

        grid = np.asarray(utils.build_mc_sampling_grid(recording_time, M_samples=10))

        counts = [
            int(np.sum((grid >= s) & (grid <= e)))
            for s, e in zip(recording_time.start, recording_time.end)
        ]
        assert min(counts) > 0

    @pytest.mark.parametrize(
        "recording_time",
        [IntervalSet(0, 1.0), IntervalSet(start=[0.0, 10.0], end=[1.0, 11.0])],
        ids=["one_epoch", "two_epochs"],
    )
    def test_jittered_samples_lie_inside_the_epochs(self, recording_time):
        """Jittering a stratification point never moves it out of the recording."""
        M_samples = 10
        grid = utils.build_mc_sampling_grid(recording_time, M_samples)
        X = PredictorsPPGLM(
            times=jnp.linspace(0.0, 11.0, 20), predictor_ids=jnp.zeros(20, dtype=int)
        )

        draw = jax.vmap(
            lambda key: (
                log_likelihood._draw_mc_sample(
                    X, key, M_samples, recording_time.tot_length(), grid
                ).times
            )
        )
        samples = np.asarray(draw(jax.random.split(jax.random.PRNGKey(0), 200))).ravel()

        inside = np.any(
            [
                (samples >= s) & (samples <= e)
                for s, e in zip(recording_time.start, recording_time.end)
            ],
            axis=0,
        )
        np.testing.assert_array_equal(inside, True)

    def test_estimator_integrates_over_the_recording(self):
        """(T / M) * sum_m g(tau_m) is unbiased for the integral of g over the recording."""
        recording_time = IntervalSet(0, 1.0)
        M_samples, n_keys = 10, 20000
        T = recording_time.tot_length()
        grid = utils.build_mc_sampling_grid(recording_time, M_samples)
        X = PredictorsPPGLM(
            times=jnp.linspace(0.0, 1.0, 20), predictor_ids=jnp.zeros(20, dtype=int)
        )

        # g(t) = t, so the estimator must average to the integral T ** 2 / 2;
        # 20000 keys put the standard error at 6.5e-5, well inside the tolerance
        estimate = jax.vmap(
            lambda key: (
                (T / M_samples)
                * jnp.sum(
                    log_likelihood._draw_mc_sample(X, key, M_samples, T, grid).times
                )
            )
        )(jax.random.split(jax.random.PRNGKey(0), n_keys))

        np.testing.assert_allclose(jnp.mean(estimate), T**2 / 2, atol=1e-3)


STATIC_ARGNAMES = (
    "inverse_link_function",
    "M_samples",
    "recording_time",
    "n_basis_funcs",
    "scan_size",
    "max_window",
    "eval_function",
)


@pytest.mark.requires_x64
class TestJit:
    def test_negative_log_likelihood_is_jittable(self):
        """The nll compiles with params, X, y and the key as the only traced arguments."""
        dataset = create_dataset()
        args = (dataset["params"], dataset["X"], dataset["y"], dataset["random_key"])
        kwargs = nll_kwargs(dataset)

        jitted = jax.jit(
            log_likelihood._negative_log_likelihood, static_argnames=STATIC_ARGNAMES
        )

        np.testing.assert_allclose(
            jitted(*args, **kwargs),
            log_likelihood._negative_log_likelihood(*args, **kwargs),
        )

    @pytest.mark.parametrize(
        "traced_argname, expectation",
        [
            ("M_samples", pytest.raises(TypeError, match="Shapes must be")),
            ("n_basis_funcs", pytest.raises(jax.errors.ConcretizationTypeError)),
            ("scan_size", pytest.raises(jax.errors.TracerBoolConversionError)),
            ("max_window", pytest.raises(ValueError, match="Non-hashable static")),
        ],
    )
    def test_shape_arguments_cannot_be_traced(self, traced_argname, expectation):
        """Every argument that sets an array shape must reach the nll as a compile-time constant."""
        dataset = create_dataset()
        args = (dataset["params"], dataset["X"], dataset["y"], dataset["random_key"])
        kwargs = nll_kwargs(dataset)
        # closed over, so that the IntervalSet does not fail the compilation first
        recording_time = kwargs.pop("recording_time")

        def nll(params, X, y, random_key, **kw):
            return log_likelihood._negative_log_likelihood(
                params, X, y, random_key, recording_time=recording_time, **kw
            )

        jitted = jax.jit(
            nll,
            static_argnames=tuple(
                name
                for name in STATIC_ARGNAMES
                if name not in (traced_argname, "recording_time")
            ),
        )

        with expectation:
            jitted(*args, **kwargs)

    def test_jits_once_recording_time_is_an_array(self):
        """The nll compiles once it takes the recording length instead of the IntervalSet."""
        dataset = create_dataset()
        args = (dataset["params"], dataset["X"], dataset["y"], dataset["random_key"])
        kwargs = nll_kwargs(
            dataset, recording_time=dataset["recording_time"].tot_length()
        )

        jitted = jax.jit(
            log_likelihood._negative_log_likelihood,
            static_argnames=tuple(n for n in STATIC_ARGNAMES if n != "recording_time"),
        )

        assert jnp.isfinite(jitted(*args, **kwargs))
