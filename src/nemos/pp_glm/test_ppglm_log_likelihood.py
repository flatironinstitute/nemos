import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pynapple import IntervalSet

from nemos.pp_glm import log_likelihood, utils
from nemos.pp_glm.validation import to_pp_glm_params, to_pp_glm_params_with_key
from nemos.basis import RaisedCosineLogEval

jax.config.update("jax_enable_x64", True)

def create_basis(n_basis_funcs=4, history_window=0.01):
    """Use nemos RC Eval basis and return the evaluate method"""
    basis = RaisedCosineLogEval(n_basis_funcs, bounds=(0, history_window), fill_value=0)
    return lambda pts: basis.evaluate(pts)

def create_params(n_neurons, n_basis_funcs, seed=0, all_to_one=False):
    """Use PP-GLM params structures"""
    if all_to_one:
        params = to_pp_glm_params((
            jnp.ones(n_neurons * n_basis_funcs),
            jnp.atleast_1d(jnp.zeros(1)),
        ))
    else:
        params = to_pp_glm_params((
            jnp.ones((n_neurons * n_basis_funcs, n_neurons)),
            jnp.atleast_1d(jnp.zeros(n_neurons)),
        ))
    random_key = jax.random.PRNGKey(seed).astype(jnp.float64)
    return to_pp_glm_params_with_key(params, random_key)

def create_dataset(n_neurons=5, n_spikes=400, sim_time=5.0, M_samples=100,
                 n_basis_funcs=4, history_window=0.01, scan_size=3, seed=0, all_to_one=False):
    """Create a fake dataset (without running an actual simulation) for fitting an all-to-all or
    all-to-one coupled model. Returns preprocessed inputs, model hyperparams and arbitrary PP-GLM params"""
    recording_time = IntervalSet(0, sim_time)
    M_grid = utils.build_mc_sampling_grid(recording_time, M_samples)
    eval_function = create_basis(n_basis_funcs, history_window)

    np.random.seed(seed)
    spike_times = np.sort(np.random.uniform(0, sim_time, n_spikes))
    spike_ids = np.random.choice(np.arange(n_neurons), n_spikes)

    X = jnp.vstack((spike_times, spike_ids))
    y = jnp.vstack((X, jnp.arange(spike_times.size)))

    if all_to_one:
        n_target = 0
        y = y[:, y[1] == n_target]

    max_window = int(utils.compute_max_window_size(
        jnp.array([-history_window, 0]), X[0], X[0]
    ))
    X, y = utils.adjust_indices_and_spike_times(X, history_window, max_window, y)

    params_with_key = create_params(n_neurons, n_basis_funcs, seed, all_to_one)

    return dict(
        params_with_key=params_with_key,
        X=X, y=y,
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

def create_dataset_single_spike(spike_time, history_window=0.01, n_basis_funcs=4, M_samples=100, seed=0):
    """
    Create the minimal dataset for a single-neuron, single-spike scenario.
    """
    n_neurons = 1
    sim_time = spike_time + 1.0
    recording_time = IntervalSet(0, sim_time)
    M_grid = utils.build_mc_sampling_grid(recording_time, M_samples)
    eval_function = create_basis(n_basis_funcs, history_window)

    spike_times = jnp.array([spike_time])
    spike_ids = jnp.array([0])

    X = jnp.vstack((spike_times, spike_ids)) # shape (2, 1)
    y = jnp.vstack((X, jnp.arange(1)))  # shape (3, 1)

    max_window = int(utils.compute_max_window_size(
        jnp.array([-history_window, 0.0]), X[0], X[0]
    ))
    X, y = utils.adjust_indices_and_spike_times(X, history_window, max_window, y)

    params = to_pp_glm_params((
        jnp.ones((n_neurons * n_basis_funcs, n_neurons)),
        jnp.atleast_1d(0.),
    ))
    random_key = jax.random.PRNGKey(seed).astype(jnp.float64)
    params_with_key = to_pp_glm_params_with_key(params, random_key)

    return dict(
        params_with_key=params_with_key,
        X=X, y=y,
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
    def test_reshape_coef_for_scan(self):
        """Test that reshaping works correctly for 1d and 2d scenarios"""
        # 1d (single postsynaptic neuron)
        n_predictors, n_bases = 5, 4
        w = jnp.ones(n_predictors * n_bases)
        out = utils.reshape_coef_for_scan(w, n_bases)

        assert out.shape == (n_predictors, n_bases, 1)

        # 2d (population)
        n_predictors, n_bases, n_target = 5, 4, 3
        w = jnp.ones((n_predictors * n_bases, n_target))
        out = utils.reshape_coef_for_scan(w, n_bases)

        assert out.shape == (n_predictors, n_bases, n_target)

    def test_reshape_input_for_scan(self):
        """Test that reshaping works properly and that padding length and value are correct"""
        # when divisible, padding length is 0
        times = jnp.ones((3, 8))
        out, pad_val, pad_len = utils.reshape_input_for_scan(times, scan_size=2)
        assert out.shape == (4, 2, 3) # (n_scans, scan_size, n_channels)
        assert pad_len == 0

        # when not divisible, padding fills to next multiple
        times = jnp.ones((3, 9))
        out, pad_val, pad_len = utils.reshape_input_for_scan(times, scan_size=2)
        assert out.shape == (5, 2, 3) # (n_scans, scan_size, n_channels)
        assert pad_len == 1

        #test that padding is the last spike and that it's all the same
        times = jnp.stack([jnp.arange(5, dtype=jnp.float64),
                           jnp.arange(5, dtype=jnp.float64)])
        out, pad_val, pad_len = utils.reshape_input_for_scan(times, scan_size=3)
        padding = out[:, -pad_len:]

        assert np.all(padding == padding[:, [0]])
        np.testing.assert_array_equal(pad_val, times[:, -1])

    def test_build_mc_sampling_grid(self):
        """test that the grid is built correctly with multiple epochs"""
        recording_time = IntervalSet(start=[0.0, 6.0], end=[4.0, 10.0])
        grid = utils.build_mc_sampling_grid(recording_time, M_samples=100)

        # assert grid size is exactly M_samples
        assert grid.shape[0] == 100

        # assert that all grid points are within epochs
        in_epoch = np.any(
            [(grid >= s) & (grid <= e) for s, e in zip(recording_time.start, recording_time.end)], axis=0
        )

        assert np.all(in_epoch)

        # test that edge case when the number of samples is less than the number of epochs
        # raises an error
        starts = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        ends = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        recording_time = IntervalSet(start=starts, end=ends)

        with pytest.raises(ValueError):
            utils.build_mc_sampling_grid(recording_time, M_samples=3)

    def test_adjust_indices_and_spike_times(self):
        """Test that shapes and indices are shifted correctly and that padding
        is outside the history window"""

        # returns already preprocessed X and y
        dataset = create_dataset()

        # test X shape increase by history window
        n_spk_original = dataset["y"].shape[1] # unchanged
        assert dataset["X"].shape[1] == n_spk_original + dataset["max_window"]

        # test y index is shifted by max_window
        assert dataset["y"][-1,0] == dataset["max_window"]

        # test padding values are out of bound and basis evals to 0
        bound = dataset["recording_time"].start[0] - dataset["history_window"]
        padding = dataset["X"][:, :dataset["max_window"]]
        assert np.all(padding[0] < bound)

        first_spike = dataset["y"][:, 0]
        dts = first_spike[0] - padding[0]
        basis_at_dts = dataset["eval_function"](dts)

        np.testing.assert_array_equal(basis_at_dts, 0)

    def test_compute_max_window_size(self):
        """Test max window is selected correctly with any non-empty dataset"""

        # test the edge case with a single spike dataset
        dataset = create_dataset_single_spike(1.0)

        max_window = int(utils.compute_max_window_size(
            jnp.array([-dataset["history_window"], 0]), dataset["X"][0], dataset["X"][0]
        ))

        assert max_window == 0

        # one reference spike, multiple events
        ref = jnp.array([1.0])
        events = jnp.array([0.1, 0.5, 0.990, 0.993, 0.995]) # the last 3 fall within 0.01 s
        max_window = int(utils.compute_max_window_size(
            jnp.array([-dataset["history_window"], 0]), ref, events
        ))

        assert max_window == 3

        # multiple spikes, multiple events
        ref = jnp.array([1.0, 3.0])
        events = jnp.array([0.990, 0.993, 0.995,            # 3 events within ref 1
                            2.990, 2.993, 2.995, 2.999])    # 4 events within ref 2

        max_window = int(utils.compute_max_window_size(
            jnp.array([-dataset["history_window"], 0]), ref, events
        ))

        assert max_window == 4


class TestLogLikelihood:
    def test_single_spike_dataset(self):
        """
        Test the edge case with a single spike in the dataset.

        The negative log-likelihood should still be valid but only include the bias contribution.
        """
        dataset = create_dataset_single_spike(5.0)

        params_with_key = dataset["params_with_key"]
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
            X,
            y,
            params_with_key.params,
            params_with_key.random_key.astype(jnp.uint32),
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
        bias_contrib = params_with_key.params.intercept
        log_lam_y = log_likelihood._log_likelihood_scan(
            X,
            y,
            params_with_key.params,
            log_likelihood._compute_lam_tilde_single,
            inverse_link_function,
            n_basis_funcs,
            max_window,
            scan_size,
            eval_function,
            log=True
        )

        np.testing.assert_almost_equal(log_lam_y, bias_contrib)

    def _test_for_loop_ppglm_ll(self, all_to_one):
        """
        Test the model nll computation against a numpy loop implementation.

        Validates that log-likelihood computed using vmap over multiple lax.scan matches a naive
        loop implementation. Also validates that the padding added to the vmap input to maintain
        fixed shape is subtracted correctly.
        """
        dataset = create_dataset(all_to_one=all_to_one)

        params_with_key = dataset["params_with_key"]
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

        ## first ll term
        # jax lax scan + vmap for a single postsynaptic neuron
        log_lam_y_scan = log_likelihood._log_likelihood_scan(
            X,
            y,
            params_with_key.params,
            log_likelihood._compute_lam_tilde_single,
            inverse_link_function,
            n_basis_funcs,
            max_window,
            scan_size,
            eval_function,
            log=True
        )

        # numpy loop
        weights, bias = params_with_key.params.coef, params_with_key.params.intercept
        if all_to_one:
            weights = weights.reshape(-1, n_basis_funcs, 1)
        else:
            weights = weights.reshape(weights.shape[1], -1, weights.shape[1])

        log_lam_y_loop = 0
        for sp in range(y.shape[1]):
            i = y[:, sp]
            slice_start = i[-1].astype(int) - max_window
            slice_end = i[-1].astype(int)
            spk_in_window = X[:, slice_start:slice_end]
            dts = i[0] - spk_in_window[0]
            basis_at_dts = eval_function(dts)
            selected_w = weights[spk_in_window[1].astype(int), :, i[1].astype(int)]
            lam_tilde = np.sum(basis_at_dts * selected_w) + bias[i[1].astype(int)]
            log_lam_y_loop += np.log(inverse_link_function(lam_tilde))


        np.testing.assert_almost_equal(log_lam_y_scan, log_lam_y_loop)

        ## second ll term
        # uses the same random key as _negative_log_likelihood below
        mc_samples = log_likelihood._draw_mc_sample(
            X,
            params_with_key.random_key.astype(jnp.uint32),
            M_samples,
            recording_time,
            M_grid
        )

        # jax lax scan + vmap for all postsynaptic neurons
        mc_est_scan = log_likelihood._log_likelihood_scan(
            X,
            mc_samples,
            params_with_key.params,
            log_likelihood._compute_lam_tilde_all,
            inverse_link_function,
            n_basis_funcs,
            max_window,
            scan_size,
            eval_function,
            log=False
        )

        # numpy loop
        mc_est_loop = 0
        for sp in range(M_samples):
            i = mc_samples[:, sp]
            slice_start = i[-1].astype(int) - max_window
            slice_end = i[-1].astype(int)
            spk_in_window = X[:, slice_start:slice_end]
            dts = i[0] - spk_in_window[0]
            basis_at_dts = eval_function(dts)
            selected_w = weights[spk_in_window[1].astype(int)]
            lam_tilde = np.sum(basis_at_dts[:,:,None] * selected_w, axis=(0,1)) + bias
            mc_est_loop += inverse_link_function(lam_tilde).sum()

        np.testing.assert_almost_equal(mc_est_scan, mc_est_loop)

        ## full nll computation
        loss_scan = log_likelihood._negative_log_likelihood(
            X,
            y,
            params_with_key.params,
            params_with_key.random_key.astype(jnp.uint32),
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
        loss_loop = ((recording_time.tot_length() / M_samples) * mc_est_loop) - log_lam_y_loop

        np.testing.assert_almost_equal(loss_loop, loss_scan)

    def test_for_loop_single_neuron_ppglm_ll(self):
        """
        Test the model nll computation against a numpy loop implementation for a single neuron dataset.
        """
        self._test_for_loop_ppglm_ll(all_to_one=True)

    def test_for_loop_population_ppglm_ll(self):
        """
        Test the model nll computation against a numpy loop implementation for a population dataset.
        """
        self._test_for_loop_ppglm_ll(all_to_one=False)