"""Tests for the stochastic optimization interface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos import solvers
from nemos.batching import ArrayDataLoader
from nemos.callbacks import (
    Callback,
    CallbackList,
    SolverConvergenceCallback,
    TrainingContext,
    _normalize_callbacks,
)
from nemos.regularizer import UnRegularized

_stochastic_solver_names = [
    "GradientDescent[optimistix]",
    "GradientDescent[optax+optimistix]",
    "ProximalGradient[optimistix]",
    "SVRG[nemos]",
    "ProxSVRG[nemos]",
]
_non_stochastic_solver_names = [
    "LBFGS",
    "BFGS",
    "NonlinearCG",
]

# Build list of stochastic solvers, conditionally including JAXopt
_stochastic_solver_classes = [
    solvers.OptimistixNAG,
    solvers.OptimistixFISTA,
    solvers.OptimistixOptaxGradientDescent,
    solvers.WrappedSVRG,
    solvers.WrappedProxSVRG,
]

# Build list of non-stochastic solvers for testing unsupported stochastic_run
_non_stochastic_solver_classes = [
    solvers.OptimistixBFGS,
    solvers.OptimistixOptaxLBFGS,
    solvers.OptimistixNonlinearCG,
]

if solvers.JAXOPT_AVAILABLE:
    _stochastic_solver_names.extend(
        [
            "GradientDescent[jaxopt]",
            "ProximalGradient[jaxopt]",
        ]
    )
    _non_stochastic_solver_names.extend(
        [
            "LBFGS[jaxopt]",
            "BFGS[jaxopt]",
            "NonlinearCG[jaxopt]",
        ]
    )
    _stochastic_solver_classes.extend(
        [
            solvers.JaxoptGradientDescent,
            solvers.JaxoptProximalGradient,
        ]
    )
    _non_stochastic_solver_classes.extend(
        [
            solvers.JaxoptBFGS,
            solvers.JaxoptLBFGS,
            solvers.JaxoptNonlinearCG,
        ]
    )


class TestSolverStochasticSupport:
    """Tests for solver stochastic support flags and utilities."""

    @pytest.mark.parametrize(
        "solver_name", _stochastic_solver_names + _non_stochastic_solver_names
    )
    def test_supports_stochastic(self, solver_name):
        """Test the right solvers have stochastic support."""
        expectation = solver_name in _stochastic_solver_names
        assert solvers.supports_stochastic(solver_name) == expectation

    def test_list_stochastic_solvers(self):
        """Test list_stochastic_solvers returns expected solvers."""
        assert set([s.full_name for s in solvers.list_stochastic_solvers()]) == set(
            _stochastic_solver_names
        )

    def test_unknown_solver_raises(self):
        """Test unknown solver name raises ValueError."""
        with pytest.raises(ValueError, match="No solver registered"):
            solvers.supports_stochastic("NonExistentSolver")


class TestCallbackSystem:
    """Tests for the callback infrastructure in stochastic optimization."""

    def test_normalize_none(self):
        """None produces a no-op Callback."""
        cb = _normalize_callbacks(None)
        assert isinstance(cb, Callback)

    @pytest.mark.parametrize(
        "cb_class", [Callback, CallbackList, SolverConvergenceCallback]
    )
    def test_normalize_callback_passthrough(self, cb_class):
        """A single Callback is returned as-is."""
        cb = cb_class()
        assert _normalize_callbacks(cb) is cb

    def test_normalize_list(self):
        """A list of Callbacks is wrapped in a CallbackList."""
        cl = _normalize_callbacks([Callback(), SolverConvergenceCallback()])
        assert isinstance(cl, CallbackList)
        assert len(cl._callbacks) == 2

    def test_normalize_invalid_raises(self):
        """Non-callback types raise TypeError."""
        with pytest.raises(TypeError, match="callbacks must be"):
            _normalize_callbacks(123)

    def test_training_context_stop(self):
        """request_stop sets should_stop and stop_reason."""
        ctx = TrainingContext(solver=None)
        assert not ctx.should_stop
        ctx.request_stop("test reason")
        assert ctx.should_stop
        assert ctx.stop_reason == "test reason"

    def test_callback_hooks_are_noop(self):
        """Base Callback hooks don't raise."""
        cb = Callback()
        ctx = TrainingContext(solver=None)
        cb.on_train_begin(ctx)
        cb.on_train_end(ctx)
        cb.on_epoch_begin(ctx)
        cb.on_epoch_end(ctx)
        cb.on_batch_begin(ctx)
        cb.on_batch_end(ctx)

    def test_callback_list_is_callback(self):
        """CallbackList is a Callback subclass."""
        cl = CallbackList([Callback()])
        assert isinstance(cl, Callback)

    def test_callback_list_dispatches(self):
        """CallbackList dispatches to all callbacks."""
        calls = []

        class RecordingCallback(Callback):
            def __init__(self, name):
                self.name = name

            def on_epoch_end(self, ctx):
                calls.append(self.name)

        cl = CallbackList([RecordingCallback("a"), RecordingCallback("b")])
        ctx = TrainingContext(solver=None)
        cl.on_epoch_end(ctx)
        assert calls == ["a", "b"]


class TestGLMStochasticFit:
    """Tests for GLM.stochastic_fit method."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(123)
        X = np.random.randn(200, 5)
        y = np.random.poisson(np.exp(X @ np.arange(5) * 0.1))
        return X, y

    def _default_solver_kwargs(self, solver_name, **overrides):
        """Build solver kwargs with stochastic-safe defaults."""
        solver_kwargs = {"stepsize": 0.001, "maxiter": 100, **overrides}
        solver_class = solvers.get_solver(solver_name).implementation
        if "acceleration" in solver_class.get_accepted_arguments():
            solver_kwargs.setdefault("acceleration", False)
        return solver_kwargs

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_stochastic_fit(self, simple_data, solver):
        """Test basic stochastic_fit functionality."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        model = nmo.glm.GLM(
            solver_name=solver, solver_kwargs=self._default_solver_kwargs(solver)
        )
        model.stochastic_fit(loader, num_epochs=5)

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.coef_.shape == (5,)

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_stochastic_fit_with_init_params(self, simple_data, solver):
        """Test stochastic_fit with provided initial parameters."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32)

        # tiny stepsize, so it doesn't move far from the initial params
        solver_kwargs = self._default_solver_kwargs(solver)
        solver_kwargs["stepsize"] = 1e-16

        model = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)

        # start from wrong initial params that's far from true params
        init_coef = jnp.arange(5)[::-1]
        init_intercept = jnp.zeros(1)
        init_params = (init_coef, init_intercept)

        model.stochastic_fit(loader, num_epochs=1, init_params=init_params)

        np.testing.assert_allclose(model.coef_, init_coef, atol=1e-2)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_default_convergence_monitoring(self, simple_data, solver):
        """Test that default callbacks (SolverConvergenceCallback) enables convergence monitoring, None disables."""
        X, y = simple_data
        n_epochs = 100
        batch_size = 100
        loader = ArrayDataLoader(X, y, batch_size=batch_size, shuffle=True)

        solver_kwargs = self._default_solver_kwargs(solver, maxiter=10_000)

        # get parameters that are close to the optimum
        model_fitted = nmo.glm.GLM(solver_name="LBFGS", solver_kwargs={"tol": 1e-8})
        model_fitted.fit(X, y)
        final_params = (model_fitted.coef_, model_fitted.intercept_)

        # Default: SolverConvergenceCallback (should stop early)
        model_stopping = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_stopping.stochastic_fit(
            loader,
            num_epochs=n_epochs,
            init_params=final_params,
        )

        # No callbacks: runs for all epochs
        model_no_stopping = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_no_stopping.stochastic_fit(
            loader,
            num_epochs=n_epochs,
            init_params=final_params,
            callbacks=None,
        )

        n_steps_taken_stopping = model_stopping.solver_state_.stats.num_steps
        n_steps_taken_no_stopping = model_no_stopping.solver_state_.stats.num_steps

        batches_per_epoch = int(np.ceil(X.shape[0] / batch_size))
        assert n_steps_taken_no_stopping == batches_per_epoch * n_epochs
        assert n_steps_taken_stopping < n_steps_taken_no_stopping

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_custom_epoch_callback_stops_early(self, simple_data, solver):
        """Test that a custom epoch callback stops optimization early."""
        X, y = simple_data
        num_epochs = 5
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        solver_kwargs = self._default_solver_kwargs(solver, maxiter=10_000)

        # Baseline: no callbacks
        model_baseline = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_baseline.stochastic_fit(loader, num_epochs=num_epochs, callbacks=None)

        # Early stop: callback requests stop after first epoch
        class StopAfterFirstEpoch(Callback):
            def on_epoch_end(self, ctx):
                if ctx.epoch >= 0:
                    ctx.request_stop("test stop")

        model_early = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_early.stochastic_fit(
            loader, num_epochs=num_epochs, callbacks=StopAfterFirstEpoch()
        )

        batches_per_epoch = int(np.ceil(loader.n_samples / loader.batch_size))
        assert model_early.solver_state_.stats.num_steps == batches_per_epoch
        assert (
            model_early.solver_state_.stats.num_steps
            < model_baseline.solver_state_.stats.num_steps
        )

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_batch_callback_stops_early(self, simple_data, solver):
        """Test that a batch callback requesting stop halts optimization early."""
        X, y = simple_data
        num_epochs = 5
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        solver_kwargs = self._default_solver_kwargs(solver, maxiter=10_000)

        # Baseline: no callback
        model_baseline = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_baseline.stochastic_fit(loader, num_epochs=num_epochs, callbacks=None)

        # Early stop: callback requests stops on first batch
        class StopOnFirstBatch(Callback):
            def on_batch_end(self, ctx):
                ctx.request_stop("first batch")

        model_early = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_early.stochastic_fit(
            loader,
            num_epochs=num_epochs,
            callbacks=StopOnFirstBatch(),
        )

        assert model_early.solver_state_.stats.num_steps == 1
        assert (
            model_early.solver_state_.stats.num_steps
            < model_baseline.solver_state_.stats.num_steps
        )

    @pytest.mark.parametrize("solver", ["LBFGS", "BFGS", "NonlinearCG"])
    def test_unsupported_solver_raises(self, simple_data, solver):
        """Test that unsupported solver raises error."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32)

        model = nmo.glm.GLM(
            solver_name=solver,
            solver_kwargs={"maxiter": 100},
        )

        with pytest.raises(ValueError, match="does not support stochastic"):
            model.stochastic_fit(loader)

    # TODO: Update this if they are implemented.
    def test_scale_and_dof_are_none(self, simple_data):
        """Test that scale_ and dof_resid_ are None after stochastic_fit."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32)

        model = nmo.glm.GLM(
            solver_name="GradientDescent",
            solver_kwargs={"stepsize": 0.001, "maxiter": 100, "acceleration": False},
        )
        model.stochastic_fit(loader, num_epochs=1)

        assert model.scale_ is None
        assert model.dof_resid_ is None


class TestPopulationGLMStochasticFit:
    """Tests for PopulationGLM.stochastic_fit method."""

    @pytest.fixture
    def population_data(self):
        """Generate simple population test data."""
        np.random.seed(123)
        X = np.random.randn(200, 5)
        y = np.random.poisson(np.exp(X @ np.random.randn(5, 3) * 0.1))
        return X, y

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_basic_population_stochastic_fit(self, population_data, solver):
        """Test basic stochastic_fit for PopulationGLM."""
        X, y = population_data
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        solver_kwargs = {"stepsize": 0.001, "maxiter": 100}
        solver_class = solvers.get_solver(solver).implementation
        if "acceleration" in solver_class.get_accepted_arguments():
            solver_kwargs["acceleration"] = False

        model = nmo.glm.PopulationGLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model.stochastic_fit(loader, num_epochs=5)

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.coef_.shape == (5, 3)
        assert model.intercept_.shape == (3,)


class TestSolverStochasticRun:
    """Tests for solver stochastic_run method directly."""

    def _default_solver_kwargs(self, solver_class):
        solver_kwargs = {
            "stepsize": 0.01,
            "regularizer": UnRegularized(),
            "regularizer_strength": None,
            "has_aux": False,
        }
        if "acceleration" in solver_class.get_accepted_arguments():
            solver_kwargs["acceleration"] = False

        return solver_kwargs

    @pytest.fixture
    def simple_loss_and_data(self):
        """Create a simple loss function and data loader."""
        np.random.seed(123)
        X = np.random.randn(1000, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(1000) * 0.1

        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        def loss(params, X, y):
            pred = X @ params
            return jnp.mean((pred - y) ** 2)

        return loss, loader

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_stochastic_run(self, simple_loss_and_data, solver_class):
        """Test solvers' stochastic_run."""
        loss, loader = simple_loss_and_data

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))

        init_params = jnp.zeros(3)
        params, state, aux = solver.stochastic_run(init_params, loader, num_epochs=10)

        assert params.shape == (3,)
        # Should have learned something close to [1, 2, 3]
        np.testing.assert_array_almost_equal(params, [1.0, 2.0, 3.0], decimal=0)

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_epoch_callback_stops_early(self, simple_loss_and_data, solver_class):
        """Test that an epoch callback requesting stop halts optimization."""
        loss, loader = simple_loss_and_data

        class StopAfterEpoch(Callback):
            def on_epoch_end(self, ctx):
                ctx.request_stop("test")

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))
        init_params = jnp.zeros(3)
        _, state, _ = solver.stochastic_run(
            init_params, loader, num_epochs=5, callback=StopAfterEpoch()
        )

        # Should have stopped after 1 epoch
        batches_per_epoch = int(np.ceil(loader.n_samples / loader.batch_size))
        assert state.stats.num_steps == batches_per_epoch

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_batch_callback_stops_early(self, simple_loss_and_data, solver_class):
        """Test that a batch callback requesting stop halts optimization."""
        loss, loader = simple_loss_and_data

        class StopOnFirstBatch(Callback):
            def on_batch_end(self, ctx):
                ctx.request_stop("first batch")

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))
        init_params = jnp.zeros(3)
        _, state, _ = solver.stochastic_run(
            init_params, loader, num_epochs=5, callback=StopOnFirstBatch()
        )
        assert state.stats.num_steps == 1

    @pytest.mark.parametrize("solver_class", _non_stochastic_solver_classes)
    def test_unsupported_solver_raises(self, simple_loss_and_data, solver_class):
        """Test that unsupported solver raises NotImplementedError."""
        loss, loader = simple_loss_and_data

        solver_kwargs = self._default_solver_kwargs(solver_class)
        solver_kwargs.pop("stepsize", None)
        solver_kwargs.pop("acceleration", None)

        solver = solver_class(loss, **solver_kwargs)

        init_params = jnp.zeros(3)
        with pytest.raises(NotImplementedError, match="does not support stochastic"):
            solver.stochastic_run(init_params, loader, num_epochs=1)

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_invalid_num_epochs_raises(self, simple_loss_and_data, solver_class):
        """Test that num_epochs < 1 raises ValueError."""
        loss, loader = simple_loss_and_data

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))

        init_params = jnp.zeros(3)
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            solver.stochastic_run(init_params, loader, num_epochs=0)

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_maxiter_does_not_stop_stochastic(self, simple_loss_and_data, solver_class):
        """Test that maxiter does not limit stochastic optimization; only num_epochs does."""
        loss, loader = simple_loss_and_data

        solver_kwargs = self._default_solver_kwargs(solver_class)
        # Set maxiter=1, which would stop after 1 step if it were respected
        solver = solver_class(loss, maxiter=1, **solver_kwargs)

        num_epochs = 3
        init_params = jnp.zeros(3)
        _, state, _ = solver.stochastic_run(init_params, loader, num_epochs=num_epochs)

        n_steps = state.stats.num_steps
        batches_per_epoch = int(np.ceil(loader.n_samples / loader.batch_size))
        expected_steps = batches_per_epoch * num_epochs
        assert n_steps == expected_steps

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_acceleration_not_allowed(self, simple_loss_and_data, solver_class):
        """Test that solvers that have acceleration argument have those turned off for stochastic optimization."""
        loss, loader = simple_loss_and_data

        solver_kwargs = self._default_solver_kwargs(solver_class)
        if "acceleration" in solver_class.get_accepted_arguments():
            solver_kwargs["acceleration"] = True
        else:
            pytest.skip("Solver doesn't have acceleration argument.")

        solver = solver_class(loss, **solver_kwargs)

        init_params = jnp.zeros(3)
        with pytest.raises(ValueError, match="Turn off acceleration"):
            solver.stochastic_run(init_params, loader, num_epochs=10)

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_linesearch_not_allowed(self, simple_loss_and_data, solver_class):
        """Test that linesearch is not allowed when using stochastic optimization."""
        if "svrg" in solver_class.__name__.lower():
            pytest.skip("SVRG doesn't have linesearch.")

        loss, loader = simple_loss_and_data

        # not giving stepsize uses linesearch in solvers that have it
        solver_kwargs = self._default_solver_kwargs(solver_class)
        solver_kwargs.pop("stepsize", None)

        solver = solver_class(loss, **solver_kwargs)

        init_params = jnp.zeros(3)
        with pytest.raises(ValueError, match="Turn off linesearch"):
            solver.stochastic_run(init_params, loader, num_epochs=10)

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_multiple_callbacks(self, simple_loss_and_data, solver_class):
        """Test that multiple callbacks are all invoked."""
        loss, loader = simple_loss_and_data

        class CounterCallback(Callback):
            def __init__(self, multiplier: float):
                self.multiplier = multiplier
                self.epoch_counts = []

            def on_epoch_end(self, ctx):
                self.epoch_counts.append(self.multiplier * ctx.epoch)

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))
        init_params = jnp.zeros(3)
        num_epochs = 3
        cb1 = CounterCallback(1)
        cb2 = CounterCallback(2)
        solver.stochastic_run(
            init_params,
            loader,
            num_epochs=num_epochs,
            callback=CallbackList([cb1, cb2]),
        )
        assert cb1.epoch_counts == [0, 1, 2]
        assert cb2.epoch_counts == [0, 2, 4]

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_no_callback_runs_all_epochs(self, simple_loss_and_data, solver_class):
        """Test that default no-op callback runs all epochs without any monitoring."""
        loss, loader = simple_loss_and_data

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))
        init_params = jnp.zeros(3)
        num_epochs = 3

        _, state, _ = solver.stochastic_run(init_params, loader, num_epochs=num_epochs)

        n_steps = state.stats.num_steps
        batches_per_epoch = int(np.ceil(loader.n_samples / loader.batch_size))
        expected_steps = batches_per_epoch * num_epochs
        assert n_steps == expected_steps


@pytest.mark.requires_x64
def test_svrg_compute_full_gradient_streaming():
    np.random.seed(123)

    # setting N // batch_size = 0 to have a nice scaling for sum-style losses
    N = 1000
    batch_size = 100

    X = np.random.randn(N, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(N) * 0.1
    loader = ArrayDataLoader(X, y, batch_size=batch_size, shuffle=True)
    params = np.random.randn(3)

    def loss_mean(params, X, y):
        pred = X @ params
        return jnp.mean((pred - y) ** 2)

    def loss_sum(params, X, y):
        pred = X @ params
        return jnp.sum((pred - y) ** 2)

    full_grad_mean = jax.grad(loss_mean)(params, X, y)
    streaming_grad_mean = solvers.SVRG(loss_mean)._compute_full_gradient_streaming(
        params, loader.__iter__
    )
    np.testing.assert_array_almost_equal(full_grad_mean, streaming_grad_mean)

    # for sum-style losses the gradients for each batch are averaged in the streaming method
    # but added in the jax.grad
    full_grad_sum = jax.grad(loss_sum)(params, X, y)
    streaming_grad_sum = solvers.SVRG(loss_sum)._compute_full_gradient_streaming(
        params, loader.__iter__
    )
    n_batches = N / batch_size
    np.testing.assert_array_almost_equal(full_grad_sum, n_batches * streaming_grad_sum)
