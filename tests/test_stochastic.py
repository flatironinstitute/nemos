"""Tests for the stochastic optimization interface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos import solvers
from nemos.batching import ArrayDataLoader, _PreprocessedDataLoader, is_data_loader
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


class TestGLMStochasticFit:
    """Tests for GLM.stochastic_fit method."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(123)
        X = np.random.randn(200, 5)
        y = np.random.poisson(np.exp(X @ np.random.randn(5) * 0.1))
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

        n_steps_taken = model._solver.get_optim_info(model.solver_state_).num_steps

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.coef_.shape == (5,)
        assert n_steps_taken == (X.shape[0] + 32 - 1) // 32 * 5

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_stochastic_fit_with_init_params(self, simple_data, solver):
        """Test stochastic_fit with provided initial parameters."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32)

        model = nmo.glm.GLM(
            solver_name=solver, solver_kwargs=self._default_solver_kwargs(solver)
        )

        init_coef = jnp.zeros(5)
        init_intercept = jnp.zeros(1)
        init_params = (init_coef, init_intercept)

        model.stochastic_fit(loader, num_epochs=5, init_params=init_params)

        assert model.coef_ is not None

    @pytest.mark.requires_x64
    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_bool_convergence_monitoring(self, simple_data, solver):
        """Test that convergence_criterion=False disables convergence monitoring, True doesn't."""
        X, y = simple_data
        n_epochs = 100
        batch_size = 100
        loader = ArrayDataLoader(X, y, batch_size=batch_size, shuffle=True)

        solver_kwargs = self._default_solver_kwargs(solver, maxiter=10_000)

        # get parameters that are close to the optimum
        model_fitted = nmo.glm.GLM(solver_name="LBFGS", solver_kwargs={"tol": 1e-8})
        model_fitted.fit(X, y)
        final_params = (model_fitted.coef_, model_fitted.intercept_)

        model_stopping = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_stopping.stochastic_fit(
            loader,
            num_epochs=n_epochs,
            init_params=final_params,
            convergence_criterion=True,
        )

        model_no_stopping = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_no_stopping.stochastic_fit(
            loader,
            num_epochs=n_epochs,
            init_params=final_params,
            convergence_criterion=False,
        )

        n_steps_taken_stopping = model_stopping.optim_info_.num_steps
        n_steps_taken_no_stopping = model_no_stopping.optim_info_.num_steps

        assert (
            n_steps_taken_no_stopping
            == (X.shape[0] + batch_size - 1) // batch_size * n_epochs
        )
        assert n_steps_taken_stopping < n_steps_taken_no_stopping

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_callable_convergence_criterion_stops_early(self, simple_data, solver):
        """Test that a callable convergence_criterion stops optimization early."""
        X, y = simple_data
        num_epochs = 5
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        solver_kwargs = self._default_solver_kwargs(solver, maxiter=10_000)

        # Baseline: no convergence monitoring
        model_baseline = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_baseline.stochastic_fit(loader, num_epochs=num_epochs)

        # Early stop: criterion that fires after first epoch
        def _stop(params, prev, state, prev_state, aux, epoch):
            return epoch >= 0

        model_early = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_early.stochastic_fit(
            loader, num_epochs=num_epochs, convergence_criterion=_stop
        )

        assert model_early.optim_info_.num_steps < model_baseline.optim_info_.num_steps

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_batch_callback_stops_early(self, simple_data, solver):
        """Test that a batch_callback returning True stops optimization early."""
        X, y = simple_data
        num_epochs = 5
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        solver_kwargs = self._default_solver_kwargs(solver, maxiter=10_000)

        # Baseline: no callback
        model_baseline = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_baseline.stochastic_fit(loader, num_epochs=num_epochs)

        # Early stop: callback that fires on first batch
        def _stop(params, state, aux, batch_idx, epoch):
            return True

        model_early = nmo.glm.GLM(solver_name=solver, solver_kwargs=solver_kwargs)
        model_early.stochastic_fit(
            loader,
            num_epochs=num_epochs,
            convergence_criterion=False,
            batch_callback=_stop,
        )

        assert model_early.optim_info_.num_steps == 1
        assert model_early.optim_info_.num_steps < model_baseline.optim_info_.num_steps

    @pytest.mark.parametrize("bad_value", [0, 1, "stop", 0.5])
    def test_invalid_convergence_criterion_raises(self, simple_data, bad_value):
        """Test that non-bool, non-callable convergence_criterion raises ValueError."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32)

        model = nmo.glm.GLM(
            solver_name="GradientDescent",
            solver_kwargs=self._default_solver_kwargs("GradientDescent"),
        )

        with pytest.raises(ValueError, match="convergence_criterion"):
            model.stochastic_fit(loader, num_epochs=1, convergence_criterion=bad_value)

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

    @pytest.mark.parametrize("solver_name", _stochastic_solver_names)
    def test_convergence_criterion_accepts_jax_scalar_bool(
        self, simple_loss_and_data, solver_name
    ):
        """Test convergence callback handles JAX scalar booleans like Python bool."""
        loss, loader = simple_loss_and_data
        solver_class = solvers.get_solver(solver_name).implementation

        init_params = jnp.zeros(3)

        solver_py = solver_class(loss, **self._default_solver_kwargs(solver_class))
        _, state_py, _ = solver_py.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            convergence_criterion=lambda *args: True,
        )
        steps_py = solver_py.get_optim_info(state_py).num_steps

        solver_jax = solver_class(loss, **self._default_solver_kwargs(solver_class))
        _, state_jax, _ = solver_jax.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            convergence_criterion=lambda *args: jnp.array(True),
        )
        steps_jax = solver_jax.get_optim_info(state_jax).num_steps

        assert steps_jax == steps_py

    @pytest.mark.parametrize("solver_name", _stochastic_solver_names)
    def test_batch_callback_accepts_jax_scalar_bool(
        self, simple_loss_and_data, solver_name
    ):
        """Test batch callback handles JAX scalar booleans like Python bool."""
        loss, loader = simple_loss_and_data
        solver_class = solvers.get_solver(solver_name).implementation

        init_params = jnp.zeros(3)

        solver_py = solver_class(loss, **self._default_solver_kwargs(solver_class))
        _, state_py, _ = solver_py.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            batch_callback=lambda *args: True,
        )
        steps_py = solver_py.get_optim_info(state_py).num_steps

        solver_jax = solver_class(loss, **self._default_solver_kwargs(solver_class))
        _, state_jax, _ = solver_jax.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            batch_callback=lambda *args: jnp.array(True),
        )
        steps_jax = solver_jax.get_optim_info(state_jax).num_steps

        assert steps_jax == steps_py

    @pytest.mark.parametrize("solver_name", _stochastic_solver_names)
    @pytest.mark.parametrize(
        "bad_flag", [1, 1.0, np.array(1.0), jnp.array(1.0), np.bool_(True).astype(int)]
    )
    def test_convergence_criterion_non_bool_scalar_raises(
        self, simple_loss_and_data, bad_flag, solver_name
    ):
        """Test convergence callback rejects non-boolean scalar return values."""
        loss, loader = simple_loss_and_data
        solver_class = solvers.get_solver(solver_name).implementation

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))

        with pytest.raises(TypeError, match="scalar boolean"):
            solver.stochastic_run(
                jnp.zeros(3),
                loader,
                num_epochs=2,
                convergence_criterion=lambda *args: bad_flag,
            )

    @pytest.mark.parametrize("solver_name", _stochastic_solver_names)
    @pytest.mark.parametrize(
        "bad_flag", [1, 1.0, np.array(1.0), jnp.array(1.0), np.bool_(True).astype(int)]
    )
    def test_batch_callback_non_bool_scalar_raises(
        self, simple_loss_and_data, bad_flag, solver_name
    ):
        """Test batch callback rejects non-boolean scalar return values."""
        loss, loader = simple_loss_and_data
        solver_class = solvers.get_solver(solver_name).implementation

        solver = solver_class(loss, **self._default_solver_kwargs(solver_class))

        with pytest.raises(TypeError, match="scalar boolean"):
            solver.stochastic_run(
                jnp.zeros(3),
                loader,
                num_epochs=2,
                batch_callback=lambda *args: bad_flag,
            )

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

        n_steps = solver.get_optim_info(state).num_steps
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
