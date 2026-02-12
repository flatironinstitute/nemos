"""Tests for the stochastic optimization interface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos import solvers
from nemos.batching import ArrayDataLoader, _PreprocessedDataLoader, is_data_loader

_stochastic_solver_names = [
    "GradientDescent",
    "ProximalGradient",
    "SVRG",
    "ProxSVRG",
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


class TestArrayDataLoader:
    """Tests for ArrayDataLoader class."""

    def test_basic_creation(self):
        """Test basic DataLoader creation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        assert loader.n_samples == 100

    def test_basic_creation_variadic(self):
        """Test basic DataLoader creation with >2 data arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = ArrayDataLoader(X, y, z, batch_size=32)

        assert loader.n_samples == 100

    def test_sample_batch(self):
        """Test sample_batch returns correct shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        X_batch, y_batch = loader.sample_batch()
        assert X_batch.shape == (32, 5)
        assert y_batch.shape == (32,)

    def test_sample_batch_variadic(self):
        """Test sample_batch returns correct shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = ArrayDataLoader(X, y, z, batch_size=32)

        batch_data = loader.sample_batch()
        assert isinstance(batch_data, tuple)
        assert len(batch_data) == 3
        assert batch_data[0].shape == (32, 5)
        assert batch_data[1].shape == (32,)
        assert batch_data[2].shape == (32, 3)

    # TODO: Not sure if this should be the intended behavior.
    def test_sample_batch_deterministic(self):
        """Test sample_batch is deterministic."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        X_batch1, y_batch1 = loader.sample_batch()
        X_batch2, y_batch2 = loader.sample_batch()

        np.testing.assert_array_equal(X_batch1, X_batch2)
        np.testing.assert_array_equal(y_batch1, y_batch2)

    def test_iteration_yields_all_data(self):
        """Test that iteration covers all samples."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=False)

        all_X = []
        all_y = []
        for X_batch, y_batch in loader:
            all_X.append(X_batch)
            all_y.append(y_batch)

        X_concat = jnp.concatenate(all_X)
        y_concat = jnp.concatenate(all_y)

        assert X_concat.shape[0] == 100
        assert y_concat.shape[0] == 100

    def test_iteration_variadic(self):
        """Test that iteration works with >2 data arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = ArrayDataLoader(X, y, z, batch_size=32)

        for x, y, z in loader:
            assert x.shape[0] == y.shape[0] == z.shape[0]

    def test_re_iterable(self):
        """Test that DataLoader is re-iterable."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=False)

        # First iteration
        batches_1 = list(loader)
        # Second iteration
        batches_2 = list(loader)

        assert len(batches_1) == len(batches_2)
        for (X1, y1), (X2, y2) in zip(batches_1, batches_2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_shuffle(self):
        """Test shuffling produces different order (statistically)."""
        X = np.arange(1000).reshape(1000, 1)
        y = np.arange(1000)
        loader = ArrayDataLoader(
            X, y, batch_size=100, shuffle=True, key=jax.random.key(123)
        )

        X_batch1, _ = next(iter(loader))
        X_batch2, _ = next(iter(loader))

        # With shuffling, subsequent iterations should have different order
        # (extremely unlikely to be the same)
        assert not np.array_equal(X_batch1, X_batch2)

    def test_invalid_batch_size(self):
        """Test that batch_size <= 0 raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            ArrayDataLoader(X, y, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            ArrayDataLoader(X, y, batch_size=-1)

    def test_mismatched_samples(self):
        """Test that mismatched X and y raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="same number of samples"):
            ArrayDataLoader(X, y, batch_size=32)


class TestPreprocessedDataLoader:
    """Tests for _PreprocessedDataLoader class."""

    def test_preprocessing_applied(self):
        """Test that preprocessing is applied to batches."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=False)

        # Preprocessing function that scales X by 2
        def preprocess(X, y):
            return X * 2, y

        wrapped = _PreprocessedDataLoader(loader, preprocess)

        X_batch, y_batch = next(iter(wrapped))
        X_orig, _ = next(iter(loader))

        np.testing.assert_array_almost_equal(X_batch, X_orig * 2)

    def test_sample_batch_cached(self):
        """Test that sample_batch result is cached."""
        call_count = 0

        def preprocess(X, y):
            nonlocal call_count
            call_count += 1
            return X, y

        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)
        wrapped = _PreprocessedDataLoader(loader, preprocess)

        wrapped.sample_batch()
        wrapped.sample_batch()
        wrapped.sample_batch()

        assert call_count == 1  # Only called once


class TestIsDataLoader:
    """Tests for is_data_loader function."""

    def test_array_data_loader(self):
        """Test that ArrayDataLoader is recognized."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        assert is_data_loader(loader)

    def test_dict_not_data_loader(self):
        """Test that dict is not recognized as DataLoader."""
        assert not is_data_loader({"X": np.zeros((10, 5))})

    def test_list_not_data_loader(self):
        """Test that list is not recognized as DataLoader."""
        assert not is_data_loader([np.zeros((10, 5)), np.zeros(10)])

    def test_preprocessed_data_loader_is_data_loader(self):
        """Test that _PreprocessedDataLoader is recognized."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        def preprocess(X_batch, y_batch):
            return X_batch, y_batch

        wrapped = _PreprocessedDataLoader(loader, preprocess)
        assert is_data_loader(wrapped)


class TestSolverStochasticSupport:
    """Tests for solver stochastic support flags and utilities."""

    @pytest.mark.parametrize(
        "solver_name", _stochastic_solver_names + _non_stochastic_solver_names
    )
    def test_supports_stochastic(self, solver_name):
        """Test the right solvers have stochastic suppoer."""
        expectation = solver_name in _stochastic_solver_names
        assert solvers.supports_stochastic(solver_name) == expectation

    def test_list_stochastic_solvers(self):
        """Test list_stochastic_solvers returns expected solvers."""
        assert set(solvers.list_stochastic_solvers()) == set(_stochastic_solver_names)

    def test_unknown_solver_raises(self):
        """Test unknown solver name raises ValueError."""
        # TODO: Update when registry handles this.
        with pytest.raises(ValueError, match="Unknown solver"):
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

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_stochastic_fit(self, simple_data, solver):
        """Test basic stochastic_fit functionality."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        model = nmo.glm.GLM(
            solver_name=solver,
            solver_kwargs={"stepsize": 0.001, "maxiter": 100},
        )
        model.stochastic_fit(loader, num_epochs=5)

        n_steps_taken = model._solver.get_optim_info(model.solver_state_).num_steps

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.coef_.shape == (5,)
        assert n_steps_taken == (X.shape[0] // 32 + 1) * 5

    @pytest.mark.parametrize("solver", _stochastic_solver_names)
    def test_stochastic_fit_with_init_params(self, simple_data, solver):
        """Test stochastic_fit with provided initial parameters."""
        X, y = simple_data
        loader = ArrayDataLoader(X, y, batch_size=32)

        model = nmo.glm.GLM(
            solver_name=solver,
            solver_kwargs={"stepsize": 0.001, "maxiter": 100},
        )

        init_coef = jnp.zeros(5)
        init_intercept = jnp.zeros(1)
        init_params = (init_coef, init_intercept)

        model.stochastic_fit(loader, num_epochs=5, init_params=init_params)

        assert model.coef_ is not None

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
            solver_kwargs={"stepsize": 0.001, "maxiter": 100},
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

        model = nmo.glm.PopulationGLM(
            solver_name=solver,
            solver_kwargs={"stepsize": 0.001, "maxiter": 100},
        )
        model.stochastic_fit(loader, num_epochs=5)

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.coef_.shape == (5, 3)
        assert model.intercept_.shape == (3,)


class TestSolverStochasticRun:
    """Tests for solver stochastic_run method directly."""

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

        from nemos.regularizer import UnRegularized

        solver_kwargs = {"stepsize": 0.01}
        if "acceleration" in solver_class.get_accepted_arguments():
            solver_kwargs["acceleration"] = False

        solver = solver_class(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            **solver_kwargs,
        )

        init_params = jnp.zeros(3)
        params, state, aux = solver.stochastic_run(init_params, loader, num_epochs=10)

        assert params.shape == (3,)
        # Should have learned something close to [1, 2, 3]
        np.testing.assert_array_almost_equal(params, [1.0, 2.0, 3.0], decimal=0)

    def test_convergence_criterion_accepts_jax_scalar_bool(self, simple_loss_and_data):
        """Test convergence callback handles JAX scalar booleans like Python bool."""
        loss, loader = simple_loss_and_data

        from nemos.regularizer import UnRegularized

        init_params = jnp.zeros(3)

        solver_py = solvers.OptimistixOptaxGradientDescent(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )
        _, state_py, _ = solver_py.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            convergence_criterion=lambda *args: True,
        )
        steps_py = solver_py.get_optim_info(state_py).num_steps

        solver_jax = solvers.OptimistixOptaxGradientDescent(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )
        _, state_jax, _ = solver_jax.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            convergence_criterion=lambda *args: jnp.array(True),
        )
        steps_jax = solver_jax.get_optim_info(state_jax).num_steps

        assert steps_jax == steps_py

    def test_batch_callback_accepts_jax_scalar_bool(self, simple_loss_and_data):
        """Test batch callback handles JAX scalar booleans like Python bool."""
        loss, loader = simple_loss_and_data

        from nemos.regularizer import UnRegularized

        init_params = jnp.zeros(3)

        # TODO: Update these to use solvers.get_solver once that is available
        solver_py = solvers.OptimistixOptaxGradientDescent(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )
        _, state_py, _ = solver_py.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            batch_callback=lambda *args: True,
        )
        steps_py = solver_py.get_optim_info(state_py).num_steps

        solver_jax = solvers.OptimistixOptaxGradientDescent(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )
        _, state_jax, _ = solver_jax.stochastic_run(
            init_params,
            loader,
            num_epochs=5,
            batch_callback=lambda *args: jnp.array(True),
        )
        steps_jax = solver_jax.get_optim_info(state_jax).num_steps

        assert steps_jax == steps_py

    @pytest.mark.parametrize(
        "bad_flag", [1, 1.0, np.array(1.0), jnp.array(1.0), np.bool_(True).astype(int)]
    )
    def test_convergence_criterion_non_bool_scalar_raises(
        self, simple_loss_and_data, bad_flag
    ):
        """Test convergence callback rejects non-boolean scalar return values."""
        loss, loader = simple_loss_and_data

        from nemos.regularizer import UnRegularized

        solver = solvers.OptimistixOptaxGradientDescent(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )

        with pytest.raises(TypeError, match="scalar boolean"):
            solver.stochastic_run(
                jnp.zeros(3),
                loader,
                num_epochs=2,
                convergence_criterion=lambda *args: bad_flag,
            )

    @pytest.mark.parametrize(
        "bad_flag", [1, 1.0, np.array(1.0), jnp.array(1.0), np.bool_(True).astype(int)]
    )
    def test_batch_callback_non_bool_scalar_raises(
        self, simple_loss_and_data, bad_flag
    ):
        """Test batch callback rejects non-boolean scalar return values."""
        loss, loader = simple_loss_and_data

        from nemos.regularizer import UnRegularized

        solver = solvers.OptimistixOptaxGradientDescent(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )

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

        from nemos.regularizer import UnRegularized

        solver = solver_class(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
        )

        init_params = jnp.zeros(3)
        with pytest.raises(NotImplementedError, match="does not support stochastic"):
            solver.stochastic_run(init_params, loader, num_epochs=1)

    @pytest.mark.parametrize("solver_class", _stochastic_solver_classes)
    def test_invalid_num_epochs_raises(self, simple_loss_and_data, solver_class):
        """Test that num_epochs < 1 raises ValueError."""
        loss, loader = simple_loss_and_data

        from nemos.regularizer import UnRegularized

        solver = solver_class(
            loss,
            UnRegularized(),
            regularizer_strength=None,
            has_aux=False,
            stepsize=0.01,
        )

        init_params = jnp.zeros(3)
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            solver.stochastic_run(init_params, loader, num_epochs=0)
