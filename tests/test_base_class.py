from typing import Union
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from numpy.typing import NDArray

from nemos.base_class import Base
from nemos.base_regressor import BaseRegressor
from nemos.regularizer import Ridge


class MockBaseRegressorInvalid(BaseRegressor):
    """
    Mock model that intentionally doesn't implement all the required abstract methods.
    Used for testing the instantiation of incomplete concrete classes.
    """

    def __init__(self, std_param: int = 0):
        self.std_param = std_param
        super().__init__()

    def predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        pass

    def score(self, X, y, score_type="pseudo-r2-McFadden"):
        pass

    def _get_optimal_solver_params_config(self):
        return None, None, None


class BadEstimator(Base):
    def __init__(self, param1, *args):
        super().__init__()
        pass


def test_init(mock_regressor):
    """Test the initialization of the MockBaseRegressor class."""
    assert mock_regressor.std_param == 2


def test_get_params(mock_regressor_nested):
    """Test the get_params method."""
    params = mock_regressor_nested.get_params(deep=True)
    assert params["std_param"] == 2
    assert params["other_param__std_param"] == 1


def test_set_params(mock_regressor):
    """Test the set_params method."""
    model = mock_regressor
    model.set_params(std_param=1)
    assert model.std_param == 1


def test_invalid_set_params(mock_regressor):
    """Test invalid parameter setting using the set_params method."""
    model = mock_regressor
    with pytest.raises(
        ValueError, match="Invalid parameter 'invalid_param' for estimator"
    ):
        model.set_params(invalid_param="invalid")


def test_get_param_names(mock_regressor):
    """Test retrieval of parameter names using the _get_param_names method."""
    param_names = mock_regressor._get_param_names()
    # As per your implementation, _get_param_names should capture the constructor arguments
    assert "std_param" in param_names


# To ensure abstract methods aren't callable
def test_abstract_class():
    """Ensure that abstract methods aren't callable."""
    with pytest.raises(TypeError, match="Can't instantiate abstract"):
        BaseRegressor()


def test_invalid_concrete_class():
    """Ensure that classes missing implementation of required abstract methods raise errors."""
    with pytest.raises(TypeError, match="Can't instantiate abstract"):
        MockBaseRegressorInvalid()


def test_empty_set(mock_regressor):
    """Check that an empty set_params returns self."""
    assert mock_regressor.set_params() is mock_regressor


def test_glm_varargs_error():
    """Test that variable number of argument in __init__ is not allowed."""
    bad_estimator = BadEstimator(1)
    with pytest.raises(
        RuntimeError,
        match="scikit-learn estimators should always specify their parameters",
    ):
        bad_estimator._get_param_names()


class TestInstantiateSolverOverrides:
    """Tests that optional params in _instantiate_solver fall back to or override instance attributes."""

    @pytest.fixture
    def regressor(self, mock_regressor):
        return mock_regressor

    @pytest.fixture
    def mock_solver_cls(self):
        mock_cls = MagicMock()
        mock_cls.get_accepted_arguments.return_value = []
        mock_instance = MagicMock()
        mock_instance.fun = MagicMock()
        mock_cls.return_value = mock_instance
        return mock_cls

    @pytest.fixture
    def mock_get_solver(self, mock_solver_cls):
        spec = MagicMock()
        spec.implementation = mock_solver_cls
        return MagicMock(return_value=spec)

    @pytest.mark.parametrize("solver_name_override, expected", [
        (None, None),  # None → falls back to self.solver_name, resolved at runtime
        ("LBFGS[optax+optimistix]", "LBFGS[optax+optimistix]"),
    ])
    def test_solver_name_resolution(self, regressor, mock_get_solver, solver_name_override, expected):
        if expected is None:
            expected = regressor.solver_name
        with patch("nemos.base_regressor.solvers.get_solver", mock_get_solver):
            regressor._instantiate_solver(lambda p, X, y: None, None, solver_name=solver_name_override)
        mock_get_solver.assert_called_once_with(expected)

    @pytest.mark.parametrize("regularizer_override", [None, Ridge()])
    def test_regularizer_resolution(self, regressor, mock_solver_cls, mock_get_solver, regularizer_override):
        expected = regularizer_override if regularizer_override is not None else regressor.regularizer
        with patch("nemos.base_regressor.solvers.get_solver", mock_get_solver):
            regressor._instantiate_solver(lambda p, X, y: None, None, regularizer=regularizer_override)
        assert mock_solver_cls.call_args.args[1] == expected

    @pytest.mark.parametrize("strength_override", [None, 2.0])
    def test_regularizer_strength_resolution(self, regressor, mock_solver_cls, mock_get_solver, strength_override):
        expected = strength_override if strength_override is not None else regressor.regularizer_strength
        with patch("nemos.base_regressor.solvers.get_solver", mock_get_solver):
            regressor._instantiate_solver(lambda p, X, y: None, None, regularizer_strength=strength_override)
        assert mock_solver_cls.call_args.args[2] == expected

    @pytest.mark.parametrize("solver_kwargs_override, extra_accepted_args", [
        (None, []),
        ({"tol": 1e-4}, ["tol"]),
    ])
    def test_solver_kwargs_resolution(self, regressor, mock_solver_cls, mock_get_solver, solver_kwargs_override, extra_accepted_args):
        mock_solver_cls.get_accepted_arguments.return_value = extra_accepted_args
        expected = solver_kwargs_override if solver_kwargs_override is not None else regressor.solver_kwargs
        with patch("nemos.base_regressor.solvers.get_solver", mock_get_solver):
            regressor._instantiate_solver(lambda p, X, y: None, None, solver_kwargs=solver_kwargs_override)
        actual_kwargs = mock_solver_cls.call_args.kwargs
        for k, v in expected.items():
            assert actual_kwargs[k] == v
