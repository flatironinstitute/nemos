from typing import Union

import jax.numpy as jnp
import pytest
from numpy.typing import NDArray

from nemos.base_class import Base
from nemos.base_regressor import BaseRegressor


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


def set_params(mock_regressor):
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
        RuntimeError, match="GLM estimators should always specify their parameters"
    ):
        bad_estimator._get_param_names()
