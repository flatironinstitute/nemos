from typing import Literal, Union

import jax
import jax.numpy as jnp
import pytest
from numpy.typing import NDArray

import nemos as nmo
from nemos.base_class import Base, BaseRegressor
from nemos.utils import check_invalid_entry, convert_to_jnp_ndarray


@pytest.fixture
def mock_regressor():
    return MockBaseRegressor()


# Sample subclass to test instantiation and methods
class MockBaseRegressor(BaseRegressor):
    """
    Mock implementation of the BaseRegressor abstract class for testing purposes.
    Implements all required abstract methods as empty methods.
    """

    def __init__(self, std_param: int = 0):
        """Initialize a MockBaseRegressor instance with optional standard parameters."""
        self.std_param = std_param
        super().__init__()

    def fit(self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]):
        pass

    def predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        pass

    def score(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        **kwargs,
    ) -> jnp.ndarray:
        pass

    def simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        feed_forward_input: Union[NDArray, jnp.ndarray],
        **kwargs,
    ):
        pass


class MockRegressorNested(MockBaseRegressor):
    def __init__(self, other_param: int, std_param: int = 0):
        super().__init__(std_param=std_param)
        self.other_param = MockBaseRegressor(std_param=other_param)


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

    def score(
        self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]
    ) -> jnp.ndarray:
        pass


class BadEstimator(Base):
    def __init__(self, param1, *args):
        super().__init__()
        pass


def test_init():
    """Test the initialization of the MockBaseRegressor class."""
    model = MockBaseRegressor(std_param=2)
    assert model.std_param == 2


def test_get_params():
    """Test the get_params method."""
    model = MockRegressorNested(other_param=1, std_param=2)
    params = model.get_params(deep=True)
    assert params["std_param"] == 2
    assert params["other_param__std_param"] == 1


def set_params():
    """Test the set_params method."""
    model = MockBaseRegressor()
    model.set_params(std_param=1)
    assert model.std_param == 1


def test_invalid_set_params():
    """Test invalid parameter setting using the set_params method."""
    model = MockBaseRegressor()
    with pytest.raises(
        ValueError, match="Invalid parameter 'invalid_param' for estimator"
    ):
        model.set_params(invalid_param="invalid")


def test_get_param_names():
    """Test retrieval of parameter names using the _get_param_names method."""
    param_names = MockBaseRegressor._get_param_names()
    # As per your implementation, _get_param_names should capture the constructor arguments
    assert "std_param" in param_names


def test_convert_to_jnp_ndarray():
    """Test data conversion to JAX NumPy arrays."""
    data = [1, 2, 3]
    (jnp_data,) = convert_to_jnp_ndarray(data)
    assert isinstance(jnp_data, jnp.ndarray)
    assert jnp.all(jnp_data == jnp.array(data, dtype=jnp.float32))


def test_check_invalid_entry():
    """Test validation of data arrays."""
    valid_data = jnp.array([1, 2, 3])
    invalid_data_nan = jnp.array([1, 2, jnp.nan])
    invalid_data_inf = jnp.array([1, jnp.inf, 2])
    check_invalid_entry(valid_data)
    with pytest.raises(ValueError, match="Input array contains NaN"):
        check_invalid_entry(invalid_data_nan)
    with pytest.raises(ValueError, match="Input array contains Inf"):
        check_invalid_entry(invalid_data_inf)


# To ensure abstract methods aren't callable
def test_abstract_class():
    """Ensure that abstract methods aren't callable."""
    with pytest.raises(TypeError, match="Can't instantiate abstract"):
        BaseRegressor()


def test_invalid_concrete_class():
    """Ensure that classes missing implementation of required abstract methods raise errors."""
    with pytest.raises(TypeError, match="Can't instantiate abstract"):
        model = MockBaseRegressorInvalid()


def test_preprocess_fit(mock_data, mock_regressor):
    X, y = mock_data
    X_out, y_out, params_out = mock_regressor._preprocess_fit(X, y)
    assert X_out.shape == X.shape
    assert y_out.shape == y.shape
    assert params_out[0].shape == (2, 2)  # Mock data shapes
    assert params_out[1].shape == (2,)


def test_preprocess_fit_empty_data(mock_regressor):
    """Test behavior with empty data input."""
    X, y = jnp.array([[]]), jnp.array([])
    with pytest.raises(ValueError):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_mismatched_shapes(mock_regressor):
    """Test behavior with mismatched X and y shapes."""
    X = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([1, 2, 3])
    with pytest.raises(ValueError):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_invalid_datatypes(mock_regressor):
    """Test behavior with invalid data types."""
    X = "invalid_data_type"
    y = "invalid_data_type"
    with pytest.raises(TypeError):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_with_nan_in_X(mock_regressor):
    """Test behavior with NaN values in data."""
    X = jnp.array([[[1, 2], [jnp.nan, 4]]])
    y = jnp.array([[1, 2]])
    with pytest.raises(ValueError, match="Input array contains"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_with_inf_in_X(mock_regressor):
    """Test behavior with inf values in data."""
    X = jnp.array([[[1, 2], [jnp.inf, 4]]])
    y = jnp.array([[1, 2]])
    with pytest.raises(ValueError, match="Input array contains"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_with_nan_in_y(mock_regressor):
    """Test behavior with NaN values in data."""
    X = jnp.array([[[1, 2], [2, 4]]])
    y = jnp.array([[1, jnp.nan]])
    with pytest.raises(ValueError, match="Input array contains"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_with_inf_in_y(mock_regressor):
    """Test behavior with inf values in data."""
    X = jnp.array([[[1, 2], [2, 4]]])
    y = jnp.array([[1, jnp.inf]])
    with pytest.raises(ValueError, match="Input array contains"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_higher_dimensional_data_X(mock_regressor):
    """Test behavior with higher-dimensional input data."""
    X = jnp.array([[[[1, 2], [3, 4]]]])
    y = jnp.array([[1, 2]])
    with pytest.raises(ValueError, match="X must be three-dimensional"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_higher_dimensional_data_y(mock_regressor):
    """Test behavior with higher-dimensional input data."""
    X = jnp.array([[[[1, 2], [3, 4]]]])
    y = jnp.array([[[1, 2]]])
    with pytest.raises(ValueError, match="y must be two-dimensional"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_lower_dimensional_data_X(mock_regressor):
    """Test behavior with lower-dimensional input data."""
    X = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[1, 2]])
    with pytest.raises(ValueError, match="X must be three-dimensional"):
        mock_regressor._preprocess_fit(X, y)


def test_preprocess_fit_lower_dimensional_data_y(mock_regressor):
    """Test behavior with lower-dimensional input data."""
    X = jnp.array([[[[1, 2], [3, 4]]]])
    y = jnp.array([1, 2])
    with pytest.raises(ValueError, match="y must be two-dimensional"):
        mock_regressor._preprocess_fit(X, y)


# Preprocess Simulate Tests
def test_preprocess_simulate_empty_data(mock_regressor):
    """Test behavior with empty feedforward_input."""
    feedforward_input = jnp.array([[[]]])
    params_f = (jnp.array([[]]), jnp.array([]))
    with pytest.raises(ValueError, match="Model parameters have inconsistent shapes."):
        mock_regressor._preprocess_simulate(feedforward_input, params_f)


def test_preprocess_simulate_invalid_datatypes(mock_regressor):
    """Test behavior with invalid feedforward_input datatype."""
    feedforward_input = "invalid_data_type"
    params_f = (jnp.array([[]]),)
    with pytest.raises(
        TypeError,
        match="Value 'invalid_data_type' with dtype .+ is not a valid JAX array type.",
    ):
        mock_regressor._preprocess_simulate(feedforward_input, params_f)


def test_preprocess_simulate_with_nan(mock_regressor):
    """Test behavior with NaN values in feedforward_input."""
    feedforward_input = jnp.array([[[jnp.nan]]])
    params_f = (jnp.array([[1]]), jnp.array([1]))
    with pytest.raises(ValueError, match="Input array contains"):
        mock_regressor._preprocess_simulate(feedforward_input, params_f)


def test_preprocess_simulate_with_inf(mock_regressor):
    """Test behavior with infinite values in feedforward_input."""
    feedforward_input = jnp.array([[[jnp.inf]]])
    params_f = (jnp.array([[1]]), jnp.array([1]))
    with pytest.raises(ValueError, match="Input array contains"):
        mock_regressor._preprocess_simulate(feedforward_input, params_f)


def test_preprocess_simulate_higher_dimensional_data(mock_regressor):
    """Test behavior with improperly dimensional feedforward_input."""
    feedforward_input = jnp.array([[[[1]]]])
    params_f = (jnp.array([[1]]), jnp.array([1]))
    with pytest.raises(ValueError, match="X must be three-dimensional"):
        mock_regressor._preprocess_simulate(feedforward_input, params_f)


def test_preprocess_simulate_invalid_init_y(mock_regressor):
    """Test behavior with invalid init_y provided."""
    feedforward_input = jnp.array([[[1]]])
    params_f = (jnp.array([[1]]), jnp.array([1]))
    init_y = jnp.array([[[1]]])
    params_r = (jnp.array([[1]]),)
    with pytest.raises(ValueError, match="y must be two-dimensional"):
        mock_regressor._preprocess_simulate(
            feedforward_input, params_f, init_y, params_r
        )


def test_preprocess_simulate_feedforward(mock_regressor):
    """Test that the preprocessing works."""
    feedforward_input = jnp.array([[[1]]])
    params_f = (jnp.array([[1]]), jnp.array([1]))
    (ff,) = mock_regressor._preprocess_simulate(feedforward_input, params_f)
    assert jnp.all(ff == feedforward_input)


def test_empty_set(mock_regressor):
    """Check that an empty set_params returns self."""
    assert mock_regressor.set_params() is mock_regressor


@pytest.mark.parametrize("device_name", ["cpu", "gpu", "tpu", "unknown"])
def test_target_device_put(device_name: Literal["cpu", "gpu", "tpu"], mock_regressor):
    """Test that put works.

    Put array to device and checks that the device is matched after put, if device is found.
    Raise error otherwise.
    """
    raise_exception = not any(
        device_name in device.device_kind.lower() for device in jax.local_devices()
    )
    x = jnp.array([1])
    if raise_exception:
        if device_name != "tpu":
            with pytest.raises(RuntimeError, match=f"Unknown backend"):
                mock_regressor.device_put(x, device=device_name)
        else:
            with pytest.raises(
                RuntimeError, match=f"Backend '{device_name}' failed to initialize: "
            ):
                mock_regressor.device_put(x, device=device_name)
    else:
        (x,) = mock_regressor.device_put(x, device=device_name)
        assert x.device().device_kind == device_name


def test_glm_varargs_error():
    """Test that variable number of argument in __init__ is not allowed."""
    bad_estimator = BadEstimator(1)
    with pytest.raises(
        RuntimeError, match="GLM estimators should always specify their parameters"
    ):
        bad_estimator._get_param_names()
