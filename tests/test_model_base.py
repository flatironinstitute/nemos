import pytest
from typing import Union, Tuple
import jax.numpy as jnp
from numpy.typing import NDArray
from neurostatslib.model_base import Model  # Adjust this import to your module name

# Sample subclass to test instantiation and methods
class MockModel(Model):

    def __init__(self, std_param: int = 0, **kwargs):
        self.std_param = std_param
        super().__init__(**kwargs)

    def fit(self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]):
        pass

    def predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        pass

    def score(self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        pass

    def _predict(self, params: Tuple[jnp.ndarray, jnp.ndarray], X: NDArray) -> jnp.ndarray:
        pass

    def _score(self, X: jnp.ndarray, y: jnp.ndarray, params: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        pass

class MockModel_Invalid(Model):
    """
    Mock model that doesn't implement `_score` abstract method
    """

    def __init__(self, std_param: int = 0, **kwargs):
        self.std_param = std_param
        super().__init__(**kwargs)

    def fit(self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]):
        pass

    def predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        pass

    def score(self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        pass

    def _predict(self, params: Tuple[jnp.ndarray, jnp.ndarray], X: NDArray) -> jnp.ndarray:
        pass



def test_init():
    model = MockModel(param1="test", param2=2)
    assert model.param1 == "test"
    assert model.param2 == 2
    assert model.std_param == 0

def test_get_params():
    model = MockModel(param1="test", param2=2)
    params = model.get_params(deep=True)
    assert params["param1"] == "test"
    assert params["param2"] == 2
    assert params["std_param"] == 0

def set_params():
    model = MockModel(param1="init_param")
    model.set_params(param1="changed")
    model.set_params(std_param=1)
    assert model.param1 == "changed"
    assert model.std_param == 1

def test_invalid_set_params():
    model = MockModel()
    with pytest.raises(ValueError):
        model.set_params(invalid_param="invalid")

def test_get_param_names():
    param_names = MockModel._get_param_names()
    # As per your implementation, _get_param_names should capture the constructor arguments
    assert "std_param" in param_names

def test_convert_to_jnp_ndarray():
    data = [1, 2, 3]
    jnp_data, = Model._convert_to_jnp_ndarray(data)
    assert isinstance(jnp_data, jnp.ndarray)
    assert jnp.all(jnp_data == jnp.array(data, dtype=jnp.float32))

def test_has_invalid_entry():
    valid_data = jnp.array([1, 2, 3])
    invalid_data = jnp.array([1, 2, jnp.nan])
    assert not Model._has_invalid_entry(valid_data)
    assert Model._has_invalid_entry(invalid_data)

# To ensure abstract methods aren't callable
def test_abstract_class():
    with pytest.raises(TypeError, match="Can't instantiate abstract"):
        Model()

def test_invalid_concrete_class():
    with pytest.raises(TypeError, match="Can't instantiate abstract"):
        model = MockModel_Invalid()



