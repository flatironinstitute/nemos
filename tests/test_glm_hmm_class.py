import jax.numpy as jnp
import pytest

import nemos as nmo

INSTANTIATE_MODEL_ONLY = [
    {"obs_model": "Poisson", "simulate": False},
    {"obs_model": "Bernoulli", "simulate": False},
    {"obs_model": "NegativeBinomial", "simulate": False},
    {"obs_model": "Gamma", "simulate": False},
]

INSTANTIATE_MODEL_AND_SIMULATE = [
    {"obs_model": "Poisson", "simulate": True},
    {"obs_model": "Bernoulli", "simulate": True},
    {"obs_model": "NegativeBinomial", "simulate": True},
    {"obs_model": "Gamma", "simulate": True},
]


@pytest.mark.parametrize(
    "instantiate_glm_hmm",
    INSTANTIATE_MODEL_AND_SIMULATE,
    indirect=True,
)
def test_get_fit_attrs(instantiate_glm_hmm):
    X, y, model, params, rates, latents = instantiate_glm_hmm
    expected_state = {
        "coef_": None,
        "glm_params_": None,
        "initial_prob_": None,
        "intercept_": None,
        "transition_prob_": None,
    }
    assert model._get_fit_state() == expected_state
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)
    assert all(val is not None for val in model._get_fit_state().values())
    assert model._get_fit_state().keys() == expected_state.keys()


@pytest.mark.parametrize(
    "instantiate_glm_hmm",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_validate_lower_dimensional_data_X(instantiate_glm_hmm):
    """Test behavior with lower-dimensional input data."""
    model = instantiate_glm_hmm[2]
    X = jnp.array([1, 2])
    y = jnp.array([1, 2])
    with pytest.raises(ValueError, match="X must be two-dimensional"):
        model._validate(X, y, model._initialize_parameters(X, y))


@pytest.mark.parametrize(
    "instantiate_glm_hmm",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_preprocess_fit_higher_dimensional_data_y(instantiate_glm_hmm):
    """Test behavior with higher-dimensional input data."""
    model = instantiate_glm_hmm[2]
    X = jnp.array([[[1, 2], [3, 4]]])
    y = jnp.array([[[1, 2]]])
    with pytest.raises(ValueError, match="y must be one-dimensional"):
        model._validate(X, y, model._initialize_parameters(X, y))


@pytest.mark.parametrize(
    "instantiate_glm_hmm",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_validate_higher_dimensional_data_X(instantiate_glm_hmm):
    """Test behavior with higher-dimensional input data."""
    model = instantiate_glm_hmm[2]
    X = jnp.array([[[[1, 2], [3, 4]]]])
    y = jnp.array([1, 2])
    with pytest.raises(ValueError, match="X must be two-dimensional"):
        model._validate(X, y, model._initialize_parameters(X, y))
