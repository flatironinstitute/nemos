from contextlib import nullcontext as does_not_raise
from numbers import Number
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import instantiate_base_regressor_subclass
from test_base_regressor_subclasses import (
    INSTANTIATE_MODEL_AND_SIMULATE,
    INSTANTIATE_MODEL_ONLY,
)

import nemos as nmo
from nemos._observation_model_builder import (
    instantiate_observation_model,
)
from nemos._regularizer_builder import instantiate_regularizer
from nemos.glm_hmm.params import GLMHMMParams, GLMParams
from nemos.typing import FeaturePytree
from nemos.utils import _get_name

# FILTER FOR GLM HMM
INSTANTIATE_MODEL_ONLY = INSTANTIATE_MODEL_ONLY.copy()
INSTANTIATE_MODEL_ONLY = [v for v in INSTANTIATE_MODEL_ONLY if "GLMHMM" in v["model"]]
INSTANTIATE_MODEL_AND_SIMULATE = INSTANTIATE_MODEL_AND_SIMULATE.copy()
INSTANTIATE_MODEL_AND_SIMULATE = [
    v for v in INSTANTIATE_MODEL_AND_SIMULATE if "GLMHMM" in v["model"]
]

DEFAULT_GLM_COEF_SHAPE = {
    "GLMHMM": (2, 3),
}


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_get_fit_attrs(instantiate_base_regressor_subclass):
    fixture = instantiate_base_regressor_subclass
    expected_state = {
        "coef_": None,
        "dof_resid_": None,
        "initial_prob_": None,
        "intercept_": None,
        "scale_": None,
        "solver_state_": None,
        "transition_prob_": None,
    }
    assert fixture.model._get_fit_state() == expected_state
    fixture.model.solver_kwargs = {"maxiter": 1}


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_AND_SIMULATE,
    indirect=True,
)
class TestGLMHMM:
    """
    Unit tests for the GLMHMM class that do not depend on the observation model.
    i.e. tests that do not call observation model methods, or tests that do not check the output when
    observation model methods are called (e.g. error testing for input validation)
    """

    @pytest.fixture
    def fit_weights_dimensionality_expectation(
        self, instantiate_base_regressor_subclass
    ):
        """
        Fixture to define the expected behavior for test_fit_weights_dimensionality based on the type of GLM class.
        """
        model_cls = instantiate_base_regressor_subclass.model.__class__
        if "Population" in model_cls.__name__:
            # FILL IN WHEN POPULATION CLASS IS DEFINED
            # NOTE THAT THE FIXTURE WILL MAKE TESTS FAIL FOR POPULATION, WHICH IS
            # ENOUGH TO REMIND US TO FILL THIS IN
            return {}
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Invalid parameter dimensionality",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"Invalid parameter dimensionality",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"Invalid parameter dimensionality",
                ),
            }

    @pytest.fixture
    def initialize_solver_weights_dimensionality_expectation(
        self, instantiate_base_regressor_subclass
    ):
        name = instantiate_base_regressor_subclass.model.__class__.__name__
        if "Population" in name:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                3: does_not_raise(),
            }
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            }

    #
    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    def test_initialize_solver_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `initialize_solver` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        n_samples, n_features = fixture.X.shape

        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros((n_features, 3))
        elif dim_weights == 3:
            init_w = jnp.zeros(
                (n_features, fixture.y.shape[1] if fixture.y.ndim > 1 else 1, 3)
            )
        else:
            init_w = jnp.zeros((n_features, 3) + (1,) * (dim_weights - 2))
        with expectation:
            params = fixture.model.initialize_params(
                fixture.X,
                fixture.y,
            )
            params = tuple([init_w, *params[1:]])
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, params
            )

    @pytest.mark.parametrize(
        "dim_intercepts",
        [0, 1, 2, 3],
    )
    def test_initialize_solver_intercepts_dimensionality(
        self,
        dim_intercepts,
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `initialize_solver` method with intercepts of different dimensionalities.

        Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        n_samples, n_features = fixture.X.shape
        is_population = "Population" in fixture.model.__class__.__name__
        if (dim_intercepts == 2 and is_population) or (
            dim_intercepts == 1 and not is_population
        ):
            expectation = does_not_raise()
        elif dim_intercepts == 0:
            expectation = pytest.raises(
                ValueError, match=r"GLM intercept must be of shape"
            )
        else:

            expectation = pytest.raises(
                ValueError, match=r"Invalid parameter dimensionality"
            )
        if is_population:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_b = (
                jnp.zeros((3,) + (1,) * (dim_intercepts - 1))
                if dim_intercepts > 0
                else jnp.array([])
            )
            init_w = jnp.zeros((n_features, 3))
        with expectation:
            params = fixture.model.initialize_params(
                fixture.X,
                fixture.y,
            )
            params = tuple([init_w, init_b, *params[2:]])
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, params
            )

    """
    Parameterization used by test_fit_init_params_type and test_initialize_solver_init_params_type
    Contains the expected behavior and separate initial parameters for regular and population GLMs

    Note: init_params is expected to be a 5-tuple (coef, intercept, scale, initial_prob, transition_prob)
    """
    fit_init_params_type_init_params = (
        "expectation, init_params_glm, init_params_population_glm",
        [
            # Valid case: proper arrays for all 5 params
            (
                does_not_raise(),
                (
                    jnp.zeros((2, 3)),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    jnp.zeros((2, 3, 3)),
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),  # scale for population
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # Wrong length tuple (not 5 elements)
            (
                pytest.raises(ValueError, match="Params must have length 5"),
                (jnp.zeros((1, 2, 3)), jnp.zeros((3,))),  # Only 2 elements
                (jnp.zeros((1, 2, 3)), jnp.zeros((3, 3))),
            ),
            # Dict instead of tuple for coef (X is array, so dict coef raises error)
            # Different observation models may raise AttributeError or TypeError
            (
                pytest.raises((AttributeError, TypeError)),
                (
                    dict(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    dict(p1=jnp.zeros((2, 3, 3)), p2=jnp.zeros((2, 2, 3))),
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # FeaturePytree for coef (X is array, so FeaturePytree coef raises TypeError)
            (
                pytest.raises(TypeError, match=r"X and coef have mismatched structure"),
                (
                    FeaturePytree(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    FeaturePytree(p1=jnp.zeros((1, 3, 3)), p2=jnp.zeros((1, 2, 3))),
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # Scalar instead of tuple (wrong type)
            (pytest.raises(ValueError, match="Params must have length 5"), 0, 0),
            # Set instead of tuple (wrong type, not subscriptable, raises ValueError about length)
            (
                pytest.raises(ValueError, match="Params must have length 5"),
                {0, 1},
                {0, 1},
            ),
            # String as intercept (wrong type)
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                (
                    jnp.zeros((2, 3)),
                    "",
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    jnp.zeros((2, 3, 3)),
                    "",
                    jnp.ones((3, 3)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # String as coef (wrong type)
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                (
                    "",
                    jnp.zeros((3,)),
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    "",
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
        ],
    )

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    def test_initialize_solver_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.

        Ensure that the provided initial parameters are array-like.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm
        with expectation:
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, init_params
            )

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_glm_coef(
        self,
        delta_n_features,
        expectation,
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_w = jnp.zeros((fixture.X.shape[1] + delta_n_features, 3))
            init_b = jnp.ones(3)
        with expectation:
            params = fixture.model.initialize_params(
                fixture.X,
                fixture.y,
            )
            params = tuple([init_w, init_b, *params[2:]])
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, params
            )
