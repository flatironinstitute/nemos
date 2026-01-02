from contextlib import nullcontext as does_not_raise
from numbers import Number
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest
from conftest import instantiate_base_regressor_subclass
from test_base_regressor_subclasses import (
    INSTANTIATE_MODEL_AND_SIMULATE,
    INSTANTIATE_MODEL_ONLY,
)

import nemos as nmo
from nemos import tree_utils
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
    fixture.model.fit(fixture.X, fixture.y)
    assert all(val is not None for val in fixture.model._get_fit_state().values())
    assert fixture.model._get_fit_state().keys() == expected_state.keys()


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

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    @pytest.mark.requires_x64
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
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
            init_w = jnp.zeros(DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__])
        else:
            init_w = jnp.zeros(
                DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__]
                + (1,) * (dim_weights - 2)
            )
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    init_w,
                    fixture.params.glm_params.intercept,
                    jnp.exp(fixture.params.glm_scale.log_scale),
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (
                0,
                pytest.raises(ValueError, match=r"Invalid parameter dimensionality"),
            ),
            (1, does_not_raise()),
            (
                2,
                pytest.raises(ValueError, match=r"Invalid parameter dimensionality"),
            ),
            (
                3,
                pytest.raises(ValueError, match=r"Invalid parameter dimensionality"),
            ),
        ],
    )
    @pytest.mark.requires_x64
    def test_fit_intercepts_dimensionality(
        self, dim_intercepts, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        if dim_intercepts == 1:
            init_b = jnp.ones(
                DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__][1]
            )
        else:
            init_b = jnp.ones((1,) * dim_intercepts)
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    fixture.params.glm_params.coef,
                    init_b,
                    jnp.exp(fixture.params.glm_scale.log_scale),
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
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
    @pytest.mark.requires_x64
    def test_fit_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm

        with expectation:
            fixture.model.fit(fixture.X, fixture.y, init_params=init_params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.requires_x64
    def test_fit_n_feature_consistency_weights(
        self,
        delta_n_features,
        instantiate_base_regressor_subclass,
        expectation,
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_w = jnp.zeros((fixture.X.shape[1] + delta_n_features, 3))
            init_b = jnp.ones(
                3,
            )
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    init_w,
                    init_b,
                    jnp.exp(fixture.params.glm_scale.log_scale),
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

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

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            (np.array([[np.nan], [0]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [np.nan]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([np.nan, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([0, np.nan]), does_not_raise()),
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, 1, 2]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                np.array([[0], [0], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [np.nan], [0], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                does_not_raise(),
            ),
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, np.nan, 2, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, 1, np.nan, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs in middle - should fail
            (
                np.array([[0], [np.nan], [np.nan], [0]]),
                np.array([0, 1, 2, 3]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            # Multiple consecutive NaNs at start - should pass
            (
                np.array([[np.nan], [np.nan], [0]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs at end - should pass
            (
                np.array([[0], [np.nan], [np.nan]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # All NaNs - should fail (caught by parent validation)
            (
                np.array([[np.nan], [np.nan]]),
                np.array([np.nan, np.nan]),
                pytest.raises(ValueError),
            ),
            # No NaNs - should pass
            (np.array([[0], [1]]), np.array([0, 1]), does_not_raise()),
            # NaN at start of second epoch - should pass
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 2], [2, 5]),
                ),
                np.zeros(5),
                does_not_raise(),
            ),
            # Both X and y with NaNs in middle at different positions - should fail
            (
                np.array([[0], [np.nan], [0], [0]]),
                np.array([0, 1, np.nan, 3]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            # Both X and y with NaNs in middle at same position - should fail
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
        ],
    )
    def test_nan_between_epochs(
        self,
        instantiate_base_regressor_subclass,
        X,
        y,
        expectation,
    ):
        """Test that NaN values are only allowed at epoch boundaries, not in the middle.

        The GLM-HMM forward-backward algorithm requires continuous time-series data within
        each epoch. This validation ensures data quality by rejecting datasets with NaN
        values that would break the algorithm's recurrence relations.

        Test coverage:
        - NaNs at start/end of data → allowed (epoch boundaries)
        - NaNs in middle of epoch → rejected (breaks continuity)
        - Multiple consecutive NaNs at borders → allowed
        - Multiple consecutive NaNs in middle → rejected
        - All NaN data → rejected (caught by parent validation)
        - No NaNs → allowed (trivial case)
        - NaNs at epoch boundaries in multi-epoch pynapple data → allowed
        - NaNs in both X and y → properly combined and validated

        This strict validation prevents runtime errors in the forward-backward algorithm
        and ensures users provide properly formatted time-series data.
        """
        fixture = instantiate_base_regressor_subclass
        model = fixture.model
        with expectation:
            model._validator.validate_inputs(X, y)


@pytest.mark.parametrize(
    "regularizer", ["Ridge", "UnRegularized", "Lasso", "ElasticNet"]
)
@pytest.mark.parametrize(
    "obs_model",
    [
        "PoissonObservations",
        "BernoulliObservations",
        "GammaObservations",
    ],
)
@pytest.mark.parametrize(
    "solver_name",
    [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "NonlinearCG",
        "ProximalGradient",
        "SVRG",
        "ProxSVRG",
    ],
)
@pytest.mark.parametrize(
    "model_class, fit_state_attrs",
    [
        (
            nmo.glm_hmm.GLMHMM,
            {
                "coef_": jnp.zeros((2, 3)),
                "intercept_": jnp.array([1.0, 1.0, 1.0]),
                "scale_": 2.0 * jnp.ones(3),
                "solver_state_": None,
                "transition_prob_": None,
                "initial_prob_": None,
                "dof_resid_": None,
            },
        ),
    ],
)
def test_save_and_load(
    regularizer,
    obs_model,
    solver_name,
    tmp_path,
    fit_state_attrs,
    model_class,
):
    """
    Test saving and loading a model with various observation models and regularizers.
    Ensure all parameters are preserved.
    """
    if (
        regularizer == "Lasso"
        or regularizer == "GroupLasso"
        or regularizer == "ElasticNet"
        and solver_name not in ["ProximalGradient", "ProxSVRG"]
    ):
        pytest.skip(
            f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
        )

    kwargs = dict(
        n_states=3,
        observation_model=obs_model,
        solver_name=solver_name,
        regularizer=regularizer,
        regularizer_strength=2.0,
        solver_kwargs={"tol": 10**-6},
    )

    if regularizer == "UnRegularized":
        kwargs.pop("regularizer_strength")

    model = model_class(**kwargs)

    initial_params = model.get_params()
    # set fit states
    for key, val in fit_state_attrs.items():
        setattr(model, key, val)
        initial_params[key] = val

    # Save
    save_path = tmp_path / "test_model.npz"
    model.save_params(save_path)

    # Load
    loaded_model = nmo.load_model(save_path)
    loaded_params = loaded_model.get_params()
    fit_state = loaded_model._get_fit_state()
    loaded_params.update(fit_state)

    # Assert matching keys and values
    assert set(initial_params.keys()) == set(
        loaded_params.keys()
    ), "Parameter keys mismatch after load."

    for key in initial_params:
        init_val = initial_params[key]
        load_val = loaded_params[key]
        if isinstance(init_val, (int, float, str, type(None))):
            assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
        elif isinstance(init_val, dict):
            assert (
                init_val == load_val
            ), f"{key} dict mismatch: {init_val} != {load_val}"
        elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
            assert np.allclose(
                np.array(init_val), np.array(load_val)
            ), f"{key} array mismatch"
        elif isinstance(init_val, Callable):
            assert _get_name(init_val) == _get_name(
                load_val
            ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"


@pytest.mark.parametrize("regularizer", ["Ridge"])
@pytest.mark.parametrize(
    "obs_model",
    [
        "PoissonObservations",
    ],
)
@pytest.mark.parametrize(
    "solver_name",
    [
        "ProxSVRG",
    ],
)
@pytest.mark.parametrize(
    "model_class, fit_state_attrs",
    [
        (
            nmo.glm_hmm.GLMHMM,
            {
                "coef_": jnp.zeros((2, 3)),
                "intercept_": jnp.array([1.0, 1.0, 1.0]),
                "scale_": 2.0,
                "dof_resid_": None,
                "solver_state_": None,
                "transition_prob_": None,
                "initial_prob_": None,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "mapping_dict, expectation",
    [
        ({}, does_not_raise()),
        (
            {
                "observation_model": nmo.observation_models.GammaObservations,
                "regularizer": nmo.regularizer.Lasso,
                "inverse_link_function": lambda x: x**2,
            },
            pytest.warns(UserWarning, match="The following keys have been replaced"),
        ),
        (
            {
                "observation_model": nmo.observation_models.GammaObservations(),  # fails, only class or callable
                "regularizer": nmo.regularizer.Lasso,
                "inverse_link_function": lambda x: x**2,
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "observation_model": "GammaObservations",  # fails, only class or callable
                "regularizer": nmo.regularizer.Lasso,
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "regularizer": nmo.regularizer.Lasso,
                "regularizer_strength": 3.0,  # fails, only class or callable
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "solver_kwargs": {"tol": 10**-1},
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "some__nested__dictionary": {"tol": 10**-1},
            },
            pytest.raises(
                ValueError,
                match="The following keys in your mapping do not match",
            ),
        ),
        # valid mapping dtype, invalid name
        (
            {
                "some__nested__dictionary": nmo.regularizer.Ridge,
            },
            pytest.raises(
                ValueError,
                match="The following keys in your mapping do not match ",
            ),
        ),
    ],
)
def test_save_and_load_with_custom_mapping(
    regularizer,
    obs_model,
    solver_name,
    mapping_dict,
    tmp_path,
    fit_state_attrs,
    model_class,
    expectation,
):
    """
    Test saving and loading a model with various observation models and regularizers.
    Ensure all parameters are preserved.
    """

    if (
        regularizer == "Lasso"
        or regularizer == "GroupLasso"
        and solver_name not in ["ProximalGradient", "SVRG", "ProxSVRG"]
    ):
        pytest.skip(
            f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
        )

    model = model_class(
        n_states=3,
        observation_model=obs_model,
        solver_name=solver_name,
        regularizer=regularizer,
        regularizer_strength=2.0,
    )

    initial_params = model.get_params()
    # set fit states
    for key, val in fit_state_attrs.items():
        setattr(model, key, val)
        initial_params[key] = val

    # Save
    save_path = tmp_path / "test_model.npz"
    model.save_params(save_path)

    # Load
    with expectation:
        loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
        loaded_params = loaded_model.get_params()
        fit_state = loaded_model._get_fit_state()
        loaded_params.update(fit_state)

        # Assert matching keys and values
        assert set(initial_params.keys()) == set(
            loaded_params.keys()
        ), "Parameter keys mismatch after load."

        unexpected_keys = set(mapping_dict) - set(initial_params)
        raise_exception = bool(unexpected_keys)
        if raise_exception:
            with pytest.raises(
                ValueError, match="mapping_dict contains unexpected keys"
            ):
                raise ValueError(
                    f"mapping_dict contains unexpected keys: {unexpected_keys}"
                )

        for key in initial_params:
            init_val = initial_params[key]
            load_val = loaded_params[key]

            if key == "observation_model__inverse_link_function":
                if "observation_model" in mapping_dict:
                    continue
            if key in mapping_dict:
                if key == "observation_model":
                    if isinstance(mapping_dict[key], str):
                        mapping_obs = instantiate_observation_model(mapping_dict[key])
                    else:
                        mapping_obs = mapping_dict[key]
                    assert _get_name(mapping_obs) == _get_name(
                        load_val
                    ), f"{key} observation model mismatch: {mapping_dict[key]} != {load_val}"
                elif key == "regularizer":
                    if isinstance(mapping_dict[key], str):
                        mapping_reg = instantiate_regularizer(mapping_dict[key])
                    else:
                        mapping_reg = mapping_dict[key]
                    assert _get_name(mapping_reg) == _get_name(
                        load_val
                    ), f"{key} regularizer mismatch: {mapping_dict[key]} != {load_val}"
                elif key == "solver_name":
                    assert (
                        mapping_dict[key] == load_val
                    ), f"{key} solver name mismatch: {mapping_dict[key]} != {load_val}"
                elif key == "regularizer_strength":
                    assert (
                        mapping_dict[key] == load_val
                    ), f"{key} regularizer strength mismatch: {mapping_dict[key]} != {load_val}"
                continue

        if isinstance(init_val, (int, float, str, type(None))):
            assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"

        elif isinstance(init_val, dict):
            assert (
                init_val == load_val
            ), f"{key} dict mismatch: {init_val} != {load_val}"

        elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
            assert np.allclose(
                np.array(init_val), np.array(load_val)
            ), f"{key} array mismatch"

        elif isinstance(init_val, Callable):
            assert _get_name(init_val) == _get_name(
                load_val
            ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"


def test_save_and_load_nested_class(nested_regularizer, tmp_path):
    """Test that save and load works with nested classes."""
    model = nmo.glm_hmm.GLMHMM(
        n_states=3, regularizer=nested_regularizer, regularizer_strength=1.0
    )
    save_path = tmp_path / "test_model.npz"
    model.save_params(save_path)

    mapping_dict = {
        "regularizer": nested_regularizer.__class__,
        "regularizer__func": jnp.exp,
    }
    with pytest.warns(UserWarning, match="The following keys have been replaced"):
        loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)

    assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
    assert isinstance(
        loaded_model.regularizer.sub_regularizer,
        nested_regularizer.sub_regularizer.__class__,
    )
    assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]

    # change mapping
    mapping_dict = {
        "regularizer": nested_regularizer.__class__,
        "regularizer__sub_regularizer": nmo.regularizer.Ridge,
        "regularizer__func": lambda x: x**2,
    }
    with pytest.warns(UserWarning, match="The following keys have been replaced"):
        loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
    assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
    assert isinstance(loaded_model.regularizer.sub_regularizer, nmo.regularizer.Ridge)
    assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Bernoulli", "simulate": True}],
    indirect=True,
)
def test_save_and_load_fitted_model(instantiate_base_regressor_subclass, tmp_path):
    """
    Test saving and loading a fitted model with various observation models and regularizers.
    Ensure all parameters are preserved.
    """
    fixture = instantiate_base_regressor_subclass
    fixture.model.coef_ = fixture.params.glm_params.coef
    fixture.model.intercept_ = fixture.params.glm_params.intercept
    fixture.model.transition_prob_ = jnp.exp(
        fixture.params.hmm_params.log_transition_prob
    )
    fixture.model.initial_prob_ = jnp.exp(fixture.params.hmm_params.log_initial_prob)
    fixture.model.scale_ = jnp.zeros_like(fixture.params.glm_params.intercept) * 11.0
    fixture.model.dof_resid_ = 1.0
    initial_params = fixture.model.get_params()
    fit_state = fixture.model._get_fit_state()
    initial_params.update(fit_state)

    # Save
    save_path = tmp_path / "test_model.npz"
    fixture.model.save_params(save_path)

    # Load
    loaded_model = nmo.load_model(save_path)
    loaded_params = loaded_model.get_params()
    fit_state = loaded_model._get_fit_state()
    fit_state.pop("solver_state_")
    initial_params.pop("solver_state_")
    loaded_params.update(fit_state)

    # Assert states are close
    for k, v in fit_state.items():
        print(f"CHECKING {k}")
        msg = f"{k} mismatch after load."
        if isinstance(v, jnp.ndarray) or isinstance(v, Number):
            assert np.allclose(initial_params[k], v), msg
        else:
            assert all(np.allclose(a, b) for a, b in zip(initial_params[k], v)), msg


@pytest.mark.parametrize(
    "X, y, expected_new_session",
    [
        (np.ones((3, 1)), np.ones((3,)), jnp.array([1, 0, 0])),
        (np.ones((3, 1)), np.array([0, np.nan, 0]), jnp.array([1, 0, 1])),
        (
            nap.TsdFrame(
                t=np.arange(3),
                d=np.zeros((3, 3)),
            ),
            nap.Tsd(
                t=np.arange(3),
                d=np.zeros((3,)),
            ),
            jnp.array([1, 0, 0]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(3),
                d=np.zeros((3, 3)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
            ),
            nap.Tsd(
                t=np.arange(3),
                d=np.zeros((3,)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
            ),
            jnp.array([1, 0, 1]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(5),
                d=np.zeros((5, 3)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(5),
                d=np.array([0, 0, np.nan, np.nan, 0]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            jnp.array([1, 0, 1, 1, 1]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 3)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.array([0, 0, np.nan, np.nan, 0, 0]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            jnp.array([1, 0, 1, 1, 1, 0]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.array([[0], [0], [np.nan], [np.nan], [0], [0]]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.zeros(6),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            jnp.array([1, 0, 1, 1, 1, 0]),
        ),
        # NaN at the very end of data
        (
            np.ones((4, 1)),
            np.array([0, 1, 2, np.nan]),
            jnp.array([1, 0, 0, 0]),
        ),
        # X and y both have NaNs at different positions
        (
            np.array([[0], [np.nan], [2], [3], [np.nan]]),
            np.array([0, 1, np.nan, 3, 4]),
            jnp.array([1, 0, 1, 1, 0]),
        ),
        # X and y both have NaNs at same position
        (
            np.array([[0], [np.nan], [2], [3]]),
            np.array([0, np.nan, 2, 3]),
            jnp.array([1, 0, 1, 0]),
        ),
        # Entire epoch is NaN (with pynapple)
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 1)),
                time_support=nap.IntervalSet([0, 2, 4], [1, 3, 5]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.array([0, 0, np.nan, np.nan, 3, 4]),
                time_support=nap.IntervalSet([0, 2, 4], [1, 3, 5]),
            ),
            jnp.array([1, 0, 1, 1, 1, 0]),
        ),
        # Multiple NaNs at the end
        (
            np.ones((5, 1)),
            np.array([0, 1, 2, np.nan, np.nan]),
            jnp.array([1, 0, 0, 0, 1]),
        ),
    ],
)
@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Poisson", "simulate": False}],
    indirect=True,
)
def test__get_is_new_session(
    X, y, expected_new_session, instantiate_base_regressor_subclass
):
    """Test that session boundaries are correctly identified from epoch starts and NaN positions.

    The GLM-HMM requires proper session segmentation to reset hidden states at discontinuities.
    This test verifies that:
    1. Epoch start times (from pynapple time_support) are marked as new sessions
    2. Positions immediately following NaN values are marked as new sessions
    3. Combined NaNs from both X and y are handled correctly

    This ensures the forward-backward algorithm can properly handle gaps in time-series data
    without incorrectly propagating state information across discontinuities.
    """
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    is_new_session = model._get_is_new_session(X, y)
    assert jnp.all(is_new_session == expected_new_session)


@pytest.mark.parametrize(
    "X, y",
    [
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 1)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.arange(6),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
        ),
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 1)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.array([0, 1, np.nan, np.nan, 2, 3]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Poisson", "simulate": False}],
    indirect=True,
)
def test__get_is_new_session_and_drop_nan(X, y, instantiate_base_regressor_subclass):
    """Test that session markers remain valid after NaN removal.

    Critical integration test verifying that the NaN handling pipeline preserves session
    boundaries correctly. When NaNs are dropped from the data, the new_session indicators
    must still point to the correct first valid sample of each epoch.

    This test ensures:
    1. Number of sessions equals number of epochs after NaN removal
    2. Session markers correctly identify the first valid value of each epoch

    This is essential because the model marks positions *after* NaNs as new sessions
    before dropping them, ensuring session boundaries survive the filtering step.
    """
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    is_new_session = model._get_is_new_session(X, y)

    _, drop_y, is_new_session = tree_utils.drop_nans(X.d, y.d, is_new_session)
    assert is_new_session.sum() == len(X.time_support)
    first_valid_per_epoch = []
    for ep in y.time_support:
        yep = np.asarray(y.get(ep.start[0], ep.end[0]))
        Xep = np.asarray(X.get(ep.start[0], ep.end[0])).reshape(yep.shape[0], -1)
        is_nan = np.isnan(yep) | np.any(np.isnan(Xep), axis=1)
        first_valid_per_epoch.append(yep[~is_nan][0])
    assert np.all(
        np.array(first_valid_per_epoch) == drop_y[is_new_session.astype(bool)]
    )
