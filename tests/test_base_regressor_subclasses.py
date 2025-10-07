import inspect
import itertools
import warnings
from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp
import scipy.stats as sts
import statsmodels.api as sm
from numba import njit

import nemos as nmo
from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS

MODEL_REGISTRY = {
    "GLM": nmo.glm.GLM,
    "PopulationGLM": nmo.glm.PopulationGLM,
    "GLMHMM": nmo.glm_hmm.GLMHMM,
}

HARD_CODED_GET_PARAMS_KEYS = {
    "GLMHMM": {
        "dirichlet_prior_alphas_init_prob",
        "dirichlet_prior_alphas_transition",
        "initialize_glm_params",
        "initialize_init_proba",
        "initialize_transition_proba",
        "inverse_link_function",
        "maxiter",
        "n_states",
        "observation_model",
        "regularizer",
        "regularizer_strength",
        "seed",
        "solver_kwargs",
        "solver_name",
        "tol",
    },
    "GLM": {
        "inverse_link_function",
        "observation_model",
        "regularizer",
        "regularizer_strength",
        "solver_kwargs",
        "solver_name",
    },
    "PopulationGLM": {
        "inverse_link_function",
        "observation_model",
        "regularizer",
        "regularizer_strength",
        "solver_kwargs",
        "solver_name",
        "feature_mask",
    },
}

# as of now, all models are glm type... in the future this may change.
MODEL_WITH_LINK_FUNCTION_REGISTRY = {
    "GLM": nmo.glm.GLM,
    "PopulationGLM": nmo.glm.PopulationGLM,
    "GLMHMM": nmo.glm_hmm.GLMHMM,
}

DEFAULTS = {"GLMHMM": dict(n_states=3), "GLM": dict(), "PopulationGLM": dict()}


INSTANTIATE_MODEL_ONLY = [
    {"model": m, "obs_model": o, "simulate": False}
    for m, o in itertools.product(MODEL_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS)
]

INSTANTIATE_MODEL_AND_SIMULATE = [
    {"model": m, "obs_model": o, "simulate": True}
    for m, o in itertools.product(MODEL_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS)
]

INSTANTIATE_MODEL_ONLY_LINK = [
    {"model": m, "obs_model": o, "simulate": False}
    for m, o in itertools.product(
        MODEL_WITH_LINK_FUNCTION_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS
    )
]

INSTANTIATE_MODEL_AND_SIMULATE_LINK = [
    {"model": m, "obs_model": o, "simulate": True}
    for m, o in itertools.product(
        MODEL_WITH_LINK_FUNCTION_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS
    )
]


def test_all_defaults_assigned():
    PARS_WITHOUT_DEFAULTS = {}
    for k, cls in MODEL_REGISTRY.items():
        PARS_WITHOUT_DEFAULTS[k] = {
            k: v
            for k, v in inspect.signature(cls.__init__).parameters.items()
            if v.default == inspect.Parameter.empty
        }

        PARS_WITHOUT_DEFAULTS[k].pop("self")
        PARS_WITHOUT_DEFAULTS[k].pop("kwargs", None)

    for k, pars_without_default in PARS_WITHOUT_DEFAULTS.items():
        if DEFAULTS[k].keys() != pars_without_default.keys():
            raise ValueError(
                f"The model {k} has the following required parameters:\n"
                f"{list(pars_without_default.keys())}\n"
                f"But the provided defaults are:\n{list(DEFAULTS[k].keys())}.\n"
                "Therefore, the following parameters do not have a default:\n"
                f"{set(pars_without_default.keys()).difference(DEFAULTS[k].keys())}\n"
                f"Please update the ``DEFAULTS`` dictionary for model {k}.\n"
            )


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
class TestModelCommons:
    """
    Unit tests for methods common to all BaseRegressor subclasses.
    """

    #######################
    # Test model.__init__
    #######################
    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            # test solver at initialization, where test_regularizers.py tests solvers with set_params
            (None, does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            ("LBFGS", does_not_raise()),
            ("NonlinearCG", does_not_raise()),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The solver: 1 is not allowed "),
            ),
        ],
    )
    def test_init_solver_type(
        self, solver_name, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        model = instantiate_base_regressor_subclass[2].__class__
        pars = DEFAULTS[model.__name__].copy()
        pars.update(dict(solver_name=solver_name))
        with expectation:
            model(**pars)

    @pytest.mark.parametrize(
        "regularizer, expectation",
        [
            # regularizer with class objects are tested in test_regularizers.py
            # so here we only test the string input names and None type
            (None, does_not_raise()),
            ("UnRegularized", does_not_raise()),
            ("Ridge", does_not_raise()),
            ("Lasso", does_not_raise()),
            ("ElasticNet", does_not_raise()),
            ("GroupLasso", does_not_raise()),
            (
                nmo.regularizer.Ridge,
                pytest.raises(
                    TypeError, match="The regularizer should be either a string from "
                ),
            ),
        ],
    )
    def test_init_regularizer_type(
        self, regularizer, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test initialization with different regularizer types.
        Test that an error is raised if a non-compatible regularizer is passed.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Unused parameter `regularizer_strength`.*",
            )
            model_cls = instantiate_base_regressor_subclass[2].__class__
            with expectation:
                model_cls(
                    **DEFAULTS[model_cls.__name__],
                    regularizer=regularizer,
                    regularizer_strength=1,
                )

    def test_get_params(self, instantiate_base_regressor_subclass):
        """
        Test that get_params() contains expected values.
        """
        model_cls = instantiate_base_regressor_subclass[2].__class__

        expected_keys = HARD_CODED_GET_PARAMS_KEYS[model_cls.__name__]
        model = model_cls(**DEFAULTS[model_cls.__name__])
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())

        # passing params
        model = model_cls(
            **DEFAULTS[model_cls.__name__],
            solver_name="LBFGS",
            regularizer="UnRegularized",
        )
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())

        # changing regularizer
        model.set_params(
            **DEFAULTS[model_cls.__name__],
            regularizer="Ridge",
            regularizer_strength=1.0,
        )
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY_LINK,
    indirect=True,
)
class TestLinkFunctionModels:
    def test_non_differentiable_inverse_link(self, instantiate_base_regressor_subclass):
        model = instantiate_base_regressor_subclass[2]

        # define a jax non-diff function
        non_diff = lambda y: jnp.asarray(njit(lambda x: x)(np.atleast_1d(y)))

        with pytest.raises(
            ValueError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            model.inverse_link_function = non_diff
        with pytest.raises(
            ValueError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            model.__class__(
                **DEFAULTS[model.__class__.__name__], inverse_link_function=non_diff
            )

    @pytest.mark.parametrize(
        "link_function",
        [
            jnp.exp,
            lambda x: jnp.exp(x) if isinstance(x, jnp.ndarray) else "not a number",
        ],
    )
    def test_initialization_link_returns_scalar(
        self,
        link_function,
        instantiate_base_regressor_subclass,
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.0), (jnp.ndarray, float))
        model = instantiate_base_regressor_subclass[2]

        if raise_exception:
            with pytest.raises(
                ValueError,
                match="The `inverse_link_function` must handle scalar inputs correctly",
            ):
                model.set_params(
                    **DEFAULTS[model.__class__.__name__],
                    inverse_link_function=link_function,
                )
        else:
            model.set_params(
                **DEFAULTS[model.__class__.__name__],
                inverse_link_function=link_function,
            )

    @pytest.mark.parametrize(
        "link_function",
        [jnp.exp, np.exp, lambda x: 1 / x, sm.families.links.Log()],
    )
    def test_initialization_link_is_jax(
        self,
        link_function,
        instantiate_base_regressor_subclass,
    ):
        """Check that the observation model initializes when a callable is passed."""
        model = instantiate_base_regressor_subclass[2]

        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        if raise_exception:
            with pytest.raises(
                ValueError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray",
            ):
                model.__class__(
                    **DEFAULTS[model.__class__.__name__],
                    inverse_link_function=link_function,
                )
        else:
            model.__class__(
                **DEFAULTS[model.__class__.__name__],
                inverse_link_function=link_function,
            )

    @pytest.mark.parametrize(
        "link_function, expectation",
        [
            (jax.scipy.special.expit, does_not_raise()),
            (
                sp.special.expit,
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
            (jax.scipy.stats.norm.cdf, does_not_raise()),
            (
                sts.norm.cdf,
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
            (
                np.exp,
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
            (lambda x: x, does_not_raise()),
            (
                sm.families.links.Log(),
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
        ],
    )
    def test_initialization_link_is_jax_set_params(
        self, link_function, instantiate_base_regressor_subclass, expectation
    ):
        model_cls = instantiate_base_regressor_subclass[2].__class__

        with expectation:
            model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                inverse_link_function=link_function
            )

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable(
        self, link_function, instantiate_base_regressor_subclass
    ):
        """Check that the observation model initializes when a callable is passed."""
        model_cls = instantiate_base_regressor_subclass[2].__class__
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                model_cls(
                    **DEFAULTS[model_cls.__name__], inverse_link_function=link_function
                )
        else:
            model_cls(
                **DEFAULTS[model_cls.__name__], inverse_link_function=link_function
            )

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable_set_params(
        self, link_function, instantiate_base_regressor_subclass
    ):
        """Check that the observation model initializes when a callable is passed."""
        model_cls = instantiate_base_regressor_subclass[2].__class__
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                    inverse_link_function=link_function
                )
        else:
            model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                inverse_link_function=link_function
            )


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [
        {"model": m, "obs_model": "Poisson", "simulate": False}
        for m in MODEL_REGISTRY.keys()
    ],
    indirect=True,
)
class TestObservationModel:
    @pytest.mark.parametrize(
        "observation, expectation",
        [
            (nmo.observation_models.PoissonObservations(), does_not_raise()),
            (nmo.observation_models.GammaObservations(), does_not_raise()),
            (nmo.observation_models.BernoulliObservations(), does_not_raise()),
            (nmo.observation_models.NegativeBinomialObservations(), does_not_raise()),
            (
                nmo.regularizer.Regularizer,
                pytest.raises(
                    AttributeError,
                    match="The provided object does not have the required",
                ),
            ),
            (
                1,
                pytest.raises(
                    AttributeError,
                    match="The provided object does not have the required",
                ),
            ),
        ],
    )
    def test_init_observation_type(
        self, observation, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test initialization with different observation models. Check if an appropriate exception is raised
        when the observation model does not have the required attributes.
        """
        model_cls = instantiate_base_regressor_subclass[2].__class__
        with expectation:
            model_cls(**DEFAULTS[model_cls.__name__], observation_model=observation)
