"""Initialization functions and related utility functions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..glm import GLM, PopulationGLM
from ..glm.initialize_parameters import initialize_intercept_matching_mean_rate
from ..glm.params import GLMUserParams
from ..hmm.initialize_parameters import (
    InitFunctionHMM,
    KMeansInitializer,
    _resolve_init_funcs,
    _validate_init_funcs_kwargs,
)
from ..pytrees import FeaturePytree
from ..type_casting import cast_to_jax
from ..typing import DESIGN_INPUT_TYPE
from .algorithm_configs import has_fixed_scale

RANDOM_KEY = jax.Array


class InitFunctionGLM(Protocol):
    """Protocol for GLM parameter initialization functions (coefficients, intercept, scale)."""

    def __call__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        inverse_link_function: Callable,
        session_starts: Optional[NDArray | jnp.ndarray],
        random_key: jax.Array,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Initialize GLM parameters and scale."""
        ...


@cast_to_jax(dtype=None)
def random_glm_params_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    inverse_link_function: Callable,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: Optional[jax.Array] = None,
    std_dev=0.001,
) -> GLMUserParams:
    """
    Initialize GLM coefficients and intercept with random normal values.

    Generates random GLM parameters for each HMM state by sampling from a normal
    distribution scaled by `std_dev`.

    Parameters
    ----------
    n_states : int
        Number of HMM states.
    X : DESIGN_INPUT_TYPE
        Design matrix with shape (n_samples, n_features).
    y : jnp.ndarray
        Observations, shape (n_samples,) or (n_samples, n_neurons).
    inverse_link_function :
        Inverse link function of the GLM.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions.
        Unused for this initialization, but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key for reproducibility. Default is PRNGKey(123).
    std_dev :
        The standard deviation of the normal distribution that generates the coefficients.
        Default is 0.001.

    Returns
    -------
    coef : Any
        Coefficient, a pytree with the same structure as X with leaves
         of shape (n_features, n_neurons, n_states).
    intercept : jnp.ndarray
        Intercept array of shape (n_neurons, n_states).
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(123)
    is_one_dim = y.ndim == 1
    n_neurons = 1 if is_one_dim else y.shape[1]
    if isinstance(X, FeaturePytree):
        X = X.data
    # small random noisy coef
    coef = jax.tree_util.tree_map(
        lambda x: std_dev
        * jax.random.normal(random_key, (x.shape[1], n_neurons, n_states)),
        X,
    )
    # mean-rate
    intercept = initialize_intercept_matching_mean_rate(inverse_link_function, y)
    intercept = jnp.tile(intercept[:, jnp.newaxis], (1, n_states))
    if is_one_dim:
        coef = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), coef)
        intercept = jnp.squeeze(intercept, axis=0)
    return coef, intercept


class KMeansInitializerGLM(KMeansInitializer):
    """
    Initializer class that uses KMeans clustering to initialize HMM parameters.

    This class fits a KMeans model to the combined predictors and output data to assign states, then computes
    initial state probabilities and transition probabilities based on the assigned states. It can be used to provide
    a more informed initialization for HMM parameters based on the structure of the data.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    inverse_link_function :
        Inverse link function of the GLM.
    observation_model :
        Observation model of the GLM.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        Keyword arguments defining the GLM model: observation model, regularization etc.
    random_key :
        Random key for reproducibility of KMeans initialization.
    """

    def __init__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        inverse_link_function: Callable,
        observation_model: str,
        session_starts: Optional[jnp.ndarray] = None,
        glm_kwargs: Optional[Dict[str, Any]] = None,
        random_key: int | jax.Array = 0,
    ):
        super().__init__(
            n_states,
            X,
            y,
            session_starts,
            random_key=random_key,
        )
        self._X = jax.tree_util.tree_map(jnp.asarray, X)
        if isinstance(self._X, FeaturePytree):
            self._X = self._X.data
        self._y = jnp.asarray(y)
        self.inverse_link_function = inverse_link_function
        self.observation_model = observation_model
        self.glm_kwargs = glm_kwargs if glm_kwargs is not None else {}
        if self._y.ndim == 1:
            self._glm_models = {
                i: GLM(
                    observation_model=observation_model,
                    inverse_link_function=inverse_link_function,
                    **self.glm_kwargs,
                )
                for i in range(self.n_states)
            }
            self._is_population = False
        else:
            self._glm_models = {
                i: PopulationGLM(
                    observation_model=observation_model,
                    inverse_link_function=inverse_link_function,
                    **self.glm_kwargs,
                )
                for i in range(self.n_states)
            }
            self._is_population = True

    def fit(self) -> "KMeansInitializerGLM":
        """Fit one GLM per state on samples assigned to that state by KMeans."""
        states = self.states.astype(bool)
        for i, (state_mask, model) in enumerate(
            zip(states.T, self._glm_models.values())
        ):
            X_state, y_state = self._X[state_mask], self._y[state_mask]
            if X_state.shape[0] == 0:
                raise ValueError(
                    f"KMeans assigned 0 samples to state {i}. Try reducing n_states "
                    f"or using a different initialization method."
                )
            model.fit(X_state, y_state)
        return self

    def glm_params(self) -> GLMUserParams:
        """Generate GLM parameters for initialization."""
        if self._glm_models[0].coef_ is None:
            self.fit()
        key = jax.random.PRNGKey(self.random_key)
        states = self.states.astype(bool)
        coef, intercept = random_glm_params_init(
            states.shape[1],
            self._X,
            self._y,
            self.inverse_link_function,
            std_dev=0.0,
            random_key=key,
        )
        for i, m in enumerate(self._glm_models.values()):
            coef = jax.tree_util.tree_map(
                lambda c, mc: c.at[..., i].set(mc), coef, m.coef_
            )
            if self._is_population:
                intercept = intercept.at[..., i].set(m.intercept_)
            else:
                intercept = intercept.at[i : i + 1].set(m.intercept_)
        return coef, intercept

    def scale(self) -> jnp.ndarray:
        """KMeans-based scale estimate."""
        is_population = self._y.ndim > 1
        if is_population:
            n_neurons = self._y.shape[1]
            scale = jnp.ones((n_neurons, self.n_states))
        else:
            scale = jnp.ones((self.n_states,))

        if has_fixed_scale(self._glm_models[0].observation_model):
            return scale

        if self._glm_models[0].scale_ is None:
            self.fit()

        for i, m in enumerate(self._glm_models.values()):
            scale = scale.at[..., i].set(m.scale_)
        return scale


def kmeans_glm_params_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    observation_model: str,
    session_starts: Optional[jnp.ndarray] = None,
    glm_kwargs: Optional[dict] = None,
    random_key: Optional[jax.Array] = None,
    initializer: Optional[KMeansInitializer] = None,
):
    """
    Initialize the glm parameters using KMeans clustering.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    inverse_link_function :
        Inverse link function of the GLM.
    observation_model :
        Observation model of the GLM.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        GLM parameters as keyword arguments. These are passed at model initialization.
    random_key :
        Random key for reproducibility of KMeans initialization.
    initializer :
        Optional instance of KMeansInitializer to use for computing initial probabilities.

    Returns
    -------
    coef, intercept :
        The glm coefficients and intercept.
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(123)
    if initializer is None:
        initializer = KMeansInitializerGLM(
            n_states,
            X,
            y,
            inverse_link_function,
            observation_model,
            session_starts,
            glm_kwargs,
            random_key,
        )
    return initializer.glm_params()


def kmeans_scale_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    observation_model: str,
    session_starts: Optional[jnp.ndarray] = None,
    glm_kwargs: Optional[dict] = None,
    random_key: Optional[jax.Array] = None,
    initializer: Optional[KMeansInitializer] = None,
):
    """
    Initialize the scale parameters using KMeans clustering.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    inverse_link_function :
        Inverse link function of the GLM.
    observation_model :
        Observation model of the GLM.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        GLM parameters as keyword arguments. These are passed at model initialization.
    random_key :
        Random key for reproducibility of KMeans initialization.
    initializer :
        Optional instance of KMeansInitializer to use for computing initial probabilities.

    Returns
    -------
    scale :
        The inital scale.
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(123)
    if initializer is None:
        initializer = KMeansInitializerGLM(
            n_states,
            X,
            y,
            inverse_link_function,
            observation_model,
            session_starts,
            glm_kwargs,
            random_key,
        )
    return initializer.scale()


@cast_to_jax(dtype=None)
def constant_scale_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: Optional[jax.Array] = None,
    scale_val: float = 1.0,
):
    """
    Initialize scale to a constant value.

    Creates a scale parameter array where all elements are set to the same value.

    Parameters
    ----------
    n_states : int
        Number of HMM states.
    X : DESIGN_INPUT_TYPE
        Design matrix, unused but included for API consistency.
    y : NDArray | jnp.ndarray
        Observations, used to determine number of neurons from shape.
    inverse_link_function :
        Inverse link function of the GLM. Unused for this initialization, but included for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,). Unused for this initialization,
        but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key, unused for this initialization, but included for API consistency.
    scale_val : float
        The constant value to initialize all scale parameters to. Default is 1.0.

    Returns
    -------
    scale : jnp.ndarray
        Scale array of shape (n_states,) for single neuron or (n_neurons, n_states)
        for multiple neurons, with all values set to `scale_val`.
    """
    is_one_dim = y.ndim == 1
    n_neurons = 1 if is_one_dim else y.shape[1]
    scale = jnp.full((n_neurons, n_states), scale_val, dtype=float)
    if is_one_dim:
        scale = jnp.squeeze(scale, axis=0)
    return scale


DEFAULT_INIT_FUNCTIONS_GLMHMM = {
    "glm_params_init": random_glm_params_init,
    "glm_params_init_kwargs": {},
    "glm_params_init_custom": False,
    "scale_init": constant_scale_init,
    "scale_init_kwargs": {},
    "scale_init_custom": False,
}

AVAIL_INIT_FUNCTIONS_GLM = {
    "glm_params_init": {
        "random": random_glm_params_init,
        "kmeans": kmeans_glm_params_init,
    },
    "scale_init": {"constant": constant_scale_init, "kmeans": kmeans_scale_init},
}

GLMHMM_INITIALIZATION_FN_DICT = dict[
    Literal[
        "glm_params_init",
        "glm_params_init_kwargs",
        "glm_params_init_custom",
        "scale_init",
        "scale_init_kwargs",
        "scale_init_custom",
    ],
    InitFunctionGLM | InitFunctionHMM | dict[str, Any] | bool,
]


def setup_glm_hmm_initialization(
    glm_params_init: Optional[str | Callable] = None,
    glm_params_init_kwargs: Optional[dict] = None,
    scale_init: Optional[str | Callable] = None,
    scale_init_kwargs: Optional[dict] = None,
    init_funcs: Optional[dict | GLMHMM_INITIALIZATION_FN_DICT] = None,
) -> GLMHMM_INITIALIZATION_FN_DICT:
    """
    Set up GLM-HMM initialization functions based on user input, merging with defaults.

    This function takes user-specified initialization functions (either as strings for built-in functions or callables
    for custom functions) for both initial state probabilities and transition probabilities, and merges
    them with default initialization functions for ones that are not provided. It ensures that the provided functions
    and kwargs are valid and conform to expected signatures.
    Note that this function assumes pre-validated function keys, this is convenient because keys are model specific.

    Parameters
    ----------
    glm_params_init:
        Initialization function for the GLM coefficient and intercept.
    glm_params_init_kwargs:
        Keyword arguments to pass to the GLM coefficient and intercept initialization function.
    scale_init:
        Initialization function for the scale of the observation model.
    scale_init_kwargs:
        Kwargs of the initialization function for the scale of the observation model.
    init_funcs :
        Existing dictionary of initialization functions to update. If None, a fresh copy of
        ``DEFAULT_INIT_FUNCTIONS_GLMHMM`` is used. Keys must already be validated before calling
        this function; use the class ``initialization_funcs`` setter for key validation.

    Returns
    -------
    init_funcs :
        Updated dictionary of initialization functions based on provided inputs.
    """

    if init_funcs is None:
        glm_init_funcs = DEFAULT_INIT_FUNCTIONS_GLMHMM.copy()
    else:
        glm_init_funcs = init_funcs.copy()

    # update functions and kwargs for glm params and scale
    # if a function is passed but not kwargs, kwargs will be reset
    # if kwargs are passed but not function, kwargs will be validated against existing function in init_funcs
    if glm_params_init is not None:
        (
            glm_init_funcs["glm_params_init"],
            glm_init_funcs["glm_params_init_kwargs"],
            glm_init_funcs["glm_params_init_custom"],
        ) = _resolve_init_funcs(
            "glm_params_init",
            glm_params_init,
            glm_params_init_kwargs,
            AVAIL_INIT_FUNCTIONS_GLM,
            protocol=InitFunctionGLM,
        )
    elif glm_params_init_kwargs is not None:
        glm_init_funcs["glm_params_init_kwargs"] = _validate_init_funcs_kwargs(
            glm_init_funcs["glm_params_init"],
            glm_params_init_kwargs,
            protocol=InitFunctionGLM,
        )

    if scale_init is not None:
        (
            glm_init_funcs["scale_init"],
            glm_init_funcs["scale_init_kwargs"],
            glm_init_funcs["scale_init_custom"],
        ) = _resolve_init_funcs(
            "scale_init",
            scale_init,
            scale_init_kwargs,
            AVAIL_INIT_FUNCTIONS_GLM,
            protocol=InitFunctionGLM,
        )
    elif scale_init_kwargs is not None:
        glm_init_funcs["scale_init_kwargs"] = _validate_init_funcs_kwargs(
            glm_init_funcs["scale_init"],
            scale_init_kwargs,
            protocol=InitFunctionGLM,
        )
    return glm_init_funcs


def generate_glm_hmm_initial_model_params(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: int | jax.Array = 123,
    init_funcs: Optional[dict] = None,
) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """
    Generate initial GLM model parameters for a GLM-HMM.

    Calls the specified initialization functions for GLM coefficients, intercept, and scale,
    splitting the random key to ensure independent random states for each component.
    HMM parameters (initial and transition probabilities) are initialized separately via
    :meth:`~nemos.hmm.BaseHMM._hmm_params_initialization`.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,) or (n_samples, n_neurons).
    inverse_link_function :
        Inverse link function of the GLM, passed to GLM parameter and scale initialization functions.
    session_starts :
        Boolean array of shape (n_samples,) indicating the start of new sessions. If None, a single
        session is assumed.
    random_key :
        Key for random number generation. Split into two independent subkeys for GLM params
        and scale respectively.
    init_funcs :
        Dictionary containing initialization functions and their kwargs for GLM parameters.
        Can be constructed with :func:`setup_glm_hmm_initialization`. GLM-specific keys missing
        from the dict fall back to ``GLM_INIT_FUNCS`` defaults.

    Returns
    -------
    coef :
        GLM coefficients, a pytree matching the structure of X with leaves of shape
        ``(n_features, n_neurons, n_states)`` or ``(n_features, n_states)`` for single-neuron.
    intercept :
        Intercept array of shape ``(n_neurons, n_states)`` or ``(n_states,)`` for single-neuron.
    scale :
        Scale array of shape ``(n_states,)`` for single-neuron or ``(n_neurons, n_states)`` for population.

    See Also
    --------
    :func:`nemos.glm_hmm.initialize_parameters.setup_glm_hmm_initialization`
        Function to set up the initialization functions and their kwargs based on user input.
    """
    glm_init_funcs = {} if init_funcs is None else init_funcs.copy()
    if isinstance(random_key, int):
        random_key = jax.random.PRNGKey(random_key)

    glm_params_key, scale_key = jax.random.split(random_key, 2)

    glm_params_init = (
        glm_init_funcs.get("glm_params_init")
        or DEFAULT_INIT_FUNCTIONS_GLMHMM["glm_params_init"]
    )
    glm_params_init_kwargs = glm_init_funcs.get("glm_params_init_kwargs") or {}

    if (
        glm_params_init is kmeans_glm_params_init
        and "observation_model" not in glm_params_init_kwargs
    ):
        raise ValueError(
            "The 'kmeans' GLM parameter initializer requires 'observation_model' to be "
            "provided in 'glm_params_init_kwargs'."
        )

    coef, intercept = glm_params_init(
        n_states=n_states,
        X=X,
        y=y,
        inverse_link_function=inverse_link_function,
        session_starts=session_starts,
        random_key=glm_params_key,
        **glm_params_init_kwargs,
    )

    scale_init_fn = (
        glm_init_funcs.get("scale_init") or DEFAULT_INIT_FUNCTIONS_GLMHMM["scale_init"]
    )
    scale_init_kwargs = glm_init_funcs.get("scale_init_kwargs") or {}

    if (
        scale_init_fn is kmeans_scale_init
        and "observation_model" not in scale_init_kwargs
    ):
        raise ValueError(
            "The 'kmeans' scale initializer requires 'observation_model' to be "
            "provided in 'scale_init_kwargs'."
        )

    scale = scale_init_fn(
        n_states=n_states,
        X=X,
        y=y,
        inverse_link_function=inverse_link_function,
        session_starts=session_starts,
        random_key=scale_key,
        **scale_init_kwargs,
    )

    return coef, intercept, scale
