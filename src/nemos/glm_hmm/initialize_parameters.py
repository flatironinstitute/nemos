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
    DEFAULT_INIT_FUNCTIONS,
    InitFunctionHMM,
    KMeansInitializer,
    _resolve_init_funcs,
    _validate_init_funcs_keys,
    _validate_init_funcs_kwargs,
    generate_hmm_initial_params,
    setup_hmm_initialization,
)
from ..type_casting import cast_to_jax
from ..typing import DESIGN_INPUT_TYPE
from .algorithm_configs import (
    has_fixed_scale,
)

RANDOM_KEY = jax.Array


class InitFunctionGLM(Protocol):
    """Protocol for HMM probability initialization functions (initial and transition)."""

    def __call__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        inverse_link_function: Callable,
        random_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Initialize HMM probabilities."""
        ...


@cast_to_jax(dtype=None)
def random_glm_params_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    inverse_link_function: Callable,
    random_key=jax.random.PRNGKey(123),
    std_dev=0.001,
) -> GLMUserParams:
    """
    Initialize GLM coefficients and intercept with random normal values.

    Generates random GLM parameters for each HMM state by sampling from a normal
    distribution scaled by 0.1.

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
    random_key : jax.random.PRNGKey
        Random key for reproducibility. Default is PRNGKey(123).
    std_dev :
        The standard deviation of the normal distribution that generates the coefficients.
        Default is 0.001.

    Returns
    -------
    coef : jnp.ndarray
        Coefficient matrix of shape (n_features, n_neurons, n_states).
    intercept : jnp.ndarray
        Intercept array of shape (n_neurons, n_states).
    """
    n_features = X.shape[1]
    is_one_dim = y.ndim == 1
    n_neurons = 1 if is_one_dim else y.shape[1]

    # small random noisy coef
    coef = std_dev * jax.random.normal(random_key, (n_features, n_neurons, n_states))
    # mean-rate
    intercept = initialize_intercept_matching_mean_rate(inverse_link_function, y)
    intercept = jnp.tile(intercept[:, jnp.newaxis], (1, n_states))
    if is_one_dim:
        coef = jnp.squeeze(coef, axis=1)
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
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        Keyword arguments defining the GLM model: observation model, regularization etc.
    minimum_prob :
        Minimum probability added to each state to avoid zero probabilities.
        Note that probabilities will be renormalized after adding this minimum value, so the final
        probabilities will not be exactly this value.
    random_key :
        Random key for reproducibility of KMeans initialization.
    """

    def __init__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        inverse_link_function: Callable,
        is_new_session: Optional[jnp.ndarray] = None,
        glm_kwargs: Optional[Dict[str, Any]] = None,
        minimum_prob: float = 0.02,
        random_key: int | jax.Array = 0,
    ):
        super().__init__(
            n_states,
            X,
            y,
            is_new_session,
            minimum_prob=minimum_prob,
            random_key=random_key,
        )
        self._X = jnp.asarray(X)
        self._y = jnp.asarray(y)
        self.inverse_link_function = inverse_link_function
        self.glm_kwargs = glm_kwargs if glm_kwargs is not None else {}
        if self._y.ndim == 1:
            self._glm_models = {i: GLM(**self.glm_kwargs) for i in range(self.n_states)}
        else:
            self._glm_models = {
                i: PopulationGLM(**self.glm_kwargs) for i in range(self.n_states)
            }

    def glm_params(self) -> GLMUserParams:
        """Generate glm parameters for initialization."""
        if isinstance(self.random_key, int):
            key = jax.random.PRNGKey(self.random_key)
        sub, _ = jax.random.split(key)
        states = self.states.astype(bool)
        coef, intercept = random_glm_params_init(
            states.shape[1],
            self._X,
            self._y,
            self.inverse_link_function,
            random_key=sub,
            std_dev=0.0,
        )
        # initialize
        for i, state_mask in enumerate(states.T):
            model = self._glm_models[i]
            X_state, y_state = self._X[state_mask], self._y[state_mask]
            model.fit(X_state, y_state)
            coef = coef.at[:, i].set(model.coef_)
            intercept = intercept.at[i : i + 1].set(model.intercept_)

        return coef, intercept

    def scale(self) -> jnp.ndarray:
        """KMeans-based scale estimate."""
        is_population = self._y.ndim > 1
        # initialize scale
        if is_population:
            n_neurons = self._y.shape[1]
            scale = jnp.ones((self.n_states, n_neurons))
        else:
            scale = jnp.ones((self.n_states,))

        # return fast for fixed scale
        if has_fixed_scale(self._glm_models[0].observation_model):
            return scale

        # if not fitted, run a fit
        if self._glm_models[0].scale_ is None:
            self.glm_params()
        for i, m in enumerate(self._glm_models.values()):
            scale = scale.at[i].set(m.scale_)
        return scale


def kmeans_glm_params_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    is_new_session: Optional[jnp.ndarray] = None,
    glm_kwargs: Optional[dict] = None,
    minimum_prob: float = 0.02,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
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
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        GLM parameters as keyword arguments. These are passed at model initialization.
    minimum_prob :
        Minimum probability added to each state to avoid zero probabilities.
        Note that probabilities will be renormalized after adding this minimum value, so the final
        probabilities will not be exactly this value.
    random_key :
        Random key for reproducibility of KMeans initialization.
    initializer :
        Optional instance of KMeansInitializer to use for computing initial probabilities.

    Returns
    -------
    coef, intercept :
        The glm coefficients and intercept.
    """
    if initializer is None:
        initializer = KMeansInitializerGLM(
            n_states,
            X,
            y,
            inverse_link_function,
            is_new_session,
            glm_kwargs,
            minimum_prob,
            random_key,
        )
    return initializer.glm_params()


def kmeans_scale_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    is_new_session: Optional[jnp.ndarray] = None,
    glm_kwargs: Optional[dict] = None,
    minimum_prob: float = 0.02,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
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
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        GLM parameters as keyword arguments. These are passed at model initialization.
    minimum_prob :
        Minimum probability added to each state to avoid zero probabilities.
        Note that probabilities will be renormalized after adding this minimum value, so the final
        probabilities will not be exactly this value.
    random_key :
        Random key for reproducibility of KMeans initialization.
    initializer :
        Optional instance of KMeansInitializer to use for computing initial probabilities.

    Returns
    -------
    scale :
        The inital scale.
    """
    if initializer is None:
        initializer = KMeansInitializerGLM(
            n_states,
            X,
            y,
            inverse_link_function,
            is_new_session,
            glm_kwargs,
            minimum_prob,
            random_key,
        )
    return initializer.scale()


@cast_to_jax(dtype=None)
def constant_scale_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    random_key=jax.random.PRNGKey(124),
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


GLM_INIT_FUCS = {
    "glm_params_init": random_glm_params_init,
    "glm_params_init_kwargs": {},
    "scale_init": constant_scale_init,
    "scale_init_kwargs": {},
}
DEFAULT_INIT_FUNCTIONS_GLMHMM = DEFAULT_INIT_FUNCTIONS.copy()
DEFAULT_INIT_FUNCTIONS_GLMHMM.update(GLM_INIT_FUCS)

AVAIL_INIT_FUNCTIONS_GLM = {
    "glm_params_init": {
        "random": random_glm_params_init,
        "kmeans": kmeans_glm_params_init,
    },
    "glm_scale_init": {"constant": constant_scale_init, "kmeans": kmeans_scale_init},
}

INITIALIZATION_FN_DICT = dict[
    Literal[
        "initial_proba_init",
        "initial_proba_init_kwargs",
        "initial_proba_init_custom",
        "transition_proba_init",
        "transition_proba_init_kwargs",
        "transition_proba_init_custom",
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
    initial_proba_init: Optional[str | Callable] = None,
    initial_proba_init_kwargs: Optional[dict] = None,
    transition_proba_init: Optional[str | Callable] = None,
    transition_proba_init_kwargs: Optional[dict] = None,
    glm_params_init: Optional[str | Callable] = None,
    glm_params_init_kwargs: Optional[dict] = None,
    scale_init: Optional[str | Callable] = None,
    scale_init_kwargs: Optional[dict] = None,
    init_funcs: Optional[dict | INITIALIZATION_FN_DICT] = None,
) -> INITIALIZATION_FN_DICT:
    """
    Set up HMM initialization functions based on user input, merging with defaults.

    This function takes user-specified initialization functions (either as strings for built-in functions or callables
    for custom functions) for both initial state probabilities and transition probabilities, validates them, and merges
    them with default initialization functions for ones that are not provided. It ensures that the provided functions
    and kwargs are valid and conform to expected signatures.

    Parameters
    ----------
    initial_proba_init :
        User-specified initialization function for initial state probabilities. Can be a string key for built-in
        functions or a custom callable. If None, the default build-in function will be used.
    initial_proba_init_kwargs :
        Keyword arguments to pass to the initial state probability initialization function.
    transition_proba_init :
        User-specified initialization function for transition probabilities. Can be a string key for built-in functions
        or a custom callable. If None, the default built-in function will be used.
    transition_proba_init_kwargs :
        Keyword arguments to pass to the transition probability initialization function.
    glm_params_init:
        Initialization function for the GLM coefficient and intercept.
    glm_params_init_kwargs:
        IKeyword arguments to pass to the GLM coefficient and intercept initialization function.
    scale_init:
        Initialization function for the scale of the observation model.
    scale_init_kwargs:
        Kwargs of the initialization function for the scale of the observation model.
    init_funcs :
        Existing dictionary of initialization functions to update. If None, a new dictionary will be created.
        If the dictionary is missing any keys, they will be backfilled with defaults.

    Returns
    -------
    init_funcs :
        Updated or initialized dictionary based on provided inputs.
    """

    if init_funcs is None:
        init_funcs = DEFAULT_INIT_FUNCTIONS_GLMHMM.copy()
        glm_init_funcs = GLM_INIT_FUCS.copy()
    else:

        glm_init_funcs = {k: v for k, v in init_funcs.items() if k in GLM_INIT_FUCS}
        # check for unexpected/unknown keys in init_funcs and backfill with defaults
        # note that the hmm init function falidation will be done in the setup
        init_funcs = _validate_init_funcs_keys(init_funcs, glm_init_funcs)

    hmm_init_funcs = {k: v for k, v in init_funcs.items() if k not in GLM_INIT_FUCS}
    hmm_init_funcs = setup_hmm_initialization(
        initial_proba_init,
        initial_proba_init_kwargs,
        transition_proba_init,
        transition_proba_init_kwargs,
        init_funcs=hmm_init_funcs,
    )
    # update functions and kwargs for init prob and transition prob
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
            available_init_funcs=AVAIL_INIT_FUNCTIONS_GLM,
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
            available_init_funcs=AVAIL_INIT_FUNCTIONS_GLM,
        )
    elif scale_init_kwargs is not None:
        glm_init_funcs["scale_init_kwargs"] = _validate_init_funcs_kwargs(
            glm_init_funcs["scale_init"],
            scale_init_kwargs,
            protocol=InitFunctionGLM,
        )
    glm_init_funcs.update(hmm_init_funcs)
    return glm_init_funcs


def generate_glm_hmm_initial_params(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    random_key: int | jax.Array = 123,
    init_funcs: Optional[dict] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate initial HMM parameters using the provided initialization functions.

    This function calls the specified initialization functions for initial state probabilities
    stored in the `init_funcs` dictionary passing `n_states` and any additional stored kwargs.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    random_key :
        Optional key for random number generation, if needed by the initialization functions. The key is split to
        ensure different random states for initial probabilities and transition probabilities.
    init_funcs :
        Dictionary containing the initialization functions and their kwargs for both initial state probabilities
        and transition probabilities. This dictionary can be set up using the `setup_hmm_initialization` function.
        If not provided, or if specific functions are missing, defaults will be used.

    Returns
    -------
    initial_probs :
        Initial state probability vector of shape (n_states,) that sums to 1.
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states) where each row sums to 1.

    See Also
    --------
    :func:`nemos.hmm.setup_hmm_initialization`
        Function to set up the initialization functions and their kwargs based on user input.
    """
    # check for unexpected/unknown keys in init_funcs if user provided dictionary not made by setup_hmm_initialization
    init_funcs = {} if init_funcs is None else init_funcs
    glm_init_funcs = {k: v for k, v in init_funcs.items() if k in GLM_INIT_FUCS}
    hmm_init_funcs = {k: v for k, v in init_funcs.items() if k not in GLM_INIT_FUCS}

    if isinstance(random_key, int):
        random_key = jax.random.PRNGKey(random_key)

    # glm, scale, hmm
    glm_params_key, scale_key, hmm_key = jax.random.split(random_key, 3)

    # validate glm specific
    glm_init_funcs = _validate_init_funcs_keys(
        glm_init_funcs, default_init_funcs=GLM_INIT_FUCS
    )

    # grab glm initialization function
    glm_params_init = (
        glm_init_funcs["glm_params_init"] or GLM_INIT_FUCS["glm_params_init"]
    )

    # set to empty kwargs if set to None
    glm_params_init_kwargs = glm_init_funcs["glm_params_init_kwargs"] or {}

    # split seed into a new key
    # compute probabilities and validate if custom functions are used
    coef, intercept = glm_params_init(
        n_states=n_states,
        X=X,
        y=y,
        inverse_link_function=inverse_link_function,
        random_key=glm_params_key,
        **glm_params_init_kwargs,
    )

    initial_probs, transition_matrix = generate_hmm_initial_params(
        n_states,
        X,
        y,
        hmm_key,
        hmm_init_funcs,
    )

    return coef, intercept, initial_probs, transition_matrix
