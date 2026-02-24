"""Initialization functions and related utility functions."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..glm.initialize_parameters import initialize_intercept_matching_mean_rate
from ..glm.params import GLMUserParams
from ..type_casting import is_numpy_array_like
from ..typing import DESIGN_INPUT_TYPE
from .params import GLMHMMUserParams

RANDOM_KEY = jax.Array


class InitFunctionHMM(Protocol):
    """Protocol for HMM probability initialization functions (initial and transition)."""

    def __call__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        random_key: RANDOM_KEY,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize HMM probabilities."""
        ...


class InitFuncGLMParams(Protocol):
    """Protocol for GLM parameters (coef, intercept) initialization functions."""

    def __call__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        inverse_link_function: Callable,
        random_key: RANDOM_KEY,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize GLM coefficients and intercept."""
        ...


class InitFuncGLMScale(Protocol):
    """Protocol for GLM scale initialization functions."""

    def __call__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        random_key: RANDOM_KEY,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Initialize GLM scale parameter."""
        ...


INITIALIZATION_FN_DICT = dict[
    Literal[
        "glm_params_init", "scale_init", "initial_proba_init", "transition_proba_init"
    ],
    InitFunctionHMM | InitFuncGLMScale | InitFuncGLMParams,
]


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


def sticky_transition_proba_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    random_key: jax.numpy.ndarray = jax.random.PRNGKey(123),
    prob_stay=0.95,
) -> jnp.ndarray:
    """
    Initialize transition probabilities with sticky dynamics.

    Creates a transition probability matrix that favors staying in the current state.
    The diagonal entries (probability of staying in current state) are set to `prob_stay`,
    while off-diagonal entries (probability of transitioning to other states) are set to
    (1 - prob_stay) / (n_states - 1).

    Parameters
    ----------
    n_states : int
        Number of HMM states. Must be greater than 1.
    X : DESIGN_INPUT_TYPE
        Design matrix, unused but included for API consistency.
    y : NDArray | jnp.ndarray
        Observations, unused but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key, unused for this particular initialization, but added for API consistency.
    prob_stay : float
        Probability of staying in the current state. Default is 0.95.

    Returns
    -------
    transition_matrix : jnp.ndarray
        Transition probability matrix of shape (n_states, n_states).
    """
    # assume n_state is > 1
    if n_states == 1:
        prob_leave = 0.0
    else:
        prob_leave = (1 - prob_stay) / (n_states - 1)
    return jnp.full((n_states, n_states), prob_leave) + jnp.diag(
        (prob_stay - prob_leave) * jnp.ones(n_states)
    )


def uniform_transition_proba_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    random_key: jax.numpy.ndarray = jax.random.PRNGKey(123),
) -> jnp.ndarray:
    """
    Initialize transition probabilities with uniform dynamics.

    Creates a transition probability matrix that assign equal probability of
    transitioning to any of the states.

    Parameters
    ----------
    n_states : int
        Number of HMM states. Must be greater than 1.
    X : DESIGN_INPUT_TYPE
        Design matrix, unused but included for API consistency.
    y : NDArray | jnp.ndarray
        Observations, unused but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key, unused for this particular initialization, but added for API consistency.

    Returns
    -------
    transition_matrix : jnp.ndarray
        Transition probability matrix of shape (n_states, n_states).
    """
    prob_transition = 1.0 / n_states
    return jnp.full((n_states, n_states), prob_transition, dtype=float)


def uniform_initial_proba_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    random_key=jax.random.PRNGKey(124),
) -> jnp.ndarray:
    """
    Initialize initial state probabilities from a uniform distribution.

    Generates random initial state probabilities by sampling from a uniform
    distribution and normalizing to ensure they sum to 1.

    Parameters
    ----------
    n_states : int
        Number of HMM states.
    X : DESIGN_INPUT_TYPE
        Design matrix, unused but included for API consistency.
    y : NDArray | jnp.ndarray
        Observations, unused but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key for reproducibility. Default is PRNGKey(124).

    Returns
    -------
    initial_probs : jnp.ndarray
        Initial state probability vector of shape (n_states,) that sums to 1.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nemos.glm_hmm.initialize_parameters import uniform_initial_proba_init
    >>>
    >>> # Generate initial probabilities for 3 states
    >>> n_states = 3
    >>> X_dummy, y_dummy = jnp.ones((3, 2)), jnp.ones(3)
    >>> init_probs = uniform_initial_proba_init(n_states, X_dummy, y_dummy)
    >>> init_probs.shape
    (3,)
    >>> jnp.isclose(jnp.sum(init_probs), 1.0)
    Array(True, dtype=bool)
    """
    prob = jnp.ones((n_states,), dtype=float)
    return prob / jnp.sum(prob)


AVAILABLE_INIT_FUNCTIONS = {
    "glm_params_init": {
        "random": random_glm_params_init,
    },
    "scale_init": {
        "constant": constant_scale_init,
    },
    "transition_proba_init": {
        "sticky": sticky_transition_proba_init,
        "uniform": uniform_transition_proba_init,
    },
    "initial_proba_init": {
        "uniform": uniform_initial_proba_init,
    },
}

_IO_AVAILABLE_INIT_FUNCTIONS = AVAILABLE_INIT_FUNCTIONS.copy()
_IO_AVAILABLE_INIT_FUNCTIONS["glm_params_init"].update(
    {
        "nemos.glm_hmm.initialize_parameters.random_glm_params_init": random_glm_params_init
    }
)
_IO_AVAILABLE_INIT_FUNCTIONS["scale_init"].update(
    {"nemos.glm_hmm.initialize_parameters.constant_scale_init": constant_scale_init}
)
_IO_AVAILABLE_INIT_FUNCTIONS["transition_proba_init"].update(
    {
        "nemos.glm_hmm.initialize_parameters.sticky_transition_proba_init": sticky_transition_proba_init
    }
)
_IO_AVAILABLE_INIT_FUNCTIONS["initial_proba_init"].update(
    {
        "nemos.glm_hmm.initialize_parameters.uniform_initial_proba_init": uniform_initial_proba_init
    }
)

DEFAULT_INIT_FUNCTION: INITIALIZATION_FN_DICT = {
    "glm_params_init": random_glm_params_init,
    "scale_init": constant_scale_init,
    "transition_proba_init": sticky_transition_proba_init,
    "initial_proba_init": uniform_initial_proba_init,
}


def glm_hmm_initialization(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    random_key=jax.random.PRNGKey(123),
    initialization_funcs: Optional[dict] = None,
    initialization_kwargs: Optional[dict] = None,
) -> GLMHMMUserParams:
    """
    Initialize all GLM-HMM parameters.

    Coordinates initialization of GLM parameters (coefficients and intercept) and
    HMM parameters (initial state probabilities and transition matrix) using the
    specified initialization functions.

    Parameters
    ----------
    n_states : int
        Number of HMM states.
    X : DESIGN_INPUT_TYPE
        Design matrix with shape (n_samples, n_features).
    y : NDArray | jnp.ndarray
        Observations, shape (n_samples,) or (n_samples, n_units).
    inverse_link_function:
        The inverse link function of the GLM.
    random_key : jax.random.PRNGKey
        Random key for reproducibility. Default is PRNGKey(123).
    initialization_funcs : dict, optional
        Dictionary mapping parameter names to initialization functions. Valid keys are:
        - 'glm_params_init': Function to initialize GLM coefficients and intercept
        - 'initial_proba_init': Function to initialize initial state probabilities
        - 'transition_proba_init': Function to initialize transition probabilities
        If None, uses DEFAULT_INIT_FUNCTION. If partial dict, missing keys use defaults.
    initialization_kwargs : dict, optional
        The initialization function keyword arguments. Keys must be a subset of those of
        `initialization_funcs`.

    Returns
    -------
    coef : jnp.ndarray
        Coefficient matrix of shape (n_features, n_states) or
         (n_features, n_neurons, n_states).
    intercept : jnp.ndarray
        Intercept array of shape (n_states,) or (n_neurons, n_states).
    scale: jnp.ndarray
        Scale of the GLM, shape (n_states,) or (n_neurons, n_states).
    initial_proba : jnp.ndarray
        Initial state probability vector of shape (n_states,).
    transition_proba : jnp.ndarray
        Transition probability matrix of shape (n_states, n_states).
    """
    if initialization_funcs is None:
        initialization_funcs = DEFAULT_INIT_FUNCTION
    else:
        initialization_funcs = _resolve_init_funcs_registry(initialization_funcs)
    if initialization_kwargs is None:
        initialization_kwargs = {}
    random_key, subkey = jax.random.split(random_key)
    kwargs = initialization_kwargs.get("glm_params_init", {})
    coef, intercept = initialization_funcs["glm_params_init"](
        n_states, X, y, inverse_link_function, subkey, **kwargs
    )
    random_key, subkey = jax.random.split(random_key)
    kwargs = initialization_kwargs.get("scale_init", {})
    scale = initialization_funcs["scale_init"](n_states, X, y, subkey, **kwargs)
    random_key, subkey = jax.random.split(random_key)
    kwargs = initialization_kwargs.get("initial_proba_init", {})
    initial_proba = initialization_funcs["initial_proba_init"](
        n_states, X, y, subkey, **kwargs
    )
    _, subkey = jax.random.split(random_key)
    kwargs = initialization_kwargs.get("transition_proba_init", {})
    transition_proba = initialization_funcs["transition_proba_init"](
        n_states, X, y, subkey, **kwargs
    )
    return coef, intercept, scale, initial_proba, transition_proba


def _resolve_init_funcs_registry(
    registry: INITIALIZATION_FN_DICT | None,
) -> INITIALIZATION_FN_DICT:
    """
    Merge and validate a partial initialization registry with defaults.

    Takes a partial registry (with only some keys) and merges it with the default
    registry, validating each provided function.

    Parameters
    ----------
    registry :
        Dictionary mapping parameter names to initialization functions. Must contain
        only valid keys from DEFAULT_INIT_FUNCTION.

    Returns
    -------
    updated_registry :
        Complete registry with user-provided functions merged with defaults.

    Raises
    ------
    KeyError
        If registry contains invalid keys.
    """
    if registry is None:
        registry = DEFAULT_INIT_FUNCTION
    elif not set(registry.keys()).issubset(DEFAULT_INIT_FUNCTION.keys()):
        invalid = set(registry.keys()).difference(DEFAULT_INIT_FUNCTION.keys())
        raise KeyError(
            f"Invalid key(s) for initialization dictionary: {invalid}.\n"
            f"Valid keys are {DEFAULT_INIT_FUNCTION.keys()}."
        )
    updated_registry = DEFAULT_INIT_FUNCTION.copy()
    for func_name, func in registry.items():
        updated_registry[func_name] = _resolve_init_func(func_name, func)
    return updated_registry


def _is_native_init_registry(registry: INITIALIZATION_FN_DICT) -> bool:
    """Return true if a function is a native initialization function."""
    return all(
        [fn in AVAILABLE_INIT_FUNCTIONS[key].values() for key, fn in registry.items()]
    )


def _resolve_init_func(
    func_name: str, init_func: Callable | str
) -> InitFunctionHMM | InitFuncGLMScale | InitFuncGLMParams:
    """
    Validate and resolve an initialization function.

    Checks that the provided initialization function has the correct signature
    (at least 4 parameters: n_states, X, y, key, with any additional parameters
    having default values). Can accept either a string name or a callable.

    Parameters
    ----------
    func_name : str
        Name of the initialization function (e.g., 'glm_params_init').
    init_func : Callable, str, or None
        Initialization function to validate. Can be:
        - None: returns the default function
        - str: looks up the function by name in AVAILABLE_INIT_FUNCTIONS
        - Callable: validates and returns the function

    Returns
    -------
    init_func : Callable
        Validated initialization function.

    Raises
    ------
    ValueError
        If the function signature is invalid (wrong number of parameters or
        extra parameters without defaults), or if string name is not found.
    TypeError
        If init_func is not callable, string, or None.
    """
    if init_func is None:
        # assign default
        return DEFAULT_INIT_FUNCTION[func_name]

    # Handle string inputs (lookup by name)
    elif isinstance(init_func, str):
        available = _IO_AVAILABLE_INIT_FUNCTIONS.get(func_name, {})
        if init_func not in available:
            raise ValueError(
                f"Unknown initialization method '{init_func}' for '{func_name}'.\n"
                f"Available methods: {list(available.keys())}"
            )
        return available[init_func]

    # Check if it's a pre-defined function
    elif callable(init_func):
        # Check if it's in the available functions
        available_funcs = [
            f
            for funcs in AVAILABLE_INIT_FUNCTIONS.get(func_name, {}).values()
            for f in [funcs]
        ]
        if init_func in available_funcs:
            return init_func

        # Validate signature for custom functions
        sig = inspect.signature(init_func)
        n_params = len(sig.parameters)

        # GLM params init needs inverse_link_function, so expects 5 params
        # Other init functions expect 4 params (n_states, X, y, key)
        expected_params = 5 if func_name == "glm_params_init" else 4
        param_desc = (
            "(n_states, X, y, inverse_link_function, key)"
            if func_name == "glm_params_init"
            else "(n_states, X, y, key)"
        )

        # Check minimum number of parameters
        if n_params < expected_params:
            param_names = list(sig.parameters.keys())
            raise ValueError(
                f"'{func_name}' initialization function must have at least {expected_params} parameters: "
                f"{param_desc}, but got {n_params} parameter(s): {param_names}.\n"
            )

        # Check that extra parameters have defaults
        params = list(sig.parameters.values())
        params_without_defaults = [
            p.name
            for p in params[expected_params:]
            if p.default is inspect.Parameter.empty
        ]

        if params_without_defaults:
            raise ValueError(
                f"All parameters beyond the required {expected_params} {param_desc} must have default values.\n"
                f"Parameters without defaults: {params_without_defaults}"
            )

        return init_func

    # Provide appropriate signature based on function type
    if func_name == "glm_params_init":
        signature_desc = (
            "(n_states: int, X: DESIGN_INPUT_TYPE, y: jnp.ndarray, "
            "inverse_link_function: Callable, key: jax.random.PRNGKey, **kwargs)"
        )
    else:
        signature_desc = (
            "(n_states: int, X: DESIGN_INPUT_TYPE, y: jnp.ndarray, "
            "key: jax.random.PRNGKey, **kwargs)"
        )

    raise TypeError(
        f"Invalid initialization function: {func_name}.\n"
        "The initialization function should be:\n"
        "- A string (e.g., 'random', 'sticky', 'uniform')\n"
        f"- A callable with signature: {signature_desc}"
    )


def _resolve_dirichlet_priors(
    alphas: Any, expected_shape: Tuple[int, ...]
) -> jnp.ndarray | None:
    """Validate and convert Dirichlet prior alpha parameters.

    Parameters
    ----------
    alphas :
        Dirichlet prior alpha parameters. Can be None or array-like.
    expected_shape :
        Expected shape of the alpha parameter array.

    Returns
    -------
    jnp.ndarray | None
        Validated alpha parameters as a JAX array, or None if input is None.

    Raises
    ------
    ValueError
        If the shape doesn't match expected_shape or if any alpha < 1.
    TypeError
        If alphas is not None or array-like.
    """
    if alphas is None:
        return None
    elif is_numpy_array_like(alphas)[1]:
        alphas = jnp.asarray(alphas, dtype=float)
        if alphas.shape != expected_shape:
            raise ValueError(
                f"Dirichlet prior alpha parameters for initial state probabilities "
                f"must have shape ``{expected_shape}``, "
                f"but got shape ``{alphas.shape}``."
            )
        if not jnp.all(alphas >= 1):
            raise ValueError(
                f"Dirichlet prior alpha parameters must be >= 1, but got values < 1"
                f":\n{alphas}"
            )
        return alphas
    else:
        raise TypeError(
            f"Invalid type for Dirichlet prior alpha parameters: ``{type(alphas).__name__}``. "
            f"Must be None or an array-like object of shape ``{expected_shape}`` with strictly positive values."
        )


def _resolve_init_kwargs(
    func_name: str,
    init_func: Callable,
    kwargs: dict | None,
) -> dict:
    """
    Validate keyword arguments for a single initialization function.

    Checks that provided kwargs match optional parameters in the function's
    signature (parameters beyond the required positional ones).

    Parameters
    ----------
    func_name : str
        Registry key for the initialization function (e.g., 'glm_params_init').
    init_func : Callable
        The initialization function to validate kwargs against.
    kwargs : dict or None
        Keyword arguments to validate. If None or empty, returns empty dict.

    Returns
    -------
    dict
        The validated kwargs (unchanged if valid).

    Raises
    ------
    ValueError
        If any kwargs don't match the function's optional parameters.
    """
    if kwargs is None or kwargs == {}:
        return {}

    # for checkers
    kwargs: dict = kwargs
    sig = inspect.signature(init_func)
    params = list(sig.parameters.values())

    # hardcoded minimum number of parameters (mandatory parameters)
    expected_params = 5 if func_name == "glm_params_init" else 4

    # list extra kwargs names
    extra_params = [p.name for p in params[expected_params:]]
    available = all(k in extra_params for k in kwargs)
    if not available:
        unavailable = [k for k in kwargs if k not in extra_params]
        func_display = getattr(init_func, "__name__", str(init_func))
        err = (
            f"Invalid keyword argument(s) {unavailable} in "
            f"``initialization_kwargs['{func_name}']``.\n"
            f"The function '{func_display}' accepts: "
        )
        if extra_params:
            err += f"{extra_params}."
        else:
            err += "no extra keyword arguments."
        raise ValueError(err)
    return kwargs


def _resolve_init_kwargs_registry(
    init_kwargs: dict | None,
    init_func_registry: dict,
) -> dict:
    """
    Validate keyword arguments for all initialization functions in a registry.

    Iterates through the initialization function registry and validates
    corresponding kwargs for each function.

    Parameters
    ----------
    init_kwargs : dict or None
        Dictionary mapping function names to their kwargs. Keys must be valid
        initialization function names (e.g., 'glm_params_init', 'scale_init').
        If None or empty, returns dict with empty kwargs for each function.
    init_func_registry : dict
        Dictionary mapping function names to initialization functions.

    Returns
    -------
    dict
        Dictionary with validated kwargs for each function name in
        DEFAULT_INIT_FUNCTION. Missing keys are filled with empty dicts.

    Raises
    ------
    ValueError
        If any kwargs are invalid for their corresponding function.
    """
    if init_kwargs is None or init_kwargs == {}:
        return {k: {} for k in DEFAULT_INIT_FUNCTION}

    for func_name, init_kwargs_func in init_func_registry.items():
        if func_name in init_kwargs:
            init_kwargs[func_name] = _resolve_init_kwargs(
                func_name, init_kwargs_func, init_kwargs[func_name]
            )
        else:
            init_kwargs[func_name] = {}

    return init_kwargs
