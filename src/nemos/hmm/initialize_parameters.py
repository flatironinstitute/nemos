import inspect
import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Tuple, Protocol
from ..validation import _suggest_keys
from ..type_casting import is_numpy_array_like


class InitFunctionHMM(Protocol):
    """Protocol for HMM probability initialization functions (initial and transition)."""

    def __call__(
        self,
        n_states: int,
        random_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize HMM probabilities."""
        ...


def sticky_transition_proba_init(
    n_states: int,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
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
    n_states :
        Number of HMM states. Must be greater than 1.
    seed :
        Random key, unused for this particular initialization, but added for API consistency.
    prob_stay :
        Probability of staying in the current state. Default is 0.95.

    Returns
    -------
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states).

    Examples
    --------
    >>> from nemos.glm_hmm.initialize_parameters import sticky_transition_proba_init
    >>>
    >>> # Generate transition probabilities for 3 states with sticky dynamics
    >>> n_states = 3
    >>> transition_matrix = sticky_transition_proba_init(n_states, prob_stay=0.95)
    >>> print(transition_matrix)
    [[0.95  0.025 0.025]
     [0.025 0.95  0.025]
     [0.025 0.025 0.95 ]]
    """
    # assume n_state is > 1
    if n_states == 1:
        prob_stay = 1.0
        prob_leave = 0.0
    else:
        prob_leave = (1 - prob_stay) / (n_states - 1)
    return jnp.full((n_states, n_states), prob_leave) + jnp.diag(
        (prob_stay - prob_leave) * jnp.ones(n_states)
    )


def uniform_transition_proba_init(
    n_states: int,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
) -> jnp.ndarray:
    """
    Initialize transition probabilities with uniform dynamics.

    Creates a transition probability matrix that assign equal probability of
    transitioning to any of the states.

    Parameters
    ----------
    n_states :
        Number of HMM states. Must be greater than 1.
    seed :
        Random key, unused for this particular initialization, but added for API consistency.

    Returns
    -------
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states).

    Examples
    --------
    >>> from nemos.glm_hmm.initialize_parameters import uniform_transition_proba_init
    >>>
    >>> # Generate transition probabilities for 3 states with uniform dynamics
    >>> n_states = 3
    >>> transition_matrix = uniform_transition_proba_init(n_states)
    >>> print(transition_matrix)
    [[0.33333333 0.33333333 0.33333333]
     [0.33333333 0.33333333 0.33333333]
     [0.33333333 0.33333333 0.33333333]]
    """
    prob_transition = 1.0 / n_states
    return jnp.full((n_states, n_states), prob_transition, dtype=float)


def random_transition_proba_init(
    n_states: int,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
) -> jnp.ndarray:
    """
    Initialize transition probabilities randomly.

    Creates a random transition probability matrix by sampling from a normal
    distribution and normalizing to ensure rows sum to 1.

    Parameters
    ----------
    n_states :
        Number of HMM states. Must be greater than 1.
    random_key :
        Random key for reproducibility of random initialization.

    Returns
    -------
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nemos.glm_hmm.initialize_parameters import random_transition_proba_init
    >>>
    >>> # Generate random transition probabilities for 3 states
    >>> n_states = 3
    >>> transition_matrix = random_transition_proba_init(n_states)
    >>> transition_matrix.shape
    (3, 3)
    >>> jnp.allclose(transition_matrix.sum(axis=1), 1.0)
    Array(True, dtype=bool)
    """
    prob_transition = jax.random.normal(random_key, (n_states, n_states), dtype=float)
    return prob_transition / prob_transition.sum(axis=1, keepdims=True)


def uniform_initial_proba_init(
    n_states: int,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
) -> jnp.ndarray:
    """
    Initialize initial state probabilities as a uniform distribution.

    Creates an initial probability matrix that assigns equal probability of
    starting in any state.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    random_key :
        Random key, unused for this particular initialization, but added for API consistency.

    Returns
    -------
    initial_probs :
        Initial state probability vector of shape (n_states,) that sums to 1.

    Examples
    --------
    >>> from nemos.glm_hmm.initialize_parameters import uniform_initial_proba_init
    >>>
    >>> # Generate initial probabilities for 3 states
    >>> n_states = 3
    >>> init_probs = uniform_initial_proba_init(n_states)
    >>> print(init_probs)
    [0.33333333 0.33333333 0.33333333]
    """
    prob = jnp.ones((n_states,), dtype=float)
    return prob / jnp.sum(prob)


def random_initial_proba_init(
    n_states: int,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
) -> jnp.ndarray:
    """
    Randomly initialize initial state probabilities.

    Generates random initial state probabilities by sampling from a normal
    distribution and normalizing to ensure they sum to 1.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    random_key :
        Random key for reproducibility of random initialization.

    Returns
    -------
    initial_probs :
        Initial state probability vector of shape (n_states,) that sums to 1.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nemos.glm_hmm.initialize_parameters import random_initial_proba_init
    >>>
    >>> # Generate initial probabilities for 3 states
    >>> n_states = 3
    >>> init_probs = random_initial_proba_init(n_states)
    >>> init_probs.shape
    (3,)
    >>> jnp.isclose(jnp.sum(init_probs), 1.0)
    Array(True, dtype=bool)
    """
    prob = jax.random.normal(random_key, (n_states,), dtype=float)
    return prob / jnp.sum(prob)


AVAILABLE_INIT_FUNCTIONS = {
    "transition_proba_init": {
        "sticky": sticky_transition_proba_init,
        "uniform": uniform_transition_proba_init,
        "random": random_transition_proba_init,
    },
    "initial_proba_init": {
        "uniform": uniform_initial_proba_init,
        "random": random_initial_proba_init,
    },
}

DEFAULT_INIT_FUNCTIONS = {
    "initial_proba_init": uniform_initial_proba_init,
    "initial_proba_init_kwargs": {},
    "transition_proba_init": sticky_transition_proba_init,
    "transition_proba_init_kwargs": {},
}


def setup_hmm_initialization(
    initial_proba_init: Optional[str | Callable] = None,
    initial_proba_init_kwargs: Optional[dict] = None,
    transition_proba_init: Optional[str | Callable] = None,
    transition_proba_init_kwargs: Optional[dict] = None,
    init_funcs: Optional[dict] = None,
):
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
    init_funcs :
        Existing dictionary of initialization functions to update. If None, a new dictionary will be created.

    Returns
    -------
    init_funcs :
        Updated or initialized dictionary based on provided inputs.
    """

    if init_funcs is None:
        init_funcs = DEFAULT_INIT_FUNCTIONS.copy()
    else:
        # check for unexpected/unknown keys in init_funcs and backfill with defaults
        init_funcs = _validate_init_funcs_keys(init_funcs)

    # update functions and kwargs for init prob and transition prob
    # if a function is passed but not kwargs, kwargs will be reset
    # if kwargs are passed but not function, kwargs will be validated against existing function in init_funcs
    if initial_proba_init is not None:
        init_funcs["initial_proba_init"], init_funcs["initial_proba_init_kwargs"] = (
            _resolve_init_funcs(
                "initial_proba_init", initial_proba_init, initial_proba_init_kwargs
            )
        )
    elif initial_proba_init_kwargs is not None:
        init_funcs["initial_proba_init_kwargs"] = _validate_init_funcs_kwargs(
            init_funcs["initial_proba_init"], initial_proba_init_kwargs
        )
    if transition_proba_init is not None:
        (
            init_funcs["transition_proba_init"],
            init_funcs["transition_proba_init_kwargs"],
        ) = _resolve_init_funcs(
            "transition_proba_init", transition_proba_init, transition_proba_init_kwargs
        )
    elif transition_proba_init_kwargs is not None:
        init_funcs["transition_proba_init_kwargs"] = _validate_init_funcs_kwargs(
            init_funcs["transition_proba_init"], transition_proba_init_kwargs
        )

    return init_funcs


def _validate_init_funcs_keys(init_funcs: dict) -> None:
    """Validate that the keys in the init_funcs dictionary are as expected. Set missing values to defaults."""
    unexpected_keys = init_funcs.keys() - DEFAULT_INIT_FUNCTIONS.keys()
    if unexpected_keys:
        suggested_keys = _suggest_keys(unexpected_keys, DEFAULT_INIT_FUNCTIONS.keys())
        error_msg = (
            (
                f" Unexpected key: '{key}'. Did you mean '{suggestion}'?"
                if suggestion
                else f" Unknown key: '{key}'"
            )
            for key, suggestion in suggested_keys
        )
        raise ValueError(
            "Unexpected or unknown keys found in 'init_funcs' dictionary. \n"
            + "\n".join(error_msg)
        )
    # resolve with defaults and make copy
    init_funcs = DEFAULT_INIT_FUNCTIONS | init_funcs
    return init_funcs


def _resolve_init_funcs(
    key: str, value: str | Callable, kwargs: Optional[dict] = None
) -> Tuple[Callable, dict]:
    """
    Validate a provided initialization function.

    This function checks if the provided value is a valid string key for a built-in initialization function or
    validates the function if it is a custom callable.

    Parameters
    ----------
    key :
        The name of the parameter being initialized (e.g., 'initial_proba_init' or 'transition_proba_init').
    value :
        The user-provided initialization function, either as a string key for built-in functions or as a custom
        callable.
    kwargs :
        Optional keyword arguments to pass to the initialization function.

    Returns
    -------
    :
        Validated initialization function and its kwargs (if any) to be used for HMM parameter initialization.

    Raises
    ------
    ValueError
        If the provided string key does not correspond to a valid built-in function or if the custom function does not
        have the expected signature.
    TypeError
        If the provided value is neither a string nor a callable function.
    """
    if isinstance(value, str):
        if value not in AVAILABLE_INIT_FUNCTIONS[key]:
            raise ValueError(
                f"Invalid initialization function name '{value}' for '{key}'. "
                f"Available options are: {list(AVAILABLE_INIT_FUNCTIONS[key].keys())}."
            )
        kwargs = _validate_init_funcs_kwargs(
            AVAILABLE_INIT_FUNCTIONS[key][value], kwargs
        )
        return AVAILABLE_INIT_FUNCTIONS[key][value], kwargs
    elif callable(value):
        return _validate_custom_init_func(key, value, kwargs)
    else:
        raise TypeError(
            f"Initialization function for '{key}' must be either a string or a callable. "
            f"Got {type(value)} instead."
        )


def _validate_init_funcs_kwargs(func: Callable, kwargs: dict | None) -> dict:
    """Validate that the provided kwargs match the expected signature of the initialization functions."""
    if kwargs is None:
        return {}
    if "n_states" in kwargs:
        raise ValueError(
            "Keyword argument 'n_states' should not be provided in kwargs for initialization functions."
        )
    sig = inspect.signature(func)
    bads = kwargs.keys() - sig.parameters.keys()
    if bads:
        raise ValueError(
            f"Invalid keyword argument for initialization function '{func.__name__}'. "
            f"Unexpected keys: {sorted(bads)}. "
            f"Expected parameters are: {sorted(sig.parameters.keys())}."
        )
    return kwargs


def _validate_custom_init_func(
    key: str, func: Callable, kwargs: Optional[dict] = None
) -> Callable:
    """
    Validate a custom initialization function against the expected signature.

    Checks that the provided function is callable and has the required input parameters (n_states, seed) and correct
    output shape.

    Parameters
    ----------
    key : str
        The name of the parameter being initialized (e.g., 'initial_proba_init' or 'transition_proba_init').
    func : Callable
        The user-provided initialization function to validate.
    kwargs : Optional[dict]
        Optional keyword arguments to pass to the initialization function.

    Returns
    -------
    func :
        The validated initialization function.
    kwargs :
        The validated kwargs to be passed to the initialization function.

    Raises
    ------
    ValueError
        If the function does not have the expected signature.
    ValueError
        If the function does not return an array of the expected shape.
    """
    protocol_sig = inspect.signature(InitFunctionHMM.__call__)
    required_params = {
        name
        for name, param in protocol_sig.parameters.items()
        if name != "self"
        and param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }
    sig = inspect.signature(func)
    missing = required_params - sig.parameters.keys()
    if missing:
        raise ValueError(
            f"Custom initialization function must have the parameters {sorted(required_params)}. "
            f"Missing: {sorted(missing)}."
        )
    kwargs = _validate_init_funcs_kwargs(func, kwargs)

    # test output for 3 states
    out = func(n_states=3, **kwargs)
    if key == "initial_proba_init":
        if out.shape != (3,):
            raise ValueError(
                f"Custom initialization function for '{key}' must return an array of shape "
                f"(n_states,)."
            )
        if not jnp.isclose(out.sum(), 1.0):
            raise ValueError(
                f"Custom initialization function for '{key}' must return an array that sums to 1."
            )
    else:
        if out.shape != (3, 3):
            raise ValueError(
                f"Custom initialization function for '{key}' must return an array of shape "
                f"(n_states,n_states)."
            )
        if not jnp.allclose(jnp.sum(out, axis=1), 1.0):
            raise ValueError(
                f"Custom initialization function for '{key}' must return a matrix with rows that sum to 1. "
            )

    return func, kwargs


def generate_hmm_initial_params(
    n_states: int, init_funcs: dict = {}, seed: int = 123
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate initial HMM parameters using the provided initialization functions.

    This function calls the specified initialization functions for initial state probabilities
    stored in the `init_funcs` dictionary passing `n_states` and any additional stored kwargs.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    init_funcs :
        Dictionary containing the initialization functions and their kwargs for both initial state probabilities
        and transition probabilities. This dictionary can be set up using the `setup_hmm_initialization` function.
        If not provided, or if specific functions are missing, defaults will be used.
    seed :
        Optional seed for random number generation, if needed by the initialization functions. This is used globally,
        but is overwritten by function-specific random_keys if they are provided in the `init_funcs` kwargs.

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
    init_funcs = _validate_init_funcs_keys(init_funcs)

    # grab initial probability initialization function
    # fallback to defaults if set to None
    initial_proba_init = (
        init_funcs["initial_proba_init"] or DEFAULT_INIT_FUNCTIONS["initial_proba_init"]
    )
    # set to empty kwargs if set to None
    initial_proba_init_kwargs = init_funcs["initial_proba_init_kwargs"] or {}

    # do the same for transition matrix
    transition_proba_init = (
        init_funcs["transition_proba_init"]
        or DEFAULT_INIT_FUNCTIONS["transition_proba_init"]
    )
    transition_proba_init_kwargs = init_funcs["transition_proba_init_kwargs"] or {}

    # split seed into two random keys, overwritten by user-provided random keys
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    if "random_key" not in initial_proba_init_kwargs:
        initial_proba_init_kwargs["random_key"] = keys[0]
    if "random_key" not in transition_proba_init_kwargs:
        transition_proba_init_kwargs["random_key"] = keys[1]

    initial_probs = initial_proba_init(n_states, **initial_proba_init_kwargs)
    transition_matrix = transition_proba_init(n_states, **transition_proba_init_kwargs)

    return initial_probs, transition_matrix


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
