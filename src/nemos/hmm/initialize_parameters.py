import inspect
import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from typing import Any, Callable, Optional, Tuple

from ..typing import DESIGN_INPUT_TYPE


class InitFunctionHMM(Protocol):
    """Protocol for HMM probability initialization functions (initial and transition)."""

    def __call__(
        self,
        n_states: int,
        seed: int,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize HMM probabilities."""
        ...


def sticky_transition_proba_init(
    n_states: int,
    seed: int = 123,
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
    "transition_proba_init": {
        "sticky": sticky_transition_proba_init,
        "uniform": uniform_transition_proba_init,
    },
    "initial_proba_init": {
        "uniform": uniform_initial_proba_init,
    },
}

DEFAULT_INIT_FUNCTIONS = {
    "initial_proba_init": uniform_initial_proba_init,
    "initial_proba_init_kwargs": None,
    "transition_proba_init": sticky_transition_proba_init,
    "transition_proba_init_kwargs": None,
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
        init_funcs = DEFAULT_INIT_FUNCTIONS

    if initial_proba_init is not None:
        init_funcs["initial_proba_init"], init_funcs["initial_proba_init_kwargs"] = (
            _resolve_init_funcs(
                "initial_proba_init", initial_proba_init, initial_proba_init_kwargs
            )
        )

    if transition_proba_init is not None:
        (
            init_funcs["transition_proba_init"],
            init_funcs["transition_proba_init_kwargs"],
        ) = _resolve_init_funcs(
            "transition_proba_init", transition_proba_init, transition_proba_init_kwargs
        )

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
        if kwargs is not None:
            kwargs = _validate_init_kwargs(AVAILABLE_INIT_FUNCTIONS[key][value], kwargs)
        return AVAILABLE_INIT_FUNCTIONS[key][value], kwargs
    elif callable(value):
        return _validate_custom_init_func(key, value, kwargs)
    else:
        raise TypeError(
            f"Initialization function for '{key}' must be either a string or a callable. "
            f"Got {type(value)} instead."
        )


def _validate_init_kwargs(func: Callable, kwargs: dict) -> dict:
    """Validate that the provided kwargs match the expected signature of the initialization functions."""
    sig = inspect.signature(func)
    bads = kwargs.keys() - sig.parameters.keys()
    if bads:
        raise ValueError(
            f"Invalid kwargs for initialization function '{func.__name__}'. "
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

    if kwargs is not None:
        kwargs = _validate_init_kwargs(func, kwargs)

    out = func(n_states=3, seed=0, **kwargs)
    expected = (
        [(3, 3), "(n_states, n_states)"]
        if key == "transition_proba_init"
        else [(3,), "(n_states,)"]
    )
    if tuple(out.shape) != expected[0]:
        raise ValueError(
            f"Custom initialization function for '{key}' must return an array of shape "
            f"{expected[1]}."
        )

    return func, kwargs
