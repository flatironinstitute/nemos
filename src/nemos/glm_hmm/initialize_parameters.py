"""Initialization functions and related utility functions."""

import inspect
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..type_casting import is_numpy_array_like
from ..typing import DESIGN_INPUT_TYPE


def random_projection_and_intercept_init(
    n_states: int, X: DESIGN_INPUT_TYPE, random_key=jax.random.PRNGKey(123)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Initialize projection weights with random normal values.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Design matrix with shape (n_samples, n_features).
    random_key
        Random key for reproducibility. Default is PRNGKey(123).

    Returns
    -------
    :
        Projection weight matrix of shape (n_features, n_states).
    """
    n_features = X.shape[1]
    random_param = 0.1 * jax.random.normal(random_key, (n_features + 1, n_states))
    return random_param[:-1], random_param[-1:]


def sticky_transition_proba_init(n_states: int, prob_stay=0.95):
    """
    Initialize transition probabilities with sticky dynamics.

    Creates a transition probability matrix that favors staying in the current state.

    Parameters
    ----------
    n_states :
        Number of HMM states. Must be greater than 1.
    prob_stay :
        Probability of staying in the current state. Default is 0.95.

    Returns
    -------
    :
        Transition probability matrix of shape (n_states, n_states).
    """
    # assume n_state is > 1
    prob_leave = (1 - prob_stay) / (n_states - 1)
    return jnp.full((n_states, n_states), prob_leave) + jnp.diag(
        (prob_stay - prob_leave) * jnp.ones(n_states)
    )


def uniform_initial_proba_init(n_states: int, random_key=jax.random.PRNGKey(124)):
    """
    Initialize initial state probabilities from a uniform distribution.

    Generates random initial state probabilities by sampling from a uniform
    distribution and normalizing to ensure they sum to 1.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    random_key :
        Random key for reproducibility. Default is PRNGKey(124).

    Returns
    -------
    :
        Initial state probability vector of shape (n_states,) that sums to 1.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nemos.glm_hmm.parameters_initialization import uniform_initial_proba_init
    >>>
    >>> # Generate initial probabilities for 3 states
    >>> n_states = 3
    >>> init_probs = uniform_initial_proba_init(n_states)
    >>> init_probs.shape
    (3,)
    >>> jnp.isclose(jnp.sum(init_probs), 1.0)
    Array(True, dtype=bool)
    >>>
    >>> # Use a custom random key for reproducibility
    >>> key = jax.random.PRNGKey(42)
    >>> init_probs = uniform_initial_proba_init(n_states, random_key=key)
    >>> jnp.isclose(jnp.sum(init_probs), 1.0)
    Array(True, dtype=bool)
    """
    prob = jax.random.uniform(random_key, (n_states,), minval=0, maxval=1)
    return prob / jnp.sum(prob)


AVAILABLE_PROJECTION_AND_INTERCEPT_INIT = [random_projection_and_intercept_init]
AVAILABLE_TRANSITION_PROBA_INIT = [sticky_transition_proba_init]
AVAILABLE_INITIAL_PROBA_INIT = [uniform_initial_proba_init]

INIT_FUNCTION_MAPPING = {
    "projection_init": {
        "random_projection_and_intercept": random_projection_and_intercept_init
    },
    "transition_proba": {"sticky_transition_proba": sticky_transition_proba_init},
    "initial_proba": {"uniform": uniform_initial_proba_init},
}


def resolve_projection_and_intercept_init_function(
    projection_and_intercept_init: Callable | str | ArrayLike | Any,
) -> (
    Callable[
        [int, DESIGN_INPUT_TYPE, jax.random.PRNGKey], Tuple[jnp.ndarray, jnp.ndarray]
    ]
    | Tuple[jnp.ndarray, jnp.ndarray]
):
    """
    Validate and resolve a projection weight initialization specification.

    This function handles multiple input formats for initializing projection weights
    in HMM-based models. It validates the input and returns either the provided
    function/array or resolves a string name to the corresponding initialization function.

    Parameters
    ----------
    projection_and_intercept_init :
        The projection weights and intercept initialization specification. Can be:
        - A pre-defined initialization function from AVAILABLE_PROJECTION_INIT
        - str: name of a standard initialization method (e.g., "random")
        - array-like: explicit projection weight values (n_features, n_states)
        - Callable: custom initialization function with signature
          (n_states: int, X: DESIGN_INPUT_TYPE, key: jax.random.PRNGKey, **kwargs) -> NDArray
          where any additional keyword arguments must have default values

    Returns
    -------
    :
        A validated initialization function or the provided array-like object.

    Raises
    ------
    ValueError
        - If the string name is not recognized
        - If a callable has fewer than 3 required parameters
        - If a callable has extra parameters without default values
    TypeError
        If the input is not a recognized type (string, callable, or array-like).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax
    >>> from nemos.glm_hmm.initialize_parameters import resolve_projection_and_intercept_init_function
    >>>
    >>> n_features, n_states = 4, 3
    >>>
    >>> # Use a named initialization method
    >>> init_fn = resolve_projection_and_intercept_init_function("random_projection")
    >>> callable(init_fn)
    True
    >>>
    >>> # Use explicit weights
    >>> weights = jnp.zeros((n_features, n_states))
    >>> init_array = resolve_projection_and_intercept_init_function(weights)
    >>> init_array
    Array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32)
    >>>
    >>> # Use custom function with optional parameters
    >>> def custom_init(n_states, X, key, scale=1.0):
    ...     return jax.random.normal(key, (n_states, X.shape[1])) * scale
    >>> init_fn = resolve_projection_and_intercept_init_function(custom_init)
    >>> callable(init_fn)
    True
    """
    # Check if it's a pre-defined function
    if projection_and_intercept_init in AVAILABLE_PROJECTION_AND_INTERCEPT_INIT:
        return projection_and_intercept_init

    # Handle string names
    elif isinstance(projection_and_intercept_init, str):
        mapping = INIT_FUNCTION_MAPPING["projection_init"]
        if projection_and_intercept_init not in mapping:
            available = ", ".join(f"'{k}'" for k in mapping.keys())
            raise ValueError(
                f"Unknown projection initialization method: '{projection_and_intercept_init}'.\n"
                f"Available methods are: {available}"
            )
        return mapping[projection_and_intercept_init]

    # Handle array-like inputs
    elif is_numpy_array_like(projection_and_intercept_init)[1]:
        return jnp.asarray(projection_and_intercept_init, float)

    # Handle callable inputs
    elif callable(projection_and_intercept_init):
        sig = inspect.signature(projection_and_intercept_init)
        n_params = len(sig.parameters)

        # Check minimum number of parameters
        if n_params < 3:
            param_names = list(sig.parameters.keys())
            raise ValueError(
                f"Projection initialization function must have at least 3 parameters: "
                f"(n_states, X, key), but got {n_params} parameter(s): {param_names}.\n"
                f"Signature should be: Callable[[int, DESIGN_INPUT_TYPE, jax.random.PRNGKey], NDArray]"
            )

        # Check that extra parameters have defaults
        params = list(sig.parameters.values())
        params_without_defaults = [
            p.name for p in params[3:] if p.default is inspect.Parameter.empty
        ]

        if params_without_defaults:
            raise ValueError(
                f"All parameters beyond the required 3 (n_states, X, key) must have default values.\n"
                f"Parameters without defaults: {params_without_defaults}"
            )

        return projection_and_intercept_init

    # Invalid type
    valid_strings_formatted = ", ".join(
        f"'{s}'" for s in INIT_FUNCTION_MAPPING["projection_init"].keys()
    )
    raise TypeError(
        f"Invalid projection weight initialization type: {type(projection_and_intercept_init).__name__}.\n"
        "The projection weight initialization must be one of:\n"
        f"  - A string from: {valid_strings_formatted}\n"
        "  - An array-like object with shape (n_states, n_features)\n"
        "  - A callable with signature: (n_states: int, X: DESIGN_INPUT_TYPE, "
        "key: jax.random.PRNGKey, **kwargs) -> NDArray"
    )


def resolve_transition_proba_init_function(
    transition_prob: Callable | str | ArrayLike | Any,
) -> Callable[[int], jnp.ndarray] | jnp.ndarray:
    """
    Validate and resolve a transition probability initialization specification.

    This function handles multiple input formats for initializing transition probability
    matrices in HMM-based models. It validates the input and returns either the provided
    function/array or resolves a string name to the corresponding initialization function.

    Parameters
    ----------
    transition_prob :
        The transition probability initialization specification. Can be:
        - A pre-defined initialization function from AVAILABLE_TRANSITION_PROBA_INIT
        - str: name of a standard initialization method (e.g., "sticky_transition_proba")
        - array-like: explicit transition probability matrix (n_states, n_states)
        - Callable: custom initialization function with signature
          (n_states: int, **kwargs) -> NDArray
          where any additional keyword arguments must have default values

    Returns
    -------
    :
        A validated initialization function or the provided initial parameters as a jax array.
        If a function, it must return a transition probability matrix of shape (n_states, n_states).

    Raises
    ------
    ValueError
        - If the string name is not recognized
        - If a callable has fewer than 1 required parameter
        - If a callable has extra parameters without default values
    TypeError
        If the input is not a recognized type (string, callable, or array-like).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax
    >>> from nemos.glm_hmm.initialize_parameters import resolve_transition_proba_init_function
    >>>
    >>> n_states = 3
    >>> # Use a named initialization method
    >>> init_fn = resolve_transition_proba_init_function("sticky_transition_proba")
    >>> callable(init_fn)
    True
    >>>
    >>> # Use explicit transition matrix
    >>> trans_matrix = jnp.eye(n_states) * 0.9 + 0.1 / n_states
    >>> init_array = resolve_transition_proba_init_function(trans_matrix)
    >>> init_array
    Array([[0.93333334, 0.03333334, 0.03333334],
       [0.03333334, 0.93333334, 0.03333334],
       [0.03333334, 0.03333334, 0.93333334]], dtype=float32)
    >>>
    >>> # Use custom function with optional parameters
    >>> def custom_init(n_states, diagonal_weight=0.8):
    ...     return jnp.eye(n_states) * diagonal_weight
    >>> init_fn = resolve_transition_proba_init_function(custom_init)
    >>> callable(init_fn)
    True
    """
    # Check if it's a pre-defined function
    if transition_prob in AVAILABLE_TRANSITION_PROBA_INIT:
        return transition_prob

    # Handle string names
    elif isinstance(transition_prob, str):
        mapping = INIT_FUNCTION_MAPPING["transition_proba"]
        if transition_prob not in mapping:
            available = ", ".join(f"'{k}'" for k in mapping.keys())
            raise ValueError(
                f"Unknown transition probability initialization method: '{transition_prob}'.\n"
                f"Available methods are: {available}"
            )
        return mapping[transition_prob]

    # Handle array-like inputs
    elif is_numpy_array_like(transition_prob)[1]:
        return jnp.asarray(transition_prob, dtype=float)

    # Handle callable inputs
    elif callable(transition_prob):
        sig = inspect.signature(transition_prob)
        n_params = len(sig.parameters)

        # Check minimum number of parameters
        if n_params < 1:
            raise ValueError(
                f"Transition probability initialization function must have at least 1 parameter: "
                f"(n_states), but got {n_params} parameter(s).\n"
                f"Signature should be: Callable[[int], NDArray]"
            )

        # Check that extra parameters have defaults
        params = list(sig.parameters.values())
        params_without_defaults = [
            p.name for p in params[1:] if p.default is inspect.Parameter.empty
        ]

        if params_without_defaults:
            raise ValueError(
                f"All parameters beyond the required 1 (n_states) must have default values.\n"
                f"Parameters without defaults: {params_without_defaults}"
            )

        return transition_prob

    # Invalid type
    else:
        valid_strings = list(INIT_FUNCTION_MAPPING["transition_proba"].keys())
        valid_strings_str = ", ".join(f"'{s}'" for s in valid_strings)

        raise TypeError(
            f"Invalid transition probability initialization type: {type(transition_prob).__name__}.\n"
            "The transition probability initialization must be one of:\n"
            f"  - A string from: {valid_strings_str}\n"
            "  - An array-like object with shape (n_states, n_states)\n"
            "  - A callable with signature: (n_states: int, **kwargs) -> NDArray"
        )


def resolve_initial_state_proba_init_function(
    initial_state_proba_init: Callable | str | ArrayLike | Any,
) -> Callable[[int, jax.random.PRNGKey], int] | jnp.ndarray:
    """
    Validate and resolve an initial state probability initialization specification.

    This function handles multiple input formats for initializing the initial state
    probability distribution (π₀) in HMM-based models. It validates the input and
    returns either the provided function/array or resolves a string name to the
    corresponding initialization function.

    Parameters
    ----------
    initial_state_proba_init :
        The initial state probability initialization specification. Can be:
        - A pre-defined initialization function from AVAILABLE_INITIAL_PROBA_INIT
        - str: name of a standard initialization method (e.g., "uniform")
        - array-like: explicit initial state probability vector (n_states,)
        - Callable: custom initialization function with signature
          (n_states: int, key: jax.random.PRNGKey, **kwargs) -> NDArray
          where any additional keyword arguments must have default values

    Returns
    -------
    init_function_or_array :
        A validated initialization function or the provided array-like object.
        The function returns an initial state index or probability vector.

    Raises
    ------
    ValueError
        - If the string name is not recognized
        - If a callable has fewer than 2 required parameters
        - If a callable has extra parameters without default values
    TypeError
        If the input is not a recognized type (string, callable, or array-like).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nemos.glm_hmm.initialize_parameters import resolve_initial_state_proba_init_function
    >>>
    >>> n_states = 3
    >>>
    >>> # Use a named initialization method
    >>> init_fn = resolve_initial_state_proba_init_function("uniform")
    >>> callable(init_fn)
    True
    >>>
    >>> # Use explicit initial state probabilities
    >>> init_probs = jnp.array([0.5, 0.3, 0.2])
    >>> init_array = resolve_initial_state_proba_init_function(init_probs)
    >>> init_array
    Array([0.5, 0.3, 0.2], dtype=float32)
    >>>
    >>> # Use custom function with optional parameters
    >>> def custom_init(n_states, key, temperature=1.0):
    ...     logits = jax.random.normal(key, (n_states,)) / temperature
    ...     return jax.nn.softmax(logits)
    >>> init_fn = resolve_initial_state_proba_init_function(custom_init)
    >>> callable(init_fn)
    True
    """
    # Check if it's a pre-defined function
    if initial_state_proba_init in AVAILABLE_INITIAL_PROBA_INIT:
        return initial_state_proba_init

    # Handle string names
    elif isinstance(initial_state_proba_init, str):
        mapping = INIT_FUNCTION_MAPPING["initial_proba"]
        if initial_state_proba_init not in mapping:
            available = ", ".join(f"'{k}'" for k in mapping.keys())
            raise ValueError(
                f"Unknown initial state initialization method: '{initial_state_proba_init}'.\n"
                f"Available methods are: {available}"
            )
        return mapping[initial_state_proba_init]

    # Handle array-like inputs
    elif is_numpy_array_like(initial_state_proba_init)[1]:
        return jnp.asarray(initial_state_proba_init, dtype=float)

    # Handle callable inputs
    elif callable(initial_state_proba_init):
        sig = inspect.signature(initial_state_proba_init)
        n_params = len(sig.parameters)

        # Check minimum number of parameters
        if n_params < 2:
            param_names = list(sig.parameters.keys())
            raise ValueError(
                f"Initial state initialization function must have at least 2 parameters: "
                f"(n_states, key), but got {n_params} parameter(s): {param_names}.\n"
                f"Signature should be: Callable[[int, jax.random.PRNGKey], int or NDArray]"
            )

        # Check that extra parameters have defaults
        params = list(sig.parameters.values())
        params_without_defaults = [
            p.name for p in params[2:] if p.default is inspect.Parameter.empty
        ]

        if params_without_defaults:
            raise ValueError(
                f"All parameters beyond the required 2 (n_states, key) must have default values.\n"
                f"Parameters without defaults: {params_without_defaults}"
            )

        return initial_state_proba_init

    # Invalid type
    else:
        valid_strings = list(INIT_FUNCTION_MAPPING["initial_proba"].keys())
        valid_strings_str = ", ".join(f"'{s}'" for s in valid_strings)

        raise TypeError(
            f"Invalid initial state initialization type: {type(initial_state_proba_init).__name__}.\n"
            "The initial state initialization must be one of:\n"
            f"  - A string from: {valid_strings_str}\n"
            "  - An array-like object with shape (n_states,)\n"
            "  - A callable with signature: (n_states: int, key: jax.random.PRNGKey, **kwargs) -> int or NDArray"
        )
