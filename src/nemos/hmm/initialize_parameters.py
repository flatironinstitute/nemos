"""Initialization functions and related utility functions for HMMs."""

import inspect
from typing import Any, Callable, Literal, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from ..type_casting import is_numpy_array_like
from ..typing import DESIGN_INPUT_TYPE
from ..validation import _suggest_keys
from .utils import initialize_new_session


class InitFunctionHMM(Protocol):
    """Protocol for HMM probability initialization functions (initial and transition)."""

    def __call__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        random_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Initialize HMM probabilities."""
        ...


def _get_protocol_parameters(protocol) -> set[str]:
    """Get the required parameters for the initialization function based on the protocol."""
    protocol_sig = inspect.signature(protocol.__call__)
    return {
        name
        for name, param in protocol_sig.parameters.items()
        if name != "self"
        and param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


INITIALIZATION_FN_DICT = dict[
    Literal[
        "initial_proba_init",
        "initial_proba_init_kwargs",
        "initial_proba_init_custom",
        "transition_proba_init",
        "transition_proba_init_kwargs",
        "transition_proba_init_custom",
    ],
    InitFunctionHMM | dict[str, Any] | bool | InitFunctionHMM | dict[str, Any] | bool,
]


def sticky_transition_proba_init(
    n_states: int,
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    prob_stay: float = 0.95,
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
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    random_key :
        Random key, unused for this particular initialization, but added for API consistency.
    prob_stay :
        Probability of staying in the current state. Default is 0.95.

    Returns
    -------
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states).

    Examples
    --------
    >>> from nemos.hmm.initialize_parameters import sticky_transition_proba_init
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
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
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
    random_key :
        Random key, unused for this particular initialization, but added for API consistency.

    Returns
    -------
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states).

    Examples
    --------
    >>> from nemos.hmm.initialize_parameters import uniform_transition_proba_init
    >>>
    >>> # Generate transition probabilities for 2 states with uniform dynamics
    >>> n_states = 2
    >>> transition_matrix = uniform_transition_proba_init(n_states)
    >>> print(transition_matrix)
    [[0.5 0.5]
     [0.5 0.5]]
    """
    prob_transition = 1.0 / n_states
    return jnp.full((n_states, n_states), prob_transition, dtype=float)


def random_transition_proba_init(
    n_states: int,
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
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
    >>> from nemos.hmm.initialize_parameters import random_transition_proba_init
    >>>
    >>> # Generate random transition probabilities for 3 states
    >>> n_states = 3
    >>> transition_matrix = random_transition_proba_init(n_states)
    >>> transition_matrix.shape
    (3, 3)
    >>> jnp.allclose(transition_matrix.sum(axis=1), 1.0)
    Array(True, dtype=bool)
    """
    prob_transition = jax.random.uniform(random_key, (n_states, n_states), dtype=float)
    return prob_transition / prob_transition.sum(axis=1, keepdims=True)


def uniform_initial_proba_init(
    n_states: int,
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
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
    >>> from nemos.hmm.initialize_parameters import uniform_initial_proba_init
    >>>
    >>> # Generate initial probabilities for 3 states
    >>> n_states = 2
    >>> init_probs = uniform_initial_proba_init(n_states)
    >>> print(init_probs)
    [0.5 0.5]
    """
    prob = jnp.ones((n_states,), dtype=float)
    return prob / jnp.sum(prob)


def random_initial_proba_init(
    n_states: int,
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
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
    >>> from nemos.hmm.initialize_parameters import random_initial_proba_init
    >>>
    >>> # Generate initial probabilities for 3 states
    >>> n_states = 3
    >>> init_probs = random_initial_proba_init(n_states)
    >>> init_probs.shape
    (3,)
    >>> jnp.isclose(jnp.sum(init_probs), 1.0)
    Array(True, dtype=bool)
    """
    prob = jax.random.uniform(random_key, (n_states,), dtype=float)
    return prob / jnp.sum(prob)


class KMeansInitializer:
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
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    random_key :
        Random key for reproducibility of KMeans initialization.
    """

    def __init__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        is_new_session: Optional[jnp.ndarray] = None,
        random_key: int | jax.Array = 0,
    ):
        if isinstance(random_key, jax.Array):
            random_key = int(random_key[-1])

        self.n_states = n_states
        self.random_key = random_key
        self.is_new_session = initialize_new_session(y.shape[0], is_new_session)
        self.model = KMeans(n_clusters=n_states, random_state=random_key)
        # concatenate pytree leaves if applicable and append y
        data = jnp.concatenate(
            jax.tree_util.tree_leaves(X) + [y if y.ndim > 1 else y[:, None]], axis=-1
        )
        self.model.fit(data)
        self.states = jax.nn.one_hot(self.model.labels_, num_classes=n_states)

    def initial_probability(self):
        """
        Compute initial state probabilities based on KMeans assigned states.

        This takes the average occurrence of each state at the start of sessions to estimate the initial state
        probabilities.
        """
        initial_probability = self.states[self.is_new_session].sum(axis=0)
        return initial_probability / initial_probability.sum()

    def transition_probability(self):
        """
        Compute transition probabilities based on KMeans assigned states.

        This computes the transition probabilities by counting the transitions between states across time points,
        excluding transitions that occur at the start of new sessions.
        """
        transition_matrix = (
            self.states[:-1][~self.is_new_session[1:]].T
            @ self.states[1:][~self.is_new_session[1:]]
        )
        return transition_matrix / transition_matrix.sum(axis=1, keepdims=True)


def kmeans_initial_proba_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    is_new_session: Optional[jnp.ndarray] = None,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    initializer: Optional[KMeansInitializer] = None,
):
    """
    Initialize initial state probabilities using KMeans clustering.

    This function creates an instance of the KMeansInitializer class (if not provided) and uses it to compute the
    initial state probabilities based on the assigned states from KMeans clustering.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    random_key :
        Random key for reproducibility of KMeans initialization.
    initializer :
        Optional instance of KMeansInitializer to use for computing initial probabilities.

    Returns
    -------
    initial_probs :
        Initial state probability vector of shape (n_states,) that sums to 1.
    """
    if initializer is None:
        initializer = KMeansInitializer(n_states, X, y, is_new_session, random_key)
    return initializer.initial_probability()


def kmeans_transition_proba_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    is_new_session: Optional[jnp.ndarray] = None,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    initializer: Optional[KMeansInitializer] = None,
):
    """
    Initialize transition probabilities using KMeans clustering.

    This function creates an instance of the KMeansInitializer class (if not provided) and uses it to compute the
    transition probabilities based on the assigned states from KMeans clustering.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    random_key :
        Random key for reproducibility of KMeans initialization.
    initializer :
        Optional instance of KMeansInitializer to use for computing initial probabilities.

    Returns
    -------
    transition_matrix :
        Transition probability matrix of shape (n_states, n_states) computed from KMeans assigned states.
    """
    if initializer is None:
        initializer = KMeansInitializer(n_states, X, y, is_new_session, random_key)
    return initializer.transition_probability()


AVAILABLE_INIT_FUNCTIONS = {
    "initial_proba_init": {
        "uniform": uniform_initial_proba_init,
        "random": random_initial_proba_init,
        "kmeans": kmeans_initial_proba_init,
    },
    "transition_proba_init": {
        "sticky": sticky_transition_proba_init,
        "uniform": uniform_transition_proba_init,
        "random": random_transition_proba_init,
        "kmeans": kmeans_transition_proba_init,
    },
}

DEFAULT_INIT_FUNCTIONS: INITIALIZATION_FN_DICT = {
    "initial_proba_init": uniform_initial_proba_init,
    "initial_proba_init_kwargs": {},
    "initial_proba_init_custom": False,
    "transition_proba_init": sticky_transition_proba_init,
    "transition_proba_init_kwargs": {},
    "transition_proba_init_custom": False,
}


def setup_hmm_initialization(
    initial_proba_init: Optional[str | Callable] = None,
    initial_proba_init_kwargs: Optional[dict] = None,
    transition_proba_init: Optional[str | Callable] = None,
    transition_proba_init_kwargs: Optional[dict] = None,
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
    init_funcs :
        Existing dictionary of initialization functions to update. If None, a new dictionary will be created.
        If the dictionary is missing any keys, they will be backfilled with defaults.

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
        (
            init_funcs["initial_proba_init"],
            init_funcs["initial_proba_init_kwargs"],
            init_funcs["initial_proba_init_custom"],
        ) = _resolve_init_funcs(
            "initial_proba_init", initial_proba_init, initial_proba_init_kwargs
        )
    elif initial_proba_init_kwargs is not None:
        init_funcs["initial_proba_init_kwargs"] = _validate_init_funcs_kwargs(
            init_funcs["initial_proba_init"], initial_proba_init_kwargs
        )
    if transition_proba_init is not None:
        (
            init_funcs["transition_proba_init"],
            init_funcs["transition_proba_init_kwargs"],
            init_funcs["transition_proba_init_custom"],
        ) = _resolve_init_funcs(
            "transition_proba_init", transition_proba_init, transition_proba_init_kwargs
        )
    elif transition_proba_init_kwargs is not None:
        init_funcs["transition_proba_init_kwargs"] = _validate_init_funcs_kwargs(
            init_funcs["transition_proba_init"], transition_proba_init_kwargs
        )

    return init_funcs


def _validate_init_funcs_keys(
    init_funcs: dict | INITIALIZATION_FN_DICT,
) -> INITIALIZATION_FN_DICT:
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
) -> Tuple[InitFunctionHMM, dict]:
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
        return AVAILABLE_INIT_FUNCTIONS[key][value], kwargs, False
    elif callable(value):
        return _validate_custom_init_func(value, kwargs)
    else:
        raise TypeError(
            f"Initialization function for '{key}' must be either a string or a callable. "
            f"Got {type(value)} instead."
        )


def _validate_init_funcs_kwargs(func: InitFunctionHMM, kwargs: dict | None) -> dict:
    """Validate that the provided kwargs match the expected signature of the initialization functions."""
    if kwargs is None:
        return {}

    reserved_params = _get_protocol_parameters(InitFunctionHMM)
    for key in reserved_params:
        if key in kwargs:
            raise ValueError(
                f"Keyword argument '{key}' is reserved and should not be provided in kwargs for initialization "
                "functions."
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
    func: Callable, kwargs: Optional[dict] = None
) -> Tuple[InitFunctionHMM, dict]:
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
    is_custom :
        A boolean indicating that this is a custom initialization function (always True for this function).

    Raises
    ------
    ValueError
        If the function does not have the expected signature.
    ValueError
        If the function does not return an array of the expected shape.
    """
    required_params = _get_protocol_parameters(InitFunctionHMM)
    sig = inspect.signature(func)
    missing = required_params - sig.parameters.keys()
    if missing:
        raise ValueError(
            f"Custom initialization function must have the parameters {sorted(required_params)}. "
            f"Missing: {sorted(missing)}."
        )
    kwargs = _validate_init_funcs_kwargs(func, kwargs)

    return func, kwargs, True


def _validate_custom_init_output(proba: jnp.ndarray, n_states: int, key: str):
    """Validate the output of a custom initialization function to ensure it has the correct shape and properties."""
    if key == "initial_proba_init":
        if proba.shape != (n_states,):
            raise ValueError(
                f"Custom initial state probability initialization function must return an array of shape "
                f"({n_states},), but got {proba.shape}."
            )
        if not jnp.isclose(proba.sum(), 1.0):
            raise ValueError(
                f"Custom initial state probabilities must sum to 1, but got sum={proba.sum()}."
            )
    elif key == "transition_proba_init":
        if proba.shape != (n_states, n_states):
            raise ValueError(
                f"Custom transition probability initialization function must return an array of shape "
                f"({n_states}, {n_states}), but got {proba.shape}."
            )
        if not jnp.allclose(proba.sum(axis=1), 1.0):
            raise ValueError(
                f"Custom transition probabilities must have rows that sum to 1, but got row sums={proba.sum(axis=1)}."
            )


def generate_hmm_initial_params(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    random_key: int | jax.Array = 123,
    init_funcs: dict = {},
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

    if isinstance(random_key, int):
        random_key = jax.random.PRNGKey(random_key)

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

    # split seed into two random keys
    # compute probabilities and validate if custom functions are used
    keys = jax.random.split(random_key, 2)
    initial_probs = initial_proba_init(
        n_states=n_states, X=X, y=y, random_key=keys[0], **initial_proba_init_kwargs
    )
    if init_funcs["initial_proba_init_custom"]:
        _validate_custom_init_output(initial_probs, n_states, "initial_proba_init")

    transition_matrix = transition_proba_init(
        n_states=n_states, X=X, y=y, random_key=keys[1], **transition_proba_init_kwargs
    )
    if init_funcs["transition_proba_init_custom"]:
        _validate_custom_init_output(
            transition_matrix, n_states, "transition_proba_init"
        )

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
                "Dirichlet prior alpha parameters for initial state probabilities "
                f"must have shape ``{expected_shape}``, "
                f"but got shape ``{alphas.shape}``."
            )
        if not jnp.all(alphas >= 1):
            raise ValueError(
                "Dirichlet prior alpha parameters must be >= 1, but got values < 1"
                f":\n{alphas}"
            )
        return alphas
    else:
        raise TypeError(
            f"Invalid type for Dirichlet prior alpha parameters: ``{type(alphas).__name__}``. "
            f"Must be None or an array-like object of shape ``{expected_shape}`` with strictly positive values."
        )
