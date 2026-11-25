"""Initialization functions and related utility functions for HMMs."""

from types import MappingProxyType
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import lazy_loader as lazy
from numpy.typing import NDArray

from ..type_casting import is_numpy_array_like
from ..typing import DESIGN_INPUT_TYPE
from .params import HMMUserParams
from .utils import initialize_session_starts

sklearn = lazy.load("sklearn")


def sticky_transition_proba_init(
    n_states: int,
    prob_stay: float = 0.95,
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
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
    prob_stay :
        Probability of staying in the current state. Default is 0.95.
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.
    random_key :
        Random key, unused for this particular initialization, but added for API consistency.

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
    session_starts: Optional[NDArray | jnp.ndarray] = None,
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
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.
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
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
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
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.

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


def dirichlet_transition_proba_init(
    n_states: int,
    alphas: Optional[float | NDArray | jnp.ndarray] = None,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Initialize transition probabilities by sampling from a Dirichlet distribution.

    Generates a random transition probability matrix by sampling each row from a Dirichlet distribution with
    specified concentration parameters (alphas).

    Parameters
    ----------
    n_states :
        Number of HMM states. Must be greater than 1.
    alphas :
        Concentration parameters for the Dirichlet distribution. Must be an array of shape (n_states, n_states)
        specifying the alpha for each transition. If None, defaults to 1.0 for all transitions (uniform).
    random_key :
        Random key for reproducibility of random initialization.
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nemos.hmm.initialize_parameters import dirichlet_transition_proba_init
    >>>
    >>> # Generate transition probabilities for 3 states with Dirichlet initialization
    >>> n_states = 3
    >>> alphas = jnp.array(
    ...     [[5.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 5.0]]
    ... )  # favor self-transitions
    >>> transition_matrix = dirichlet_transition_proba_init(
    ...     n_states, alphas=alphas, random_key=jax.random.PRNGKey(0)
    ... )
    >>> transition_matrix  # doctest: +ELLIPSIS
    Array(...)
    """
    if alphas is None:
        alphas = jnp.ones((n_states, n_states))
    prob = jax.random.dirichlet(random_key, alphas)
    return prob / prob.sum(axis=1, keepdims=True)


def uniform_initial_proba_init(
    n_states: int,
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
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
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.
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
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
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
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.

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


def dirichlet_initial_proba_init(
    n_states: int,
    alphas: Optional[float | NDArray | jnp.ndarray] = None,
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(123),
    X: Optional[DESIGN_INPUT_TYPE] = None,
    y: Optional[NDArray | jnp.ndarray] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Initialize initial state probabilities by sampling from a Dirichlet distribution.

    Generates random initial state probabilities by sampling from a Dirichlet distribution with specified
    concentration parameters (alphas).

    Parameters
    ----------
    n_states :
        Number of HMM states. Must be greater than 1.
    alphas :
        Concentration parameters for the Dirichlet distribution. Must be an array of shape (n_states,)
        specifying the alpha for each state. If None, defaults to 1.0 for all states (uniform).
    random_key :
        Random key for reproducibility of random initialization.
    X :
        Optional predictor data. Unused for this particular initialization, but added for API consistency.
    y :
        Optional output data. Unused for this particular initialization, but added for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. Unused for this
        particular initialization, but added for API consistency.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nemos.hmm.initialize_parameters import dirichlet_initial_proba_init
    >>>
    >>> # Generate initial state probabilities for 3 states with Dirichlet initialization
    >>> n_states = 3
    >>> alphas = jnp.array([5.0, 1.0, 1.0])  # favor first state
    >>> init_prob = dirichlet_initial_proba_init(
    ...     n_states, alphas=alphas, random_key=jax.random.PRNGKey(0)
    ... )
    >>> init_prob  # doctest: +ELLIPSIS
    Array(...)
    """
    if alphas is None:
        alphas = jnp.ones((n_states,))
    prob = jax.random.dirichlet(random_key, alphas)
    return prob / prob.sum()


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
    session_starts :
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
        session_starts: Optional[jnp.ndarray] = None,
        random_key: int | jax.Array = 0,
    ):
        if isinstance(random_key, jax.Array):
            random_key = int(jax.random.randint(random_key, (), 0, 2**31 - 1))

        self.n_states = n_states
        self.random_key = random_key
        self.session_starts = initialize_session_starts(X, y, session_starts)
        self.model = sklearn.cluster.KMeans(
            n_clusters=n_states, random_state=random_key
        )
        # concatenate pytree leaves if applicable and append y
        data = jnp.concatenate(
            jax.tree_util.tree_leaves(X) + [y if y.ndim > 1 else y[:, None]], axis=-1
        )
        self.model.fit(data)
        self.states = jax.nn.one_hot(self.model.labels_, num_classes=n_states)

    def initial_probability(self, minimum_prob: float = 0.02):
        """
        Compute initial state probabilities based on KMeans assigned states.

        This takes the average occurrence of each state at the start of sessions to estimate the initial state
        probabilities.

        Parameters
        ----------
        minimum_prob :
            Minimum probability added to each state to avoid zero probabilities.
            Note that probabilities will be renormalized after adding this minimum value, so the final
            probabilities will not be exactly this value.

        Returns
        -------
        initial_probability :
            Initial state probability vector of shape (n_states,) computed from KMeans assigned states.
        """
        initial_probability = self.states[self.session_starts].sum(axis=0)
        # normalize and add minimum_prob to avoid zero probabilities, then renormalize
        initial_probability = (
            initial_probability / initial_probability.sum()
        ) + minimum_prob
        return initial_probability / initial_probability.sum()

    def transition_probability(self, minimum_prob: float = 0.02):
        """
        Compute transition probabilities based on KMeans assigned states.

        This computes the transition probabilities by counting the transitions between states across time points,
        excluding transitions that occur at the start of new sessions.

        Parameters
        ----------
        minimum_prob :
            Minimum probability added to each state to avoid zero probabilities.
            Note that probabilities will be renormalized after adding this minimum value, so the final
            probabilities will not be exactly this value.

        Returns
        -------
        transition_matrix :
            Transition probability matrix of shape (n_states, n_states) computed from KMeans assigned states.
        """
        transition_matrix = (
            self.states[:-1][~self.session_starts[1:]].T
            @ self.states[1:][~self.session_starts[1:]]
        )
        # normalize and add minimum_prob to avoid zero probabilities, then renormalize
        transition_matrix = (
            transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        ) + minimum_prob
        return transition_matrix / transition_matrix.sum(axis=1, keepdims=True)


DEFAULT_INIT_FUNCTIONS = MappingProxyType(
    {
        "initial_proba_init": uniform_initial_proba_init,
        "transition_proba_init": sticky_transition_proba_init,
    }
)


def generate_hmm_initial_params(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    session_starts: NDArray | jnp.ndarray,
    random_key_pair: jax.Array = jax.random.split(jax.random.PRNGKey(1234), 2),
) -> HMMUserParams:
    """
    Generate initial HMM parameters using the default initialization functions.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data (e.g., model design for GLM) of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,).
    session_starts :
        Boolean array of shape (n_samples,) indicating the start of new sessions.
    random_key_pair :
        Optional pair of keys for random number generation, if needed by the initialization functions.
        This function assumes already formatted keys as contract, preventing silent bugs such as keys
        not being formatted appropriately by upstream models.

    Returns
    -------
    :
        ``HMMUserParams`` tuple of ``(initial_probs, transition_matrix)`` where
        ``initial_probs`` has shape ``(n_states,)`` summing to 1, and
        ``transition_matrix`` has shape ``(n_states, n_states)`` with rows summing to 1.
    """
    initial_probs = DEFAULT_INIT_FUNCTIONS["initial_proba_init"](
        n_states=n_states,
        X=X,
        y=y,
        session_starts=session_starts,
        random_key=random_key_pair[0],
    )
    transition_matrix = DEFAULT_INIT_FUNCTIONS["transition_proba_init"](
        n_states=n_states,
        X=X,
        y=y,
        session_starts=session_starts,
        random_key=random_key_pair[1],
    )

    return {"initial_prob": initial_probs, "transition_prob": transition_matrix}


def _resolve_dirichlet_priors(
    alphas: Any, expected_shape: Tuple[int, ...]
) -> jnp.ndarray | None:
    """Validate and convert Dirichlet prior alpha parameters.

    Parameters
    ----------
    alphas :
        Dirichlet prior alpha parameters. Can be None or array-like, including
        nested sequences (lists, tuples), NumPy and JAX arrays.
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
        If alphas is not None and cannot be converted to a numeric array.
    """
    if alphas is None:
        return None

    # unwrap pandas-like containers; sequences are left untouched and converted below
    unwrapped, _ = is_numpy_array_like(alphas)
    try:
        alphas = jnp.asarray(unwrapped, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(
            f"Invalid type for Dirichlet prior alpha parameters: ``{type(alphas).__name__}``. "
            f"Must be None or an array-like object of shape ``{expected_shape}`` with strictly positive values."
        )

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
