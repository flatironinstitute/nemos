"""API for the GLM-HMM model."""

from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from ..base_regressor import BaseRegressor
from ..observation_models import Observations
from ..regularizer import Regularizer
from ..typing import DESIGN_INPUT_TYPE, RegularizerStrength


def random_projection_init(
    n_states: int, X: DESIGN_INPUT_TYPE, y: NDArray, random_key=jax.random.PRNGKey(123)
):
    """Initialize projections."""
    n_features = X.shape[1]
    return 0.1 * jax.random.normal(random_key, (n_features, n_states))


def sticky_transition_proba_init(n_states: int, prob_stay=0.95):
    """Initialize transition probabilities."""
    # assume n_state is > 1
    prob_leave = (1 - prob_stay) / (n_states - 1)
    return jnp.full((n_states, n_states), prob_leave) + jnp.diag(
        (prob_stay - prob_leave) * jnp.ones(n_states)
    )


def uniform_initial_proba_init(n_states: int, random_key=jax.random.PRNGKey(124)):
    """Initialize initial state probabilities."""
    prob = jax.random.uniform(random_key, (n_states,), minval=0, maxval=1)
    return prob / jnp.sum(prob)


class GLMHMM(BaseRegressor):
    """GLM-HMM model."""

    def __init__(
        self,
        n_states: int,
        observation_model: Observations = "Bernoulli",
        inverse_link_function: Callable = jax.lax.logistic,
        regularizer: Union[
            str, Regularizer
        ] = "UnRegularized",  # how does it work for the analytical M-step?
        # do all regularization make sense for this?
        regularizer_strength: Optional[
            RegularizerStrength
        ] = None,  # do we regularize all params or only projection?
        # - there is regularization but doesn't follow the current logic.
        # - one prior for transition and initial proba, to get an analytical m-step still
        dirichlet_prior_init_state: jnp.ndarray | None = None,  # (n_state, )
        dirichlet_prior_transition: jnp.ndarray | None = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        initialize_init_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | Literal
        ) = uniform_initial_proba_init,
        initialize_transition_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | Literal
        ) = sticky_transition_proba_init,
        initialize_projections: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | Literal
        ) = random_projection_init,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

    def fit(
        self, X: DESIGN_INPUT_TYPE, y: Union[NDArray, jnp.ndarray, nap.Tsd]
    ) -> "GLMHMM":
        """Fit the GLM-HMM model to the data."""
        pass

    def predict(
        self,
        X: DESIGN_INPUT_TYPE,
        predict_type: Literal["most-likely", "per-state", "average"] = "most-likely",
    ) -> jnp.ndarray | nap.Tsd | nap.TsdFrame:
        """Compute predicted firing rate pet state."""
        pass

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        """Compute the model score."""
        pass

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[nap.Tsd, nap.Tsd]:
        """Simulate spikes from the model, returns neural activity and states."""
        pass

    def predict_proba(
        self,  #
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: NDArray,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the smoothing posteriors over states."""
        pass

    def decode_state(
        self, X: Union[DESIGN_INPUT_TYPE, ArrayLike], y: ArrayLike
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the most likely states over samples."""
        pass

    def _predict_and_compute_loss(
        self,
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        pass

    # INITIALIZATIONS
    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        params: Optional = None,
    ) -> Union[Any, NamedTuple]:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization."""
        pass

    def initialize_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params,
    ) -> Union[Any, NamedTuple]:
        """Initialize the state of the solver for running fit and update."""
        pass

    # CHECKS FOR PARAMS AND INPUTS
    @staticmethod
    def _check_params(
        params: Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike],
        data_type: Optional[jnp.dtype] = None,
    ) -> Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        pass

    @staticmethod
    def _check_input_dimensionality(
        X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        pass

    @staticmethod
    def _check_input_and_params_consistency(
        params,  # TODO: Add signature
        X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """Validate the number of features in model parameters and input arguments.

        Raises
        ------
        ValueError
            - if the number of features is inconsistent between params[1] and X
              (when provided).

        """
        pass

    # Save
    def save_params(
        self,
        filename: Union[str, Path],
        fit_attrs: dict,
        string_attrs: list = None,
    ):
        """Save model params."""
        pass

    # SVRG specific optimization not available.
    def _get_optimal_solver_params_config(self):
        """No optimal parameters known for SVRG in HMMGLM."""
        return None, None, None
