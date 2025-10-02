"""API for the GLM-HMM model."""

from numbers import Number
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..regularizer import Regularizer
from ..typing import DESIGN_INPUT_TYPE, RegularizerStrength
from .parameters_initialization import (
    random_projection_init,
    resolve_initial_state_proba_init_function,
    resolve_projection_init_function,
    resolve_transition_proba_init_function,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
)


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
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ) = uniform_initial_proba_init,
        initialize_transition_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ) = sticky_transition_proba_init,
        initialize_projections: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ) = random_projection_init,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self._n_states = n_states
        self.observation_model = observation_model

        # check and store initialization hyper-parameters.
        self._initialize_projections = resolve_projection_init_function(
            initialize_projections
        )
        if isinstance(self._initialize_projections, jnp.ndarray):
            self._check_initial_proj(self._initialize_projections)

        self._initialize_transition_proba = resolve_transition_proba_init_function(
            initialize_transition_proba
        )
        if isinstance(self._initialize_transition_proba, jnp.ndarray):
            self._check_initial_proj(self._initialize_transition_proba)

        self._initialize_init_state_proba = resolve_initial_state_proba_init_function(
            initialize_init_proba
        )
        if isinstance(self._initialize_init_state_proba, jnp.ndarray):
            self._check_init_state_proba(self._initialize_init_state_proba)

    def _check_initial_proj(
        self, projection_array: jax.numpy.ndarray, X: Optional[DESIGN_INPUT_TYPE] = None
    ):
        if self._n_states != projection_array.shape[1]:
            raise ValueError(
                f"Projection weights shape mismatch: the second dimension must match the number of HMM states.\n"
                f"Expected shape[1] = {self._n_states} (n_states), but got shape[1] = {projection_array.shape[1]}.\n"
                f"Projection weights shape: {projection_array.shape}"
            )
        if X is not None and projection_array.shape[0] != X.shape[1]:
            raise ValueError(
                f"Projection weights shape mismatch: the first dimension must match the number of GLM features.\n"
                f"Expected shape[0] = {X.shape[1]} (n_features), but got shape[0] = {projection_array.shape[0]}.\n"
                f"Projection weights shape: {projection_array.shape}"
            )

    def _check_transition_proba(self, transition_prob: jax.numpy.ndarray):
        if transition_prob.shape != (self._n_states, self._n_states):
            raise ValueError(
                f"Transition probability matrix shape mismatch: expected ({self._n_states}, {self._n_states}), "
                f"but got {transition_prob.shape}."
            )
        if not jnp.allclose(jnp.sum(transition_prob, axis=1), 1):
            row_sums = jnp.sum(transition_prob, axis=1)
            raise ValueError(
                f"Transition probability matrix rows must sum to 1. "
                f"Each row i represents the probability distribution of transitioning from state i. "
                f"Row sums: {row_sums}"
            )

    def _check_init_state_proba(self, initial_prob: jax.numpy.ndarray):
        if initial_prob.shape != (self._n_states,):
            raise ValueError(
                f"Initial state probability vector shape mismatch: expected ({self._n_states},), "
                f"but got {initial_prob.shape}."
            )
        if not jnp.allclose(initial_prob.sum(), 1):
            raise ValueError(
                f"Initial state probabilities must sum to 1, but got sum = {initial_prob.sum()}. "
                f"Probabilities: {initial_prob}"
            )

    @property
    def n_states(self) -> int:
        """Number of hidden states of the HMM."""
        return self._n_states

    @n_states.setter
    def n_states(self, n_states: int):
        if not isinstance(n_states, Number):
            raise TypeError(
                f"n_states must be a positive integer. "
                f"n_states is of type `{type(n_states)}` instead."
            )
        int_n_states = int(n_states)
        if int_n_states != int(n_states):
            raise TypeError(
                f"n_states must be a positive integer. `{n_states}` provided instead."
            )
        if int_n_states < 1:
            raise ValueError(
                f"n_states must be a positive integer. `{n_states}` provided instead."
            )
        self._n_states = int_n_states

    @property
    def observation_model(self) -> Union[None, obs.Observations]:
        """Getter for the ``observation_model`` attribute."""
        return self._observation_model

    @observation_model.setter
    def observation_model(self, observation: obs.Observations):
        if isinstance(observation, str):
            self._observation_model = instantiate_observation_model(observation)
            return
        # check that the model has the required attributes
        # and that the attribute can be called
        obs.check_observation_model(observation)
        self._observation_model = observation

    @property
    def inverse_link_function(self):
        """Getter for the inverse link function for the model."""
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Setter for the inverse link function for the model."""
        self._inverse_link_function = resolve_inverse_link_function(
            inverse_link_function, self._observation_model
        )

    def _set_up_initialization_funcs(
        self,
        initialize_init_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ),
        initialize_transition_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ),
        initialize_projections: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ),
    ):
        pass

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
