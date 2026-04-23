"""Initialization functions and related utility functions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..glm import GLM, PopulationGLM
from ..glm.initialize_parameters import initialize_intercept_matching_mean_rate
from ..glm.params import GLMUserParams
from ..hmm.initialize_parameters import KMeansInitializer
from ..type_casting import cast_to_jax
from ..typing import DESIGN_INPUT_TYPE

RANDOM_KEY = jax.Array


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
    is_new_session :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    minimum_prob :
        Minimum probability added to each state to avoid zero probabilities.
        Note that probabilities will be renormalized after adding this minimum value, so the final
        probabilities will not be exactly this value.
    inverse_link_function :
        Inverse link function of the GLM.
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
        self.inverse_link_function = inverse_link_function
        self.glm_kwargs = glm_kwargs if glm_kwargs is not None else {}

    def glm_params(self) -> GLMUserParams:
        """Generate glm parameters for initialization."""
        if isinstance(self.random_key, int):
            key = jax.random.PRNGKey(self.random_key)
        sub, _ = jax.random.split(key)
        states = self.states.astype(bool)
        is_one_dim = self._y.ndim == 1
        coef, intercept = random_glm_params_init(
            states.shape[1],
            self._X,
            self._y,
            self.inverse_link_function,
            random_key=sub,
            std_dev=0.0,
        )
        if is_one_dim:
            model = GLM(**self.glm_kwargs)
        else:
            model = PopulationGLM(**self.glm_kwargs)
        # initialize
        for i, state_mask in enumerate(states.T):
            X_state, y_state = self._X[state_mask], self._y[state_mask]
            model.fit(X_state, y_state)
            coef = coef.at[:, i].set(model.coef_)
            intercept = intercept.at[i : i + 1].set(model.intercept_)

        return coef, intercept
