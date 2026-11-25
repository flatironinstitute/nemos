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
from ..observation_models import Observations
from ..pytrees import FeaturePytree
from ..type_casting import cast_to_jax
from ..typing import DESIGN_INPUT_TYPE
from .algorithm_configs import has_no_extra_params


@cast_to_jax(dtype=None)
def random_glm_params_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    inverse_link_function: Callable,
    observation_model: Observations,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: Optional[jax.Array] = None,
    std_dev=0.001,
) -> GLMUserParams:
    """
    Initialize GLM coefficients and intercept with random normal values.

    Generates random GLM parameters for each HMM state by sampling from a normal
    distribution scaled by `std_dev`.

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
    observation_model :
        Observation model defining the valid range for the initial mean.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions.
        Unused for this initialization, but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key for reproducibility. Default is PRNGKey(123).
    std_dev :
        The standard deviation of the normal distribution that generates the coefficients.
        Default is 0.001.

    Returns
    -------
    coef : Any
        Coefficient, a pytree with the same structure as X with leaves
         of shape (n_features, n_neurons, n_states).
    intercept : jnp.ndarray
        Intercept array of shape (n_neurons, n_states).
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(123)
    is_one_dim = y.ndim == 1
    n_neurons = 1 if is_one_dim else y.shape[1]
    if isinstance(X, FeaturePytree):
        X = X.data
    # small random noisy coef
    coef = jax.tree_util.tree_map(
        lambda x: (
            std_dev * jax.random.normal(random_key, (x.shape[1], n_neurons, n_states))
        ),
        X,
    )
    # mean-rate
    intercept = initialize_intercept_matching_mean_rate(
        inverse_link_function,
        X,
        y,
        rate_range=observation_model.rate_range,
    )
    intercept = jnp.tile(intercept[:, jnp.newaxis], (1, n_states))
    if is_one_dim:
        coef = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), coef)
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
    observation_model :
        Observation model of the GLM.
    session_starts :
        Optional boolean array of shape (n_samples,) indicating the start of new sessions. If None
        (default), it is assumed that all data belongs to a single session.
    glm_kwargs:
        Keyword arguments defining the GLM model: observation model, regularization etc.
    random_key :
        Random key for reproducibility of KMeans initialization.
    """

    def __init__(
        self,
        n_states: int,
        X: DESIGN_INPUT_TYPE,
        y: NDArray | jnp.ndarray,
        inverse_link_function: Callable,
        observation_model: Observations | str,
        session_starts: Optional[jnp.ndarray] = None,
        glm_kwargs: Optional[Dict[str, Any]] = None,
        random_key: int | jax.Array = 0,
    ):
        super().__init__(
            n_states,
            X,
            y,
            session_starts,
            random_key=random_key,
        )
        self._X = jax.tree_util.tree_map(jnp.asarray, X)
        if isinstance(self._X, FeaturePytree):
            self._X = self._X.data
        self._y = jnp.asarray(y)
        self.inverse_link_function = inverse_link_function
        self.observation_model = observation_model
        self.glm_kwargs = glm_kwargs if glm_kwargs is not None else {}
        if self._y.ndim == 1:
            self._glm_models = {
                i: GLM(
                    observation_model=observation_model,
                    inverse_link_function=inverse_link_function,
                    **self.glm_kwargs,
                )
                for i in range(self.n_states)
            }
            self._is_population = False
        else:
            self._glm_models = {
                i: PopulationGLM(
                    observation_model=observation_model,
                    inverse_link_function=inverse_link_function,
                    **self.glm_kwargs,
                )
                for i in range(self.n_states)
            }
            self._is_population = True

    def fit(self) -> "KMeansInitializerGLM":
        """Fit one GLM per state on samples assigned to that state by KMeans."""
        states = self.states.astype(bool)
        for i, (state_mask, model) in enumerate(
            zip(states.T, self._glm_models.values())
        ):
            X_state, y_state = self._X[state_mask], self._y[state_mask]
            if X_state.shape[0] == 0:
                raise ValueError(
                    f"KMeans assigned 0 samples to state {i}. Try reducing n_states "
                    f"or using a different initialization method."
                )
            model.fit(X_state, y_state)
        return self

    def glm_params(self) -> GLMUserParams:
        """Generate GLM parameters for initialization."""
        if self._glm_models[0].coef_ is None:
            self.fit()
        key = jax.random.PRNGKey(self.random_key)
        states = self.states.astype(bool)
        coef, intercept = random_glm_params_init(
            states.shape[1],
            self._X,
            self._y,
            self.inverse_link_function,
            self._glm_models[0].observation_model,
            std_dev=0.0,
            random_key=key,
        )
        for i, m in enumerate(self._glm_models.values()):
            coef = jax.tree_util.tree_map(
                lambda c, mc: c.at[..., i].set(mc), coef, m.coef_
            )
            if self._is_population:
                intercept = intercept.at[..., i].set(m.intercept_)
            else:
                intercept = intercept.at[i : i + 1].set(m.intercept_)
        return coef, intercept

    def scale(self) -> jnp.ndarray:
        """KMeans-based scale estimate."""
        is_population = self._y.ndim > 1
        if is_population:
            n_neurons = self._y.shape[1]
            scale = jnp.ones((n_neurons, self.n_states))
        else:
            scale = jnp.ones((self.n_states,))

        if has_no_extra_params(self._glm_models[0].observation_model):
            return scale

        if self._glm_models[0].scale_ is None:
            self.fit()

        for i, m in enumerate(self._glm_models.values()):
            scale = scale.at[..., i].set(m.scale_)
        return scale


@cast_to_jax(dtype=None)
def constant_parameter_init(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    observation_model: Optional[Observations] = None,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: Optional[jax.Array] = None,
    fill_value: float = 1.0,
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
    inverse_link_function :
        Inverse link function of the GLM. Unused for this initialization, but included for API consistency.
    observation_model :
        Observation model of the GLM. Unused for this initialization, but included for API consistency.
    session_starts :
        Optional boolean array of shape (n_samples,). Unused for this initialization,
        but included for API consistency.
    random_key : jax.random.PRNGKey
        Random key, unused for this initialization, but included for API consistency.
    fill_value : float
        The constant value to initialize all scale parameters to. Default is 1.0.

    Returns
    -------
    scale : jnp.ndarray
        Scale array of shape (n_states,) for single neuron or (n_neurons, n_states)
        for multiple neurons, with all values set to `scale_val`.
    """
    is_one_dim = y.ndim == 1
    n_neurons = 1 if is_one_dim else y.shape[1]
    param = jnp.full((n_neurons, n_states), fill_value, dtype=float)
    if is_one_dim:
        param = jnp.squeeze(param, axis=0)
    return param


DEFAULT_INIT_FUNCTIONS_GLMHMM = {
    "glm_params_init": random_glm_params_init,
    "scale_init": constant_parameter_init,
    "dispersion_init": constant_parameter_init,
}


def generate_glm_hmm_initial_model_params(
    n_states: int,
    X: DESIGN_INPUT_TYPE,
    y: NDArray | jnp.ndarray,
    inverse_link_function: Callable,
    observation_model: Observations,
    session_starts: Optional[NDArray | jnp.ndarray] = None,
    random_key: int | jax.Array = 123,
) -> GLMUserParams:
    """
    Generate initial GLM model parameters for a GLM-HMM.

    Calls the default initialization functions for GLM coefficients, intercept, and each
    observation-model parameter, splitting the random key to ensure independent random states
    for each component.
    HMM parameters (initial and transition probabilities) are initialized separately via
    :meth:`~nemos.hmm.BaseHMM._hmm_params_initialization`.

    Parameters
    ----------
    n_states :
        Number of HMM states.
    X :
        Predictor data of shape (n_samples, n_features).
    y :
        Output data (e.g., neural activity) of shape (n_samples,) or (n_samples, n_neurons).
    inverse_link_function :
        Inverse link function of the GLM, passed to GLM parameter and scale initialization functions.
    observation_model :
        Observation model of the GLM, forwarded to GLM-parameter and scale init functions.
    session_starts :
        Boolean array of shape (n_samples,) indicating the start of new sessions. If None, a single
        session is assumed.
    random_key :
        Key for random number generation. Split into two independent subkeys for GLM params
        and scale respectively.

    Returns
    -------
    coef :
        GLM coefficients, a pytree matching the structure of X with leaves of shape
        ``(n_features, n_neurons, n_states)`` or ``(n_features, n_states)`` for single-neuron.
    intercept :
        Intercept array of shape ``(n_neurons, n_states)`` or ``(n_states,)`` for single-neuron.
    scale :
        Scale array of shape ``(n_states,)`` for single-neuron or ``(n_neurons, n_states)`` for population.
    """
    if isinstance(random_key, int):
        random_key = jax.random.PRNGKey(random_key)

    glm_params_key, scale_key = jax.random.split(random_key, 2)

    coef, intercept = DEFAULT_INIT_FUNCTIONS_GLMHMM["glm_params_init"](
        n_states=n_states,
        X=X,
        y=y,
        inverse_link_function=inverse_link_function,
        observation_model=observation_model,
        session_starts=session_starts,
        random_key=glm_params_key,
    )
    user_params = {"coef": coef, "intercept": intercept}
    for param_name, param_specs in observation_model.extra_params.items():
        user_params[param_name] = DEFAULT_INIT_FUNCTIONS_GLMHMM[f"{param_name}_init"](
            n_states=n_states,
            X=X,
            y=y,
            inverse_link_function=inverse_link_function,
            observation_model=observation_model,
            session_starts=session_starts,
            random_key=scale_key,
            fill_value=param_specs.default,
        )

    return user_params
