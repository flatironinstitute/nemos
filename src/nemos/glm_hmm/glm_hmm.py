"""API for the GLM-HMM model."""

from numbers import Number
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .. import tree_utils, validation
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor
from ..glm import GLM
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..pytrees import FeaturePytree
from ..regularizer import Regularizer
from ..third_party.jaxopt import jaxopt
from ..type_casting import (
    is_numpy_array_like,
    is_pynapple_tsd,
    jnp_asarray_if,
)
from ..typing import DESIGN_INPUT_TYPE, RegularizerStrength
from .expectation_maximization import (
    em_glm_hmm,
    hmm_negative_log_likelihood,
    prepare_likelihood_func,
)
from .initialize_parameters import (
    random_glm_params_init,
    resolve_glm_params_init_function,
    resolve_initial_state_proba_init_function,
    resolve_transition_proba_init_function,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
)

ModelParams = Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]


def compute_is_new_session(time: NDArray, start: NDArray) -> jnp.ndarray:
    """Compute new session indicator vector.

    Parameters
    ----------
    time:
        The timestamp associated to each sample.
    start:
        Start times of each new epoch/session.
    """
    return jax.numpy.zeros_like(time).at[jax.numpy.searchsorted(time, start)].set(1)


class GLMHMM(BaseRegressor[ModelParams]):
    """GLM-HMM model."""

    def __init__(
        self,
        n_states: int,
        observation_model: Observations = "Bernoulli",
        inverse_link_function: Callable = jax.lax.logistic,
        regularizer: Union[
            str, Regularizer
        ] = "UnRegularized",  # this applies only for the regularization of the glm coefficients.
        regularizer_strength: Optional[
            RegularizerStrength
        ] = None,  # this is used to regularize GLM coef.
        # prior to regularize init prob and transition
        dirichlet_prior_alphas_init_prob: jnp.ndarray | None = None,  # (n_state, )
        dirichlet_prior_alphas_transition: (
            jnp.ndarray | None
        ) = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        initialize_init_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ) = uniform_initial_proba_init,
        initialize_transition_proba: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], NDArray] | NDArray | str
        ) = sticky_transition_proba_init,
        initialize_glm_params: (
            Callable[[DESIGN_INPUT_TYPE, NDArray], ModelParams] | ModelParams | str
        ) = random_glm_params_init,
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self._n_states = n_states
        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function

        # check and store initialization hyperparameters.
        self.initialize_glm_params = initialize_glm_params
        self.initialize_transition_proba = initialize_transition_proba
        self._initialize_init_proba = initialize_init_proba

        # set the prior params
        self.dirichlet_prior_alphas_init_prob = dirichlet_prior_alphas_init_prob
        self.dirichlet_prior_alphas_transition = dirichlet_prior_alphas_transition
        self.seed = seed
        self.maxiter = maxiter
        self.tol = tol
        # Model log-likelihood
        # inputs: (y, firing_rate)
        # input shapes:
        #     - y: (n_samples,)
        #     - firing_rate: (n_samples,)
        # returns: a 0-dim jnp.ndarray
        self._likelihood_func: (
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None
        ) = None
        # expected negative log-likelihood as function of the GLM parameters.
        # inputs: ((coef, intercept), X, y, posteriors)
        # input shapes:
        #     - (coef, intercept): ( (n_features, n_states), (n_states,) )
        #     - X: (n_samples, n_features)
        #     - y: (n_samples,)
        #     - posteriors: (n_samples, n_states)
        self._expected_negative_log_likelihood: (
            Callable[
                [
                    Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
                    jnp.ndarray,
                    jnp.ndarray,
                    jnp.ndarray,
                ],
                jnp.ndarray,
            ]
            | None
        ) = None
        # parameters
        self.glm_params_: Tuple[dict | jnp.ndarray, jnp.ndarray] | None = None
        self.transition_prob_: jnp.ndarray | None = None
        self.initial_prob_: jnp.ndarray | None = None

    @property
    def coef_(self):
        """The GLM coefficients."""
        if self.glm_params_ is not None:
            return self.glm_params_[0]
        return None

    @property
    def intercept_(self):
        """The GLM intercepts."""
        if self.glm_params_ is not None:
            return self.glm_params_[1]
        return None

    @property
    def maxiter(self):
        """EM maximum number of iterations."""
        return self._maxiter

    @maxiter.setter
    def maxiter(self, maxiter: int):

        if not isinstance(maxiter, Number) or maxiter != int(maxiter) or maxiter <= 0:
            raise ValueError(
                f"``maxiter`` must be a strictly positive integer. {maxiter} provided."
            )
        self._maxiter = int(maxiter)

    @property
    def tol(self):
        """Tolerance for the EM algorithm convergence criterion.

        The algorithm stops when the absolute change in log-likelihood between
        consecutive iterations falls below this threshold:
        |log_likelihood_current - log_likelihood_previous| < tol

        Returns
        -------
            float: Convergence tolerance value.
        """
        return self._tol

    @tol.setter
    def tol(self, tol: float):

        if not isinstance(tol, Number) or tol <= 0:
            raise ValueError(
                f"``maxiter`` must be a strictly positive float. {tol} provided."
            )
        self._tol = float(tol)

    @property
    def initialize_glm_params(self):
        """Initialization of glm coef and intercept for the GLM."""
        return self._initialize_glm_params

    @initialize_glm_params.setter
    def initialize_glm_params(self, glm_params):
        glm_params = resolve_glm_params_init_function(glm_params)
        check_values = (
            isinstance(glm_params, tuple)
            and len(glm_params) == 2
            and all(isinstance(arr, jax.numpy.ndarray) for arr in glm_params)
        )
        if check_values:
            self._check_initial_glm_params(glm_params)
        self._initialize_glm_params = glm_params

    @property
    def initialize_init_proba(self):
        """Initialization of initial state probabilities."""
        return self._initialize_init_proba

    @initialize_init_proba.setter
    def initialize_init_proba(self, initial_prob):
        initial_prob = resolve_initial_state_proba_init_function(initial_prob)
        if isinstance(initial_prob, jnp.ndarray):
            self._check_init_state_proba(initial_prob)
        self._initialize_init_proba = initial_prob

    @property
    def initialize_transition_proba(self):
        """Initialization of the transition probabilities."""
        return self._initialize_transition_proba

    @initialize_transition_proba.setter
    def initialize_transition_proba(self, transition_prob):
        transition_prob = resolve_transition_proba_init_function(transition_prob)
        if isinstance(transition_prob, jnp.ndarray):
            self._check_init_state_proba(transition_prob)
        self._initialize_transition_proba = transition_prob

    @property
    def dirichlet_prior_alphas_init_prob(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_init_prob

    @dirichlet_prior_alphas_init_prob.setter
    def dirichlet_prior_alphas_init_prob(self, value: jnp.ndarray | None):
        if value is None:
            self._dirichlet_prior_alphas_init_prob = None
        elif is_numpy_array_like(value)[1]:
            value = jnp.asarray(value, dtype=float)
            if value.shape != (self._n_states,):
                raise ValueError(
                    f"Dirichlet prior alpha parameters for initial state probabilities "
                    f"must have shape ({self._n_states},), "
                    f"but got shape {value.shape}."
                )
            if not jnp.all(value > 0):
                raise ValueError(
                    f"Dirichlet prior alpha parameters must be strictly positive, but got values with "
                    f"zero or negative entries: {value}"
                )
            self._dirichlet_prior_alphas_init_prob = value
        else:
            raise TypeError(
                f"Invalid type for Dirichlet prior alpha parameters: {type(value).__name__}. "
                f"Must be None or an array-like object of shape ({self._n_states},) with strictly positive values."
            )

    @property
    def dirichlet_prior_alphas_transition(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_transition

    @dirichlet_prior_alphas_transition.setter
    def dirichlet_prior_alphas_transition(self, value: jnp.ndarray | None):
        if value is None:
            self._dirichlet_prior_alphas_transition = None
        elif is_numpy_array_like(value)[1]:
            value = jnp.asarray(value, dtype=float)
            if value.shape != (self._n_states, self.n_states):
                raise ValueError(
                    "Dirichlet prior alpha parameters for transition probabilities must "
                    f"have shape ({self._n_states}, {self._n_states}), "
                    f"but got shape {value.shape}."
                )
            if not jnp.all(value > 0):
                raise ValueError(
                    f"Dirichlet prior alpha parameters must be strictly positive, but got values with "
                    f"zero or negative entries: {value}"
                )
            self._dirichlet_prior_alphas_transition = value
        else:
            raise TypeError(
                f"Invalid type for Dirichlet prior alpha parameters for transition probabilities: "
                f"{type(value).__name__}. "
                f"Must be None or an array-like object of shape ({self._n_states}, {self._n_states}) "
                f"with strictly positive values."
            )

    def _check_initial_glm_params(
        self,
        params: Tuple[DESIGN_INPUT_TYPE | dict, jax.numpy.ndarray],
    ):
        coef, intercept = params
        # check the dimensionality of coeff
        err_message_shape = (
            "params[0] (GLM coefficients) must be a two-dimensional array "
            f"or nemos.pytree.FeaturePytree with array leafs of shape "
            f"(n_features, {self._n_states})."
        )
        validation.check_tree_leaves_dimensionality(
            params[0],
            expected_dim=2,
            err_message=err_message_shape,
        )

        shape_coef = set(
            jax.tree_util.tree_map(
                lambda x: x.shape[1], jax.tree_util.tree_leaves(params[0])
            )
        )
        if len(shape_coef) != 1 or shape_coef != {self._n_states}:
            raise ValueError(err_message_shape)

        if not isinstance(intercept, jnp.ndarray):
            raise ValueError(
                f"params[1] (intercept) should be a 1-dimensional array of shape ({self._n_states},). "
                f"Provided params[1] is of type {type(intercept)} instead."
            )
        elif not intercept.shape == (self._n_states,):
            raise ValueError(
                f"params[1] (intercept) should be a 1-dimensional array of shape ({self._n_states},). "
                f"Provided params[1] is of shape {intercept.shape} instead."
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

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[
            Tuple[Union[dict, ArrayLike], ArrayLike, ArrayLike, ArrayLike]
        ] = None,
    ) -> "GLMHMM":
        """Fit the GLM-HMM model to the data."""

        # define new session array
        if is_pynapple_tsd(y):
            is_new_session = compute_is_new_session(y.t, y.time_support.start)
        elif is_pynapple_tsd(X):
            is_new_session = compute_is_new_session(X.t, X.time_support.start)
        else:
            is_new_session = None

        # cast to jax
        X, y = jax.tree_util.tree_map(lambda x: jnp_asarray_if(x, dtype=float), (X, y))

        # validate the inputs & initialize solver
        init_params = self.initialize_params(X, y, init_params=init_params)

        # find non-nans
        is_valid = tree_utils.get_valid_multitree(X, y)

        # drop nans
        X = jax.tree_util.tree_map(lambda x: x[is_valid], X)
        y = jax.tree_util.tree_map(lambda x: x[is_valid], y)

        # grab data if needed (tree map won't function because param is never a FeaturePytree).
        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        self._likelihood_func, self._expected_negative_log_likelihood = (
            self._get_m_step_loss_function(y.ndim > 1)
        )
        self.initialize_state(data, y, init_params=init_params)

        # run EM
        glm_params, transition_prob, initial_prob = init_params
        (
            _,
            _,
            self.initial_prob_,
            self.transition_prob_,
            self.glm_params_,
        ) = em_glm_hmm(
            data,
            y,
            initial_prob,
            transition_prob,
            glm_params,
            self._inverse_link_function,
            self._likelihood_func,
            self._solver_run,
            is_new_session,
            self._maxiter,
            self._tol,
        )
        return self

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
        init_params: Optional[
            Tuple[Tuple[ArrayLike, ArrayLike], ArrayLike, ArrayLike]
        ] = None,
    ) -> Union[Any, NamedTuple]:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization.

        TODO: fill up docstrings.
        """
        return super().initialize_params(X, y, init_params)

    def _initialize_parameters(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> ModelParams:
        """GLM-HMM initialization."""

        key = self.seed
        init_glm_params = self._initialize_glm_params
        if callable(init_glm_params):
            key, subkey = jax.random.split(key)
            coef, intercept = init_glm_params(
                1 if y.ndim == 1 else y.shape[1], self._n_states, X, subkey
            )
            if y.ndim == 1:
                coef, intercept = jnp.squeeze(coef), jnp.squeeze(intercept)
            init_glm_params = coef, intercept

        init_transition_proba = self._initialize_transition_proba
        if callable(init_transition_proba):
            key, subkey = jax.random.split(key)
            init_transition_proba = init_transition_proba(self._n_states, subkey)

        init_proba = self._initialize_init_proba
        if callable(init_proba):
            key, subkey = jax.random.split(key)
            init_proba = init_proba(self._n_states, subkey)

        return init_glm_params, init_transition_proba, init_proba

    def _get_m_step_loss_function(self, is_population_glm: bool):
        """Prepare the loss function for the M-step of the GLM params."""
        # prepare the loss function
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            is_population_glm,
            self.observation_model.log_likelihood,
            self.observation_model._negative_log_likelihood,
            is_log=True,
        )
        inverse_link_function = self.inverse_link_function

        # closure for the static callable solver.run
        # NOTE: this is the loss function used in the numerical M-step that learns coefficient and intercept.
        def expected_negative_log_likelihood(
            glm_params, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                glm_params,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        return likelihood_func, expected_negative_log_likelihood

    def initialize_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params,
        cast_to_jax_and_drop_nans: bool = True,
    ) -> Union[Any, NamedTuple]:
        """Initialize the state of the solver for running fit and update."""
        X, y = self._preprocess_inputs(
            X, y, cast_to_jax_and_drop_nans=cast_to_jax_and_drop_nans
        )
        if self._expected_negative_log_likelihood is None:
            self._likelihood_func, self._expected_negative_log_likelihood = (
                self._get_m_step_loss_function(y.ndim > 1)
            )
        self.instantiate_solver(self._expected_negative_log_likelihood)
        opt_state = self.solver_init_state(init_params, X, y)
        return opt_state

    # CHECKS FOR PARAMS AND INPUTS
    def _check_params(
        self,
        params: Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike],
        data_type: Optional[jnp.dtype] = None,
    ) -> Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        # check that params has length 4 (coeff, intercept, transition_proba, initial_proba)
        validation.check_length(
            params,
            3,
            "GLM-HMM requires three parameters: "
            "``(glm_params, transition_proba, initial_proba)``.\n ``glm_params`` must be a "
            "length 2 tuple ``(coef, intercept)``.",
        )
        validation.check_length(
            params[0],
            2,
            "The GLM params must be a length two tuple, ``(coef, intercept)``.",
        )
        # convert to jax array (specify type if needed)
        params = validation.convert_tree_leaves_to_jax_array(
            params,
            "Initial parameters must be array-like objects (or pytrees of array-like objects) "
            "with numeric data-type!",
            data_type,
        )
        self._check_initial_glm_params(params[0])
        self._check_transition_proba(params[1])
        self._check_init_state_proba(params[2])
        return params

    @staticmethod
    def _check_input_dimensionality(
        X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        GLM._check_input_dimensionality(X=X, y=y)

    @staticmethod
    def _check_input_and_params_consistency(
        params: ModelParams,  # TODO: Add signature
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
        if X is not None:
            # check that X and params[0] have the same structure
            if isinstance(X, FeaturePytree):
                data = X.data
            else:
                data = X
            coef, intercept = params[0]
            struct1 = jax.tree_util.tree_structure(data)
            struct2 = jax.tree_util.tree_structure(coef)
            if struct1 != struct2:
                raise ValueError(
                    f"X GLM coefficients must be the tree with the same structure.\n"
                    f"X has structure {struct1} and coefficients have structure {struct2}. "
                )

            # check the consistency of the feature axis
            validation.check_tree_axis_consistency(
                coef,
                data,
                axis_1=0,
                axis_2=1,
                err_message="Inconsistent number of features. "
                f"GLM coefficients have {jax.tree_util.tree_map(lambda p: p.shape[0], coef)} features, "
                f"X has {jax.tree_util.tree_map(lambda x: x.shape[1], X)} features instead!",
            )

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

    def _get_coef_and_intercept(self) -> Tuple[Any, Any]:
        if self.glm_params_ is not None:
            return self.glm_params_
        return None, None

    def _set_coef_and_intercept(self, params):
        self.glm_params_ = params

    def update(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jaxopt.OptStep:
        """Run a single update step of the jaxopt solver."""
        pass
