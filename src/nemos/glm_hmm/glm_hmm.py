"""API for the GLM-HMM model."""

import warnings
from numbers import Number
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .. import tree_utils
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..regularizer import GroupLasso, Lasso, Regularizer, Ridge
from ..type_casting import cast_to_jax, is_pynapple_tsd
from ..typing import (
    DESIGN_INPUT_TYPE,
    FeaturePytree,
    RegularizerStrength,
    SolverState,
    StepResult,
)
from ..utils import format_repr
from .algorithm_configs import (
    get_analytical_scale_update,
    prepare_estep_log_likelihood,
    prepare_mstep_nll_objective_param,
    prepare_mstep_nll_objective_scale,
)
from .expectation_maximization import (
    GLMHMMState,
    em_glm_hmm,
    em_step,
    initialize_new_session,
)
from .initialize_parameters import (
    INITIALIZATION_FN_DICT,
    _is_native_init_registry,
    _resolve_init_funcs_registry,
    glm_hmm_initialization,
    resolve_dirichlet_priors,
)
from .params import GLMHMMParams, GLMHMMUserParams, GLMParams, GLMScale, HMMParams
from .validation import GLMHMMValidator


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


class GLMHMM(BaseRegressor[GLMHMMUserParams, GLMHMMParams]):
    """GLM-HMM model."""

    def __init__(
        self,
        n_states: int,
        observation_model: (
            Observations
            | Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial", "Gaussian"]
        ) = "Bernoulli",
        inverse_link_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[
            RegularizerStrength
        ] = None,  # this is used to regularize GLM coef.
        # prior to regularize init prob and transition
        dirichlet_prior_alphas_init_prob: Union[
            jnp.ndarray, None
        ] = None,  # (n_state, )
        dirichlet_prior_alphas_transition: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        initialization_funcs: Optional[INITIALIZATION_FN_DICT] = None,
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
        self.n_states = n_states
        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function

        # assign defaults to initialization functions
        self.initialization_funcs = initialization_funcs

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

        # fit attributes
        self.transition_prob_: jnp.ndarray | None = None
        self.initial_prob_: jnp.ndarray | None = None
        self.coef_: jnp.ndarray | None = None
        self.intercept_: jnp.ndarray | None = None
        self.solver_state_: NamedTuple | None = None
        self.scale_: jnp.ndarray | None = None
        self.dof_resid_: int | None = None

    @property
    def n_states(self) -> int:
        """Number of hidden states of the HMM."""
        return self._n_states

    @n_states.setter
    def n_states(self, n_states: int):
        # quick sanity check and assignment
        if isinstance(n_states, int) and n_states > 0:
            self._n_states = n_states
            self._validator: GLMHMMValidator = GLMHMMValidator(n_states=n_states)
            return

        # further checks for other valid numeric types (like non-negative float with no-decimals)
        if not isinstance(n_states, Number):
            raise TypeError(
                f"n_states must be a positive integer. "
                f"n_states is of type ``{type(n_states)}`` instead."
            )

        # provided a non-integer number (check that has no decimals)
        int_n_states = int(n_states)
        if int_n_states != int(n_states):
            raise TypeError(
                f"n_states must be a positive integer. ``{n_states}`` provided instead."
            )
        elif int_n_states < 1:
            raise ValueError(
                f"n_states must be a positive integer. ``{n_states}`` provided instead."
            )
        self._n_states = int_n_states
        self._validator = GLMHMMValidator(n_states=n_states)

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
                f"``tol`` must be a strictly positive float. {tol} provided."
            )
        self._tol = float(tol)

    @property
    def initialization_funcs(self):
        """Dictionary of initialization functions for model parameters."""
        return self._initialization_funcs

    @initialization_funcs.setter
    def initialization_funcs(self, initialization_funcs: INITIALIZATION_FN_DICT):
        self._initialization_funcs = _resolve_init_funcs_registry(initialization_funcs)

    @property
    def dirichlet_prior_alphas_init_prob(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_init_prob

    @dirichlet_prior_alphas_init_prob.setter
    def dirichlet_prior_alphas_init_prob(self, value: jnp.ndarray | None):
        self._dirichlet_prior_alphas_init_prob = resolve_dirichlet_priors(
            value, (self._n_states,)
        )

    @property
    def dirichlet_prior_alphas_transition(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_transition

    @dirichlet_prior_alphas_transition.setter
    def dirichlet_prior_alphas_transition(self, value: jnp.ndarray | None):
        self._dirichlet_prior_alphas_transition = resolve_dirichlet_priors(
            value, (self._n_states, self._n_states)
        )

    @property
    def observation_model(self) -> obs.Observations:
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

    @property
    def seed(self):
        """Random seed as a jax PRNG key."""
        return self._seed

    @seed.setter
    def seed(self, value):
        value = jnp.asarray(value)
        # Validate it's a JAX PRNG key
        if value.shape != (2,) or value.dtype != jnp.uint32:
            raise TypeError(
                f"seed must be a JAX PRNG key (jax.random.PRNGKey). "
                f"Got {type(value)} with shape {getattr(value, 'shape', 'N/A')}"
            )
        self._seed = value

    def _model_specific_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> GLMHMMParams:
        """GLM-HMM initialization."""
        user_params = glm_hmm_initialization(self._n_states, X, y, self._seed)

        # check if registry uses NeMoS init funcs
        is_nemos_init = _is_native_init_registry(self._initialization_funcs)
        if is_nemos_init:
            # skip validation and just cast
            return self._validator.to_model_params(user_params)

        # params casting with validation
        model_params = self._validator.validate_and_cast_params(user_params)
        self._validator.validate_consistency(model_params, X=X, y=y)
        return model_params

    def _initialize_optimization_and_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params: GLMHMMParams,
    ) -> SolverState:
        """Initialize the EM functions."""
        # glm params m-step setup
        is_population = y.ndim > 1
        log_likelihood = prepare_estep_log_likelihood(
            is_population_glm=is_population,
            observation_model=self.observation_model,
        )
        objective = prepare_mstep_nll_objective_param(
            is_population_glm=is_population,
            observation_model=self._observation_model,
            inverse_link_function=self._inverse_link_function,
        )
        _, _, glm_params_update_fn = self._instantiate_solver(objective)

        scale_update_fn = get_analytical_scale_update(
            is_population_glm=is_population, observation_model=self._observation_model
        )
        if scale_update_fn is None:
            objective_scale = prepare_mstep_nll_objective_scale(
                is_population_glm=is_population,
                observation_model=self._observation_model,
            )
            _, _, scale_update_fn = self._instantiate_solver(objective_scale)

        # cannot wrap is_new_session, that's to be calculated at each update form the provided X and y.
        # for consistency, do not make a partial of that argument in run as well.
        self._optimization_run = eqx.Partial(
            em_glm_hmm,
            inverse_link_function=self.inverse_link_function,
            log_likelihood_func=log_likelihood,
            m_step_fn_glm_params=glm_params_update_fn,
            m_step_fn_glm_scale=scale_update_fn,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        self._optimization_update = eqx.Partial(
            em_step,
            inverse_link_function=self.inverse_link_function,
            log_likelihood_func=log_likelihood,
            m_step_fn_glm_params=glm_params_update_fn,
            m_step_fn_glm_scale=scale_update_fn,
        )

        def init_state_fn(*args, **kwargs) -> SolverState:
            state = GLMHMMState(
                data_log_likelihood=-jnp.array(jnp.inf),
                previous_data_log_likelihood=-jnp.array(jnp.inf),
                log_likelihood_history=jnp.full(self.maxiter, jnp.nan),
                iterations=0,
            )
            return state

        self._optimization_init_state = init_state_fn
        return init_state_fn()

    @staticmethod
    def _get_is_new_session(
        X: DESIGN_INPUT_TYPE, y: ArrayLike | nap.Tsd | nap.TsdFrame
    ) -> jnp.ndarray | None:
        # define new session array
        if is_pynapple_tsd(y):
            is_new_session = compute_is_new_session(y.t, y.time_support.start)
        elif is_pynapple_tsd(X):
            is_new_session = compute_is_new_session(X.t, X.time_support.start)
        else:
            is_new_session = initialize_new_session(y.shape[0], None)
        return is_new_session

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
    ) -> "GLMHMM":
        """Fit the GLM-HMM model to the data."""
        self._validator.validate_inputs(X=X, y=y)
        is_new_session = self._get_is_new_session(X, y)

        # validate the inputs & initialize solver
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )

        # filter for non-nans, grab data if needed
        data, y, is_new_session = self._preprocess_inputs(X, y, is_new_session)

        # make sure is_new_session starts with a 1
        is_new_session = is_new_session.at[0].set(True)

        # set up optimization
        self._initialize_optimization_and_state(data, y, init_params)

        # run EM
        (
            fit_params,
            self.solver_state_,
        ) = self._optimization_run(
            init_params, X=data, y=y, is_new_session=is_new_session
        )

        if self.solver_state_.iterations == self.maxiter:
            warnings.warn(
                "The fit did not converge. "
                "Consider the following:"
                "\n1) Enable float64 with ``jax.config.update('jax_enable_x64', True)``"
                "\n2) Increase the ``maxiter`` parameter (max number of iterations of the EM) "
                "or increase the ``tol`` parameter (tolerance).",
                RuntimeWarning,
            )

        # assign fit attributes
        self._set_model_params(fit_params)
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(X)
        return self

    def _estimate_resid_degrees_of_freedom(
        self, X: DESIGN_INPUT_TYPE, n_samples: Optional[int] = None
    ):
        """
        Estimate the degrees of freedom of the residuals.

        Parameters
        ----------
        self :
            A fitted GLM model.
        X :
            The design matrix.
        n_samples :
            The number of samples observed. If not provided, n_samples is set to ``X.shape[0]``. If the fit is
            batched, the n_samples could be larger than ``X.shape[0]``.

        Returns
        -------
        :
            An estimate of the degrees of freedom of the residuals.
        """
        # Convert a pytree to a design-matrix with pytrees
        X = jnp.hstack(jax.tree_util.tree_leaves(X))

        if n_samples is None:
            n_samples = X.shape[0]
        else:
            if not isinstance(n_samples, int):
                raise TypeError(
                    "`n_samples` must either `None` or of type `int`. Type {type(n_sample)} provided "
                    "instead!"
                )

        params = self._get_model_params()
        coef = params.glm_params.coef
        if coef.ndim == 3:
            n_neurons = coef.shape[1]
        else:
            n_neurons = 1

        dof_intercept_and_hmm = (
            self._n_states * n_neurons  # intercept
            + (
                self._n_states - 1
            )  # init prob (n values but sum to 1, so n-1 free values)
            + (self._n_states - 1) * self._n_states
        )  # transition n n-dim vectors that sum to 1

        # if the regularizer is lasso use the non-zero
        # coef as an estimate of the dof
        # see https://arxiv.org/abs/0712.0881
        if isinstance(self.regularizer, (GroupLasso, Lasso)):
            resid_dof = sum(
                tree_utils.pytree_map_and_reduce(
                    lambda x: ~jnp.isclose(x, jnp.zeros_like(x)),
                    lambda x: sum([jnp.sum(i, axis=0) for i in x]),
                    coef,
                )
            )
            return n_samples - resid_dof - dof_intercept_and_hmm
        elif isinstance(self.regularizer, Ridge):
            # for Ridge, use the tot parameters (X.shape[1] + intercept)
            return (
                n_samples - (X.shape[1] * self.n_states) - dof_intercept_and_hmm
            ) * jnp.ones(n_neurons)
        else:
            # for UnRegularized, use the rank
            rank = jnp.linalg.matrix_rank(X)
            return (n_samples - rank - dof_intercept_and_hmm) * jnp.ones(n_neurons)

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

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: NDArray,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the smoothing posteriors over-states."""
        pass

    def filter_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: NDArray,
    ):
        """Compute the filtering posteriors over-states."""
        pass

    def decode_state(
        self, X: Union[DESIGN_INPUT_TYPE, ArrayLike], y: ArrayLike
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the most likely states over samples."""
        pass

    def _compute_loss(
        self,
        params: GLMHMMParams,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Loss function (expected HMM log-likelihood) without validation."""
        pass

    def save_params(
        self,
        filename: Union[str, Path],
    ):
        """Save model params."""
        # initialize saving dictionary
        fit_attrs = self._get_fit_state()
        fit_attrs.pop("solver_state_")
        string_attrs = ["inverse_link_function"]
        self._save_params(filename, fit_attrs, string_attrs)

    # SVRG specific optimization not available.
    def _get_optimal_solver_params_config(self):
        """No optimal parameters known for SVRG in HMMGLM."""
        return None, None, None

    def _get_model_params(self) -> GLMHMMParams:
        glm_params = GLMParams(self.coef_, self.intercept_)
        scale = GLMScale(jnp.log(self.scale_))
        hmm_params = HMMParams(
            jnp.log(self.initial_prob_), jnp.log(self.transition_prob_)
        )
        return GLMHMMParams(glm_params, scale, hmm_params)

    def _set_model_params(self, params: GLMHMMParams):
        self.coef_ = params.glm_params.coef
        self.intercept_ = params.glm_params.intercept
        self.scale_ = jnp.exp(params.glm_scale.log_scale)
        self.initial_prob_ = jnp.exp(params.hmm_params.log_initial_prob)
        self.transition_prob_ = jnp.exp(params.hmm_params.log_transition_prob)

    def update(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """Run a single update step of the jaxopt solver."""
        is_new_session = self._get_is_new_session(X, y)

        # cast to jax and find non-nans
        X, y = cast_to_jax(tree_utils.drop_nans)(X, y)

        # grab the data
        data = X.data if isinstance(X, FeaturePytree) else X

        # wrap into GLM params, this assumes params are well-structured,
        # if initialization is done via `initialize_solver_and_state` it
        # should be fine
        params = self._validator.to_model_params(params)

        # perform a one-step update
        updated_params, updated_state = self._optimization_update(
            params, opt_state, data, y, *args, is_new_session=is_new_session, **kwargs
        )

        # store params and state
        self._set_model_params(updated_params)
        self.solver_state_ = updated_state

        # estimate the scale
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(
            X, n_samples=n_samples
        )
        return self._validator.from_model_params(updated_params), updated_state

    def __repr__(self) -> str:
        """Hierarchical repr for the GLMHMM class."""
        return format_repr(
            self, multiline=True, use_name_keys=["inverse_link_function"]
        )
