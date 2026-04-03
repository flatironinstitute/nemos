"""Negative Binomial GLM with joint parameter and scale learning."""

from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..observation_models import NegativeBinomialObservations
from ..regularizer import Regularizer, UnRegularized
from ..solvers import SolverAdapterState
from ..typing import DESIGN_INPUT_TYPE
from .base_glm import BaseGLM
from .params import GLMParams, GLMUserParams, NBGLMUserParams


class SolverState(eqx.Module):
    """Store scale and glm parameters solver states."""

    glm_params: SolverAdapterState
    scale: SolverAdapterState


class NBState(eqx.Module):
    """Negative Binomial optimization state."""

    data_log_likelihood: float | jnp.ndarray
    previous_data_log_likelihood: float | jnp.ndarray
    solver_state: SolverState
    iterations: int


def _extract_fun_value(state: SolverAdapterState) -> float | None:
    """Extract the function value from the solver state."""
    if not isinstance(state, SolverAdapterState):
        # force recompute loss if the solver state is custom
        # this prevents unexpected behavior at the cost of an
        # extra loss computation.
        return None
    state = state.solver_state
    # jaxopt convention
    if hasattr(state, "value"):
        fval = state.value
    # optimistix convention
    elif hasattr(state, "f"):
        fval = state.f
    elif hasattr(state, "f_info"):
        fval = state.f_info.f
    else:
        # should not hit here, this error is for developers
        raise ValueError(
            "SolverAdapterState must store the value function following "
            "one of the convention above."
        )
    return fval


def check_log_likelihood_increment(state: NBState, tol: float) -> jnp.ndarray:
    """
    Check EM convergence using absolute tolerance on log-likelihood.

    Parameters
    ----------
    state :
        Current EM state containing likelihood history.
    tol :
        Absolute tolerance threshold.

    Returns
    -------
    : Array
        Boolean indicating convergence.
    """
    delta = jnp.abs(state.data_log_likelihood - state.previous_data_log_likelihood)
    return delta < tol


@partial(jax.jit, static_argnames=["solver_run_params"])
def _param_update_step(
    init_params: GLMParams,
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    log_scale: float,
    solver_run_params,
):
    params, state, aux = solver_run_params(init_params, X, y, jnp.exp(log_scale))
    return params, state, aux


@partial(jax.jit, static_argnames=["solver_run_scale"])
def _scale_update_step(log_scale, X, y, init_params, solver_run_scale):
    log_scale, state, aux = solver_run_scale(log_scale, X, y, init_params)
    return log_scale, state, aux


def _joint_update(
    init_params: Tuple[GLMParams, float],
    init_state: NBState,
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    solver_run_params: Callable,
    solver_run_scale: Callable,
):
    init_glm_params, init_log_scale = init_params[0], init_params[1]
    new_params, new_state_params, new_aux_params = _param_update_step(
        init_glm_params, X, y, init_log_scale, solver_run_params
    )
    new_scale, new_state_scale, new_aux_scale = _scale_update_step(
        init_log_scale, X, y, new_params, solver_run_scale
    )
    new_iterations = init_state.iterations + 1
    func_val = _extract_fun_value(new_state_scale)
    if func_val is None:
        raise ValueError("Solver state must store the value function.")
    new_state = NBState(
        data_log_likelihood=func_val,
        previous_data_log_likelihood=init_state.data_log_likelihood,
        solver_state=SolverState(glm_params=new_state_params, scale=new_state_scale),
        iterations=new_iterations,
    )
    return (new_params, new_scale), new_state, (new_aux_params, new_aux_scale)


def _joint_run(
    init_params: Tuple[GLMParams, float],
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    solver_run_params: Callable,
    solver_run_scale: Callable,
    initialize_state: Callable,
    scale_loss: Callable,
    tol: float,
    maxiter: int,
):
    init_glm_params, init_scale = init_params[0], init_params[1]

    _step_params = eqx.Partial(
        _param_update_step,
        X=X,
        y=y,
        solver_run_params=solver_run_params,
    )

    _step_scale = eqx.Partial(
        _scale_update_step,
        X=X,
        y=y,
        solver_run_scale=solver_run_scale,
    )

    init_state = initialize_state(init_params, X, y)

    def stopping_condition_while(carry):
        _, _, new_state = carry
        return ~check_log_likelihood_increment(new_state, tol)

    def body_fn(carry):
        params, log_scale, state = carry
        new_params, state_params, aux_scale = _step_params(
            init_params=params, log_scale=log_scale
        )
        new_log_scale, state_scale, aux_params = _step_scale(
            log_scale=log_scale, init_params=new_params
        )
        func_val = _extract_fun_value(state_scale)
        if func_val is None:
            # extra compute might be needed for custom solvers
            # not storing the func value in standard attrs.
            func_val = scale_loss(new_log_scale, X, y, new_params)
        new_state = NBState(
            data_log_likelihood=func_val,
            previous_data_log_likelihood=state.data_log_likelihood,
            iterations=state.iterations + 1,
            solver_state=SolverState(glm_params=state_params, scale=state_scale),
        )
        return new_params, new_log_scale, new_state

    init_carry = init_glm_params, init_scale, init_state
    params, log_scale, state = eqx.internal.while_loop(
        stopping_condition_while,
        body_fn,
        init_carry,
        max_steps=maxiter,
        kind="lax",
    )
    return (params, log_scale), state, None


class NBGLM(BaseGLM[GLMUserParams, GLMParams]):
    """Negative Binomial GLM model."""

    def __init__(
        self,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Any = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
        tol: float = 1e-5,
        maxiter: int = 500,
    ):
        observation_model = NegativeBinomialObservations()
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self.tol = tol
        self.maxiter = maxiter

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[NBGLMUserParams] = None,
    ):
        """Fit the NB-GLM model."""
        self._validator.validate_inputs(X, y)

        # filter for non-nans, grab data if needed
        data, y = self._preprocess_inputs(X, y)
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )
        init_log_scale = jnp.log(self.observation_model.scale)
        self._initialize_optimizer_and_state(
            (init_params, init_log_scale),
            data,
            y,
        )
        params, state, aux = self._optimization_run(
            (init_params, init_log_scale),
            data,
            y,
        )
        self._set_model_params(params)
        self.aux_ = aux
        self.solver_state_ = state

    def _compute_loss(
        self,
        params: GLMParams,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        predicted_rate = self._predict(params, X)
        return self._observation_model._negative_log_likelihood(
            y, predicted_rate, lambda x: jnp.sum(jnp.mean(x, axis=0)), *args, **kwargs
        )

    def _initialize_optimizer_and_state(
        self,
        init_params: Tuple[GLMParams, jnp.ndarray],
        X: dict[str, jnp.ndarray] | jnp.ndarray,
        y: jnp.ndarray,
        *args,
    ) -> NBState:
        solver_params = self._instantiate_solver(
            self._compute_loss,
            init_params[0],
        )

        def _scale_loss(log_scale, X, y, params):
            return -self.observation_model.log_likelihood(
                y, self._predict(params, X), scale=jnp.exp(log_scale)
            )

        solver_scale = self._instantiate_solver(
            _scale_loss,
            init_params[1],
            regularizer=UnRegularized(),
            regularizer_strength=None,
            solver_name="LBFGS",
            solver_kwargs={
                "tol": 1e-6 if init_params[1].dtype is jnp.float32 else 1e-12
            },
        )

        def optimization_update(params, state, X, y):
            return _joint_update(
                params, state, X, y, solver_params.run, solver_scale.run
            )

        def optimization_init_state(params, X, y):
            state_params = solver_params.init_state(params[0], X, y, jnp.exp(params[1]))
            state_scale = solver_scale.init_state(params[1], X, y, params[0])
            return NBState(
                data_log_likelihood=-jnp.array(jnp.inf),
                previous_data_log_likelihood=-jnp.array(jnp.inf),
                solver_state=SolverState(glm_params=state_params, scale=state_scale),
                iterations=0,
            )

        self._solver = {"glm_params": solver_params, "scale": solver_scale}

        def optimization_run(params, X, y):
            return _joint_run(
                params,
                X,
                y,
                solver_params.run,
                solver_scale.run,
                scale_loss=_scale_loss,
                tol=self.tol,
                maxiter=self.maxiter,
                initialize_state=optimization_init_state,
            )

        self._optimization_run = optimization_run
        self._optimization_update = optimization_update
        self._optimization_init_state = optimization_init_state
        return self._optimization_init_state(init_params, X, y)

    def _set_model_params(self, params: Tuple[GLMParams, jnp.ndarray]):
        self.coef_, self.intercept_ = self._validator.from_model_params(params[0])
        self.scale_ = jnp.exp(params[1])

    def _get_model_params(self):
        glm_params = self._validator.to_model_params(self.coef_, self.intercept_)
        log_scale = jnp.log(self.scale_)
        return glm_params, log_scale
