from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..glm import GLM
from ..observation_models import NegativeBinomialObservations
from ..regularizer import Regularizer, UnRegularized
from ..solvers import SolverState
from ..typing import DESIGN_INPUT_TYPE
from .params import GLMParams, NBGLMUserParams


class NBState(eqx.Module):
    data_log_likelihood: float | jnp.ndarray
    previous_data_log_likelihood: float | jnp.ndarray
    state_params: SolverState | None
    state_scale: SolverState | None
    iterations: int


def _extract_fun_value(state: SolverState) -> float | None:
    """Extract the function value from the solver state."""
    if hasattr(state, "value"):
        fval = state.value
    elif hasattr(state, "f"):
        fval = state.f.item() if hasattr(state.f, "item") else state.f
    elif hasattr(state, "f_info"):
        fval = state.f_info.f
        fval = fval.item() if hasattr(fval, "item") else fval
    else:
        fval = None
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
        state_params=new_state_params,
        state_scale=new_state_scale,
        iterations=new_iterations,
    )
    return (new_params, new_scale), new_state, (new_aux_params, new_aux_scale)


def _joint_run(
    init_params: Tuple[GLMParams, float],
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    solver_run_params: Callable,
    solver_run_scale: Callable,
    init_state_fn: Callable,
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

    init_state = init_state_fn(init_params, X, y)

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
            raise ValueError("Solver state must store the value function.")
        new_state = NBState(
            data_log_likelihood=func_val,
            previous_data_log_likelihood=state.data_log_likelihood,
            iterations=state.iterations + 1,
            state_scale=state_scale,
            state_params=state_params,
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


class NBGLM(GLM):
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
        self._initialize_optimization_and_state(
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
        # self.optim_info_ = (
        # self._solver.get_optim_info(state.state_params),
        # self._solver.get_optim_info(state.state_scale))

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

    def _initialize_optimization_and_state(
        self,
        init_params: Tuple[GLMParams, jnp.ndarray],
        X: dict[str, jnp.ndarray] | jnp.ndarray,
        y: jnp.ndarray,
        *args,
    ) -> SolverState:
        init_state_params, update_params, run_params = self._instantiate_solver(
            self._compute_loss,
            init_params[0],
        )

        def _scale_loss(log_scale, X, y, params):
            return -self.observation_model.log_likelihood(
                y, self._predict(params, X), scale=jnp.exp(log_scale)
            )

        init_state_scale, update_scale, run_scale = self._instantiate_solver(
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
            return _joint_update(params, state, X, y, run_params, run_scale)

        def optimization_init_state(params, X, y):
            state_glm_params = init_state_params(params[0], X, y, jnp.exp(params[1]))
            state_scale = init_state_scale(params[1], X, y, params[0])
            return NBState(
                data_log_likelihood=-jnp.array(jnp.inf),
                previous_data_log_likelihood=-jnp.array(jnp.inf),
                state_params=state_glm_params,
                state_scale=state_scale,
                iterations=0,
            )

        def optimization_run(params, X, y):
            return _joint_run(
                params,
                X,
                y,
                run_params,
                run_scale,
                tol=self.tol,
                maxiter=self.maxiter,
                init_state_fn=optimization_init_state,
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
