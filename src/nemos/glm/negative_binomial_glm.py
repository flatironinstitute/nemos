from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy._typing import ArrayLike

from ..glm import GLM
from ..observation_models import NegativeBinomialObservations
from ..regularizer import Regularizer
from ..solvers import AbstractSolver, SolverState
from ..typing import DESIGN_INPUT_TYPE
from .params import GLMParams, NBGLMUserParams


class NBState(eqx.Module):
    data_log_likelihood: float | jnp.ndarray
    previous_data_log_likelihood: float | jnp.ndarray
    state_params: SolverState | None
    state_scale: SolverState | None
    iterations: int


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


@partial(jax.jit, static_argnames=["solver_params"])
def _param_update_step(
    init_params: GLMParams,
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    log_scale: float,
    solver_params,
):
    params, state, aux = solver_params(init_params, X, y, jnp.exp(log_scale))
    return params, state, aux


@partial(jax.jit, static_argnames=["solver_scale"])
def _scale_update_step(log_scale, X, y, init_params, solver_scale):
    log_scale, state, aux = _param_update_step(init_params, X, y, log_scale)
    return log_scale, state, aux


def _joint_fit(
    init_params: Tuple[GLMParams, float],
    X: DESIGN_INPUT_TYPE,
    y: jnp.ndarray,
    solver_params: AbstractSolver,
    solver_scale: AbstractSolver,
    tol: float,
    maxiter: int,
):
    init_glm_params, init_scale = init_params[0], init_params[1]

    _step_params = eqx.Partial(
        _param_update_step,
        X=X,
        y=y,
        solver_params=solver_params,
    )

    _step_scale = eqx.Partial(
        _scale_update_step,
        X=X,
        y=y,
        solver_scale=solver_scale,
    )

    init_state = NBState(
        data_log_likelihood=-jnp.array(jnp.inf),
        previous_data_log_likelihood=-jnp.array(jnp.inf),
        state_params=None,
        state_scale=None,
        iterations=0,
    )

    def stopping_condition_while(carry):
        _, new_state = carry
        return ~check_log_likelihood_increment(new_state, tol)

    def body_fn(carry):
        params, log_scale, state = carry
        new_params, state_params, _ = _step_params(params, log_scale)
        new_log_scale, state_scale, _ = _step_scale(params, log_scale)
        func_val = solver_scale.get_optim_info(state_scale).function_val
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

    init_carry = init_glm_params, jnp.log(init_scale), init_state
    params, log_scale, state = eqx.internal.while_loop(
        stopping_condition_while,
        body_fn,
        init_carry,
        max_steps=maxiter,
        kind="lax",
    )
    return (params, jnp.exp(log_scale)), state, None


class NBGLM(GLM):
    def __init__(
        self,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Any = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
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

        self._initialize_optimization_and_state(
            init_params, data, y, self.observation_model.scale
        )
        params, state, aux = self._optimization_run(
            init_params, data, y, self.observation_model.scale
        )

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
        init_params: Tuple[GLMParams, float],
        X: dict[str, jnp.ndarray] | jnp.ndarray,
        y: jnp.ndarray,
        *args,
    ) -> SolverState:
        self._instantiate_solver(
            self.compute_loss,
            init_params[0],
        )
