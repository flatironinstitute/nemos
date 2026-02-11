from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import nemos as nmo
from nemos import tree_utils
from nemos.solvers import AbstractSolver, OptimizationInfo
from nemos.solvers._aux_helpers import wrap_aux

pytestmark = pytest.mark.solver_related


class OptaxAdamState(NamedTuple):
    iter_num: jnp.ndarray
    opt_state: optax.OptState
    value: jnp.ndarray
    error: jnp.ndarray


class OptaxAdam(AbstractSolver[OptaxAdamState]):
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params=None,
        learning_rate: float = 1e-3,
        tol: float = 1e-6,
        maxiter: int = 200,
        **solver_init_kwargs,
    ):
        self.tol = tol
        self.maxiter = maxiter
        self._optim = optax.adam(learning_rate)

        loss = regularizer.penalized_loss(
            unregularized_loss, init_params, regularizer_strength
        )

        if has_aux:
            self._loss_with_aux = loss
        else:
            self._loss_with_aux = wrap_aux(loss)

        self._value_and_grad = jax.value_and_grad(self._loss_with_aux, has_aux=True)

    def init_state(self, init_params, *args):
        (value, _), _ = self._value_and_grad(init_params, *args)
        opt_state = self._optim.init(init_params)
        return OptaxAdamState(
            iter_num=jnp.array(0),
            opt_state=opt_state,
            value=value,
            error=jnp.array(jnp.inf),
        )

    def _compute_error(self, new_params, params):
        return tree_utils.tree_l2_norm(tree_utils.tree_sub(new_params, params))

    def _step(self, params, opt_state, *args):
        (value, aux), grads = self._value_and_grad(params, *args)
        updates, new_opt_state = self._optim.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        error = self._compute_error(new_params, params)
        return value, aux, new_params, new_opt_state, error

    def update(self, params, state, *args):
        value, aux, new_params, new_opt_state, error = self._step(
            params, state.opt_state, *args
        )
        new_state = OptaxAdamState(
            iter_num=state.iter_num + 1,
            opt_state=new_opt_state,
            value=value,
            error=error,
        )
        return new_params, new_state, aux

    def run(self, init_params, *args):
        init_state = self.init_state(init_params, *args)

        def cond(loop_state):
            _, _, error, iter_num = loop_state
            return jnp.logical_and(iter_num < self.maxiter, error > self.tol)

        def body(loop_state):
            params, opt_state, _, iter_num = loop_state
            _, _, new_params, new_opt_state, error = self._step(
                params, opt_state, *args
            )
            return new_params, new_opt_state, error, iter_num + 1

        params, opt_state, error, iter_num = jax.lax.while_loop(
            cond,
            body,
            (init_params, init_state.opt_state, init_state.error, init_state.iter_num),
        )
        final_value, aux = self._loss_with_aux(params, *args)
        state = OptaxAdamState(
            iter_num=iter_num,
            opt_state=opt_state,
            value=final_value,
            error=error,
        )
        return params, state, aux

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"learning_rate", "tol", "maxiter"}

    def get_optim_info(self, state: OptaxAdamState) -> OptimizationInfo:
        num_steps = state.iter_num.item()
        return OptimizationInfo(
            function_val=state.value.item(),
            num_steps=num_steps,
            converged=state.error.item() <= self.tol,
            reached_max_steps=(num_steps >= self.maxiter),
        )


@pytest.mark.requires_x64
def test_custom_solver_integration(poissonGLM_model_instantiation):
    X, y, model, _, _ = poissonGLM_model_instantiation

    original_registry = nmo.solvers._solver_registry._registry.copy()
    original_defaults = nmo.solvers._solver_registry._defaults.copy()
    original_allowed = nmo.regularizer.UnRegularized._allowed_solvers

    try:
        nmo.solvers.register(
            "Adam",
            OptaxAdam,
            backend="custom",
            validate=True,
            test_ridge_without_aux=True,
        )
        nmo.regularizer.UnRegularized.allow_solver("Adam")

        model.solver_name = "Adam[custom]"
        model.solver_kwargs = {"learning_rate": 5e-3, "tol": 1e-10, "maxiter": 2000}
        model.fit(X, y)

        assert isinstance(model._solver, OptaxAdam)

        ref_model = nmo.glm.GLM(
            model.observation_model,
            regularizer=model.regularizer,
            solver_name="LBFGS",
            solver_kwargs={"tol": 1e-10},
        )
        ref_model.fit(X, y)

        assert np.allclose(model.coef_, ref_model.coef_, atol=1e-10)
        assert np.allclose(model.intercept_, ref_model.intercept_, atol=1e-10)
    finally:
        nmo.solvers._solver_registry._registry.clear()
        nmo.solvers._solver_registry._defaults.clear()
        nmo.solvers._solver_registry._registry.update(original_registry)
        nmo.solvers._solver_registry._defaults.update(original_defaults)
        nmo.regularizer.UnRegularized._allowed_solvers = original_allowed
