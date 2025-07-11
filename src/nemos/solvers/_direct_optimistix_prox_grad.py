"""Implementing ProximalGradient with FISTA as an Optimistix IterativeSolver."""

from ._optimistix_solvers import OptimistixAdapter

from functools import partial
import inspect

import operator
import optimistix as optx
import equinox as eqx

from optimistix._custom_types import Y, Aux

from typing import Any, Callable

from jaxtyping import Array, Int, Float, PyTree, Bool

import jax
import jax.numpy as jnp


# isn't this already defined somewhere?
def tree_sub(x, y):
    return jax.tree.map(operator.sub, x, y)


# from jaxopt
# alternatively, could use the definition from optax
def tree_add_scalar_mul(tree_x: PyTree, scalar, tree_y):
    return jax.tree_util.tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


class ProxGradState(eqx.Module):
    iter_num: Int[Array, ""]
    stepsize: Float[Array, ""]
    velocity: PyTree
    t: Float[Array, ""]

    terminate: Bool[Array, ""]
    # result: optx._solution.RESULTS


class ProximalGradient(optx.AbstractMinimiser[Y, Aux, ProxGradState]):
    prox: Callable
    regularizer_strength: float

    atol: float
    rtol: float
    norm: Callable

    maxls: int = 15
    decrease_factor: float = 0.5

    def init(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> ProxGradState:
        del fn, args, options, f_struct, aux_struct, tags
        return ProxGradState(
            iter_num=jnp.asarray(0),
            velocity=y,
            t=jnp.asarray(1.0),
            stepsize=jnp.asarray(1.0),
            terminate=jnp.asarray(False),
        )

    def step(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
        tags: frozenset[object],
    ) -> tuple[Y, ProxGradState, Aux]:
        del tags

        # TODO might want to store value_and_grad_fun instead of doing this
        # if we need the gradient anyway?
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_val, lin_fn, _ = jax.linearize(
            lambda _y: fn(_y, args), state.velocity, has_aux=True
        )
        grad = optx._misc.lin_to_grad(
            lin_fn, state.velocity, autodiff_mode=autodiff_mode
        )

        fun_without_aux = lambda x, args: fn(x, args)[0]
        next_y, new_stepsize = self.fista_line_search(
            fun_without_aux,
            state.velocity,
            f_val,
            grad,
            state.stepsize,
            args,
        )

        new_stepsize = jnp.where(
            new_stepsize <= 1e-6,
            jnp.array(1.0),
            new_stepsize / self.decrease_factor,
        )

        next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * state.t**2))
        diff_y = tree_sub(next_y, y)
        next_vel = tree_add_scalar_mul(next_y, (state.t - 1) / next_t, diff_y)

        new_fun_val, new_aux = fn(next_y, args)

        # NOTE do we want to use Cauchy for consistency with other solvers
        # or the other to be consistent with JAXopt?
        terminate = optx._misc.cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            diff_y,
            f_val,
            new_fun_val - f_val,
        )
        # terminate = (optx.two_norm(tree_sub(next_y, y)) / new_stepsize) < self.atol

        next_state = ProxGradState(
            iter_num=state.iter_num + 1,
            velocity=next_vel,
            t=next_t,
            stepsize=jnp.asarray(new_stepsize),
            terminate=terminate,
            # result=optx._solution.RESULTS.successful,
        )

        return next_y, next_state, new_aux

    # adapted from JAXopt
    def fista_line_search(
        self,
        fun: Callable,
        x: Y,
        x_fun_val: Float[Array, ""],
        grad: Y,
        stepsize: Float[Array, ""],
        args: PyTree[Any],
    ) -> tuple[Y, Float[Array, ""]]:
        # epsilon of current dtype for robust checking of
        # sufficient decrease condition
        eps = jnp.finfo(x_fun_val.dtype).eps

        def cond_fun(carry):
            next_x, stepsize = carry

            new_fun_val = fun(next_x, args)

            diff_x = tree_sub(next_x, x)
            sqdist = optx._misc.sum_squares(diff_x)

            # NOTE verbatim from JAXopt
            # The expression below checks the sufficient decrease condition
            # f(next_x) < f(x) + dot(grad_f(x), diff_x) + (0.5/stepsize) ||diff_x||^2
            # where the terms have been reordered for numerical stability.
            fun_decrease = stepsize * (new_fun_val - x_fun_val)
            expected_decrease = (
                stepsize * optx._misc.tree_dot(diff_x, grad) + 0.5 * sqdist
            )

            return fun_decrease > expected_decrease + eps

        def body_fun(carry):
            stepsize = carry[1]
            new_stepsize = stepsize * self.decrease_factor
            next_x = tree_add_scalar_mul(x, -new_stepsize, grad)
            next_x = self.prox(next_x, self.regularizer_strength, new_stepsize)
            return next_x, new_stepsize

        init_x = tree_add_scalar_mul(x, -stepsize, grad)
        init_x = self.prox(init_x, self.regularizer_strength, stepsize)
        init_val = (init_x, stepsize)

        # TODO make kind dependent on the adjoint used?
        # "lax" for implicit, "checkpointed" for RecursiveCheckpointAdjoint
        return eqx.internal.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=init_val,
            max_steps=self.maxls,
            kind="lax",
        )

    def terminate(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx._solution.RESULTS]:
        del fn, y, args, options, tags

        # return state.terminate, state.result
        return state.terminate, optx._solution.RESULTS.successful

    def postprocess(
        self,
        fn: Callable,
        y: Y,
        aux: Aux,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
        tags: frozenset[object],
        result: optx._solution.RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        del fn, args, options, state, tags, result
        return y, aux, {}


class DirectProximalGradient(OptimistixAdapter):
    _solver_cls = ProximalGradient
    _proximal = True

    @property
    def maxiter(self):
        return self.config.max_steps
