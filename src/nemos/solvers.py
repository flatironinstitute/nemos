from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random
from jax._src.typing import ArrayLike
from jaxopt import OptStep
from jaxopt._src import loop
from jaxopt.tree_util import (
    tree_add,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_scalar_mul,
    tree_sub,
    tree_zeros_like,
)

# copying jax.random's annotation
KeyArrayLike = ArrayLike


class SVRGState(NamedTuple):
    epoch_num: int
    key: KeyArrayLike
    error: float
    stepsize: float


class SVRG:
    def __init__(
        self,
        fun: Callable,
        maxiter: int = 1000,
        key: Optional[KeyArrayLike] = None,
        m: Optional[int] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-5,
        batch_size: int = 1,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.key = key
        self.m = m  # number of iterations in the inner loop
        self.stepsize = stepsize
        self.tol = tol
        self.loss_gradient = jit(grad(self.fun, argnums=(0,)))
        self.batch_size = batch_size

    def init_state(self, init_params, *args, **kwargs):
        state = SVRGState(
            epoch_num=1,
            key=self.key if self.key is not None else random.PRNGKey(0),
            error=jnp.inf,
            stepsize=self.stepsize,
        )
        return state

    def update(self, xs, state, *args, **kwargs):
        X, y = args
        N = X.shape[0]
        if self.m is not None:
            m = self.m
        else:
            m = (N + self.batch_size - 1) // self.batch_size

        # TODO if X is prohibitively large, do this in batches and average over them
        # but then should the whole matrix even be passed here, or should the batching be done somewhere else?
        df_xs = self.loss_gradient(xs, X, y)[0]

        def inner_loop_body(_, carry):
            xk, key = carry

            key, subkey = random.split(key)
            i = random.randint(subkey, (self.batch_size,), 0, N)

            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]

            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )

            xk = tree_add_scalar_mul(xk, -state.stepsize, gk)

            return (xk, key)

        xk, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                xs,
                # conversion needed for compatibility with glm.update
                state.key.astype(jnp.uint32),
            ),
        )

        error = self._error(xk, xs, state.stepsize)
        next_state = SVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
            error=error,
            stepsize=state.stepsize,
        )
        return OptStep(params=xk, state=next_state)

    def run(self, init_params, *args, **kwargs):
        def body_fun(step):
            xs, state = step
            return self.update(xs, state, *args, **kwargs)

        def cond_fun(step):
            _, state = step
            return (state.epoch_num <= self.maxiter) & (state.error >= self.tol)

        init_state = self.init_state(init_params)
        final_xs, final_state = loop.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=OptStep(params=init_params, state=init_state),
            maxiter=self.maxiter,
            jit=True,
        )
        return OptStep(params=final_xs, state=final_state)

    # @staticmethod
    # def _error(x, x_prev, stepsize):
    #    diff_norm = tree_l2_norm(tree_sub(x, x_prev))
    #    return diff_norm / stepsize

    @staticmethod
    def _error(x, x_prev, stepsize):
        return tree_l2_norm(tree_sub(x, x_prev)) / tree_l2_norm(x_prev)


class ProxSVRG(SVRG):
    def __init__(
        self,
        fun: Callable,
        prox: Callable,
        maxiter: int = 1000,
        key: Optional[KeyArrayLike] = None,
        m: Optional[int] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-5,
        batch_size: int = 1,
    ):
        super().__init__(fun, maxiter, key, m, stepsize, tol, batch_size)
        self.proximal_operator = prox

    def update(self, xs, state, *args, **kwargs):
        """
        Performs the inner loop of Prox-SVRG
        """
        prox_lambda, X, y = args
        N = X.shape[0]
        if self.m is not None:
            m = self.m
        else:
            m = (N + self.batch_size - 1) // self.batch_size

        df_xs = self.loss_gradient(xs, X, y)[0]

        def inner_loop_body(_, carry):
            xk, x_sum, key = carry
            key, subkey = random.split(key)
            i = random.randint(subkey, (self.batch_size,), 0, N)

            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]

            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )

            xk = tree_add_scalar_mul(xk, -state.stepsize, gk)
            xk = self.proximal_operator(xk, state.stepsize * prox_lambda)

            x_sum = tree_add(x_sum, xk)

            return (xk, x_sum, key)

        _, x_sum, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                xs,
                tree_zeros_like(xs),
                # conversion needed for compatibility with glm.update
                state.key.astype(jnp.uint32),
            ),
        )

        xs_prev = xs
        xs = tree_scalar_mul(1 / m, x_sum)
        error = self._error(xs, xs_prev, state.stepsize)
        next_state = SVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
            error=error,
            stepsize=state.stepsize,
        )
        return OptStep(params=xs, state=next_state)
