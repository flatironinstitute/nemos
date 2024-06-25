from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random
from jaxopt import OptStep
from jaxopt._src import loop
from jaxopt._src.tree_util import (
    tree_add,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_scalar_mul,
    tree_sub,
)


class SVRGState(NamedTuple):
    epoch_num: int
    key: jax.Array
    error: float


class SVRG:
    def __init__(
        self,
        fun,
        maxiter: int = 100,
        key=None,
        m: Optional[int] = None,
        lr: float = 1e-3,
        tol: float = 1e-5,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.key = key
        self.m = m  # number of iterations in the inner loop
        self.lr = lr
        self.tol = tol
        self.loss_gradient = jit(grad(self.fun, argnums=(0,)))

    def init_state(self, init_params, *args, **kwargs):
        state = SVRGState(
            epoch_num=1,
            key=self.key if self.key is not None else random.PRNGKey(0),
            error=jnp.inf,
        )
        return state

    def update(self, xs, state, *args, **kwargs):
        X, y = args
        m = self.m if self.m is not None else X.shape[0]
        N = X.shape[0]

        df_xs = self.loss_gradient(xs, X, y)[0]

        def inner_loop_body(_, carry):
            xk, key = carry
            key, subkey = random.split(key)
            i = random.randint(subkey, (), 0, N)
            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]
            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )
            xk = tree_add_scalar_mul(xk, -self.lr, gk)
            return (xk, key)

        xk, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (xs, state.key),
        )

        error = self._error(tree_sub(xk, xs), self.lr)
        next_state = SVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
            error=error,
        )
        return OptStep(params=xk, state=next_state)

    def run(self, init_params, *args, **kwargs):
        def body_fun(carry):
            xs, state = carry
            return self.update(xs, state, *args, **kwargs)

        def cond_fun(carry):
            _, state = carry
            return (state.epoch_num <= self.maxiter) & (state.error >= self.tol)

        init_state = self.init_state(init_params)
        final_xs, final_state = loop.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=OptStep(params=init_params, state=init_state),
            maxiter=self.maxiter,
            jit=jit,
        )
        return OptStep(params=final_xs, state=final_state)

    @staticmethod
    def _error(diff_x, stepsize):
        diff_norm = tree_l2_norm(diff_x)
        return diff_norm / stepsize


class ProxSVRG(SVRG):
    def __init__(
        self,
        fun,
        prox,
        maxiter: int = 100,
        key=None,
        m: Optional[int] = None,
        lr: float = 1e-3,
        tol: float = 1e-5,
    ):
        super().__init__(fun, maxiter, key, m, lr, tol)
        self.proximal_operator = prox

    def update(self, xs, state, *args, **kwargs):
        """
        Performs the inner loop of Prox-SVRG
        """
        prox_lambda, X, y = args
        m = self.m if self.m is not None else X.shape[0]
        N = X.shape[0]

        df_xs = self.loss_gradient(xs, X, y)[0]

        def inner_loop_body(_, carry):
            xk, x_sum, key = carry
            key, subkey = random.split(key)
            i = random.randint(subkey, (), 0, N)
            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]
            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )
            xk = tree_add_scalar_mul(xk, -self.lr, gk)
            xk = self.proximal_operator(xk, self.lr * prox_lambda)
            x_sum = tree_add(x_sum, xk)
            return (xk, x_sum, key)

        x_sum_init = jax.tree_util.tree_map(jnp.zeros_like, xs)

        _, x_sum, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (xs, x_sum_init, state.key),
        )

        xs_prev = xs
        xs = tree_scalar_mul(1 / m, x_sum)
        error = self._error(tree_sub(xs, xs_prev), self.lr)
        next_state = SVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
            error=error,
        )
        return OptStep(params=xs, state=next_state)
