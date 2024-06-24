import warnings
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jaxopt import OptStep
from jaxopt._src.tree_util import tree_l2_norm, tree_sub


class SVRGState(NamedTuple):
    epoch_num: int
    key: jax._src.prng.PRNGKeyArray
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

        # self.loss_history = []

    def init_state(self, init_params, *args, **kwargs):
        state = SVRGState(
            # params=init_params,
            epoch_num=1,
            key=self.key if self.key is not None else random.PRNGKey(0),
            error=jnp.inf,
        )
        return state

    def update(self, xs, state, *args, **kwargs):
        """
        Performs the inner loop of SVRG
        """
        key = state.key

        X, y = args
        m = self.m if self.m is not None else X.shape[0]
        N = X.shape[0]

        # compute full gradient
        df_xs = self.loss_gradient(xs, X, y)[0]

        xk = xs
        for _ in range(m):
            # sample datapoint index
            key, subkey = random.split(key)
            i = random.randint(subkey, (), 0, N).item()

            # compute stochastic gradients
            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]

            # Compute variance-reduced gradient
            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )

            # Update xk
            xk = jax.tree_util.tree_map(lambda xk, gk: xk - self.lr * gk, xk, gk)

        # difference to the previous outer loop's end
        error = self._error(tree_sub(xk, xs), self.lr)
        # update the outer loop's variable
        # could be done in the return, but this is more explicit
        xs = xk

        next_state = SVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
            error=error,
        )

        return OptStep(params=xs, state=next_state)

    def run(self, init_params, *args, **kwargs):
        state = self.init_state(init_params)
        xs = init_params

        for s in range(self.maxiter):
            # self.loss_history.append(self.fun(xs, *args, **kwargs).item())
            xs, state = self.update(xs, state, *args)

            if state.error < self.tol:
                break

        return OptStep(params=xs, state=state)

    def _error(self, diff_x, stepsize):
        diff_norm = tree_l2_norm(diff_x)
        return diff_norm / stepsize

    def _body_fun(self, inputs):
        (params, state), (args, kwargs) = inputs
        return self.update(params, state, *args, **kwargs), (args, kwargs)

    def _cond_fun(self, inputs):
        _, state = inputs[0]
        return state.error > self.tol


class ProxSVRG(SVRG):
    def __init__(
        self,
        fun,
        prox=None,
        maxiter: int = 100,
        key=None,
        m: Optional[int] = None,
        lr: float = 1e-3,
        tol: float = 1e-5,
    ):
        super().__init__(fun, maxiter, key, m, lr, tol)

        self.proximal_operator = prox

        # self.loss_history = []

    def update(self, xs, state, *args, **kwargs):
        """
        Performs the inner loop of Prox-SVRG
        """
        key = state.key

        prox_lambda, X, y = args

        m = self.m if self.m is not None else X.shape[0]
        N = X.shape[0]

        # compute full gradient
        df_xs = self.loss_gradient(xs, X, y)[0]

        xk = xs
        x_sum = jax.tree_util.tree_map(jnp.zeros_like, xs)

        for _ in range(m):
            # sample datapoint index
            key, subkey = random.split(key)
            i = random.randint(subkey, (), 0, N).item()

            # compute stochastic gradients
            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]

            # Compute variance-reduced gradient
            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )

            xk = jax.tree_util.tree_map(
                lambda xk, gk: xk - self.lr * gk,
                xk,
                gk,
            )
            xk = self.proximal_operator(xk, self.lr * prox_lambda)

            # Accumulate xk for averaging
            x_sum = jax.tree_util.tree_map(lambda sum, x: sum + x, x_sum, xk)

        # difference to the previous outer loop's end
        xs_prev = xs
        # Update xs as the average of inner loop iterations
        xs = jax.tree_util.tree_map(lambda sum: sum / m, x_sum)
        # compute the difference
        error = self._error(tree_sub(xs, xs_prev), self.lr)

        next_state = SVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
            error=error,
        )

        return OptStep(params=xs, state=next_state)
