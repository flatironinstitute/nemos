from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, random


# Soft-thresholding function
def prox_lasso(x, lambda_):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lambda_, 0)


# Proximal operator for ridge regularization
def prox_ridge(x, lambda_):
    return x / (1 + lambda_)


# Proximal operator with no regularization
def prox_identity(x, lambda_):
    return x


class ProxSVRGState(NamedTuple):
    # params: dict
    epoch_num: int
    key: jax._src.prng.PRNGKeyArray


class ProxSVRG:
    def __init__(
        self,
        fun,
        prox=None,
        max_iter: int = 100,
        key=None,
        m: Optional[int] = None,
        lr: float = 1e-3,
        prox_lambda: float = 0.0,
    ):
        self.fun = fun
        self.max_iter = max_iter
        self.proximal_operator = prox if prox is not None else prox_identity
        self.key = key
        self.m = m  # number of iterations in the inner loop
        self.lr = lr

        # TODO I guess the proximal operator already should have the value of lamda baked in
        # how is this done in ProxGradient / nemos / jaxopt's lasso prox?
        self.prox_lambda = prox_lambda

        self.loss_gradient = jit(grad(self.fun, argnums=(0,)))

        self.loss_history = []

    def init_state(self, init_params):
        state = ProxSVRGState(
            # params=init_params,
            epoch_num=1,
            key=self.key if self.key is not None else random.PRNGKey(0),
        )
        return state

    def update(self, xs, state, X, y):
        """
        Performs the inner loop of Prox-SVRG
        """
        key = state.key

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

            # Update xk
            xk = jax.tree_util.tree_map(
                lambda xk, gk: self.proximal_operator(
                    xk - self.lr * gk, self.lr * self.prox_lambda
                ),
                xk,
                gk,
            )

            # Accumulate xk for averaging
            x_sum = jax.tree_util.tree_map(lambda sum, x: sum + x, x_sum, xk)

        # Update xs as the average of inner loop iterations
        xs = jax.tree_util.tree_map(lambda sum: sum / m, x_sum)

        next_state = ProxSVRGState(
            epoch_num=state.epoch_num + 1,
            key=key,
        )

        return xs, next_state

    def run(
        self,
        init_params,
        data: tuple,
        hyperparams_prox=None,
    ):
        if hyperparams_prox is not None:
            raise NotImplementedError

        X, y = data

        state = self.init_state(init_params)
        xs = init_params

        for s in range(self.max_iter):
            self.loss_history.append(self.fun(xs, X, y).item())
            xs, state = self.update(xs, state, X, y)

        return xs, state
