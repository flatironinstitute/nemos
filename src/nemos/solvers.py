from functools import partial
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
    iter_num: int
    key: KeyArrayLike
    error: float
    stepsize: float
    N: Optional[int] = None
    xs: Optional[tuple] = None
    df_xs: Optional[tuple] = None


class SVRG:
    def __init__(
        self,
        fun: Callable,
        maxiter: int = 1000,
        key: Optional[KeyArrayLike] = None,
        # N: Optional[int] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-5,
        batch_size: int = 1,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.key = key
        # self.N = N  # number of overall data points
        self.stepsize = stepsize
        self.tol = tol
        self.loss_gradient = jit(grad(self.fun, argnums=(0,)))
        self.batch_size = batch_size

    def init_state(self, init_params, *args, **kwargs):
        if len(args) > 0:
            X, y = args
            assert isinstance(X, ArrayLike)
            assert isinstance(y, ArrayLike)
            assert X.shape[0] == y.shape[1]

            N = X.shape[0]
            df_xs = self.loss_gradient(init_params, X, y)
        else:
            N = None
            df_xs = None

        state = SVRGState(
            iter_num=1,
            key=self.key if self.key is not None else random.PRNGKey(0),
            error=jnp.inf,
            stepsize=self.stepsize,
            N=N,
            xs=init_params,
            df_xs=df_xs,
        )
        return state

    def get_m(self, N: int):
        """
        Number of iterations needed to cover N data points with samples of self.batch_size
        """
        return (N + self.batch_size - 1) // self.batch_size

    def _sgd_update(self, xs, state, *args, **kwargs):
        X, y = args
        N = X.shape[0] if state.N is None else state.N
        m = self.get_m(N)

        # TODO if X is prohibitively large, do this in batches and average over them
        # but then should the whole matrix even be passed here, or should the batching be done somewhere else?
        df_xs = self.loss_gradient(xs, X, y)[0]

        def inner_loop_body(_, carry):
            xk, key, X, y = carry

            key, subkey = random.split(key)
            i = random.randint(subkey, (self.batch_size,), 0, N)

            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]

            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )

            xk = tree_add_scalar_mul(xk, -state.stepsize, gk)

            return (xk, key, X, y)

        xk, key, _, _ = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                xs,
                # conversion needed for compatibility with glm.update
                state.key.astype(jnp.uint32),
                X,
                y,
            ),
        )

        error = self._error(xk, xs, state.stepsize)
        next_state = SVRGState(
            iter_num=state.iter_num + 1,
            key=key,
            error=error,
            stepsize=state.stepsize,
        )
        return OptStep(params=xk, state=next_state)

    @partial(jit, static_argnums=(0,))
    def update(self, xk, state, *args, **kwargs):
        # batch data
        x, y = args

        # if the state hasn't been initialized with the full gradient,
        # the best we can do is initialize df_xs with the gradient of the current mini-batch
        # not the full gradient, but less noisy than any xk
        if state.xs is None or state.df_xs is None:
            state = SVRGState(
                stepsize=state.stepsize,
                iter_num=state.iter_num,
                key=state.key,
                error=state.error,
                xs=xk,
                df_xs=self.loss_gradient(xk, x, y)[0],
            )

        xs, df_xs = state.xs, state.df_xs

        # don't carry x and y, they should be cached on the first call
        def loop_body(i, xk):
            # no random sampling, just iterate through the data points
            dfik_xk = self.loss_gradient(xk, x[i], y[i])[0]
            dfik_xs = self.loss_gradient(xs, x[i], y[i])[0]
            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )
            xk = tree_add_scalar_mul(xk, -state.stepsize, gk)
            return xk

        xk = lax.fori_loop(0, x.shape[0], loop_body, xk)

        # xs is updated outside for this implementation
        # because we only want to update it after a whole sweep
        # through the data, not after every mini-batch
        next_state = SVRGState(
            stepsize=state.stepsize,
            iter_num=state.iter_num + 1,
            key=state.key,
            error=state.error,
            xs=state.xs,
            df_xs=state.df_xs,
        )

        return OptStep(params=xk, state=next_state)

    # def update(self, xk, state, *args, **kwargs):
    #    return self._sgd_update(xk, state, *args, **kwargs)

    def run(self, init_params, *args, **kwargs):
        def body_fun(step):
            xs, state = step
            return self._sgd_update(xs, state, *args, **kwargs)

        def cond_fun(step):
            _, state = step
            return (state.iter_num <= self.maxiter) & (state.error >= self.tol)

        init_state = self.init_state(init_params, *args)

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
        # N: Optional[int] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-5,
        batch_size: int = 1,
    ):
        super().__init__(
            fun,
            maxiter,
            key,
            # N,
            stepsize,
            tol,
            batch_size,
        )
        self.proximal_operator = prox

    def update(self, xk, state, *args, **kwargs):
        raise NotImplementedError

    def _sgd_update(self, xs, state, *args, **kwargs):
        """
        Performs the inner loop of Prox-SVRG
        """
        prox_lambda, X, y = args
        N = X.shape[0] if state.N is None else state.N
        m = self.get_m(N)

        df_xs = self.loss_gradient(xs, X, y)[0]

        def inner_loop_body(_, carry):
            xk, x_sum, key, X, y = carry
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

            return (xk, x_sum, key, X, y)

        _, x_sum, key, _, _ = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                xs,
                tree_zeros_like(xs),
                # conversion needed for compatibility with glm.update
                state.key.astype(jnp.uint32),
                X,
                y,
            ),
        )

        xs_prev = xs
        xs = tree_scalar_mul(1 / m, x_sum)
        error = self._error(xs, xs_prev, state.stepsize)
        next_state = SVRGState(
            iter_num=state.iter_num + 1,
            key=key,
            error=error,
            stepsize=state.stepsize,
        )
        return OptStep(params=xs, state=next_state)
