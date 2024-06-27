from functools import partial
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random
from jax._src.typing import ArrayLike
from jaxopt import OptStep
from jaxopt._src import loop
from jaxopt.prox import prox_none
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


class ProxSVRG:
    def __init__(
        self,
        fun: Callable,
        prox: Callable,
        maxiter: int = 1000,
        key: Optional[KeyArrayLike] = None,
        # N: Optional[int] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-5,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.key = key
        # self.N = N  # number of overall data points
        self.stepsize = stepsize
        self.tol = tol
        self.loss_gradient = jit(grad(self.fun, argnums=(0,)))

        self.proximal_operator = prox

    def init_state(self, init_params, *args, **kwargs):
        if len(args) < 2:
            N = None
            df_xs = None
        else:
            prox_lambda, X, y = args

            assert isinstance(X, ArrayLike)
            assert isinstance(y, ArrayLike)
            assert X.shape[0] == y.shape[0]

            N = X.shape[0]
            df_xs = self.loss_gradient(init_params, X, y)[0]

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
        prox_lambda, X, y = args
        return super().init_state(init_params, X, y, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update(self, xs, state, *args, **kwargs):
        """
        Performs the inner loop of Prox-SVRG
        """
        prox_lambda, X, y = args
        m = X.shape[0]  # number of iterations

        # if the state hasn't been initialized with the full gradient,
        # the best we can do is initialize df_xs with the gradient of the current mini-batch
        # not the full gradient, but less noisy than any xk
        # if state.xs is None or state.df_xs is None:
        #    state = state._replace(
        #        xs=xs,
        #        df_xs=self.loss_gradient(xs, X, y)[0],
        #    )

        df_xs = state.df_xs
        # assert jax.tree_util.tree_map(
        #    lambda params_a, params_b: jnp.all(jnp.isclose(params_a, params_b)),
        #    xs,
        #    state.xs,
        # )

        def inner_loop_body(i, carry):
            xk, x_sum, key = carry
            # key, subkey = random.split(key)
            # i = random.randint(subkey, (), 0, N)

            dfik_xk = self.loss_gradient(xk, X[i, :], y[i])[0]
            dfik_xs = self.loss_gradient(xs, X[i, :], y[i])[0]

            gk = jax.tree_util.tree_map(
                lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs
            )

            xk = tree_add_scalar_mul(xk, -state.stepsize, gk)
            # xk = self.proximal_operator(xk, state.stepsize * prox_lambda)
            xk = self.proximal_operator(xk, prox_lambda, scaling=state.stepsize)

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

        # final value is not xk at the last iteration
        # but an average over the xk values through the loop
        xs = tree_scalar_mul(1 / m, x_sum)
        next_state = state._replace(
            iter_num=state.iter_num + 1,
        )
        return OptStep(params=xs, state=next_state)

    @partial(jit, static_argnums=(0,))
    def run(self, init_params, *args, **kwargs):
        prox_lambda, X, y = args

        # this method assumes that args hold the full data
        def body_fun(step):
            xs_prev, state = step
            # evaluate and store the full gradient with the params from the last inner loop
            state = state._replace(
                df_xs=self.loss_gradient(xs_prev, X, y)[0],
            )

            # update xs with the final xk after running through the whole data
            xs, state = self.update(xs_prev, state, prox_lambda, X, y, **kwargs)

            state = state._replace(
                xs=xs,
                error=self._error(xs, xs_prev, state.stepsize),
            )

            return OptStep(params=xs, state=state)

        def cond_fun(step):
            _, state = step
            return (state.iter_num <= self.maxiter) & (state.error >= self.tol)

        init_state = self.init_state(init_params, prox_lambda, X, y)

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


class SVRG(ProxSVRG):
    def __init__(
        self,
        fun: Callable,
        maxiter: int = 1000,
        key: Optional[KeyArrayLike] = None,
        # N: Optional[int] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-5,
    ):
        super().__init__(
            fun,
            prox_none,
            maxiter,
            key,
            # N,
            stepsize,
            tol,
        )

    def init_state(self, init_params, *args, **kwargs):
        # substitute None for prox_lambda
        if len(args) == 2:
            args = (None, *args)
        return super().init_state(init_params, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update(self, xk, state, *args, **kwargs):
        if len(args) == 2:
            args = (None, *args)
        return super().update(xk, state, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def run(self, init_params, *args, **kwargs):
        args = (None, *args)
        return super().run(init_params, *args, **kwargs)
