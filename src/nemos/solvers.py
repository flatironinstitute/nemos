from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random
from jax._src.typing import ArrayLike
from jaxopt import OptStep
from jaxopt._src import loop
from jaxopt._src.proximal_gradient import fista_line_search
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

# copying from .glm to avoid circular import
# TODO might want to move these into a .typing module?
ModelParams = Tuple[jnp.ndarray, jnp.ndarray]


class SVRGState(NamedTuple):
    iter_num: int
    key: KeyArrayLike
    error: float
    stepsize: float
    # N: Optional[int] = None
    loss_log: ArrayLike
    xs: Optional[tuple] = None
    df_xs: Optional[tuple] = None
    x_av: Optional[tuple] = None


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
        batch_size: Optional[int] = None,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.key = key
        # self.N = N  # number of overall data points
        self.stepsize = stepsize
        self.tol = tol
        self.loss_gradient = jit(grad(self.fun, argnums=(0,)))

        self.batch_size = batch_size

        self.proximal_operator = prox

    def init_state(self, init_params, *args, **kwargs):
        df_xs = None
        if kwargs.get("init_full_gradient", False):
            prox_lambda, X, y = args

            assert isinstance(X, ArrayLike)
            assert isinstance(y, ArrayLike)
            assert X.shape[0] == y.shape[0]

            # N = X.shape[0]
            df_xs = self.loss_gradient(init_params, X, y)[0]

        state = SVRGState(
            iter_num=0,
            key=self.key if self.key is not None else random.PRNGKey(0),
            error=jnp.inf,
            stepsize=self.stepsize,
            # N=N,
            loss_log=jnp.empty((self.maxiter,)),
            xs=init_params,
            df_xs=df_xs,
            x_av=init_params,
        )
        return state

    @partial(jit, static_argnums=(0,))
    def _xk_update(self, xk, xs, df_xs, stepsize, prox_lambda, x, y):
        dfik_xk = self.loss_gradient(xk, x, y)[0]
        dfik_xs = self.loss_gradient(xs, x, y)[0]

        gk = jax.tree_util.tree_map(lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs)

        next_xk = tree_add_scalar_mul(xk, -stepsize, gk)

        # next_xk = self.proximal_operator(next_xk, state.stepsize * prox_lambda)
        next_xk = self.proximal_operator(next_xk, prox_lambda, scaling=stepsize)

        return next_xk

    @partial(jit, static_argnums=(0,))
    def update(self, x0: ModelParams, state: SVRGState, *args, **kwargs):
        # return self._update_per_point(x0, state, *args, **kwargs)
        return self._update_per_batch(x0, state, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def _update_per_batch(self, x0: ModelParams, state: SVRGState, *args, **kwargs):
        # NOTE this doesn't update state.x_av, that has to be done outside

        prox_lambda, X, y = args
        xs, df_xs = state.xs, state.df_xs

        xk = self._xk_update(x0, xs, df_xs, state.stepsize, prox_lambda, X, y)

        # update the state
        # storing the average over the inner loop to potentially use it in the run loop
        state = state._replace(
            iter_num=state.iter_num + 1,
        )

        # returning the average might help stabilize things and allow for a larger step size
        return OptStep(params=xk, state=state)

    @partial(jit, static_argnums=(0,))
    def _update_per_point(self, x0: ModelParams, state: SVRGState, *args, **kwargs):
        """
        Performs the inner loop of Prox-SVRG

        Parameters
        ----------
        x0 : ModelParams
            Parameters at the end of the previous update, used as the starting point for the current update.
            When updating on the whole data, `update` is called from within `run` which is used by `GLM.fit`, then this is the last anchor point.
            When updating on a mini-batch, `update` is called by `GLM.update`, then this has to be the parameters after updating on the last mini-batch, and.
        state : SVRGState
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        *args
            Assumed to be of length 3 and is packed out as:
                prox_lambda, X, y = args
            where prox_lambda is the strength of the regularization (which can be None), and X and y are the data.

        Returns
        -------
        OptStep
            xs : ModelParams
                Average of the parameters over the last inner loop.
            state : SVRGState
                Updated state.
        """
        prox_lambda, X, y = args
        m = X.shape[0]  # number of iterations
        N = X.shape[0]  # number of data points

        xs, df_xs = state.xs, state.df_xs

        # assert jax.tree_util.tree_map(
        #    lambda params_a, params_b: jnp.all(jnp.isclose(params_a, params_b)),
        #    xs,
        #    state.xs,
        # )

        def inner_loop_body(i, carry):
            xk, x_sum, key = carry
            key, subkey = random.split(key)
            ind = random.randint(subkey, (), 0, N)
            # ind = i

            xk = self._xk_update(
                xk, xs, df_xs, state.stepsize, prox_lambda, X[ind, :], y[ind]
            )

            x_sum = tree_add(x_sum, xk)

            return (xk, x_sum, key)

        xk, x_sum, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                x0,
                tree_zeros_like(xs),
                # conversion needed for compatibility with glm.update
                state.key.astype(jnp.uint32),
            ),
        )

        # update the state
        # storing the average over the inner loop to potentially use it in the run loop
        state = state._replace(
            iter_num=state.iter_num + 1,
            key=key,
            x_av=tree_scalar_mul(1 / m, x_sum),
        )

        # returning the average might help stabilize things and allow for a larger step size
        return OptStep(params=xk, state=state)
        # return OptStep(params=state.x_av, state=state)

    @property
    def _update_used_in_run(self):
        # return self._update_per_point
        return self._update_per_random_batch

    @partial(jit, static_argnums=(0,))
    def run(self, init_params: ModelParams, *args, **kwargs):
        prox_lambda, X, y = args

        init_state = self.init_state(
            init_params,
            prox_lambda,
            X,
            y,
            init_full_gradient=True,
        )
        assert init_state.xs is not None
        assert init_state.df_xs is not None

        # evaluate the loss for the initial parameters, aka iter_num=0
        init_state = init_state._replace(
            loss_log=init_state.loss_log.at[0].set(self.fun(init_params, X, y)),
        )

        # this method assumes that args hold the full data
        def body_fun(step):
            xs_prev, state = step

            # evaluate and store the full gradient with the params from the last inner loop
            state = state._replace(
                df_xs=self.loss_gradient(xs_prev, X, y)[0],
            )

            # run an update over the whole data
            xk, state = self._update_used_in_run(
                xs_prev, state, prox_lambda, X, y, **kwargs
            )

            # update xs with the final xk or an average over the inner loop's iterations
            xs = xk
            # xs = state.x_av

            state = state._replace(
                xs=xs,
                error=self._error(xs, xs_prev, state.stepsize),
                loss_log=state.loss_log.at[state.iter_num].set(self.fun(xs, X, y)),
            )

            return OptStep(params=xs, state=state)

        def cond_fun(step):
            _, state = step
            return (state.iter_num <= self.maxiter) & (state.error >= self.tol)

        final_xs, final_state = loop.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=OptStep(params=init_params, state=init_state),
            maxiter=self.maxiter,
            jit=True,
        )
        return OptStep(params=final_xs, state=final_state)

    @partial(jit, static_argnums=(0,))
    def _update_per_random_batch(
        self, x0: ModelParams, state: SVRGState, *args, **kwargs
    ):
        prox_lambda, X, y = args

        N, d = X.shape[0]  # number of data points x number of dimensions
        m = (N + self.batch_size - 1) // self.batch_size  # number of iterations

        xs, df_xs = state.xs, state.df_xs

        def inner_loop_body(i, carry):
            xk, x_sum, key = carry
            key, subkey = random.split(key)
            ind = random.randint(subkey, (self.batch_size,), 0, N)

            xk = self._xk_update(
                xk, xs, df_xs, state.stepsize, prox_lambda, X[ind, :], y[ind]
            )

            x_sum = tree_add(x_sum, xk)

            return (xk, x_sum, key)

        xk, x_sum, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                x0,
                tree_zeros_like(xs),
                # conversion needed for compatibility with glm.update
                state.key.astype(jnp.uint32),
            ),
        )

        # update the state
        # storing the average over the inner loop to potentially use it in the run loop
        state = state._replace(
            iter_num=state.iter_num + 1,
            key=key,
            x_av=tree_scalar_mul(1 / m, x_sum),
        )

        # returning the average might help stabilize things and allow for a larger step size
        return OptStep(params=xk, state=state)
        # return OptStep(params=state.x_av, state=state)

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
        batch_size: Optional[int] = None,
    ):
        super().__init__(
            fun,
            prox_none,
            maxiter,
            key,
            # N,
            stepsize,
            tol,
            batch_size,
        )

    def init_state(self, init_params: ModelParams, *args, **kwargs):
        # substitute None for prox_lambda
        if len(args) == 2:
            args = (None, *args)
        return super().init_state(init_params, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update(self, x0: ModelParams, state: SVRGState, *args, **kwargs):
        if len(args) == 2:
            args = (None, *args)
        return super().update(x0, state, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def run(self, init_params: ModelParams, *args, **kwargs):
        args = (None, *args)
        return super().run(init_params, *args, **kwargs)
