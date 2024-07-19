from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.flatten_util
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

from .base_class import DESIGN_INPUT_TYPE
from .tree_utils import tree_slice

# copying jax.random's annotation
KeyArrayLike = ArrayLike


class SVRGState(NamedTuple):
    iter_num: int
    key: KeyArrayLike
    error: float
    stepsize: float
    # loss_log: jnp.ndarray
    xs: Optional[tuple] = None
    df_xs: Optional[tuple] = None
    x_av: Optional[tuple] = None


class ProxSVRG:
    """
    Prox-SVRG solver

    Borrowing from jaxopt.ProximalGradient, this solver minimizes:

      objective(params, hyperparams_prox, *args, **kwargs) =
        fun(params, *args, **kwargs) + non_smooth(params, hyperparams_prox)

    Attributes
    ----------
    fun: Callable
        smooth function of the form ``fun(x, *args, **kwargs)``.
    prox: Callable
        proximity operator associated with the function ``non_smooth``.
        It should be of the form ``prox(params, hyperparams_prox, scale=1.0)``.
        See ``jaxopt.prox`` for examples.
    maxiter : int
        Maximum number of epochs to run the optimization for.
    key : jax.random.PRNGkey
        jax PRNGKey to start with. Used for sampling random data points.
    stepsize : float
        Constant step size to use.
    tol: float
        Tolerance level for the error when comparing parameters
        at the end of consecutive epochs to check for convergence.
    batch_size: int
        Number of data points to sample per inner loop iteration.

    Example
    -------
    def loss_fn(params, X, y):
        ...

    svrg = ProxSVRG(loss_fn, prox_fun)
    params, state = svrg.run(init_params, prox_lambda, X, y)

    References
    ----------
    Prox-SVRG - https://arxiv.org/abs/1403.4699v1
    SVRG - https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html
    """

    def __init__(
        self,
        fun: Callable,
        prox: Callable,
        maxiter: int = 10_000,
        key: Optional[KeyArrayLike] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-3,
        batch_size: int = 1,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.key = key
        self.stepsize = stepsize
        self.tol = tol
        self.loss_gradient = jit(grad(self.fun))
        self.batch_size = batch_size
        self.proximal_operator = prox

    def init_state(
        self,
        init_params: Any,
        hyperparams_prox: Any,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_full_gradient: bool = False,
    ):
        """
        Initialize the solver state

        Parameters
        ----------
        init_params : Any
            pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        hyperparams_prox : float
            Parameters of the proximal operator, in our case the regularization strength.
            Not used here, but required to be consistent with the jaxopt API.
        X :
            Input data.
        y :
            Output data.
        init_full_gradient : bool, default False
            Whether to calculate the full gradient at the initial parameters,
            assuming that X, y are the full data set, and store this gradient in the initial state.
        """
        df_xs = None

        if init_full_gradient:
            df_xs = self.loss_gradient(init_params, X, y)

        state = SVRGState(
            iter_num=0,
            key=self.key if self.key is not None else random.key(0),
            error=jnp.inf,
            stepsize=self.stepsize,
            # loss_log=jnp.zeros((self.maxiter,)),
            xs=init_params,
            df_xs=df_xs,
            x_av=init_params,
        )
        return state

    def _update_loss_log(
        self,
        loss_log: jnp.ndarray,
        i: int,
        params: Any,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update an entry in the array used for storing the log of the loss throughout the optimization.

        Parameters
        ----------
        loss_log : jnp.ndarray
            1D array storing the loss values.
        i : int
            Index at which to update, most likely the current iteration number.
        params : Any
            Parameters with which to evaluate the loss.
        X, y : jnp.ndarray
            Input and output data.

        Returns
        -------
        Updated loss log.
        """
        return loss_log.at[i].set(self.fun(params, X, y))

    @partial(jit, static_argnums=(0,))
    def _xk_update(
        self,
        xk: Any,
        xs: Any,
        df_xs: Any,
        stepsize: float,
        prox_lambda: float,
        x: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> Any:
        """
        Body of the inner loop of Prox-SVRG that takes a step.

        Parameters
        ----------
        xk : pytree
            Current parameters.
        xs : pytree
            Anchor point.
        df_xs : pytree
            Full gradient at the anchor point.
        stepsize : float
            Step size.
        prox_lambda : float or None
            Regularization strength.
        x : jnp.ndarray
            Input data point or mini-batch.
        y : jnp.ndarray
            Output data point or mini-batch.
        """
        dfik_xk = self.loss_gradient(xk, x, y)
        dfik_xs = self.loss_gradient(xs, x, y)

        gk = jax.tree_util.tree_map(lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs)

        next_xk = tree_add_scalar_mul(xk, -stepsize, gk)

        # next_xk = self.proximal_operator(next_xk, stepsize * prox_lambda)
        next_xk = self.proximal_operator(next_xk, prox_lambda, scaling=stepsize)

        return next_xk

    @partial(jit, static_argnums=(0,))
    def update(
        self,
        current_params: Any,
        state: SVRGState,
        prox_lambda: float,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> OptStep:
        """
        Perform a single parameter update on the passed data (no random sampling or loops)
        and increment `state.iter_num`.

        Please note that this is called by `GLM.update`, but repeated calls to `GLM.update`
        on mini-batches passed to it will not result in running the full (Prox-)SVRG,
        and parts of the algorithm will have to be implemented outside.

        Parameters
        ----------
        current_params : Any
            Parameters at the end of the previous update, used as the starting point for the current update.
        state : SVRGState
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        prox_lambda : float
            Regularization strength.
        X :
            Input data.
        y : jnp.ndarray
            Output data.


        Returns
        -------
        OptStep
            xs : Any
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state : SVRGState
                Updated state.
        """
        # return self._update_per_random_samples(current_params, state, prox_lambda, X, y)
        if state.df_xs is None:
            raise ValueError(
                "Full gradient at the anchor point (state.df_xs) has to be set. "
                + "Try passing init_full_gradient=True to ProxSVRG.init_state or GLM.initialize_solver."
            )
        return self._update_on_batch(current_params, state, prox_lambda, X, y)

    @partial(jit, static_argnums=(0,))
    def _update_on_batch(
        self,
        current_params: Any,
        state: SVRGState,
        prox_lambda: float,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> OptStep:
        """
        Update parameters given a mini-batch of data and increment iteration/epoch number in state.

        Note that this method doesn't update state.x_av, state.xs, state.df_xs, that has to be done outside.
        """
        # NOTE this doesn't update state.x_av, state.xs, state.df_xs, that has to be done outside

        next_params = self._xk_update(
            current_params, state.xs, state.df_xs, state.stepsize, prox_lambda, X, y
        )

        state = state._replace(
            iter_num=state.iter_num + 1,
        )

        return OptStep(params=next_params, state=state)

    @partial(jit, static_argnums=(0,))
    def run(
        self,
        init_params: Any,
        prox_lambda: float,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called from `GLM.fit` and assumes that X and y are the full data set.

        Parameters
        ----------
        init_params : Any
            Initial parameters to start from.
        prox_lambda : float
            Regularization strength.
        X :
            Input data.
        y : jnp.ndarray
            Output data.

        Returns
        -------
        OptStep
            final_xs : Any
                Parameters at the end of the last innner loop.
                (... or the average of the parameters over the last inner loop)
            final_state : SVRGState
                Final optimizer state.
        """
        # initialize the state, including the full gradient at the initial parameters
        init_state = self.init_state(
            init_params,
            prox_lambda,
            X,
            y,
            init_full_gradient=True,
        )

        # evaluate the loss for the initial parameters, aka iter_num=0
        # init_state = init_state._replace(
        #    loss_log=self._update_loss_log(init_state.loss_log, 0, init_params, X, y)
        # )

        # this method assumes that args hold the full data
        def body_fun(step):
            xs_prev, state = step

            # evaluate and store the full gradient with the params from the last inner loop
            state = state._replace(
                df_xs=self.loss_gradient(xs_prev, X, y),
            )

            # run an update over the whole data
            xk, state = self._update_per_random_samples(
                xs_prev, state, prox_lambda, X, y
            )

            # update xs with the final xk or an average over the inner loop's iterations
            xs = xk
            # xs = state.x_av

            state = state._replace(
                xs=xs,
                error=self._error(xs, xs_prev, state.stepsize),
                # loss_log=self._update_loss_log(
                #    state.loss_log, state.iter_num, xs, X, y
                # ),
            )

            return OptStep(params=xs, state=state)

        # at the end of each epoch, check for convergence or reaching the max number of epochs
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
    def _update_per_random_samples(
        self,
        current_params: Any,
        state: SVRGState,
        prox_lambda: float,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> OptStep:
        """
        Performs the inner loop of Prox-SVRG sweeping through approximately one full epoch,
        updating the parameters after sampling a mini-batch on each iteration.

        Parameters
        ----------
        current_params : Any
            Parameters at the end of the previous update, used as the starting point for the current update.
        state : SVRGState
            Optimizer state at the end of the previous sweep.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        prox_lambda : float
            Regularization strength. Can be None.
        X :
            Input data
        y :
            Output data.

        Returns
        -------
        OptStep
            xs : Any
                Parameters at the end of the last inner loop.
                (... or the average of the parameters over the last inner loop)
            state : SVRGState
                Updated state.
        """

        N = y.shape[0]  # number of data points x number of dimensions

        m = (N + self.batch_size - 1) // self.batch_size  # number of iterations
        # m = N

        xs, df_xs = state.xs, state.df_xs

        def inner_loop_body(_, carry):
            xk, x_sum, key = carry

            # sample mini-batch or data point
            key, subkey = random.split(key)
            ind = random.randint(subkey, (self.batch_size,), 0, N)

            # perform a single update on the mini-batch or data point
            xk = self._xk_update(
                xk, xs, df_xs, state.stepsize, prox_lambda, tree_slice(X, ind), y[ind]
            )

            # update the sum used for the averaging
            x_sum = tree_add(x_sum, xk)

            return (xk, x_sum, key)

        xk, x_sum, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (
                current_params,
                tree_zeros_like(xs),  # initialize the sum to zero
                state.key,
            ),
        )

        # update the state
        # storing the average over the inner loop to potentially use it in the run loop
        state = state._replace(
            iter_num=state.iter_num + 1,
            key=key,
            x_av=tree_scalar_mul(1 / m, x_sum),
        )

        # the next anchor point is the parameters at the end of the inner loop
        # or the average over the inner loop
        return OptStep(params=xk, state=state)
        # return OptStep(params=state.x_av, state=state)

    # @staticmethod
    # def _error(x, x_prev, stepsize):
    #    diff_norm = tree_l2_norm(tree_sub(x, x_prev))
    #    return diff_norm / stepsize

    @staticmethod
    def _error(x, x_prev, stepsize):
        return tree_l2_norm(tree_sub(x, x_prev)) / tree_l2_norm(x_prev)

    # @staticmethod
    # def _error(x, x_prev, stepsize):
    #    # adapted from scikit-learn's SAG solver
    #    diff = tree_sub(x, x_prev)
    #    flat_diff, _ = jax.flatten_util.ravel_pytree(diff)
    #    return jnp.max(jnp.abs(flat_diff))


class SVRG(ProxSVRG):
    """
    SVRG solver

    Equivalent to ProxSVRG with prox as the identity function and prox_lambda=None.

    Attributes
    ----------
    fun: Callable
        smooth function of the form ``fun(x, *args, **kwargs)``.
    maxiter : int
        Maximum number of epochs to run the optimization for.
    key : jax.random.PRNGkey
        jax PRNGKey to start with. Used for sampling random data points.
    stepsize : float
        Constant step size to use.
    tol: float
        Tolerance level for the error when comparing parameters
        at the end of consecutive epochs to check for convergence.
    batch_size: int
        Number of data points to sample per inner loop iteration.

    Example
    -------
    def loss_fn(params, X, y):
        ...

    svrg = SVRG(loss_fn)
    params, state = svrg.run(init_params, X, y)

    References
    ----------
    Prox-SVRG - https://arxiv.org/abs/1403.4699v1
    SVRG - https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html
    """

    def __init__(
        self,
        fun: Callable,
        maxiter: int = 10_000,
        key: Optional[KeyArrayLike] = None,
        stepsize: float = 1e-3,
        tol: float = 1e-3,
        batch_size: int = 1,
    ):
        super().__init__(
            fun,
            prox_none,
            maxiter,
            key,
            stepsize,
            tol,
            batch_size,
        )

    def init_state(self, init_params: Any, *args, **kwargs):
        """
        Initialize the solver state

        Parameters
        ----------
        init_params : Any
            pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        args:
            If 2 positional arguments are passed, they are assumed to be X and y,
            and prox_lambda is substituted with None.

            (prox_lambda : Optional, float)
                Regularization strength.
                When calling `SVRG.init_state` by hand, this doesn't need to be passed,
                but within `SVRG.run` None is passed.
            X :
                Input data.
            y :
                Output data.

        init_full_gradient : bool, default False
            Whether to calculate the full gradient at the initial parameters,
            assuming that X, y are the full data set, and store this gradient in the initial state.
        """
        # substitute None for prox_lambda
        if len(args) == 2:
            args = (None, *args)
        return super().init_state(init_params, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update(self, current_params: Any, state: SVRGState, *args, **kwargs):
        """
        Perform a single parameter update on the passed data (no random sampling or loops)
        and increment `state.iter_num`.

        Please note that this is called by `GLM.update`, but repeated calls to `GLM.update`
        on mini-batches passed to it will not result in running the full (Prox-)SVRG,
        and parts of the algorithm will have to be implemented outside.

        Parameters
        ----------
        current_params : Any
            Parameters at the end of the previous update, used as the starting point for the current update.
        state : SVRGState
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        args:
            If 2 positional arguments are passed, they are assumed to be X and y,
            and prox_lambda is substituted with None.

            (prox_lambda : Optional, float)
                Regularization strength.
                When calling `SVRG.update` or `GLM.update` by hand, this doesn't need to be passed.
            X :
                Input data.
            y : jnp.ndarray
                Output data.

        Returns
        -------
        OptStep
            xs : Any
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state : SVRGState
                Updated state.
        """
        if len(args) == 2:
            args = (None, *args)
        return super().update(current_params, state, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def run(self, init_params: Any, *args, **kwargs):
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called from `GLM.fit` and assumes that X and y are the full data set.

        Parameters
        ----------
        init_params : Any
            Initial parameters to start from.
        prox_lambda : float
            Regularization strength.
        args:
            If 2 positional arguments are passed, they are assumed to be X and y,
            and prox_lambda is substituted with None.

            (prox_lambda : Optional, float)
                Regularization strength.
            X :
                Input data.
            y : jnp.ndarray
                Output data.

        Returns
        -------
        OptStep
            final_xs : Any
                Parameters at the end of the last innner loop.
                (... or the average of the parameters over the last inner loop)
            final_state : SVRGState
                Final optimizer state.
        """
        if len(args) == 2:
            args = (None, *args)
        return super().run(init_params, *args, **kwargs)
