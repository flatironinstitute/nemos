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

from .tree_utils import (
    tree_add,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_scalar_mul,
    tree_slice,
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
    xs: Optional[tuple] = None
    df_xs: Optional[tuple] = None


class ProxSVRG:
    """
    Prox-SVRG solver

    Borrowing from jaxopt.ProximalGradient, this solver minimizes:

      objective(params, hyperparams_prox, *args, **kwargs) =
        fun(params, *args, **kwargs) + non_smooth(params, hyperparams_prox)

    Attributes
    ----------
    fun: Callable
        Smooth function of the form ``fun(x, *args, **kwargs)``.
    prox: Callable
        Proximal operator associated with the function ``non_smooth``.
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
        *args,
        init_full_gradient: bool = False,
    ) -> SVRGState:
        """
        Initialize the solver state

        Parameters
        ----------
        init_params : Any
            Pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        hyperparams_prox : float
            Parameters of the proximal operator, in our case the regularization strength.
            Not used here, but required to be consistent with the jaxopt API.
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
                    Output data.
        init_full_gradient : bool, default False
            Whether to calculate the full gradient at the initial parameters,
            assuming that args hold the full data set, and store this gradient in the initial state.

        Returns
        -------
        state : SVRGState
            Initialized optimizer state
        """
        df_xs = None
        if init_full_gradient:
            df_xs = self.loss_gradient(init_params, *args)

        state = SVRGState(
            iter_num=0,
            key=self.key if self.key is not None else random.key(123),
            error=jnp.inf,
            stepsize=self.stepsize,
            xs=init_params,
            df_xs=df_xs,
        )
        return state

    @partial(jit, static_argnums=(0,))
    def _xk_update(
        self,
        xk: Any,
        xs: Any,
        df_xs: Any,
        stepsize: float,
        prox_lambda: float,
        *args,
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
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Hyperparameters passed to `prox` and positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input datapoint or mini-batch.
                y :
                    Output datapoint or mini-batch.
        """
        dfik_xk = self.loss_gradient(xk, *args)
        dfik_xs = self.loss_gradient(xs, *args)

        gk = jax.tree_util.tree_map(lambda a, b, c: a - b + c, dfik_xk, dfik_xs, df_xs)

        next_xk = tree_add_scalar_mul(xk, -stepsize, gk)

        next_xk = self.proximal_operator(next_xk, prox_lambda, scaling=stepsize)

        return next_xk

    @partial(jit, static_argnums=(0,))
    def update(
        self,
        current_params: Any,
        state: SVRGState,
        prox_lambda: float,
        *args,
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
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
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
        return self._update_on_batch(current_params, state, prox_lambda, *args)

    @partial(jit, static_argnums=(0,))
    def _update_on_batch(
        self,
        current_params: Any,
        state: SVRGState,
        prox_lambda: float,
        *args,
    ) -> OptStep:
        """
        Update parameters given a mini-batch of data and increment iteration/epoch number in state.

        Note that this method doesn't update state.xs, state.df_xs, that has to be done outside.

        Parameters
        ----------
        current_params : Any
            Parameters at the end of the previous update, used as the starting point for the current update.
        state : SVRGState
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        prox_lambda : float
            Regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
                    Output data.

        Returns
        -------
        OptStep
            xs : Any
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state : SVRGState
                Updated state.
        """
        # NOTE this doesn't update state.xs, state.df_xs, that has to be done outside

        next_params = self._xk_update(
            current_params, state.xs, state.df_xs, state.stepsize, prox_lambda, *args
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
        *args,
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
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
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
            *args,
            init_full_gradient=True,
        )

        return self._run(init_params, init_state, prox_lambda, *args)

    @partial(jit, static_argnums=(0,))
    def _run(
        self,
        init_params: Any,
        init_state: SVRGState,
        prox_lambda: float,
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called from `GLM.fit` and assumes that X and y are the full data set.
        Assumes the state has been initialized, which works a bit differently for SVRG and ProxSVRG.

        Parameters
        ----------
        init_params : Any
            Initial parameters to start from.
        init_state : SVRGState
            Initialized optimizer state returned by `ProxSVRG.init_state`
        prox_lambda : float
            Regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
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

        # this method assumes that args hold the full data
        def body_fun(step):
            xs_prev, state = step

            # evaluate and store the full gradient with the params from the last inner loop
            state = state._replace(
                df_xs=self.loss_gradient(xs_prev, *args),
            )

            # run an update over the whole data
            xk, state = self._update_per_random_samples(
                xs_prev, state, prox_lambda, *args
            )

            # update xs with the final xk or an average over the inner loop's iterations
            xs = xk

            state = state._replace(
                xs=xs,
                error=self._error(xs, xs_prev, state.stepsize),
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
        *args,
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
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
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
        n_points_per_arg = {leaf.shape[0] for leaf in jax.tree.leaves(args)}
        if not len(n_points_per_arg) == 1:
            raise ValueError("All arguments must have the same sized first dimension.")
        N = n_points_per_arg.pop()

        m = (N + self.batch_size - 1) // self.batch_size  # number of iterations
        # m = N

        xs, df_xs = state.xs, state.df_xs

        def inner_loop_body(_, carry):
            xk, key = carry

            # sample mini-batch or data point
            key, subkey = random.split(key)
            ind = random.randint(subkey, (self.batch_size,), 0, N)

            # perform a single update on the mini-batch or data point
            xk = self._xk_update(
                xk,
                xs,
                df_xs,
                state.stepsize,
                prox_lambda,
                *(tree_slice(arg, ind) for arg in args),
            )

            return (xk, key)

        xk, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (current_params, state.key),
        )

        # update the state
        # storing the average over the inner loop to potentially use it in the run loop
        state = state._replace(
            iter_num=state.iter_num + 1,
            key=key,
        )

        # the next anchor point is the parameters at the end of the inner loop
        # (or the average over the inner loop)
        return OptStep(params=xk, state=state)

    @staticmethod
    def _error(x, x_prev, stepsize):
        """
        Calculate the magnitude of the update relative to the parameters.
        Used for terminating the algorithm if a certain tolerance is reached.

        Params
        ------
        x :
            Parameter values after the update.
        x_prev :
            Previous parameter values.

        Returns
        -------
        Scaled update magnitude.
        """
        # stepsize is an argument to be consistent with jaxopt
        return tree_l2_norm(tree_sub(x, x_prev)) / tree_l2_norm(x_prev)


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

    def init_state(self, init_params: Any, *args, **kwargs) -> SVRGState:
        """
        Initialize the solver state

        Parameters
        ----------
        init_params : Any
            pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
                    Output data.

        init_full_gradient : bool, default False
            Whether to calculate the full gradient at the initial parameters,
            assuming that args hold the full data set, and store this gradient in the initial state.

        Returns
        -------
        state : SVRGState
            Initialized optimizer state
        """
        # substitute None for prox_lambda
        return super().init_state(init_params, None, *args, **kwargs)

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
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
                    Output data.

        Returns
        -------
        OptStep
            xs : Any
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state : SVRGState
                Updated state.
        """
        # substitute None for prox_lambda
        return super().update(current_params, state, None, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def run(
        self,
        init_params: Any,
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called from `GLM.fit` and assumes that X and y are the full data set.

        Parameters
        ----------
        init_params : Any
            Initial parameters to start from.
        args:
            Positional arguments passed to loss function `fun` and its gradient.
            For GLMs it is:
                X :
                    Input data.
                y :
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
        # don't have to pass prox_lambda here
        init_state = self.init_state(
            init_params,
            *args,
            init_full_gradient=True,
        )

        # substitute None for prox_lambda
        return self._run(init_params, init_state, None, *args)
