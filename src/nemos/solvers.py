import warnings
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import grad, jit, lax, random
from jaxopt import OptStep
from jaxopt._src import loop
from jaxopt.prox import prox_none

from .tree_utils import tree_add_scalar_mul, tree_l2_norm, tree_slice, tree_sub
from .typing import KeyArrayLike, Pytree


class SVRGState(NamedTuple):
    """
    Optimizer state for (Prox)SVRG.

    Fields
    ------
    iter_num :
        Current epoch or iteration number.
    key :
        Random key to use when sampling data points or mini-batches.
    error :
        Scaled difference (~distance) between subsequent parameter values
        used to monitor convergence.
    stepsize :
        Step size of the individual gradient steps.
    xs :
        Anchor/reference point where the full gradient is calculated in the SVRG algorithm.
        Naming follows pseudocode in [1]_
    df_xs :
        Full gradient at the anchor/reference point.

    References
    ----------
    .. 1. [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).](https://arxiv.org/abs/2010.00892)
    """

    iter_num: int
    key: KeyArrayLike
    error: float
    stepsize: float
    xs: Optional[Pytree] = None
    df_xs: Optional[Pytree] = None


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

    Examples
    --------
    >>> def loss_fn(params, X, y):
    >>>    ...
    >>>
    >>> svrg = ProxSVRG(loss_fn, prox_fun)
    >>> params, state = svrg.run(init_params, prox_lambda, X, y)

    References
    ----------
    .. 1. [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).](https://arxiv.org/abs/2010.00892)
    .. 2. [Xiao, Lin, and Tong Zhang. "A proximal stochastic gradient method with progressive variance reduction." SIAM Journal on Optimization 24.4 (2014): 2057-2075.](https://arxiv.org/abs/1403.4699v1)
    .. 3. [Johnson, Rie, and Tong Zhang. "Accelerating stochastic gradient descent using predictive variance reduction." Advances in neural information processing systems 26 (2013).](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
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
        init_params: Pytree,
        hyperparams_prox: Any,
        *args,
        init_full_gradient: bool = False,
    ) -> SVRGState:
        """
        Initialize the solver state

        Parameters
        ----------
        init_params :
            Pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        hyperparams_prox :
            Parameters of the proximal operator, in our case the regularization strength.
            Not used here, but required to be consistent with the jaxopt API.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.
        init_full_gradient : bool, default False
            Whether to calculate the full gradient at the initial parameters,
            assuming that args hold the full data set, and store this gradient in the initial state.

        Returns
        -------
        state :
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
    def _xk_update_step(
        self,
        xk: Pytree,
        xs: Pytree,
        df_xs: Pytree,
        stepsize: float,
        prox_lambda: Union[float, None],
        *args,
    ) -> Pytree:
        """
        Body of the inner loop of Prox-SVRG that takes a step.

        Parameters
        ----------
        xk :
            Current parameters.
        xs :
            Anchor point.
        df_xs :
            Full gradient at the anchor point.
        stepsize :
            Step size.
        prox_lambda :
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        Returns
        -------
        next_xk :
            Parameter values after applying the update.
        """
        # gradient on batch_{i_k} evaluated at the current parameters (xk)
        grad_of_fik_at_xk = self.loss_gradient(xk, *args)
        # gradient on batch_{i_k} evaluated at the anchor point (xs)
        grad_of_fik_at_xs = self.loss_gradient(xs, *args)

        # SVRG gradient estimate
        gk = jax.tree_util.tree_map(
            lambda a, b, c: a - b + c, grad_of_fik_at_xk, grad_of_fik_at_xs, df_xs
        )

        # x_{k+1} = x_{k} - stepsize * g_{k}
        next_xk = tree_add_scalar_mul(xk, -stepsize, gk)

        # apply the proximal operator
        next_xk = self.proximal_operator(next_xk, prox_lambda, scaling=stepsize)

        return next_xk

    @partial(jit, static_argnums=(0,))
    def update(
        self,
        current_params: Pytree,
        state: SVRGState,
        prox_lambda: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Perform a single parameter update on the passed data (no random sampling or loops)
        and increment `state.iter_num`.

        Please note that this gets called by `BaseRegressor._solver_update` (e.g., as called by `GLM.update`),
        but repeated calls to `(Prox)SVRG.update` (so in turn e.g. to `GLM.update`) on mini-batches passed to it
        will not result in running the full (Prox-)SVRG, and parts of the algorithm will have to be implemented outside.

        Parameters
        ----------
        current_params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        prox_lambda :
            Regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.


        Returns
        -------
        OptStep
            xs :
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state :
                Updated state.

        Raises
        ------
        ValueError
            The parameter update needs a value for the full gradient at the anchor point, which needs the full data
            to be calculated and is expected to be stored in state.df_xs. So if state.df_xs is None, a ValueError is raised.
        """
        if state.df_xs is None:
            raise ValueError(
                "Full gradient at the anchor point (state.df_xs) has to be set. "
                + "Try passing init_full_gradient=True to ProxSVRG.init_state or GLM.initialize_solver."
            )
        return self._update_on_batch(current_params, state, prox_lambda, *args)

    @partial(jit, static_argnums=(0,))
    def _update_on_batch(
        self,
        current_params: Pytree,
        state: SVRGState,
        prox_lambda: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Update parameters given a mini-batch of data and increment iteration/epoch number in state.

        Note that this method doesn't update state.xs, state.df_xs, that has to be done outside.

        Parameters
        ----------
        current_params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        prox_lambda :
            Regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        Returns
        -------
        OptStep
            xs :
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state :
                Updated state.
        """
        next_params = self._xk_update_step(
            current_params, state.xs, state.df_xs, state.stepsize, prox_lambda, *args
        )

        state = state._replace(
            iter_num=state.iter_num + 1,
        )

        return OptStep(params=next_params, state=state)

    @partial(jit, static_argnums=(0,))
    def run(
        self,
        init_params: Pytree,
        prox_lambda: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called by `BaseRegressor._solver_run` (e.g. as called by `GLM.fit`) and assumes that X and y are the full data set.

        Parameters
        ----------
        init_params :
            Initial parameters to start from.
        prox_lambda :
            Regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        Returns
        -------
        OptStep
            final_params :
                Parameters at the end of the last innner loop.
                (... or the average of the parameters over the last inner loop)
            final_state :
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
        init_params: Pytree,
        init_state: SVRGState,
        prox_lambda: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called by `BaseRegressor._solver_run` (e.g. as called by `GLM.fit`) and assumes that X and y are the full data set.
        Assumes the state has been initialized, which works a bit differently for SVRG and ProxSVRG.

        Parameters
        ----------
        init_params :
            Initial parameters to start from.
        init_state :
            Initialized optimizer state returned by `ProxSVRG.init_state`
        prox_lambda :
            Regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.
        Returns
        -------
        OptStep
            final_params :
                Parameters at the end of the last innner loop.
                (... or the average of the parameters over the last inner loop)
            final_state :
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

        final_params, final_state = loop.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=OptStep(params=init_params, state=init_state),
            maxiter=self.maxiter,
            jit=True,
        )
        return OptStep(params=final_params, state=final_state)

    @partial(jit, static_argnums=(0,))
    def _update_per_random_samples(
        self,
        current_params: Pytree,
        state: SVRGState,
        prox_lambda: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Performs the inner loop of Prox-SVRG sweeping through approximately one full epoch,
        updating the parameters after sampling a mini-batch on each iteration.

        Parameters
        ----------
        current_params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous sweep.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        prox_lambda :
            Regularization strength. Can be None.
        args :
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        Returns
        -------
        OptStep
            xs :
                Parameters at the end of the last inner loop.
                (... or the average of the parameters over the last inner loop)
            state :
                Updated state.

        Raises
        ------
        ValueError
            If not all arguments in args have the same sized first dimension.
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
            xk = self._xk_update_step(
                xk,
                xs,
                df_xs,
                state.stepsize,
                prox_lambda,
                *tree_slice(args, ind),
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

    Examples
    --------
    >>> def loss_fn(params, X, y):
    >>>    ...
    >>>
    >>> svrg = SVRG(loss_fn)
    >>> params, state = svrg.run(init_params, X, y)

    References
    ----------
    .. 1. [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).](https://arxiv.org/abs/2010.00892)
    .. 2. [Xiao, Lin, and Tong Zhang. "A proximal stochastic gradient method with progressive variance reduction." SIAM Journal on Optimization 24.4 (2014): 2057-2075.](https://arxiv.org/abs/1403.4699v1)
    .. 3. [Johnson, Rie, and Tong Zhang. "Accelerating stochastic gradient descent using predictive variance reduction." Advances in neural information processing systems 26 (2013).](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
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

    def init_state(self, init_params: Pytree, *args, **kwargs) -> SVRGState:
        """
        Initialize the solver state

        Parameters
        ----------
        init_params :
            pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        init_full_gradient : bool, default False
            Whether to calculate the full gradient at the initial parameters,
            assuming that args hold the full data set, and store this gradient in the initial state.

        Returns
        -------
        state :
            Initialized optimizer state
        """
        # substitute None for prox_lambda
        return super().init_state(init_params, None, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update(
        self, current_params: Pytree, state: SVRGState, *args, **kwargs
    ) -> OptStep:
        """
        Perform a single parameter update on the passed data (no random sampling or loops)
        and increment `state.iter_num`.

        Please note that this gets called by `BaseRegressor._solver_update` (e.g., as called by `GLM.update`),
        but repeated calls to `(Prox)SVRG.update` (so in turn e.g. to `GLM.update`) on mini-batches passed to it
        will not result in running the full (Prox-)SVRG, and parts of the algorithm will have to be implemented outside.

        Parameters
        ----------
        current_params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (xs) and the gradient at the anchor point (df_xs) already set.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        Returns
        -------
        OptStep
            xs :
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state :
                Updated state.

        Raises
        ------
        ValueError
            The parameter update needs a value for the full gradient at the anchor point, which needs the full data
            to be calculated and is expected to be stored in state.df_xs. So if state.df_xs is None, a ValueError is raised.
        """
        # substitute None for prox_lambda
        return super().update(current_params, state, None, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def run(
        self,
        init_params: Pytree,
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called by `BaseRegressor._solver_run` (e.g. as called by `GLM.fit`) and assumes that X and y are the full data set.

        Parameters
        ----------
        init_params :
            Initial parameters to start from.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(current_params, *args)`),
            most likely input and output data.
            They are expected to be Pytrees with arrays or FeaturePytree as their leaves, with all of their
            leaves having the same sized first dimension (corresponding to the number of data points).
            For GLMs these are:
                X : DESIGN_INPUT_TYPE
                    Input data.
                y : jnp.ndarray
                    Output data.

        Returns
        -------
        OptStep
            final_params :
                Parameters at the end of the last innner loop.
                (... or the average of the parameters over the last inner loop)
            final_state :
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


def softplus_poisson_optimal_stepsize(
    X: jnp.ndarray, y: jnp.ndarray, batch_size: int, n_power_iters: Optional[int] = None
):
    """
    Calculate the optimal stepsize to use for SVRG with a GLM that uses
    Poisson observations and softplus inverse link function.

    Parameters
    ----------
    X : jnp.ndarray
        Input data.
    y : jnp.ndarray
        Output data.
    batch_size : int
        Mini-batch size, i.e. number of data points sampled for
        each inner update of SVRG.
    n_power_iters: int, optional, default None
        If None, build the XDX matrix (which has a shape of n_features x n_features)
        and find its eigenvalues directly.
        If an integer, it is the max number of iterations to run the power
        iteration for when finding the largest eigenvalue.

    Returns
    -------
    stepsize : scalar jax array
        Optimal stepsize to use
    """
    L_max, L = _softplus_poisson_L_max_and_L(jnp.array(X), jnp.array(y), n_power_iters)

    stepsize = _calc_alpha(batch_size, X.shape[0], L_max, L)

    return stepsize


# not using the previous one to avoid calculating L and L_max twice
def softplus_poisson_optimal_batch_and_stepsize(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_power_iters: Optional[int] = None,
    default_batch_size: int = 1,
    default_stepsize: float = 1e-3,
):
    """
    Calculate the optimal batch size and step size to use for SVRG with a GLM
    that uses Poisson observations and softplus inverse link function.

    Parameters
    ----------
    X : jnp.ndarray
        Input data.
    y : jnp.ndarray
        Output data.
    n_power_iters: int, optional, default None
        If None, build the XDX matrix (which has a shape of n_features x n_features)
        and find its eigenvalues directly.
        If an integer, it is the max number of iterations to run the power
        iteration for when finding the largest eigenvalue.
    default_batch_size : int
        Batch size to fall back on if the calculation fails.
    default_stepsize: float
        Step size to fall back on if the calculation fails.

    Returns
    -------
    batch_size : int
        Optimal batch size to use.
    stepsize : scalar jax array
        Optimal stepsize to use.
    """
    L_max, L = _softplus_poisson_L_max_and_L(jnp.array(X), jnp.array(y), n_power_iters)

    batch_size = jnp.floor(_calc_b_hat(X.shape[0], L_max, L))

    if not jnp.isfinite(batch_size):
        batch_size = default_batch_size
        stepsize = default_stepsize

        warnings.warn(
            f"Could not determine batch and step size automatically. Falling back on the default values of {batch_size} and {default_stepsize}."
        )
    else:
        stepsize = _calc_alpha(batch_size, X.shape[0], L_max, L)

    return int(batch_size), stepsize


def _softplus_poisson_L_max_and_L(
    X: jnp.ndarray, y: jnp.ndarray, n_power_iters: Optional[int] = None
):
    """
    Calculate the smoothness constant and maximum smoothness constant for SVRG
    assuming that the optimized function is the log-likelihood of a Poisson GLM
    with a softplus inverse link function.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.
    n_power_iters :
        If None, calculate X.T @ D @ X and its largest eigenvalue directly.
        If an integer, the umber of power iterations to use to calculate the largest eigenvalue.

    Returns
    -------
    L_max, L :
        Maximum smoothness constant and smoothness constant.
    """
    L = _softplus_poisson_L(X, y, n_power_iters)
    L_max = _softplus_poisson_L_max(X, y)

    return L_max, L


def _softplus_poisson_L_multiply(X, y, v):
    """
    Perform the multiplication of v with X.T @ D @ X without forming the full X.T @ D @ X,
    and iterating through the rows of X and y instead.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.
    v :
        d-dimensionl vector.

    Returns
    -------
    X.T @ D @ X @ v
    """
    N, _ = X.shape

    def body_fun(i, current_sum):
        return current_sum + (0.17 * y[i] + 0.25) * jnp.outer(X[i, :], X[i, :]) @ v

    v_new = jax.lax.fori_loop(0, N, body_fun, v)

    return v_new / N


def _softplus_poisson_L_with_power_iteration(X, y, n_power_iters: int = 5):
    """
    Instead of calculating X.T @ D @ X and its largest eigenvalue directly,
    calculate it using the power method and by iterating through X and y,
    forming a small product at a time.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.
    n_power_iters :
        Number of power iterations.

    Returns
    -------
    The largest eigenvalue of X.T @ D @ X
    """
    # key is fixed to random.key(0)
    _, d = X.shape

    # initialize to random d-dimensional vector
    v = random.normal(jax.random.key(0), (d,))

    # run the power iteration until convergence or the max steps
    for _ in range(n_power_iters):
        v_prev = v.copy()
        v = _softplus_poisson_L_multiply(X, y, v)

        if jnp.allclose(v_prev, v):
            break

    # calculate the eigenvalue
    return jnp.linalg.norm(_softplus_poisson_L_multiply(X, y, v)) / jnp.linalg.norm(v)


def _softplus_poisson_XDX(X, y):
    """
    Calculate the X.T @ D @ X matrix for use in calculating the smoothness constant L.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.

    Returns
    -------
    XDX :
        d x d matrix
    """
    N, d = X.shape

    def body_fun(i, current_sum):
        return current_sum + (0.17 * y[i] + 0.25) * jnp.outer(X[i, :], X[i, :])

    # xi = jax.lax.dynamic_slice(X, (i, 0), (1, d)).reshape((d,))
    # yi = jax.lax.dynamic_slice(y, (i, 0), (1, 1))
    # return current_sum + (0.17 * yi + 0.25) * jnp.outer(xi, xi)

    # will be d x d
    XDX = jax.lax.fori_loop(0, N, body_fun, jnp.zeros((d, d)))

    return XDX / N


def _softplus_poisson_L(
    X: jnp.ndarray, y: jnp.ndarray, n_power_iters: Optional[int] = None
):
    """
    Calculate the smoothness constant from data, assuming that the optimized
    function is the log-likelihood of a Poisson GLM with a softplus inverse link function.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.

    Returns
    -------
    L :
        Smoothness constant of f.
    """
    if n_power_iters is None:
        # calculate XDX and its largest eigenvalue directly
        return jnp.sort(jnp.linalg.eigvals(_softplus_poisson_XDX(X, y)).real)[-1]
    else:
        # use the power iteration to calculate the larget eigenvalue
        return _softplus_poisson_L_with_power_iteration(X, y, n_power_iters)


def _softplus_poisson_L_max(X: jnp.ndarray, y: jnp.ndarray):
    """
    Calculate the maximum smoothness constant from data, assuming that
    the optimized function is the log-likelihood of a Poisson GLM with
    a softplus inverse link function.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.

    Returns
    -------
    L_max :
        Maximum smoothness constant among f_{i}.
    """
    N, _ = X.shape

    def body_fun(i, current_max):
        return jnp.maximum(
            current_max, jnp.linalg.norm(X[i, :]) ** 2 * (0.17 * y[i] + 0.25)
        )

    L_max = jax.lax.fori_loop(0, N, body_fun, jnp.array([0.0]))

    return L_max[0]


def _calc_b_hat(N: int, L_max: float, L: float):
    """
    Calculate optimal batch size according to Sebbouh et al. 2019.

    Parameters
    ----------
    N :
        Overall number of data points.
    L_max :
        Maximum smoothness constant among f_{i}.
    L :
        Smoothness constant.

    Returns
    -------
    b_hat :
        Optimal batch size for the optimization.
    """
    with jax.experimental.enable_x64():
        return jnp.sqrt(N / 2 * (3 * L_max - L) / (N * L - 3 * L_max))


def _calc_alpha(b: int, N: int, L_max: float, L: float):
    """
    Calculate optimal step size according to Sebbouh et al. 2019.

    Parameters
    ----------
    b :
        Mini-batch size.
    N :
        Overall number of data points.
    L_max :
        Maximum smoothness constant among f_{i}.
    L :
        Smoothness constant.

    Returns
    -------
    alpha :
        Optimal step size for the optimization.
    """
    with jax.experimental.enable_x64():
        return 1 / 2 * b * (N - 1) / (3 * (N - b) * L_max + N * (b - 1) * L)
