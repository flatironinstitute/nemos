from functools import partial
from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import grad, jit, lax, random
from jaxopt import OptStep
from jaxopt._src import loop
from jaxopt.prox import prox_none

from ..tree_utils import tree_add_scalar_mul, tree_l2_norm, tree_slice, tree_sub
from ..typing import KeyArrayLike, Pytree


class SVRGState(NamedTuple):
    """
    Optimizer state for (Prox)SVRG.

    Attributes
    ----------
    iter_num :
        Current epoch or iteration number.
    key :
        Random key to use when sampling data points or mini-batches.
    error :
        Scaled difference (~distance) between subsequent parameter values
        used to monitor convergence.
    stepsize :
        Step size of the individual gradient steps.
    reference_point :
        Anchor/reference/snapshot point where the full gradient is calculated in the SVRG algorithm.
        Corresponds to $x_{s}$ in the pseudocode[$^{[1]}$](#references).
    full_grad_at_reference_point :
        Full gradient at the anchor/reference point.

    # References
    ------------
    [1] [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik.
        "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).
        ](https://arxiv.org/abs/2010.00892)
    """

    iter_num: int
    key: KeyArrayLike
    error: float
    stepsize: float
    reference_point: Optional[Pytree] = None
    full_grad_at_reference_point: Optional[Pytree] = None


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
    >>> import numpy as np
    >>> from jaxopt.prox import prox_lasso
    >>> loss_fn = lambda params, X, y: ((X.dot(params) - y)**2).sum()
    >>> svrg = ProxSVRG(loss_fn, prox_lasso)
    >>> hyperparams_prox = 0.1
    >>> params, state = svrg.run(np.zeros(2), hyperparams_prox, np.ones((10, 2)), np.zeros(10))


    References
    ----------
    [1] [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik.
    "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).
    ](https://arxiv.org/abs/2010.00892)

    [2] [Xiao, Lin, and Tong Zhang.
    "A proximal stochastic gradient method with progressive variance reduction."
    SIAM Journal on Optimization 24.4 (2014): 2057-2075.](https://arxiv.org/abs/1403.4699v1)

    [3] [Johnson, Rie, and Tong Zhang.
    "Accelerating stochastic gradient descent using predictive variance reduction."
    Advances in neural information processing systems 26 (2013).
    ](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
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
        *args,
    ) -> SVRGState:
        """
        Initialize the solver state

        Parameters
        ----------
        init_params :
            Pytree containing the initial parameters.
            For GLMs it's a tuple of (W, b)
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
        state :
            Initialized optimizer state
        """
        state = SVRGState(
            iter_num=0,
            key=self.key if self.key is not None else random.key(123),
            error=jnp.inf,
            stepsize=self.stepsize,
            reference_point=init_params,
            full_grad_at_reference_point=None,
        )
        return state

    @partial(jit, static_argnums=(0,))
    def _inner_loop_param_update_step(
        self,
        params: Pytree,
        reference_point: Pytree,
        full_grad_at_reference_point: Pytree,
        stepsize: float,
        hyperparams_prox: Union[float, None],
        *args,
    ) -> Pytree:
        """
        Body of the inner loop of Prox-SVRG that takes a step.

        Parameters
        ----------
        params :
            Current parameters.
        reference_point :
            Anchor point.
        full_grad_at_reference_point :
            Full gradient at the anchor point.
        stepsize :
            Step size.
        hyperparams_prox :
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
        next_params :
            Parameter values after applying the update.
        """
        # gradient on batch_{i_k} evaluated at the current parameters
        # gradient of f_{i_k} at x_{k} in the pseudocode of Gower et al. 2020
        minibatch_grad_at_current_params = self.loss_gradient(params, *args)
        # gradient on batch_{i_k} evaluated at the anchor point
        # gradient of f_{i_k} at x_{k} in the pseudocode of Gower et al. 2020
        minibatch_grad_at_reference_point = self.loss_gradient(reference_point, *args)

        # SVRG gradient estimate
        gk = jax.tree_util.tree_map(
            lambda a, b, c: a - b + c,
            minibatch_grad_at_current_params,
            minibatch_grad_at_reference_point,
            full_grad_at_reference_point,
        )

        # x_{k+1} = x_{k} - stepsize * g_{k}
        next_params = tree_add_scalar_mul(params, -stepsize, gk)

        # apply the proximal operator
        next_params = self.proximal_operator(
            next_params, hyperparams_prox, scaling=stepsize
        )

        return next_params

    @partial(jit, static_argnums=(0,))
    def update(
        self,
        params: Pytree,
        state: SVRGState,
        hyperparams_prox: Union[float, None],
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
        params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (`reference_point`) and the gradient at the anchor point
            (`full_grad_at_reference_point`) already set.
        hyperparams_prox :
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
            reference_point :
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state :
                Updated state.

        Raises
        ------
        ValueError
            The parameter update needs a value for the full gradient at the anchor point, which needs the full data
            to be calculated and is expected to be stored in `state.full_grad_at_reference_point`. So if
            `state.full_grad_at_reference_point` is None, a ValueError is raised.
        """
        if state.full_grad_at_reference_point is None:
            raise ValueError(
                "Full gradient at the anchor point (state.full_grad_at_reference_point) has to be set."
            )
        return self._update_on_batch(params, state, hyperparams_prox, *args)

    @partial(jit, static_argnums=(0,))
    def _update_on_batch(
        self,
        params: Pytree,
        state: SVRGState,
        hyperparams_prox: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Update parameters given a mini-batch of data and increment iteration/epoch number in state.

        Note that this method doesn't update `state.reference_point`, `state.full_grad_at_reference_point`,
        that has to be done outside.

        Parameters
        ----------
        params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (`reference_point`) and the gradient at the anchor point
            (`full_grad_at_reference_point`) already set.
        hyperparams_prox :
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
            reference_point :
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state :
                Updated state.
        """
        next_params = self._inner_loop_param_update_step(
            params,
            state.reference_point,
            state.full_grad_at_reference_point,
            state.stepsize,
            hyperparams_prox,
            *args,
        )

        state = state._replace(
            iter_num=state.iter_num + 1,
        )

        return OptStep(params=next_params, state=state)

    @partial(jit, static_argnums=(0,))
    def run(
        self,
        init_params: Pytree,
        hyperparams_prox: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called by `BaseRegressor._solver_run` (e.g. as called by `GLM.fit`) and assumes
        that X and y are the full data set.

        Parameters
        ----------
        init_params :
            Initial parameters to start from.
        hyperparams_prox :
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
            *args,
        )

        return self._run(init_params, init_state, hyperparams_prox, *args)

    @partial(jit, static_argnums=(0,))
    def _run(
        self,
        init_params: Pytree,
        init_state: SVRGState,
        hyperparams_prox: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called by `BaseRegressor._solver_run` (e.g. as called by `GLM.fit`) and assumes that
        X and y are the full data set.
        Assumes the state has been initialized, which works a bit differently for SVRG and ProxSVRG.

        Parameters
        ----------
        init_params :
            Initial parameters to start from.
        init_state :
            Initialized optimizer state returned by `ProxSVRG.init_state`
        hyperparams_prox :
            Hyperparameters to `prox`, most commonly regularization strength.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
            prev_reference_point, state = step

            # evaluate and store the full gradient with the params from the last inner loop
            state = state._replace(
                full_grad_at_reference_point=self.loss_gradient(
                    prev_reference_point, *args
                )
            )

            # run an update over the whole data
            params, state = self._update_per_random_samples(
                prev_reference_point, state, hyperparams_prox, *args
            )

            # update reference point (x_{s}) with the final parameters (x_{m}) or an average over
            # the inner loop's iterations
            # note that the average is currently not implemented
            reference_point = params

            state = state._replace(
                reference_point=reference_point,
                error=self._error(
                    reference_point, prev_reference_point, state.stepsize
                ),
            )

            return OptStep(params=reference_point, state=state)

        # at the end of each epoch, check for convergence or reaching the max number of epochs
        def cond_fun(step):
            _, state = step
            return (state.iter_num <= self.maxiter) & (state.error >= self.tol)

        # initialize the full gradient at the anchor point
        # the anchor point is init_params at first
        init_state = init_state._replace(
            full_grad_at_reference_point=self.loss_gradient(init_params, *args)
        )

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
        params: Pytree,
        state: SVRGState,
        hyperparams_prox: Union[float, None],
        *args,
    ) -> OptStep:
        """
        Performs the inner loop of Prox-SVRG sweeping through approximately one full epoch,
        updating the parameters after sampling a mini-batch on each iteration.

        Parameters
        ----------
        params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous sweep.
            Needs to have the current anchor point (`reference_point`) and the gradient at the anchor point
            (`full_grad_at_reference_point`) already set.
        hyperparams_prox :
            Hyperparameters to `prox`, most commonly regularization strength. Can be None.
        args :
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
            next_params :
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

        def inner_loop_body(_, carry):
            params, key = carry

            # sample mini-batch or data point
            key, subkey = random.split(key)
            ind = random.randint(subkey, (self.batch_size,), 0, N)

            # perform a single update on the mini-batch or data point
            next_params = self._inner_loop_param_update_step(
                params,
                state.reference_point,
                state.full_grad_at_reference_point,
                state.stepsize,
                hyperparams_prox,
                *tree_slice(args, ind),
            )

            return (next_params, key)

        next_params, key = lax.fori_loop(
            0,
            m,
            inner_loop_body,
            (params, state.key),
        )

        # update the state
        # storing the average over the inner loop to potentially use it in the run loop
        state = state._replace(
            iter_num=state.iter_num + 1,
            key=key,
        )

        return OptStep(params=next_params, state=state)

    @staticmethod
    def _error(x, x_prev, stepsize):
        """
        Calculate the magnitude of the update relative to the stepsize.
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
        return tree_l2_norm(tree_sub(x, x_prev)) / stepsize


class SVRG(ProxSVRG):
    """
    SVRG solver.

    This solver implements "Algorithm 3" of [1]. Equivalent to ProxSVRG with prox as the identity
    function and hyperparams_prox=None.

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
    >>> import numpy as np
    >>> loss_fn = lambda params, X, y: ((X.dot(params) - y)**2).sum()
    >>> svrg = SVRG(loss_fn)
    >>> params, state = svrg.run(np.zeros(2), np.ones((10, 2)), np.zeros(10))

    References
    ----------
    [1] [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik.
    "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).
    ](https://arxiv.org/abs/2010.00892)

    [2] [Xiao, Lin, and Tong Zhang. "A proximal stochastic gradient method with progressive variance reduction."
    SIAM Journal on Optimization 24.4 (2014): 2057-2075.](https://arxiv.org/abs/1403.4699v1)

    [3] [Johnson, Rie, and Tong Zhang. "Accelerating stochastic gradient descent using predictive variance reduction."
    Advances in neural information processing systems 26 (2013).
    ](https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
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
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
        state :
            Initialized optimizer state
        """
        return super().init_state(init_params, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update(self, params: Pytree, state: SVRGState, *args, **kwargs) -> OptStep:
        """
        Perform a single parameter update on the passed data (no random sampling or loops)
        and increment `state.iter_num`.

        Please note that this gets called by `BaseRegressor._solver_update` (e.g., as called by `GLM.update`),
        but repeated calls to `(Prox)SVRG.update` (so in turn e.g. to `GLM.update`) on mini-batches passed to it
        will not result in running the full (Prox-)SVRG, and parts of the algorithm will have to be implemented outside.

        Parameters
        ----------
        params :
            Parameters at the end of the previous update, used as the starting point for the current update.
        state :
            Optimizer state at the end of the previous update.
            Needs to have the current anchor point (`reference_point`) and the gradient at the anchor point
            (`full_grad_at_reference_point`) already set.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
            reference_point :
                Parameters after taking one step defined in the inner loop of Prox-SVRG.
            state :
                Updated state.

        Raises
        ------
        ValueError
            The parameter update needs a value for the full gradient at the anchor point, which needs the full data
            to be calculated and is expected to be stored in `state.full_grad_at_reference_point`.
            So if `state.full_grad_at_reference_point` is None, a ValueError is raised.
        """
        # substitute None for hyperparams_prox
        return super().update(params, state, None, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def run(
        self,
        init_params: Pytree,
        *args,
    ) -> OptStep:
        """
        Run a whole optimization until convergence or until `maxiter` epochs are reached.
        Called by `BaseRegressor._solver_run` (e.g. as called by `GLM.fit`) and assumes that
        X and y are the full data set.

        Parameters
        ----------
        init_params :
            Initial parameters to start from.
        args:
            Positional arguments passed to loss function `fun` and its gradient (e.g. `fun(params, *args)`),
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
        # don't have to pass hyperparams_prox here
        init_state = self.init_state(init_params, *args)

        # substitute None for hyperparams_prox
        return self._run(init_params, init_state, None, *args)
