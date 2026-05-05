"""Newton-based optimization solvers.

This module provides second-order optimization routines based on Newton's method,
with optional line search and pluggable linear solvers via ``lineax``. The
implementations operate on arbitrary parameter pytrees and are optionally jit-able.
"""

from functools import wraps
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import equinox as eqx
import lineax
import optax
from ._abstract_solver import AbstractSolver, OptimizationInfo
from ..typing import Params, StepResult


class NewtonState(eqx.Module):
    """State of the Newton optimization process.

    Attributes
    ----------
    iter_num :
        Number of Newton iterations performed.
    converged :
        Whether the convergence criterion was satisfied.
    grad_norm :
        L2 norm of the gradient at the final iterate.
    params :
        Current parameter values (flattened array, for incremental updates).
    ls_state :
        Optax line search state (for incremental updates).
    """

    iter_num: int
    converged: bool
    grad_norm: float
    params: Optional[jnp.ndarray] = None
    ls_state: Optional[Any] = None


class Newton(AbstractSolver[NewtonState]):
    """Newton optimizer operating on arbitrary parameter pytrees.

    This class implements a full Newton optimization loop, including gradient
    and Hessian evaluation, linear system solves, and optional line search.
    The linear solver is configured per-problem via the ``setup_hessian`` method,
    allowing different Hessian solution strategies without creating subclasses.

    This class also implements the NeMoS ``AbstractSolver`` interface, enabling
    integration with the full NeMoS framework for model fitting.

    Parameters
    ----------
    func :
        Scalar-valued objective function with signature ``(params, *args)``.
    maxiter :
        Maximum number of Newton iterations.
    tol :
        Convergence tolerance on the gradient L2 norm.
    jit :
        Whether to JIT-compile the optimization loop.
    force_autodiff_hessian :
        If ``True``, always use automatic differentiation to compute the Hessian,
        even if one is configured via ``setup_hessian``. Default is ``False``,
        which uses the configured Hessian function if available, falling back
        to autodiff if not.

    Notes
    -----
    The optimization is performed in flattened parameter space using
    :func:`jax.flatten_util.ravel_pytree`, but inputs and outputs remain
    structured as pytrees.

    The Hessian and linear solver strategy should be configured via
    ``setup_hessian()`` before calling ``run()``.
    """

    def __init__(
        self,
        func: Callable,
        maxiter: int = 30,
        tol: float = 1e-6,
        jit: bool = True,
        force_autodiff_hessian: bool = False,
    ):
        self.func = func
        self.maxiter = maxiter
        self.tol = tol
        self.jit = jit
        self.force_autodiff_hessian = force_autodiff_hessian

        # Cache val_and_grad to avoid recomputation each step
        self._val_and_grad = jax.value_and_grad(func)

        # Internal line search (backtracking by default)
        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        # Hessian configuration — initialized to None, set via setup_hessian()
        self._hess_fn = None
        self._hess_tag = None
        self._linear_solver = None

    def setup_hessian(
        self,
        hess_fn: Optional[Callable] = None,
        hess_tag: Optional[lineax.AbstractTag] = None,
    ) -> None:
        """Configure the Hessian computation and linear solver strategy.

        Parameters
        ----------
        hess_fn :
            Optional callable returning the Hessian in flattened parameter space.
            If ``None`` or if ``force_autodiff_hessian`` is ``True``, the Hessian
            is computed automatically using :func:`jax.hessian`.
        hess_tag :
            Optional lineax tag indicating structure of the Hessian (e.g.,
            ``lineax.positive_semidefinite_tag`` for Cholesky, ``None`` for general).
            This influences the choice of linear solver.

        Notes
        -----
        If ``hess_tag`` is ``None``, an LU-based solver is used for general matrices.
        If ``hess_tag`` indicates positive-semidefiniteness, a Cholesky solver is used.
        """
        self._hess_fn = hess_fn
        self._hess_tag = hess_tag

        # Select linear solver based on Hessian structure tag
        if hess_tag is lineax.positive_semidefinite_tag:
            self._linear_solver = lineax.Cholesky()
        else:
            # Default to LU for general or unspecified Hessians
            self._linear_solver = lineax.LU()

    def _solve_linear_system(self, H: jnp.ndarray, g_flat: jnp.ndarray) -> jnp.ndarray:
        """Solve ``H @ step = -g`` using the configured linear solver.

        The Hessian may be wrapped with a tag before factorization to indicate
        structural properties (e.g., symmetry or positive-definiteness).
        """
        if self._linear_solver is None:
            raise RuntimeError(
                "Linear solver not configured. Call setup_hessian() before run()."
            )

        # Wrap the Hessian operator with the configured tag
        if self._hess_tag is not None:
            operator = lineax.MatrixLinearOperator(H, self._hess_tag)
        else:
            operator = lineax.MatrixLinearOperator(H)

        solution = lineax.linear_solve(operator, -g_flat, solver=self._linear_solver)
        return solution.value

    def run(self, init_params: Params, *args: Any) -> StepResult:
        """Run the Newton optimization loop.

        Parameters
        ----------
        init_params :
            Initial parameter values (arbitrary pytree).
        *args :
            Additional arguments passed to the objective function.

        Returns
        -------
        tuple[Any, NewtonState]
            Optimized parameters and final solver state.

        Raises
        ------
        RuntimeError
            If ``setup_hessian()`` has not been called.
        """
        if self._linear_solver is None:
            raise RuntimeError(
                "Linear solver not configured. Call setup_hessian() before run()."
            )

        maxiter = self.maxiter

        # Initialize state
        state = self.init_state(init_params, *args)
        params = init_params

        # Run the optimization loop (optionally JIT-compiled)
        if self.jit:
            # JIT-compiled loop using eqx.internal.while_loop
            def cond(carry):
                state, _ = carry
                return (~state.converged) & (state.iter_num < maxiter)

            def body(carry):
                _, params = carry
                params, state, _ = self.update(params, state, *args)
                return state, params

            init_carry = (state, params)
            final_state, final_params = eqx.internal.while_loop(
                cond, body, init_carry, kind=None
            )
        else:
            # Pure-Python loop (eager execution)
            for _ in range(maxiter):
                params, state, _ = self.update(params, state, *args)
                if state.converged:
                    break

            final_params = params
            final_state = state

        return final_params, final_state, None

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        """Return the set of keyword arguments accepted by the solver.

        Returns
        -------
        set[str]
            Keywords that can be passed to configure the solver.
        """
        return {"hess", "maxiter", "tol", "jit", "force_autodiff_hessian"}

    def init_state(self, init_params: Any, *args: Any) -> NewtonState:
        """Initialize solver state for the given parameters.

        This prepares the solver state for incremental updates via ``update()``.
        The returned state can be passed to ``update()`` along with the initial
        parameters to perform the first Newton step.

        Parameters
        ----------
        init_params :
            Initial parameters (pytree).
        *args :
            Additional arguments passed to the objective function (unused).

        Returns
        -------
        NewtonState
            Initialized state with flattened parameters and line search state.
        """
        if self._linear_solver is None:
            raise RuntimeError(
                "Linear solver not configured. Call setup_hessian() before init_state()."
            )

        params_flat, _ = ravel_pytree(init_params)
        ls_init_state = self._line_search.init(params_flat)

        return NewtonState(
            iter_num=0,
            converged=False,
            grad_norm=float("inf"),
            params=params_flat,
            ls_state=ls_init_state,
        )

    def update(self, params: Params, state: NewtonState, *args: Any) -> StepResult:
        """Perform a single Newton iteration (incremental update).

        Continues the optimization from the state returned by the previous
        ``run()`` or ``update()`` call, performing one additional Newton step.

        Parameters
        ----------
        params :
            Current parameters (ignored; state contains flattened version).
        state :
            The `NewtonState` from the previous ``run()`` or ``update()``.
        *args :
            Additional arguments passed to the objective function.

        Returns
        -------
        tuple[Any, NewtonState]
            Updated parameters and state after one Newton iteration.

        Raises
        ------
        RuntimeError
            If ``setup_hessian()`` has not been called or if state was not
            initialized with ``init_state()`` or a prior ``run()``/``update()``.
        """
        if self._linear_solver is None:
            raise RuntimeError(
                "Linear solver not configured. Call setup_hessian() before update()."
            )

        if state.params is None or state.ls_state is None:
            raise RuntimeError(
                "State not properly initialized. Use init_state() or run() first."
            )

        func = self.func
        tol = self.tol

        # Unravel the stored flat parameters
        p_flat = state.params
        _, unravel = ravel_pytree(params)  # Use the pytree structure from input

        # value_fn in flat space — needed by optax linesearches
        def value_fn_flat(p_flat):
            return func(unravel(p_flat), *args)

        # Hessian in flat space
        hess_fn = self._hess_fn
        force_autodiff = self.force_autodiff_hessian

        def hessian_flat(p_flat):
            if not force_autodiff and hess_fn is not None:
                return hess_fn(unravel(p_flat), *args)
            return jax.hessian(value_fn_flat)(p_flat)

        # Single Newton step — use cached val_and_grad
        params_tree = unravel(p_flat)
        f0, g_tree = self._val_and_grad(params_tree, *args)
        g_flat, _ = ravel_pytree(g_tree)
        H = hessian_flat(p_flat)
        step = self._solve_linear_system(H, g_flat)

        # Apply line search
        scaled_step, new_ls_state = self._line_search.update(
            step,
            state.ls_state,
            p_flat,
            value=f0,
            grad=g_flat,
            value_fn=value_fn_flat,
        )
        new_p_flat = p_flat + scaled_step

        # Compute gradient norm for convergence check
        gnorm = jnp.linalg.norm(g_flat)
        converged = gnorm < tol

        new_state = NewtonState(
            iter_num=state.iter_num + 1,
            converged=bool(converged),
            grad_norm=float(gnorm),
            params=new_p_flat,
            ls_state=new_ls_state,
        )

        return unravel(new_p_flat), new_state, None

    def _get_optim_info(self, state: NewtonState, **kwargs: Any) -> OptimizationInfo:
        """Extract optimization information from the Newton solver state.

        Parameters
        ----------
        state :
            The NewtonState returned by the solver.
        **kwargs :
            Unused keyword arguments.

        Returns
        -------
        OptimizationInfo
            Summary of the optimization run.
        """
        return OptimizationInfo(
            function_val=None,
            num_steps=state.iter_num,
            converged=jnp.array(state.converged),
            reached_max_steps=jnp.array(state.iter_num >= self.maxiter),
        )


# ---------------------------------------------------------------------------
# GLM-specific analytic Hessian factory
# ---------------------------------------------------------------------------


def _elementwise_derivative(f: Callable) -> Callable:
    """Construct the element-wise derivative of a function using forward-mode AD.

    Parameters
    ----------
    f :
        A function acting element-wise on an array.

    Returns
    -------
    Callable
        A function that computes the derivative of ``f`` evaluated element-wise.
    """

    @wraps(f)
    def df(x):
        _, grad = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return grad

    return df


def _var_func_of_mu(model) -> Callable:
    """Return the variance function V(mu) for a GLM observation model.

    Parameters
    ----------
    model :
        A GLM instance with an ``observation_model`` attribute.

    Returns
    -------
    Callable
        A function mapping the mean ``mu`` to the variance ``V(mu)``.

    Raises
    ------
    NotImplementedError
        If the observation model is not recognized.
    """
    obs_name = model.observation_model.__class__.__name__
    var_funcs = {
        "PoissonObservations": lambda mu: mu,
        "GammaObservations": lambda mu: mu**2,
        "GaussianObservations": lambda mu: jnp.ones_like(mu),
        "BernoulliObservations": lambda mu: mu * (1.0 - mu),
    }
    if obs_name not in var_funcs:
        raise NotImplementedError(f"No variance function defined for {obs_name!r}")
    return var_funcs[obs_name]


def define_hess(model, regularizer_strength: float = 0.0) -> Callable:
    """Construct the analytic Fisher-scoring Hessian for a GLM.

    This function returns a callable that computes the Hessian of the penalized
    mean log-likelihood for a generalized linear model using the Fisher scoring
    approximation. This avoids the computational cost of automatic
    differentiation-based Hessian evaluation.

    Parameters
    ----------
    model :
        A GLM instance providing an inverse link function and observation model.
    regularizer_strength :
        Strength of L2 regularization applied to the model coefficients. The
        intercept term is not regularized.

    Returns
    -------
    Callable
        A function with signature ``(params, X, *args) -> (d, d)`` returning the
        Hessian matrix in flattened parameter space, where ``d`` is the number of
        parameters (including intercept).

    Notes
    -----
    The returned Hessian corresponds to the Fisher information matrix scaled by
    the number of samples, consistent with a mean loss formulation.
    """
    gprime = _elementwise_derivative(model.inverse_link_function)
    var_of_mu = _var_func_of_mu(model)
    lam = regularizer_strength

    def hess(params, *args):
        X = args[0]
        n, p = X.shape
        eta = X @ params.coef + params.intercept
        mu = model.inverse_link_function(eta)
        # Fisher weights: (g'(eta))^2 / V(mu) / n  — 1/n matches the mean loss.
        w = gprime(eta) ** 2 / var_of_mu(mu) / n
        X_aug = jnp.concatenate([X, jnp.ones((n, 1))], axis=1)
        H = X_aug.T @ (w[:, None] * X_aug)
        # L2 regularisation on coefficients only.
        if lam > 0.0:
            H = H.at[:p, :p].add(lam * jnp.eye(p))
        return H

    return hess
