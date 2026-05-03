"""Newton-based optimization solvers.

This module provides second-order optimization routines based on Newton's method,
with optional line search and pluggable linear solvers via ``lineax``. The
implementations operate on arbitrary parameter pytrees and are optionally jit-able.
"""

from functools import wraps
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import lineax
import optax
from ._abstract_solver import AbstractSolver, OptimizationInfo, SolverAdapterState


class NewtonState(NamedTuple):
    """State of the Newton optimization process.

    Attributes
    ----------
    iter_num :
        Number of Newton iterations performed.
    converged :
        Whether the convergence criterion was satisfied.
    grad_norm :
        L2 norm of the gradient at the final iterate.
    """

    iter_num: int
    converged: bool
    grad_norm: float


class BaseNewton:
    """Generic Newton optimizer operating on arbitrary parameter pytrees.

    This class implements a full Newton optimization loop, including gradient
    and Hessian evaluation, linear system solves, and optional line search. It
    serves as a reusable base for specific Newton variants that differ only in
    how the linear system is solved.

    Subclasses specify a ``linear_solver`` (from ``lineax``) and may override
    :meth:`_solve_linear_system` to enforce additional structure such as symmetry
    or positive definiteness.

    Parameters
    ----------
    func :
        Scalar-valued objective function with signature ``(params, *args)``.
    hess :
        Optional callable returning the Hessian in flattened parameter space.
        If ``None``, the Hessian is computed automatically using
        :func:`jax.hessian`.
    maxiter :
        Maximum number of Newton iterations.
    tol :
        Convergence tolerance on the gradient L2 norm.
    line_search :
        Optax-compatible line search transformation. If ``None``, a default
        backtracking line search is used. Passing ``optax.identity()``
        disables line search.
    jit :
        Whether to JIT-compile the optimization loop.

    Notes
    -----
    The optimization is performed in flattened parameter space using
    :func:`jax.flatten_util.ravel_pytree`, but inputs and outputs remain
    structured as pytrees.
    """

    # Subclasses override this.
    linear_solver: lineax.AbstractLinearSolver = None

    def __init__(
        self,
        func: Callable,
        hess: Optional[Callable] = None,
        maxiter: int = 30,
        tol: float = 1e-6,
        line_search: optax.GradientTransformation | None = None,
        jit: bool = True,
    ):
        self.func = func
        self.hess = hess
        self.maxiter = maxiter
        self.tol = tol
        self.line_search = (
            line_search
            if line_search is not None
            else optax.scale_by_backtracking_linesearch(max_backtracking_steps=30)
        )
        self.jit = jit

    def _solve_linear_system(self, H: jnp.ndarray, g_flat: jnp.ndarray) -> jnp.ndarray:
        """Solve ``H @ step = -g`` using :attr:`linear_solver`.

        The base implementation wraps the Hessian in a
        :class:`lineax.MatrixLinearOperator` and delegates to
        :attr:`linear_solver`.  Subclasses may override this to enforce
        structural properties (e.g. symmetry) before factorisation.
        """
        operator = lineax.MatrixLinearOperator(H)
        solution = lineax.linear_solve(operator, -g_flat, solver=self.linear_solver)
        return solution.value

    def run(self, init_params, *args) -> tuple[Any, NewtonState]:
        func = self.func
        val_and_grad = jax.value_and_grad(func)
        tol = self.tol
        maxiter = self.maxiter
        line_search = self.line_search

        params_flat, unravel = ravel_pytree(init_params)

        # value_fn in flat space — needed by optax linesearches
        def value_fn_flat(p_flat):
            return func(unravel(p_flat), *args)

        # Hessian in flat space
        hess_fn = self.hess

        def hessian_flat(p_flat):
            if hess_fn is not None:
                return hess_fn(unravel(p_flat), *args)
            return jax.hessian(value_fn_flat)(p_flat)

        # Single Newton step: returns (descent_direction, gradient, f0)
        def newton_step(p_flat):
            params = unravel(p_flat)
            f0, g_tree = val_and_grad(params, *args)
            g_flat, _ = ravel_pytree(g_tree)
            H = hessian_flat(p_flat)
            step = self._solve_linear_system(H, g_flat)
            return step, g_flat, f0

        # Optax linesearch state initialisation
        # init() requires a sample params pytree; we use the flat array directly
        # since our entire loop operates in flat space.
        ls_init_state = line_search.init(params_flat)

        # While-loop body and condition
        def body(carry):
            p_flat, ls_state, i, _converged, _gnorm = carry
            step, g_flat, f0 = newton_step(p_flat)

            # optax convention: pass *updates* (the raw descent direction)
            # and let the linesearch scale them.  Extra kwargs carry the
            # information the linesearch needs to evaluate the Armijo / Wolfe
            # conditions without a redundant forward pass.
            # When line_search=optax.identity(), scales by 1.
            scaled_step, new_ls_state = line_search.update(
                step,
                ls_state,
                p_flat,
                value=f0,
                grad=g_flat,
                value_fn=value_fn_flat,
            )
            new_flat = p_flat + scaled_step

            gnorm = jnp.linalg.norm(g_flat)
            return new_flat, new_ls_state, i + 1, gnorm < tol, gnorm

        def cond(carry):
            _, _ls, i, converged, _ = carry
            return (~converged) & (i < maxiter)

        # Initialise carry
        _, g0_tree = val_and_grad(init_params, *args)
        g0_flat, _ = ravel_pytree(g0_tree)
        init_carry = (
            params_flat,
            ls_init_state,
            jnp.array(0),
            jnp.array(False),
            jnp.linalg.norm(g0_flat),
        )

        # ------------------------------------------------------------------
        # Run loop (optionally JIT-compiled)
        # ------------------------------------------------------------------
        while_loop = jax.lax.while_loop
        if not self.jit:
            # Pure-Python fallback: mimics while_loop semantics but stays eager.
            def while_loop(cond_fn, body_fn, init_val):  # noqa: F811
                val = init_val
                while cond_fn(val):
                    val = body_fn(val)
                return val

        final_flat, _, n_iter, converged, grad_norm = while_loop(cond, body, init_carry)

        state = NewtonState(
            iter_num=int(n_iter),
            converged=bool(converged),
            grad_norm=float(grad_norm),
        )
        return unravel(final_flat), state


class _NewtonCholesky(BaseNewton):
    """Newton optimizer using a Cholesky factorization for the linear subproblem.

    This implementation assumes that the Hessian is symmetric positive-definite,
    which is typically satisfied for convex objectives such as generalized
    linear models. Under this assumption, Cholesky decomposition provides an
    efficient and numerically stable solution to the Newton system.

    The Hessian is symmetrized prior to factorization to mitigate numerical
    asymmetries.

    Notes
    -----
    This class is intended as a low-level numerical backend and does not implement
    any solver interface directly.
    """

    linear_solver: lineax.AbstractLinearSolver = lineax.Cholesky()

    def _solve_linear_system(self, H: jnp.ndarray, g_flat: jnp.ndarray) -> jnp.ndarray:
        # Symmetrise to suppress floating-point asymmetry before factorisation.
        H_sym = (H + H.T) / 2.0
        operator = lineax.MatrixLinearOperator(H_sym, lineax.positive_semidefinite_tag)
        solution = lineax.linear_solve(operator, -g_flat, solver=self.linear_solver)
        return solution.value


class _NewtonLU(BaseNewton):
    """Newton optimizer using LU decomposition for the linear subproblem.

    This variant is suitable for problems where the Hessian may be indefinite or
    non-symmetric, such as non-convex objectives or regions near saddle points.

    Compared to Cholesky-based methods, LU decomposition is more general but may
    be less efficient for well-conditioned positive-definite systems.
    """

    linear_solver: lineax.AbstractLinearSolver = lineax.LU()


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


# ---------------------------------------------------------------------------
# NeMoS adapters
# ---------------------------------------------------------------------------


class NewtonCholesky(AbstractSolver):
    """Adapter that wires :class:`NewtonCholesky` into the NeMoS solver protocol.

    Parameters
    ----------
    unregularized_loss :
        Base loss function before regularization.
    regularizer :
        Object providing a ``penalized_loss`` method.
    regularizer_strength :
        Strength of the regularization penalty.
    has_aux :
        Whether the loss returns auxiliary outputs. Currently unsupported.
    init_params :
        Initial parameter values, used to construct the penalized loss.
    hess :
        Optional analytic Hessian function. If not provided, the Hessian is
        computed using automatic differentiation.
    maxiter :
        Maximum number of Newton iterations.
    tol :
        Convergence tolerance on the gradient norm.
    line_search :
        Optax-compatible line search transformation. If ``None``, a default
        backtracking line search is used.
    jit :
        Whether to JIT-compile the optimization loop.

    Notes
    -----
    This solver performs full-batch optimization via :meth:`run`. Incremental
    updates via :meth:`update` are not supported.
    """

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer,
        regularizer_strength: float,
        has_aux: bool,
        init_params,
        hess: Optional[Callable] = None,
        maxiter: int = 30,
        tol: float = 1e-6,
        line_search: optax.GradientTransformation | None = None,
        jit: bool = True,
        mode: str = "analytic",
    ):
        if has_aux:
            raise ValueError("Auxiliary output from the loss is not supported.")

        penalized = regularizer.penalized_loss(
            unregularized_loss, init_params, regularizer_strength
        )
        self.penalized_loss = penalized
        self._solver = _NewtonCholesky(
            func=penalized,
            hess=hess,
            maxiter=maxiter,
            tol=tol,
            line_search=line_search,
            jit=jit,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"hess", "maxiter", "tol", "line_search", "jit"}

    def init_state(self, init_params, *args) -> NewtonState:
        return NewtonState(iter_num=0, converged=False, grad_norm=float("inf"))

    def run(self, init_params, *args) -> tuple[Any, NewtonState, None]:
        params, state = self._solver.run(init_params, *args)
        return params, state, None

    def update(self, params, state, *args):
        raise NotImplementedError(
            "NewtonCholeskySolver does not support incremental updates; use run()."
        )

    def _get_optim_info(self, state: NewtonState, **kwargs) -> OptimizationInfo:
        return OptimizationInfo(
            function_val=None,
            num_steps=state.iter_num,
            converged=jnp.array(state.converged),
            reached_max_steps=jnp.array(state.iter_num >= self.maxiter),
        )
