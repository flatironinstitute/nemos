"""Newton-Cholesky solver sketch for GLMs.

Standalone class — not yet wired to the NeMoS solver protocol.

Interface
---------
func : (params, *args) -> scalar    penalized loss (gradient will be computed separately)
hess : (params, *args) -> (d, d)    full Hessian of func in flat-param space;
                                    defaults to jax.hessian(func)

The caller owns all loss-specific structure (1/n scaling, X_aug, regularization).
The solver is fully generic.


Notes
-----
## Linear Solver

For a general newton step we need to deal with the case of a non-positive (semi-)definite
hessian matrix. That is not happening for a GLM, but it may be the case for non-convex problems.
For example if the algorithm is moving towards a saddle
point instead of a local minimum.

The Cholevsky solve implemented here requires the hessian to be positive (or negative) semi-definite,
 this makes the solve more efficient.

I think we need to implement two solvers, that are basically the same but have a different `linear_solver` attribute.
Each solver we can deal with different linear operator (hessian matrix) structures:

- A Newton-Cholesky solver, as in `scikit-learn` for convex problems:
    - Use Lineax Cholesky (already a dependency): https://docs.kidger.site/lineax/api/solvers/#lineax.Cholesky

- A Newton general solver:
    - Use Lineax LU or SVD (for ill-posed or indefinite problems);


According to the structure, we could pick one of the linear solver options in Lineax (which is already a
dependency via optimistix). If you look into the docs below you can already see a decision tree on which solvers are
available:

- https://docs.kidger.site/lineax/api/solvers/#lineax.AutoLinearSolver
  This auto-selects the solver based on the structure of the matrix. This auto selection have to inspect
  the matrix, so it may be not what we want, we should know the problem structure on a model by model basis.


For this PR, we should focus on newton-cholesky only but we should predispose it so that the linear solve is an
attribute, so it is trivial to extend. What I would do:

class BaseNewton:
    linear_solver: lineax.AbstractLinearSolver = None

    ... implementation of the newton algorithm


class NewtonCholesky(BaseNewton):
    linear_solver: lineax.Cholesky()


# for the future
class Newton(BaseNewton):
    def __init__(self, is_singular: bool):
        if is_singular:
            self.linear_solver = lineax.LU()
        else:
            self.linear_solver = lineax.SVD()


## Line search

Line searches are steps that are triggered after a parameter updated and should prevent overfitting and guarantee
convergence when the solver is near the optimum. Here I implemented directly the Amijo line search but it is readily
available in optax and optimistix:

- https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_backtracking_linesearch
- https://docs.kidger.site/optimistix/api/searches/searches/#optimistix.BacktrackingArmijo


In my experience, optax is straight-forward to implement, while optimistix requires you to reuse some of their
machinery, it will be clear if you look at the signature of the two implementations.

If you choose to go with optax, you can use the `chain` mechanism to chain together the newton step followed by
the line search:

- https://optax.readthedocs.io/en/latest/api/combining_optimizers.html

Overall, we do not need any of the two, all we need is to have a solver that compiles and execute fast. Using optax,
or optimistix would maybe make the code cleaner but I don't care as much.

## While loop

The run method runs a while loop; the choice of the while loop makes the solver more or less flexible:

- jit compiled or not
- differentiable (forward or backward)
- unrolled vs non-unrolled loop

I like how the old jaxopt did things, they had a single entry point while loop with some configurations which
gives flexibility, then one can pass parameters to the solver to define the loop
(`jit=True/False, unrolled=True/False` etc.):

- https://github.com/google/jaxopt/blob/main/jaxopt/_src/loop.py

The jit compilation is interesting because we may not need the compilation for small problems, and exposing the
parameter will allow that.
"""

from functools import wraps
from typing import Callable, NamedTuple, Optional

import jax
import jax.scipy.linalg as jsl
from jax.flatten_util import ravel_pytree

from nemos.glm import GLM


class NewtonCholState(NamedTuple):
    iter_num: int
    converged: bool
    grad_norm: float


class NewtonCholesky:
    def __init__(
        self,
        func: Callable,
        grad: Optional[Callable] = None,
        hess: Optional[Callable] = None,
        maxiter: int = 30,
        tol: float = 1e-6,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
    ):
        self.func = func
        self.hess = hess
        self.maxiter = maxiter
        self.tol = tol
        self.c = armijo_c
        self.rho = armijo_rho

    def run(self, init_params, *args):
        func = self.func
        # important trick
        # (more efficient than calling the func and
        # the gradient separately, since it re-uses the forward pass and the tape
        # to compute the gradient).
        val_and_grad = jax.value_and_grad(func)
        hess_fn = self.hess
        c, rho = self.c, self.rho
        tol = self.tol
        maxiter = self.maxiter

        params_flat, unravel = ravel_pytree(init_params)

        def hessian(params_flat):
            if hess_fn is not None:
                return hess_fn(unravel(params_flat), *args)
            return jax.hessian(lambda p: func(unravel(p), *args))(params_flat)

        def newton_step(params_flat):
            params = unravel(params_flat)
            f0, g_tree = val_and_grad(params, *args)
            g_flat, _ = ravel_pytree(g_tree)
            H = hessian(params_flat)
            step_flat = -jsl.cho_solve(jsl.cho_factor(H), g_flat)
            return step_flat, g_flat, f0

        def armijo(params_flat, step_flat, f0, slope):
            def cond(alpha):
                return (
                    func(unravel(params_flat + alpha * step_flat), *args)
                    > f0 + c * alpha * slope
                )

            return jax.lax.while_loop(cond, lambda a: a * rho, jnp.array(1.0))

        def body(carry):
            params_flat, i, _converged, _gnorm = carry
            step_flat, g_flat, f0 = newton_step(params_flat)
            slope = g_flat @ step_flat
            alpha = armijo(params_flat, step_flat, f0, slope)
            new_flat = params_flat + alpha * step_flat
            grad_norm = jnp.linalg.norm(g_flat)
            return new_flat, i + 1, grad_norm < tol, grad_norm

        def cond(carry):
            _, i, converged, _ = carry
            return ~converged & (i < maxiter)

        _, g0_tree = val_and_grad(init_params, *args)
        g0_flat, _ = ravel_pytree(g0_tree)
        init_carry = (
            params_flat,
            jnp.array(0),
            jnp.array(False),
            jnp.linalg.norm(g0_flat),
        )

        final_flat, n_iter, converged, grad_norm = jax.lax.while_loop(
            cond, body, init_carry
        )
        return unravel(final_flat), NewtonCholState(n_iter, converged, grad_norm)


# ---------------------------------------------------------------------------
# GLM-specific Hessian factory
# ---------------------------------------------------------------------------


def elementwise_derivative(f):
    @wraps(f)
    def df(x):
        _, grad = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return grad

    return df


def _var_func_of_mu(model):
    obs_name = model.observation_model.__class__.__name__
    var_funcs = {
        "PoissonObservations": lambda mu: mu,
        "GammaObservations": lambda mu: mu**2,
        "GaussianObservations": lambda mu: jnp.ones_like(mu),
        "BernoulliObservations": lambda mu: mu * (1 - mu),
    }
    if obs_name not in var_funcs:
        raise NotImplementedError(f"No variance function defined for {obs_name}")
    return var_funcs[obs_name]


def define_hess(model: GLM, regularizer_strength: float = 0.0) -> Callable:
    """Return the full Fisher-scoring Hessian of the penalized mean log-likelihood.

    Returned callable: (params, *args) -> (d, d) matrix, d = n_features + 1.
    Caller passes this as `hess` to NewtonCholesky.
    """
    gprime = elementwise_derivative(model.inverse_link_function)
    var_of_mu = _var_func_of_mu(model)
    lam = regularizer_strength

    def hess(params, *args):
        X = args[0]
        n, p = X.shape
        eta = X @ params.coef + params.intercept
        mu = model.inverse_link_function(eta)
        w = gprime(eta) ** 2 / var_of_mu(mu) / n  # (n,) — 1/n matches mean loss
        X_aug = jnp.concatenate([X, jnp.ones((n, 1))], axis=1)
        H = X_aug.T @ (w[:, None] * X_aug)
        # add ridge on coef only (intercept is not regularized)
        H = H.at[:p, :p].add(lam * jnp.eye(p))
        return H

    return hess


# ---------------------------------------------------------------------------
# NeMoS solver protocol adapter
# ---------------------------------------------------------------------------


class NewtonCholeskySolver:
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params,
        hess: Optional[Callable] = None,
        maxiter: int = 30,
        tol: float = 1e-6,
    ):
        assert not has_aux, "aux output not supported"
        penalized = regularizer.penalized_loss(
            unregularized_loss, init_params, regularizer_strength
        )
        self._solver = NewtonCholesky(
            func=penalized,
            hess=hess,
            maxiter=maxiter,
            tol=tol,
        )

    @classmethod
    def get_accepted_arguments(cls):
        return {"hess", "maxiter", "tol"}

    def init_state(self, init_params, *args):
        return NewtonCholState(0, False, float("inf"))

    def run(self, init_params, *args):
        params, state = self._solver.run(init_params, *args)
        return params, state, None

    def update(self, params, state, *args):
        raise NotImplementedError("use run()")


# ---------------------------------------------------------------------------
# Quick-and-dirty test subclass
# ---------------------------------------------------------------------------


class NewtonGLM(GLM):
    def fit(self, X, y):
        lam = self.regularizer_strength or 0.0
        h = define_hess(self, regularizer_strength=lam)
        self.solver_kwargs = {**(self.solver_kwargs or {}), "hess": h}
        return super().fit(X, y)


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.append("/Users/ebalzani/Code/nemos/scripts/benchmarking/")
    from time import perf_counter

    import jax.numpy as jnp
    import numpy as np
    import pynapple as nap
    from sklearn.linear_model import PoissonRegressor

    import nemos as nmo

    jax.config.update("jax_enable_x64", True)

    nmo.solvers.register("NewtonCholesky", NewtonCholeskySolver, backend="custom")
    nmo.regularizer.Ridge.allow_solver("NewtonCholesky")
    nmo.regularizer.UnRegularized.allow_solver("NewtonCholesky")
    nmo.solvers.set_default_backend("NewtonCholesky", "custom")

    def get_data():
        """Model design."""
        path = nmo.fetch.fetch_data("Mouse32-140822.nwb")
        data = nap.load_file(path)
        spikes = data["units"]
        epochs = data["epochs"]
        wake_ep = epochs[epochs.tags == "wake"]
        spikes = spikes.getby_category("location")["adn"]
        spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)
        y = spikes.count(0.01, ep=wake_ep)
        X = nmo.basis.RaisedCosineLogConv(5, window_size=80).compute_features(y)
        X, y = X.d, y.d
        keep = np.all(~np.isnan(X), axis=1)
        return X[keep], y[keep]

    X, y = get_data()
    skl = PoissonRegressor(alpha=0.01, solver="newton-cholesky", max_iter=1000)
    XX, yy = np.array(X), np.array(y[:, 0])
    t0 = perf_counter()
    skl.fit(XX, yy)
    t1 = perf_counter()
    print("skl:", t1 - t0)
    for label, mdl in [
        (
            "BFGS",
            nmo.glm.GLM(
                regularizer="Ridge", regularizer_strength=0.01, solver_name="LBFGS"
            ),
        ),
        (
            "NewtonCholesky",
            NewtonGLM(
                regularizer="Ridge",
                regularizer_strength=0.01,
                solver_name="NewtonCholesky",
            ),
        ),
    ]:
        y0 = y[:, 0]
        t0 = perf_counter()
        mdl.fit(X, y0)
        t1 = perf_counter()
        state = mdl.solver_state_
        if hasattr(state, "stats"):  # optimistix-based
            converged = bool(state.stats.converged)
            n_iter = int(state.stats.num_steps)
            extra = ""
        else:  # NewtonCholState
            converged = bool(state.converged)
            n_iter = int(state.iter_num)
            extra = f"  grad_norm: {state.grad_norm:.2e}"
        print(
            f"{label:20s}  {t1-t0:.3f}s  converged: {converged}  iters: {n_iter}{extra}"
        )
