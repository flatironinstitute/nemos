import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nemos._hess import Full, HessianTag, PositiveDefinite, PositiveSemiDefinite
from nemos.regularizer import Ridge, UnRegularized
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.solvers._newton import Newton, NewtonState
from nemos.tree_utils import pytree_map_and_reduce

N = 8

_PD_TAG = HessianTag(structure=Full, property=PositiveDefinite)
_PSD_TAG = HessianTag(structure=Full, property=PositiveSemiDefinite)


def _dtype_tol(dtype, scale=1.0, rtol_factor=100.0):
    """Tolerance proportional to machine precision."""
    eps = float(jnp.finfo(dtype).eps)
    return rtol_factor * eps * max(1.0, float(scale))


def _make_pd_problem(n, dtype, rng=None):
    """Return a well-conditioned PD quadratic and its analytic optimum."""
    if rng is None:
        rng = np.random.default_rng(0)

    # QR gives an orthogonal matrix, so we can prescribe the condition number.
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = np.linspace(1.0, 5.0, n)

    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)
    x_star = np.linalg.solve(A, b)

    return (
        jnp.asarray(A, dtype=dtype),
        jnp.asarray(b, dtype=dtype),
        jnp.asarray(x_star, dtype=dtype),
    )


def _make_psd_problem(n, dtype, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)

    eigvals = np.linspace(1.0, 5.0, n - 2)

    A = np.diag(np.concatenate([eigvals, np.zeros(2)]))

    coeffs = rng.standard_normal(n - 2)

    b = np.concatenate([coeffs, np.zeros(2)])

    x_star = np.concatenate(
        [
            coeffs / eigvals,
            np.zeros(2),
        ]
    )

    x0_null = np.zeros(n)
    x0_null[-2:] = [1.5, -0.75]

    x0 = x0_null.copy()

    Q_null = np.eye(n)[:, -2:]

    return (
        jnp.asarray(A, dtype=dtype),
        jnp.asarray(b, dtype=dtype),
        jnp.asarray(x_star, dtype=dtype),
        jnp.asarray(x0, dtype=dtype),
        jnp.asarray(x0_null, dtype=dtype),
        jnp.asarray(Q_null, dtype=dtype),
    )


def _quadratic_loss_and_hessian(A, b):
    """f(x) = 1/2 x'Ax - b'x."""

    def loss(params, *args):
        return _quadratic_loss(A, b)(params, *args)

    def hess(params, *args):
        return A

    return loss, hess


def _quadratic_loss(A, b):
    """Quadratic loss for autodiff-Hessian tests."""

    def loss(params, *args):
        return 0.5 * jnp.dot(params, A @ params) - jnp.dot(b, params)

    return loss


def _make_solver(loss_fn, hess_fn, hess_tag, init_params, **kwargs):
    """Wire up a Newton solver with sensible defaults for tests."""
    solver = Newton(
        loss_fn,
        regularizer=UnRegularized(),
        regularizer_strength=0.0,
        has_aux=False,
        init_params=init_params,
        jit=False,
        **kwargs,
    )
    solver.setup_hessian(hess_fn=hess_fn, hess_tag=hess_tag)
    return solver


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.requires_x64
def test_pd_quadratic_convergence(dtype):
    """Newton-Cholesky solves a PD quadratic exactly in one Newton step."""
    A, b, x_star = _make_pd_problem(6, dtype)

    loss, hess = _quadratic_loss_and_hessian(A, b)
    x0 = jnp.zeros_like(x_star)

    tol = _dtype_tol(dtype, np.linalg.norm(np.asarray(x_star)))

    solver = _make_solver(
        loss,
        hess,
        _PD_TAG,
        x0,
        tol=tol,
        shift_const=0.0,
    )

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps == 2

    np.testing.assert_allclose(
        x_opt,
        x_star,
        atol=tol,
        rtol=0.0,
    )

    residual = A @ x_opt - b
    np.testing.assert_allclose(
        residual,
        0.0,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.requires_x64
def test_run_converges_on_pd_quadratic():
    """Solver must converge to the exact minimiser of a strongly convex quadratic."""
    A, b, x_star = _make_pd_problem(N, np.float64)
    loss, hess = _quadratic_loss_and_hessian(A, b)
    x0 = jnp.zeros(N)
    solver = _make_solver(loss, hess, _PD_TAG, x0)

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged), "Solver did not converge on a PD quadratic."
    assert (
        state.stats.num_steps == 2
    ), "Solver did not converge in 2 step on a PD quadratic."
    np.testing.assert_allclose(
        x_opt,
        x_star,
        atol=1e-6,
        err_msg="Solver solution does not match analytic solution.",
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.requires_x64
def test_pd_quadratic_convergence_autodiff(dtype):
    """Newton-Cholesky solves a PD quadratic using an autodiff Hessian."""
    A, b, x_star = _make_pd_problem(6, dtype)

    loss = _quadratic_loss(A, b)
    x0 = jnp.zeros_like(x_star)

    tol = _dtype_tol(dtype, np.linalg.norm(np.asarray(x_star)))

    solver = _make_solver(
        loss,
        None,
        _PD_TAG,
        x0,
        tol=tol,
        shift_const=0.0,
    )

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps == 2

    np.testing.assert_allclose(
        x_opt,
        x_star,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.requires_x64
def test_psd_singular_quadratic_preserves_null_space(dtype):
    """Newton-Cholesky converges to a true minimizer while preserving x0's null-space component."""
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss, hess = _quadratic_loss_and_hessian(A, b)

    tol = _dtype_tol(
        dtype,
        np.linalg.norm(np.asarray(b)),
        rtol_factor=500,
    )

    solver = _make_solver(
        loss,
        hess,
        _PSD_TAG,
        x0,
        shift_const=1.0,
        maxiter=100,
        tol=tol,
    )

    x_opt, state, _ = solver.run(x0)

    # The solver must actually converge
    assert bool(state.stats.converged)

    # No numerical failure
    assert bool(jnp.all(jnp.isfinite(x_opt)))

    # The final point must satisfy the original first-order condition
    residual = A @ x_opt - b
    np.testing.assert_allclose(
        residual,
        0.0,
        atol=tol,
        rtol=0.0,
    )

    initial_null = Q_null.T @ x0
    final_null = Q_null.T @ x_opt

    np.testing.assert_allclose(
        final_null,
        initial_null,
        atol=tol,
        rtol=0.0,
    )

    expected = x_star + x0_null

    np.testing.assert_allclose(
        x_opt,
        expected,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.requires_x64
def test_psd_singular_quadratic_preserves_null_space_autodiff(dtype):
    """Newton-Cholesky converges to a true minimizer while preserving x0's null-space component, using autodiff."""
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss, _ = _quadratic_loss_and_hessian(A, b)

    tol = _dtype_tol(
        dtype,
        np.linalg.norm(np.asarray(b)),
        rtol_factor=500,
    )

    solver = _make_solver(
        loss,
        None,
        _PSD_TAG,
        x0,
        shift_const=1.0,
        maxiter=100,
        tol=tol,
    )

    x_opt, state, _ = solver.run(x0)

    # The solver must converge
    assert bool(state.stats.converged)

    # No numerical failure
    assert bool(jnp.all(jnp.isfinite(x_opt)))

    # The final point must satisfy the original first-order condition
    residual = A @ x_opt - b
    np.testing.assert_allclose(
        residual,
        0.0,
        atol=tol,
        rtol=0.0,
    )

    initial_null = Q_null.T @ x0
    final_null = Q_null.T @ x_opt

    np.testing.assert_allclose(
        final_null,
        initial_null,
        atol=tol,
        rtol=0.0,
    )

    expected = x_star + x0_null

    np.testing.assert_allclose(
        x_opt,
        expected,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_pd_quadratic_scale_equivariance(dtype, scale):
    A, b, x_star = _make_pd_problem(6, dtype)

    x0 = jnp.zeros_like(x_star)
    tol = _dtype_tol(dtype, np.linalg.norm(np.asarray(x_star)))

    loss, hess = _quadratic_loss_and_hessian(
        scale * A,
        scale * b,
    )

    solver = _make_solver(
        loss,
        None,
        _PD_TAG,
        x0,
        tol=tol,
        shift_const=0.0,
    )

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps == 2

    np.testing.assert_allclose(
        x_opt,
        x_star,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_psd_quadratic_scale_equivariance(dtype, scale):
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss = _quadratic_loss(scale * A, scale * b)

    solver = _make_solver(
        loss,
        None,
        _PSD_TAG,
        x0,
        shift_const=1.0,
        maxiter=100,
    )

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert bool(jnp.all(jnp.isfinite(x_opt)))

    expected = x_star + x0_null

    tol = _dtype_tol(
        dtype,
        np.linalg.norm(np.asarray(expected)),
        rtol_factor=1000,
    )

    np.testing.assert_allclose(
        x_opt,
        expected,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "regr_setup",
    [
        "linear_regression",
        "ridge_regression",
        "linear_regression_tree",
        "ridge_regression_tree",
    ],
)
@pytest.mark.requires_x64
def test_newton_linear_or_ridge_regression(request, regr_setup):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton_params, state, _ = Newton(
        loss,
        regularizer=UnRegularized(),
        regularizer_strength=0.0,
        has_aux=False,
        tol=10**-12,
        init_params=param_init,
    ).run(param_init, X, y)
    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
        all,
        params,
        newton_params,
    )


@pytest.mark.parametrize(
    "regr_setup, regularizer",
    [
        ("linear_regression", UnRegularized()),
        ("ridge_regression", Ridge()),
        ("linear_regression_tree", UnRegularized()),
        ("ridge_regression_tree", Ridge()),
    ],
)
@pytest.mark.requires_x64
def test_newton_init_state_default(request, regr_setup, regularizer):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton = Newton(
        loss,
        regularizer=regularizer,
        regularizer_strength=0.5,
        has_aux=True,
        tol=10**-12,
        init_params=param_init,
    )
    state = newton.init_state(param_init, X, y)

    assert isinstance(state, NewtonState)
    assert state.grad_norm == jnp.array(jnp.inf)
    assert isinstance(state.stats, OptimizationInfo)
    assert state.stats.num_steps == 0
    assert state.stats.converged == jnp.array(False)
    assert jnp.isnan(state.stats.function_val)
    assert state.stats.converged == jnp.array(False)
    assert state.stats.reached_max_steps == jnp.array(False)
    assert isinstance(state.ls_state, optax.ScaleByBacktrackingLinesearchState)
