import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nemos._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    NegativeDefinite,
    PositiveDefinite,
    PositiveSemiDefinite,
    Symmetric,
)
from nemos.regularizer import Ridge, UnRegularized
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.solvers._newton import Newton, NewtonState
from nemos.tree_utils import pytree_map_and_reduce

N = 8

_PD_TAG = HessianTag(structure=Full, property=PositiveDefinite)
_PSD_TAG = HessianTag(structure=Full, property=PositiveSemiDefinite)
_GENERAL_TAG = HessianTag(structure=Full, property=General)
_SYMMETRIC_TAG = HessianTag(structure=Full, property=Symmetric)
_ND_TAG = HessianTag(structure=Full, property=NegativeDefinite)


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


def _make_solver(loss_fn, hess_fn, hess_tag, init_params, jit=False, **kwargs):
    """Wire up a Newton solver with sensible defaults for tests."""
    solver = Newton(
        loss_fn,
        regularizer=UnRegularized(),
        regularizer_strength=0.0,
        has_aux=False,
        init_params=init_params,
        jit=jit,
        **kwargs,
    )
    solver.setup_hessian(hess_fn=hess_fn, hess_tag=hess_tag)
    return solver


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("jit", [False, True])
def test_pd_quadratic_convergence(dtype, jit):
    """Newton-Cholesky solves a PD quadratic exactly in one Newton step."""
    A, b, x_star = _make_pd_problem(6, dtype)

    loss, hess = _quadratic_loss_and_hessian(A, b)
    x0 = jnp.zeros_like(x_star)

    solver = _make_solver(loss, hess, _PD_TAG, x0, shift_const=0.0, jit=jit)

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps == 2

    np.testing.assert_allclose(x_opt, x_star, atol=solver.tol, rtol=solver.rtol)

    residual = A @ x_opt - b
    np.testing.assert_allclose(residual, 0.0, atol=solver.tol, rtol=solver.rtol)


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_run_converges_on_pd_quadratic(jit):
    """Solver must converge to the exact minimiser of a strongly convex quadratic."""
    A, b, x_star = _make_pd_problem(N, np.float64)
    loss, hess = _quadratic_loss_and_hessian(A, b)
    x0 = jnp.zeros(N)
    solver = _make_solver(loss, hess, _PD_TAG, x0, jit=jit)

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


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("jit", [False, True])
def test_pd_quadratic_convergence_autodiff(dtype, jit):
    """Newton-Cholesky solves a PD quadratic using an autodiff Hessian."""
    A, b, x_star = _make_pd_problem(6, dtype)

    loss = _quadratic_loss(A, b)
    x0 = jnp.zeros_like(x_star)

    solver = _make_solver(loss, None, _PD_TAG, x0, shift_const=0.0, jit=jit)

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps == 2

    np.testing.assert_allclose(x_opt, x_star, atol=solver.tol, rtol=solver.rtol)


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("jit", [False, True])
def test_psd_singular_quadratic_preserves_null_space(dtype, jit):
    """Newton-Cholesky converges to a true minimizer while preserving x0's null-space component."""
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss, hess = _quadratic_loss_and_hessian(A, b)

    solver = _make_solver(
        loss, hess, _PSD_TAG, x0, shift_const=1.0, maxiter=100, jit=jit
    )

    x_opt, state, _ = solver.run(x0)

    # The solver must actually converge
    assert bool(state.stats.converged)

    # No numerical failure
    assert bool(jnp.all(jnp.isfinite(x_opt)))

    # The final point must satisfy the original first-order condition
    residual = A @ x_opt - b
    np.testing.assert_allclose(residual, 0.0, atol=solver.tol, rtol=solver.rtol)

    initial_null = Q_null.T @ x0
    final_null = Q_null.T @ x_opt

    np.testing.assert_allclose(
        final_null, initial_null, atol=solver.tol, rtol=solver.rtol
    )

    expected = x_star + x0_null

    np.testing.assert_allclose(x_opt, expected, atol=solver.tol, rtol=solver.rtol)


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("jit", [False, True])
def test_psd_singular_quadratic_preserves_null_space_autodiff(dtype, jit):
    """Newton-Cholesky converges to a true minimizer while preserving x0's null-space component, using autodiff."""
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss, _ = _quadratic_loss_and_hessian(A, b)

    solver = _make_solver(
        loss, None, _PSD_TAG, x0, shift_const=1.0, maxiter=100, jit=jit
    )

    x_opt, state, _ = solver.run(x0)

    # The solver must converge
    assert bool(state.stats.converged)

    # No numerical failure
    assert bool(jnp.all(jnp.isfinite(x_opt)))

    # The final point must satisfy the original first-order condition
    residual = A @ x_opt - b
    np.testing.assert_allclose(residual, 0.0, atol=solver.tol, rtol=solver.rtol)

    initial_null = Q_null.T @ x0
    final_null = Q_null.T @ x_opt

    np.testing.assert_allclose(
        final_null, initial_null, atol=solver.tol, rtol=solver.rtol
    )

    expected = x_star + x0_null

    np.testing.assert_allclose(x_opt, expected, atol=solver.tol, rtol=solver.rtol)


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("jit", [False, True])
def test_pd_quadratic_scale_equivariance(dtype, scale, jit):
    A, b, x_star = _make_pd_problem(6, dtype)

    x0 = jnp.zeros_like(x_star)

    loss, hess = _quadratic_loss_and_hessian(
        scale * A,
        scale * b,
    )

    solver = _make_solver(loss, None, _PD_TAG, x0, shift_const=0.0, jit=jit)

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps == 2

    np.testing.assert_allclose(x_opt, x_star, atol=solver.tol, rtol=solver.rtol)


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("jit", [False, True])
def test_psd_quadratic_scale_equivariance(dtype, scale, jit):
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss = _quadratic_loss(scale * A, scale * b)

    solver = _make_solver(
        loss, None, _PSD_TAG, x0, shift_const=1.0, maxiter=100, jit=jit
    )

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert bool(jnp.all(jnp.isfinite(x_opt)))

    expected = x_star + x0_null

    np.testing.assert_allclose(x_opt, expected, atol=solver.tol, rtol=solver.rtol)


@pytest.mark.requires_x64
@pytest.mark.parametrize(
    "regr_setup",
    [
        "linear_regression",
        "ridge_regression",
        "linear_regression_tree",
        "ridge_regression_tree",
    ],
)
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
        has_aux=False,
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


_MOD_TAGS = [_GENERAL_TAG, _SYMMETRIC_TAG, _ND_TAG]


def _modified_direction(
    H: jax.Array, grad: jax.Array, hess_tag: HessianTag, jit: bool
) -> jax.Array:
    params = jnp.zeros_like(grad)
    loss = _quadratic_loss(H, jnp.zeros_like(grad))
    solver = _make_solver(loss, lambda params, *args: H, hess_tag, params, jit=jit)
    solver.init_state(params)
    return solver._newton_direction(grad, H, params)


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_produces_identical_directions(jit):
    """
    General, Symmetric and NegativeDefinite go down the same branch,
    so they must produce identical steps on the same problem.
    """
    H = jnp.asarray(
        [
            [-2.0, 0.5, 0.0],
            [0.5, 3.0, 0.25],
            [0.0, 0.25, -1.0],
        ],
        dtype=jnp.float64,
    )
    grad = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    directions = [
        _modified_direction(H, grad, hess_tag, jit=jit) for hess_tag in _MOD_TAGS
    ]
    for direction in directions[1:]:
        np.testing.assert_allclose(
            direction,
            directions[0],
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_general_pd_direction_matches_cholesky(jit):
    """
    With all eigenvalues well above the floor,
    the general path gives the same step as the Cholesky path,
    and solves a quadratic in one step.
    """
    A, _, _ = _make_pd_problem(6, jnp.float64)
    grad = jnp.asarray([0.5, -1.0, 2.0, 0.25, -0.75, 1.5])

    general_step = _modified_direction(A, grad, _GENERAL_TAG, jit=jit)
    expected = jnp.linalg.solve(A, -grad)

    np.testing.assert_allclose(
        general_step,
        expected,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("hess_tag", _MOD_TAGS)
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_direction_matches_sqrtm_reference(hess_tag, jit):
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    eigenvalues = np.asarray([-5.0, -2.0, 0.5, 1.5, 4.0])

    H_np = Q @ np.diag(eigenvalues) @ Q.T
    grad_np = rng.standard_normal(5)

    import scipy

    absolute_hessian = scipy.linalg.sqrtm(H_np @ H_np)
    absolute_hessian = np.real_if_close(absolute_hessian)
    expected = scipy.linalg.solve(
        absolute_hessian,
        -grad_np,
        assume_a="pos",
    )

    direction = _modified_direction(
        jnp.asarray(H_np), jnp.asarray(grad_np), hess_tag, jit=jit
    )

    np.testing.assert_allclose(
        direction,
        expected,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_direction_is_invariant_to_eigenvalue_signs(jit):
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    magnitudes = np.asarray([0.5, 1.0, 2.0, 4.0])
    grad = jnp.asarray(rng.standard_normal(4))

    reference = None

    for signs in itertools.product([-1.0, 1.0], repeat=4):
        eigenvalues = magnitudes * np.asarray(signs)
        H = jnp.asarray(Q @ np.diag(eigenvalues) @ Q.T)

        direction = _modified_direction(H, grad, _GENERAL_TAG, jit=jit)

        if reference is None:
            reference = direction
        else:
            np.testing.assert_allclose(
                direction,
                reference,
                atol=1e-10,
                rtol=1e-10,
            )


@pytest.mark.requires_x64
@pytest.mark.parametrize(
    "eigenvalues",
    [
        [0.5, 1.0, 2.0, 4.0],
        [-0.5, -1.0, -2.0, -4.0],
        [-4.0, -0.5, 1.0, 3.0],
        [0.0, -2.0, 1.0, 4.0],
        [1e-8, -1e-7, 1.0, -3.0],
    ],
)
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_direction_is_descent(eigenvalues, jit):
    rng = np.random.default_rng(6)
    n = len(eigenvalues)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    H = jnp.asarray(Q @ np.diag(eigenvalues) @ Q.T)
    grad = jnp.asarray(rng.standard_normal(n))

    direction = _modified_direction(H, grad, _GENERAL_TAG, jit=jit)
    slope = float(jnp.vdot(grad, direction))

    modified_eigenvalues = np.maximum(np.abs(eigenvalues), 1e-6)
    squared_grad_norm = float(jnp.vdot(grad, grad))

    lower_bound = -squared_grad_norm / modified_eigenvalues.min()
    upper_bound = -squared_grad_norm / modified_eigenvalues.max()

    assert slope < 0.0
    assert lower_bound - 1e-10 <= slope
    assert slope <= upper_bound + 1e-10


@pytest.mark.requires_x64
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_direction_applies_eigenvalue_floor(sign, jit):
    delta = 1e-6
    eigenvalues = jnp.asarray(
        [
            0.0,
            sign * 0.5 * delta,
            sign * delta,
            sign * 2.0 * delta,
        ],
        dtype=jnp.float64,
    )
    H = jnp.diag(eigenvalues)
    grad = jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=jnp.float64)
    direction = _modified_direction(H, grad, _GENERAL_TAG, jit=jit)
    expected = -grad / jnp.maximum(jnp.abs(eigenvalues), delta)
    np.testing.assert_allclose(
        direction,
        expected,
        atol=1e-8,
        rtol=1e-12,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("x0", [-0.2, 0.2])
@pytest.mark.parametrize("hess_tag", [_GENERAL_TAG, _SYMMETRIC_TAG])
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_newton_escapes_quartic_saddle_region(x0, hess_tag, jit):
    def loss(params, *args):
        del args
        x = params[0]
        return 0.25 * x**4 - 0.5 * x**2

    def hess(params, *args):
        del args
        x = params[0]
        return jnp.asarray([[3.0 * x**2 - 1.0]])

    init_params = jnp.asarray([x0], dtype=jnp.float64)
    expected = jnp.asarray([np.sign(x0)], dtype=jnp.float64)

    assert hess(init_params)[0, 0] < 0.0

    solver = _make_solver(
        loss, hess, hess_tag, init_params, maxiter=30, tol=1e-10, rtol=1e-8, jit=jit
    )

    params, state, _ = solver.run(init_params)

    assert bool(state.stats.converged)
    assert state.stats.num_steps < 20
    np.testing.assert_allclose(params, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_direction_is_computed_blockwise(jit):
    H = jnp.asarray(
        [
            [[-2.0, 0.5], [0.5, 3.0]],
            [[1.0, -0.25], [-0.25, -4.0]],
        ],
        dtype=jnp.float64,
    )
    grad = jnp.asarray(
        [[1.0, -2.0], [0.5, 1.5]],
        dtype=jnp.float64,
    )
    params = jnp.zeros_like(grad)

    tag = HessianTag(
        structure=BlockDiagonal,
        property=General,
        batch_axes=0,
    )

    def loss(params, *args):
        del args
        return jnp.sum(params**2)

    solver = _make_solver(loss, lambda params, *args: H, tag, params, jit=jit)
    solver.init_state(params)

    direction = solver._newton_direction(grad, H, params)

    expected = []
    for block, block_grad in zip(np.asarray(H), np.asarray(grad)):
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        modified = np.maximum(np.abs(eigenvalues), 1e-6)
        expected.append(eigenvectors @ ((eigenvectors.T @ -block_grad) / modified))

    np.testing.assert_allclose(
        direction,
        np.asarray(expected),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_direction_preserves_pytree_flattening_order(jit):
    H_dense = jnp.asarray(
        [
            [-2.0, 0.5, 0.25],
            [0.5, 3.0, -0.75],
            [0.25, -0.75, 1.0],
        ],
        dtype=jnp.float64,
    )
    linear = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    params = {
        "a": jnp.asarray([0.2, -0.4], dtype=jnp.float64),
        "b": jnp.asarray([0.7], dtype=jnp.float64),
    }

    def flatten(tree):
        return jnp.concatenate([tree["a"], tree["b"]])

    def loss(tree, *args):
        del args
        vector = flatten(tree)
        return 0.5 * vector @ H_dense @ vector + linear @ vector

    grad = jax.grad(loss)(params)
    H = jax.hessian(loss)(params)

    solver = _make_solver(loss, lambda params, *args: H, _GENERAL_TAG, params, jit=jit)
    solver.init_state(params)

    direction = solver._newton_direction(grad, H, params)

    grad_flat = np.concatenate([np.asarray(grad["a"]), np.asarray(grad["b"])])
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(H_dense))
    modified = np.maximum(np.abs(eigenvalues), 1e-6)
    expected = eigenvectors @ ((eigenvectors.T @ -grad_flat) / modified)

    np.testing.assert_allclose(
        direction["a"],
        expected[:2],
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        direction["b"],
        expected[2:],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.requires_x64
@pytest.mark.parametrize("jit", [False, True])
def test_eigh_newton_solves_rosenbrock_from_indefinite_region(jit):
    def loss(params, *args):
        del args
        x, y = params
        return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2

    x0 = jnp.asarray([0.0, 2.0], dtype=jnp.float64)
    hess = jax.hessian(loss)

    assert jnp.linalg.det(hess(x0)) < 0.0

    solver = _make_solver(
        loss,
        hess,
        _GENERAL_TAG,
        x0,
        jit=jit,
        maxiter=20,
        tol=1e-8,
        rtol=1e-8,
    )

    params, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert state.stats.num_steps < 100
    np.testing.assert_allclose(
        params,
        jnp.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )
