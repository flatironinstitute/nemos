import jax
import jax.numpy as jnp
from contextlib import nullcontext as does_not_raise
import numpy as np
import pytest

from nemos.regularizer import UnRegularized
from nemos.solvers._hess import (
    Full,
    General,
    HessianTag,
    PositiveDefinite,
    PositiveSemiDefinite,
)
from nemos.solvers._newton import (
    NewtonCholesky,
    _add_diagonal_shift,
    _compute_diagonal_shift,
)


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
        return _quadratic_loss(params, *args)

    def hess(params, *args):
        return A

    return loss, hess


def _quadratic_loss(A, b):
    """Quadratic loss for autodiff-Hessian tests."""

    def loss(params, *args):
        return 0.5 * jnp.dot(params, A @ params) - jnp.dot(b, params)

    return loss


def _make_solver(loss_fn, hess_fn, hess_tag, init_params, **kwargs):
    """Wire up a NewtonCholesky solver with sensible defaults for tests."""
    solver = NewtonCholesky(
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


def _dense_H():
    """3x3 diagonal matrix with max diagonal entry 100."""
    return jnp.diag(jnp.array([1.0, 10.0, 100.0]))


def _pytree_H():
    return {
        "w": {
            "w": jnp.diag(jnp.array([100.0, 50.0])),
            "b": jnp.zeros((2, 2)),  # square, but NOT a diagonal block
        },
        "b": {
            "w": jnp.zeros((2, 2)),  # square, but NOT a diagonal block
            "b": jnp.diag(jnp.array([1.0, 1.0])),
        },
    }


_H_CASES = [
    pytest.param(_dense_H, 3, 100.0, id="dense"),
    pytest.param(_pytree_H, 2, 100.0, id="pytree"),
]


class TestComputeDiagonalShift:
    """Unit tests for the _compute_diagonal_shift helper."""

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_exact_formula(self, H_factory, n, max_diag):
        """Tau must equal shift_const * n * eps * max_diag."""
        H = H_factory()
        shift_const = 2.0
        tau = _compute_diagonal_shift(H, shift_const)
        eps = float(jnp.finfo(jax.tree.leaves(H)[0].dtype).eps)
        expected = shift_const * n * eps * max_diag
        np.testing.assert_allclose(tau, expected)

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_shift_const_is_linear_multiplier(self, H_factory, n, max_diag):
        """Doubling shift_const must exactly double tau."""
        H = H_factory()
        tau1 = _compute_diagonal_shift(H, 1.0)
        tau2 = _compute_diagonal_shift(H, 2.0)
        np.testing.assert_allclose(tau2, 2.0 * tau1)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_uses_dtype_eps(self, H_factory, n, max_diag):
        """Tau tracks the eps of the actual leaf dtype."""
        H = H_factory()
        # cast all leaves to float32 / float64
        H32 = jax.tree.map(lambda x: x.astype(jnp.float32), H)
        H64 = jax.tree.map(lambda x: x.astype(jnp.float64), H)
        tau32 = _compute_diagonal_shift(H32, 1.0)
        tau64 = _compute_diagonal_shift(H64, 1.0)
        assert tau64 < tau32, "float64 shift should be smaller than float32 shift."

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_off_diagonal_entries_ignored(self, H_factory, n, max_diag):
        """Changing off-diagonal values must not affect tau."""
        H = H_factory()

        def perturb_off_diagonal(h):
            """Add 1e6 to every off-diagonal entry of a 2-D leaf."""
            if h.ndim == 2:
                return h + 1e6 * (1 - jnp.eye(*h.shape, dtype=h.dtype))
            return h

        H_perturbed = jax.tree.map(perturb_off_diagonal, H)

        tau_orig = _compute_diagonal_shift(H, 1.0)
        tau_pert = _compute_diagonal_shift(H_perturbed, 1.0)
        np.testing.assert_allclose(
            tau_pert,
            tau_orig,
            err_msg="Off-diagonal perturbation changed tau.",
        )


class TestAddDiagonalShift:
    """Unit tests for the _add_diagonal_shift helper."""

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_diagonal_blocks_increase_by_tau(self, H_factory, n, max_diag):
        """Diagonal Hessian blocks have tau added to their diagonal."""
        H = H_factory()
        tau = 3.7
        H_mod = _add_diagonal_shift(H, tau)

        original = dict(jax.tree_util.tree_leaves_with_path(H))
        modified = dict(jax.tree_util.tree_leaves_with_path(H_mod))

        for path, h in original.items():
            if len(path) % 2 == 0 and path[: len(path) // 2] == path[len(path) // 2 :]:
                np.testing.assert_allclose(
                    jnp.diagonal(modified[path]),
                    jnp.diagonal(h) + tau,
                    err_msg=f"Diagonal block {path} was not shifted.",
                )

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_off_diagonal_blocks_unchanged(self, H_factory, n, max_diag):
        """Off-diagonal Hessian blocks are completely unchanged."""
        H = H_factory()
        tau = 5.0
        H_mod = _add_diagonal_shift(H, tau)

        original = dict(jax.tree_util.tree_leaves_with_path(H))
        modified = dict(jax.tree_util.tree_leaves_with_path(H_mod))

        for path, h in original.items():
            is_diagonal = (
                len(path) % 2 == 0 and path[: len(path) // 2] == path[len(path) // 2 :]
            )

            if not is_diagonal:
                np.testing.assert_array_equal(
                    modified[path],
                    h,
                    err_msg=f"Off-diagonal block {path} was modified.",
                )

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_zero_shift_is_identity_op(self, H_factory, n, max_diag):
        """A zero shift leaves the entire Hessian unchanged."""
        H = H_factory()
        H_mod = _add_diagonal_shift(H, 0.0)

        for original, modified in zip(
            jax.tree.leaves(H),
            jax.tree.leaves(H_mod),
        ):
            np.testing.assert_array_equal(
                modified,
                original,
                err_msg="Hessian changed with tau=0.",
            )

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_pytree_structure_preserved(self, H_factory, n, max_diag):
        """The Hessian pytree structure is preserved."""
        H = H_factory()
        H_mod = _add_diagonal_shift(H, 1.0)

        assert jax.tree_util.tree_structure(H_mod) == jax.tree_util.tree_structure(H)

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_shapes_preserved(self, H_factory, n, max_diag):
        """Every Hessian leaf retains its original shape."""
        H = H_factory()
        H_mod = _add_diagonal_shift(H, 7.0)

        for original, modified in zip(
            jax.tree.leaves(H),
            jax.tree.leaves(H_mod),
        ):
            assert modified.shape == original.shape

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_dtypes_preserved(self, H_factory, n, max_diag):
        """Every Hessian leaf retains its original dtype."""
        H = H_factory()
        H_mod = _add_diagonal_shift(H, 1.0)

        for original, modified in zip(
            jax.tree.leaves(H),
            jax.tree.leaves(H_mod),
        ):
            assert modified.dtype == original.dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.requires_x64
def test_pd_quadratic_one_step(dtype):
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
    assert not bool(state.diverged)
    assert state.stats.num_steps == 1

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

    # At the optimum the Newton decrement should vanish.
    assert float(state.newton_decrement) <= tol


@pytest.mark.requires_x64
def test_run_converges_on_pd_quadratic():
    """Solver must converge to the exact minimiser of a strongly convex quadratic."""
    A, b, x_star = _make_pd_problem(N, np.float64)
    loss, hess = _quadratic_loss_and_hessian(A, b)
    x0 = jnp.zeros(N)
    solver = _make_solver(loss, hess, _PD_TAG, x0)

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged), "Solver did not converge on a PD quadratic."
    assert state.stats.num_steps == 1, (
        "Solver did not converge in 1 step on a PD quadratic."
    )
    np.testing.assert_allclose(
        x_opt,
        x_star,
        atol=1e-6,
        err_msg="Solver solution does not match analytic solution.",
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.requires_x64
def test_pd_quadratic_one_step_autodiff(dtype):
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
    assert state.stats.num_steps == 1

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
    assert not bool(state.diverged)

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

    loss, hess = _quadratic_loss_and_hessian(A, b)

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
    assert not bool(state.diverged)

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
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.requires_x64
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
    assert state.stats.num_steps == 1.0

    np.testing.assert_allclose(
        x_opt,
        x_star,
        atol=tol,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.requires_x64
def test_psd_quadratic_scale_equivariance(dtype, scale):
    A, b, x_star, x0, x0_null, Q_null = _make_psd_problem(6, dtype)

    loss = _quadratic_loss(scale * A, scale * b)

    solver = _make_solver(
        loss,
        None,
        _PSD_TAG,
        x0,
        shift_const=1.0,
        maxiter=1000,
    )

    x_opt, state, _ = solver.run(x0)

    assert bool(state.stats.converged)
    assert not bool(state.diverged)
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
    "property_, expectation",
    [
        pytest.param(
            PositiveDefinite,
            does_not_raise(),
            id="positive-definite",
        ),
        pytest.param(
            PositiveSemiDefinite,
            does_not_raise(),
            id="positive-semi-definite",
        ),
        pytest.param(
            General,
            pytest.raises(ValueError, match="positive"),
            id="general",
        ),
    ],
)
def test_hessian_property_gating(property_, expectation):
    A, b, _ = _make_pd_problem(4, jnp.float64)

    loss, hess = _quadratic_loss_and_hessian(A, b)

    with expectation:
        _make_solver(
            loss,
            hess,
            HessianTag(structure=Full, property=property_),
            jnp.zeros(4),
        )
