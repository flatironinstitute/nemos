import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.regularizer import UnRegularized
from nemos.solvers._newton import (
    NewtonCholesky,
    _add_diagonal_shift,
    _compute_diagonal_shift,
)

# Helpers


def _make_pd_problem(n, dtype, rng=None):
    """Return (A, b, x_star) for a strongly convex quadratic f = 0.5 x'Ax - b'x."""
    if rng is None:
        rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n)).astype(dtype)
    A = (M @ M.T + n * np.eye(n, dtype=dtype)).astype(dtype)
    b = rng.standard_normal(n).astype(dtype)
    x_star = np.linalg.solve(A, b)
    return jnp.array(A), jnp.array(b), jnp.array(x_star)


def _make_psd_singular_problem(n, rank, dtype, rng=None):
    """Return (A, b, x_min_norm) with rank-deficient A ⪰ 0 and b ∈ range(A).

    x_min_norm = A† b is the minimum-norm minimizer (i.e. the solution with
    zero null-space component, reached when x0 = 0).
    """
    if rng is None:
        rng = np.random.default_rng(1)
    V = rng.standard_normal((n, rank)).astype(dtype)
    s = rng.uniform(0.5, 2.0, rank).astype(dtype)
    A = (V * s) @ V.T  # rank-deficient PSD
    # b in range(A): pick a random coefficient vector in R^rank
    c = rng.standard_normal(rank).astype(dtype)
    b = V @ (s * c)  # = A @ (V c / s … unnormalized, but b = A w for w = V c/s)
    # minimum-norm solution: A† b = V diag(1/s) V' b  (Moore-Penrose)
    x_min_norm = V @ (c / s * s)  # = V @ c  (simplifies)
    # double-check: A @ x_min_norm ≈ b
    return jnp.array(A), jnp.array(b), jnp.array(x_min_norm)


def _quadratic_loss_and_hessian(A, b):
    """Return (loss_fn, hess_fn) for f(x) = 0.5 x'Ax - b'x."""

    def loss(params, *args):
        x = params
        return 0.5 * jnp.dot(x, A @ x) - jnp.dot(b, x)

    def hess(params, *args):
        return A

    return loss, hess


def _make_newton_cholesky_solver(loss_fn, hess_fn, hess_tag, init_params, **kwargs):
    """Convenience wrapper that wires up a NewtonCholesky solver."""
    solver = NewtonCholesky(
        loss_fn,
        regularizer=UnRegularized(),
        regularizer_strength=0.0,
        has_aux=False,
        init_params=init_params,
        jit=False,  # easier debugging in tests
        **kwargs,
    )
    solver.setup_hessian(
        hess_fn=hess_fn,
        hess_tag=hess_tag,
    )
    return solver


def _dense_H():
    """3x3 diagonal matrix with max diagonal entry 100."""
    return jnp.diag(jnp.array([1.0, 10.0, 100.0]))


def _pytree_H():
    """Two-parameter pytree Hessian; dominant block is ("w","w") with max_diag=100, n=2."""
    return {
        "w": {"w": jnp.diag(jnp.array([100.0, 50.0])), "b": jnp.zeros((2, 1))},
        "b": {"w": jnp.zeros((1, 2)), "b": jnp.diag(jnp.array([1.0]))},
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
    def test_adds_to_diagonal_only(self, H_factory, n, max_diag):
        """Off-diagonal entries must not change."""
        H = H_factory()
        tau = 0.5
        H_mod = _add_diagonal_shift(H, tau)
        # For each on-diagonal leaf, diagonal increases by tau; off-diagonal unchanged.
        orig_leaves = jax.tree.leaves(H)
        mod_leaves = jax.tree.leaves(H_mod)
        for orig, mod in zip(orig_leaves, mod_leaves):
            if orig.ndim == 2 and orig.shape[0] == orig.shape[1]:
                np.testing.assert_allclose(jnp.diagonal(mod), jnp.diagonal(orig) + tau)
                mask = ~jnp.eye(orig.shape[0], dtype=bool)
                np.testing.assert_allclose(mod[mask], orig[mask])
            else:
                # cross / rectangular leaf: unchanged entirely
                np.testing.assert_allclose(mod, orig)

    @pytest.mark.parametrize("H_factory, n, max_diag", _H_CASES)
    def test_zero_shift_is_identity_op(self, H_factory, n, max_diag):
        """A zero shift must leave every leaf unchanged."""
        H = H_factory()
        H_mod = _add_diagonal_shift(H, 0.0)
        jax.tree.map(lambda orig, mod: np.testing.assert_allclose(mod, orig), H, H_mod)
