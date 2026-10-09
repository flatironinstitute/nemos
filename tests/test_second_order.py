"""Machinery the second-order solvers share, tested once for all of them.

``Newton``, ``ProximalNewton`` and ``ProximalLBFGS`` differ in where their direction comes
from -- a linear solve, a proximal quadratic subproblem on the assembled Hessian, or the
same subproblem on a limited-memory model -- and agree on everything that happens to the
direction afterwards: the slope gate, the backtracking search, and how a step that is not
taken is reported. That shared part lives here; what is specific to one solver, such as
``Newton``'s identity-shift ladder or how a GLM drives them, stays in its own file.

``solver._line_search`` is an ``ArmijoBacktracking`` for the smooth solver and a
``TsengYunBacktracking`` for the two proximal ones. Several tests below reach into it,
so three parts of it are worth stating once:

- ``_slope_descent_value(params, step, grad, fval)`` returns ``(slope, descent, value)``.
  ``value`` is the objective being decreased, the smooth loss plus the penalty where
  there is one, and it does not depend on ``step``, so a placeholder step may be passed
  to read it. ``slope`` is a vector built so that ``tree_dot(slope, step)`` is the
  Tseng & Yun slope ``grad f . step + P(params + step) - P(params)``; ``descent`` is
  that contraction. The smooth case returns ``(grad, grad . step, fval)``, the same
  three roles with an empty penalty, so tests reading them cover all three solvers.
- ``fun`` is the objective the search evaluates at each trial point, called as
  ``fun(params, *args)``.
- ``_line_search`` is the ``optax`` transformation underneath, a static field, so a test
  starving its budget swaps it with ``dataclasses.replace``.

The state it returns is a ``LineSearchState``: ``linesearch_state`` is ``optax``'s own,
and ``loss_value`` / ``descent`` / ``step_taken`` are what the solver reads back.
"""

import dataclasses

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import pytest

from nemos.regularizer import Lasso, Ridge
from nemos.solvers import Newton, ProximalNewton

# A point with both zero and non-zero coefficients: the zeros are where the L1 penalty is
# not differentiable, which is the whole reason the composite slope is needed.
_KINKED_PARAMS = np.array([0.7, 0.0, -0.4, 0.0, 0.25, 0.1])
_PENALTY_STRENGTH = 0.01


# Squared error, averaged. Its Hessian is the constant ``(2/n) X^T X``, so nothing below
# depends on a particular curvature.
def _mse(params, X, y):
    return jnp.power(y - jnp.dot(X, params), 2).mean()


# Both second-order solvers share ``_apply_or_reject``, and the slope reaching it is built
# differently by each: the plain gradient for the smooth solver, the composite Delta for
# the proximal one.
_GATE_CASES = [
    pytest.param(Newton, Ridge, 0.1, id="Newton-Ridge"),
    pytest.param(ProximalNewton, Lasso, _PENALTY_STRENGTH, id="ProximalNewton-Lasso"),
]


@pytest.mark.parametrize("solver_cls, regularizer_cls, strength", _GATE_CASES)
@pytest.mark.parametrize(
    "make_step, slope_sign",
    [
        pytest.param(lambda grad: jax.tree.map(lambda g: -g, grad), -1.0, id="descent"),
        pytest.param(jnp.zeros_like, 0.0, id="stationary"),
        pytest.param(lambda grad: grad, 1.0, id="ascent"),
    ],
)
@pytest.mark.requires_x64
def test_second_order_solvers_step_only_on_a_descent_slope(
    solver_cls, regularizer_cls, strength, make_step, slope_sign
):
    """``_apply_or_reject`` runs the line search on a negative slope and on nothing else.

    The gate is ``tree_dot(slope, step)``, a slope rather than a flag, so it has to be
    compared against zero: a positive value certifies nothing, and the Armijo test it
    would then run puts its threshold above the current value, accepting an increase.
    A proximal step with a positive ``Delta`` is what an inexact subproblem solve returns.
    """
    np.random.seed(0)
    X = np.random.normal(size=(200, _KINKED_PARAMS.size))
    y = np.random.normal(size=200)
    params = jnp.asarray(_KINKED_PARAMS)

    solver = solver_cls(
        _mse,
        regularizer=regularizer_cls(),
        regularizer_strength=strength,
        has_aux=False,
        init_params=params,
        tol=1e-12,
    )
    state = solver.init_state(params, X, y)
    (fval, _), grad = solver._fval_and_grad(params, X, y)
    step = make_step(grad)

    updates, new_ls_state = solver._line_search.update(
        params, step, grad, fval, state.ls_state, X, y
    )
    descent = new_ls_state.descent
    assert np.sign(float(descent)) == slope_sign

    new_params, new_ls_state, no_step_found = solver._apply_or_reject(
        params, step, grad, state, fval, X, y
    )
    # a stationary slope is not a failure: it is the optimum, and the zero step it leaves
    # behind is what the convergence test is supposed to read
    assert bool(no_step_found) == (slope_sign > 0)

    if slope_sign < 0:
        assert not np.allclose(new_params, params), "a descent step must be taken"
    else:
        np.testing.assert_array_equal(np.asarray(new_params), np.asarray(params))
        # the rejected branch returns the state untouched, so the next iteration
        # restarts the search from the same stepsize
        np.testing.assert_array_equal(
            np.asarray(new_ls_state.linesearch_state.learning_rate),
            np.asarray(state.ls_state.linesearch_state.learning_rate),
        )


@pytest.mark.requires_x64
def test_prox_newton_reports_a_nan_slope_as_no_step_found():
    """A diverged subproblem must be reported, not stepped on and not called converged.

    A NaN slope certifies nothing, so the step is rejected. Rejection alone would leave
    ``y_diff`` at zero for the next Cauchy test to read as convergence, so
    ``_apply_or_reject`` reports it instead and the iterate stays finite.
    """
    np.random.seed(0)
    X = np.random.normal(size=(200, _KINKED_PARAMS.size))
    y = np.random.normal(size=200)
    params = jnp.asarray(_KINKED_PARAMS)

    solver = ProximalNewton(
        _mse,
        regularizer=Lasso(),
        regularizer_strength=_PENALTY_STRENGTH,
        has_aux=False,
        init_params=params,
        tol=1e-12,
    )
    state = solver.init_state(params, X, y)
    (fval, _), grad = solver._fval_and_grad(params, X, y)
    step = jax.tree.map(lambda g: jnp.full_like(g, jnp.nan), grad)

    updates, new_ls_state = solver._line_search.update(
        params, step, grad, fval, state.ls_state, X, y
    )
    descent = new_ls_state.descent
    assert np.isnan(float(descent))

    new_params, new_ls_state, no_step_found = solver._apply_or_reject(
        params, step, grad, state, fval, X, y
    )
    assert bool(no_step_found)
    np.testing.assert_array_equal(np.asarray(new_params), np.asarray(params))


# ``no_step_found`` has three producers: a slope that does not certify descent (covered
# above), a line search that spends its budget without reaching sufficient decrease, and
# -- at the loop level -- any of them ending the run without claiming convergence.
_STALL_CASES = [
    pytest.param(Newton, Ridge, 0.1, id="Newton-Ridge"),
    pytest.param(ProximalNewton, Lasso, _PENALTY_STRENGTH, id="ProximalNewton-Lasso"),
]


def _stall_problem(solver_cls, regularizer_cls, strength, **kwargs):
    np.random.seed(0)
    X = np.random.normal(size=(200, _KINKED_PARAMS.size))
    y = np.random.normal(size=200)
    params = jnp.asarray(_KINKED_PARAMS)
    solver = solver_cls(
        _mse,
        regularizer=regularizer_cls(),
        regularizer_strength=strength,
        has_aux=False,
        init_params=params,
        tol=1e-12,
        **kwargs,
    )
    return solver, X, y, params


def _starve(line_search, max_backtracking_steps=1):
    """The same search with a budget too small to rescue any step.

    ``_line_search`` is a static field of an ``eqx.Module``, so it is replaced rather
    than written through.
    """
    return dataclasses.replace(
        line_search,
        _line_search=optax.scale_by_backtracking_linesearch(max_backtracking_steps),
    )


@pytest.mark.parametrize("solver_cls, regularizer_cls, strength", _STALL_CASES)
@pytest.mark.requires_x64
def test_second_order_solvers_reject_an_exhausted_line_search(
    solver_cls, regularizer_cls, strength
):
    """optax returns its last trial whether or not sufficient decrease was reached.

    Starving ``max_backtracking_steps`` makes it return a trial that raises the
    objective; ``info.decrease_error`` is positive exactly then, so the step must not be
    applied. Without the check the iterate moves to that worse point.
    """
    solver, X, y, params = _stall_problem(solver_cls, regularizer_cls, strength)
    solver._line_search = _starve(solver._line_search)
    state = solver.init_state(params, X, y)
    (fval, _), grad = solver._fval_and_grad(params, X, y)
    # far past any stepsize one halving can rescue, and still a descent direction so the
    # slope gate passes and the line search is the only thing that can reject it
    step = jax.tree.map(lambda g: -1e6 * g, grad)

    slope, descent, value = solver._line_search._slope_descent_value(
        params, step, grad, fval
    )
    assert float(descent) < 0.0

    new_params, new_ls_state, no_step_found = solver._apply_or_reject(
        params, step, grad, state, fval, X, y
    )

    assert float(new_ls_state.linesearch_state.info.decrease_error) > 0.0, (
        "the search must have failed"
    )
    assert bool(no_step_found)
    np.testing.assert_array_equal(np.asarray(new_params), np.asarray(params))
    # the trial optax offered was worse than the starting point, which is why the
    # check is needed
    value_fn = lambda p: solver._line_search.fun(p, X, y)  # noqa: E731
    updates, _ = solver._line_search._line_search.update(
        step,
        state.ls_state.linesearch_state,
        params,
        value=value,
        grad=slope,
        value_fn=value_fn,
    )
    offered = jax.tree.map(lambda p, u: p + u, params, updates)
    assert float(value_fn(offered)) > float(value_fn(params))


@pytest.mark.parametrize("solver_cls, regularizer_cls, strength", _STALL_CASES)
@pytest.mark.requires_x64
def test_second_order_solvers_end_a_stalled_run_without_claiming_convergence(
    solver_cls, regularizer_cls, strength, monkeypatch
):
    """A rejected step ends ``run`` as a stall, distinguishable from a solved run.

    A rejection leaves ``y_diff`` at zero, which ``cauchy_termination`` reads as
    convergence, so the loop has to stop on ``no_step_found`` instead. ``OptimizationInfo``
    then reports neither ``converged`` nor ``reached_max_steps``.
    """
    solver, X, y, params = _stall_problem(
        solver_cls, regularizer_cls, strength, maxiter=50
    )
    # reject unconditionally, whatever direction the solver produces, so the test
    # covers how the loop handles a rejection and not one particular cause of it.
    # ``Loop`` is a frozen ``eqx.Module``, so the class is patched rather than the
    # instance the solver holds.
    monkeypatch.setattr(
        solver_cls,
        "_apply_or_reject",
        lambda self, p, step, grad, state, fval, line_search, *args: (
            p,
            state.ls_state,
            jnp.array(True),
        ),
    )

    final_params, final_state, _ = solver.run(params, X, y)

    assert bool(final_state.no_step_found)
    assert not bool(final_state.stats.converged)
    assert not bool(final_state.stats.reached_max_steps)
    assert int(final_state.stats.num_steps) == 1, "the run must stop on the first stall"
    np.testing.assert_array_equal(np.asarray(final_params), np.asarray(params))


@pytest.mark.parametrize("solver_cls, regularizer_cls, strength", _STALL_CASES)
@pytest.mark.requires_x64
def test_second_order_solvers_do_not_read_a_flat_optimum_as_a_failed_search(
    solver_cls, regularizer_cls, strength
):
    """A search that fails at a stationary point has found the optimum, not a breakdown.

    The Armijo test compares the composite objective at the trial point against its
    value at the current one. That current value is ``value`` below, what
    ``_slope_descent_value`` returns: the smooth loss plus the penalty. Near the optimum
    the difference between the two is smaller than the rounding error on the objective
    itself, so the comparison carries no information, no trial passes it, and the search
    spends its whole budget and reports ``decrease_error > 0``. That is the normal way a
    run ends, and calling it ``no_step_found`` would make every tight run report failure.

    What tells this apart from a direction that actually blew up is the slope. Here it is
    below ``eps * abs(value)``, the scale at which the objective can no longer be
    resolved; a blown-up direction arrives with a slope many orders of magnitude above
    it.
    """
    solver, X, y, params = _stall_problem(solver_cls, regularizer_cls, strength)
    solver._line_search = _starve(solver._line_search)
    state = solver.init_state(params, X, y)
    (fval, _), grad = solver._fval_and_grad(params, X, y)
    # Recreate what the search sees at the optimum: a slope under the rounding scale.
    eps = np.finfo(np.float64).eps
    # ``grad`` is a placeholder step here; ``value`` does not depend on it
    *_, value = solver._line_search._slope_descent_value(params, grad, grad, fval)
    # along -grad the slope is -step_length * ||grad||^2, so solve for the length
    step_length = 0.1 * eps * abs(value) / lx.internal.tree_dot(grad, grad)
    step = jax.tree.map(lambda g: -step_length * g, grad)

    _, descent, _ = solver._line_search._slope_descent_value(params, step, grad, fval)
    descent = float(descent)
    assert descent < 0.0
    assert abs(descent) <= eps * abs(float(value))

    new_params, new_ls_state, no_step_found = solver._apply_or_reject(
        params, step, grad, state, fval, X, y
    )

    assert float(new_ls_state.linesearch_state.info.decrease_error) > 0.0, (
        "the search must have failed"
    )
    assert not bool(no_step_found), "a flat optimum is convergence, not a stall"
    np.testing.assert_array_equal(np.asarray(new_params), np.asarray(params))


@pytest.mark.parametrize("solver_cls, regularizer_cls, strength", _STALL_CASES)
@pytest.mark.requires_x64
def test_second_order_solvers_converge_at_a_tolerance_below_the_search_noise_floor(
    solver_cls, regularizer_cls, strength
):
    """End to end: a run asking for more precision than the line search can resolve.

    ``tol=1e-12`` on an objective of order one is below the point where the backtracking
    search stops being able to show decrease, so the last iterations rest on the
    stationary branch above. The run has to finish as ``converged``, with the iterate at
    the optimum. Reporting ``no_step_found`` there was the regression, and this test
    checks for it.
    """
    solver, X, y, params = _stall_problem(
        solver_cls, regularizer_cls, strength, maxiter=500
    )
    final_params, final_state, _ = solver.run(params, X, y)

    assert bool(final_state.stats.converged)
    assert not bool(final_state.no_step_found)
    assert not bool(final_state.stats.reached_max_steps)
    assert np.all(np.isfinite(np.asarray(final_params)))

    # and it is the optimum: no descent direction of any length improves the objective.
    # ``_line_search.fun`` is the penalized objective for all three solvers, which is
    # the one being minimized -- ``solver.fun`` is the smooth part alone for a proximal
    # solver, and a step downhill on that is not a counterexample.
    def objective(c):
        return float(solver._line_search.fun(c, X, y))

    (fval, _), grad = solver._fval_and_grad(final_params, X, y)
    for scale in (1e-4, 1e-6, 1e-8):
        trial = jax.tree.map(lambda p, g: p - scale * g, final_params, grad)
        assert objective(trial) >= objective(final_params) - 1e-12
