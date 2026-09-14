from __future__ import annotations

import importlib
import pkgutil
from copy import deepcopy

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import pytest

import nemos as nmo
from conftest import (
    all_subclasses,
    freeze_first_coef_leaf,
    initialize_feature_mask_for_population_glm,
)
from nemos._hess import MatrixProperty, MatrixStructure
from nemos._inspect_utils import is_abstract
from nemos.base_regressor import BaseRegressor
from nemos.glm import GLM, PopulationGLM
from nemos.glm.classifier_glm import ClassifierGLM, ClassifierPopulationGLM
from nemos.glm.params import GLMParams
from nemos.regularizer import GroupLasso, Lasso, Regularizer, Ridge, UnRegularized
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.solvers._newton import Newton, NewtonState, ProximalNewton
from nemos.tree_utils import pytree_map_and_reduce

# Import every submodule so all BaseRegressor subclasses are registered before the
# parametrizations below are collected (same idiom as test_model_params).
for _, _modname, _ in pkgutil.walk_packages(nmo.__path__, prefix="nemos."):
    importlib.import_module(_modname)

# Register every test here as solver-related
pytestmark = pytest.mark.solver_related


# The two second-order solvers. Everything ``Newton`` converges to, ``ProximalNewton``
# converges to as well: they differ in how the penalty is reached (a proximal operator
# rather than the penalized loss) and in the convergence test, not in the optimum. So the
# solver-agnostic tests below run for both, and only the genuinely divergent behaviour gets
# a dedicated test.
_NEWTON_SOLVERS = ("Newton", "ProximalNewton")

_SOLVERS = pytest.mark.parametrize("solver_name", _NEWTON_SOLVERS)

_SOLVER_CLASSES = {"Newton": Newton, "ProximalNewton": ProximalNewton}


def _solver_regularizers(solver_name):
    """Auto-discover every regularizer that advertises ``solver_name`` as allowed.

    The block-diagonal Hessian equals the full Hessian's diagonal blocks only if the
    penalty is additive (so the Hessian factorizes into loss + penalty terms). Additivity
    is currently baked into ``Regularizer.penalized_loss``; parametrizing over the
    discovered set means a future Newton-eligible regularizer that breaks additivity is
    caught here instead of silently mis-regularizing the Newton step.

    Membership is an exact match against the ``_allowed_solvers`` tuple, so ``"Newton"``
    does not also select ``"ProximalNewton"``. The two therefore get their own correct
    sets, and ``ProximalNewton`` picks up the nonsmooth penalties (``Lasso``,
    ``ElasticNet``, ``GroupLasso``) that ``Newton`` is not allowed for.
    """

    return sorted(
        (
            cls
            for cls in all_subclasses(Regularizer)
            if cls.__module__.startswith("nemos")
            and solver_name in getattr(cls, "_allowed_solvers", ())
        ),
        key=lambda cls: cls.__name__,
    )


def _solver_regularizer_cases():
    """``(solver_name, regularizer_cls)`` pairs, discovered rather than listed."""
    return [
        pytest.param(solver_name, cls, id=f"{solver_name}-{cls.__name__}")
        for solver_name in _NEWTON_SOLVERS
        for cls in _solver_regularizers(solver_name)
    ]


def _strength_for(regularizer_cls, value=0.1):
    """``None`` for the unpenalized regularizer, ``value`` for every other.

    ``ElasticNet`` expands a scalar into an ``(alpha, l1_ratio)`` pair in the model's
    setter, so a caller needing the value the regularizer actually sees must read
    ``model.regularizer_strength`` back rather than reuse ``value``.
    """
    return None if regularizer_cls is UnRegularized else value


def _reference_solver(regularizer_cls):
    """A first-order solver allowed for ``regularizer_cls``, to compare an optimum against.

    ``LBFGS`` cannot be used for the nonsmooth penalties, so those fall back to
    ``ProximalGradient``.
    """
    allowed = getattr(regularizer_cls, "_allowed_solvers", ())
    return "LBFGS" if "LBFGS" in allowed else "ProximalGradient"


def _block_diagonal_models():
    """Model classes that declare a block-diagonal Hessian.

    Discovered rather than listed. The block path assembles the penalty Hessian by vmapping
    the regularizer over neurons, pairing the model's ``batch_axes`` against the strength,
    so a new block-diagonal model joins the check below on arrival rather than when someone
    remembers to add it.
    """
    return sorted(
        (
            cls
            for cls in all_subclasses(BaseRegressor)
            if cls.__module__.startswith("nemos")
            and not is_abstract(cls)
            and cls._hess_structure is MatrixStructure.BLOCK_DIAGONAL
        ),
        key=lambda cls: cls.__name__,
    )


# Data for each block-diagonal model, in both ``coef`` layouts. Only the pytree layout
# distinguishes a prefix-spelled ``batch_axes`` (``GLMParams(1, 0)``, what every in-tree
# model uses) from a per-leaf one, and the two are not interchangeable.
_BLOCK_MODEL_FIXTURES = {
    PopulationGLM: (
        "population_poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation_pytree",
    ),
    ClassifierPopulationGLM: (
        "population_classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation_pytree",
    ),
}

_BLOCK_MODEL_CASES = [
    pytest.param(
        fixture_name,
        id=f"{cls.__name__}-{'pytree' if fixture_name.endswith('_pytree') else 'array'}",
    )
    for cls, fixture_names in _BLOCK_MODEL_FIXTURES.items()
    for fixture_name in fixture_names
]


def _per_neuron_strength(coef):
    """Ridge strength shaped like ``coef`` and varying along the neuron axis (axis 1).

    A strength that is constant across neurons is numerically indistinguishable from a
    scalar one, so it would not detect a wrong neuron axis in the vmapped penalty
    Hessian (``Regularizer._filter_kwargs_batch_axes``). Varying it across neurons makes
    the block and the full Hessian disagree if the axes are mismatched.
    """

    def per_leaf(leaf):
        per_neuron = 0.1 * (1 + jnp.arange(leaf.shape[1]))
        return jnp.broadcast_to(
            per_neuron.reshape((1, leaf.shape[1]) + (1,) * (leaf.ndim - 2)), leaf.shape
        )

    return jax.tree.map(per_leaf, coef)


# scalar vs. parameter-shaped strength: the second exercises the strength expansion and
# the per-ingredient batch axes inside the regularizer's penalty Hessian.
_STRENGTHS = pytest.mark.parametrize(
    "make_strength",
    [lambda coef: 0.1, _per_neuron_strength],
    ids=["scalar_strength", "per_neuron_strength"],
)


@_SOLVERS
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
def test_newton_linear_or_ridge_regression(request, regr_setup, solver_name):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton_params, state, _ = _SOLVER_CLASSES[solver_name](
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


@_SOLVERS
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
def test_newton_init_state_default(request, regr_setup, regularizer, solver_name):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton = _SOLVER_CLASSES[solver_name](
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


@_SOLVERS
def _init_params_for(glm_class):
    """Well-shaped initial params for ``_instantiate_solver``.

    Setting the solver up resolves the Hessian tag against the parameters being fitted, so
    it reads the tree: which leaves exist, and which of them are active. A bare array has
    neither.
    """
    if issubclass(glm_class, PopulationGLM):
        return GLMParams(coef=jnp.zeros((2, 3)), intercept=jnp.zeros(3))
    return GLMParams(coef=jnp.zeros(2), intercept=jnp.zeros(1))


@_SOLVERS
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_instantiate_solver(regularizer_name, glm_class, solver_name):
    glm = glm_class(
        regularizer=regularizer_name,
        solver_name=solver_name,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    solver = glm._instantiate_solver(glm._compute_loss, _init_params_for(glm_class))

    assert glm.solver_name == solver_name
    # exact type, not ``isinstance``: ``ProximalNewton`` subclasses ``Newton``, so an
    # isinstance check cannot tell the two apart and would pass for the wrong solver
    assert type(solver) is _SOLVER_CLASSES[solver_name]


@pytest.mark.requires_x64
@_SOLVERS
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize(
    "model_fixture",
    ["poissonGLM_model_instantiation", "population_poissonGLM_model_instantiation"],
)
@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_newton_matches_first_order_solver_with_frozen_params(
    regularizer_name, model_fixture, freeze, request, solver_name
):
    """A second-order solver differentiates the combined loss with respect to the active
    subtree, so it must land on the same optimum as a partition-agnostic first-order solver.

    Both freezing modes are covered: a frozen intercept drops a Hessian row, while a
    frozen ``coef`` leaf carves a block out of the ``coef`` block itself.

    For ``ProximalNewton`` this additionally checks that the proximal operator is built
    against the active subtree: a prox constructed over the full tree would shrink frozen
    leaves and move the optimum away from the reference.
    """
    X, y, model, true_params, _ = request.getfixturevalue(model_fixture)

    def build(name):
        m = type(model)(
            regularizer=regularizer_name,
            regularizer_strength=1.0 if regularizer_name == "Ridge" else None,
            solver_name=name,
            solver_kwargs={"tol": 10**-12},
        )
        if freeze == "intercept":
            m.fit_intercept = False
        else:
            freeze_first_coef_leaf(m, true_params)
        return m

    newton = build(solver_name).fit(X, y)
    reference = build("LBFGS").fit(X, y)

    assert newton.solver_name == solver_name
    np.testing.assert_allclose(newton.coef_, reference.coef_, atol=1e-5)
    np.testing.assert_allclose(newton.intercept_, reference.intercept_, atol=1e-5)


@_SOLVERS
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize(
    "model_fixture",
    ["poissonGLM_model_instantiation", "population_poissonGLM_model_instantiation"],
)
@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_newton_leaves_frozen_params_untouched(
    regularizer_name, model_fixture, freeze, request, solver_name
):
    """The frozen leaves come back bit-identical, not merely close: a Hessian that
    silently included them would move them by a small but nonzero amount.

    For ``ProximalNewton`` the proximal operator is the second way a frozen leaf could get
    moved, since a prox applied over the full tree would shrink it."""
    X, y, model, true_params, _ = request.getfixturevalue(model_fixture)

    frozen_model = type(model)(
        regularizer=regularizer_name,
        regularizer_strength=1.0 if regularizer_name == "Ridge" else None,
        solver_name=solver_name,
    )
    if freeze == "intercept":
        frozen_model.fit_intercept = False
        frozen_model.fit(X, y)
        np.testing.assert_array_equal(
            frozen_model.intercept_, np.zeros_like(true_params.intercept)
        )
    else:
        pinned = freeze_first_coef_leaf(frozen_model, true_params)
        frozen_model.fit(X, y)
        fitted = jax.tree_util.tree_leaves(frozen_model.coef_)[0]
        np.testing.assert_array_equal(fitted, jax.tree_util.tree_leaves(pinned)[0])


@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_defaults_to_newton_regardless_of_freezing(glm_class, fit_intercept):
    """Ridge defaults to Newton because its penalized Hessian is positive definite.
    Newton is partition-aware, so freezing the intercept no longer forces a fallback
    to the regularizer's first-order default."""
    model = glm_class(regularizer="Ridge", fit_intercept=fit_intercept)
    assert model.solver_name == "Newton"
    assert Ridge().default_solver != "Newton"  # the fallback would have been visible


@pytest.mark.requires_x64
@_SOLVERS
@pytest.mark.parametrize("regularizer_name", ["Ridge"])
@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_newton_population_glm_feature_mask_with_frozen_params(
    regularizer_name, freeze, request, solver_name
):
    """The mask, the active axes and the frozen axes all index the neuron axis of the
    block Hessian. A wrong ``in_axes`` survives the unmasked tests, so pair the mask
    with a freeze and check the solver still lands where a first-order solver does.

    Ridge only, and for ``Newton`` only by necessity: a masked-out coefficient has zero
    gradient *and* zero curvature, so an unpenalized masked Hessian is singular and the
    Newton step is NaN. That is unrelated to parameter freezing (it reproduces with nothing
    frozen) and is tracked in #580. ``ProximalNewton`` survives the same singular Hessian
    because it only multiplies by it, which is asserted separately.
    """
    X, y, model, true_params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation_pytree"
    )
    mask = initialize_feature_mask_for_population_glm(X, y.shape[1])
    # zero a block so the mask is not the identity: a masked-out coefficient must not
    # pick up curvature from the neuron it is masked away from
    first = sorted(mask)[0]
    mask[first] = mask[first].at[:, 0].set(0.0)

    def build(name):
        m = nmo.glm.PopulationGLM(
            regularizer=regularizer_name,
            regularizer_strength=1.0 if regularizer_name == "Ridge" else None,
            solver_name=name,
            solver_kwargs={"tol": 10**-12},
            feature_mask=mask,
        )
        if freeze == "intercept":
            m.fit_intercept = False
        else:
            freeze_first_coef_leaf(m, true_params)
        return m

    newton = build(solver_name).fit(X, y)
    reference = build("LBFGS").fit(X, y)

    for key in mask:
        np.testing.assert_allclose(newton.coef_[key], reference.coef_[key], atol=1e-5)
    np.testing.assert_allclose(newton.intercept_, reference.intercept_, atol=1e-5)


@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_population_glm_hess_fn_drops_frozen_leaves(freeze, request):
    """``_get_hess_fn`` returns the active block only: every frozen leaf position is
    ``None`` in the returned pytree, and the surviving blocks carry the neuron axis."""
    X, y, model, true_params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation_pytree"
    )
    model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1.0)
    if freeze == "intercept":
        model.fit_intercept = False
    else:
        freeze_first_coef_leaf(model, true_params)

    params = model._model_specific_initialization(X, y)
    active, frozen = model._partition_active(params)
    hess = model._get_hess_fn(frozen=frozen)(active, X, y)

    n_neurons = y.shape[1]
    if freeze == "intercept":
        assert hess.intercept is None
        assert all(row.intercept is None for row in hess.coef.values())
    else:
        pinned = sorted(model.fix_params[0])[0]
        assert active.coef[pinned] is None
        assert hess.coef[pinned] is None
        # the frozen leaf is dropped as a column too, not just as a row
        assert all(
            row.coef[pinned] is None for row in hess.coef.values() if row is not None
        )

    # every surviving block is stacked on the neuron axis
    for block in jax.tree_util.tree_leaves(hess):
        assert block.shape[0] == n_neurons


def test_feature_mask_reassignment_invalidates_solver(request):
    """The loss and the Hessian both read ``_feature_mask`` at call time, so a solver
    built against the previous mask is stale and must be torn down."""
    X, y, model, *_ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1.0)
    params = model._model_specific_initialization(X, y)
    active, frozen = model._partition_active(params)
    model._initialize_optimizer_and_state(active, X, y, frozen_params=frozen)
    assert model.solver is not None

    model.feature_mask = initialize_feature_mask_for_population_glm(X, y.shape[1])
    assert model.solver is None
    assert model.optimizer_run is None


@_SOLVERS
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize(
    "glm_class",
    [
        nmo.glm.GLM,
        nmo.glm.PopulationGLM,
        nmo.glm.ClassifierGLM,
        nmo.glm.ClassifierPopulationGLM,
    ],
)
def test_newton_glm_passes_solver_kwargs(regularizer_name, glm_class, solver_name):
    solver_kwargs = {
        "maxiter": np.random.randint(1, 100),
        "jit": False,
        "tol": 1e-6,
        # ``rtol`` is read by ``ProximalNewton._converged`` and merely stored by
        # ``Newton``; both accept it, so it belongs in the shared set
        "rtol": 1e-7,
    }
    if solver_name == "ProximalNewton":
        solver_kwargs |= {"inner_iter": 7, "inner_atol": 1e-9, "inner_rtol": 1e-9}

    glm = glm_class(
        regularizer=regularizer_name,
        solver_name=solver_name,
        solver_kwargs=solver_kwargs,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    solver = glm._instantiate_solver(glm._compute_loss, _init_params_for(glm_class))

    for k, v in solver_kwargs.items():
        assert getattr(solver, k) == v


@_SOLVERS
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_initialize_state(
    glm_class, regularizer_name, linear_regression, solver_name
):
    X, y, _, _, _ = linear_regression

    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    reg_cls = getattr(nmo.regularizer, regularizer_name)
    reg = reg_cls()

    glm = glm_class(
        regularizer=reg,
        solver_name=solver_name,
        inverse_link_function=jax.nn.softplus,
        observation_model=nmo.observation_models.PoissonObservations(),
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )

    init_params = glm.initialize_params(X, y)
    state = glm.initialize_optimizer_and_state(init_params, X, y)

    assert isinstance(state, NewtonState)
    assert state.grad_norm == jnp.array(jnp.inf)
    assert isinstance(state.stats, OptimizationInfo)
    assert state.stats.num_steps == 0
    assert state.stats.converged == jnp.array(False)
    assert jnp.isnan(state.stats.function_val)
    assert state.stats.converged == jnp.array(False)
    assert state.stats.reached_max_steps == jnp.array(False)
    assert isinstance(state.ls_state, optax.ScaleByBacktrackingLinesearchState)


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_glm_converges(request, solver_name, regularizer_cls, structure):
    """A second-order-fitted GLM should converge and return finite parameters."""
    X, y, model, _, _ = request.getfixturevalue(
        "poissonGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    model.solver_name = solver_name
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_glm_converges(
    request, solver_name, regularizer_cls, feature_mask
):
    """A second-order-fitted PopulationGLM should converge and return finite parameters."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    model.solver_name = solver_name

    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


def _full_autodiff_model(model):
    """Copy ``model`` with the block Hessian replaced by one dense autodiff matrix.

    Dropping ``_get_hess_fn`` sends ``Newton`` to ``jax.hessian`` of the penalized loss,
    and the declaration has to follow it: what the copy assembles is a single full matrix,
    not one block per neuron, so it neither has a block structure nor a batch axis to name.
    """
    full_model = deepcopy(model)
    full_model._get_hess_fn = lambda frozen=None: None
    full_model._hess_structure = MatrixStructure.FULL
    full_model._hess_batch_axes = None
    return full_model


@pytest.mark.requires_x64
@_SOLVERS
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_glm_matches_full_autodiff(
    request, feature_mask, solver_name
):
    """A block-Hessian fit should match a full autodiff model that does not vmap over subproblems."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model.regularizer = "Ridge"
    model.regularizer_strength = 0.1
    model.solver_name = solver_name
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = _full_autodiff_model(model)

    full_model.fit(X, y)
    model.fit(X, y)
    np.testing.assert_allclose(full_model.coef_, model.coef_, atol=1e-3)


def test_every_block_diagonal_model_has_fixtures():
    """Registry and discovery must agree, so no block-diagonal model goes unchecked.

    ``test_newton_block_diagonal_matches_full_autodiff_update`` is parametrized from
    ``_BLOCK_MODEL_FIXTURES``, so a model missing from it would be skipped rather than fail.
    This test is what turns that silence into a failure.
    """
    discovered = {cls.__name__ for cls in _block_diagonal_models()}
    registered = {cls.__name__ for cls in _BLOCK_MODEL_FIXTURES}
    assert discovered == registered, (
        f"declare a block-diagonal Hessian but are absent from _BLOCK_MODEL_FIXTURES, so "
        f"they are never checked against the full Hessian: {sorted(discovered - registered)}. "
        f"Registered but no longer block-diagonal: {sorted(registered - discovered)}."
    )


@pytest.mark.requires_x64
@_SOLVERS
@_STRENGTHS
@pytest.mark.parametrize("feature_mask", [True, False])
@pytest.mark.parametrize("fixture_name", _BLOCK_MODEL_CASES)
def test_newton_block_diagonal_matches_full_autodiff_update(
    request, fixture_name, feature_mask, make_strength, solver_name
):
    """One update() on the block Hessian must match a full autodiff model.

    Runs for every model declaring a block-diagonal Hessian, in both ``coef`` layouts and
    under a scalar and a per-neuron strength. The block path vmaps the regularizer's penalty
    Hessian over neurons, so a mismatch between the model's ``batch_axes`` and the strength
    surfaces here and nowhere else: the scalar strength carries no neuron axis to get wrong.
    """
    X, y, model, params, _ = request.getfixturevalue(fixture_name)
    model.regularizer = "Ridge"
    model.solver_name = solver_name
    model.regularizer_strength = make_strength(params.coef)
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = _full_autodiff_model(model)

    p0 = model.initialize_params(X, y)
    state0 = model.initialize_optimizer_and_state(p0, X, y)
    state0_full = full_model.initialize_optimizer_and_state(p0, X, y)

    p_full, state_full = full_model.update(p0, state0_full, X, y)
    p, state = model.update(p0, state0, X, y)

    # params match
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5),
        p,
        p_full,
    )

    # check that update actually changed the parameters
    changed = any(
        not np.allclose(a, b) for a, b in zip(jax.tree.leaves(p0), jax.tree.leaves(p))
    )
    assert changed, "Did not update."


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_glm_block_hessian_matches_full(
    request, feature_mask, regularizer_cls, solver_name
):
    """
    The vmapped per-neuron Hessian should equal the diagonal neuron-blocks of the
    full autodiff Hessian, and the full Hessian should be block-diagonal across neurons.

    Both Hessians are rendered as dense matrices (per neuron) via a flatten/unflatten of
    the parameter pytree, so the comparison is on the actual matrices the Newton solve
    consumes. Parametrized over every eligible regularizer: the block/full match holds only
    for additive penalties, so a non-additive one would fail here.

    Both solvers are covered, and the comparison is valid for each because the two models
    use the same solver: for ``Newton`` both sides carry the penalty curvature, for
    ``ProximalNewton`` neither does.
    """
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = _strength_for(regularizer_cls)
    model.solver_name = solver_name
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = _full_autodiff_model(model)

    p0 = model.initialize_params(X, y)
    model.initialize_optimizer_and_state(p0, X, y)
    full_model.initialize_optimizer_and_state(p0, X, y)

    p = GLMParams(*p0)

    H_full = full_model._solver._hessian(p, X, y)
    H_block = model._solver._hessian(p, X, y)

    n_neurons = p.intercept.shape[0]
    struct_neuron = jax.eval_shape(
        lambda: GLMParams(coef=p.coef[:, 0], intercept=p.intercept[0])
    )
    to_matrix = lambda block: lx.PyTreeLinearOperator(block, struct_neuron).as_matrix()

    for n in range(n_neurons):
        full_block = GLMParams(
            coef=GLMParams(
                coef=H_full.coef.coef[:, n, :, n],
                intercept=H_full.coef.intercept[:, n, n],
            ),
            intercept=GLMParams(
                coef=H_full.intercept.coef[n, :, n],
                intercept=H_full.intercept.intercept[n, n],
            ),
        )
        block = GLMParams(
            coef=GLMParams(
                coef=H_block.coef.coef[n], intercept=H_block.coef.intercept[n]
            ),
            intercept=GLMParams(
                coef=H_block.intercept.coef[n], intercept=H_block.intercept.intercept[n]
            ),
        )
        np.testing.assert_allclose(
            to_matrix(block),
            to_matrix(full_block),
            atol=1e-8,
            err_msg=f"Block Hessian for neuron {n} does not match the full diagonal block.",
        )

    # verify no cross-neuron coupling in the full Hessian
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                continue
            np.testing.assert_allclose(
                H_full.coef.coef[:, i, :, j],
                0.0,
                atol=1e-8,
                err_msg=f"Off-diagonal coef block ({i}, {j}) is nonzero.",
            )
            np.testing.assert_allclose(
                H_full.intercept.intercept[i, j],
                0.0,
                atol=1e-8,
                err_msg=f"Off-diagonal intercept block ({i}, {j}) is nonzero.",
            )


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_classifier_glm_converges(
    request, solver_name, regularizer_cls, structure
):
    """A second-order-fitted ClassifierGLM should converge and return finite parameters."""
    X, y, model, _, _ = request.getfixturevalue(
        "classifierGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    model.solver_name = solver_name
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_classifier_population_glm_converges(
    request, solver_name, regularizer_cls, feature_mask
):
    """A second-order-fitted ClassifierPopulationGLM converges and returns finite params."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    model.solver_name = solver_name
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@_SOLVERS
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_classifier_glm_matches_full_autodiff(
    request, feature_mask, solver_name
):
    """A block-Hessian classifier fit should match a full autodiff model that does not vmap."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation"
    )
    model.regularizer = "Ridge"
    model.regularizer_strength = 0.1
    model.solver_name = solver_name
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = _full_autodiff_model(model)

    full_model.fit(X, y)
    model.fit(X, y)
    np.testing.assert_allclose(full_model.coef_, model.coef_, atol=1e-3)


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_classifier_glm_block_hessian_matches_full(
    request, feature_mask, regularizer_cls, solver_name
):
    """The vmapped per-neuron Hessian should equal the diagonal neuron-blocks of the
    full autodiff Hessian, and the full Hessian should be block-diagonal across neurons.

    Both Hessians are rendered as dense matrices (per neuron) via a flatten/unflatten of
    the parameter pytree, so the comparison is on the actual matrices the Newton solve
    consumes. Parametrized over every eligible regularizer: the block/full match holds only
    for additive penalties, so a non-additive one would fail here.

    Both solvers are covered, and the comparison is valid for each because the two models
    use the same solver: for ``Newton`` both sides carry the penalty curvature, for
    ``ProximalNewton`` neither does.

    Notes
    -----
    A relevant failure mode is if a **non-additive** regularizer is introduced, Newton
    is allowed for it, and the hessian is block-diagonal tagged.
    Newton assumes the additivity of the penalty when creating the hessian with a tree-add.
    The eventual bugfixes will be two:
    1. Disallow Newton for the regularizer,
    2. Do not assume a block diagonal hessian. The full path just uses plain jax.hess(loss).
    """
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = _strength_for(regularizer_cls)
    model.solver_name = solver_name
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = _full_autodiff_model(model)

    p0 = model.initialize_params(X, y)
    model.initialize_optimizer_and_state(p0, X, y)
    full_model.initialize_optimizer_and_state(p0, X, y)

    # encode the labels exactly as ``update`` does before handing off to the solver
    y_enc = jax.nn.one_hot(model._label_encoder.encode(y, safe=False), model.n_classes)
    p = GLMParams(*p0)

    # full: nested GLMParams coupling every (neuron, class); block: leading axis batches neurons
    H_full = full_model._solver._hessian(p, X, y_enc)
    H_block = model._solver._hessian(p, X, y_enc)

    n_neurons = p.intercept.shape[0]
    # single-neuron parameter structure used to flatten each block to a dense matrix
    struct_neuron = jax.eval_shape(
        lambda: GLMParams(coef=p.coef[:, 0], intercept=p.intercept[0])
    )
    to_matrix = lambda block: lx.PyTreeLinearOperator(block, struct_neuron).as_matrix()

    for n in range(n_neurons):
        full_block = GLMParams(
            coef=GLMParams(
                coef=H_full.coef.coef[:, n, :, :, n, :],
                intercept=H_full.coef.intercept[:, n, :, n, :],
            ),
            intercept=GLMParams(
                coef=H_full.intercept.coef[n, :, :, n, :],
                intercept=H_full.intercept.intercept[n, :, n, :],
            ),
        )
        block = GLMParams(
            coef=GLMParams(
                coef=H_block.coef.coef[n], intercept=H_block.coef.intercept[n]
            ),
            intercept=GLMParams(
                coef=H_block.intercept.coef[n], intercept=H_block.intercept.intercept[n]
            ),
        )
        np.testing.assert_allclose(
            to_matrix(block),
            to_matrix(full_block),
            atol=1e-8,
            err_msg=f"Block Hessian for neuron {n} does not match the full diagonal block.",
        )

    # the block solve is only exact if the full Hessian has no cross-neuron coupling
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                continue
            np.testing.assert_allclose(
                H_full.coef.coef[:, i, :, :, j, :],
                0.0,
                atol=1e-8,
                err_msg=f"Off-diagonal coef block ({i}, {j}) is nonzero.",
            )
            np.testing.assert_allclose(
                H_full.intercept.intercept[i, :, j, :],
                0.0,
                atol=1e-8,
                err_msg=f"Off-diagonal intercept block ({i}, {j}) is nonzero.",
            )


class _FullHessianGLM(GLM):
    """GLM supplying its own unpenalized Hessian while keeping the inherited ``Full`` tag.

    Every in-tree model that overrides ``_get_hess_fn`` is tagged ``BLOCK_DIAGONAL``
    (``PopulationGLM`` and its classifier subclass), so this is the only way to reach the
    unbatched branches: ``batch_axes=None`` in ``BaseRegressor._instantiate_solver`` and the
    early return it triggers in ``Regularizer._get_hess_fn``.
    """

    def _get_hess_fn(self, frozen=None):
        def loss(params, X, y):
            # mirrors the in-tree implementations: differentiate the combined loss
            # with respect to the active subtree alone
            rate = self._predict(eqx.combine(params, frozen), X)
            return self._observation_model._negative_log_likelihood(y, rate)

        return jax.hessian(loss)


@pytest.mark.requires_x64
@pytest.mark.parametrize("solver_name, regularizer_cls", _solver_regularizer_cases())
def test_newton_unbatched_model_hessian_matches_differentiated_loss(
    request, regularizer_cls, solver_name
):
    """``_hessian`` must be the Hessian of the smooth objective the solver differentiates.

    That is the single invariant ``setup_hessian`` maintains, and each solver satisfies it
    for the opposite reason: ``Newton`` differentiates the penalized loss, so
    ``_penalize_hessian`` adds the penalty's curvature to the model's likelihood term;
    ``ProximalNewton`` differentiates the unregularized loss and reaches the penalty through
    its prox, so no curvature is added. Comparing against
    ``jax.hessian(solver.fun)`` checks both without special-casing either, and would catch a
    penalty added twice as readily as one omitted.

    Uses a model supplying an unbatched Hessian, which is the only way to reach
    ``batch_axes=None`` in ``BaseRegressor._instantiate_solver`` and the early return it
    triggers in ``Regularizer._get_hess_fn``.
    """
    X, y, model, _, _ = request.getfixturevalue("poissonGLM_model_instantiation")
    model = _FullHessianGLM(
        observation_model=model.observation_model,
        regularizer=regularizer_cls(),
        regularizer_strength=_strength_for(regularizer_cls),
        solver_name=solver_name,
    )

    p0 = model.initialize_params(X, y)
    model.initialize_optimizer_and_state(p0, X, y)
    p = GLMParams(*p0)

    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=1e-8),
        model._solver._hessian(p, X, y),
        jax.hessian(model._solver.fun)(p, X, y),
    )


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_regularizer_change(request, model_instantiation_type):
    """Changing regularizer should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model.solver is not None, "Solver should be set after initialization."

    model.regularizer = "UnRegularized"
    assert model._solver is None, "_solver must be None after regularizer change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_strength_change(request, model_instantiation_type):
    """Changing regularizer_strength should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.regularizer_strength = 0.5
    assert (
        model._solver is None
    ), "_solver must be None after regularizer_strength change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_observation_model_change(
    request, model_instantiation_type
):
    """Changing observation_model should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.observation_model = "Gaussian"
    assert model._solver is None, "_solver must be None after observation_model change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_solver_name_change(request, model_instantiation_type):
    """Changing solver_name should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.solver_name = "LBFGS"
    assert model._solver is None, "_solver must be None after solver_name change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_solver_kwargs_change(
    request, model_instantiation_type
):
    """Changing solver_kwargs should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.solver_kwargs = {"maxiter": 50}
    assert model._solver is None, "_solver must be None after solver_kwargs change."


def test_glm_hess_structure():
    """A GLM assembles one dense Hessian, so it declares no block layout and no batch axis."""
    assert GLM._hess_structure is MatrixStructure.FULL
    assert GLM._hess_batch_axes is None


def test_population_glm_hess_structure():
    """A PopulationGLM assembles one block per neuron.

    The batch is the neuron axis of each parameter: axis 1 of ``coef``, axis 0 of
    ``intercept``.
    """
    assert PopulationGLM._hess_structure is MatrixStructure.BLOCK_DIAGONAL
    assert PopulationGLM._hess_batch_axes == GLMParams(1, 0)


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_unpenalized_loss_is_only_semidefinite(glm_class):
    """The loss alone is convex for a convexity-preserving link, and no more than that.

    A coefficient block is ``X.T W X``, which any rank-deficient design makes singular.
    """
    assert glm_class()._resolve_hess_property() is MatrixProperty.POSITIVE_SEMI_DEFINITE


# Whether a Ridge-penalized model resolves a definite Hessian. The penalty curves every
# coefficient but skips the intercept, so the verdict comes down to what the loss certifies
# there: a GLM's intercept block is ``1.T W 1``, positive for any design, while a softmax is
# flat along a uniform shift of the intercept and so certifies nothing.
_RIDGE_TAG_CASES = [
    pytest.param(
        "poissonGLM_model_instantiation", MatrixProperty.POSITIVE_DEFINITE, id="GLM"
    ),
    pytest.param(
        "population_poissonGLM_model_instantiation",
        MatrixProperty.POSITIVE_DEFINITE,
        id="PopulationGLM",
    ),
    pytest.param(
        "classifierGLM_model_instantiation",
        MatrixProperty.POSITIVE_SEMI_DEFINITE,
        id="ClassifierGLM",
    ),
    pytest.param(
        "population_classifierGLM_model_instantiation",
        MatrixProperty.POSITIVE_SEMI_DEFINITE,
        id="ClassifierPopulationGLM",
    ),
]


@pytest.mark.parametrize("fixture_name, expected_property", _RIDGE_TAG_CASES)
def test_ridge_tag_is_definite_when_the_loss_certifies_the_intercept(
    request, fixture_name, expected_property
):
    """Ridge is definite on the coefficients and flat on the intercept.

    The sum is therefore definite exactly for the models whose loss certifies the
    intercept.
    """
    _, _, model, params, *_ = request.getfixturevalue(fixture_name)
    model.regularizer = "Ridge"
    model.regularizer_strength = 0.1
    model.solver_name = "Newton"

    solver = model._instantiate_solver(model._compute_loss, params)
    assert solver._hess_tag.property is expected_property


@pytest.mark.parametrize("fixture_name, _", _RIDGE_TAG_CASES)
def test_unregularized_tag_is_not_definite(request, fixture_name, _):
    """With no penalty the tag is the loss's own.

    The intercept alone is certified, which leaves the coefficients, and so the whole
    matrix, uncertified.
    """
    _, _, model, params, *_ = request.getfixturevalue(fixture_name)
    model.regularizer = "UnRegularized"
    model.regularizer_strength = None
    model.solver_name = "Newton"

    solver = model._instantiate_solver(model._compute_loss, params)
    assert solver._hess_tag.property is MatrixProperty.POSITIVE_SEMI_DEFINITE


@pytest.mark.parametrize("regularizer_name", ["Lasso", "GroupLasso"])
def test_non_smooth_penalties_resolve_no_tag(regularizer_name):
    """A penalty with no second derivative describes no curvature, and says so with
    ``None`` rather than with a tag claiming nothing: ``combine_hessian_tags`` propagates
    it, so Newton claims nothing about the sum either.

    These regularizers do not allow Newton, so the tag is read off the regularizer rather
    than off an instantiated solver.
    """
    regularizer = getattr(nmo.regularizer, regularizer_name)()
    params = _init_params_for(GLM)
    assert regularizer._resolve_hess_tag(params, 0.1) is None


def _installed_newton(model, X, y):
    """Return the model's ``Newton``, after ``init_state`` picked the linear solver."""
    model.initialize_optimizer_and_state(model.initialize_params(X, y), X, y)
    return model._solver


def _assert_linear_solver(solver, expected_cls):
    """Assert the linear solver Newton picked, and the operator tags that go with it."""
    assert isinstance(solver._linear_solver, expected_cls)
    if expected_cls is lx.Cholesky:
        assert (
            solver._operator_tags == lx.positive_semidefinite_tag
        ), f"Expected ``positive_semidefinite_tag`` for Cholesky solver. Got ``{solver._operator_tags}`` instead!"
    else:
        assert solver._operator_tags == ()
        assert (
            solver._linear_solver.well_posed is False
        ), "Solver is well posed but shouldn't for the given tag."


_LINEAR_SOLVER_CASES = [
    pytest.param(fixture_name, regularizer_name, expected_cls, id=test_id)
    for fixture_name, regularizer_name, expected_cls, test_id in [
        ("poissonGLM_model_instantiation", "Ridge", lx.Cholesky, "GLM-Ridge"),
        (
            "population_poissonGLM_model_instantiation",
            "Ridge",
            lx.Cholesky,
            "PopulationGLM-Ridge",
        ),
        (
            "classifierGLM_model_instantiation",
            "Ridge",
            lx.AutoLinearSolver,
            "ClassifierGLM-Ridge",
        ),
        (
            "population_classifierGLM_model_instantiation",
            "Ridge",
            lx.AutoLinearSolver,
            "ClassifierPopulationGLM-Ridge",
        ),
        (
            "poissonGLM_model_instantiation",
            "UnRegularized",
            lx.AutoLinearSolver,
            "GLM-UnRegularized",
        ),
        (
            "population_poissonGLM_model_instantiation",
            "UnRegularized",
            lx.AutoLinearSolver,
            "PopulationGLM-UnRegularized",
        ),
        (
            "classifierGLM_model_instantiation",
            "UnRegularized",
            lx.AutoLinearSolver,
            "ClassifierGLM-UnRegularized",
        ),
        (
            "population_classifierGLM_model_instantiation",
            "UnRegularized",
            lx.AutoLinearSolver,
            "ClassifierPopulationGLM-UnRegularized",
        ),
    ]
]


@pytest.mark.parametrize(
    "fixture_name, regularizer_name, expected_cls", _LINEAR_SOLVER_CASES
)
def test_linear_solver_follows_the_resolved_tag(
    request, fixture_name, regularizer_name, expected_cls
):
    """A definite tag selects ``lx.Cholesky``, a weaker one ``lx.AutoLinearSolver``."""
    X, y, model, *_ = request.getfixturevalue(fixture_name)
    model.regularizer = regularizer_name
    model.regularizer_strength = None if regularizer_name == "UnRegularized" else 0.1
    model.solver_name = "Newton"

    _assert_linear_solver(_installed_newton(model, X, y), expected_cls)


@pytest.mark.parametrize(
    "fixture_name, regularizer_name, expected_cls", _LINEAR_SOLVER_CASES
)
def test_fit_resolves_the_same_linear_solver(
    request, fixture_name, regularizer_name, expected_cls
):
    """``fit`` reaches the same linear solver as ``initialize_optimizer_and_state``.

    The two differ only by ``_optimize_solver_params``, which sets solver kwargs and leaves
    the tag alone.
    """
    X, y, model, *_ = request.getfixturevalue(fixture_name)
    model.regularizer = regularizer_name
    model.regularizer_strength = None if regularizer_name == "UnRegularized" else 0.1
    model.solver_name = "Newton"

    model.fit(X, y)

    _assert_linear_solver(model._solver, expected_cls)


@pytest.mark.requires_x64
def test_newton_without_hessian_tag_uses_auto_linear_solver(linear_regression):
    """With no tag set, ``init_state`` falls back to one that claims nothing."""
    X, y, _, params, loss = linear_regression

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton = Newton(
        loss,
        regularizer=UnRegularized(),
        regularizer_strength=0.0,
        has_aux=False,
        init_params=param_init,
    )
    assert newton._hess_tag is None

    newton.init_state(param_init, X, y)

    assert newton._hess_tag.property is MatrixProperty.SYMMETRIC
    assert newton._hess_tag.structure is MatrixStructure.FULL
    assert not any(jax.tree_util.tree_leaves(newton._hess_tag.flat_on))
    assert not any(jax.tree_util.tree_leaves(newton._hess_tag.definite_on))
    _assert_linear_solver(newton, lx.AutoLinearSolver)


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_default_solver_is_newton_for_ridge(glm_class):
    """Ridge-penalized GLM should default to Newton solver."""
    model = glm_class(regularizer="Ridge", regularizer_strength=0.1)
    assert model.solver_name == "Newton"


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_default_solver_is_not_newton_for_unregularized(glm_class):
    """Unregularized GLM should NOT default to Newton."""
    model = glm_class(regularizer="UnRegularized")
    assert model.solver_name != "Newton"


@_SOLVERS
@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_solver_name_respected_when_explicitly_set(glm_class, solver_name):
    """An explicitly set second-order solver_name should be respected."""
    model = glm_class(regularizer="UnRegularized", solver_name=solver_name)
    assert model.solver_name == solver_name


@_SOLVERS
@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_solver_type_after_fit(request, model_instantiation_type, solver_name):
    """After fit(), model._solver should be an instance of the requested solver."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.regularizer = "Ridge"
    model.solver_name = solver_name
    model.fit(X, y)

    # exact type: ``ProximalNewton`` subclasses ``Newton``, so isinstance cannot separate them
    assert type(model._solver) is _SOLVER_CLASSES[solver_name]


@_SOLVERS
@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_update_increments_step_count(
    request, model_instantiation_type, solver_name
):
    """Each call to update() should increment the step counter by exactly 1."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.solver_name = solver_name
    init_params = model.initialize_params(X, y)
    state = model.initialize_optimizer_and_state(init_params, X, y)
    assert state.stats.num_steps == 0

    _, state1 = model.update(init_params, state, X, y)
    assert state1.stats.num_steps == 1

    _, state2 = model.update(model.get_model_params(), state1, X, y)
    assert state2.stats.num_steps == 2


@_SOLVERS
@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_maxiter_respected(request, model_instantiation_type, solver_name):
    """Setting maxiter=1 should bound the solver to at most 1 step."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.regularizer = "Ridge"
    model.solver_name = solver_name
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)

    n_steps = int(model.solver_state_.stats.num_steps)
    assert n_steps <= 1, f"Expected at most 1 step, got {n_steps}"


@_SOLVERS
@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_newton_invalid_kwarg_raises(glm_class, solver_name):
    """Passing an unrecognised kwarg should raise a NameError immediately."""
    with pytest.raises(NameError, match="not a kwarg"):
        glm_class(
            regularizer="Ridge",
            regularizer_strength=0.1,
            solver_name=solver_name,
            solver_kwargs={"totally_fake_kwarg": 99},
        )


@_SOLVERS
def test_second_order_solvers_store_rtol(solver_name):
    """``rtol`` is advertised by both solvers, so both must accept and keep it.

    ``ProximalNewton`` previously advertised it without taking it, so ``solver_kwargs``
    validation passed and construction then raised ``TypeError``.
    """
    model = GLM(
        regularizer="Ridge",
        regularizer_strength=0.1,
        solver_name=solver_name,
        solver_kwargs={"rtol": 1e-3},
    )
    solver = model._instantiate_solver(model._compute_loss, _init_params_for(GLM))
    assert solver.rtol == 1e-3


@pytest.mark.requires_x64
def test_prox_newton_rtol_loosens_convergence(request):
    """``rtol`` must reach the Cauchy test, not merely sit on the instance.

    ``ProximalNewton._converged`` calls ``cauchy_termination(self.rtol, self.tol, ...)``, so a
    relative tolerance well above the absolute one has to stop the run sooner. Being stored
    is not enough: that is exactly what the unreachable ``0.0`` default did before. Measured
    on this problem, 9 steps at ``rtol=0.0`` against 4 at ``rtol=1e-2``.
    """
    X, y, model, _, _ = request.getfixturevalue("poissonGLM_model_instantiation")

    def steps_at(rtol):
        fitted = GLM(
            regularizer="Lasso",
            regularizer_strength=1e-3,
            solver_name="ProximalNewton",
            solver_kwargs={"tol": 1e-12, "rtol": rtol, "maxiter": 500},
        ).fit(X, y)
        assert bool(fitted.solver_state_.stats.converged)
        return int(fitted.solver_state_.stats.num_steps)

    strict = steps_at(0.0)
    loose = steps_at(1e-2)
    assert loose < strict, (
        f"a loose rtol must terminate sooner, got {loose} steps at rtol=1e-2 against "
        f"{strict} at rtol=0.0 -- rtol is not reaching the convergence test"
    )


# Squared error, averaged. Its Hessian is the constant ``(2/n) X^T X``, which makes every
# reference below a closed form rather than another solver's output.
def _mse(params, X, y):
    return jnp.power(y - jnp.dot(X, params), 2).mean()


def _run_prox_newton(loss, X, y, init, regularizer, strength, **kwargs):
    solver = ProximalNewton(
        loss,
        regularizer=regularizer,
        regularizer_strength=strength,
        has_aux=False,
        init_params=init,
        tol=1e-12,
        maxiter=500,
        **kwargs,
    )
    return solver, *solver.run(init, X, y)


@pytest.mark.requires_x64
def test_prox_newton_matches_closed_form_lasso():
    """On an orthonormal design the Lasso solution is a soft-threshold, in closed form.

    With ``X^T X = I`` the averaged squared error separates per coordinate, and minimizing
    ``(1/n)(b^2 - 2 b z) + lam |b|`` over each gives ``b = soft(z, n lam / 2)`` for
    ``z = X^T y``. ``lam`` is chosen so both branches of the threshold are exercised: a
    threshold above every ``|z|`` returns all zeros and would pass while testing nothing.

    Measured agreement is 4.3e-8 at the default ``inner_atol``/``inner_rtol`` of 1e-8, and
    tightening those moves it to 5.1e-9, so the floor is set by the inner solve. The 1e-6
    bound below sits ~20x above the measured value.
    """
    np.random.seed(0)
    n, n_features = 200, 6
    X, _ = np.linalg.qr(np.random.normal(size=(n, n_features)))
    y = np.random.normal(size=n)
    np.testing.assert_allclose(X.T @ X, np.eye(n_features), atol=1e-12)

    strength = 0.005
    z = X.T @ y
    threshold = n * strength / 2.0
    expected = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
    # the point of the chosen strength: neither branch is empty
    assert 0 < (expected != 0).sum() < n_features

    init = jnp.zeros(n_features)
    _, params, state, _ = _run_prox_newton(_mse, X, y, init, Lasso(), strength)

    np.testing.assert_allclose(params, expected, atol=1e-8)
    assert bool(state.stats.converged)
    # the prox produces exact zeros, not merely small numbers
    np.testing.assert_array_equal(np.asarray(params) == 0.0, expected == 0.0)


@pytest.mark.requires_x64
def test_prox_newton_reduces_to_newton_without_penalty():
    """With ``P = 0`` the composite objective is smooth and both solvers must agree.

    ``ProximalNewton`` differs from ``Newton`` only in how it reaches the penalty and in
    its convergence test, so an unpenalized problem is where the two have to coincide --
    on each other and on the least-squares solution.
    """
    np.random.seed(0)
    X = np.random.normal(size=(200, 4))
    y = np.random.normal(size=200)
    ols, *_ = np.linalg.lstsq(X, y, rcond=-1)

    init = jnp.zeros(4)
    _, prox_params, _, _ = _run_prox_newton(_mse, X, y, init, UnRegularized(), None)
    newton_params, _, _ = Newton(
        _mse,
        regularizer=UnRegularized(),
        regularizer_strength=None,
        has_aux=False,
        init_params=init,
        tol=1e-12,
    ).run(init, X, y)

    np.testing.assert_allclose(prox_params, ols, atol=1e-6)
    np.testing.assert_allclose(prox_params, newton_params, atol=1e-6)


@pytest.mark.requires_x64
def test_prox_newton_singular_hessian_converges(request):
    """A rank-deficient design gives a singular ``H``, which is not by itself a problem.

    ``H`` is only multiplied, never inverted, so the solve does not need definiteness. The
    subproblem also stays bounded below here because ``grad f`` lies in ``range(X^T)``
    while ``ker H = ker X``, leaving the two orthogonal -- asserted directly, since it is
    the condition the class docstring names.

    Starting from zero, no iterate acquires a component in ``ker H``, so the run lands on
    the minimum-norm least-squares solution that ``lstsq`` returns.
    """
    np.random.seed(0)
    n = 200
    Z = np.random.normal(size=(n, 3))
    X = np.column_stack([Z, Z[:, 0], Z[:, 1] + Z[:, 2]])
    y = np.random.normal(size=n)
    assert np.linalg.matrix_rank(X) == 3

    min_norm, *_ = np.linalg.lstsq(X, y, rcond=None)
    init = jnp.zeros(X.shape[1])
    _, params, state, _ = _run_prox_newton(_mse, X, y, init, UnRegularized(), None)

    hess = (2.0 / n) * X.T @ X
    eigvals = np.linalg.eigvalsh(hess)
    assert eigvals.min() > -1e-10, "Hessian must be positive semidefinite"
    assert (
        eigvals.min() < 1e-10
    ), "Hessian must be singular for this test to mean anything"

    # grad f orthogonal to ker H: the condition that bounds the subproblem below
    grad = -(2.0 / n) * X.T @ (y - X @ np.asarray(params))
    _, singular_values, right_vectors = np.linalg.svd(hess)
    null_space = right_vectors[singular_values < 1e-10 * singular_values.max()]
    assert null_space.shape[0] == 2
    for direction in null_space:
        assert abs(grad @ direction) < 1e-10

    assert bool(state.stats.converged)
    np.testing.assert_allclose(params, min_norm, atol=1e-6)


@pytest.mark.requires_x64
def test_prox_newton_prox_applies_across_a_pytree():
    """The prox is applied to every leaf of a pytree, and the structure survives.

    ``b = 0`` is the exact Lasso optimum whenever ``lam >= ||grad f(0)||_inf``, which makes
    an all-zero solution a derived reference rather than a guess. Doubling that bound puts
    the problem strictly inside the regime, so every leaf must come back exactly zero.
    """
    np.random.seed(0)
    n = 200
    X = {
        "input_1": np.random.normal(size=(n, 2)),
        "input_2": np.random.normal(size=(n, 3)),
    }
    y = np.random.normal(size=n)

    def loss(params, X, y):
        pred = sum(jnp.dot(X[k], params[k]) for k in X)
        return jnp.power(y - pred, 2).mean()

    zeros = {k: jnp.zeros(v.shape[1]) for k, v in X.items()}
    grad_at_zero = jax.grad(loss)(zeros, X, y)
    strength = 2.0 * max(
        float(jnp.max(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(grad_at_zero)
    )
    # start away from the optimum: initializing at zero would let a solver that never
    # moved pass this test
    init = {k: jnp.asarray(np.random.normal(size=v.shape[1])) for k, v in X.items()}
    assert all(np.any(np.asarray(leaf) != 0.0) for leaf in init.values())

    _, params, state, _ = _run_prox_newton(loss, X, y, init, Lasso(), strength)

    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(init)
    assert bool(state.stats.converged)
    for key in X:
        np.testing.assert_array_equal(
            np.asarray(params[key]), np.zeros(X[key].shape[1])
        )


@pytest.mark.requires_x64
def test_prox_newton_autodiff_hessian_matches_supplied_hessian():
    """Without ``setup_hessian`` the solver autodiffs its smooth loss; the two agree.

    ``_build_cache`` falls back to ``jax.hessian(self.fun)`` when no Hessian was supplied.
    For a squared-error loss the analytic Hessian is the constant ``(2/n) X^T X``, so the
    two paths must produce the same iterates, not merely similar ones.
    """
    np.random.seed(0)
    n = 200
    X = np.random.normal(size=(n, 4))
    y = np.random.normal(size=n)
    init = jnp.zeros(4)

    _, autodiff_params, _, _ = _run_prox_newton(_mse, X, y, init, Lasso(), 0.005)

    solver = ProximalNewton(
        _mse,
        regularizer=Lasso(),
        regularizer_strength=0.005,
        has_aux=False,
        init_params=init,
        tol=1e-12,
        maxiter=500,
    )
    solver.setup_hessian(lambda params, X, y: (2.0 / n) * X.T @ X)
    supplied_params, _, _ = solver.run(init, X, y)

    np.testing.assert_allclose(autodiff_params, supplied_params, atol=1e-10)


@pytest.mark.requires_x64
def test_prox_newton_indefinite_hessian_does_not_report_success():
    """An indefinite ``H`` is outside the solver's contract, and it says so.

    The subproblem ``min_d grad^T d + 0.5 d^T H d + P(b + d)`` is unbounded below along a
    direction of negative curvature whenever ``P`` grows at most linearly there, so no
    minimizer exists for any inner solver to find. Fixing that needs Hessian modification
    (damping or a trust region), which this solver does not do.

    The assertion is deliberately weak: whatever comes back, the solver must not return a
    finite point while reporting convergence. Pinning the exact NaN output would freeze an
    implementation detail rather than the contract.
    """
    np.random.seed(0)
    n = 200
    X = np.random.normal(size=(n, 4))
    y = np.random.normal(size=n)
    init = jnp.zeros(4)

    solver = ProximalNewton(
        _mse,
        regularizer=Lasso(),
        regularizer_strength=0.005,
        has_aux=False,
        init_params=init,
        tol=1e-12,
        maxiter=50,
    )
    indefinite = np.diag([2.0, 1.0, 0.5, -1.0])
    assert np.linalg.eigvalsh(indefinite).min() < 0
    solver.setup_hessian(lambda params, X, y: indefinite)

    params, state, _ = solver.run(init, X, y)

    finite = bool(np.all(np.isfinite(np.asarray(params))))
    assert not (finite and bool(state.stats.converged))


# Constants of the backtracking search ``Newton.__init__`` builds: ``max_backtracking_steps``
# is set there, the rest are ``optax.scale_by_backtracking_linesearch`` defaults. The
# reference below reproduces the algorithm from them, so a change on either side shows up
# as a disagreement rather than as a silently different search.
_SLOPE_RTOL = 1e-4
_DECREASE_FACTOR = 0.8
_INCREASE_FACTOR = 1.5
_MAX_LEARNING_RATE = 1.0
_MAX_BACKTRACKING_STEPS = 30

# A point with both zero and non-zero coefficients: the zeros are where the penalty is
# not differentiable, which is the whole reason the composite slope is needed.
_KINKED_PARAMS = np.array([0.7, 0.0, -0.4, 0.0, 0.25, 0.1])
_PENALTY_STRENGTH = 0.01

# Three groups over the six coefficients, so a group straddles a zero and a non-zero one.
_GROUP_MASK = np.zeros((3, _KINKED_PARAMS.size))
_GROUP_MASK[0, :2] = 1.0
_GROUP_MASK[1, 2:4] = 1.0
_GROUP_MASK[2, 4:] = 1.0


def _mse_grad(params, X, y):
    """Gradient of ``_mse``, differentiated by hand rather than taken from the solver."""
    return (-2.0 / X.shape[0]) * X.T @ (y - X @ np.asarray(params))


def _l1_penalty(params):
    return _PENALTY_STRENGTH * np.abs(np.asarray(params)).sum()


def _group_l2_penalty(params):
    """``strength * sum_j sqrt(dim(beta_j)) ||beta_j||_2``, the formula ``GroupLasso`` documents."""
    params = np.asarray(params)
    return _PENALTY_STRENGTH * sum(
        np.sqrt(group.sum()) * np.linalg.norm(params[group.astype(bool)])
        for group in _GROUP_MASK
    )


def _step_off_a_zero(params):
    """A unit step along the first zero coefficient, where the penalty has its kink."""
    step = np.zeros_like(params)
    step[np.flatnonzero(np.asarray(params) == 0.0)[0]] = 1.0
    return step


# Both nonsmooth penalties allowed for ``ProximalNewton``, each with the penalty written
# out in numpy so the expected slope does not come from the object under test. The
# regularizers are built inside the test: a ``GroupLasso`` constructed at collection time
# stores a float32 mask, since ``requires_x64`` has not switched precision on yet.
_PENALTY_CASES = [
    pytest.param(Lasso, _l1_penalty, id="Lasso"),
    pytest.param(
        lambda: GroupLasso(mask=_GROUP_MASK), _group_l2_penalty, id="GroupLasso"
    ),
]

# Steps chosen for the sign of ``P(b + d) - P(b)`` they force, since that difference is
# the term the composite slope adds. The tiny variants make ``penalty_diff / ||d||^2``
# large, which is where the encoding could lose the term to cancellation, and the null
# step is the ``sq_norm > 0`` guard.
_STEP_CASES = [
    pytest.param(lambda b: -b, -1.0, id="to_the_origin"),
    pytest.param(lambda b: -1e-6 * b, -1.0, id="to_the_origin_tiny"),
    pytest.param(_step_off_a_zero, 1.0, id="off_a_zero_coefficient"),
    pytest.param(
        lambda b: 1e-6 * _step_off_a_zero(b), 1.0, id="off_a_zero_coefficient_tiny"
    ),
    pytest.param(np.zeros_like, 0.0, id="null_step"),
]


def _line_search_inputs_at(regularizer, strength, params, step, X, y):
    """The gradient at ``params`` and the line-search inputs built from it."""
    solver = ProximalNewton(
        _mse,
        regularizer=regularizer,
        regularizer_strength=strength,
        has_aux=False,
        init_params=params,
        tol=1e-12,
    )
    solver.init_state(params, X, y)
    (fval, _), grad = solver._gradient(params, X, y)
    return grad, solver._line_search_inputs(params, step, grad, fval, X, y)


@pytest.mark.parametrize("make_regularizer, penalty", _PENALTY_CASES)
@pytest.mark.parametrize("make_step, penalty_change", _STEP_CASES)
@pytest.mark.requires_x64
def test_prox_newton_line_search_slope_is_the_composite_delta(
    make_regularizer, penalty, make_step, penalty_change
):
    """Contracted with the step, the slope handed to ``optax`` is the Tseng & Yun (2009) ``Delta``.

    ``optax`` forms ``vdot(step, slope)`` and never differentiates ``slope``, so the
    override encodes ``Delta = grad f^T d + P(b + d) - P(b)`` by adding
    ``[P(b + d) - P(b)] / ||d||^2 * d`` to the gradient. It is ``Delta``, not
    ``grad f^T d``, that certifies descent of the nonsmooth objective, so the identity is
    what makes the stock Armijo search correct here.

    Every reference is numpy: the penalty formulas, the hand-differentiated gradient of
    ``_mse`` and the composite value. Worst measured relative error over these cases is
    1.4e-11, on the tiny steps where the ``1/||d||^2`` scaling costs digits; the 1e-9
    bound is ~70x above it.
    """
    np.random.seed(0)
    X = np.random.normal(size=(200, _KINKED_PARAMS.size))
    y = np.random.normal(size=200)
    params = jnp.asarray(_KINKED_PARAMS)
    step = make_step(_KINKED_PARAMS)

    _, (value, slope, value_fn) = _line_search_inputs_at(
        make_regularizer(), _PENALTY_STRENGTH, params, jnp.asarray(step), X, y
    )

    penalty_diff = penalty(_KINKED_PARAMS + step) - penalty(_KINKED_PARAMS)
    assert np.sign(penalty_diff) == penalty_change, (
        "the step must move the penalty in the parametrized direction, otherwise the "
        "case tests the smooth slope only"
    )

    smooth = float(np.mean((y - X @ _KINKED_PARAMS) ** 2))
    np.testing.assert_allclose(value, smooth + penalty(_KINKED_PARAMS), rtol=1e-12)

    expected = _mse_grad(_KINKED_PARAMS, X, y) @ step + penalty_diff
    np.testing.assert_allclose(
        lx.internal.tree_dot(slope, jnp.asarray(step)), expected, rtol=1e-9, atol=1e-15
    )
    # the null step divides 0 by 0 unless the guard holds
    assert np.all(np.isfinite(np.asarray(slope)))

    moved = _KINKED_PARAMS + step
    np.testing.assert_allclose(
        value_fn(jnp.asarray(moved)),
        float(np.mean((y - X @ moved) ** 2)) + penalty(moved),
        rtol=1e-12,
    )


@pytest.mark.requires_x64
def test_prox_newton_line_search_slope_is_the_gradient_when_unpenalized():
    """With ``P = 0`` the added term vanishes and the search gets the plain smooth inputs.

    The penalty difference is identically zero, so the carrier vector must reduce to the
    gradient exactly -- a spurious term here would push a correct Armijo search off a
    smooth objective it already handles.
    """
    np.random.seed(0)
    X = np.random.normal(size=(200, _KINKED_PARAMS.size))
    y = np.random.normal(size=200)
    params = jnp.asarray(_KINKED_PARAMS)
    step = jnp.asarray(-_KINKED_PARAMS)

    grad, (value, slope, value_fn) = _line_search_inputs_at(
        UnRegularized(), None, params, step, X, y
    )

    np.testing.assert_array_equal(np.asarray(slope), np.asarray(grad))
    np.testing.assert_allclose(
        value, float(np.mean((y - X @ _KINKED_PARAMS) ** 2)), rtol=1e-12
    )
    np.testing.assert_allclose(
        value_fn(params + step), float(np.mean(y**2)), rtol=1e-12
    )


def _tseng_yun_backtracking(objective, params, step, delta, prev_stepsize):
    """Armijo backtracking on a composite objective, Tseng & Yun (2009), in plain numpy.

    Tries ``a = min(1.5 * a_prev, 1) * 0.8^k`` for ``k = 0, 1, ...`` and accepts the first
    satisfying ``F(b + a d) <= F(b) + c a Delta``; ``a = 0`` if the objective is not a
    number, and the last ``a`` tried if the budget runs out. Returns the stepsize and the
    number of objective evaluations, so a run matches the reference step by step and not
    only at its endpoint.
    """
    value = objective(params)
    stepsize = min(_INCREASE_FACTOR * prev_stepsize, _MAX_LEARNING_RATE)
    error, evaluations = np.inf, 0
    for iter_num in range(_MAX_BACKTRACKING_STEPS + 1):
        if error <= 0.0:
            break
        if iter_num > 0:
            stepsize *= _DECREASE_FACTOR
        residual = (
            objective(params + stepsize * step) - value - stepsize * _SLOPE_RTOL * delta
        )
        error = np.inf if np.isnan(residual) else max(residual, 0.0)
        evaluations += 1
    return (0.0 if np.isinf(error) else stepsize), evaluations


@pytest.mark.parametrize("start", [np.zeros_like(_KINKED_PARAMS), _KINKED_PARAMS])
@pytest.mark.parametrize("scale", [1.0, 5.0, 50.0])
@pytest.mark.parametrize("prev_stepsize", [1.0, 0.4])
@pytest.mark.requires_x64
def test_prox_newton_backtracking_matches_tseng_yun_reference(
    start, scale, prev_stepsize
):
    """The optax search, fed the rewritten inputs, is the Tseng & Yun search written directly.

    ``_apply_or_reject`` is compared against ``_tseng_yun_backtracking``, which owes optax
    nothing but the four constants: same accepted stepsize, same number of objective
    evaluations, same iterate. ``scale`` inflates the proximal Newton direction past the
    minimizer of the local model so the full step is rejected and the loop actually
    backtracks -- 6 and 16 evaluations at 5x and 50x against 1 at the Newton step --
    and ``prev_stepsize`` exercises the warm start, whose candidate ``min(1.5 a, 1)``
    saturates at 1.0 and does not at 0.4.

    The comparison is bit-exact in practice (stepsizes equal, iterates within 1.1e-16)
    because the accepted step clears the sufficient-decrease test by 3e-3 to 5e-1 here,
    far above the precision at which the two objectives could disagree. It also has teeth
    on the composite value function: passing the smooth loss alone in its place accepts
    0.4096 rather than 0.32768 at ``scale=5``.
    """
    np.random.seed(0)
    X = np.random.normal(size=(200, _KINKED_PARAMS.size))
    y = np.random.normal(size=200)
    params = jnp.asarray(start)

    def objective(coef):
        coef = np.asarray(coef)
        return float(np.mean((y - X @ coef) ** 2) + _l1_penalty(coef))

    solver = ProximalNewton(
        _mse,
        regularizer=Lasso(),
        regularizer_strength=_PENALTY_STRENGTH,
        has_aux=False,
        init_params=params,
        tol=1e-12,
    )
    state = solver.init_state(params, X, y)
    # the search reads its previous stepsize off the state; setting it directly keeps the
    # warm start a parametrized axis instead of a by-product of a trajectory
    state = eqx.tree_at(
        lambda s: s.ls_state.learning_rate, state, jnp.asarray(prev_stepsize)
    )

    (fval, _), grad = solver._gradient(params, X, y)
    H = solver._hessian(params, X, y)
    step = jax.tree.map(lambda d: scale * d, solver._newton_direction(grad, H, params))
    _, slope, _ = solver._line_search_inputs(params, step, grad, fval, X, y)
    delta = float(lx.internal.tree_dot(slope, step))
    assert delta < 0.0, "the reference only terminates on a descent direction"

    new_params, ls_state = solver._apply_or_reject(
        params, step, grad, state, fval, X, y
    )
    expected_stepsize, expected_evaluations = _tseng_yun_backtracking(
        objective, start, np.asarray(step), delta, prev_stepsize
    )

    np.testing.assert_allclose(ls_state.learning_rate, expected_stepsize, rtol=1e-12)
    assert int(ls_state.info.num_linesearch_steps) == expected_evaluations
    np.testing.assert_allclose(
        new_params, start + expected_stepsize * np.asarray(step), rtol=1e-12, atol=1e-15
    )
    assert objective(new_params) < objective(start)


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
    (fval, _), grad = solver._gradient(params, X, y)
    step = make_step(grad)

    _, slope, _ = solver._line_search_inputs(params, step, grad, fval, X, y)
    descent = float(lx.internal.tree_dot(slope, step))
    assert np.sign(descent) == slope_sign

    new_params, new_ls_state = solver._apply_or_reject(
        params, step, grad, state, fval, X, y
    )

    if slope_sign < 0:
        assert not np.allclose(new_params, params), "a descent step must be taken"
    else:
        np.testing.assert_array_equal(np.asarray(new_params), np.asarray(params))
        # the rejected branch returns the state untouched, so the next iteration
        # restarts the search from the same stepsize
        np.testing.assert_array_equal(
            np.asarray(new_ls_state.learning_rate),
            np.asarray(state.ls_state.learning_rate),
        )


@pytest.mark.requires_x64
def test_prox_newton_does_not_read_a_nan_slope_as_stationary():
    """A blown-up direction must reach the iterate, not be rejected as a null step.

    Rejection leaves ``params`` where they were, so ``y_diff`` is zero and the next
    Cauchy test reports convergence -- the failure would be announced as success. This is
    the ``jnp.isnan`` half of the gate in ``_apply_or_reject``, and it is reachable: with
    an indefinite Hessian the subproblem is unbounded, and the iterates overflow to NaN
    through a sequence of perfectly good descent directions.
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
    (fval, _), grad = solver._gradient(params, X, y)
    step = jax.tree.map(lambda g: jnp.full_like(g, jnp.nan), grad)

    _, slope, _ = solver._line_search_inputs(params, step, grad, fval, X, y)
    assert np.isnan(float(lx.internal.tree_dot(slope, step)))

    new_params, _ = solver._apply_or_reject(params, step, grad, state, fval, X, y)
    assert not np.all(np.isfinite(np.asarray(new_params)))
