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
from conftest import all_subclasses, initialize_feature_mask_for_population_glm
from nemos._inspect_utils import is_abstract
from nemos.base_regressor import BaseRegressor
from nemos.glm import GLM, PopulationGLM
from nemos.glm.classifier_glm import ClassifierGLM, ClassifierPopulationGLM
from nemos.glm.params import GLMParams
from nemos.regularizer import Regularizer, Ridge, UnRegularized
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.solvers._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    NegativeDefinite,
    PositiveDefinite,
    PositiveSemiDefinite,
    Symmetric,
    _expand_property,
    weaken_property,
)
from nemos.solvers._newton import Newton, NewtonState
from nemos.tree_utils import pytree_map_and_reduce

# Import every submodule so all BaseRegressor subclasses are registered before the
# parametrizations below are collected (same idiom as test_model_params).
for _, _modname, _ in pkgutil.walk_packages(nmo.__path__, prefix="nemos."):
    importlib.import_module(_modname)

# Register every test here as solver-related
pytestmark = pytest.mark.solver_related


def _newton_regularizers():
    """Auto-discover every regularizer that advertises Newton as an allowed solver.

    The block-diagonal Hessian equals the full Hessian's diagonal blocks only if the
    penalty is additive (so the Hessian factorizes into loss + penalty terms). Additivity
    is currently baked into ``Regularizer.penalized_loss``; parametrizing over the
    discovered set means a future Newton-eligible regularizer that breaks additivity is
    caught here instead of silently mis-regularizing the Newton step.
    """

    return sorted(
        (
            cls
            for cls in all_subclasses(Regularizer)
            if cls.__module__.startswith("nemos")
            and "Newton" in getattr(cls, "_allowed_solvers", ())
        ),
        key=lambda cls: cls.__name__,
    )


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
            and getattr(cls, "_hess_tag", None) is not None
            and cls._hess_tag.structure is BlockDiagonal
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


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_instantiate_solver(regularizer_name, glm_class):
    glm = glm_class(
        regularizer=regularizer_name,
        solver_name="Newton",
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    solver = glm._instantiate_solver(glm._compute_loss, np.zeros(1))

    # currently glm._solver is a Wrapped(Prox)SVRG
    assert glm.solver_name == "Newton"
    assert isinstance(solver, Newton)


def _freeze_first_coef_leaf(model, params):
    """Build a ``_fix_params`` spec pinning the first ``coef`` leaf to its true value.

    Freezing a coefficient leaf rather than the intercept is what exercises the
    ``coef`` block of the Hessian: the intercept is a single row, so dropping it
    leaves the interesting block untouched.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params.coef)
    # cast the pinned leaf: ``fit`` casts its inputs, so an un-cast spec would be
    # compared against a fitted leaf of a different dtype
    spec = jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(leaves[0])] + [None] * (len(leaves) - 1)
    )
    model._fix_params = GLMParams(spec, None)
    return spec


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize(
    "model_fixture",
    ["poissonGLM_model_instantiation", "population_poissonGLM_model_instantiation"],
)
@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_newton_matches_first_order_solver_with_frozen_params(
    regularizer_name, model_fixture, freeze, request
):
    """Newton differentiates the combined loss with respect to the active subtree,
    so it must land on the same optimum as a partition-agnostic first-order solver.

    Both freezing modes are covered: a frozen intercept drops a Hessian row, while a
    frozen ``coef`` leaf carves a block out of the ``coef`` block itself.
    """
    X, y, model, true_params, _ = request.getfixturevalue(model_fixture)

    def build(solver_name):
        m = type(model)(
            regularizer=regularizer_name,
            regularizer_strength=1.0 if regularizer_name == "Ridge" else None,
            solver_name=solver_name,
            solver_kwargs={"tol": 10**-12},
        )
        if freeze == "intercept":
            m.fit_intercept = False
        else:
            _freeze_first_coef_leaf(m, true_params)
        return m

    newton = build("Newton").fit(X, y)
    reference = build("LBFGS").fit(X, y)

    assert newton.solver_name == "Newton"
    np.testing.assert_allclose(newton.coef_, reference.coef_, atol=1e-5)
    np.testing.assert_allclose(newton.intercept_, reference.intercept_, atol=1e-5)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize(
    "model_fixture",
    ["poissonGLM_model_instantiation", "population_poissonGLM_model_instantiation"],
)
@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_newton_leaves_frozen_params_untouched(
    regularizer_name, model_fixture, freeze, request
):
    """The frozen leaves come back bit-identical, not merely close: a Hessian that
    silently included them would move them by a small but nonzero amount."""
    X, y, model, true_params, _ = request.getfixturevalue(model_fixture)

    frozen_model = type(model)(
        regularizer=regularizer_name,
        regularizer_strength=1.0 if regularizer_name == "Ridge" else None,
        solver_name="Newton",
    )
    if freeze == "intercept":
        frozen_model.fit_intercept = False
        frozen_model.fit(X, y)
        np.testing.assert_array_equal(
            frozen_model.intercept_, np.zeros_like(true_params.intercept)
        )
    else:
        pinned = _freeze_first_coef_leaf(frozen_model, true_params)
        # the spec decides which leaves the solver skips, not what they are worth: a
        # frozen leaf is held at whatever the initialization put there. The intercept
        # is filled from the spec at init time, a coef leaf is not, so the pinned value
        # is handed in through ``init_params`` for the fitted leaf to be comparable.
        zeros = jax.tree_util.tree_map(jnp.zeros_like, true_params.coef)
        init_params = (
            eqx.combine(pinned, zeros),
            jnp.zeros_like(true_params.intercept),
        )
        frozen_model.fit(X, y, init_params=init_params)
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
@pytest.mark.parametrize("regularizer_name", ["Ridge"])
@pytest.mark.parametrize("freeze", ["intercept", "coef_leaf"])
def test_newton_population_glm_feature_mask_with_frozen_params(
    regularizer_name, freeze, request
):
    """The mask, the active axes and the frozen axes all index the neuron axis of the
    block Hessian. A wrong ``in_axes`` survives the unmasked tests, so pair the mask
    with a freeze and check Newton still lands where a first-order solver does.

    Ridge only: a masked-out coefficient has zero gradient *and* zero curvature, so an
    unpenalized masked Hessian is singular and the Newton step is NaN. That is unrelated
    to parameter freezing (it reproduces with nothing frozen) and is tracked in #580.
    """
    X, y, model, true_params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation_pytree"
    )
    mask = initialize_feature_mask_for_population_glm(X, y.shape[1])
    # zero a block so the mask is not the identity: a masked-out coefficient must not
    # pick up curvature from the neuron it is masked away from
    first = sorted(mask)[0]
    mask[first] = mask[first].at[:, 0].set(0.0)

    def build(solver_name):
        m = nmo.glm.PopulationGLM(
            regularizer=regularizer_name,
            regularizer_strength=1.0 if regularizer_name == "Ridge" else None,
            solver_name=solver_name,
            solver_kwargs={"tol": 10**-12},
            feature_mask=mask,
        )
        if freeze == "intercept":
            m.fit_intercept = False
        else:
            _freeze_first_coef_leaf(m, true_params)
        return m

    newton = build("Newton").fit(X, y)
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
        _freeze_first_coef_leaf(model, true_params)

    params = model._model_specific_initialization(X, y)
    active, frozen = model._partition_active(params)
    hess = model._get_hess_fn(frozen=frozen)(active, X, y)

    n_neurons = y.shape[1]
    if freeze == "intercept":
        assert hess.intercept is None
        assert all(row.intercept is None for row in hess.coef.values())
    else:
        pinned = sorted(model._fix_params.coef)[0]
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
def test_newton_glm_passes_solver_kwargs(regularizer_name, glm_class):
    solver_kwargs = {
        "maxiter": np.random.randint(1, 100),
        "jit": False,
        "tol": 1e-6,
    }

    glm = glm_class(
        regularizer=regularizer_name,
        solver_name="Newton",
        solver_kwargs=solver_kwargs,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    solver = glm._instantiate_solver(glm._compute_loss, np.zeros(1))

    for k, v in solver_kwargs.items():
        assert getattr(solver, k) == v


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_initialize_state(glm_class, regularizer_name, linear_regression):
    X, y, _, _, _ = linear_regression

    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    reg_cls = getattr(nmo.regularizer, regularizer_name)
    reg = reg_cls()

    glm = glm_class(
        regularizer=reg,
        solver_name="Newton",
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
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_glm_converges(request, regularizer_cls, structure):
    """Newton-fitted GLM should converge and return finite parameters."""
    X, y, model, _, _ = request.getfixturevalue(
        "poissonGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_glm_converges(request, regularizer_cls, feature_mask):
    """Newton-fitted PopulationGLM should converge and return finite parameters."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3

    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_glm_matches_full_autodiff(request, feature_mask):
    """Newton-fitted PopulationGLM should match a full autodiff model that does not vmap over subproblems."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model.regularizer = "Ridge"
    model.regularizer_strength = 0.1
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = deepcopy(model)
    full_model._get_hess_fn = lambda frozen=None: None
    full_model._hess_tag = HessianTag(structure=Full, property=PositiveSemiDefinite)

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
@_STRENGTHS
@pytest.mark.parametrize("feature_mask", [True, False])
@pytest.mark.parametrize("fixture_name", _BLOCK_MODEL_CASES)
def test_newton_block_diagonal_matches_full_autodiff_update(
    request, fixture_name, feature_mask, make_strength
):
    """One Newton update() on the block Hessian must match a full autodiff model.

    Runs for every model declaring a block-diagonal Hessian, in both ``coef`` layouts and
    under a scalar and a per-neuron strength. The block path vmaps the regularizer's penalty
    Hessian over neurons, so a mismatch between the model's ``batch_axes`` and the strength
    surfaces here and nowhere else: the scalar strength carries no neuron axis to get wrong.
    """
    X, y, model, params, _ = request.getfixturevalue(fixture_name)
    model.regularizer = "Ridge"
    model.solver_name = "Newton"
    model.regularizer_strength = make_strength(params.coef)
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = deepcopy(model)
    full_model._get_hess_fn = lambda frozen=None: None
    full_model._hess_tag = HessianTag(structure=Full, property=PositiveSemiDefinite)

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
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_glm_block_hessian_matches_full(
    request, feature_mask, regularizer_cls
):
    """
    The vmapped per-neuron Hessian should equal the diagonal neuron-blocks of the
    full autodiff Hessian, and the full Hessian should be block-diagonal across neurons.

    Both Hessians are rendered as dense matrices (per neuron) via a flatten/unflatten of
    the parameter pytree, so the comparison is on the actual matrices the Newton solve
    consumes. Parametrized over every Newton-eligible regularizer: the block/full match
    holds only for additive penalties, so a non-additive one would fail here.
    """
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = None if regularizer_cls is UnRegularized else 0.1
    model.solver_name = "Newton"
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = deepcopy(model)
    full_model._get_hess_fn = lambda frozen=None: None
    full_model._hess_tag = HessianTag(structure=Full, property=PositiveSemiDefinite)

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
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_classifier_glm_converges(request, regularizer_cls, structure):
    """Newton-fitted ClassifierGLM should converge and return finite parameters."""
    X, y, model, _, _ = request.getfixturevalue(
        "classifierGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_classifier_population_glm_converges(
    request, regularizer_cls, feature_mask
):
    """Newton-fitted ClassifierPopulationGLM should converge and return finite parameters."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation"
    )
    model.regularizer = regularizer_cls()
    model.regularizer_strength = 1e-3
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_classifier_glm_matches_full_autodiff(request, feature_mask):
    """Newton-fitted ClassifierPopulationGLM should match a full autodiff model that does not vmap over subproblems."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation"
    )
    model.regularizer = "Ridge"
    model.regularizer_strength = 0.1
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = deepcopy(model)
    full_model._get_hess_fn = lambda frozen=None: None
    full_model._hess_tag = HessianTag(structure=Full, property=PositiveSemiDefinite)

    full_model.fit(X, y)
    model.fit(X, y)
    np.testing.assert_allclose(full_model.coef_, model.coef_, atol=1e-3)


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
@pytest.mark.parametrize("feature_mask", [True, False])
def test_newton_population_classifier_glm_block_hessian_matches_full(
    request, feature_mask, regularizer_cls
):
    """The vmapped per-neuron Hessian should equal the diagonal neuron-blocks of the
    full autodiff Hessian, and the full Hessian should be block-diagonal across neurons.

    Both Hessians are rendered as dense matrices (per neuron) via a flatten/unflatten of
    the parameter pytree, so the comparison is on the actual matrices the Newton solve
    consumes. Parametrized over every Newton-eligible regularizer: the block/full match
    holds only for additive penalties, so a non-additive one would fail here.

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
    model.regularizer_strength = None if regularizer_cls is UnRegularized else 0.1
    model.solver_name = "Newton"
    if feature_mask:
        model._feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )

    full_model = deepcopy(model)
    full_model._get_hess_fn = lambda frozen=None: None
    full_model._hess_tag = HessianTag(structure=Full, property=PositiveSemiDefinite)

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

    Every in-tree model that overrides ``_get_hess_fn`` is tagged ``BlockDiagonal``
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
@pytest.mark.parametrize("regularizer_cls", _newton_regularizers())
def test_newton_unbatched_model_hessian_includes_penalty(request, regularizer_cls):
    """A model-supplied Hessian that is not block-diagonal must still get the penalty added."""
    X, y, model, _, _ = request.getfixturevalue("poissonGLM_model_instantiation")
    model = _FullHessianGLM(
        observation_model=model.observation_model,
        regularizer=regularizer_cls(),
        regularizer_strength=0.1,
        solver_name="Newton",
    )

    p0 = model.initialize_params(X, y)
    model.initialize_optimizer_and_state(p0, X, y)
    p = GLMParams(*p0)

    # the model contributes the likelihood term only, so the Hessian the solve consumes must
    # equal the autodiff Hessian of the penalized loss the solver actually minimizes
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


def test_glm_hess_tag_structure():
    """GLM Hessian should have Full structure."""
    assert GLM._hess_tag.structure is Full


def test_glm_hess_tag_property_unregularized():
    """Unpenalized GLM Hessian is only positive semidefinite."""
    assert GLM._hess_tag.property is PositiveSemiDefinite


def test_population_glm_hess_tag_structure():
    """PopulationGLM Hessian should have BlockDiagonal structure."""
    assert PopulationGLM._hess_tag.structure is BlockDiagonal


def test_population_glm_hess_tag_property():
    """PopulationGLM certifies only PositiveSemiDefinite at the class level, like GLM
    and ClassifierPopulationGLM: the unpenalized block is singular whenever a
    coefficient carries no curvature (a masked-out feature). Ridge promotes it to
    PositiveDefinite via ``_hess_property_override``, checked just below."""
    assert PopulationGLM._hess_tag.property is PositiveSemiDefinite
    assert PopulationGLM._hess_tag.property is GLM._hess_tag.property


def test_population_glm_hess_tag_batch_axes():
    """PopulationGLM _hess_tag should carry per-neuron batch_axes as a GLMParams."""
    from nemos.glm.params import GLMParams

    tag = PopulationGLM._hess_tag
    assert tag.batch_axes is not None
    assert isinstance(tag.batch_axes, GLMParams)


@pytest.mark.parametrize("glm_class", [GLM, PopulationGLM])
def test_hess_property_override(glm_class):
    """Ridge-penalized GLM should override Hessian property to PositiveDefinite."""
    model = glm_class(regularizer="Ridge", regularizer_strength=0.1)
    assert model._hess_property_override() is PositiveDefinite


@pytest.mark.parametrize("glm_class", [ClassifierGLM, ClassifierPopulationGLM])
def test_hess_property_override_classification(glm_class):
    """Ridge-penalized GLM should override Hessian property to PositiveDefinite."""
    model = glm_class(regularizer="Ridge", regularizer_strength=0.1)
    assert model._hess_property_override() is None


@pytest.mark.parametrize(
    "regularizer_name",
    ["UnRegularized", "Lasso", "GroupLasso"],
)
def test_hess_property_override_non_ridge(regularizer_name):
    """Non-Ridge GLMs should return None from _hess_property_override."""
    strength = 0.1 if regularizer_name != "UnRegularized" else None
    model = GLM(regularizer=regularizer_name, regularizer_strength=strength)
    assert model._hess_property_override() is None


@pytest.mark.parametrize(
    "prop, expected",
    [
        (PositiveDefinite, PositiveSemiDefinite),
        (PositiveSemiDefinite, PositiveSemiDefinite),
        (NegativeDefinite, Symmetric),
        (Symmetric, Symmetric),
        (General, General),
    ],
)
def test_weaken_property(prop, expected):
    """Zeroing out directions costs strictness and keeps the sign."""
    assert weaken_property(prop) is expected


@pytest.mark.parametrize(
    "prop",
    [PositiveDefinite, PositiveSemiDefinite, NegativeDefinite, Symmetric, General],
)
def test_weaken_property_is_idempotent(prop):
    """Nothing is left to weaken after one pass: no strictness survives it."""
    assert weaken_property(weaken_property(prop)) is weaken_property(prop)


@pytest.mark.parametrize(
    "prop",
    [PositiveDefinite, PositiveSemiDefinite, NegativeDefinite, Symmetric, General],
)
def test_weaken_property_implied_by_original(prop):
    """The weakened property must be one the original already implies."""
    assert weaken_property(prop) in _expand_property(prop)


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


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_solver_name_respected_when_explicitly_set(glm_class):
    """Explicitly setting solver_name='Newton' should be respected."""
    model = glm_class(regularizer="UnRegularized", solver_name="Newton")
    assert model.solver_name == "Newton"


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_solver_type_after_fit(request, model_instantiation_type):
    """After fit(), model._solver should be a Newton instance."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.regularizer = "Ridge"
    model.fit(X, y)
    from nemos.solvers._newton import Newton

    assert isinstance(model._solver, Newton)


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_update_increments_step_count(request, model_instantiation_type):
    """Each call to update() should increment the step counter by exactly 1."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    init_params = model.initialize_params(X, y)
    state = model.initialize_optimizer_and_state(init_params, X, y)
    assert state.stats.num_steps == 0

    _, state1 = model.update(init_params, state, X, y)
    assert state1.stats.num_steps == 1

    _, state2 = model.update(model.get_model_params(), state1, X, y)
    assert state2.stats.num_steps == 2


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_maxiter_respected(request, model_instantiation_type):
    """Setting maxiter=1 should bound the solver to at most 1 step."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.regularizer = "Ridge"
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)

    n_steps = int(model.solver_state_.stats.num_steps)
    assert n_steps <= 1, f"Expected at most 1 step, got {n_steps}"


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_newton_invalid_kwarg_raises(glm_class):
    """Passing an unrecognised kwarg should raise a NameError immediately."""
    with pytest.raises(NameError, match="not a kwarg"):
        glm_class(
            regularizer="Ridge",
            regularizer_strength=0.1,
            solver_name="Newton",
            solver_kwargs={"totally_fake_kwarg": 99},
        )
