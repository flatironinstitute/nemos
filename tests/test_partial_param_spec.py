"""Tests for the partial parameter specification (``fix_params``) machinery."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from sklearn.base import clone

from nemos.batching import ArrayDataLoader
from nemos.callbacks import Callback
from nemos.glm.params import GLMParams
from nemos.solvers._abstract_solver import AbstractSolver
from nemos.solvers._no_op import NoOpSolver
from nemos.tree_utils import tree_broadcast_prefix


class _RecordingCallback(Callback):
    """Records which hooks the solver invoked, in order."""

    def __init__(self):
        self.calls = []

    def on_train_begin(self, ctx):
        self.calls.append("on_train_begin")

    def on_train_end(self, ctx):
        self.calls.append("on_train_end")

    def on_pass_begin(self, ctx):
        self.calls.append("on_pass_begin")

    def on_pass_end(self, ctx):
        self.calls.append("on_pass_end")

    def on_batch_begin(self, ctx):
        self.calls.append("on_batch_begin")

    def on_batch_end(self, ctx):
        self.calls.append("on_batch_end")


# conftest fixtures, picked to vary the coef pytree structure and the intercept shape
MODEL_FIXTURES = [
    "poissonGLM_model_instantiation",  # coef: one array
    "poissonGLM_model_instantiation_pytree",  # coef: dict of two arrays
    "population_poissonGLM_model_instantiation",  # 2-D coef, (n_neurons,) intercept
    "population_poissonGLM_model_instantiation_pytree",
    "classifierGLM_model_instantiation",
]

COEF_MODES = ["learn_all", "pin_all", "pin_first"]


def _coef_spec(model, X, y, mode):
    """Coef spec mirroring ``X``: ``None`` learns a leaf, an array pins it."""
    if mode == "learn_all":
        return None

    empty_coef = model._validator.get_empty_params(X, y).coef
    if mode == "pin_all":
        return jax.tree_util.tree_map(jnp.ones_like, empty_coef)

    seen = []

    def pin_first(leaf):
        seen.append(None)
        return jnp.ones_like(leaf) if len(seen) == 1 else None

    return jax.tree_util.tree_map(pin_first, empty_coef)


def _pinned_intercept(model, X, y):
    """A pinned intercept of the shape this model expects."""
    return jnp.full_like(model._validator.get_empty_params(X, y).intercept, 0.5)


def _leaves(tree):
    """Leaves of ``tree``, treating ``None`` as a leaf rather than an empty node."""
    return jax.tree_util.tree_leaves(tree, is_leaf=lambda x: x is None)


def _configure(model, X, y, coef_mode, intercept_pinned, fit_intercept):
    """Apply a spec through the public setters, returning the model."""
    intercept = _pinned_intercept(model, X, y) if intercept_pinned else None
    model.fix_params = (_coef_spec(model, X, y, coef_mode), intercept)
    model.fit_intercept = fit_intercept
    return model


@pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
@pytest.mark.parametrize("coef_mode", COEF_MODES)
@pytest.mark.parametrize("intercept_pinned", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
class TestSpecAlgebra:
    """``_active_filter_spec`` / ``_frozen_values`` / ``_partition_active``."""

    def test_active_spec_matches_fix_params(
        self, request, model_fixture, coef_mode, intercept_pinned, fit_intercept
    ):
        """A coef leaf is active iff the user left it ``None`` in the spec."""
        X, y, model = request.getfixturevalue(model_fixture)[:3]
        _configure(model, X, y, coef_mode, intercept_pinned, fit_intercept)

        active = model._active_filter_spec()
        if coef_mode == "learn_all":
            # a wholly-unset spec collapses to the single prefix value ``True``
            assert all(bool(leaf) for leaf in _leaves(active.coef))
        else:
            expected = jax.tree_util.tree_map(
                lambda spec: spec is None,
                model.fix_params[0],
                is_leaf=lambda x: x is None,
            )
            assert _leaves(active.coef) == _leaves(expected)

    def test_intercept_active_only_when_free_and_unpinned(
        self, request, model_fixture, coef_mode, intercept_pinned, fit_intercept
    ):
        """Either mechanism freezes the intercept; it is active only if neither fires."""
        X, y, model = request.getfixturevalue(model_fixture)[:3]
        _configure(model, X, y, coef_mode, intercept_pinned, fit_intercept)

        expected = fit_intercept and not intercept_pinned
        assert bool(model._active_filter_spec().intercept) is expected

    def test_filter_spec_and_frozen_values_are_complements(
        self, request, model_fixture, coef_mode, intercept_pinned, fit_intercept
    ):
        """A leaf is inactive iff ``_frozen_values`` holds a concrete array for it."""
        X, y, model = request.getfixturevalue(model_fixture)[:3]
        _configure(model, X, y, coef_mode, intercept_pinned, fit_intercept)

        # expand against params, not frozen: None leaves would swallow the broadcast
        params = model._validator.get_empty_params(X, y)
        active = tree_broadcast_prefix(model._active_filter_spec(), params)
        frozen = model._frozen_values(X, y)

        for is_active, frozen_leaf in zip(_leaves(active), _leaves(frozen)):
            assert bool(is_active) is (
                frozen_leaf is None
            ), f"active={is_active!r} but frozen={frozen_leaf!r}"

    def test_partition_recombines_exactly(
        self, request, model_fixture, coef_mode, intercept_pinned, fit_intercept
    ):
        """``_partition_active`` loses nothing: the halves recombine to the input."""
        X, y, model = request.getfixturevalue(model_fixture)[:3]
        _configure(model, X, y, coef_mode, intercept_pinned, fit_intercept)

        params = jax.tree_util.tree_map(
            jnp.ones_like, model._validator.get_empty_params(X, y)
        )
        recombined = eqx.combine(*model._partition_active(params))

        assert jax.tree_util.tree_structure(recombined) == jax.tree_util.tree_structure(
            params
        )
        for got, want in zip(_leaves(recombined), _leaves(params)):
            np.testing.assert_array_equal(got, want)


class TestInterceptPrecedence:
    """``fit_intercept=False`` and a pinned intercept are two paths to one frozen leaf."""

    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_pinned_intercept_wins_over_fit_intercept(
        self, poissonGLM_model_instantiation, fit_intercept
    ):
        """A pinned intercept is honoured whatever ``fit_intercept`` says."""
        X, y, model = poissonGLM_model_instantiation[:3]
        pinned = _pinned_intercept(model, X, y)
        model.fix_params = (None, pinned)
        model.fit_intercept = fit_intercept

        np.testing.assert_array_equal(model._frozen_values(X, y).intercept, pinned)

    def test_fit_intercept_false_pins_zero_when_unpinned(
        self, poissonGLM_model_instantiation
    ):
        """With nothing pinned, ``fit_intercept=False`` supplies the zero itself."""
        X, y, model = poissonGLM_model_instantiation[:3]
        model.fit_intercept = False

        frozen = model._frozen_values(X, y)
        np.testing.assert_array_equal(
            frozen.intercept, jnp.zeros_like(frozen.intercept)
        )
        assert bool(model._active_filter_spec().intercept) is False

    def test_default_freezes_nothing_and_warns_nothing(
        self, poissonGLM_model_instantiation, recwarn
    ):
        """The default combination leaves the intercept free and warns about nothing."""
        X, y, model = poissonGLM_model_instantiation[:3]

        assert model._frozen_values(X, y).intercept is None
        assert bool(model._active_filter_spec().intercept) is True
        assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []

    def test_flipping_fit_intercept_changes_the_partition(
        self, poissonGLM_model_instantiation
    ):
        """The two mechanisms stay consistent when the flag flips after construction."""
        X, y, model = poissonGLM_model_instantiation[:3]
        assert bool(model._active_filter_spec().intercept) is True

        model.fit_intercept = False
        frozen = model._frozen_values(X, y)
        assert bool(model._active_filter_spec().intercept) is False
        np.testing.assert_array_equal(
            frozen.intercept, jnp.zeros_like(frozen.intercept)
        )


@pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
@pytest.mark.parametrize("coef_mode", COEF_MODES)
@pytest.mark.parametrize("intercept_pinned", [False, True])
def test_fix_params_survives_save_load(
    request, tmp_path, model_fixture, coef_mode, intercept_pinned
):
    """``fix_params`` round-trips through ``.npz``, ``None`` leaves included."""
    X, y, model = request.getfixturevalue(model_fixture)[:3]
    _configure(model, X, y, coef_mode, intercept_pinned, fit_intercept=True)

    path = tmp_path / "model.npz"
    model.save_params(path)
    loaded = nmo.load_model(path)

    before, after = model.fix_params, loaded.fix_params
    is_none = lambda x: x is None
    assert jax.tree_util.tree_structure(
        after, is_leaf=is_none
    ) == jax.tree_util.tree_structure(before, is_leaf=is_none)
    for got, want in zip(_leaves(after), _leaves(before)):
        if want is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, want)


class CoefModule(eqx.Module):
    """Custom pytree node, standing in for a user-defined coef container."""

    a: jnp.ndarray
    b: object


def test_user_defined_node_spec_cannot_be_loaded(
    poissonGLM_model_instantiation, tmp_path
):
    """A spec built on a user-defined pytree node saves, then fails to load.

    Only nemos-native containers are rebuilt at load time, so a foreign node reaches the
    archive as an object array. Saving still succeeds: unserializable pieces are meant to
    be supplied back through ``mapping_dict``, which accepts callables and classes only
    and so cannot carry a parameter spec.
    """
    X, y, model = poissonGLM_model_instantiation[:3]
    model.fix_params = (CoefModule(jnp.ones((3,)), None), None)

    path = tmp_path / "model.npz"
    model.save_params(path)

    with pytest.raises(ValueError, match="allow_pickle=False"):
        nmo.load_model(path)


def _pin_everything(model, X, y):
    """A ``fix_params`` spec leaving nothing active."""
    empty = model._validator.get_empty_params(X, y)
    return (
        jax.tree_util.tree_map(jnp.ones_like, empty.coef),
        jnp.full_like(empty.intercept, 0.25),
    )


def _convergence_warnings(recwarn):
    return [w for w in recwarn if "did not converge" in str(w.message)]


class TestEveryParameterFixed:
    """Entry points return the pinned values rather than driving an empty solver."""

    @pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
    def test_fit_returns_the_pinned_values(self, request, model_fixture):
        """``fit`` skips the solver and reports the fixed values."""
        X, y, model = request.getfixturevalue(model_fixture)[:3]
        coef, intercept = _pin_everything(model, X, y)
        model.fix_params = (coef, intercept)

        with pytest.warns(UserWarning, match="Every parameter is fixed"):
            model.fit(X, y)

        for got, want in zip(_leaves(model.coef_), _leaves(coef)):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(model.intercept_, intercept)

    def test_update_returns_the_pinned_values(self, poissonGLM_model_instantiation):
        """A single ``update`` step is the identity when nothing is active."""
        X, y, model = poissonGLM_model_instantiation[:3]
        coef, intercept = _pin_everything(model, X, y)
        model.fix_params = (coef, intercept)

        with pytest.warns(UserWarning, match="Every parameter is fixed"):
            state = model.initialize_optimizer_and_state((coef, intercept), X, y)
            model.update((coef, intercept), state, X, y)

        np.testing.assert_array_equal(model.coef_, coef)
        np.testing.assert_array_equal(model.intercept_, intercept)

    def test_fit_does_not_warn_about_convergence(
        self, poissonGLM_model_instantiation, recwarn
    ):
        """The stand-in state reports convergence, so no second warning fires."""
        X, y, model = poissonGLM_model_instantiation[:3]
        model.fix_params = _pin_everything(model, X, y)

        model.fit(X, y)

        assert _convergence_warnings(recwarn) == []

    def test_no_op_solver_is_installed(self, poissonGLM_model_instantiation):
        """The empty-tree case is served by ``NoOpSolver``, not a real solver."""
        X, y, model = poissonGLM_model_instantiation[:3]
        model.fix_params = _pin_everything(model, X, y)

        with pytest.warns(UserWarning, match="Every parameter is fixed"):
            model.fit(X, y)

        assert isinstance(model.solver, NoOpSolver)

    def test_freeing_one_leaf_runs_the_real_solver(
        self, poissonGLM_model_instantiation, recwarn
    ):
        """Leaving any leaf active restores the normal path, with no warning."""
        X, y, model = poissonGLM_model_instantiation[:3]
        _, intercept = _pin_everything(model, X, y)
        model.fix_params = (None, intercept)

        model.fit(X, y)

        assert not isinstance(model.solver, NoOpSolver)
        assert [
            w for w in recwarn if "Every parameter is fixed" in str(w.message)
        ] == []

    @pytest.mark.parametrize("solver_name", ["LBFGS", "Newton"])
    def test_fit_pins_values_for_hessian_and_gradient_solvers(
        self, poissonGLM_model_instantiation, solver_name
    ):
        """Newton needs a Hessian the stand-in never supplies, and still short-circuits.

        ``setup_hessian`` is only reached while instantiating a real solver, which the
        empty-tree case skips, so no Newton-specific hook is required of the stand-in.
        """
        X, y, model = poissonGLM_model_instantiation[:3]
        coef, intercept = _pin_everything(model, X, y)
        model.set_params(fix_params=(coef, intercept), solver_name=solver_name)

        with pytest.warns(UserWarning, match="Every parameter is fixed"):
            model.fit(X, y)

        np.testing.assert_array_equal(model.coef_, coef)
        np.testing.assert_array_equal(model.intercept_, intercept)
        assert isinstance(model.solver, NoOpSolver)

    def test_stochastic_fit_returns_the_pinned_values(
        self, poissonGLM_model_instantiation
    ):
        """``stochastic_fit`` reaches the solver directly, so it needs its own coverage."""
        X, y, model = poissonGLM_model_instantiation[:3]
        coef, intercept = _pin_everything(model, X, y)
        model.set_params(fix_params=(coef, intercept), solver_name="SVRG")

        with pytest.warns(UserWarning, match="Every parameter is fixed"):
            model.stochastic_fit(ArrayDataLoader(X, y, batch_size=32), n_passes=2)

        np.testing.assert_array_equal(model.coef_, coef)
        np.testing.assert_array_equal(model.intercept_, intercept)

    def test_stochastic_fit_pins_a_frozen_leaf_exactly(
        self, poissonGLM_model_instantiation, recwarn
    ):
        """With coef free, the real stochastic solver runs and never moves the intercept."""
        X, y, model = poissonGLM_model_instantiation[:3]
        _, intercept = _pin_everything(model, X, y)
        model.set_params(fix_params=(None, intercept), solver_name="SVRG")

        model.stochastic_fit(ArrayDataLoader(X, y, batch_size=32), n_passes=2)

        np.testing.assert_array_equal(model.intercept_, intercept)
        assert not isinstance(model.solver, NoOpSolver)
        assert [
            w for w in recwarn if "Every parameter is fixed" in str(w.message)
        ] == []

    def test_stochastic_fit_skips_the_data_passes(self, poissonGLM_model_instantiation):
        """No pass or batch runs, so only the train-level callbacks fire.

        Iterating the loader could not change a fully fixed tree, and for out-of-memory
        data it would read everything to no effect.
        """
        X, y, model = poissonGLM_model_instantiation[:3]
        model.set_params(fix_params=_pin_everything(model, X, y), solver_name="SVRG")
        callback = _RecordingCallback()

        with pytest.warns(UserWarning, match="Every parameter is fixed"):
            model.stochastic_fit(
                ArrayDataLoader(X, y, batch_size=32), n_passes=3, callbacks=callback
            )

        assert callback.calls == ["on_train_begin", "on_train_end"]


def test_no_op_solver_covers_the_solver_interface():
    """``NoOpSolver`` replaces any configured solver, so it must implement all of it.

    The model reaches for solver methods after initialization — ``stochastic_run`` is one
    — and a missing one only shows up as an ``AttributeError`` mid-fit. Comparing the
    public surface catches that when the interface grows, e.g. a Hessian mixin.
    """
    expected = {
        name
        for name in dir(AbstractSolver)
        if not name.startswith("_") and callable(getattr(AbstractSolver, name, None))
    }

    missing = sorted(name for name in expected if not hasattr(NoOpSolver, name))
    assert missing == []


def _solver_grad_norm(model, active, X, y):
    """Norm of the gradient of the loss the solver minimized, evaluated at ``active``.

    ``_solver_loss_fun`` is the penalized loss closed over the frozen leaves and taking
    the active subtree alone. The unpenalized model loss is the wrong object: its
    gradient keeps a residual equal to the penalty gradient, no matter the tolerance.
    """
    grad = jax.grad(model._solver_loss_fun)(active, X, y)
    return float(
        jnp.sqrt(
            sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grad))
        )
    )


def _grad_norm_under_other_spec(model, active, X, y, fix_params, full_params):
    """``_solver_grad_norm`` at ``active``, but with different values frozen.

    The frozen values are closed over when the solver is built, so the contrast needs a
    second model carrying the alternative spec.
    """
    other = clone(model)
    other.set_params(fix_params=fix_params)
    other.initialize_optimizer_and_state(full_params, X, y)
    return _solver_grad_norm(other, active, X, y)


@pytest.mark.requires_x64
class TestFrozenValuesEnterTheObjective:
    """Pinned values take part in the objective, not reattached to the result.

    Preservation alone cannot detect a frozen leaf dropped from the linear predictor: the
    returned parameters look identical either way. The active gradient of the solver's own
    loss vanishes at the fitted point only if the pinned value was in the objective, so
    freezing a different value there must spoil stationarity.
    """

    def test_pinned_intercept_is_used_by_the_loss(self, poissonGLM_model_instantiation):
        """coef converges to the optimum *given* the pinned intercept."""
        X, y, model = poissonGLM_model_instantiation[:3]
        pinned = _pinned_intercept(model, X, y)
        model.set_params(fix_params=(None, pinned), solver_kwargs={"tol": 1e-12})

        model.fit(X, y)

        np.testing.assert_allclose(model.intercept_, pinned)
        active, _ = model._partition_active(GLMParams(model.coef_, model.intercept_))
        zeros = jnp.zeros_like(pinned)
        # float64 with tol=1e-12 lands at ~1e-13; three orders of margin, no more
        assert _solver_grad_norm(model, active, X, y) < 1e-10
        assert (
            _grad_norm_under_other_spec(
                model, active, X, y, (None, zeros), (model.coef_, zeros)
            )
            > 1e-2
        )

    def test_pinned_coef_leaf_is_used_by_the_loss(
        self, poissonGLM_model_instantiation_pytree
    ):
        """The free coef leaves converge to the optimum *given* the pinned leaf."""
        X, y, model = poissonGLM_model_instantiation_pytree[:3]
        spec = _coef_spec(model, X, y, "pin_first")
        model.set_params(fix_params=(spec, None), solver_kwargs={"tol": 1e-12})

        model.fit(X, y)

        pinned_key = next(key for key, val in spec.items() if val is not None)
        np.testing.assert_allclose(model.coef_[pinned_key], spec[pinned_key])
        active, _ = model._partition_active(GLMParams(model.coef_, model.intercept_))
        zeroed_spec = {
            key: jnp.zeros_like(val) if key == pinned_key else val
            for key, val in spec.items()
        }
        zeroed_coef = {
            key: jnp.zeros_like(leaf) if key == pinned_key else leaf
            for key, leaf in model.coef_.items()
        }
        # float64 with tol=1e-12 lands at ~1e-13; three orders of margin, no more
        assert _solver_grad_norm(model, active, X, y) < 1e-10
        assert (
            _grad_norm_under_other_spec(
                model,
                active,
                X,
                y,
                (zeroed_spec, None),
                (zeroed_coef, model.intercept_),
            )
            > 1e-2
        )
