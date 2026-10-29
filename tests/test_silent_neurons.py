"""Silent-output regression coverage across compatible solver/regularizer pairs."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos.solvers import CHOLESKY_ERR_MSG

REGULARIZERS = ["UnRegularized", "Ridge", "Lasso", "GroupLasso", "ElasticNet"]
SOLVER_REGULARIZER_PAIRS = [
    pytest.param(spec.full_name, regularizer, id=f"{spec.full_name}-{regularizer}")
    for spec in nmo.solvers.list_available_solvers()
    for regularizer in REGULARIZERS
    if spec.algo_name in getattr(nmo.regularizer, regularizer)().allowed_solvers
]


@pytest.mark.parametrize("solver_name,regularizer_name", SOLVER_REGULARIZER_PAIRS)
@pytest.mark.parametrize("case", ["glm", "partial_population", "silent_population"])
def test_silent_neurons_fit(solver_name, regularizer_name, case):
    rng = np.random.default_rng(632)
    X = rng.normal(size=(40, 2))
    y = np.zeros(40) if case == "glm" else np.zeros((40, 3))
    if case == "partial_population":
        y[:, 0] = np.tile([0.0, 1.0], 20)
    model_class = nmo.glm.GLM if case == "glm" else nmo.glm.PopulationGLM
    if regularizer_name == "GroupLasso":
        mask = np.ones((1, 2)) if case == "glm" else np.ones((1, 2, 3))
        regularizer = nmo.regularizer.GroupLasso(mask=mask)
    else:
        regularizer = getattr(nmo.regularizer, regularizer_name)()
    kwargs = {"maxiter": 30, "tol": 1e-4}
    if solver_name.startswith("Newton["):
        kwargs["linear_solver"] = "identity_shift"
    if "SVRG[" in solver_name:
        kwargs.update(batch_size=40, stepsize=0.01)
    model = model_class(
        solver_name=solver_name,
        regularizer=regularizer,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 0.1,
        solver_kwargs=kwargs,
    )
    # This is a finite-fit regression, not a claim of finite MLE convergence.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The fit did not converge", category=RuntimeWarning
        )
        with pytest.warns(UserWarning, match="boundary mean activity"):
            model.fit(X, y)
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_).all()
    # Check the fitted Poisson rates directly. PopulationGLM.predict currently
    # re-wraps GroupLasso masks with y=None, an independent prediction-path bug.
    predicted = np.exp(X @ np.asarray(model.coef_) + np.asarray(model.intercept_))
    assert np.isfinite(predicted).all()
    silent_predictions = predicted if case != "partial_population" else predicted[:, 1:]
    assert np.all(silent_predictions <= 0.5 / len(y) + 1e-6)


@pytest.mark.parametrize("linear_solver", ["eigh", "identity_shift"])
@pytest.mark.parametrize("population", [False, True])
def test_newton_safe_linear_solvers(linear_solver, population):
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = np.zeros((40, 3)) if population else np.zeros(40)
    model_class = nmo.glm.PopulationGLM if population else nmo.glm.GLM
    model = model_class(
        solver_name="Newton",
        regularizer="Ridge",
        regularizer_strength=0.1,
        solver_kwargs={"linear_solver": linear_solver},
    )
    with pytest.warns(UserWarning, match="boundary mean activity"):
        model.fit(X, y)
    assert jnp.all(jnp.isfinite(model.intercept_))


def test_newton_cholesky_failure_has_actionable_message():
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = np.zeros((40, 3))
    y[[1, 5, 7], 0] = 1
    model = nmo.glm.PopulationGLM(
        solver_name="Newton",
        regularizer="Ridge",
        regularizer_strength=0.1,
        solver_kwargs={"linear_solver": "cholesky"},
    )
    # Underflow makes the silent-neuron intercept Hessian exactly singular.
    initial = (np.zeros((2, 3)), np.array([-4.0, -1000.0, -1000.0]))
    with pytest.raises(ValueError, match="solver_kwargs=.*identity_shift") as exc:
        model.fit(X, y, init_params=initial)
    assert "'linear_solver': 'eigh'" in str(exc.value)
    assert CHOLESKY_ERR_MSG in str(exc.value.__cause__)


@pytest.mark.parametrize(
    "message", ["unrelated solver failure", CHOLESKY_ERR_MSG.split(". ")[0]]
)
def test_unrelated_solver_runtime_error_is_preserved(monkeypatch, message):
    failure = RuntimeError(message)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(nmo.solvers.Newton, "run", fail)
    model = nmo.glm.GLM(solver_name="Newton")
    with pytest.raises(RuntimeError) as exc:
        model.fit(np.ones((10, 2)), np.ones(10))
    assert exc.value is failure
