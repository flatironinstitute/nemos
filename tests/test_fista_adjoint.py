import os

import optimistix as optx
import pytest

import nemos as nmo

# everything is solver-related here
pytestmark = pytest.mark.solver_related


@pytest.fixture(autouse=True)
def skip_if_override_solver(pytestconfig):
    """Skip these tests when overriding GradientDescent or ProximalGradient implementation."""
    override = pytestconfig.getini("override_solver")
    if override:
        algo, _ = override.split(":", 1)
        if algo in ("GradientDescent", "ProximalGradient"):
            pytest.skip(
                "override_solver changes defaults; FISTA adjoint tests require optimistix defaults"
            )


@pytest.fixture
def optimistix_solver_registry():
    """Point GLM solver registry at the Optimistix implementations for this module."""
    registry = nmo.solvers._solver_registry
    original_registry = registry._registry.copy()
    original_defaults = registry._defaults.copy()
    try:
        registry.register(
            "GradientDescent",
            nmo.solvers.OptimistixNAG,
            backend="optimistix",
            replace=True,
            default=True,
            validate=False,
        )
        registry.register(
            "ProximalGradient",
            nmo.solvers.OptimistixFISTA,
            backend="optimistix",
            replace=True,
            default=True,
            validate=False,
        )
        yield registry._registry
    finally:
        registry._registry.clear()
        registry._registry.update(original_registry)
        registry._defaults.clear()
        registry._defaults.update(original_defaults)


@pytest.mark.parametrize(
    "adjoint",
    [optx.ImplicitAdjoint(), optx.RecursiveCheckpointAdjoint()],
)
@pytest.mark.parametrize(
    "solver_name",
    ["GradientDescent", "ProximalGradient"],
)
@pytest.mark.skipif(
    os.getenv("NEMOS_SOLVER_BACKEND") != "optimistix",
    reason="Only run with the Optimistix backend",
)
def test_glm_passes_adjoint_to_optimistix_config(
    optimistix_solver_registry, adjoint, solver_name
):
    glm = nmo.glm.GLM(
        regularizer="Ridge",
        regularizer_strength=0.1,
        solver_name=solver_name,
        solver_kwargs={"adjoint": adjoint},
    )
    glm._instantiate_solver(glm.compute_loss, None)

    solver_adapter = glm._solver
    assert isinstance(solver_adapter.config.adjoint, type(adjoint))

    # not true because GLM.instantiate_solver does a deepcopy
    # assert solver_adapter.config.adjoint is adjoint


@pytest.mark.parametrize(
    ("adjoint", "expected_kind"),
    [
        (optx.ImplicitAdjoint(), "lax"),
        (optx.RecursiveCheckpointAdjoint(), "checkpointed"),
    ],
)
@pytest.mark.parametrize(
    "solver_name",
    ["GradientDescent", "ProximalGradient"],
)
@pytest.mark.skipif(
    os.getenv("NEMOS_SOLVER_BACKEND") != "optimistix",
    reason="Only run with the Optimistix backend",
)
def test_fista_while_loop_kind_matches_adjoint(
    optimistix_solver_registry, adjoint, expected_kind, solver_name
):
    glm = nmo.glm.GLM(
        regularizer="Ridge",
        regularizer_strength=0.1,
        solver_name=solver_name,
        solver_kwargs={"adjoint": adjoint},
    )
    glm._instantiate_solver(glm.compute_loss, None)

    fista_solver = glm._solver._solver
    assert fista_solver.while_loop_kind == expected_kind


@pytest.mark.parametrize(
    ("adjoint", "while_loop_kind"),
    [
        (optx.ImplicitAdjoint(), "bounded"),
        (optx.RecursiveCheckpointAdjoint(), "lax"),
    ],
)
@pytest.mark.parametrize(
    "solver_name",
    ["GradientDescent", "ProximalGradient"],
)
@pytest.mark.skipif(
    os.getenv("NEMOS_SOLVER_BACKEND") != "optimistix",
    reason="Only run with the Optimistix backend",
)
def test_fista_explicit_while_loop_kind_overrides_adjoint(
    optimistix_solver_registry, adjoint, while_loop_kind, solver_name
):
    glm = nmo.glm.GLM(
        regularizer="Ridge",
        regularizer_strength=0.1,
        solver_name=solver_name,
        solver_kwargs={"adjoint": adjoint, "while_loop_kind": while_loop_kind},
    )
    glm._instantiate_solver(glm.compute_loss, None)

    fista_solver = glm._solver._solver
    assert fista_solver.while_loop_kind == while_loop_kind


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        # "population_poissonGLM_model_instantiation",
    ],
)
@pytest.mark.parametrize(
    "adjoint", [optx.ImplicitAdjoint(), optx.RecursiveCheckpointAdjoint()]
)
@pytest.mark.parametrize("while_loop_kind", ["bounded", "lax", "checkpointed"])
@pytest.mark.skipif(
    os.getenv("NEMOS_SOLVER_BACKEND") != "optimistix",
    reason="Only run with the Optimistix backend",
)
@pytest.mark.filterwarnings(r"ignore:.*fit did not converge.*:RuntimeWarning")
def test_fit_succeeds_with_mismatched_adjoint_and_while_loop_kind(
    optimistix_solver_registry,
    request,
    model_instantiation_type,
    adjoint,
    while_loop_kind,
):
    data = request.getfixturevalue(model_instantiation_type)
    X, y = data[:2]
    model = data[2]

    # explicitly pass a while_loop_kind that does not match the adjoint-derived default
    model.set_params(
        solver_name="ProximalGradient",
        solver_kwargs={
            "adjoint": adjoint,
            "while_loop_kind": while_loop_kind,
            "maxiter": 5,
        },
    )
    model.fit(X, y)

    solver_adapter = model._solver
    assert isinstance(solver_adapter.config.adjoint, type(adjoint))
    assert solver_adapter._solver.while_loop_kind == while_loop_kind

    assert model._get_fit_state()["solver_state_"] is not None
