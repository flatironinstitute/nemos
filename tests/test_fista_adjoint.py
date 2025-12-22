import os

import optimistix as optx
import pytest

import nemos as nmo

# everything is solver-related here
pytestmark = pytest.mark.solver_related


@pytest.fixture
def optimistix_solver_registry(monkeypatch):
    """Point GLM solver registry at the Optimistix implementations for this module."""
    registry = nmo.solvers.solver_registry.copy()
    optimistix_registry = registry | {
        "GradientDescent": nmo.solvers.OptimistixNAG,
        "ProximalGradient": nmo.solvers.OptimistixFISTA,
    }
    monkeypatch.setattr(nmo.solvers, "solver_registry", optimistix_registry)
    return optimistix_registry


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
    glm.instantiate_solver(glm.compute_loss)

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
    glm.instantiate_solver(glm.compute_loss)

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
    glm.instantiate_solver(glm.compute_loss)

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
