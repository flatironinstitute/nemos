import pytest

from contextlib import nullcontext as does_not_raise

import nemos as nmo
from nemos.solvers._jaxopt_solvers import JAXOPT_AVAILABLE

pytestmark = pytest.mark.solver_related


class SolverAOne(nmo.solvers.SolverProtocol):
    pass


class SolverATwo(nmo.solvers.SolverProtocol):
    pass


class SolverBOne(nmo.solvers.SolverProtocol):
    pass


@pytest.fixture
def isolated_registry(monkeypatch):
    """Provide an isolated, empty registry for testing."""
    test_registry = {}
    test_defaults = {}

    monkeypatch.setattr(nmo.solvers._solver_registry, "_registry", test_registry)
    monkeypatch.setattr(nmo.solvers._solver_registry, "_defaults", test_defaults)

    return test_registry, test_defaults


def test_register_rejects_non_solver_protocol_class():
    class NotASolver:
        pass

    with pytest.raises(
        TypeError, match=r"NotASolver doesn't implement SolverProtocol\."
    ):
        nmo.solvers._solver_registry.register(
            "NotASolverAlgo", NotASolver, backend="custom"
        )


@pytest.mark.parametrize(
    "name,algo,backend,expectation",
    [
        (
            "myalgo[mybackend]",
            "myalgo",
            "mybackend",
            does_not_raise(),
        ),
        (
            "myalgo",
            "myalgo",
            None,
            does_not_raise(),
        ),
        (
            "myalgo[mybackend",
            "myalgo",
            "mybackend",
            pytest.raises(ValueError, match="Found opening"),
        ),
        (
            "mya]alg",
            None,
            None,
            pytest.raises(ValueError, match="without opening"),
        ),
        (
            "mya[alg",
            None,
            None,
            pytest.raises(ValueError, match="does not end with closing"),
        ),
        (
            "mya[alg]asdf",
            None,
            None,
            pytest.raises(ValueError, match="middle of the solver name"),
        ),
        # Multiple opening brackets
        (
            "al[go[backend]",
            None,
            None,
            pytest.raises(ValueError, match="multiple opening"),
        ),
        # Multiple closing brackets
        (
            "algo[back]end]",
            None,
            None,
            pytest.raises(ValueError, match="multiple closing"),
        ),
        # Empty algo name
        (
            "[backend]",
            None,
            None,
            pytest.raises(ValueError, match="Algorithm name cannot be an empty string"),
        ),
        # Empty backend
        (
            "algo[]",
            None,
            None,
            pytest.raises(ValueError, match="Backend name cannot be an empty string"),
        ),
        # Empty string (no backend specified)
        (
            "",
            None,
            None,
            pytest.raises(ValueError, match="Algorithm name cannot be an empty string"),
        ),
    ],
)
def test_parse_name(name, algo, backend, expectation):
    with expectation:
        res_algo, res_backend = nmo.solvers._solver_registry._parse_name(name)

        assert algo == res_algo
        assert backend == res_backend


def test_resolve_backend_one_no_default(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

    assert (
        nmo.solvers._solver_registry._resolve_backend("algo_a", False) == "backend_one"
    )


def test_resolve_backend_multiple_default(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=True)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

    assert (
        nmo.solvers._solver_registry._resolve_backend("algo_a", False) == "backend_one"
    )


def test_resolve_backend_multiple_no_default(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

    with pytest.raises(ValueError, match="Multiple backends and no default"):
        nmo.solvers._solver_registry._resolve_backend("algo_a", False)


def test_resolve_backend_raise_if_given():
    with pytest.raises(ValueError, match="algorithm name only"):
        nmo.solvers._solver_registry._resolve_backend("algo_a[backend_one]", True)


def test_get_solver_easy(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

    spec = nmo.solvers.get_solver("algo_a")
    assert spec.backend == "backend_one"
    assert spec.implementation == SolverAOne


def test_get_solver_default_respected(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=True)

    spec = nmo.solvers.get_solver("algo_a")
    assert spec.backend == "backend_two"
    assert spec.implementation == SolverATwo


def test_get_solver_nondefault_backend(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=True)

    spec = nmo.solvers.get_solver("algo_a[backend_one]")
    assert spec.backend == "backend_one"
    assert spec.implementation == SolverAOne


def test_get_solver_multiple_backends_no_default(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

    with pytest.raises(ValueError, match="Multiple backends and no default"):
        nmo.solvers.get_solver("algo_a")


def test_get_solver_raise_if_algo_not_in_registry(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

    with pytest.raises(ValueError, match="No solver registered"):
        nmo.solvers.get_solver("algo_b")


def test_get_solver_invalid_backend(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

    with pytest.raises(ValueError, match="backend not available"):
        nmo.solvers.get_solver("algo_a[random_backend]")


def test_get_solver_wrong_format(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

    with pytest.raises(ValueError, match="reserved for specifying"):
        nmo.solvers.get_solver("algo_a[random_bac]kend")


def test_register_adds_to_registry(isolated_registry):
    registry, defaults = isolated_registry

    nmo.solvers.register("algo_a", SolverAOne, "backend_one")

    assert "algo_a" in registry
    assert "backend_one" in registry["algo_a"]
    assert registry["algo_a"]["backend_one"].implementation == SolverAOne


def test_register_without_replace_raises_if_exists(isolated_registry):
    registry, defaults = isolated_registry

    nmo.solvers.register("algo_a", SolverAOne, "backend_one")

    with pytest.raises(ValueError, match="already registered"):
        nmo.solvers.register("algo_a", SolverATwo, "backend_one", replace=False)


def test_register_with_replace_overwrites(isolated_registry):
    registry, defaults = isolated_registry

    nmo.solvers.register("algo_a", SolverAOne, "backend_one")
    assert registry["algo_a"]["backend_one"].implementation == SolverAOne

    nmo.solvers.register("algo_a", SolverATwo, "backend_one", replace=True)
    assert registry["algo_a"]["backend_one"].implementation == SolverATwo


def test_register_with_default_sets_default(isolated_registry):
    registry, defaults = isolated_registry

    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=True)

    assert defaults["algo_a"] == "backend_two"


def test_register_validation(isolated_registry):
    registry, defaults = isolated_registry

    class BadSolver(SolverAOne):
        def init_state(self, wrong_param_name, *args):
            pass

    nmo.solvers.register("algo_a", BadSolver, "backend_one", validate=False)

    with pytest.raises(ValueError, match="Incompatible signature"):
        nmo.solvers.register("algo_b", BadSolver, "backend_one", validate=True)


def test_set_default(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=True)
    nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

    assert nmo.solvers.get_solver("algo_a").backend == "backend_one"

    nmo.solvers.set_default("algo_a", "backend_two")

    assert nmo.solvers.get_solver("algo_a").backend == "backend_two"


def test_set_default_raises_if_algo_not_available(isolated_registry):
    registry, defaults = isolated_registry

    with pytest.raises(ValueError, match="No solver registered"):
        nmo.solvers.set_default("algo_a", "backend_two")


def test_set_default_raises_if_backend_not_available(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one")

    with pytest.raises(ValueError, match="backend not available"):
        nmo.solvers.set_default("algo_a", "backend_two")


def test_list_algo_backends(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one")
    nmo.solvers.register("algo_a", SolverATwo, "backend_two")
    nmo.solvers.register("algo_b", SolverBOne, "backend_one")

    assert set(nmo.solvers.list_algo_backends("algo_a")) == {
        "backend_one",
        "backend_two",
    }

    assert set(nmo.solvers.list_algo_backends("algo_b")) == {
        "backend_one",
    }


def test_list_algo_backends_raises(isolated_registry):
    registry, defaults = isolated_registry

    with pytest.raises(ValueError, match="No solver registered"):
        nmo.solvers.list_algo_backends("algo_a")


def test_list_available_solvers(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one")
    nmo.solvers.register("algo_a", SolverATwo, "backend_two")
    nmo.solvers.register("algo_b", SolverBOne, "backend_one")

    solvers_in_registry = nmo.solvers.list_available_solvers()

    expected = [
        nmo.solvers.SolverSpec("algo_a", "backend_one", SolverAOne),
        nmo.solvers.SolverSpec("algo_a", "backend_two", SolverATwo),
        nmo.solvers.SolverSpec("algo_b", "backend_one", SolverBOne),
    ]

    assert solvers_in_registry == expected


def test_list_available_algorithms(isolated_registry):
    registry, defaults = isolated_registry
    nmo.solvers.register("algo_a", SolverAOne, "backend_one")
    nmo.solvers.register("algo_a", SolverATwo, "backend_two")
    nmo.solvers.register("algo_b", SolverBOne, "backend_one")

    assert nmo.solvers.list_available_algorithms() == ["algo_a", "algo_b"]

    nmo.solvers.register("algo_c", SolverBOne, "backend_one")
    assert nmo.solvers.list_available_algorithms() == ["algo_a", "algo_b", "algo_c"]


@pytest.mark.skipif(not JAXOPT_AVAILABLE, reason="jaxopt not installed")
def test_list_available_solvers_includes_jaxopt():
    solvers = nmo.solvers.list_available_solvers()
    jaxopt_solvers = [s for s in solvers if s.backend == "jaxopt"]

    assert len(jaxopt_solvers) > 0
    jaxopt_algos = {s.algo_name for s in jaxopt_solvers}
    assert jaxopt_algos == {
        "GradientDescent",
        "ProximalGradient",
        "LBFGS",
        "BFGS",
        "NonlinearCG",
    }


@pytest.mark.skipif(JAXOPT_AVAILABLE, reason="jaxopt is installed")
def test_list_available_solvers_excludes_jaxopt_when_unavailable():
    solvers = nmo.solvers.list_available_solvers()
    jaxopt_solvers = [s for s in solvers if s.backend == "jaxopt"]

    assert len(jaxopt_solvers) == 0
