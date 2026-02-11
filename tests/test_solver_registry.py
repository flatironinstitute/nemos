from contextlib import nullcontext as does_not_raise

import pytest

import nemos as nmo
from nemos.solvers._jaxopt_solvers import JAXOPT_AVAILABLE

pytestmark = pytest.mark.solver_related


# Mock solver classes for testing
# On Python >=3.11 _BaseSolver definition is not required,
# but on 3.10 inheriting from a Protocol creates an __init__ with the
# signature (self, *args, **kwargs) instead of inheriting the defined one
class _BaseSolver(nmo.solvers.SolverProtocol):
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params=None,
        **solver_init_kwargs,
    ):
        pass


class SolverAOne(_BaseSolver):
    pass


class SolverATwo(_BaseSolver):
    pass


class SolverBOne(_BaseSolver):
    pass


@pytest.fixture
def isolated_registry(monkeypatch):
    """Provide an isolated, empty registry for testing."""
    test_registry = {}
    test_defaults = {}

    monkeypatch.setattr(nmo.solvers._solver_registry, "_registry", test_registry)
    monkeypatch.setattr(nmo.solvers._solver_registry, "_defaults", test_defaults)

    return test_registry, test_defaults


class TestParseName:
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
            (
                "al[go[backend]",
                None,
                None,
                pytest.raises(ValueError, match="multiple opening"),
            ),
            (
                "algo[back]end]",
                None,
                None,
                pytest.raises(ValueError, match="multiple closing"),
            ),
            (
                "[backend]",
                None,
                None,
                pytest.raises(
                    ValueError, match="Algorithm name cannot be an empty string"
                ),
            ),
            (
                "algo[]",
                None,
                None,
                pytest.raises(
                    ValueError, match="Backend name cannot be an empty string"
                ),
            ),
            (
                "",
                None,
                None,
                pytest.raises(
                    ValueError, match="Algorithm name cannot be an empty string"
                ),
            ),
        ],
    )
    def test_parse_name(self, name, algo, backend, expectation):
        with expectation:
            res_algo, res_backend = nmo.solvers._solver_registry._parse_name(name)

            assert algo == res_algo
            assert backend == res_backend


class TestResolveBackend:
    def test_one_no_default(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

        assert (
            nmo.solvers._solver_registry._resolve_backend("algo_a", False)
            == "backend_one"
        )

    def test_multiple_default(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=True)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

        assert (
            nmo.solvers._solver_registry._resolve_backend("algo_a", False)
            == "backend_one"
        )

    def test_multiple_no_default(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

        with pytest.raises(ValueError, match="Multiple backends and no default"):
            nmo.solvers._solver_registry._resolve_backend("algo_a", False)

    def test_raise_if_given(self):
        with pytest.raises(ValueError, match="algorithm name only"):
            nmo.solvers._solver_registry._resolve_backend("algo_a[backend_one]", True)


class TestRegister:
    def test_rejects_non_solver_protocol_class(self):
        class NotASolver:
            pass

        with pytest.raises(
            TypeError, match=r"NotASolver doesn't implement SolverProtocol\."
        ):
            nmo.solvers._solver_registry.register(
                "NotASolverAlgo", NotASolver, backend="custom"
            )

    def test_adds_to_registry(self, isolated_registry):
        registry, defaults = isolated_registry

        nmo.solvers.register("algo_a", SolverAOne, "backend_one")

        assert "algo_a" in registry
        assert "backend_one" in registry["algo_a"]
        assert registry["algo_a"]["backend_one"].implementation == SolverAOne

    def test_register_second_implementation(self, isolated_registry):
        registry, defaults = isolated_registry

        nmo.solvers.register("algo_a", SolverAOne, "backend_one")
        nmo.solvers.register("algo_a", SolverATwo, "backend_two")

        assert "algo_a" in registry
        assert "backend_one" in registry["algo_a"]
        assert "backend_two" in registry["algo_a"]

    def test_without_replace_raises_if_exists(self, isolated_registry):
        registry, defaults = isolated_registry

        nmo.solvers.register("algo_a", SolverAOne, "backend_one")

        with pytest.raises(ValueError, match="already registered"):
            nmo.solvers.register("algo_a", SolverATwo, "backend_one", replace=False)

    def test_with_replace_overwrites(self, isolated_registry):
        registry, defaults = isolated_registry

        nmo.solvers.register("algo_a", SolverAOne, "backend_one")
        assert registry["algo_a"]["backend_one"].implementation == SolverAOne

        nmo.solvers.register("algo_a", SolverATwo, "backend_one", replace=True)
        assert registry["algo_a"]["backend_one"].implementation == SolverATwo

    def test_with_default_sets_default(self, isolated_registry):
        registry, defaults = isolated_registry

        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=True)

        assert defaults["algo_a"] == "backend_two"

    def test_validation(self, isolated_registry):
        registry, defaults = isolated_registry

        class BadSolver(SolverAOne):
            def init_state(self, wrong_param_name, *args):
                pass

        nmo.solvers.register("algo_a", BadSolver, "backend_one", validate=False)

        with pytest.raises(ValueError, match="Incompatible signature"):
            nmo.solvers.register("algo_b", BadSolver, "backend_one", validate=True)


class TestGetSolver:
    def test_single_backend(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

        spec = nmo.solvers.get_solver("algo_a")
        assert spec.backend == "backend_one"
        assert spec.implementation == SolverAOne

    def test_default_respected(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=True)

        spec = nmo.solvers.get_solver("algo_a")
        assert spec.backend == "backend_two"
        assert spec.implementation == SolverATwo

    def test_explicit_backend(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=True)

        spec = nmo.solvers.get_solver("algo_a[backend_one]")
        assert spec.backend == "backend_one"
        assert spec.implementation == SolverAOne

    def test_multiple_backends_no_default_raises(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

        with pytest.raises(ValueError, match="Multiple backends and no default"):
            nmo.solvers.get_solver("algo_a")

    def test_algo_not_in_registry_raises(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

        with pytest.raises(ValueError, match="No solver registered"):
            nmo.solvers.get_solver("algo_b")

    def test_invalid_backend_raises(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

        with pytest.raises(ValueError, match="backend not available"):
            nmo.solvers.get_solver("algo_a[random_backend]")

    def test_wrong_format_raises(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=False)

        with pytest.raises(ValueError, match="reserved for specifying"):
            nmo.solvers.get_solver("algo_a[random_bac]kend")


class TestSetDefault:
    def test_changes_default(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one", default=True)
        nmo.solvers.register("algo_a", SolverATwo, "backend_two", default=False)

        assert nmo.solvers.get_solver("algo_a").backend == "backend_one"

        nmo.solvers.set_default("algo_a", "backend_two")

        assert nmo.solvers.get_solver("algo_a").backend == "backend_two"

    def test_raises_if_algo_not_available(self, isolated_registry):
        registry, defaults = isolated_registry

        with pytest.raises(ValueError, match="No solver registered"):
            nmo.solvers.set_default("algo_a", "backend_two")

    def test_raises_if_backend_not_available(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one")

        with pytest.raises(ValueError, match="backend not available"):
            nmo.solvers.set_default("algo_a", "backend_two")


class TestListFunctions:
    def test_list_algo_backends(self, isolated_registry):
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

    def test_list_algo_backends_raises_if_not_registered(self, isolated_registry):
        registry, defaults = isolated_registry

        with pytest.raises(ValueError, match="No solver registered"):
            nmo.solvers.list_algo_backends("algo_a")

    def test_list_available_solvers(self, isolated_registry):
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

    def test_list_available_algorithms(self, isolated_registry):
        registry, defaults = isolated_registry
        nmo.solvers.register("algo_a", SolverAOne, "backend_one")
        nmo.solvers.register("algo_a", SolverATwo, "backend_two")
        nmo.solvers.register("algo_b", SolverBOne, "backend_one")

        assert nmo.solvers.list_available_algorithms() == ["algo_a", "algo_b"]

        nmo.solvers.register("algo_c", SolverBOne, "backend_one")
        assert nmo.solvers.list_available_algorithms() == ["algo_a", "algo_b", "algo_c"]
