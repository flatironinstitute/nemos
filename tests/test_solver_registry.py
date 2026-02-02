import pytest

from contextlib import nullcontext as does_not_raise

import nemos as nmo

pytestmark = pytest.mark.solver_related


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
