import pytest

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
