import numpy as np
import pytest

import neurostatslib.basis as basis


def pytest_generate_tests(metafunc):
    # called once per each test function
    if not (
        hasattr(metafunc.function, "__qualname__")
        and "." in metafunc.function.__qualname__
    ):
        # skip if not class
        return
    if not "params" in metafunc.cls.__dict__:
        # skip if params is not defined
        return
    funcarglist = metafunc.cls.params[metafunc.function.__name__]

    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


