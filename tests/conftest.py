import numpy as np
import pytest

import neurostatslib.basis as basis


def pytest_generate_tests(metafunc):
    """This function is a pytest hook that is called once per each test function.

        It automatically parametrizes test functions within test classes based on the defined
        'params' attribute.

        Parameters:
        -----------
            metafunc (pytest.Metafunc): The metafunc object containing information about the test function.

        Example usage:
            class TestMathOperations:
                # Define the params attribute with a dictionary of parameter lists for each test function.
                params = {
                    'test_addition': [{'a': 1, 'b': 2}, {'a': -1, 'b': -2}],
                    'test_subtraction': [{'a': 5, 'b': 3}, {'a': 10, 'b': 7}],
                }

                def test_addition(self, a, b):
                    assert a + b == a + b

                def test_subtraction(self, a, b):
                    assert a - b == a - b

        In this example, the 'test_addition' and 'test_subtraction' test functions will be automatically
        parametrized with the values provided in the 'params' attribute.

        """
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
