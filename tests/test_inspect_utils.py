import pytest

import nemos._inspect_utils.inspect_utils as inspect_utils


@pytest.fixture(scope="module")
def function(request):
    func = None
    if request.param == 1:

        def func(x, *args, y=0, **kwargs):
            return

    elif request.param == 2:

        def func(x, y, *args, z=0, **kwargs):
            return

    elif request.param == 3:

        def func(x, y, z, *args, w=0, **kwargs):
            return

    return func


@pytest.mark.parametrize(
    "function, pos, var", [(1, 1, 2), (2, 2, 2), (3, 3, 2)], indirect=["function"]
)
def test_count_var_args(function, pos, var):
    num_pos, num_var = inspect_utils.count_positional_and_var_args(function)
    assert (num_pos, num_var) == (pos, var)
