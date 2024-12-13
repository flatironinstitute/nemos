import pytest
from myst_nb import glue

import nemos._documentation_utils._myst_nb_glue as myst_utils


@pytest.fixture
def functions():

    def func(x, y, z):
        # func body
        return x + y + z

    def func_multiline_init(
        long_variable_names,
        multi_line_init_with_black,
        multi_line_init_with_black_1,
        multi_line_init_with_black_3,
    ):
        # func_multiline_init body
        return

    def func_multiline_return(
        long_variable_names,
        multi_line_init_with_black,
        multi_line_init_with_black_1,
        multi_line_init_with_black_3,
    ):
        # func_multiline_return body
        return (
            long_variable_names,
            multi_line_init_with_black,
            multi_line_init_with_black_1,
            multi_line_init_with_black_3,
        )

    @myst_utils.capture_print
    def func_decorated(
        long_variable_names,
        multi_line_init_with_black,
        multi_line_init_with_black_1,
        multi_line_init_with_black_3,
    ):
        # func_decorated body
        return (
            long_variable_names,
            multi_line_init_with_black,
            multi_line_init_with_black_1,
            multi_line_init_with_black_3,
        )

    return (
        (func, "# func body"),
        (func_multiline_init, "# func_multiline_init body"),
        (func_multiline_return, "# func_multiline_return body"),
        (func_decorated, "# func_decorated body"),
    )


def test_parse_func_body(functions):
    for func, expected in functions:
        assert myst_utils.extract_body_exclude_def_and_return(func) == expected


def test_capture_out():
    @myst_utils.capture_print
    def func(x):
        print(x)
        return 2 * x

    out, doubled = func(4)
    assert doubled == 8
    assert out == "4\n"


def test_glue_formatted_str():
    out = myst_utils.FormattedString("string")
    capture_glue = myst_utils.capture_print(glue)
    out, _ = capture_glue("out-string", out)
    assert out == "string\n"
    out2, _ = capture_glue("not-formatted", "string")
    assert out2 == "'string'\n"


def test_gluing_does_not_display():
    func = myst_utils.capture_print(myst_utils.glue_two_step_convolve)
    out, _ = func()
    assert out == ""
