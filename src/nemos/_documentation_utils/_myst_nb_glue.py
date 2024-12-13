"""
Utilities for capturing and gluing function outputs for interactive documentation.

This module provides tools to enhance documentation workflows by enabling the capture
of print outputs and source code from Python functions. These outputs can then be
formatted and embedded into Jupyter Notebooks or static documentation using the
`myst-nb` glue functionality.

## Main Features:
- `capture_print`: A decorator to capture print statements from a function and return them alongside the function's
   output.
- `extract_body_exclude_def_and_return`: A utility to extract and format the body of a function, excluding its
   definition and return statement.
- `FormattedString`: A helper class to wrap and format text for use with the glue functionality.

"""

import inspect
import textwrap

import numpy as np

try:
    from myst_nb import glue
except ImportError:
    raise ImportError(
        "Missing optional dependency 'myst-nb'."
        " Please use pip or "
        "conda to install 'myst-nb'."
    )
import io
import sys
from functools import wraps

from .. import basis as nmo_basis
from .. import convolve


class FormattedString:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        # Return the text as it would be printed
        return self.text


def capture_print(func):
    """
    Decorator to capture print output from a function.

    Parameters
    ----------
    func : function
        The function whose print output you want to capture.

    Returns
    -------
    function
        A wrapped function that captures print output and returns it along
        with the original return value.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to a StringIO object
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            # Call the function
            result = func(*args, **kwargs)
            # Return captured output and the function's return value
            return captured_output.getvalue(), result
        finally:
            # Restore original stdout
            sys.stdout = original_stdout

    return wrapper


def extract_body_exclude_def_and_return(func):
    """
    Extracts the body of a given function, unindents it, and excludes the function
    definition and return statement.

    Parameters
    ----------
    func :
        A Python function object.

    Returns
    -------
    :
        The unindented body of the function without the definition and return statement.
    """
    # Get the source code of the function
    source = inspect.getsource(func)

    # Use textwrap.dedent to unindent the source code
    source_lines = source.splitlines()

    # Exclude the def line
    body_lines = []
    skip_first_line = True  # Skip the first `def` line
    for line in source_lines:
        if line.strip().endswith("):"):
            skip_first_line = False
            continue
        if skip_first_line:
            continue
        if line.strip().startswith(
            "return"
        ):  # Stop processing at the `return` statement
            break
        body_lines.append(line)

    return textwrap.dedent("\n".join(body_lines))


@capture_print
def two_step_convolve_cell_body(basis, inp, out):
    # setup the kernels
    basis.set_kernel()
    print(f"Kernel shape (window_size, n_basis_funcs): {basis.kernel_.shape}")

    # apply the convolution
    out_two_steps = convolve.create_convolutional_predictor(basis.kernel_, inp)
    print(f"Convolution output shape: {out_two_steps.shape}")

    # then reshape to 2D
    out_two_steps = out_two_steps.reshape(
        inp.shape[0], inp.shape[1] * inp.shape[2] * basis.n_basis_funcs
    )

    # check that this is equivalent to the output of compute_features
    print(f"All matching: {np.array_equal(out_two_steps, out, equal_nan=True)}")


def glue_two_step_convolve():
    """Run the cell-body, capture out, and glue output strings."""
    inp = np.random.randn(50, 3, 2)

    bas = nmo_basis.RaisedCosineLinearConv(n_basis_funcs=5, window_size=6)
    out = bas.compute_features(inp)

    print_out, _ = two_step_convolve_cell_body(bas, inp, out)
    formatted_string = FormattedString(print_out)

    # Glue the string with explicit formatting for Markdown
    glue("two-step-convolution", formatted_string, display=False)

    # glue second cell
    source = FormattedString(
        extract_body_exclude_def_and_return(two_step_convolve_cell_body)
    )
    glue("two-step-convolution-source-code", source, display=False)
