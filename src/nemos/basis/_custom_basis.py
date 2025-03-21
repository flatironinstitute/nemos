"""Custom Basis Class.

Facilitate the construction of a custom basis class.
"""

import inspect
import itertools
import re
from copy import deepcopy
from numbers import Number
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..base_class import Base
from ..utils import format_repr
from . import AdditiveBasis, MultiplicativeBasis
from ._basis_mixin import BasisMixin, set_input_shape_state
from ._composition_utils import (
    _check_valid_shape_tuple,
    count_positional_and_var_args,
    infer_input_dimensionality,
    is_basis_like,
    multiply_basis_by_integer,
    raise_basis_to_power,
)


def simplify_func_repr(string: str):
    """
    Simplify function repr.

    Simplify function repr by dropping the address and replace "functools.partial" with "partial".

    Parameters
    ----------
    string:
        The string repr of a parameter.

    Returns
    -------
        A simplified version of the string.

    """
    pattern = r"<function (<\w+>|\w+) at 0x[0-9a-fA-F]+>"
    new_string = deepcopy(string)
    for match in re.finditer(pattern, string):
        func_name = match.group(1)
        orig_repr = match.group(0)
        new_string = re.sub(orig_repr, func_name, new_string, count=1)
    new_string = re.sub(r"functools\.partial", "partial", new_string)
    return new_string


class FunctionList:
    def __init__(self, funcs, name="b"):
        self.funcs_ = list(funcs)  # store as list
        self.name = name  # optional label prefix

    def __getitem__(self, idx):
        return self.funcs_[idx]

    def __len__(self):
        return len(self.funcs_)

    def __iter__(self):
        return iter(self.funcs_)

    def __repr__(self):
        if len(self.funcs_) <= 2:
            return simplify_func_repr(repr(self.funcs_))
        return f"{simplify_func_repr(repr(self.funcs_[0]))}, ..., {simplify_func_repr(repr(self.funcs_[-1]))}"


def apply_f_vectorized(
    func: Callable[[NDArray], NDArray], *xi: NDArray, ndim_input: int = 1, **kwargs
):
    """Iterate over the output dim and apply the function to all input combination."""

    # check if no dimension needs vectorization
    if all(x.ndim - 1 == ndim_input for x in xi):
        return func(*xi, **kwargs)[..., np.newaxis]

    # compute the flat shape of the dimension that must be vectorized.
    flat_vec_dims = (
        (
            range(1)
            if x.ndim - 1 == ndim_input
            else range(int(np.prod(x.shape[1 + ndim_input :])))
        )
        for x in xi
    )
    xi_reshape = [
        (
            x[..., np.newaxis]
            if x.ndim - 1 == ndim_input
            else x.reshape(*x.shape[: 1 + ndim_input], -1)
        )
        for x in xi
    ]
    return np.concatenate(
        [
            func(*(x[..., i] for i, x in zip(index, xi_reshape)), **kwargs)[
                ..., np.newaxis
            ]
            for index in itertools.product(*flat_vec_dims)
        ],
        axis=-1,
    )


def check_valid_shape(shape):
    is_numeric = not all(isinstance(i, Number) for i in shape)
    is_int = all(i == int(i) for i in shape) if is_numeric else False
    if not (is_numeric and is_int):
        raise ValueError("`output_shape` must be an iterable of integers.")


class CustomBasis(BasisMixin, Base):
    """
    Custom basis class.

    Create a custom basis class from a list of callables (the basis functions).

    Parameters
    ----------
    funcs:
        List of basis functions.
    ndim_input:
        Dimensionality of the input for each sample, i.e. if your time series is of shape ``(n_samples, n, m)``,
        ``ndim_input`` is two.
    output_shape:
        Shape of the output excluding the number of samples.
    basis_kwargs:
        Additional keyword arguments to pass to the basis function.
    label:
        The label of the basis function.

    Examples
    --------
    >>> import numpy as np
    >>> import nemos as nmo
    >>> from functools import partial
    >>> # Define a function
    >>> def decay_exp(x, rate, shift=0):
    ...     return np.exp(-rate * (x + shift)**2)
    >>> # Define a list of basis functions
    >>> funcs = [partial(decay_exp, rate=r) for r in np.linspace(0, 1, 10)]
    >>> bas = nmo.basis.CustomBasis(funcs=funcs, basis_kwargs=dict(shift=1))
    >>> X = bas.compute_features(np.linspace(0, 1, 100))
    """

    def __init__(
        self,
        funcs: List[Callable[[NDArray, ...], NDArray]],
        ndim_input: int = 1,
        output_shape: Tuple[int, ...] = (),
        basis_kwargs: Optional[Any] = None,
        label: Optional[str] = None,
    ):
        self.funcs = funcs
        self.ndim_input = int(ndim_input)

        output_shape = tuple(output_shape)

        self.output_shape = output_shape

        # nomenclature is confusing, should rename this to _n_args_compute_features
        self._n_input_dimensionality = infer_input_dimensionality(self)
        self._n_basis_funcs = len(self.funcs)

        # store args and kwargs
        basis_kwargs = basis_kwargs if basis_kwargs is not None else {}
        sig = inspect.signature(apply_f_vectorized)
        invalid_kwargs = {
            p.name
            for p in sig.parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        if invalid_kwargs.intersection(basis_kwargs.keys()):
            raise ValueError(
                f"Invalid kwargs name in ``basis_kwargs``: {invalid_kwargs.intersection(basis_kwargs.keys())}. "
                "Change parameter name in your function definition."
            )

        self.basis_kwargs = basis_kwargs
        super().__init__(label=label)

    @property
    def funcs(self) -> "FunctionList":
        return self._funcs

    @funcs.setter
    def funcs(self, val: Iterable[Callable[[NDArray, ...], NDArray]]):
        val = FunctionList(val)

        if not all(isinstance(f, Callable) for f in val):
            raise ValueError("User must provide an iterable of callable.")

        if hasattr(self, "_n_input_dimensionality"):
            inp_dim = sum(count_positional_and_var_args(f)[0] for f in val)
            if inp_dim != self._n_input_dimensionality:
                raise ValueError(
                    "The number of input time series required by the CustomBasis must be consistent. "
                    "Redefine a CustomBasis for a different number of inputs."
                )
        self._funcs = val

    def set_output_shape(self, output_shape: Tuple[int] | int):
        if isinstance(output_shape, tuple):
            _check_valid_shape_tuple(output_shape)
            self.output_shape = output_shape
        if isinstance(output_shape, int):
            if output_shape == 1:
                self.output_shape = ()
            if output_shape > 1:
                self.output_shape = (output_shape,)
            else:
                raise ValueError(
                    f"Output shape must be strictly positive (> 0), {output_shape} provided instead."
                )
        raise TypeError(
            "`output_shape` must be a tuple of positive integers or a positive integer.}"
        )

    def compute_features(self, *xi):
        self.set_input_shape(*xi)
        out = self.evaluate(*xi)
        # first dim is samples, the last the concatenated features
        self.output_shape = out.shape[1:-1]
        # return a model design
        return out.reshape(xi[0].shape[0], -1)

    def evaluate(self, *xi: NDArray):
        """Evaluate funcs in a vectorized form."""
        return np.concatenate(
            [
                apply_f_vectorized(
                    f, *xi, **self.basis_kwargs, ndim_input=self.ndim_input
                )
                for f in self.funcs
            ],
            axis=-1,
        )

    @set_input_shape_state(
        states=("_input_shape_product", "_input_shape_", "_label", "_out_shape")
    )
    def __sklearn_clone__(self) -> "CustomBasis":
        """Clone the basis while preserving attributes related to input shapes.

        This method ensures that input shape attributes (e.g., `_input_shape_product`,
        `_input_shape_`) are preserved during cloning. Reinitializing the class
        as in the regular sklearn clone would drop these attributes, rendering
        cross-validation unusable.
        """
        klass = self.__class__(**self.get_params())
        return klass

    @property
    def n_output_features(self):
        if self.input_shape is None:
            return None
        # Computation for number of output features:
        # 1. Compute the number of vectorized dimensions:
        #    - discard axis corresponding to input dimensionality
        #    - multiply the shape of the remaining axis
        # 2. multiply by the number of basis and the shape of the output
        vec_inp = np.prod([shape[self.ndim_input :] for shape in self._input_shape_])
        return int(vec_inp * np.prod(self.output_shape) * len(self.funcs))

    @staticmethod
    def _reshape_concatenated_arrays(
        array: NDArray, bas: "CustomBasis", axis: int
    ) -> NDArray:
        # reshape the arrays to match input shapes
        shape = list(array.shape)
        array = array.reshape(
            shape[:axis]
            + [
                *bas.output_shape,
                *(i for shape in bas._input_shape_ for i in shape[bas.ndim_input :]),
                len(bas.funcs),
            ]
            + shape[axis + 1 :]
        )
        return array

    def __add__(self, other: BasisMixin) -> AdditiveBasis:
        return AdditiveBasis(self, other)

    def __rmul__(self, other: BasisMixin | int) -> BasisMixin:
        return self.__mul__(other)

    def __mul__(self, other: BasisMixin | int) -> BasisMixin:
        if isinstance(other, int):
            return multiply_basis_by_integer(self, other)

        if not is_basis_like(other):
            raise TypeError(
                "Basis multiplicative factor should be a Basis object or a positive integer!"
            )

        return MultiplicativeBasis(self, other)

    def __pow__(self, exponent) -> BasisMixin:
        return raise_basis_to_power(self, exponent)

    def __repr__(self):
        return format_repr(self, multiline=True)
