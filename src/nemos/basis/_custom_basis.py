"""Custom Basis Class.

Facilitate the construction of a custom basis class.
"""

import itertools
from numbers import Number
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..base_class import Base
from ._basis_mixin import BasisMixin, set_input_shape_state
from ._composition_utils import (
    count_positional_and_var_args,
    infer_input_dimensionality,
    _check_valid_shape_tuple,
)


def apply_f_vectorized(
    f: Callable[[NDArray], NDArray], *xi: NDArray, ndim_input: int = 1
):
    """Iterate over the output dim and apply the function to all input combination."""

    # check if no dimension needs vectorization
    if all(x.ndim - 1 == ndim_input for x in xi):
        return f(*xi)[..., np.newaxis]

    # compute the flat shape of the dimension that must be vectorized.
    flat_vec_dims = (
        (
            range(1)
            if x.ndim - 1 == ndim_input
            else range(np.prod(x.shape[1 + ndim_input :]))
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
            f(*(x[..., i] for i, x in zip(index, xi_reshape)))[..., np.newaxis]
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
    def __init__(
        self,
        funcs: List[Callable[[NDArray, ...], NDArray]],
        ndim_input: int = 1,
        output_shape: Tuple[int, ...] = (),
        label: Optional[str] = None,
    ):
        self.funcs = funcs
        self.ndim_input = int(ndim_input)

        output_shape = tuple(output_shape)

        self.output_shape = output_shape

        # nomenclature is confusing, should rename this to _n_args_compute_features
        self._n_input_dimensionality = infer_input_dimensionality(self)
        self._n_basis_funcs = len(self.funcs)
        super().__init__(label=label)

    @property
    def funcs(self) -> List[Callable[[NDArray, ...], NDArray]]:
        return self._funcs

    @funcs.setter
    def funcs(self, val: Iterable[Callable[[NDArray, ...], NDArray]]):
        val = list(val)

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
                raise ValueError(f"Output shape must be strictly positive (> 0), {output_shape} provided instead.")
        raise TypeError("`output_shape` must be a tuple of positive integers or a positive integer.}")

    @staticmethod
    def _check_all_same_shape(*x: NDArray):
        if not len(set(xi.shape for xi in x)) == 1:
            raise ValueError("All inputs must have the same shape.")

    def _check_input_dimensionality(self, xi: Tuple) -> None:
        # TODO fill in the check, if possible
        pass

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
                apply_f_vectorized(f, *xi, ndim_input=self.ndim_input)
                for f in self.funcs
            ],
            axis=-1,
        )

    @set_input_shape_state(states=("_input_shape_product", "_input_shape_", "_label", "_out_shape"))
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
        # 2. Each basis contributes to the same number of outputs
        vec_inp = len(self.funcs) * np.prod([shape[self.ndim_input:] for shape in self._input_shape_])
        return int(vec_inp * np.prod(self.output_shape))


    @staticmethod
    def _reshape_concatenated_arrays(array: NDArray, bas: "CustomBasis", axis: int) -> NDArray:
        # reshape the arrays to match input shapes
        shape = list(array.shape)
        array = array.reshape(
            shape[:axis]
            + [*bas.output_shape, *(i for shape in bas._input_shape_ for i in shape[bas.ndim_input:]), len(bas.funcs)]
            + shape[axis + 1:]
        )
        return array

