"""Custom Basis Class.

Facilitate the construction of a custom basis class.
"""
from typing import Optional, Callable, Iterable, Tuple, List
from numpy.typing import NDArray

from ._composition_utils import count_positional_and_var_args
import numpy as np


def _default_compute_num_output_features(*shapes: Tuple[int,...]):
    return sum(np.prod(s, dtype=int) for s in shapes).item()


def apply_f_vectorized(f: Callable[[NDArray], NDArray], *xi: NDArray, ndim_input: int = 1):
    if ndim_input == 0 or xi[0].ndim - 1 == ndim_input:
        return f(*xi)[...,np.newaxis]

    # assume all xi have the same shape
    x_shape = xi[0].shape

    # compute the expected input shape after moving axis
    shape_after_mv_axis = tuple(
        x_shape[i]
        for i in (0, *range(1 + ndim_input, len(x_shape)), *range(1, 1 + ndim_input))
    )

    # count the axis we are vectorizing over
    n_vec_input = len(shape_after_mv_axis[:-ndim_input]) - 1

    # move axis and reshape lazily
    def move_axes_gen(x_tuple):
        for x in x_tuple:
            yield np.moveaxis(x, range(1, 1 + ndim_input), range(len(x.shape) - ndim_input, len(x.shape)))

    def reshape_gen(x_tuple):
        for x in x_tuple:
            yield x.reshape(-1, *shape_after_mv_axis[-ndim_input:])

    xi_lazy = reshape_gen(move_axes_gen(xi))

    # apply the function to the correct array shape.
    out = f(*xi_lazy)  # this is (n_samples * n_vec_inputs, *out_shape)
    # split back to original func & add axis for concat
    return np.moveaxis(
        np.reshape(out, (*shape_after_mv_axis[:-ndim_input], *out.shape[1:])),
        range(1, 1 + n_vec_input),
        range(len(shape_after_mv_axis[:-ndim_input]) + len(out.shape[1:]) - n_vec_input,
              len(shape_after_mv_axis[:-ndim_input]) + len(out.shape[1:]))
    )[..., np.newaxis]


class CustomBasis:
    def __init__(
            self,
            *funcs: Callable[[NDArray,...], NDArray],
            ndim_input: int = 1,
            ndim_output: int = 1,
            calculate_n_output_features: Optional[Callable[[Tuple[int,...]],  int]] = None,
            label: Optional[str]=None
    ):
        self.funcs = funcs
        self.ndim_input = int(ndim_input)
        self.ndim_output = int(ndim_output)
        self.calculate_n_output_features = calculate_n_output_features
        # nomenclature is confusing, should rename this to _n_args_compute_features
        self._n_input_dimensionality = sum(count_positional_and_var_args(f)[0] for f in self.funcs)
        


    @property
    def funcs(self) -> List[Callable[[NDArray,...], NDArray]]:
        return self._funcs

    @funcs.setter
    def funcs(self, val: Iterable[Callable[[NDArray,...], NDArray]]):
        val = list(val)

        if not all(isinstance(f, Callable) for f in val):
            raise ValueError("User must provide an iterable of callable.")

        if hasattr(self, "_n_input_dimensionality"):
            inp_dim = sum(count_positional_and_var_args(f)[0] for f in val)
            if inp_dim != self._n_input_dimensionality:
                raise ValueError("The number of input time series required by the CustomBasis must be consistent. "
                                 "Redefine a CustomBasis for a different number of inputs.")
        self._funcs = val

    @property
    def calculate_n_output_features(self):
        return self._calculate_n_output_features

    @calculate_n_output_features.setter
    def calculate_n_output_features(self, val: Optional[Callable[[Tuple[int,...]],  int]]):
        if val is None:
            self._calculate_n_output_features = _default_compute_num_output_features
            return 
        
        if not isinstance(val, Callable):
            raise ValueError("When provided, `calculate_n_output_features` must be callable receiving a tuple of "
                             "input shapes and return the number of output features that ``compute_features`` "
                             "will return.")

    @staticmethod
    def _unpack_inputs(funcs, *x):
        cc = 0
        for f in funcs:
            n_inp, _ = count_positional_and_var_args(f)
            yield x[cc:cc + n_inp]
            cc += n_inp

    
    def evaluate(self, *x: NDArray):

        if not len(set(xi.shape for xi in x)) == 1:
            raise ValueError("All inputs must have the same shape.")

        return np.concatenate(
            [               
                apply_f_vectorized(f, *xi, ndim_input=self.ndim_input)
                for xi, f in zip(self._unpack_inputs(self.funcs, *x), self.funcs)
            ],
            axis=-1
        )
