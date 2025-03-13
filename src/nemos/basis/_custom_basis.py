"""Custom Basis Class.

Facilitate the construction of a custom basis class.
"""
import itertools
from typing import Optional, Callable, Iterable, Tuple, List
from numpy.typing import NDArray

from ._composition_utils import count_positional_and_var_args, infer_input_dimensionality
from ._basis_mixin import BasisMixin
from ..base_class import Base
import numpy as np


def apply_f_vectorized(f: Callable[[NDArray], NDArray], *xi: NDArray, ndim_input: int = 1):
    """Iterate over the output dim and apply the function to all input combination."""

    # check if no dimension needs vectorization
    if all(x[0].ndim - 1 == ndim_input for x in xi):
        return f(*xi)[...,np.newaxis]

    # compute the flat shape of the dimension that must be vectorized.
    flat_vec_dims = (range(1) if x.ndim - 1 == ndim_input else range(np.prod(x.shape[1+ndim_input:])) for x in xi)
    xi_reshape = [x[...,np.newaxis] if x.ndim - 1 == ndim_input else  x.reshape(*x.shape[:1+ndim_input], -1) for x in xi]
    return np.concatenate(
        [
            f(*(x[..., i] for i, x in zip(index, xi_reshape)))[..., np.newaxis]
            for index in itertools.product(*flat_vec_dims)
        ],
        axis = -1
    )


class CustomBasis(BasisMixin, Base):
    def __init__(
            self,
            funcs: List[Callable[[NDArray,...], NDArray]],
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
        self._n_input_dimensionality = infer_input_dimensionality(self)
        self._n_basis_funcs = len(self.funcs)
        super().__init__(label=label)


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
            self._calculate_n_output_features = None
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

    @staticmethod
    def _check_all_same_shape(*x: NDArray):
        if not len(set(xi.shape for xi in x)) == 1:
            raise ValueError("All inputs must have the same shape.")

    def _check_input_dimensionality(self, xi: Tuple) -> None:
        # TODO fill in the check, if possible
        pass

    def compute_features(self, *xi):
        pass

    def evaluate(self, *x: NDArray):
        """Evaluate funcs in a vectorized form."""
        return np.concatenate(
            [               
                apply_f_vectorized(f, *xi, ndim_input=self.ndim_input)
                for xi, f in zip(self._unpack_inputs(self.funcs, *x), self.funcs)
            ],
            axis=-1
        )
