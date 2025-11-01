"""Custom Basis Class.

Facilitate the construction of a custom basis class.
"""

from __future__ import annotations

import inspect
import itertools
import re
from copy import deepcopy
from numbers import Number
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from nemos.typing import FeatureMatrix

from ..base_class import Base
from ..type_casting import support_pynapple
from ..utils import format_repr
from . import AdditiveBasis, MultiplicativeBasis
from ._basis_mixin import BasisMixin, BasisTransformerMixin, set_input_shape_state
from ._check_basis import _check_transform_input
from ._composition_utils import (
    _check_unique_shapes,
    _check_valid_shape_tuple,
    add_docstring,
    count_positional_and_var_args,
    infer_input_dimensionality,
    is_basis_like,
    multiply_basis_by_integer,
    promote_to_transformer,
    raise_basis_to_power,
    set_input_shape,
)

if TYPE_CHECKING:
    from . import TransformerBasis


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
    pattern = r"<function (.+) at 0x[0-9a-fA-F]+>"
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

    @staticmethod
    def _unwrap(func):
        return func.__wrapped__ if hasattr(func, "__wrapped__") else func

    def __repr__(self):
        unwrap = [self._unwrap(f) for f in self.funcs_]
        if len(unwrap) <= 2:
            return simplify_func_repr(repr(unwrap))
        return f"[{simplify_func_repr(repr(unwrap[0]))}, ..., {simplify_func_repr(repr(unwrap[-1]))}]"


def apply_f_vectorized(
    func: Callable[[NDArray], NDArray], *xi: NDArray, ndim_input: int = 1, **kwargs
):
    """Iterate over the output dim and apply the function to all input combination."""

    # check if no dimension needs vectorization
    if all(x.ndim == ndim_input for x in xi):
        return func(*xi, **kwargs)[..., np.newaxis]

    # Get the vectorized shape (should be the same for all inputs)
    vec_shape = xi[0].shape[ndim_input:]

    # Generate all combinations of vectorized indices in the correct order
    vec_indices = itertools.product(*[range(dim) for dim in vec_shape])

    # Collect results for each vectorized index combination
    results = []
    for indices in vec_indices:
        # Extract slices for this combination of indices
        slices = [x[(slice(None),) * ndim_input + indices] for x in xi]

        # Apply function to the slices
        result = func(*slices, **kwargs)
        results.append(result[..., np.newaxis])

    # Concatenate along the last axis
    return np.concatenate(results, axis=-1)


def check_valid_shape(shape):
    is_numeric = not all(isinstance(i, Number) for i in shape)
    is_int = all(i == int(i) for i in shape) if is_numeric else False
    if not (is_numeric and is_int):
        raise ValueError("`output_shape` must be an iterable of integers.")


class CustomBasis(BasisMixin, BasisTransformerMixin, Base):
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
        Shape of the output excluding the number of samples. Set automatically when `compute_features` is called.
    basis_kwargs:
        Additional keyword arguments to pass to the basis function.
    pynapple_support:
        Enable pynapple support if True.
    label:
        The label of the basis function.
    is_complex : bool, optional
        Whether the basis should be treated as complex. This flag ensures that
        multiplication with other bases behaves correctly: two real bases, or a real
        and a complex basis, can be multiplied, but two complex bases cannot. This
        restriction exists because after multiplication, ``basis.compute_features``
        does not distinguish between real and imaginary components, which would lead
        to incorrect outputs.

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
    >>> bas
    CustomBasis(
        funcs=[partial(decay_exp, rate=np.float64(0.0)), ..., partial(decay_exp, rate=np.float64(1.0))],
        ndim_input=1,
        basis_kwargs={'shift': 1},
        pynapple_support=True,
        is_complex=False
    )
    >>> samples = np.linspace(0, 1, 50)
    >>> X = bas.compute_features(samples)
    >>> X.shape
    (50, 10)
    >>> # Can be composed with other basis (including other custom basis)
    >>> add = bas + bas
    >>> X = add.compute_features(samples, samples)
    >>> X.shape
    (50, 20)
    """

    def __init__(
        self,
        funcs: List[Callable[[NDArray], NDArray]] | Callable[[NDArray], NDArray],
        ndim_input: int = 1,
        output_shape: Optional[Tuple[int, ...] | int] = None,
        basis_kwargs: Optional[dict] = None,
        pynapple_support: bool = True,
        label: Optional[str] = None,
        is_complex: bool = False,
    ):
        self._pynapple_support = bool(pynapple_support)
        self.funcs = funcs
        self.ndim_input = int(ndim_input)

        if output_shape is None:
            output_shape = ()

        self.output_shape = output_shape

        self._input_shape_product = None

        self._n_input_dimensionality = infer_input_dimensionality(self)
        self._n_basis_funcs = len(self.funcs)

        self.basis_kwargs = basis_kwargs
        self._is_complex = bool(is_complex)
        super().__init__(label=label)

    @property
    def is_complex(self):
        # custom classes could be complex or real, so the attribute
        # is an instance attribute not a class attribute
        return self._is_complex

    @property
    def pynapple_support(self) -> bool:
        """Support pynapple Tsd/TsdFrame/TsdTensor as inputs."""
        return self._pynapple_support

    @property
    def funcs(self):
        """User defined list of basis functions."""
        return self._funcs

    @funcs.setter
    def funcs(self, val: Iterable[Callable[[NDArray, ...], NDArray]]):
        if isinstance(val, Callable):
            val = [val]
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

    @property
    def n_basis_funcs(self) -> int:
        """The number of basis."""
        return self._n_basis_funcs

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The shape of the output excluding the number of samples and the number of basis functions."""
        return self._output_shape

    @output_shape.setter
    def output_shape(self, output_shape: Tuple[int] | int):
        if isinstance(output_shape, int):
            if output_shape == 1:
                self._output_shape = ()
            elif output_shape > 1:
                self._output_shape = (output_shape,)
            else:
                raise ValueError(
                    f"Output shape must be strictly positive (> 0), {output_shape} provided instead."
                )
        else:
            try:
                output_shape = tuple(output_shape)
            except TypeError:
                raise TypeError(
                    "`output_shape` must be an iterable of positive integers or a positive integer."
                )
            _check_valid_shape_tuple(output_shape)
            self._output_shape = output_shape

    @property
    def basis_kwargs(self) -> dict:
        """Additional keyword arguments to pass to the basis functions."""
        return self._basis_kwargs

    @basis_kwargs.setter
    def basis_kwargs(self, basis_kwargs: dict):
        # store args and kwargs
        basis_kwargs = basis_kwargs if basis_kwargs is not None else {}
        if not isinstance(basis_kwargs, dict):
            raise ValueError("`basis_kwargs` must be a dictionary.")
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
        self._basis_kwargs = basis_kwargs

    def compute_features(
        self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        """
        Apply the basis transformation to the input data.

        This method applies each function in ``self.funcs`` to the input arrays ``*xi``.
        These functions are called with the arguments ``(*xi, **self.basis_kwargs)`` and must return
        an array of shape ``(n_samples, ...)``, where the first dimension corresponds to the number of samples,
        and the output must have at least one dimension (i.e., ``ndim >= 1``).

        The outputs of all function calls are reshaped into 2D arrays with shape ``(n_samples, n_output)``, and
        then concatenated along the feature axis (second dimension) to form the full design matrix.

        If the input arrays have more dimensions than ``self.ndim_input``, the function calls are automatically
        vectorized over the additional axes. This is done using Python loops, which may be slow. For better
        performance, users are encouraged to provide fully vectorized functions.

        Parameters
        ----------
        *xi :
            Input arrays. Each must have at least ``self.ndim_input`` dimensions. If extra dimensions are present,
            they are interpreted as batch or window dimensions over which the basis functions are applied.

        Returns
        -------
        :
            The resulting design matrix, with one row per sample and one column per output feature.

        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> from functools import partial
        >>> def power_func(n, x):
        ...     return x ** n
        >>> bas = nmo.basis.CustomBasis([partial(power_func, 1), partial(power_func, 2)])
        >>> bas.compute_features(np.arange(1, 4))
        array([[1., 1.],
               [2., 4.],
               [3., 9.]])
        """
        xi = _check_transform_input(self, *xi)
        if any(x.ndim < self.ndim_input for x in xi):
            invalid_dims = [x.ndim for x in xi if x.ndim < self.ndim_input]
            raise ValueError(
                f"Each input must have at least {self.ndim_input} dimensions, as required by this basis. "
                f"However, some inputs have fewer dimensions: {invalid_dims}."
            )
        _check_unique_shapes(*xi, basis=self)
        set_input_shape(self, *xi)
        design_matrix = self.evaluate(
            *xi
        )  # (n_samples, *n_output_shape, n_vec_dim, n_basis)
        # return a model design
        return design_matrix.reshape((xi[0].shape[0], -1))

    def evaluate(self, *xi: NDArray):
        """
        Evaluate the basis functions in a vectorized form at the given sample points.

        Parameters
        ----------
        *xi :
            The samples at which the basis functions are evaluated. Each element in `xi` corresponds
            to an input dimension, and must be broadcastable to a common shape along the sample axis.
            The shape of each input array should be (n_samples, ...) where the first axis indexes samples.

        Returns
        -------
        basis_funcs :
            The basis functions evaluated at the given input points, with shape
            (n_samples, n_vect_input * n_basis_funcs), n_vect_input is the number of inputs that are
            vectorized.

        Notes
        -----
        This method supports both NumPy and pynapple inputs. If pynapple support is enabled,
        the inputs and outputs are automatically cast using the configured backend (e.g., JAX or NumPy).
        Evaluation is performed by applying a vectorized function over each basis function and
        concatenating the results along the last axis.

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CustomBasis
        >>> basis = CustomBasis(funcs=[lambda x: x, lambda x: x**2])
        >>> x = np.linspace(0, 1, 10)
        >>> out = basis.evaluate(x)
        >>> out.shape
        (10, 2)
        >>> # vectorize over 3 inputs
        >>> out = basis.evaluate(np.random.randn(10, 3))
        >>> out.shape
        (10, 3, 2)

        """
        if self._pynapple_support:
            conv_type = "numpy" if nap.nap_config.backend == "numba" else "jax"
            apply_func = support_pynapple(conv_type)(apply_f_vectorized)
        else:
            apply_func = apply_f_vectorized

        # Get individual function results
        func_results = [
            apply_func(f, *xi, **self.basis_kwargs, ndim_input=self.ndim_input)
            for f in self.funcs
        ]

        # Stack functions first, then reorder
        stacked = np.stack(
            func_results, axis=-1
        )  # (n_samples, *out_shape, n_vec_features, n_funcs)
        self.output_shape = stacked.shape[1:-2]

        # no vectorization
        if all(x.ndim == self.ndim_input for x in xi):
            stacked = stacked[..., 0, :]
        return stacked

    @set_input_shape_state(states=("_input_shape_product", "_input_shape_", "_label"))
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
        """The number of output features, i.e. the number of columns of the design matrix."""
        if self.input_shape is None:
            return None
        # Computation for number of output features:
        # 1. Compute the number of vectorized dimensions:
        #    - discard axis corresponding to input dimensionality
        #    - multiply the shape of the remaining axis
        # 2. multiply by the number of basis and the shape of the output

        # Note that self._input_shape_ could be:
        #   1. A list of tuple, if the funcs require > 1 input.
        #   2. A single tuple, if the funcs require 1 input.
        ishape = (
            [self._input_shape_]
            if not isinstance(self._input_shape_, list)
            else self._input_shape_
        )
        vec_inp = np.prod([p for shape in ishape for p in shape[self.ndim_input - 1 :]])
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
                *(
                    i
                    for shape in bas._input_shape_
                    for i in shape[bas.ndim_input - 1 :]
                ),
                len(bas.funcs),
            ]
            + shape[axis + 1 :]
        )
        return array

    @promote_to_transformer
    def __add__(self, other: BasisMixin) -> AdditiveBasis:
        return AdditiveBasis(self, other)

    @promote_to_transformer
    def __rmul__(self, other: BasisMixin | int) -> BasisMixin:
        return self.__mul__(other)

    @promote_to_transformer
    def __mul__(self, other: BasisMixin | int) -> BasisMixin:
        if isinstance(other, int):
            return multiply_basis_by_integer(self, other)

        if not is_basis_like(other):
            raise TypeError(
                "Basis multiplicative factor should be a Basis object or a positive integer!"
            )

        return MultiplicativeBasis(self, other)

    @promote_to_transformer
    def __pow__(self, exponent) -> BasisMixin:
        return raise_basis_to_power(self, exponent)

    def __repr__(self, n=0):
        rep = format_repr(self, multiline=True)
        tab = "    "
        return rep.replace("\n", f"\n{tab * n}")

    def __len__(self) -> int:
        return 1

    @add_docstring("split_by_feature", BasisMixin)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> from functools import partial
        >>> def power_func(n, x):
        ...     return x ** n
        >>> bas = nmo.basis.CustomBasis([partial(power_func, 1), partial(power_func, 2)])
        >>> # define a 3 x 2 input
        >>> inp = np.arange(1, 7).reshape(3, 2)
        >>> X = bas.compute_features(inp)
        >>> X.shape  # (3, 2 * 2)
        (3, 4)
        >>> bas.split_by_feature(X)["CustomBasis"]  # spilt to (3, 2, 2)
        array([[[ 1.,  1.],
                [ 2.,  4.]],
        ...
               [[ 3.,  9.],
                [ 4., 16.]],
        ...
               [[ 5., 25.],
                [ 6., 36.]]])
        """
        # ruff: noqa: D205, D400
        return super().split_by_feature(x, axis=axis)

    def set_input_shape(self, *xi: "int | tuple[int, ...] | NDArray"):
        """
        Set the expected input shape for the basis object.

        This method sets the input shape for each input required by the funcs in the CustomBasis.
        One ``xi`` must be provided for each input, specified as an integer,
        a tuple of integers, or an array. The method calculates and stores the total number of output features
        based on the number of basis functions, the number of input per function, and the provided input shapes.

        Parameters
        ----------
        *xi :
            The input shape specifications. For every k,``xi[k]`` can be:
            - An integer: Represents the dimensionality of the input. A value of ``1`` is treated as scalar input.
            - A tuple: Represents the exact input shape excluding the first axis (sample axis).
              All elements must be integers.
            - An array: The shape is extracted, excluding the first axis (assumed to be the sample axis).

        Raises
        ------
        ValueError
            If a tuple is provided, and it contains non-integer elements.
            If not enough inputs are provided.

        Returns
        -------
        self :
            Returns the instance itself to allow method chaining.

        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> from functools import partial
        >>> # Basis with one input only
        >>> def power_func(n, x):
        ...     return x ** n
        >>> basis = nmo.basis.CustomBasis([partial(power_func, n) for n in range(1, 6)])
        >>> # Configure with an integer input:
        >>> _ = basis.set_input_shape(3)
        >>> basis.n_output_features
        15
        >>> # Configure with a tuple:
        >>> _ = basis.set_input_shape((4, 5))
        >>> basis.n_output_features
        100
        >>> # Configure with an array:
        >>> x = np.ones((10, 4, 5))
        >>> _ = basis.set_input_shape(x)
        >>> basis.n_output_features
        100
        >>> # basis with 2 inputs
        >>> def power_add_func(n, x, y):
        ...     return x ** n + y ** n
        >>> basis = nmo.basis.CustomBasis([partial(power_add_func, n) for n in range(1, 6)])
        >>> _ = basis.set_input_shape(3, 3)
        >>> basis.n_output_features
        15
        >>> _ = basis.set_input_shape((3, 2), (3, 2))
        >>> basis.n_output_features
        30
        >>> _ = basis.set_input_shape(np.ones((10, 3, 2)), (3, 2))
        >>> basis.n_output_features
        30
        """
        super().set_input_shape(*xi, allow_inputs_of_different_shape=False)
        # CustomBasis acts as a multiplicative basis in n-dimension
        # i.e. multiple inputs must have the same shape and are
        # treated in a paired-way in vectorization
        self._input_shape_ = (
            None if self._input_shape_ is None else self._input_shape_[:1]
        )
        return self

    def to_transformer(self) -> "TransformerBasis":
        """
        Turn the Basis into a TransformerBasis for use with scikit-learn.

        Returns
        -------
        :
            A transformer basis.

        Examples
        --------
        >>> from functools import partial
        >>>
        >>> import numpy as np
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.pipeline import Pipeline
        >>>
        >>> import nemos as nmo
        >>>
        >>> # load some data
        >>> x = 0.1 * np.random.normal(size=(100, 1))
        >>> y = np.random.poisson(np.exp(x[:, 0]), size=100)
        >>>
        >>>
        >>> def power_func(n, x, bias=0):
        ...     return (x + bias) ** n
        >>>
        >>>
        >>> basis = nmo.basis.CustomBasis([partial(power_func, n) for n in range(1, 6)])
        >>> basis = basis.to_transformer()
        >>> glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1.0)
        >>> pipeline = Pipeline([("basis", basis), ("glm", glm)])
        >>> param_grid = dict(
        ...     glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
        ...     basis__basis_kwargs=(dict(bias=0), dict(bias=1)),
        ... )
        >>> gridsearch = GridSearchCV(
        ...     pipeline,
        ...     param_grid=param_grid,
        ...     cv=2,
        ... )
        >>> gridsearch = gridsearch.fit(x, y)
        """
        return super().to_transformer()

    @property
    def input_shape(
        self,
    ) -> None | List[None] | Tuple[int, ...] | List[Tuple[int, ...]]:
        """Input shape as a tuple or list of tuple.

        The property mimics the behavior of atomic bases, and uses the
        assumption that _input_shape_ for custom bases is a list of
        length one.
        """
        input_shape = self._input_shape_
        if input_shape is None:
            if self._n_input_dimensionality == 1:
                return None
            else:
                return [None] * self._n_input_dimensionality
        if self._n_input_dimensionality == 1:
            return input_shape[0]
        return input_shape * self._n_input_dimensionality
