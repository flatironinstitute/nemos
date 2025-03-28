"""Mixin classes for basis."""

from __future__ import annotations

import abc
import copy
import inspect
import re
from contextlib import contextmanager
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Any, Generator, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..convolve import create_convolutional_predictor
from ..utils import _get_terminal_size
from ._composition_utils import (
    __PUBLIC_BASES__,
    _atomic_basis_label_setter_logic,
    _composite_basis_setter_logic,
    _get_root,
    _recompute_all_default_labels,
    infer_input_dimensionality,
)
from ._transformer_basis import TransformerBasis

if TYPE_CHECKING:
    from ._basis import Basis


def _is_basis(object: Any):
    """Minimal checks for attributes."""
    return all(
        hasattr(object, attrname) for attrname in ("compute_features", "get_params")
    )


def set_input_shape_state(states: Tuple[str] = ("_input_shape_product",)):
    """
    Decorator to preserve input shape-related attributes during method execution.

    This decorator ensures that the attributes `_n_basis_input_` and `_input_shape_product`
    are copied from the original object (`self`) to the returned object (`klass`)
    after the wrapped method executes. It is intended to be used with methods that
    clone or create a new instance of the class, ensuring these critical attributes
    are retained for functionality such as cross-validation.

    Parameters
    ----------
    method :
        The method to be wrapped. This method is expected to return an object
        (`klass`) that requires the `_n_basis_input_` and `_input_shape_` attributes.
    attr_list

    Returns
    -------
    :
        The wrapped method that copies `_input_shape_product` and `_input_shape_` from
        the original object (`self`) to the new object (`klass`).

    Examples
    --------
    Applying the decorator to a method:

    >>> from functools import wraps
    >>> @set_input_shape_state
    ... def __sklearn_clone__(self):
    ...     klass = self.__class__(**self.get_params())
    ...     return klass

    The `_input_shape_product` and `_input_shape_` attributes of `self` will be
    copied to `klass` after the method executes.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Call the original method
            klass = method(self, *args, **kwargs)
            # Copy the specified attributes
            for attr_name in states:
                setattr(klass, attr_name, getattr(self, attr_name, None))
            return klass

        return wrapper

    return decorator


def remap_parameters(method):
    """Remap parameter names to original."""

    @wraps(method)
    def wrapper(self, **params):
        # get key/param map
        _, map_params = self._map_parameters()

        # use mapped key if exists, or original key
        # deepcopy basis to avoid matching labels when assigning twice the same basis
        # to two different components:
        # basis.set_params(basis1=bas, basis2=bas)
        new_params = {
            map_params.get(key, key): (copy.deepcopy(val) if _is_basis(val) else val)
            for key, val in params.items()
        }

        # apply set_params
        self = method(self, **new_params)

        # re-assign ordered ids to all default labels of atomic components
        # note that the recursion always run until parent.
        if self._parent is None:
            _recompute_all_default_labels(self)

        return self

    return wrapper


class AtomicBasisMixin:
    """Mixin class for atomic bases (i.e. non-composite)."""

    def __init__(self, n_basis_funcs: int, label: Optional[str] = None):
        self._n_basis_funcs = n_basis_funcs
        self._input_shape_ = None
        self._check_n_basis_min()

        # initialize as default
        self._label = self.__class__.__name__
        # pass through the checker
        self.label = label

    @property
    def label(self) -> str:
        """Label for the basis."""
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        error = _atomic_basis_label_setter_logic(self, label)
        if error:
            raise error

    @property
    def _has_default_label(self):
        return re.match(rf"^{self.__class__.__name__}(_\d+)?$", self._label) is not None

    @set_input_shape_state(states=("_input_shape_product", "_input_shape_", "_label"))
    def __sklearn_clone__(self) -> Basis:
        """Clone the basis while preserving attributes related to input shapes.

        This method ensures that input shape attributes (e.g., `_input_shape_product`,
        `_input_shape_`) are preserved during cloning. Reinitializing the class
        as in the regular sklearn clone would drop these attributes, rendering
        cross-validation unusable.
        """
        klass = self.__class__(**self.get_params())
        return klass

    def _iterate_over_components(self) -> Generator:
        """Return a generator that iterates over all basis components.

        For atomic bases, the list is just [self].

        Returns
        -------
            A generator returning self, it will be chained in composite bases.

        """
        return (x for x in [self])

    def _generate_subtree_labels(
        self, type_label: Literal["all", "user-defined"] = "all"
    ) -> Generator[str]:
        """
        List all user-specified labels.
        """
        if type_label == "all" or (not self._has_default_label):
            yield self._label

    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Set the expected input shape for the basis object.

        This method configures the shape of the input data that the basis object expects.
        ``xi`` can be specified as an integer, a tuple of integers, or derived
        from an array. The method also calculates the total number of input
        features and output features based on the number of basis functions.

        Parameters
        ----------
        xi :
            The input shape specification.
            - An integer: Represents the dimensionality of the input. A value of ``1`` is treated as scalar input.
            - A tuple: Represents the exact input shape excluding the first axis (sample axis).
              All elements must be integers.
            - An array: The shape is extracted, excluding the first axis (assumed to be the sample axis).

        Raises
        ------
        ValueError
            If a tuple is provided and it contains non-integer elements.

        Returns
        -------
        self :
            Returns the instance itself to allow method chaining.

        Notes
        -----
        All state attributes that depends on the input must be set in this method in order for
        the API of basis to work correctly. In particular, this method is called by ``setup_basis``,
        which is equivalent to ``fit`` for a transformer. If any input dependent state
        is not set in this method, then ``compute_features`` (equivalent to ``fit_transform``) will break.

        """
        if isinstance(xi, tuple):
            if not all(isinstance(i, int) for i in xi):
                raise ValueError(
                    f"The tuple provided contains non integer values. Tuple: {xi}."
                )
            shape = xi
        elif isinstance(xi, int):
            shape = () if xi == 1 else (xi,)
        else:
            shape = xi.shape[1:]

        n_inputs = (int(np.prod(shape)),)

        self._input_shape_ = [shape]

        # total number of input time series. Used  for slicing and reshaping
        self._input_shape_product = n_inputs
        return self

    @property
    def input_shape(self) -> NDArray:
        return self._input_shape_[0] if self._input_shape_ else None


class EvalBasisMixin:
    """Mixin class for evaluational basis."""

    def __init__(self, bounds: Optional[Tuple[float, float]] = None):
        self.bounds = bounds

    def _compute_features(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor):
        """Evaluate basis at sample points.

        The basis is evaluated at the locations specified in the inputs. For example,
        ``compute_features(np.array([0, .5]))`` would return the array:

        .. code-block:: text

           b_1(0) ... b_n(0)
           b_1(.5) ... b_n(.5)

        where ``b_i`` is the i-th basis.

        Parameters
        ----------
        *xi:
            The input samples over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        Returns
        -------
        :
            A matrix with the transformed features.

        """
        out = self.evaluate(*(np.reshape(x, (x.shape[0], -1)) for x in xi))
        return np.reshape(out, (out.shape[0], -1))

    def setup_basis(self, *xi: NDArray) -> Basis:
        """
        Set all basis states.

        This method corresponds sklearn transformer ``fit``. As fit, it must receive the input and
        it must set all basis states, i.e. ``kernel_`` and all the states relative to the input shape.
        The difference between this method and the transformer ``fit`` is in the expected input structure,
        where the transformer ``fit`` method requires the inputs to be concatenated in a 2D array, while here
        each input is provided as a separate time series for each basis element.

        Parameters
        ----------
        xi:
            Input arrays.

        Returns
        -------
        :
            The basis with ready for evaluation.
        """
        self.set_input_shape(*xi)
        return self

    def _set_input_independent_states(self) -> "EvalBasisMixin":
        """
        Compute all the basis states that do not depend on the input.

        For EvalBasisMixin, this method might not perform any operation but simply return the
        instance itself, as no kernel preparation is necessary.

        Returns
        -------
        self :
            The instance itself.

        """
        return self

    @property
    def bounds(self):
        """Range of values covered by the basis."""
        return self._bounds

    @bounds.setter
    def bounds(self, values: Union[None, Tuple[float, float]]):
        """Setter for bounds."""
        if values is not None and len(values) != 2:
            raise ValueError(
                f"The provided `bounds` must be of length two. Length {len(values)} provided instead!"
            )

        # convert to float and store
        try:
            self._bounds = values if values is None else tuple(map(float, values))
        except (ValueError, TypeError):
            raise TypeError("Could not convert `bounds` to float.")

        if values is not None and values[1] <= values[0]:
            raise ValueError(
                f"Invalid bound {values}. Lower bound is greater or equal than the upper bound."
            )


class ConvBasisMixin:
    """Mixin class for convolutional basis."""

    def __init__(self, window_size: int, conv_kwargs: Optional[dict] = None):
        self.kernel_ = None
        self.window_size = window_size
        self.conv_kwargs = {} if conv_kwargs is None else conv_kwargs

    def _compute_features(self, *xi: NDArray | Tsd | TsdFrame | TsdTensor):
        """Convolve basis functions with input time series.

        A bank of basis filters is convolved with the input data. All the dimensions
        except for the sample-axis are flattened, so that the method always returns a
        matrix.

        For example, if inputs are of shape (num_samples, 2, 3), the output will be
        ``(num_samples, num_basis_funcs * 2 * 3)``.

        Parameters
        ----------
        *xi:
            The input data over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        Notes
        -----
        This method is intended to be 1-to-1 mappable to sklearn ``transform`` method of transformer. This
        means that for the method to be callable, all the state attributes have to be pre-computed in a
        method that is mappable to ``fit``, which for us is ``_fit_basis``. It is fundamental that both
        methods behaves like the corresponding transformer method, with the only difference being the input
        structure: a single (X, y) pair for the transformer, a number of time series for the Basis.

        """
        self._check_has_kernel()
        # before calling the convolve, check that the input matches
        # the expectation. We can check xi[0] only, since convolution
        # is applied at the end of the recursion on the 1D basis, ensuring len(xi) == 1.
        conv = create_convolutional_predictor(self.kernel_, *xi, **self._conv_kwargs)
        # make sure to return a matrix
        return np.reshape(conv, (conv.shape[0], -1))

    def setup_basis(self, *xi: NDArray) -> Basis:
        """
        Set all basis states.

        This method corresponds sklearn transformer ``fit``. As fit, it must receive the input and
        it must set all basis states, i.e. ``kernel_`` and all the states relative to the input shape.
        The difference between this method and the transformer ``fit`` is in the expected input structure,
        where the transformer ``fit`` method requires the inputs to be concatenated in a 2D array, while here
        each input is provided as a separate time series for each basis element.

        Parameters
        ----------
        xi:
            Input arrays.

        Returns
        -------
        :
            The basis with ready for evaluation.
        """
        self._set_kernel()
        self.set_input_shape(*xi)
        return self

    def _set_input_independent_states(self):
        """
        Compute all the basis states that do not depend on the input.

        For Conv mixin the only attribute is the kernel.
        """
        return self._set_kernel()

    def _set_kernel(self) -> "ConvBasisMixin":
        """
        Prepare or compute the convolutional kernel for the basis functions.

        This method is called to prepare the basis functions for convolution operations
        in subclasses. It computes a
        kernel based on the basis functions that will be used for convolution with the
        input data. The specifics of kernel computation depend on the subclass implementation
        and the nature of the basis functions.

        Returns
        -------
        self :
            The instance itself, modified to include the computed kernel. This
            allows for method chaining and integration into transformation pipelines.

        Notes
        -----
        Subclasses implementing this method should detail the specifics of how the kernel is
        computed and how the input parameters are utilized.

        """
        self.kernel_ = self.evaluate(np.linspace(0, 1, self.window_size))
        return self

    @property
    def window_size(self):
        """Duration of the convolutional kernel in number of samples."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        """Setter overwrite default setter the window size parameter."""
        self._check_window_size(window_size)
        self._window_size = window_size

    def _check_window_size(self, window_size):
        if window_size is None:
            raise ValueError("You must provide a window_size!")

        elif not (isinstance(window_size, int) and window_size > 0):
            raise ValueError(
                f"`window_size` must be a positive integer. {window_size} provided instead!"
            )

    @property
    def conv_kwargs(self):
        """The convolutional kwargs.

        Keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`.
        """
        return self._conv_kwargs

    @conv_kwargs.setter
    def conv_kwargs(self, values: dict):
        """Check and set convolution kwargs."""
        self._check_convolution_kwargs(values)
        self._conv_kwargs = values

    @staticmethod
    def _check_convolution_kwargs(conv_kwargs: dict):
        """Check convolution kwargs settings.

        Raises
        ------
        ValueError:
            If ``axis`` is provided as an argument, and it is different from 0
            (samples must always be in the first axis).
        ValueError:
            If ``self._conv_kwargs`` include parameters not recognized or that do not have
            default values in ``create_convolutional_predictor``.
        """
        if "axis" in conv_kwargs:
            raise ValueError(
                "Setting the `axis` parameter is not allowed. Basis requires the "
                "convolution to be applied along the first axis (`axis=0`).\n"
                "Please transpose your input so that the desired axis for "
                "convolution is the first dimension (axis=0)."
            )
        convolve_params = inspect.signature(create_convolutional_predictor).parameters
        convolve_configs = {
            key
            for key, param in convolve_params.items()
            if param.default
            # prevent user from passing
            # `basis_matrix` or `time_series` in kwargs.
            is not inspect.Parameter.empty
        }
        if not set(conv_kwargs.keys()).issubset(convolve_configs):
            # do not encourage to set axis.
            convolve_configs = convolve_configs.difference({"axis"})
            # remove the parameter in case axis=0 was passed, since it is allowed.
            invalid = (
                set(conv_kwargs.keys())
                .difference(convolve_configs)
                .difference({"axis"})
            )
            raise ValueError(
                f"Unrecognized keyword arguments: {invalid}. "
                f"Allowed convolution keyword arguments are: {convolve_configs}."
            )

    def _check_has_kernel(self) -> None:
        """Check that the kernel is pre-computed."""
        if self.kernel_ is None:
            raise RuntimeError(
                "You must call `setup_basis` before `_compute_features` for Conv basis."
            )


class BasisTransformerMixin:
    """Mixin class for constructing a transformer."""

    def to_transformer(self) -> TransformerBasis:
        """
        Turn the Basis into a TransformerBasis for use with scikit-learn.

        Examples
        --------
        Jointly cross-validating basis and GLM parameters with scikit-learn.

        >>> import nemos as nmo
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.model_selection import GridSearchCV
        >>> # load some data
        >>> X, y = np.random.normal(size=(30, 1)), np.random.poisson(size=30)
        >>> basis = nmo.basis.RaisedCosineLinearEval(10).set_input_shape(1).to_transformer()
        >>> glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1.)
        >>> pipeline = Pipeline([("basis", basis), ("glm", glm)])
        >>> param_grid = dict(
        ...     glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
        ...     basis__n_basis_funcs=(3, 5, 10, 20, 100),
        ... )
        >>> gridsearch = GridSearchCV(
        ...     pipeline,
        ...     param_grid=param_grid,
        ...     cv=5,
        ... )
        >>> gridsearch = gridsearch.fit(X, y)
        """
        return TransformerBasis(self)


class CompositeBasisMixin:
    """Mixin class for composite basis.

    Add overwrites concrete methods or defines abstract methods for composite basis
    (AdditiveBasis and MultiplicativeBasis).
    """

    _shallow_copy: bool = False

    def __init__(self, basis1: Basis, basis2: Basis, label: Optional[str] = None):
        # This step is slow if you add a very large number of bases
        self._basis1 = None
        self._basis2 = None
        # deep copy to avoid changes directly to the 1d basis to be reflected
        # in the composite basis.

        if not self.__class__._shallow_copy:
            self.basis1 = copy.deepcopy(basis1)
            self.basis2 = copy.deepcopy(basis2)
        else:
            # skip checks and shallow copy
            self._basis1 = basis1
            self._basis2 = basis2

        # set parents
        self.basis1._parent = self
        self.basis2._parent = self

        # initialize attribute
        self._label = None
        # use setter to check & set provided label
        self.label = label

        # number of input arrays that the basis receives
        self._n_input_dimensionality = infer_input_dimensionality(
            basis1
        ) + infer_input_dimensionality(basis2)

    @property
    def basis1(self):
        return self._basis1

    @basis1.setter
    def basis1(self, basis):
        if not hasattr(basis, "get_params") or not hasattr(basis, "compute_features"):
            raise TypeError(
                "`basis1` does not implement `compute_features`. "
                "The method is required for the correct behavior of the basis."
            )

        if self._basis2:
            self._set_labels(basis, self._basis2)
        if self._basis1:
            basis = _composite_basis_setter_logic(basis, self._basis1)
        self._basis1 = basis
        self._input_shape_update()

    @property
    def basis2(self):
        return self._basis2

    @basis2.setter
    def basis2(self, basis):
        if not hasattr(basis, "compute_features"):
            raise TypeError(
                "`basis2` does not implement `compute_features`. "
                "The method is required for the correct behavior of the basis."
            )
        if self._basis1:
            self._set_labels(self._basis1, basis)
        if self._basis2:
            basis = _composite_basis_setter_logic(basis, self._basis2)
        self._basis2 = basis
        self._input_shape_update()

    @property
    def _has_default_label(self):
        return self._label is None

    def _check_unique_labels(self, basis1, basis2):
        """Check that all user-defined labels in the given basis objects are unique."""

        # Include self's label in uniqueness check (if applicable)
        self_label = getattr(self, "_label", None)

        # Store basis1 labels
        seen_labels = set(basis1._generate_subtree_labels("user-defined"))
        if self_label in seen_labels:
            err_msg = (
                f"All user-provided labels of basis elements must be distinct.\n"
                f"The basis you are composing share the following labels: '{self_label}'.\n"
                "Please change the labels for one of the elements before composition."
            )
            return True, err_msg

        # Check for duplicates in basis2 without storing in set for efficiency
        for label in basis2._generate_subtree_labels("user-defined"):
            if label == self_label or label in seen_labels:
                err_msg = (
                    f"All user-provided labels of basis elements must be distinct.\n"
                    f"The basis you are composing share the following labels: '{label}'.\n"
                    "Please change the labels for one of the elements before composition."
                )
                return True, err_msg

        return False, ""

    def _set_labels(self, basis1, basis2):
        # check labels
        non_unique, err_msg = self._check_unique_labels(basis1, basis2)
        if non_unique:
            raise ValueError(err_msg)

        self.update_default_label_id(basis1, basis2)

    def _input_shape_update(self):
        # if all bases where set, then set input for composition.
        set_bases = (s is not None for s in self.input_shape)

        if all(set_bases):
            # pass down the input shapes
            self.set_input_shape(*self.input_shape)

    @property
    def label(self) -> str:
        """Label for the basis."""
        if self._label is None:
            return self._generate_label()
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        reset = (label is None) or (label == self._generate_label())
        if reset:
            self._label = None
        else:
            label = str(label)
            if label in _get_root(self)._generate_subtree_labels():
                raise ValueError(
                    f"Label '{label}' is already in use. When user-provided, label must be unique."
                )
            elif label in __PUBLIC_BASES__ and label != self.__class__.__name__:
                raise ValueError(
                    f"Cannot set basis label '{label}' for basis of type {type(self)}."
                )
            self._label = label

    @property
    def input_shape(self):
        basis1 = getattr(self, "_basis1", None)
        basis2 = getattr(self, "_basis2", None)
        # happens at initialization
        if basis2 is None and basis1 is not None:
            components = (
                basis1._iterate_over_components()
                if hasattr(basis1, "_iterate_over_components")
                else [basis1]
            )
            return [
                *(getattr(bas1, "input_shape", None) for bas1 in components),
                None,
            ]
        components1 = (
            basis1._iterate_over_components()
            if hasattr(basis1, "_iterate_over_components")
            else [basis1]
        )
        components2 = (
            basis2._iterate_over_components()
            if hasattr(basis2, "_iterate_over_components")
            else [basis2]
        )
        shapes = [
            *(getattr(bas1, "input_shape", None) for bas1 in components1),
            *(getattr(bas2, "input_shape", None) for bas2 in components2),
        ]
        return shapes

    @property
    def _input_shape_(self):
        return self.input_shape

    @property
    @abc.abstractmethod
    def n_basis_funcs(self):
        """Read only property for composite bases."""
        pass

    def _generate_subtree_labels(
        self, type_label: Literal["all", "user-defined"] = "all"
    ) -> Generator[str]:
        """
        List all user-specified labels.
        """
        if type_label == "all" or self._label:
            yield self.label
        for lab in self.basis1._generate_subtree_labels(type_label):
            yield lab
        for lab in self.basis2._generate_subtree_labels(type_label):
            yield lab

    def setup_basis(self, *xi: NDArray) -> Basis:
        """
        Set all basis states.

        This method corresponds sklearn transformer ``fit``. As fit, it must receive the input and
        it must set all basis states, i.e. ``kernel_`` and all the states relative to the input shape.
        The difference between this method and the transformer ``fit`` is in the expected input structure,
        where the transformer ``fit`` method requires the inputs to be concatenated in a 2D array, while here
        each input is provided as a separate time series for each basis element.

        Parameters
        ----------
        xi:
            Input arrays.

        Returns
        -------
        :
            The basis with ready for evaluation.
        """
        # setup both input independent
        self._set_input_independent_states()

        # and input dependent states
        self.set_input_shape(*xi)

        return self

    def _set_input_independent_states(self):
        """
        Compute the input dependent states for traversing the composite basis.

        Returns
        -------
        :
            The basis with the states stored as attributes of each component.
        """
        self.basis1._set_input_independent_states()
        self.basis2._set_input_independent_states()

    def _iterate_over_components(self):
        """Return a generator that iterates over all basis components.

        Reimplements the default behavior by iteratively calling _iterate_over_components of the
        elements.

        Returns
        -------
            A generator looping on each individual input.
        """
        components1 = (
            self.basis1._iterate_over_components()
            if hasattr(self.basis1, "_iterate_over_components")
            else [self.basis1]
        )
        components2 = (
            self.basis2._iterate_over_components()
            if hasattr(self.basis2, "_iterate_over_components")
            else [self.basis2]
        )
        return chain(
            components1,
            components2,
        )

    @contextmanager
    def _set_shallow_copy(self, value):
        """Context manager for setting the shallow copy flag in a thread safe way."""
        old_value = self.__class__._shallow_copy
        self.__class__._shallow_copy = value
        try:
            yield
        finally:
            self.__class__._shallow_copy = old_value

    @set_input_shape_state(states=("_input_shape_product", "_label"))
    def __sklearn_clone__(self) -> Basis:
        """Clone the basis while preserving attributes related to input shapes.

        This method ensures that input shape attributes (e.g., `_input_shape_product`,
        `_input_shape_`) are preserved during cloning. Reinitializing the class
        as in the regular sklearn clone would drop these attributes, rendering
        cross-validation unusable.
        The method also handles recursive cloning for composite basis structures.

        Notes
        -----
        The ``_shallow_copy`` attribute is set to True in the context, forcing a shallow copy, at
        before the klass definition, and reset to False after cloning.
        """

        with self._set_shallow_copy(True):
            # clone recursively
            basis1 = self.basis1.__sklearn_clone__()
            basis2 = self.basis2.__sklearn_clone__()

            # shallow copy init
            klass = self.__class__(basis1, basis2)

        return klass

    def set_input_shape(self, *xi: int | tuple[int, ...] | NDArray) -> Basis:
        """
        Set the expected input shape for the basis object.

        This method sets the input shape for each component basis in the basis.
        One ``xi`` must be provided for each basis component, specified as an integer,
        a tuple of integers, or an array. The method calculates and stores the total number of output features
        based on the number of basis functions in each component and the provided input shapes.

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
            If a tuple is provided and it contains non-integer elements.

        Returns
        -------
        self :
            Returns the instance itself to allow method chaining.
        """
        self._input_shape_product = (
            *self.basis1.set_input_shape(
                *xi[: self.basis1._n_input_dimensionality]
            )._input_shape_product,
            *self.basis2.set_input_shape(
                *xi[self.basis1._n_input_dimensionality :]
            )._input_shape_product,
        )
        return self

    def __repr__(self, n=0):
        cols, rows = _get_terminal_size()
        rows = rows // 4
        cols = cols
        disp_label = len(str(self.label)) < cols
        if disp_label:
            start_str = f"'{self.label}': "
        else:
            start_str = ""

        # number of nested composite bases
        n += 1
        tab = "    "
        try:
            basis1 = self.basis1.__repr__(n=n)
        except TypeError:
            basis1 = self.basis1
        try:
            basis2 = self.basis2.__repr__(n=n)
        except TypeError:
            basis2 = self.basis2
        if n < rows:
            rep = (
                start_str + f"{self.__class__.__name__}"
                f"(\n{n*tab}basis1={basis1},\n{n*tab}basis2={basis2},\n{(n-1)*tab})"
            )
        elif n == rows:
            rep = start_str + f"{self.__class__.__name__}(\n{n*tab}...\n{(n-1)*tab})"
        else:
            rep = None
        return rep

    @staticmethod
    def update_default_label_id(basis1: Basis, basis2: Basis):
        """
        Update basis2 atomic element labels.

        When composing bases, update basis2 labels in order to disambiguate them with respect to basis1 labels.
        Here we assume that each tree (basis1 and basis2) have unique basis labels if taken separately, while
        they may have overlapping labels between them.
        This function updates the label of basis2 to avoid overlaps with basis1.


        Parameters
        ----------
        basis1:
            The first basis object.
        basis2:
            The second basis object.

        Notes
        -----
        The method only operates on default labels (i.e. class names).
        User-defined labels will not be edited, and overlaps will result in `ValueError` at composite basis
        initialization.

        """
        delta_labels = dict()
        for bas in basis1._iterate_over_components():
            cls_name = bas.__class__.__name__
            if cls_name not in delta_labels:
                pattern = re.compile(rf"^{cls_name}(_\d+)?$")
                delta_labels[cls_name] = sum(
                    (
                        1
                        for b in basis1._iterate_over_components()
                        if re.match(pattern, b._label)
                    )
                )

        # update the label
        count_labels = dict()
        for bas in basis2._iterate_over_components():
            cls_name = bas.__class__.__name__
            if cls_name not in delta_labels:
                continue
            match = re.match(rf"^{cls_name}(_\d+)?$", bas._label)
            if match:
                current = count_labels.get(cls_name, 0)
                bas._label = (
                    f"{cls_name}_{current + delta_labels[cls_name]}"
                    if current + delta_labels[cls_name]
                    else cls_name
                )
                count_labels[cls_name] = 1 + current

        return

    def __sklearn_get_params__(self, deep=True):
        """
        Implements standard scikit-learn get parameters by inspecting init.

        This function will be called by Base.set_params() to get the actual param
        structure, since the inherited``get_params`` is overridden to use basis labels
        instead of the nested structure.

        Parameters
        ----------
        deep:
            If true, call itself recursively on basis implementing the method.

        Returns
        -------
        out:
            A dictionary containing the parameters. Key is the parameter
            name, value is the parameter value.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if (
                deep
                and hasattr(value, "__sklearn_get_params__")
                and not isinstance(value, type)
            ):
                item_orig = value.__sklearn_get_params__()
                deep_items = item_orig.items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def _map_parameters(self, deep=True):
        """
        Remap parameters in a given object by replacing 'basis[12]' patterns with unique labels.

        Parameters:
        -----------
        bas : object
            An object that contains a `get_params()` method returning a dictionary of parameters.

        Returns:
        --------
        param_dict_map:
            A dictionary with renamed parameters.
        key_map:
            A mapping of new parameter names to original names.

        """
        param_dict_map, key_map = self._get_params_and_key_map(
            deep=deep,
        )  # Retrieve the parameter dictionary
        # strip higher level label
        param_dict_map = self._remove_self_label_from_key(param_dict_map)
        key_map = self._remove_self_label_from_key(key_map)
        return param_dict_map, key_map

    def _get_params_and_key_map(self, deep=True) -> Tuple[dict, dict]:
        """
        From scikit-learn, get parameters by inspecting init.

        Parameters
        ----------
        deep

        Returns
        -------
        parameter_dict:
            A dictionary containing the parameters. Key is the ``basis_label "__" + parameter_name``,
            value is the parameter value.
        key_map:
            Dictionary that maps the keys of parameter_dict onto the keys based on attribute nesting.
        """
        parameter_dict = dict()
        key_map = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and not isinstance(value, type):
                if hasattr(value, "_get_params_and_key_map"):
                    item_map, key_mapping = value._get_params_and_key_map()
                    map_deep_items = item_map.items()
                    # only keep the last basis label (the leaf of the tree)
                    parameter_dict.update(
                        (
                            (
                                "__".join(k.split("__")[-2:])
                                if not _is_basis(val)
                                else val.label
                            ),
                            val,
                        )
                        for k, val in map_deep_items
                    )
                    key_map.update(
                        {
                            k: key + "__" + v
                            for k, v in zip(item_map.keys(), key_mapping.values())
                        }
                    )
                elif hasattr(value, "get_params"):
                    # hit this on atomic bases
                    deep_items = value.get_params().items()
                    parameter_dict.update(
                        {f"{value.label}__{k}": val for k, val in deep_items}
                    )
                    # assume that the basis is either 1 or 2
                    lab = "basis1" if value is self.basis1 else "basis2"
                    key_map.update(
                        {f"{value.label}__{k}": lab + "__" + k for k, _ in deep_items}
                    )

            if _is_basis(value):
                parameter_dict[value.label] = value
                key_map[value.label] = key
            else:
                parameter_dict[self.label + "__" + key] = value
                key_map[self.label + "__" + key] = key
        return parameter_dict, key_map

    def _remove_self_label_from_key(self, map_dict: dict) -> dict:
        initial_string = self.label + "__"
        return {
            k[len(initial_string) :] if k.startswith(initial_string) else k: val
            for k, val in map_dict.items()
        }

    def get_params(self, deep=True) -> dict:
        new_param_dict, _ = self._map_parameters(deep=deep)
        return new_param_dict

    @remap_parameters
    def set_params(self, **params: Any):
        return super().set_params(**params)
