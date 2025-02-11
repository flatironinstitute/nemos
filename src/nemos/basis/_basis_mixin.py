"""Mixin classes for basis."""

from __future__ import annotations

import abc
import copy
import inspect
import re
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Generator, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..convolve import create_convolutional_predictor
from ..utils import _get_terminal_size
from ._transformer_basis import TransformerBasis

if TYPE_CHECKING:
    from ._basis import Basis


__PUBLIC_BASES__ = [
    "IdentityEval",
    "HistoryConv",
    "MSplineEval",
    "MSplineConv",
    "BSplineEval",
    "BSplineConv",
    "CyclicBSplineEval",
    "CyclicBSplineConv",
    "RaisedCosineLinearEval",
    "RaisedCosineLinearConv",
    "RaisedCosineLogEval",
    "RaisedCosineLogConv",
    "OrthExponentialEval",
    "OrthExponentialConv",
    "AdditiveBasis",
    "MultiplicativeBasis",
]


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


class AtomicBasisMixin:
    """Mixin class for atomic bases (i.e. non-composite)."""

    def __init__(self, n_basis_funcs: int, label: Optional[str] = None):
        self._n_basis_funcs = n_basis_funcs
        self._input_shape_ = None
        self._check_n_basis_min()

        if (label is None) or (label == self.__class__.__name__):
            self._label = self.__class__.__name__

        else:
            self._label = str(label)

    def _generate_label(self) -> str:
        """Return label"""
        return self._label

    @property
    def label(self) -> str:
        """Label for the basis."""
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        # check default cases
        if label == self._label:
            return

        elif label is None:
            # check if already default
            self._label = self.__class__.__name__
            self._recompute_all_labels()
            return

        else:
            # check if label matches class-name plus identifier
            match = re.match(r"(.+)?_\d+$", label)
            check_string = match.group(1) if match else None
            check_string = check_string if check_string in __PUBLIC_BASES__ else label
            if check_string == self.__class__.__name__:
                self._label = check_string
                self._recompute_all_labels()
                return

        # convert to string (in case label is not string).
        if not isinstance(label, str):
            raise TypeError(
                f"'label' must be a string. Type {type(label)} was provided instead."
            )

        # get the current available labels
        current_labels = self._root()._generate_subtree_labels()
        # if check_string is one of the other classes drop
        if (check_string in current_labels) or (check_string in __PUBLIC_BASES__):
            if check_string in __PUBLIC_BASES__:
                msg = f"Cannot assign '{label}' to a basis of class {self.__class__.__name__}."
            else:
                msg = f"Label '{label}' is already in use. When user-provided, label must be unique."
            raise ValueError(msg)
        else:
            # match if it was a default
            self._update_label_from_root()
            self._label = label

    @property
    def _has_default_label(self):
        return re.match(rf"^{self.__class__.__name__}(_\d+)?$", self._label) is not None

    def _recompute_all_labels(self):
        """
        Recompute all labels matching default for self.
        """
        cls_name = self.__class__.__name__
        pattern = re.compile(rf"^{cls_name}(_\d+)?$")
        root = self._root()
        if root:
            bas_id = 0
            for bas in root._iterate_over_components():
                if re.match(pattern, bas._label):
                    bas._label = f"{cls_name}_{bas_id}" if bas_id else cls_name
                    bas_id += 1

    def _update_label_from_root(self):
        """
        Subtract 1 to each matching default label with higher ID then current.
        """
        cls_name = self.__class__.__name__
        pattern = re.compile(rf"^{cls_name}(_\d+)?$")
        match = re.match(pattern, self._label)
        if match is None:
            return
        # get the "ID" of the label
        current_id = int(match.group(1)[1:]) if match.group(1) else 0
        # subtract one to the ID of any other default label with ID > current_id
        root = self._root()
        if root:
            for bas in root._iterate_over_components():
                match = re.match(pattern, bas._label)
                if match:
                    bas_id = int(match.group(1)[1:]) if match.group(1) else 0
                    bas_id = bas_id - 1 if bas_id > current_id else bas_id
                    bas._label = f"{cls_name}_{bas_id}" if bas_id else cls_name

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
        lab = self._label
        is_default = re.match(rf"^{self.__class__.__name__}(_\d+)?$", lab) is not None
        if type_label == "all" or (not is_default):
            yield lab

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

    def _check_input_shape_consistency(self, x: NDArray):
        """Check input consistency across calls."""
        # remove sample axis and squeeze
        shape = x.shape[1:]

        initialized = self._input_shape_ is not None
        is_shape_match = self._input_shape_[0] == shape
        if initialized and not is_shape_match:
            expected_shape_str = "(n_samples, " + f"{self._input_shape_[0]}"[1:]
            expected_shape_str = expected_shape_str.replace(",)", ")")
            raise ValueError(
                f"Input shape mismatch detected.\n\n"
                f"The basis `{self.__class__.__name__}` with label '{self.label}' expects inputs with "
                f"a consistent shape (excluding the sample axis). Specifically, the shape should be:\n"
                f"  Expected: {expected_shape_str}\n"
                f"  But got:  {x.shape}.\n\n"
                "Note: The number of samples (`n_samples`) can vary between calls of `compute_features`, "
                "but all other dimensions must remain the same. If you need to process inputs with a "
                "different shape, please create a new basis instance, or set a new input shape by calling "
                "`set_input_shape`."
            )

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
        out = self._evaluate(*(np.reshape(x, (x.shape[0], -1)) for x in xi))
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
        return np.reshape(conv, shape=(conv.shape[0], -1))

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
        self.set_kernel()
        self.set_input_shape(*xi)
        return self

    def _set_input_independent_states(self):
        """
        Compute all the basis states that do not depend on the input.

        For Conv mixin the only attribute is the kernel.
        """
        return self.set_kernel()

    def set_kernel(self) -> "ConvBasisMixin":
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
        self.kernel_ = self._evaluate(np.linspace(0, 1, self.window_size))
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

    def __init__(self, basis1: Basis, basis2: Basis, label: Optional[str] = None):
        # deep copy to avoid changes directly to the 1d basis to be reflected
        # in the composite basis.
        # This step is slow if you add a very large number of bases
        self._basis1 = None
        self._basis2 = None

        self.basis1 = copy.deepcopy(basis1)
        self.basis2 = copy.deepcopy(basis2)

        # set parents
        self.basis1._parent = self
        self.basis2._parent = self

        # initialize attribute
        self._label = None
        # use setter to check & set provided label
        self.label = label

    @property
    def basis1(self):
        return self._basis1

    @basis1.setter
    def basis1(self, basis):
        if not hasattr(basis, "get_params") or not hasattr(basis, "compute_features"):
            raise TypeError(
                f"`basis1` must be an object of type `Basis`. Type {type(basis)} provided instead."
            )

        if self._basis2:
            self._set_labels(basis, self._basis2)
        self._basis1 = basis
        self._input_shape_update()

    @property
    def basis2(self):
        return self._basis2

    @basis2.setter
    def basis2(self, basis):
        if not hasattr(basis, "get_params") or not hasattr(basis, "compute_features"):
            raise TypeError(
                f"`basis2` must be an object of type `Basis`. Type {type(basis)} provided instead."
            )
        if self._basis1:
            self._set_labels(self._basis1, basis)
        self._basis2 = basis
        self._input_shape_update()

    @property
    def _has_default_label(self):
        return self._label is None

    def _check_unique_labels(self, basis1, basis2):
        """Check that all user-defined labels in the given basis objects are unique."""

        # Include self's label in uniqueness check (if applicable)
        self_label = getattr(self, "_label", None)

        # Store basis2 labels as we iterate through basis1
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
        set_bases = [s is not None for s in self.input_shape]

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
            if label in self._root()._generate_subtree_labels():
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
        if basis1 is None and basis2 is not None:
            return [
                None,
                *(bas2.input_shape for bas2 in basis2._iterate_over_components()),
            ]
        elif basis2 is None and basis1 is not None:
            return [
                *(bas1.input_shape for bas1 in basis1._iterate_over_components()),
                None,
            ]
        elif basis1 is None and basis2 is None:
            return [None, None]
        shapes = [
            *(bas1.input_shape for bas1 in basis1._iterate_over_components()),
            *(bas2.input_shape for bas2 in basis2._iterate_over_components()),
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

    def _check_input_shape_consistency(self, *xi: NDArray):
        """Check the input shape consistency for all basis elements."""
        self.basis1._check_input_shape_consistency(
            *xi[: self.basis1._n_input_dimensionality]
        )
        self.basis2._check_input_shape_consistency(
            *xi[self.basis1._n_input_dimensionality :]
        )

    def _iterate_over_components(self):
        """Return a generator that iterates over all basis components.

        Reimplements the default behavior by iteratively calling _iterate_over_components of the
        elements.

        Returns
        -------
            A generator looping on each individual input.
        """
        return chain(
            self.basis1._iterate_over_components(),
            self.basis2._iterate_over_components(),
        )

    @set_input_shape_state(states=("_input_shape_product", "_label"))
    def __sklearn_clone__(self) -> Basis:
        """Clone the basis while preserving attributes related to input shapes.

        This method ensures that input shape attributes (e.g., `_input_shape_product`,
        `_input_shape_`) are preserved during cloning. Reinitializing the class
        as in the regular sklearn clone would drop these attributes, rendering
        cross-validation unusable.
        The method also handles recursive cloning for composite basis structures.
        """
        # clone recursively
        basis1 = self.basis1.__sklearn_clone__()
        basis2 = self.basis2.__sklearn_clone__()
        klass = self.__class__(basis1, basis2)

        for attr_name in ["_input_shape_product"]:
            setattr(klass, attr_name, getattr(self, attr_name))
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
