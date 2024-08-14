"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import copy
from functools import wraps
from typing import Callable, Generator, Literal, Optional, Tuple, Union

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame
from scipy.interpolate import splev

from .base_class import Base
from .convolve import create_convolutional_predictor
from .type_casting import support_pynapple
from .utils import row_wise_kron
from .validation import check_fraction_valid_samples

FeatureMatrix = Union[NDArray, TsdFrame]

__all__ = [
    "MSplineBasis",
    "BSplineBasis",
    "CyclicBSplineBasis",
    "RaisedCosineBasisLinear",
    "RaisedCosineBasisLog",
    "OrthExponentialBasis",
    "AdditiveBasis",
    "MultiplicativeBasis",
    "TransformerBasis",
]


def __dir__() -> list[str]:
    return __all__


def check_transform_input(func: Callable) -> Callable:
    """Check input before calling basis.

    This decorator allows to raise an exception that is more readable
    when the wrong number of input is provided to __call__.
    """

    @wraps(func)
    def wrapper(self: Basis, *xi: ArrayLike, **kwargs) -> NDArray:
        xi = self._check_transform_input(*xi)
        return func(self, *xi, **kwargs)  # Call the basis

    return wrapper


def check_one_dimensional(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self: Basis, *xi: ArrayLike, **kwargs):
        if any(x.ndim != 1 for x in xi):
            raise ValueError("Input sample must be one dimensional!")
        return func(self, *xi, **kwargs)

    return wrapper


def min_max_rescale_samples(
    sample_pts: NDArray,
    bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[NDArray, float]:
    """Rescale samples to [0,1].

    Parameters
    ----------
    sample_pts:
        The original samples.
    bounds:
        Sample bounds. `bounds[0]` and `bounds[1]` are mapped to 0 and 1, respectively.
        Default are `min(sample_pts), max(sample_pts)`.

    Raises
    ------
    ValueError
        If all the samples contain invalid entries (either NaN or Inf).
        This may happen if `max(sample) < bounds[0]` or `min(sample) >  bounds[1]`.

    Warns
    -----
    UserWarning
        If more than 90% of the sample points contain NaNs or Infs.
    """
    sample_pts = sample_pts.astype(float)
    vmin = np.nanmin(sample_pts) if bounds is None else bounds[0]
    vmax = np.nanmax(sample_pts) if bounds is None else bounds[1]
    if vmin and vmax and vmax <= vmin:
        raise ValueError(
            "Invalid value range. `bounds[1]` must be larger then `bounds[0]`!"
        )
    sample_pts[(sample_pts < vmin) | (sample_pts > vmax)] = np.nan
    sample_pts -= vmin
    # this passes if `samples_pts` contains a single value
    if vmin != vmax:
        scaling = vmax - vmin
        sample_pts /= scaling
    else:
        scaling = 1.0

    check_fraction_valid_samples(
        sample_pts,
        err_msg="All the samples lie outside the [vmin, vmax] range.",
        warn_msg="More than 90% of the samples lie outside the [vmin, vmax] range.",
    )

    return sample_pts, scaling


class TransformerBasis:
    """Basis as `scikit-learn` transformers.

    This class abstracts the underlying basis function details, offering methods
    similar to scikit-learn's transformers but specifically designed for basis
    transformations. It supports fitting to data (calculating any necessary parameters
    of the basis functions), transforming data (applying the basis functions to
    data), and both fitting and transforming in one step.

    `TransformerBasis`, unlike `Basis`, is compatible with scikit-learn pipelining and
    model selection, enabling the cross-validation of the basis type and parameters,
    for example `n_basis_funcs`. See the example section below.

    Parameters
    ----------
    basis :
        A concrete subclass of `Basis`.

    Examples
    --------
    >>> from nemos.basis import BSplineBasis, TransformerBasis
    >>> from nemos.glm import GLM
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import GridSearchCV
    >>> import numpy as np
    >>> np.random.seed(123)

    >>> # Generate data
    >>> num_samples, num_features = 10000, 1
    >>> x = np.random.normal(size=(num_samples, ))  # raw time series
    >>> basis = BSplineBasis(10)
    >>> features = basis.compute_features(x)  # basis transformed time series
    >>> weights = np.random.normal(size=basis.n_basis_funcs)  # true weights
    >>> y = np.random.poisson(np.exp(features.dot(weights)))  # spike counts

    >>> # transformer can be used in pipelines
    >>> transformer = TransformerBasis(basis)
    >>> pipeline = Pipeline([ ("compute_features", transformer), ("glm", GLM()),])
    >>> pipeline.fit(x[:, None], y)  # x need to be 2D for sklearn transformer API
    >>> print(pipeline.predict(np.random.normal(size=(10, 1))))  # predict rate from new data

    >>> # TransformerBasis parameter can be cross-validated.
    >>> # 5-fold cross-validate the number of basis
    >>> param_grid = dict(compute_features__n_basis_funcs=[4, 10])
    >>> grid_cv = GridSearchCV(pipeline, param_grid, cv=5)
    >>> grid_cv.fit(x[:, None], y)
    >>> print("Cross-validated number of basis:", grid_cv.best_params_)
    """

    def __init__(self, basis: Basis):
        self._basis = copy.deepcopy(basis)

    @staticmethod
    def _unpack_inputs(X: FeatureMatrix):
        """Unpack impute without using transpose.

        Unpack horizontally stacked inputs using slicing. This works gracefully with `pynapple`,
        returning a list of Tsd objects. Attempt to unpack using *X.T will raise a `pynapple`
        exception since `pynapple` assumes that the time axis is the first axis.

        Parameters
        ----------
        X:
            The inputs horizontally stacked.

        Returns
        -------
        :
            A tuple of each individual input.

        """
        return (X[:, k] for k in range(X.shape[1]))

    def fit(self, X: FeatureMatrix, y=None):
        """
        Compute the convolutional kernels.

        If any of the 1D basis in self._basis is in "conv" mode, it computes the convolutional kernels.

        Parameters
        ----------
        X :
            The data to fit the basis functions to, shape (num_samples, num_input).
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self :
            The transformer object.
        """
        self._basis._set_kernel(*self._unpack_inputs(X))
        return self

    def transform(self, X: FeatureMatrix, y=None) -> FeatureMatrix:
        """
        Transform the data using the fitted basis functions.

        Parameters
        ----------
        X :
            The data to transform using the basis functions, shape (num_samples, num_input).
        y :
            Not used, present for API consistency by convention.

        Returns
        -------
        :
            The data transformed by the basis functions.
        """
        # transpose does not work with pynapple
        # can't use func(*X.T) to unwrap

        return self._basis._compute_features(*self._unpack_inputs(X))

    def fit_transform(self, X: FeatureMatrix, y=None) -> FeatureMatrix:
        """
        Compute the kernels and the features.

        This method is a convenience that combines fit and transform into
        one step.

        Parameters
        ----------
        X :
            The data to fit the basis functions to and then transform.
        y :
            Not used, present for API consistency by convention.

        Returns
        -------
        array-like
            The data transformed by the basis functions, after fitting the basis
            functions to the data.
        """
        return self._basis.compute_features(*self._unpack_inputs(X))

    def __getstate__(self):
        """
        Explicitly define how to pickle TransformerBasis object.

        See https://docs.python.org/3/library/pickle.html#object.__getstate__
        and https://docs.python.org/3/library/pickle.html#pickle-state
        """
        return {"_basis": self._basis}

    def __setstate__(self, state):
        """
        Define how to populate the object's state when unpickling.

        Note that during unpickling a new object is created without calling __init__.
        Needed to avoid infinite recursion in __getattr__ when unpickling.

        See https://docs.python.org/3/library/pickle.html#object.__setstate__
        and https://docs.python.org/3/library/pickle.html#pickle-state
        """
        self._basis = state["_basis"]

    def __getattr__(self, name: str):
        """
        Enable easy access to attributes of the underlying Basis object.

        Examples
        --------
        >>> from nemos import basis
        >>> bas = basis.RaisedCosineBasisLinear(5)
        >>> trans_bas = basis.TransformerBasis(bas)
        >>> bas.n_basis_funcs
        5
        >>> trans_bas.n_basis_funcs
        5
        """
        return getattr(self._basis, name)

    def __setattr__(self, name: str, value) -> None:
        """
        Allow setting _basis or the attributes of _basis with a convenient dot assignment syntax.

        Setting any other attribute is not allowed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the attribute being set is not `_basis` or an attribute of `_basis`.

        Examples
        --------
        >>> import nemos as nmo
        >>> trans_bas = nmo.basis.TransformerBasis(nmo.basis.MSplineBasis(10))
        >>> # allowed
        >>> trans_bas._basis = nmo.basis.BSplineBasis(10)
        >>> # allowed
        >>> trans_bas.n_basis_funcs = 20
        >>> # not allowed
        >>> tran_bas.random_attribute_name = "some value"
        Traceback (most recent call last):
        ...
        ValueError: Only setting _basis or existing attributes of _basis is allowed.
        """
        # allow self._basis = basis
        if name == "_basis":
            super().__setattr__(name, value)
        # allow changing existing attributes of self._basis
        elif hasattr(self._basis, name):
            setattr(self._basis, name, value)
        # don't allow setting any other attribute
        else:
            raise ValueError(
                "Only setting _basis or existing attributes of _basis is allowed."
            )

    def __sklearn_clone__(self) -> TransformerBasis:
        """
        Customize how TransformerBasis objects are cloned when used with sklearn.model_selection.

        By default, scikit-learn tries to clone the object by calling __init__ using the output of get_params,
        which fails in our case.

        For more info: https://scikit-learn.org/stable/developers/develop.html#cloning
        """
        cloned_obj = TransformerBasis(copy.deepcopy(self._basis))
        cloned_obj._basis.kernel_ = None
        return cloned_obj

    def set_params(self, **parameters) -> TransformerBasis:
        """
        Set TransformerBasis parameters.

        When used with `sklearn.model_selection`, users can set either the `_basis` attribute directly
        or the parameters of the underlying Basis, but not both.

        Examples
        --------
        >>> from nemos.basis import BSplineBasis, MSplineBasis, TransformerBasis
        >>> basis = MSplineBasis(10)
        >>> transformer_basis = TransformerBasis(basis=basis)

        >>> # setting parameters of _basis is allowed
        >>> print(transformer_basis.set_params(n_basis_funcs=8).n_basis_funcs)

        >>> # setting _basis directly is allowed
        >>> print(transformer_basis.set_params(_basis=BSplineBasis(10))._basis)

        >>> # mixing is not allowed, this will raise an exception
        >>> transformer_basis.set_params(_basis=BSplineBasis(10), n_basis_funcs=2)
        """
        new_basis = parameters.pop("_basis", None)
        if new_basis is not None:
            self._basis = new_basis
            if len(parameters) > 0:
                raise ValueError(
                    "Set either new _basis object or parameters for existing _basis, not both."
                )
        else:
            self._basis = self._basis.set_params(**parameters)

        return self

    def get_params(self, deep: bool = True) -> dict:
        """Extend the dict of parameters from the underlying Basis with _basis."""
        return {"_basis": self._basis, **self._basis.get_params(deep)}

    def __dir__(self) -> list[str]:
        """Extend the list of properties of methods with the ones from the underlying Basis."""
        return super().__dir__() + self._basis.__dir__()

    def __add__(self, other: TransformerBasis) -> TransformerBasis:
        """
        Add two TransformerBasis objects.

        Parameters
        ----------
        other
            The other TransformerBasis object to add.

        Returns
        -------
        : TransformerBasis
            The resulting Basis object.
        """
        return TransformerBasis(self._basis + other._basis)

    def __mul__(self, other: TransformerBasis) -> TransformerBasis:
        """
        Multiply two TransformerBasis objects.

        Parameters
        ----------
        other
            The other TransformerBasis object to multiply.

        Returns
        -------
        :
            The resulting Basis object.
        """
        return TransformerBasis(self._basis * other._basis)

    def __pow__(self, exponent: int) -> TransformerBasis:
        """Exponentiation of a TransformerBasis object.

        Define the power of a basis by repeatedly applying the method __mul__.
        The exponent must be a positive integer.

        Parameters
        ----------
        exponent :
            Positive integer exponent

        Returns
        -------
        :
            The product of the basis with itself "exponent" times. Equivalent to self * self * ... * self.

        Raises
        ------
        TypeError
            If the provided exponent is not an integer.
        ValueError
            If the integer is zero or negative.
        """
        # errors are handled by Basis.__pow__
        return TransformerBasis(self._basis**exponent)


class Basis(Base, abc.ABC):
    """
    Abstract base class for defining basis functions for feature transformation.

    Basis functions are mathematical constructs that can represent data in alternative,
    often more compact or interpretable forms. This class provides a template for such
    transformations, with specific implementations defining the actual behavior.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    *args :
        Only used in "conv" mode. Additional positional arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode: Literal["eval", "conv"] = "eval",
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        self.n_basis_funcs = n_basis_funcs
        self._n_input_dimensionality = 0
        self._check_n_basis_min()
        self._conv_kwargs = kwargs
        self.bounds = bounds

        # check mode
        if mode not in ["conv", "eval"]:
            raise ValueError(
                f"`mode` should be either 'conv' or 'eval'. '{mode}' provided instead!"
            )
        if mode == "conv":
            if window_size is None:
                raise ValueError(
                    "If the basis is in `conv` mode, you must provide a window_size!"
                )
            elif not (isinstance(window_size, int) and window_size > 0):
                raise ValueError(
                    f"`window_size` must be a positive integer. {window_size} provided instead!"
                )
            if bounds is not None:
                raise ValueError("`bounds` should only be set when `mode=='eval'`.")
        else:
            if kwargs:
                raise ValueError(
                    f"kwargs should only be set when mode=='conv', but '{mode}' provided instead!"
                )

        self._window_size = window_size
        self._mode = mode
        self.kernel_ = None
        self._identifiability_constraints = False

    @property
    def bounds(self):
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

    @property
    def mode(self):
        return self._mode

    @property
    def window_size(self):
        return self._window_size

    @property
    def identifiability_constraints(self):
        return self._identifiability_constraints

    @identifiability_constraints.setter
    def identifiability_constraints(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"`identifiability_constraints` must be a boolean. {type(value)} provided instead!"
            )
        self._identifiability_constraints = value

    @staticmethod
    def _apply_identifiability_constraints(X: NDArray):
        """Apply identifiability constraints to a design matrix `X`.

         Removes columns from `X` until `[1, X]` is full rank to ensure the uniqueness
         of the GLM (Generalized Linear Model) maximum-likelihood solution. This is particularly
         crucial for models using bases like BSplines and CyclicBspline, which, due to their
         construction, sum to 1 and can cause rank deficiency when combined with an intercept.

         For GLMs, this rank deficiency means that different sets of coefficients might yield
         identical predicted rates and log-likelihood, complicating parameter learning, especially
         in the absence of regularization.

        Parameters
        ----------
        X:
            The design matrix before applying the identifiability constraints.

        Returns
        -------
        :
            The adjusted design matrix with redundant columns dropped and columns mean-centered.
        """

        def add_constant(x):
            return np.hstack((np.ones((x.shape[0], 1)), x))

        rank = np.linalg.matrix_rank(add_constant(X))
        # mean center
        X -= np.nanmean(X, axis=0)
        while rank < X.shape[1] + 1:
            # drop a column
            X = X[:, :-1]
            # recompute rank
            rank = np.linalg.matrix_rank(add_constant(X))
        return X

    @check_transform_input
    def _compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        r"""
        Apply the basis transformation to the input data.

        This method operates in two modes:
        - 'eval': Evaluates the basis functions at the given sample points.
        - 'conv': Applies a convolution operation between the input data and the basis functions,
          using a window size defined at initialization.

        Parameters
        ----------
        *xi:
            The input samples over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        Returns
        -------
        :
            A matrix with the transformed features. The shape of the output depends on the operation mode:
                - If `mode == 'eval'`, the basis evaluated at the samples, or $b_i(*xi)$, where $b_i$ is a
                basis element. xi[k] must be a one-dimensional array or a pynapple Tsd.

                - If `mode == 'conv'`, a bank of basis filters (created by calling fit) is convolved with the
                samples. Samples can be a NDArray, or a pynapple Tsd/TsdFrame/TsdTensor. All the dimensions
                except for the sample-axis are flattened, so that the method always returns a matrix.
                For example, if samples are of shape (num_samples, 2, 3), the output will be
                (num_samples, num_basis_funcs * 2 * 3).
                The time-axis can be specified at basis initialization by setting the keyword argument `axis`.
                For example, if `axis == 1` your samples should be (N1, num_samples N3, ...), the output of
                transform will be (num_samples, num_basis_funcs * N1 * N3 *...).

        Raises
        ------
        ValueError:
            If an invalid mode is specified or necessary parameters for the chosen mode are missing.
        """
        # check if self.kernel_ is not None for mode="conv"
        self._check_has_kernel()
        if self.mode == "eval":  # evaluate at the sample
            return self.__call__(*xi)
        else:  # convolve, called only at the last layer
            if "axis" not in self._conv_kwargs:
                axis = 0
            else:
                axis = self._conv_kwargs["axis"]
            # convolve called at the end of any recursive call
            # this ensures that len(xi) == 1.
            conv = create_convolutional_predictor(
                self.kernel_, *xi, **self._conv_kwargs
            )
            # move the time axis to the first dimension
            new_axis = (np.arange(conv.ndim) + axis) % conv.ndim
            conv = np.transpose(conv, new_axis)
            # make sure to return a matrix
            return np.reshape(conv, newshape=(conv.shape[0], -1))

    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Compute the basis functions and transform input data into model features.

        This method is designed to be a high-level interface for transforming input
        data using the basis functions defined by the subclass. Depending on the basis'
        mode ('eval' or 'conv'), it either evaluates the basis functions at the sample
        points or performs a convolution operation between the input data and the
        basis functions.

        Parameters
        ----------
        *xi :
            Input data arrays to be transformed. The shape and content requirements
            depend on the subclass and mode of operation ('eval' or 'conv').

        Returns
        -------
        :
            Transformed features. In 'eval' mode, it corresponds to the basis functions
            evaluated at the input samples. In 'conv' mode, it consists of convolved
            input samples with the basis functions. The output shape varies based on
            the subclass and mode.

        Notes
        -----
        Subclasses should implement how to handle the transformation specific to their
        basis function types and operation modes.
        """
        if self.kernel_ is None:
            self._set_kernel(*xi)
        return self._compute_features(*xi)

    def _set_kernel(self, *xi: ArrayLike) -> Basis:
        """
        Prepare or compute the convolutional kernel for the basis functions.

        This method is called to prepare the basis functions for convolution operations
        in subclasses where the 'conv' mode is used. It typically involves computing a
        kernel based on the basis functions that will be used for convolution with the
        input data. The specifics of kernel computation depend on the subclass implementation
        and the nature of the basis functions.

        In 'eval' mode, this method might not perform any operation but simply return the
        instance itself, as no kernel preparation is necessary.

        Parameters
        ----------
        *xi :
            The input data based on which the kernel might be computed. The actual use of
            these inputs is subclass-specific and might not be applicable for all basis types.

        Returns
        -------
        self :
            The instance itself, modified to include the computed kernel if applicable. This
            allows for method chaining and integration into transformation pipelines.

        Notes
        -----
        Subclasses implementing this method should detail the specifics of how the kernel is
        computed and how the input parameters are utilized. If the basis operates in 'eval'
        mode exclusively, this method should simply return `self` without modification.
        """
        if self.mode == "conv":
            self.kernel_ = self.__call__(np.linspace(0, 1, self.window_size))
        return self

    @abc.abstractmethod
    def __call__(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Abstract method to evaluate the basis functions at given points.

        This method must be implemented by subclasses to define the specific behavior
        of the basis transformation. The implementation depends on the type of basis
        (e.g., spline, raised cosine), and it should evaluate the basis functions at
        the specified points in the domain.

        Parameters
        ----------
        *xi :
            Variable number of arguments, each representing an array of points at which
            to evaluate the basis functions. The dimensions and requirements of these
            inputs vary depending on the specific basis implementation.

        Returns
        -------
        :
            An array containing the evaluated values of the basis functions at the input
            points. The shape and structure of this array are specific to the subclass
            implementation.
        """
        pass

    def _get_samples(self, *n_samples: int) -> Generator[NDArray]:
        """Get equi-spaced samples for all the input dimensions.

        This will be used to evaluate the basis on a grid of
        points derived by the samples.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
        :
            A generator yielding numpy arrays of linspaces from 0 to 1 of sizes specified by `n_samples`.
        """
        # handling of defaults when evaluating on a grid
        # (i.e. when we cannot use max and min of samples)
        if self.bounds is None:
            mn, mx = 0, 1
        else:
            mn, mx = self.bounds
        return (np.linspace(mn, mx, n_samples[k]) for k in range(len(n_samples)))

    @support_pynapple(conv_type="numpy")
    def _check_transform_input(
        self, *xi: ArrayLike
    ) -> Tuple[Union[NDArray, Tsd, TsdFrame]]:
        """Check transform input.

        Parameters
        ----------
        xi[0],...,xi[n] :
            The input samples, each  with shape (number of samples, ).

        Raises
        ------
        ValueError
            - If the time point number is inconsistent between inputs.
            - If the number of inputs doesn't match what the Basis object requires.
            - At least one of the samples is empty.

        """
        # check that the input is array-like (i.e., whether we can cast it to
        # numeric arrays)
        try:
            # make sure array is at least 1d (so that we succeed when only
            # passed a scalar)
            xi = tuple(np.atleast_1d(np.asarray(x, dtype=float)) for x in xi)
        # ValueError here surfaces the exception with e.g., `x=np.array["a", "b"])`
        except (TypeError, ValueError):
            raise TypeError("Input samples must be array-like of floats!")

        # check for non-empty samples
        if self._has_zero_samples(tuple(len(x) for x in xi)):
            raise ValueError("All sample provided must be non empty.")

        # checks on input and outputs
        self._check_samples_consistency(*xi)
        self._check_input_dimensionality(xi)

        return xi

    def _check_has_kernel(self) -> None:
        """Check that the kernel is pre-computed."""
        if self.mode == "conv" and self.kernel_ is None:
            raise ValueError(
                "You must call `_set_kernel` before `_compute_features` when mode =`conv`."
            )

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        The i-th axis of the grid will be sampled with n_samples[i] equi-spaced points.
        The method uses numpy.meshgrid with `indexing="ij"`, returning matrix indexing
        instead of the default cartesian indexing, see Notes.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid. The length of
            n_samples must equal the number of combined bases.

        Returns
        -------
        *Xs :
            A tuple of arrays containing the meshgrid values, one element for each of the n dimension of the grid,
            where n equals to the number of inputs.
            The size of Xs[i] is (n_samples[0], ... , n_samples[n]).
        Y :
            The basis function evaluated at the samples,
            shape (n_samples[0], ... , n_samples[n], number of basis).

        Raises
        ------
        ValueError
            - If the time point number is inconsistent between inputs or if the number of inputs doesn't match what
            the Basis object requires.
            - If one of the n_samples is <= 0.

        Notes
        -----
        Setting "indexing = 'ij'" returns a meshgrid with matrix indexing. In the N-D case with inputs of size
        $M_1,...,M_N$, outputs are of shape $(M_1, M_2, M_3, ....,M_N)$.
        This differs from the numpy.meshgrid default, which uses Cartesian indexing.
        For the same input, Cartesian indexing would return an output of shape $(M_2, M_1, M_3, ....,M_N)$.

        """
        self._check_input_dimensionality(n_samples)

        if self._has_zero_samples(n_samples):
            raise ValueError("All sample counts provided must be greater than zero.")

        # get the samples
        sample_tuple = self._get_samples(*n_samples)
        Xs = np.meshgrid(*sample_tuple, indexing="ij")

        # evaluates the basis on a flat NDArray and reshape to match meshgrid output
        Y = self.__call__(*tuple(grid_axis.flatten() for grid_axis in Xs)).reshape(
            (*n_samples, self.n_basis_funcs)
        )

        return *Xs, Y

    @staticmethod
    def _has_zero_samples(n_samples: Tuple[int, ...]) -> bool:
        return any([n <= 0 for n in n_samples])

    def _check_input_dimensionality(self, xi: Tuple) -> None:
        """
        Check that the number of inputs provided by the user matches the number of inputs required.

        Parameters
        ----------
        xi[0], ..., xi[n] :
            The input samples, shape (number of samples, ).

        Raises
        ------
        ValueError
            If the number of inputs doesn't match what the Basis object requires.
        """
        if len(xi) != self._n_input_dimensionality:
            raise TypeError(
                f"Input dimensionality mismatch. This basis evaluation requires {self._n_input_dimensionality} inputs, "
                f"{len(xi)} inputs provided instead."
            )

    @staticmethod
    def _check_samples_consistency(*xi: NDArray) -> None:
        """
        Check that each input provided to the Basis object has the same number of time points.

        Parameters
        ----------
        xi[0], ..., xi[n] :
            The input samples, shape (number of samples, ).

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs.
        """
        sample_sizes = [sample.shape[0] for sample in xi]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(
                "Sample size mismatch. Input elements have inconsistent sample sizes."
            )

    @abc.abstractmethod
    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Most of the basis work with at least 1 element, but some
        such as the RaisedCosineBasisLog requires a minimum of 2 basis to be well defined.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        pass

    def __add__(self, other: Basis) -> AdditiveBasis:
        """
        Add two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to add.

        Returns
        -------
        : AdditiveBasis
            The resulting Basis object.
        """
        return AdditiveBasis(self, other)

    def __mul__(self, other: Basis) -> MultiplicativeBasis:
        """
        Multiply two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to multiply.

        Returns
        -------
        :
            The resulting Basis object.
        """
        return MultiplicativeBasis(self, other)

    def __pow__(self, exponent: int) -> MultiplicativeBasis:
        """Exponentiation of a Basis object.

        Define the power of a basis by repeatedly applying the method __multiply__.
        The exponent must be a positive integer.

        Parameters
        ----------
        exponent :
            Positive integer exponent

        Returns
        -------
        :
            The product of the basis with itself "exponent" times. Equivalent to self * self * ... * self.

        Raises
        ------
        TypeError
            If the provided exponent is not an integer.
        ValueError
            If the integer is zero or negative.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent should be an integer!")

        if exponent <= 0:
            raise ValueError("Exponent should be a non-negative integer!")

        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

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
        >>> X, y = ...  # X: features, y: neural activity
        >>> basis = nmo.basis.RaisedCosineBasisLinear(10)
        >>> glm = nmo.glm.GLM(regularizer="Ridge")
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
        >>> gridsearch.fit(X, y)
        """

        return TransformerBasis(copy.deepcopy(self))


class AdditiveBasis(Basis):
    """
    Class representing the addition of two Basis objects.

    Parameters
    ----------
    basis1 :
        First basis object to add.
    basis2 :
        Second basis object to add.

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.


    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self.n_basis_funcs = basis1.n_basis_funcs + basis2.n_basis_funcs
        super().__init__(self.n_basis_funcs, mode="eval")
        self._n_input_dimensionality = (
            basis1._n_input_dimensionality + basis2._n_input_dimensionality
        )
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _check_n_basis_min(self) -> None:
        pass

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        """
        X = np.hstack(
            (
                self._basis1.__call__(*xi[: self._basis1._n_input_dimensionality]),
                self._basis2.__call__(*xi[self._basis1._n_input_dimensionality :]),
            )
        )
        if self.identifiability_constraints:
            X = self._apply_identifiability_constraints(X)
        return X

    @check_transform_input
    def _compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Compute features for added bases and concatenate.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The features, shape (n_samples, n_basis_funcs)

        """
        # the numpy conversion is important, there is some in-place
        # array modification in basis.
        hstack_pynapple = support_pynapple(conv_type="numpy")(np.hstack)
        return hstack_pynapple(
            (
                self._basis1._compute_features(
                    *xi[: self._basis1._n_input_dimensionality]
                ),
                self._basis2._compute_features(
                    *xi[self._basis1._n_input_dimensionality :]
                ),
            ),
        )

    def _set_kernel(self, *xi: ArrayLike) -> Basis:
        """Call fit on the added basis.

        If any of the added basis is in "conv" mode, it will prepare its kernels for the convolution.

        Parameters
        ----------
        *xi:
            The sample inputs. Unused, necessary to conform to `scikit-learn` API.

        Returns
        -------
        :
            The AdditiveBasis ready to be evaluated.
        """
        self._basis1._set_kernel(*xi)
        self._basis2._set_kernel(*xi)
        return self


class MultiplicativeBasis(Basis):
    """
    Class representing the multiplication (external product) of two Basis objects.

    Parameters
    ----------
    basis1 :
        First basis object to multiply.
    basis2 :
        Second basis object to multiply.

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.

    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self.n_basis_funcs = basis1.n_basis_funcs * basis2.n_basis_funcs
        super().__init__(self.n_basis_funcs, mode="eval")
        self._n_input_dimensionality = (
            basis1._n_input_dimensionality + basis2._n_input_dimensionality
        )
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _check_n_basis_min(self) -> None:
        pass

    def _set_kernel(self, *xi: NDArray) -> Basis:
        """Call fit on the multiplied basis.

        If any of the added basis is in "conv" mode, it will prepare its kernels for the convolution.

        Parameters
        ----------
        *xi:
            The sample inputs. Unused, necessary to conform to `scikit-learn` API.

        Returns
        -------
        :
            The MultiplicativeBasis ready to be evaluated.
        """
        self._basis1._set_kernel(*xi)
        self._basis2._set_kernel(*xi)
        return self

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)
        """
        X = np.asarray(
            row_wise_kron(
                self._basis1.__call__(*xi[: self._basis1._n_input_dimensionality]),
                self._basis2.__call__(*xi[self._basis1._n_input_dimensionality :]),
                transpose=False,
            )
        )
        if self.identifiability_constraints:
            X = self._apply_identifiability_constraints(X)
        return X

    @check_transform_input
    def _compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Compute the features for the multiplied bases, and compute their outer product.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The  features, shape (n_samples, n_basis_funcs)

        """
        kron = support_pynapple(conv_type="numpy")(row_wise_kron)
        return kron(
            self._basis1._compute_features(*xi[: self._basis1._n_input_dimensionality]),
            self._basis2._compute_features(*xi[self._basis1._n_input_dimensionality :]),
            transpose=False,
        )


class SplineBasis(Basis, abc.ABC):
    """
    SplineBasis class inherits from the Basis class and represents spline basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    order : optional
        Spline order.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`

    Attributes
    ----------
    order : int
        Spline order.

    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        order: int = 2,
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        self.order = order
        super().__init__(
            n_basis_funcs,
            mode=mode,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )
        self._n_input_dimensionality = 1
        if self.order < 1:
            raise ValueError("Spline order must be positive!")

    def _generate_knots(
        self,
        sample_pts: NDArray,
        perc_low: float = 0.0,
        perc_high: float = 1.0,
        is_cyclic: bool = False,
    ) -> NDArray:
        """
        Generate knot locations for spline basis functions.

        Parameters
        ----------
        sample_pts : (n_samples,)
            The sample points.
        perc_low
            The low percentile value, between [0,1).
        perc_high
            The high percentile value, between (0,1].
        is_cyclic : optional
            Whether the spline is cyclic.

        Returns
        -------
            The knot locations for the spline basis functions.

        Raises
        ------
        AssertionError
            If the percentiles or order values are not within the valid range.
        """
        # Determine number of interior knots.
        num_interior_knots = self.n_basis_funcs - self.order
        if is_cyclic:
            num_interior_knots += self.order - 1

        # Spline basis have support on the semi-open [a, b)  interval, we add a small epsilon
        # to mx so that the so that basis_element(max(samples)) != 0
        knot_locs = np.concatenate(
            (
                np.zeros(self.order - 1),
                np.linspace(0, (1 + np.finfo(float).eps), num_interior_knots + 2),
                np.full(self.order - 1, 1 + np.finfo(float).eps),
            )
        )
        return knot_locs

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Check that the spline-basis has at least as many basis as the order.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < self.order:
            raise ValueError(
                f"{self.__class__.__name__} `order` parameter cannot be larger "
                "than `n_basis_funcs` parameter."
            )


class MSplineBasis(SplineBasis):
    r"""
    M-spline[$^1$](#references) basis functions for modeling and data transformation.

    M-splines are a type of spline basis function used for smooth curve fitting
    and data representation. They are positive and integrate to one, making them
    suitable for probabilistic models and density estimation. The order of an
    M-spline defines its smoothness, with higher orders resulting in smoother
    splines.

    This class provides functionality to create M-spline basis functions, allowing
    for flexible and smooth modeling of data. It inherits from the `SplineBasis`
    abstract class, providing specific implementations for M-splines.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions to generate. More basis functions allow for
        more flexible data modeling but can lead to overfitting.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    order :
        The order of the splines used in basis functions. Must be between [1,
        n_basis_funcs]. Default is 2. Higher order splines have more continuous
        derivatives at each interior knot, resulting in smoother basis functions.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs:
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import MSplineBasis
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> mspline_basis = MSplineBasis(n_basis_funcs, order=order)
    >>> sample_points = linspace(0, 1, 100)
    >>> basis_functions = mspline_basis(sample_points)

    References
    ----------
    [1] Ramsay, J. O. (1988). Monotone regression splines in action. Statistical science,
        3(4), 425-441.

    Notes
    -----
    MSplines must integrate to 1 over their domain (the area under the curve is 1). Therefore, if the domain
    (x-axis) of an MSpline basis is expanded by a factor of $\alpha$, the values on the co-domain (y-axis) values
    will shrink by a factor of $1/\alpha$.
    For example, over the standard bounds of (0, 1), the maximum value of the MSpline is 18.
    If we set the bounds to (0, 2), the maximum value will be 9, i.e., 18 / 2.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        order: int = 2,
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_basis_funcs,
            mode=mode,
            order=order,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(self, sample_pts: ArrayLike) -> FeatureMatrix:
        """
        Evaluate the M-spline basis functions at given sample points.

        Parameters
        ----------
        sample_pts :
            An array of sample points where the M-spline basis functions are to be
            evaluated.

        Returns
        -------
        :
            An array where each column corresponds to one M-spline basis function
            evaluated at the input sample points. The shape of the array is
            (len(sample_pts), n_basis_funcs).

        Notes
        -----
        The implementation uses a recursive definition of M-splines. Boundary
        conditions are handled such that the basis functions are positive and
        integrate to one over the domain defined by the sample points.
        """
        sample_pts, scaling = min_max_rescale_samples(sample_pts, self.bounds)
        # add knots if not passed
        knot_locs = self._generate_knots(
            sample_pts, perc_low=0.0, perc_high=1.0, is_cyclic=False
        )

        X = np.stack(
            [
                mspline(sample_pts, self.order, i, knot_locs)
                for i in range(self.n_basis_funcs)
            ],
            axis=1,
        )
        # re-normalize so that it integrates to 1 over the range.
        X /= scaling
        if self.identifiability_constraints:
            X = self._apply_identifiability_constraints(X)
        return X

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Evaluate the M-spline basis functions on a uniformly spaced grid.

        This method creates a uniformly spaced grid of sample points within the domain
        [0, 1] and evaluates all the M-spline basis functions at these points. It is
        particularly useful for visualizing the shape and distribution of the basis
        functions across their domain.

        Parameters
        ----------
        n_samples :
            The number of points in the uniformly spaced grid. A higher number of
            samples will result in a more detailed visualization of the basis functions.

        Returns
        -------
        X : NDArray
            A 1D array of uniformly spaced sample points within the domain [0, 1].
            Shape: `(n_samples,)`.
        Y : NDArray
            A 2D array where each row corresponds to the evaluated M-spline basis
            function values at the points in X. Shape: `(n_samples, n_basis_funcs)`.

        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import MSplineBasis
        >>> mspline_basis = MSplineBasis(n_basis_funcs=4, order=3)
        >>> sample_points, basis_values = mspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('M-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        >>> plt.ylabel('Basis Function Value')
        >>> plt.legend()
        >>> plt.show()
        """
        return super().evaluate_on_grid(n_samples)


class BSplineBasis(SplineBasis):
    """
    B-spline[$^1$](#references) 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    order :
        Order of the splines used in basis functions. Must lie within [1, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`

    Attributes
    ----------
    order :
        Spline order.


    References
    ----------
    1. Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
        Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5

    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        order: int = 4,
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            order=order,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(self, sample_pts: ArrayLike) -> FeatureMatrix:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the B-spline is evaluated, shape (n_samples,).

        Returns
        -------
        basis_funcs :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs).

        Raises
        ------
        AssertionError
            If the sample points are not within the B-spline knots.

        Notes
        -----
        The evaluation is performed by looping over each element and using `splev`
        from SciPy to compute the basis values.
        """
        sample_pts, _ = min_max_rescale_samples(sample_pts, self.bounds)
        # add knots
        knot_locs = self._generate_knots(sample_pts, 0.0, 1.0)

        basis_eval = bspline(
            sample_pts, knot_locs, order=self.order, der=0, outer_ok=False
        )

        if self.identifiability_constraints:
            basis_eval = self._apply_identifiability_constraints(basis_eval)
        return basis_eval

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the B-spline basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of samples.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Raised cosine basis functions, shape (n_samples, n_basis_funcs)

        Notes
        -----
        The evaluation is performed by looping over each element and using `splev` from
        SciPy to compute the basis values.
        """
        return super().evaluate_on_grid(n_samples)


class CyclicBSplineBasis(SplineBasis):
    """
    B-spline 1-dimensional basis functions for cyclic splines.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    order :
        Order of the splines used in basis functions. Order must lie within [2, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.
    order : int
        Order of the splines used in basis functions.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        order: int = 4,
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            order=order,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )
        if self.order < 2:
            raise ValueError(
                f"Order >= 2 required for cyclic B-spline, "
                f"order {self.order} specified instead!"
            )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(
        self,
        sample_pts: ArrayLike,
    ) -> FeatureMatrix:
        """Evaluate the Cyclic B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the cyclic B-spline is evaluated, shape
            (n_samples,).

        Returns
        -------
        basis_funcs :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        Notes
        -----
        The evaluation is performed by looping over each element and using `splev` from
        SciPy to compute the basis values.

        """
        sample_pts, _ = min_max_rescale_samples(sample_pts, self.bounds)
        knot_locs = self._generate_knots(sample_pts, 0.0, 1.0, is_cyclic=True)

        # for cyclic, do not repeat knots
        knot_locs = np.unique(knot_locs)

        nk = knot_locs.shape[0]

        # make sure knots are sorted
        knot_locs.sort()

        # extend knots
        xc = knot_locs[nk - self.order]
        knots = np.hstack(
            (
                knot_locs[0] - knot_locs[-1] + knot_locs[nk - self.order : nk - 1],
                knot_locs,
            )
        )

        ind = sample_pts > xc

        basis_eval = bspline(sample_pts, knots, order=self.order, der=0, outer_ok=True)
        sample_pts[ind] = sample_pts[ind] - knots.max() + knot_locs[0]

        if np.sum(ind):
            basis_eval[ind] = basis_eval[ind] + bspline(
                sample_pts[ind], knots, order=self.order, outer_ok=True, der=0
            )
        # restore points
        sample_pts[ind] = sample_pts[ind] + knots.max() - knot_locs[0]
        if self.identifiability_constraints:
            basis_eval = self._apply_identifiability_constraints(basis_eval)

        return basis_eval

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the Cyclic B-spline basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of samples.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Raised cosine basis functions, shape (n_samples, n_basis_funcs)

        Notes
        -----
        The evaluation is performed by looping over each element and using `splev` from
        SciPy to compute the basis values.
        """
        return super().evaluate_on_grid(n_samples)


class RaisedCosineBasisLinear(Basis):
    """Represent linearly-spaced raised cosine basis functions.

    This implementation is based on the cosine bumps used by Pillow et al.[$^1$](#references)
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    width :
        Width of the raised cosine. By default, it's set to 2.0.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`

    References
    ----------
    1. Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
        C. E. (2005). Prediction and decoding of retinal ganglion cell responses
        with a probabilistic spiking model. Journal of Neuroscience, 25(47),
        11003–11013. http://dx.doi.org/10.1523/jneurosci.3305-05.2005
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        width: float = 2.0,
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_basis_funcs,
            mode=mode,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )
        self._n_input_dimensionality = 1
        self._check_width(width)
        self._width = width
        # for these linear raised-cosine basis functions,
        # the samples must be rescaled to 0 and 1.
        self._rescale_samples = True

    @property
    def width(self):
        """Return width of the raised cosine."""
        return self._width

    @width.setter
    def width(self, width: float):
        self._check_width(width)
        self._width = width

    @staticmethod
    def _check_width(width: float) -> None:
        """Validate the width value.

        Parameters
        ----------
        width :
            The width value to validate.

        Raises
        ------
        ValueError
            If width <= 1 or 2*width is not a positive integer. Values that do not match
            this constraint will result in:
            - No overlap between bumps (width < 1).
            - Oscillatory behavior when summing the basis elements (2*width not integer).
        """
        if width <= 1 or (not np.isclose(width * 2, round(2 * width))):
            raise ValueError(
                f"Invalid raised cosine width. "
                f"2*width must be a positive integer, 2*width = {2 * width} instead!"
            )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(
        self,
        sample_pts: ArrayLike,
    ) -> FeatureMatrix:
        """Generate basis functions with given samples.

        Parameters
        ----------
        sample_pts :
            Spacing for basis functions, holding elements on interval [0, 1], Shape (number of samples, ).

        Raises
        ------
        ValueError
            If the sample provided do not lie in [0,1].

        """
        if self._rescale_samples:
            # note that sample points is converted to NDArray
            # with the decorator.
            # copy is necessary otherwise:
            # basis1 = nmo.basis.RaisedCosineBasisLinear(5)
            # basis2 = nmo.basis.RaisedCosineBasisLog(5)
            # additive_basis = basis1 + basis2
            # additive_basis(*([x] * 2)) would modify both inputs
            sample_pts, _ = min_max_rescale_samples(np.copy(sample_pts), self.bounds)

        peaks = self._compute_peaks()
        delta = peaks[1] - peaks[0]
        # generate a set of shifted cosines, and constrain them to be non-zero
        # over a single period, then enforce the codomain to be [0,1], by adding 1
        # and then multiply by 0.5
        basis_funcs = 0.5 * (
            np.cos(
                np.clip(
                    np.pi * (sample_pts[:, None] - peaks[None]) / (delta * self.width),
                    -np.pi,
                    np.pi,
                )
            )
            + 1
        )
        if self.identifiability_constraints:
            basis_funcs = self._apply_identifiability_constraints(basis_funcs)
        return basis_funcs

    def _compute_peaks(self) -> NDArray:
        """
        Compute the location of raised cosine peaks.

        Returns
        -------
            Peak locations of each basis element.
        """
        return np.linspace(0, 1, self.n_basis_funcs)

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of samples.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Raised cosine basis functions, shape (n_samples, n_basis_funcs)

        """
        return super().evaluate_on_grid(n_samples)

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Check that the number of basis is at least 2.

        Raises
        ------
        ValueError
            If n_basis_funcs < 2.
        """
        if self.n_basis_funcs < 2:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 2 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )


class RaisedCosineBasisLog(RaisedCosineBasisLinear):
    """Represent log-spaced raised cosine basis functions.

    Similar to `RaisedCosineBasisLinear` but the basis functions are log-spaced.
    This implementation is based on the cosine bumps used by Pillow et al.[$^1$](#references)
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    width :
        Width of the raised cosine.
    time_scaling :
        Non-negative hyper-parameter controlling the logarithmic stretch magnitude, with
        larger values resulting in more stretching. As this approaches 0, the
        transformation becomes linear.
    enforce_decay_to_zero:
        If set to True, the algorithm first constructs a basis with `n_basis_funcs + ceil(width)` elements
        and subsequently trims off the extra basis elements. This ensures that the final basis element
        decays to 0.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`

    References
    ----------
    1. Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013. http://dx.doi.org/10.1523/jneurosci.3305-05.2005
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        width: float = 2.0,
        time_scaling: float = None,
        enforce_decay_to_zero: bool = True,
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_basis_funcs,
            mode=mode,
            width=width,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )
        # The samples are scaled appropriately in the self._transform_samples which scales
        # and applies the log-stretch, no additional transform is needed.
        self._rescale_samples = False

        self.enforce_decay_to_zero = enforce_decay_to_zero
        if time_scaling is None:
            self._time_scaling = 50.0
        else:
            self._check_time_scaling(time_scaling)
            self._time_scaling = time_scaling

    @property
    def time_scaling(self):
        """Getter property for time_scaling."""
        return self._time_scaling

    @staticmethod
    def _check_time_scaling(time_scaling: float) -> None:
        if time_scaling <= 0:
            raise ValueError(
                f"Only strictly positive time_scaling are allowed, {time_scaling} provided instead."
            )

    def _transform_samples(
        self,
        sample_pts: ArrayLike,
    ) -> NDArray:
        """
        Map the sample domain to log-space.

        Parameters
        ----------
        sample_pts :
            Sample points used for evaluating the splines,
            shape (n_samples, ).

        Returns
        -------
            Transformed version of the sample points that matches the Raised Cosine basis domain,
            shape (n_samples, ).
        """
        # rescale to [0,1]
        # copy is necessary to avoid unwanted rescaling in additive/multiplicative basis.
        sample_pts, _ = min_max_rescale_samples(np.copy(sample_pts), self.bounds)
        # This log-stretching of the sample axis has the following effect:
        # - as the time_scaling tends to 0, the points will be linearly spaced across the whole domain.
        # - as the time_scaling tends to inf, basis will be small and dense around 0 and
        # progressively larger and less dense towards 1.
        log_spaced_pts = np.log(self.time_scaling * sample_pts + 1) / np.log(
            self.time_scaling + 1
        )
        return log_spaced_pts

    def _compute_peaks(self) -> NDArray:
        """
        Peak location of each log-spaced cosine basis element.

        Compute the peak location for the log-spaced raised cosine basis.
        Enforcing that the last basis decays to zero is equivalent to
        setting the last peak to a value smaller than 1.

        Returns
        -------
            Peak locations of each basis element.

        """
        if self.enforce_decay_to_zero:
            # compute the last peak location such that the last
            # basis element decays to zero at the last sample.
            last_peak = 1 - self.width / (self.n_basis_funcs + self.width - 1)
        else:
            last_peak = 1
        return np.linspace(0, last_peak, self.n_basis_funcs)

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(
        self,
        sample_pts: ArrayLike,
    ) -> FeatureMatrix:
        """Generate log-spaced raised cosine basis with given samples.

        Parameters
        ----------
        sample_pts :
            Spacing for basis functions. Samples will be rescaled to the interval [0, 1].

        Returns
        -------
        basis_funcs :
            Log-raised cosine basis functions, shape (n_samples, n_basis_funcs).

        Raises
        ------
        ValueError
            If the sample provided do not lie in [0,1].
        """
        return super().__call__(self._transform_samples(sample_pts))


class OrthExponentialBasis(Basis):
    """Set of 1D basis decaying exponential functions numerically orthogonalized.

    Parameters
    ----------
    n_basis_funcs
            Number of basis functions.
    decay_rates :
            Decay rates of the exponentials, shape (n_basis_funcs,).
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    bounds :
        The bounds for the basis domain in `mode="eval"`. The default `bounds[0]` and `bounds[1]` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bonuds, the basis will return NaN.
    **kwargs :
        Only used in "conv" mode. Additional keyword arguments that are passed to
        `nemos.convolve.create_convolutional_predictor`
    """

    def __init__(
        self,
        n_basis_funcs: int,
        decay_rates: NDArray[np.floating],
        mode="eval",
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            window_size=window_size,
            bounds=bounds,
            **kwargs,
        )
        self.decay_rates = decay_rates
        self._check_rates()
        self._n_input_dimensionality = 1

    @property
    def decay_rates(self):
        """Decay rate getter"""
        return self._decay_rates

    @decay_rates.setter
    def decay_rates(self, value: NDArray):
        """Decay rate setter."""
        value = np.asarray(value)
        if value.shape[0] != self.n_basis_funcs:
            raise ValueError(
                f"The number of basis functions must match the number of decay rates provided. "
                f"Number of basis functions provided: {self.n_basis_funcs}, "
                f"Number of decay rates provided: {value.shape[0]}"
            )
        self._decay_rates = value

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Checks that the number of basis is at least 1.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < 1:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 1 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )

    def _check_rates(self) -> None:
        """
        Check if the decay rates list has duplicate entries.

        Raises
        ------
        ValueError
            If two or more decay rates are repeated, which would result in a linearly
            dependent set of functions for the basis.
        """
        if len(set(self._decay_rates)) != len(self._decay_rates):
            raise ValueError(
                "Two or more rate are repeated! Repeating rate will result in a "
                "linearly dependent set of function for the basis."
            )

    def _check_sample_size(self, *sample_pts: NDArray) -> None:
        """Check that the sample size is greater than the number of basis.

        This is necessary for the orthogonalization procedure,
        that otherwise will return (sample_size, ) basis elements instead of the expected number.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval [0, inf).

        Raises
        ------
        ValueError
            If the number of basis element is less than the number of samples.
        """
        if sample_pts[0].size < self.n_basis_funcs:
            raise ValueError(
                "OrthExponentialBasis requires at least as many samples as basis functions!\n"
                f"Class instantiated with {self.n_basis_funcs} basis functions "
                f"but only {sample_pts[0].size} samples provided!"
            )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(
        self,
        sample_pts: NDArray,
    ) -> FeatureMatrix:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval [0,
            inf), shape (n_samples,).

        Returns
        -------
        basis_funcs
            Evaluated exponentially decaying basis functions, numerically
            orthogonalized, shape (n_samples, n_basis_funcs)

        """
        self._check_sample_size(sample_pts)
        sample_pts, _ = min_max_rescale_samples(sample_pts, self.bounds)
        valid_idx = ~np.isnan(sample_pts)
        # because of how scipy.linalg.orth works, have to create a matrix of
        # shape (n_pts, n_basis_funcs) and then transpose, rather than
        # directly computing orth on the matrix of shape (n_basis_funcs,
        # n_pts)
        exp_decay_eval = np.stack(
            [np.exp(-lam * sample_pts[valid_idx]) for lam in self._decay_rates], axis=1
        )
        # count the linear independent components (could be lower than n_basis_funcs for num precision).
        n_independent_component = np.linalg.matrix_rank(exp_decay_eval)
        # initialize output to nan
        basis_funcs = np.full(
            shape=(sample_pts.shape[0], n_independent_component), fill_value=np.nan
        )
        # orthonormalize on valid points
        basis_funcs[valid_idx] = scipy.linalg.orth(exp_decay_eval)
        if self.identifiability_constraints:
            basis_funcs = self._apply_identifiability_constraints(basis_funcs)
        return basis_funcs

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of samples.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Evaluated exponentially decaying basis functions, numerically
            orthogonalized, shape (n_samples, n_basis_funcs)

        """
        return super().evaluate_on_grid(n_samples)


def mspline(x: NDArray, k: int, i: int, T: NDArray) -> NDArray:
    """Compute M-spline basis function.

    Parameters
    ----------
    x
        Spacing for basis functions, shape (n_sample_points, ).
    k
        Order of the spline basis.
    i
        Number of the spline basis.
    T
        knot locations. should lie in interval [0, 1], shape (k + n_basis_funcs,).

    Returns
    -------
    spline
        M-spline basis function, shape (n_sample_points, ).
    """
    # Boundary conditions.
    if (T[i + k] - T[i]) < 1e-6:
        return np.zeros_like(x)

    # Special base case of first-order spline basis.
    elif k == 1:
        v = np.zeros_like(x)
        v[(x >= T[i]) & (x < T[i + 1])] = 1 / (T[i + 1] - T[i])
        return v

    # General case, defined recursively
    else:
        return (
            k
            * (
                (x - T[i]) * mspline(x, k - 1, i, T)
                + (T[i + k] - x) * mspline(x, k - 1, i + 1, T)
            )
            / ((k - 1) * (T[i + k] - T[i]))
        )


def bspline(
    sample_pts: NDArray,
    knots: NDArray,
    order: int = 4,
    der: int = 0,
    outer_ok: bool = False,
) -> NDArray:
    """
    Calculate and return the evaluation of B-spline basis.

    This function evaluates B-spline basis for given sample points. It checks for
    out of range points and optionally handles them. It also handles the NaNs if present.

    Parameters
    ----------
    sample_pts :
        An array containing sample points for which B-spline basis needs to be evaluated,
        shape (n_samples,)
    knots :
        An array containing knots for the B-spline basis. The knots are sorted in ascending order.
    order :
        The order of the B-spline basis.
    der :
        The derivative of the B-spline basis to be evaluated.
    outer_ok :
        If True, allows for evaluation at points outside the range of knots.
        Default is False, in which case an assertion error is raised when
        points outside the knots range are encountered.

    Returns
    -------
    basis_eval :
        An array containing the evaluation of B-spline basis for the given sample points.
        Shape (n_samples, n_basis_funcs).

    Raises
    ------
    AssertionError
        If `outer_ok` is False and the sample points lie outside the B-spline knots range.

    Notes
    -----
    The function uses splev function from scipy.interpolate library for the basis evaluation.
    """
    knots.sort()
    nk = knots.shape[0]

    # check for out of range points (in cyclic b-spline need_outer must be set to False)
    need_outer = any(sample_pts < knots[order - 1]) or any(
        sample_pts > knots[nk - order]
    )
    assert (
        not need_outer
    ) | outer_ok, 'sample points must lie within the B-spline knots range unless "outer_ok==True".'

    # select knots that are within the knots range (this takes care of eventual NaNs)
    in_sample = (sample_pts >= knots[0]) & (sample_pts <= knots[-1])

    if need_outer:
        reps = order - 1
        knots = np.hstack((np.ones(reps) * knots[0], knots, np.ones(reps) * knots[-1]))
        nk = knots.shape[0]
    else:
        reps = 0

    # number of basis elements
    n_basis = nk - order

    # initialize the basis element container
    basis_eval = np.full((n_basis - 2 * reps, sample_pts.shape[0]), np.nan)

    # loop one element at the time and evaluate the basis using splev
    id_basis = np.eye(n_basis, nk, dtype=np.int8)
    for i in range(reps, len(knots) - order - reps):
        basis_eval[i - reps, in_sample] = splev(
            sample_pts[in_sample], (knots, id_basis[i], order - 1), der=der
        )

    return basis_eval.T
