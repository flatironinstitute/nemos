from __future__ import annotations

import copy
from functools import wraps
from typing import TYPE_CHECKING, Generator

import numpy as np

from ..typing import FeatureMatrix

if TYPE_CHECKING:
    from ._basis import Basis


def transformer_chaining(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the wrapped function and capture its return value
        result = func(*args, **kwargs)

        # If the method returns the inner `self`, replace it with the outer `self` (no deepcopy here).
        return self if result is self.basis else result

    return wrapper


class TransformerBasis:
    """Basis as ``scikit-learn`` transformers.

    This class abstracts the underlying basis function details, offering methods
    similar to scikit-learn's transformers but specifically designed for basis
    transformations. It supports fitting to data (calculating any necessary parameters
    of the basis functions), transforming data (applying the basis functions to
    data), and both fitting and transforming in one step.

    ``TransformerBasis``, unlike ``Basis``, is compatible with scikit-learn pipelining and
    model selection, enabling the cross-validation of the basis type and parameters,
    for example ``n_basis_funcs``. See the example section below.

    Parameters
    ----------
    basis :
        A concrete subclass of ``Basis``.

    Examples
    --------
    >>> from nemos.basis import BSplineEval
    >>> from nemos.basis import TransformerBasis
    >>> from nemos.glm import GLM
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import GridSearchCV
    >>> import numpy as np
    >>> np.random.seed(123)

    >>> # Generate data
    >>> num_samples, num_features = 10000, 1
    >>> x = np.random.normal(size=(num_samples, ))  # raw time series
    >>> basis = BSplineEval(10)
    >>> features = basis.compute_features(x)  # basis transformed time series
    >>> weights = np.random.normal(size=basis.n_basis_funcs)  # true weights
    >>> y = np.random.poisson(np.exp(features.dot(weights)))  # spike counts

    >>> # transformer can be used in pipelines
    >>> transformer = TransformerBasis(basis)
    >>> pipeline = Pipeline([ ("compute_features", transformer), ("glm", GLM()),])
    >>> pipeline = pipeline.fit(x[:, None], y)  # x need to be 2D for sklearn transformer API
    >>> out = pipeline.predict(np.arange(10)[:, None]) # predict rate from new datas
    >>> # TransformerBasis parameter can be cross-validated.
    >>> # 5-fold cross-validate the number of basis
    >>> param_grid = dict(compute_features__n_basis_funcs=[4, 10])
    >>> grid_cv = GridSearchCV(pipeline, param_grid, cv=5)
    >>> grid_cv = grid_cv.fit(x[:, None], y)
    >>> print("Cross-validated number of basis:", grid_cv.best_params_)
    Cross-validated number of basis: {'compute_features__n_basis_funcs': 10}
    """

    _chainable_methods = (
        "set_kernel",
        "set_input_shape",
        "_set_input_independent_states",
        "setup_basis",
    )

    def __init__(self, basis: Basis):
        self.basis = copy.deepcopy(basis)
        self._wrapped_methods = {}  # Cache for wrapped methods

    @staticmethod
    def _check_initialized(basis):
        if basis._input_shape_product is None:
            raise RuntimeError(
                "Cannot apply TransformerBasis: the provided basis has no defined input shape. "
                "Please call `set_input_shape` before calling `fit`, `transform`, or "
                "`fit_transform`."
            )

    def _unpack_inputs(self, X: FeatureMatrix) -> Generator:
        """Unpack inputs.

        Unpack horizontally stacked inputs using slicing. This works gracefully with ``pynapple``,
        returning a list of Tsd objects.

        Parameters
        ----------
        X:
            The inputs horizontally stacked.

        Returns
        -------
        :
            A generator looping on each individual input.

        """
        n_samples = X.shape[0]
        out = (
            np.reshape(X[:, cc : cc + n_input], (n_samples, *bas.input_shape))
            for i, (bas, n_input) in enumerate(
                zip(self._iterate_over_components(), self._input_shape_product)
            )
            for cc in [sum(self._input_shape_product[:i])]
        )
        return out

    def fit(self, X: FeatureMatrix, y=None):
        """
        Check the input structure and, if necessary, compute the convolutional kernels.

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

        Raises
        ------
        RuntimeError
            If ``self.n_basis_input`` is None. Call ``self.set_input_shape`` before calling ``fit`` to avoid this.
        ValueError:
            If the number of columns in X do not ``self.n_basis_input_``.

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineEval, TransformerBasis

        >>> # Example input
        >>> X = np.random.normal(size=(100, 2))

        >>> # Define and fit tranformation basis
        >>> basis = MSplineEval(10).set_input_shape(2)
        >>> transformer = TransformerBasis(basis)
        >>> transformer_fitted = transformer.fit(X)
        """
        self._check_initialized(self.basis)
        self._check_input(X, y)
        self.basis.setup_basis(*self._unpack_inputs(X))
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

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineConv, TransformerBasis

        >>> # Example input
        >>> X = np.random.normal(size=(10000, 2))

        >>> basis = MSplineConv(10, window_size=200).set_input_shape(2)
        >>> transformer = TransformerBasis(basis)
        >>> # Before calling `fit` the convolution kernel is not set
        >>> transformer.kernel_

        >>> transformer_fitted = transformer.fit(X)
        >>> # Now the convolution kernel is initialized and has shape (window_size, n_basis_funcs)
        >>> transformer_fitted.kernel_.shape
        (200, 10)

        >>> # Transform basis
        >>> feature_transformed = transformer.transform(X)
        """
        self._check_initialized(self.basis)
        self._check_input(X, y)
        # transpose does not work with pynapple
        # can't use func(*X.T) to unwrap
        return self.basis._compute_features(*self._unpack_inputs(X))

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

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineEval, TransformerBasis

        >>> # Example input
        >>> X = np.random.normal(size=(100, 1))

        >>> # Define tranformation basis
        >>> basis = MSplineEval(10).set_input_shape(1)
        >>> transformer = TransformerBasis(basis)

        >>> # Fit and transform basis
        >>> feature_transformed = transformer.fit_transform(X)
        """
        self.fit(X, y=y)
        return self.transform(X)

    def __getstate__(self):
        """
        Explicitly define how to pickle TransformerBasis object.

        See https://docs.python.org/3/library/pickle.html#object.__getstate__
        and https://docs.python.org/3/library/pickle.html#pickle-state
        """
        # this is the only state needed at initalization
        # returning the cached wrapped methods would create
        # a circular binding of the state to self (i.e. infinite recursion when
        # unpickling).
        return {"basis": self.basis}

    def __setstate__(self, state):
        """
        Define how to populate the object's state when unpickling.

        Note that during unpickling a new object is created without calling __init__.
        Needed to avoid infinite recursion in __getattr__ when unpickling.

        See https://docs.python.org/3/library/pickle.html#object.__setstate__
        and https://docs.python.org/3/library/pickle.html#pickle-state
        """
        self.basis = state["basis"]
        self._wrapped_methods = {}  # Reinitialize the cache

    def __getattr__(self, name: str):
        """
        Enable easy access to attributes of the underlying Basis object.

        This method caches all chainable methods (methods returning self) in a dicitonary.
        These methods are created the first time they are accessed by decorating the `self.basis.name`
        and cached for future use.

        Examples
        --------
        >>> from nemos import basis
        >>> bas = basis.RaisedCosineLinearEval(5)
        >>> trans_bas = basis.TransformerBasis(bas)
        >>> bas.n_basis_funcs
        5
        >>> trans_bas.n_basis_funcs
        5
        """
        # Check if the method has already been wrapped
        if name in self._wrapped_methods:
            return self._wrapped_methods[name]

        if not hasattr(self.basis, name) or name == "to_transformer":
            raise AttributeError(f"'TransformerBasis' object has no attribute '{name}'")

        # Get the original attribute from the basis
        attr = getattr(self.basis, name)

        # If the attribute is a callable method, wrap it dynamically
        if name in self._chainable_methods:
            wrapped = transformer_chaining(attr).__get__(self)
            self._wrapped_methods[name] = wrapped  # Cache the wrapped method
            return wrapped

        # For non-callable attributes, return them directly
        return attr

    def __setattr__(self, name: str, value) -> None:
        r"""
        Allow setting basis or the attributes of basis with a convenient dot assignment syntax.

        Setting any other attribute is not allowed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the attribute being set is not ``basis`` or an attribute of ``basis``.

        Examples
        --------
        >>> import nemos as nmo
        >>> trans_bas = nmo.basis.TransformerBasis(nmo.basis.MSplineEval(10))
        >>> # allowed
        >>> trans_bas.basis = nmo.basis.BSplineEval(10)
        >>> # allowed
        >>> trans_bas.n_basis_funcs = 20
        >>> # not allowed
        >>> try:
        ...     trans_bas.rand_atrr = "some value"
        ... except ValueError as e:
        ...     print(repr(e))
        ValueError('Only setting basis or existing attributes of basis is allowed. Attempt to set `rand_atrr`.')
        """
        # allow self.basis = basis and other attrs of self to be retrievable
        if name in ["basis", "_wrapped_methods"]:
            super().__setattr__(name, value)
        # allow changing existing attributes of self.basis
        elif hasattr(self.basis, name):
            setattr(self.basis, name, value)
        # don't allow setting any other attribute
        else:
            raise ValueError(
                f"Only setting basis or existing attributes of basis is allowed. Attempt to set `{name}`."
            )

    def __sklearn_clone__(self) -> TransformerBasis:
        """
        Customize how TransformerBasis objects are cloned when used with sklearn.model_selection.

        By default, scikit-learn tries to clone the object by calling __init__ using the output of get_params,
        which fails in our case.

        For more info: https://scikit-learn.org/stable/developers/develop.html#cloning
        """
        cloned_obj = TransformerBasis(self.basis.__sklearn_clone__())
        return cloned_obj

    def __repr__(self):
        return f"Transformer({self.basis})"

    def set_params(self, **parameters) -> TransformerBasis:
        """
        Set TransformerBasis parameters.

        When used with ``sklearn.model_selection``, users can set either the ``basis`` attribute directly
        or the parameters of the underlying Basis, but not both.

        Examples
        --------
        >>> from nemos.basis import BSplineEval, MSplineEval, TransformerBasis
        >>> basis = MSplineEval(10)
        >>> transformer_basis = TransformerBasis(basis=basis)

        >>> # setting parameters of basis is allowed
        >>> print(transformer_basis.set_params(n_basis_funcs=8).n_basis_funcs)
        8
        >>> # setting basis directly is allowed
        >>> print(type(transformer_basis.set_params(basis=BSplineEval(10)).basis))
        <class 'nemos.basis.basis.BSplineEval'>
        >>> # mixing is not allowed, this will raise an exception
        >>> try:
        ...     transformer_basis.set_params(basis=BSplineEval(10), n_basis_funcs=2)
        ... except ValueError as e:
        ...     print(repr(e))
        ValueError('Set either new basis object or parameters for existing basis, not both.')
        """
        new_basis = parameters.pop("basis", None)
        if new_basis is not None:
            self.basis = new_basis
            if len(parameters) > 0:
                raise ValueError(
                    "Set either new basis object or parameters for existing basis, not both."
                )
        else:
            self.basis = self.basis.set_params(**parameters)

        return self

    def get_params(self, deep: bool = True) -> dict:
        """Extend the dict of parameters from the underlying Basis with basis."""
        return {"basis": self.basis, **self.basis.get_params(deep)}

    def __dir__(self) -> list[str]:
        """Extend the list of properties of methods with the ones from the underlying Basis."""
        unique_attrs = set(list(super().__dir__()) + list(self.basis.__dir__()))
        # discard without raising errors if not present
        unique_attrs.discard("to_transformer")
        return list(unique_attrs)

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
        return TransformerBasis(self.basis + other.basis)

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
        return TransformerBasis(self.basis * other.basis)

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
        return TransformerBasis(self.basis**exponent)

    def _check_input(self, X: FeatureMatrix, y=None):
        """Check that the input structure.

        TransformerBasis expects a 2-d array as an input. The number of columns should match the number of inputs
        the basis expects. This number can be set before the TransformerBasis is initialized, by calling
        ``Basis.set_input_shape``.

        Parameters
        ----------
        X:
            The input FeatureMatrix.

        Raises
        ------
        ValueError:
            If the input is not a 2-d array or if the number of columns does not match the expected number of inputs.
        """
        ndim = getattr(X, "ndim", None)
        if ndim is None:
            raise ValueError("The input must be a 2-dimensional array.")

        elif ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, shape (n_samples, n_features). The provided X has shape {X.shape} instead."
            )

        if X.shape[1] != sum(self._input_shape_product):
            raise ValueError(
                f"Input mismatch: expected {sum(self._input_shape_product)} inputs, but got {X.shape[1]} "
                f"columns in X.\nTo modify the required number of inputs, call `set_input_shape` before using "
                f"`fit` or `fit_transform`."
            )

        if y is not None and y.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples. "
                f"X has {X.shape[0]} samples, while y has {y.shape[0]} samples."
            )
