"""Mixin classes for basis."""

from __future__ import annotations

import copy
import inspect
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..convolve import create_convolutional_predictor
from ._transformer_basis import TransformerBasis


class EvalBasisMixin:
    """Mixin class for evaluational basis."""

    def __init__(self, bounds: Optional[Tuple[float, float]] = None):
        self.bounds = bounds

    def _compute_features(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor):
        """
        Evaluate basis at sample points.

        The basis evaluated at the samples, or :math:`b_i(*xi)`, where :math:`b_i` is a
        basis element. xi[k] must be a one-dimensional array or a pynapple Tsd/TsdFrame/TsdTensor.

        Parameters
        ----------
        *xi:
            The input samples over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        Returns
        -------
        :
            A matrix with the transformed features. The basis evaluated at the samples,
            or :math:`b_i(*xi)`, where :math:`b_i` is a basis element. xi[k] must be a one-dimensional array
            or a pynapple Tsd.

        """
        return self._evaluate(*xi)

    def _set_kernel(self) -> "EvalBasisMixin":
        """
        Prepare or compute the convolutional kernel for the basis functions.

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
        self.window_size = window_size
        self.conv_kwargs = {} if conv_kwargs is None else conv_kwargs

    def _compute_features(self, *xi: NDArray | Tsd | TsdFrame | TsdTensor):
        """
        Convolve basis functions with input time series.

        A bank of basis filters (created by calling fit) is convolved with the
        input data. Inputs can be a NDArray, or a pynapple Tsd/TsdFrame/TsdTensor. All the dimensions
        except for the sample-axis are flattened, so that the method always returns a matrix.

        For example, if inputs are of shape (num_samples, 2, 3), the output will be
        ``(num_samples, num_basis_funcs * 2 * 3)``.

        The time-axis can be specified at basis initialization by setting the keyword argument ``axis``.
        For example, if ``axis == 1`` your input should be of shape ``(N1, num_samples N3, ...)``, the output of
        transform will be of shape ``(num_samples, num_basis_funcs * N1 * N3 *...)``.

        Parameters
        ----------
        *xi:
            The input data over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        """
        if self.kernel_ is None:
            raise ValueError(
                "You must call `_set_kernel` before `_compute_features`! "
                "Convolution kernel is not set."
            )
        # before calling the convolve, check that the input matches
        # the expectation. We can check xi[0] only, since convolution
        # is applied at the end of the recursion on the 1D basis, ensuring len(xi) == 1.
        conv = create_convolutional_predictor(self.kernel_, *xi, **self._conv_kwargs)
        # make sure to return a matrix
        return np.reshape(conv, newshape=(conv.shape[0], -1))

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
        self.kernel_ = self._evaluate(np.linspace(0, 1, self.window_size))
        return self

    @property
    def window_size(self):
        """Window size as number of samples.

        Duration of the convolutional kernel in number of samples.
        """
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        """Setter for the window size parameter."""
        if window_size is None:
            raise ValueError(
                "If the basis is in `conv` mode, you must provide a window_size!"
            )

        elif not (isinstance(window_size, int) and window_size > 0):
            raise ValueError(
                f"`window_size` must be a positive integer. {window_size} provided instead!"
            )

        self._window_size = window_size

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
        >>> basis = nmo.basis.RaisedCosineLinearEval(10).to_transformer()
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
        return TransformerBasis(copy.deepcopy(self))
