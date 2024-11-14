"""Mixin classes for basis."""

from numpy.typing import ArrayLike
from ..convolve import create_convolutional_predictor
import numpy as np
from typing import Union, Tuple


class EvalBasisMixin:

    def __init__(self, *args, **kwargs):
        self._bounds = kwargs.pop("bounds", None)

    def _compute_features(self, *xi: ArrayLike):
        """
        Apply the basis transformation to the input data.

        The basis evaluated at the samples, or $b_i(*xi)$, where $b_i$ is a
        basis element. xi[k] must be a one-dimensional array or a pynapple Tsd.

        Parameters
        ----------
        *xi:
            The input samples over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        Returns
        -------
        :
            A matrix with the transformed features. Faturehe basis evaluated at the samples,
            or $b_i(*xi)$, where $b_i$ is a basis element. xi[k] must be a one-dimensional array
            or a pynapple Tsd.

        """
        return self.__call__(*xi)

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
        return self._bounds

    @bounds.setter
    def bounds(self, values: Union[None, Tuple[float, float]]):
        """Setter for bounds."""
        if values is not None and self.mode == "conv":
            raise ValueError("`bounds` should only be set when `mode=='eval'`.")

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

    def __init__(self, *args, **kwargs):
        self._window_size = kwargs.pop("window_size")

    def _compute_features(self, *xi: ArrayLike):
        """
        Apply the basis transformation to the input data.

        A bank of basis filters (created by calling fit) is convolved with the
        samples. Samples can be a NDArray, or a pynapple Tsd/TsdFrame/TsdTensor. All the dimensions
        except for the sample-axis are flattened, so that the method always returns a matrix.
        For example, if samples are of shape (num_samples, 2, 3), the output will be
        (num_samples, num_basis_funcs * 2 * 3).
        The time-axis can be specified at basis initialization by setting the keyword argument `axis`.
        For example, if `axis == 1` your samples should be (N1, num_samples N3, ...), the output of
        transform will be (num_samples, num_basis_funcs * N1 * N3 *...).

        Parameters
        ----------
        *xi:
            The input samples over which to apply the basis transformation. The samples can be passed
            as multiple arguments, each representing a different dimension for multivariate inputs.

        """
        # before calling the convolve, check that the input matches
        # the expectation. We can check xi[0] only, since convolution
        # is applied at the end of the recursion on the 1D basis, ensuring len(xi) == 1.
        conv = create_convolutional_predictor(
            self.kernel_, *xi, **self._conv_kwargs
        )
        # make sure to return a matrix
        return np.reshape(conv, newshape=(conv.shape[0], -1))

    def _set_kernel(self) -> "ConvBasisMixin":
        """
        Prepare or compute the convolutional kernel for the basis functions.

        This method is called to prepare the basis functions for convolution operations
        in subclasses where the 'conv' mode is used. It typically involves computing a
        kernel based on the basis functions that will be used for convolution with the
        input data. The specifics of kernel computation depend on the subclass implementation
        and the nature of the basis functions.

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
        self.kernel_ = self.__call__(np.linspace(0, 1, self.window_size))
        return self

    @property
    def window_size(self):
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