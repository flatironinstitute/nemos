"""Mixin classes for basis."""

from numpy.typing import ArrayLike
from ..convolve import create_convolutional_predictor
import numpy as np
from typing import Union, Tuple, Optional
import inspect

class EvalBasisMixin:

    def __init__(self, bounds: Optional[Tuple[float, float]] = None):
        self.bounds = bounds

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

    def _check_convolution_kwargs(self):
        """Check convolution kwargs settings.

        Raises
        ------
        ValueError:
            If `self._conv_kwargs` are not None.
        """
        # this should not be hit since **kwargs are not allowed at EvalBasis init.
        # still keep it for compliance with Abstract class Basis.
        if self._conv_kwargs:
            raise ValueError(
                f"kwargs should only be set when mode=='conv', but '{self._mode}' provided instead!"
            )


class ConvBasisMixin:

    def __init__(self, window_size: int):
        self.window_size = window_size

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

    def _check_convolution_kwargs(self):
        """Check convolution kwargs settings.

        Raises
        ------
        ValueError:
            - If `axis` is provided as an argument, and it is different from 0
            (samples must always be in the first axis).
            - If `self._conv_kwargs` include parameters not recognized or that do not have
            default values in `create_convolutional_predictor`.
        """
        if "axis" in self._conv_kwargs:
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
        if not set(self._conv_kwargs.keys()).issubset(convolve_configs):
            # do not encourage to set axis.
            convolve_configs = convolve_configs.difference({"axis"})
            # remove the parameter in case axis=0 was passed, since it is allowed.
            invalid = (
                set(self._conv_kwargs.keys())
                .difference(convolve_configs)
                .difference({"axis"})
            )
            raise ValueError(
                f"Unrecognized keyword arguments: {invalid}. "
                f"Allowed convolution keyword arguments are: {convolve_configs}."
            )