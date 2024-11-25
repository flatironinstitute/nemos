"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray, ArrayLike


from ._basis_mixin import EvalBasisMixin, ConvBasisMixin

from ._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from ._raised_cosine_basis import RaisedCosineBasisLinear, RaisedCosineBasisLog
from ._decaying_exponential import OrthExponentialBasis, add_orth_exp_decay_docstring
from ..typing import FeatureMatrix

__all__ = [
    "EvalMSpline",
    "ConvMSpline",
    "EvalBSpline",
    "ConvBSpline",
    "EvalCyclicBSpline",
    "ConvCyclicBSpline",
    "EvalRaisedCosineLinear",
    "ConvRaisedCosineLinear",
    "EvalRaisedCosineLog",
    "ConvRaisedCosineLog",
    "EvalOrthExponential",
    "ConvOrthExponential",
]



def __dir__() -> list[str]:
    return __all__


class EvalBSpline(EvalBasisMixin, BSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalBSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        BSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )



class ConvBSpline(ConvBasisMixin, BSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvBSpline",
            conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        BSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )


class EvalCyclicBSpline(EvalBasisMixin, CyclicBSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalCyclicBSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        CyclicBSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )


class ConvCyclicBSpline(ConvBasisMixin, CyclicBSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvCyclicBSpline",
            conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        CyclicBSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )


class EvalMSpline(EvalBasisMixin, MSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalMSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )


class ConvMSpline(ConvBasisMixin, MSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvMSpline",
            conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )


class EvalRaisedCosineLinear(EvalBasisMixin, RaisedCosineBasisLinear):
    def __init__(
            self,
            n_basis_funcs: int,
            width: float = 2.0,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalRaisedCosineLinear",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        RaisedCosineBasisLinear.__init__(
            self,
            n_basis_funcs,
            width=width,
            mode="eval",
            label=label,
        )


class ConvRaisedCosineLinear(ConvBasisMixin, RaisedCosineBasisLinear):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            width: float = 2.0,
            label: Optional[str] = "ConvRaisedCosineLinear",
            conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        RaisedCosineBasisLinear.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            width=width,
            label=label,
        )

class EvalRaisedCosineLog(EvalBasisMixin, RaisedCosineBasisLog):
    def __init__(
            self,
            n_basis_funcs: int,
            width: float = 2.0,
            time_scaling: float = None,
            enforce_decay_to_zero: bool = True,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalRaisedCosineLog",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        RaisedCosineBasisLog.__init__(
            self,
            n_basis_funcs,
            width=width,
            time_scaling=time_scaling,
            enforce_decay_to_zero=enforce_decay_to_zero,
            mode="eval",
            label=label,
        )


class ConvRaisedCosineLog(ConvBasisMixin, RaisedCosineBasisLog):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            width: float = 2.0,
            time_scaling: float = None,
            enforce_decay_to_zero: bool = True,
            label: Optional[str] = "ConvRaisedCosineLog",
            conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        RaisedCosineBasisLog.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            width=width,
            time_scaling=time_scaling,
            enforce_decay_to_zero=enforce_decay_to_zero,
            label=label,
        )


class EvalOrthExponential(EvalBasisMixin, OrthExponentialBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            decay_rates: NDArray,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalOrthExponential",
    ):
        """Set of 1D basis decaying exponential functions numerically orthogonalized.

        Parameters
        ----------
        n_basis_funcs
            Number of basis functions.
        decay_rates :
            Decay rates of the exponentials, shape ``(n_basis_funcs,)``.
        bounds :
            The bounds for the basis domain. The default ``bounds[0]`` and ``bounds[1]`` are the
            minimum and the maximum of the samples provided when evaluating the basis.
            If a sample is outside the bounds, the basis will return NaN.
        label :
            The label of the basis, intended to be descriptive of the task variable being processed.
            For example: velocity, position, spike_counts.

        Examples
        --------
        >>> import numpy as np
        >>> from numpy import linspace
        >>> from nemos.basis import ConvOrthExponential
        >>> X = np.random.normal(size=(1000, 1))
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # sample decay rates
        >>> window_size = 10
        >>> ortho_basis = EvalOrthExponential(n_basis_funcs, decay_rates)
        >>> sample_points = linspace(0, 1, 100)
        >>> # evaluate the basis
        >>> basis_functions = ortho_basis.compute_features(sample_points)

        """
        EvalBasisMixin.__init__(self, bounds=bounds)
        OrthExponentialBasis.__init__(
            self,
            n_basis_funcs,
            decay_rates=decay_rates,
            mode="eval",
            label=label,
        )

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        n_samples:
            The number of samples.

        Returns
        -------
        X :
            Array of shape ``(n_samples,)`` containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Evaluated exponentially decaying basis functions, numerically
            orthogonalized, shape ``(n_samples, n_basis_funcs)``

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import EvalOrthExponential
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = EvalOrthExponential(n_basis_funcs, decay_rates=decay_rates)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples=n_samples)

    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Compute the basis functions and transform input data into model features.

        This method is designed to be a high-level interface for transforming input
        data using the basis functions defined by the subclass. It evaluates the basis functions at the sample
        points.

        Parameters
        ----------
        *xi :
            Input data arrays to be transformed.

        Returns
        -------
        :
            Transformed features, consisting of the basis functions evaluated at the input samples.

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvOrthExponential

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvOrthExponential(10, window_size=100, decay_rates=np.arange(1, 11))
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(*xi)

    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Decompose an array along a specified axis into sub-arrays based on the number of expected inputs.

        This function takes an array (e.g., a design matrix or model coefficients) and splits it along
        a designated axis.

        **How it works:**

        - If the basis expects an input shape ``(n_samples, n_inputs)``, then the feature axis length will
        be ``total_n_features = n_inputs * n_basis_funcs``. This axis is reshaped into dimensions
        ``(n_inputs, n_basis_funcs)``.

        - If the basis expects an input of shape ``(n_samples,)``, then the feature axis length will
        be ``total_n_features = n_basis_funcs``. This axis is reshaped into ``(1, n_basis_funcs)``.

        For example, if the input array ``x`` has shape ``(1, 2, total_n_features, 4, 5)``,
        then after applying this method, it will be reshaped into ``(1, 2, n_inputs, n_basis_funcs, 4, 5)``.

        The specified axis (``axis``) determines where the split occurs, and all other dimensions
        remain unchanged. See the example section below for the most common use cases.

        Parameters
        ----------
        x :
          The input array to be split, representing concatenated features, coefficients,
          or other data. The shape of ``x`` along the specified axis must match the total
          number of features generated by the basis, i.e., ``self.n_output_features``.

          **Examples:**

          - For a design matrix: ``(n_samples, total_n_features)``

          - For model coefficients: ``(total_n_features,)`` or ``(total_n_features, n_neurons)``.

        axis : int, optional
          The axis along which to split the features. Defaults to 1.
          Use ``axis=1`` for design matrices (features along columns) and ``axis=0`` for
          coefficient arrays (features along rows). All other dimensions are preserved.

        Raises
        ------
        ValueError
          If the shape of ``x`` along the specified axis does not match ``self.n_output_features``.

        Returns
        -------
        dict
          A dictionary where:
          - **Key**: Label of the basis.
          - **Value**: the array reshaped to: ``(..., n_inputs, n_basis_funcs, ...)``

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalOrthExponential
        >>> from nemos.glm import GLM
        >>> # Define an additive basis
        >>> basis = EvalOrthExponential(n_basis_funcs=5, label="feature")
        >>> # Generate a sample input array and compute features
        >>> x = np.random.randn(20)
        >>> X = basis.compute_features(x)
        >>> # Split the feature matrix along axis 1
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        feature: shape (20, 1, 5)

        """
        return super().split_by_feature(x, axis=axis)


class ConvOrthExponential(ConvBasisMixin, OrthExponentialBasis):
    """
    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import ConvOrthExponential
    >>> X = np.random.normal(size=(1000, 1))
    >>> n_basis_funcs = 5
    >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # sample decay rates
    >>> window_size = 10
    >>> ortho_basis = ConvOrthExponential(n_basis_funcs, window_size, decay_rates)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> basis_functions = ortho_basis.compute_features(sample_points)

    """
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            decay_rates: NDArray,
            label: Optional[str] = "ConvOrthExponential",
            conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        OrthExponentialBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            decay_rates=decay_rates,
            label=label,
        )

    @add_orth_exp_decay_docstring("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import ConvOrthExponential
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = ConvOrthExponential(n_basis_funcs, window_size, decay_rates=decay_rates)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples)

    @add_orth_exp_decay_docstring("compute_features")
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvOrthExponential

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvOrthExponential(10, window_size=100, decay_rates=np.arange(1, 11))
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return OrthExponentialBasis.compute_features(self, *xi)

    @add_orth_exp_decay_docstring("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvOrthExponential
        >>> from nemos.glm import GLM
        >>> basis = ConvOrthExponential(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return OrthExponentialBasis.split_by_feature(self, x, axis=axis)


