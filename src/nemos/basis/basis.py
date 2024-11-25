"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray, ArrayLike


from ._basis_mixin import EvalBasisMixin, ConvBasisMixin

from ._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from ._raised_cosine_basis import RaisedCosineBasisLinear, RaisedCosineBasisLog, add_raised_cosine_linear_docstring, add_raised_cosine_log_docstring
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

    @add_raised_cosine_log_docstring("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import EvalRaisedCosineLog
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = EvalRaisedCosineLog(n_basis_funcs)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return RaisedCosineBasisLog.evaluate_on_grid(self, n_samples)

    @add_raised_cosine_log_docstring("compute_features")
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalRaisedCosineLog

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = EvalRaisedCosineLog(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return RaisedCosineBasisLog.compute_features(self, *xi)

    @add_raised_cosine_log_docstring("split_by_feature")
    def split_by_feature(
            self,
            x: NDArray,
            axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalRaisedCosineLog
        >>> from nemos.glm import GLM
        >>> basis = EvalRaisedCosineLog(n_basis_funcs=6, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return RaisedCosineBasisLog.split_by_feature(self, x, axis=axis)


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

    @add_raised_cosine_log_docstring("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import ConvRaisedCosineLog
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = ConvRaisedCosineLog(n_basis_funcs, window_size)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return RaisedCosineBasisLog.evaluate_on_grid(self, n_samples)

    @add_raised_cosine_log_docstring("compute_features")
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvRaisedCosineLog

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvRaisedCosineLog(10, window_size=100)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return RaisedCosineBasisLog.compute_features(self, *xi)

    @add_raised_cosine_log_docstring("split_by_feature")
    def split_by_feature(
            self,
            x: NDArray,
            axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvRaisedCosineLog
        >>> from nemos.glm import GLM
        >>> basis = ConvRaisedCosineLog(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return RaisedCosineBasisLog.split_by_feature(self, x, axis=axis)


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

    @add_orth_exp_decay_docstring("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
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

    @add_orth_exp_decay_docstring("compute_features")
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalOrthExponential

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = EvalOrthExponential(10, decay_rates=np.arange(1, 11))
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(*xi)

    @add_orth_exp_decay_docstring("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
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


