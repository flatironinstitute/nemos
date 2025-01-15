"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import ArrayLike, NDArray

from ..typing import FeatureMatrix
from ._basis import add_docstring
from ._basis_mixin import AtomicBasisMixin, ConvBasisMixin, EvalBasisMixin
from ._decaying_exponential import OrthExponentialBasis
from ._identity import HistoryBasis, IdentityBasis
from ._raised_cosine_basis import RaisedCosineBasisLinear, RaisedCosineBasisLog
from ._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from ._transformer_basis import TransformerBasis

__all__ = [
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
    "TransformerBasis",
]


def __dir__() -> list[str]:
    return __all__


class BSplineEval(EvalBasisMixin, BSplineBasis):
    """
    B-spline 1-dimensional basis functions.

    Implementation of the one-dimensional BSpline basis [1]_.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    order :
        Order of the splines used in basis functions. Must lie within ``[1, n_basis_funcs]``.
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.
    bounds :
        The bounds for the basis domain. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.


    References
    ----------
    .. [1] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline
        Techniques. Mathematics and Visualization. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-662-04919-8_5

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import BSplineEval
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> bspline_basis = BSplineEval(n_basis_funcs, order=order)
    >>> bspline_basis
    BSplineEval(n_basis_funcs=5, order=3)
    >>> sample_points = linspace(0, 1, 100)
    >>> basis_functions = bspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        order: int = 4,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "BSplineEval",
    ):

        BSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )
        EvalBasisMixin.__init__(self, bounds=bounds)

    @add_docstring("split_by_feature", BSplineBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineEval
        >>> from nemos.glm import GLM
        >>> basis = BSplineEval(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = BSplineEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", BSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import BSplineEval
        >>> bspline_basis = BSplineEval(n_basis_funcs=4, order=3)
        >>> sample_points, basis_values = bspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('B-Spline Basis Functions')
        Text(0.5, 1.0, 'B-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        Text(0.5, 0, 'Domain')
        >>> plt.ylabel('Basis Function Value')
        Text(0, 0.5, 'Basis Function Value')
        >>> l = plt.legend()
        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.BSplineEval(5)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class BSplineConv(ConvBasisMixin, BSplineBasis):
    """
    B-spline 1-dimensional basis functions.

    Implementation of the one-dimensional BSpline basis [1]_.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    window_size :
        The window size for convolution in number of samples.
    order :
        Order of the splines used in basis functions. Must lie within ``[1, n_basis_funcs]``.
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    References
    ----------
    .. [1] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In:
        Bézier and B-Spline Techniques. Mathematics and Visualization. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-662-04919-8_5

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import BSplineConv
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> bspline_basis = BSplineConv(n_basis_funcs, order=order, window_size=10)
    >>> bspline_basis
    BSplineConv(n_basis_funcs=5, window_size=10, order=3)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = bspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        window_size: int,
        order: int = 4,
        label: Optional[str] = "BSplineConv",
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

    @add_docstring("split_by_feature", BSplineBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineConv
        >>> from nemos.glm import GLM
        >>> basis = BSplineConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = BSplineConv(10, window_size=11)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", BSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import BSplineConv
        >>> bspline_basis = BSplineConv(n_basis_funcs=4, order=3, window_size=10)
        >>> sample_points, basis_values = bspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('B-Spline Basis Functions')
        Text(0.5, 1.0, 'B-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        Text(0.5, 0, 'Domain')
        >>> plt.ylabel('Basis Function Value')
        Text(0, 0.5, 'Basis Function Value')
        >>> l = plt.legend()
        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.BSplineConv(5, 10)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class CyclicBSplineEval(EvalBasisMixin, CyclicBSplineBasis):
    """
    B-spline 1-dimensional basis functions for cyclic splines.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    order :
        Order of the splines used in basis functions. Order must lie within [2, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.
    bounds :
        The bounds for the basis domain. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import CyclicBSplineEval
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> cyclic_bspline_basis = CyclicBSplineEval(n_basis_funcs, order=order)
    >>> cyclic_bspline_basis
    CyclicBSplineEval(n_basis_funcs=5, order=3)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = cyclic_bspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        order: int = 4,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "CyclicBSplineEval",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        CyclicBSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )

    @add_docstring("split_by_feature", CyclicBSplineBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CyclicBSplineEval
        >>> from nemos.glm import GLM
        >>> basis = CyclicBSplineEval(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CyclicBSplineEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = CyclicBSplineEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", CyclicBSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 Cyclic B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import CyclicBSplineEval
        >>> cbspline_basis = CyclicBSplineEval(n_basis_funcs=4, order=3)
        >>> sample_points, basis_values = cbspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('Cyclic B-Spline Basis Functions')
        Text(0.5, 1.0, 'Cyclic B-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        Text(0.5, 0, 'Domain')
        >>> plt.ylabel('Basis Function Value')
        Text(0, 0.5, 'Basis Function Value')
        >>> l = plt.legend()
        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.CyclicBSplineEval(5)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class CyclicBSplineConv(ConvBasisMixin, CyclicBSplineBasis):
    """
    B-spline 1-dimensional basis functions for cyclic splines.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    window_size :
        The window size for convolution in number of samples.
    order :
        Order of the splines used in basis functions. Order must lie within [2, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import CyclicBSplineConv
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> cyclic_bspline_basis = CyclicBSplineConv(n_basis_funcs, order=order, window_size=10)
    >>> cyclic_bspline_basis
    CyclicBSplineConv(n_basis_funcs=5, window_size=10, order=3)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = cyclic_bspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        window_size: int,
        order: int = 4,
        label: Optional[str] = "CyclicBSplineConv",
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

    @add_docstring("split_by_feature", CyclicBSplineBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CyclicBSplineConv
        >>> from nemos.glm import GLM
        >>> basis = CyclicBSplineConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CyclicBSplineConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = CyclicBSplineConv(10, window_size=11)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", CyclicBSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 Cyclic B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import CyclicBSplineConv
        >>> cbspline_basis = CyclicBSplineConv(n_basis_funcs=4, order=3, window_size=10)
        >>> sample_points, basis_values = cbspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('Cyclic B-Spline Basis Functions')
        Text(0.5, 1.0, 'Cyclic B-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        Text(0.5, 0, 'Domain')
        >>> plt.ylabel('Basis Function Value')
        Text(0, 0.5, 'Basis Function Value')
        >>> l = plt.legend()
        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.CyclicBSplineConv(5, 10)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class MSplineEval(EvalBasisMixin, MSplineBasis):
    r"""
    M-spline basis functions for modeling and data transformation.

    M-splines [1]_ are a type of spline basis function used for smooth curve fitting
    and data representation. They are positive and integrate to one, making them
    suitable for probabilistic models and density estimation. The order of an
    M-spline defines its smoothness, with higher orders resulting in smoother
    splines.

    This class provides functionality to create M-spline basis functions, allowing
    for flexible and smooth modeling of data. It inherits from the ``SplineBasis``
    abstract class, providing specific implementations for M-splines.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions to generate. More basis functions allow for
        more flexible data modeling but can lead to overfitting.
    order :
        The order of the splines used in basis functions. Must be between [1,
        n_basis_funcs]. Default is 2. Higher order splines have more continuous
        derivatives at each interior knot, resulting in smoother basis functions.
    bounds :
        The bounds for the basis domain. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    References
    ----------
    .. [1] Ramsay, J. O. (1988). Monotone regression splines in action. Statistical science,
        3(4), 425-441.

    Notes
    -----
    ``MSplines`` must integrate to 1 over their domain (the area under the curve is 1). Therefore, if the domain
    (x-axis) of an MSpline basis is expanded by a factor of :math:`\alpha`, the values on the co-domain
    (y-axis) values will shrink by a factor of :math:`1/\alpha`.
    For example, over the standard bounds of (0, 1), the maximum value of the MSpline is 18.
    If we set the bounds to (0, 2), the maximum value will be 9, i.e., 18 / 2.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import MSplineEval
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> mspline_basis = MSplineEval(n_basis_funcs, order=order)
    >>> mspline_basis
    MSplineEval(n_basis_funcs=5, order=3)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = mspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        order: int = 4,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "MSplineEval",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )

    @add_docstring("split_by_feature", MSplineBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineEval
        >>> from nemos.glm import GLM
        >>> basis = MSplineEval(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 6)

        """
        return MSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = MSplineEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", MSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import MSplineEval
        >>> mspline_basis = MSplineEval(n_basis_funcs=4, order=3)
        >>> sample_points, basis_values = mspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('M-Spline Basis Functions')
        Text(0.5, 1.0, 'M-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        Text(0.5, 0, 'Domain')
        >>> plt.ylabel('Basis Function Value')
        Text(0, 0.5, 'Basis Function Value')
        >>> l = plt.legend()
        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.MSplineEval(5)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class MSplineConv(ConvBasisMixin, MSplineBasis):
    r"""
    M-spline basis functions for modeling and data transformation.

    M-splines [1]_ are a type of spline basis function used for smooth curve fitting
    and data representation. They are positive and integrate to one, making them
    suitable for probabilistic models and density estimation. The order of an
    M-spline defines its smoothness, with higher orders resulting in smoother
    splines.

    This class provides functionality to create M-spline basis functions, allowing
    for flexible and smooth modeling of data. It inherits from the ``SplineBasis``
    abstract class, providing specific implementations for M-splines.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions to generate. More basis functions allow for
        more flexible data modeling but can lead to overfitting.
    order :
        The order of the splines used in basis functions. Must be between [1,
        n_basis_funcs]. Default is 2. Higher order splines have more continuous
        derivatives at each interior knot, resulting in smoother basis functions.
    window_size :
        The window size for convolution in number of samples.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    References
    ----------
    .. [1] Ramsay, J. O. (1988). Monotone regression splines in action. Statistical science,
        3(4), 425-441.

    Notes
    -----
    ``MSplines`` must integrate to 1 over their domain (the area under the curve is 1). Therefore, if the domain
    (x-axis) of an MSpline basis is expanded by a factor of :math:`\alpha`, the values on the co-domain
    (y-axis) values will shrink by a factor of :math:`1/\alpha`.
    For example, over the standard bounds of (0, 1), the maximum value of the MSpline is 18.
    If we set the bounds to (0, 2), the maximum value will be 9, i.e., 18 / 2.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import MSplineConv
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> mspline_basis = MSplineConv(n_basis_funcs, order=order, window_size=10)
    >>> mspline_basis
    MSplineConv(n_basis_funcs=5, window_size=10, order=3)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = mspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        window_size: int,
        order: int = 4,
        label: Optional[str] = "MSplineConv",
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

    @add_docstring("split_by_feature", MSplineBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineConv
        >>> from nemos.glm import GLM
        >>> basis = MSplineConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = MSplineConv(10, window_size=11)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", MSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import MSplineConv
        >>> mspline_basis = MSplineConv(n_basis_funcs=4, order=3, window_size=10)
        >>> sample_points, basis_values = mspline_basis.evaluate_on_grid(100)
        >>> for i in range(4):
        ...     p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
        >>> plt.title('M-Spline Basis Functions')
        Text(0.5, 1.0, 'M-Spline Basis Functions')
        >>> plt.xlabel('Domain')
        Text(0.5, 0, 'Domain')
        >>> plt.ylabel('Basis Function Value')
        Text(0, 0.5, 'Basis Function Value')
        >>> l = plt.legend()
        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.MSplineConv(5, 10)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class RaisedCosineLinearEval(EvalBasisMixin, RaisedCosineBasisLinear):
    """
    Represent linearly-spaced raised cosine basis functions.

    This implementation is based on the cosine bumps used by Pillow et al. [1]_
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    width :
        Width of the raised cosine. By default, it's set to 2.0.
    bounds :
        The bounds for the basis domain in ``mode="eval"``. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
        C. E. (2005). Prediction and decoding of retinal ganglion cell responses
        with a probabilistic spiking model. Journal of Neuroscience, 25(47),
        11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import RaisedCosineLinearEval
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = RaisedCosineLinearEval(n_basis_funcs)
    >>> raised_cosine_basis
    RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = raised_cosine_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        width: float = 2.0,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "RaisedCosineLinearEval",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        RaisedCosineBasisLinear.__init__(
            self,
            n_basis_funcs,
            width=width,
            mode="eval",
            label=label,
        )

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLinear)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import RaisedCosineLinearEval
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = RaisedCosineLinearEval(n_basis_funcs)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLinearEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = RaisedCosineLinearEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", RaisedCosineBasisLinear)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLinearEval
        >>> from nemos.glm import GLM
        >>> basis = RaisedCosineLinearEval(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.RaisedCosineLinearEval(5)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class RaisedCosineLinearConv(ConvBasisMixin, RaisedCosineBasisLinear):
    """
    Represent linearly-spaced raised cosine basis functions.

    This implementation is based on the cosine bumps used by Pillow et al. [1]_
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    width :
        Width of the raised cosine. By default, it's set to 2.0.
    window_size :
        The window size for convolution in number of samples.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
        C. E. (2005). Prediction and decoding of retinal ganglion cell responses
        with a probabilistic spiking model. Journal of Neuroscience, 25(47),
        11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import RaisedCosineLinearConv
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = RaisedCosineLinearConv(n_basis_funcs, window_size=10)
    >>> raised_cosine_basis
    RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = raised_cosine_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        window_size: int,
        width: float = 2.0,
        label: Optional[str] = "RaisedCosineLinearConv",
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

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLinear)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import RaisedCosineLinearConv
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = RaisedCosineLinearConv(n_basis_funcs, window_size)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLinearConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = RaisedCosineLinearConv(10, window_size=100)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", RaisedCosineBasisLinear)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLinearConv
        >>> from nemos.glm import GLM
        >>> basis = RaisedCosineLinearConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.RaisedCosineLinearConv(5, 10)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class RaisedCosineLogEval(EvalBasisMixin, RaisedCosineBasisLog):
    """Represent log-spaced raised cosine basis functions.

    Similar to ``RaisedCosineLinearEval`` but the basis functions are log-spaced.
    This implementation is based on the cosine bumps used by Pillow et al. [1]_
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    width :
        Width of the raised cosine.
    time_scaling :
        Non-negative hyper-parameter controlling the logarithmic stretch magnitude, with
        larger values resulting in more stretching. As this approaches 0, the
        transformation becomes linear.
    enforce_decay_to_zero:
        If set to True, the algorithm first constructs a basis with ``n_basis_funcs + ceil(width)`` elements
        and subsequently trims off the extra basis elements. This ensures that the final basis element
        decays to 0.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import RaisedCosineLogEval
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = RaisedCosineLogEval(n_basis_funcs)
    >>> raised_cosine_basis
    RaisedCosineLogEval(n_basis_funcs=5, width=2.0, time_scaling=50.0, enforce_decay_to_zero=True)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = raised_cosine_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        width: float = 2.0,
        time_scaling: float = None,
        enforce_decay_to_zero: bool = True,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "RaisedCosineLogEval",
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

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLog)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import RaisedCosineLogEval
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = RaisedCosineLogEval(n_basis_funcs)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLogEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = RaisedCosineLogEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", RaisedCosineBasisLog)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLogEval
        >>> from nemos.glm import GLM
        >>> basis = RaisedCosineLogEval(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.RaisedCosineLogEval(5)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class RaisedCosineLogConv(ConvBasisMixin, RaisedCosineBasisLog):
    """Represent log-spaced raised cosine basis functions.

    Similar to ``RaisedCosineLinearConv`` but the basis functions are log-spaced.
    This implementation is based on the cosine bumps used by Pillow et al. [1]_
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    width :
        Width of the raised cosine.
    time_scaling :
        Non-negative hyper-parameter controlling the logarithmic stretch magnitude, with
        larger values resulting in more stretching. As this approaches 0, the
        transformation becomes linear.
    enforce_decay_to_zero:
        If set to True, the algorithm first constructs a basis with ``n_basis_funcs + ceil(width)`` elements
        and subsequently trims off the extra basis elements. This ensures that the final basis element
        decays to 0.
    window_size :
        The window size for convolution. Required if mode is 'conv'.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import RaisedCosineLogConv
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = RaisedCosineLogConv(n_basis_funcs, window_size=10)
    >>> raised_cosine_basis
    RaisedCosineLogConv(n_basis_funcs=5, window_size=10, width=2.0, time_scaling=50.0, enforce_decay_to_zero=True)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = raised_cosine_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        window_size: int,
        width: float = 2.0,
        time_scaling: float = None,
        enforce_decay_to_zero: bool = True,
        label: Optional[str] = "RaisedCosineLogConv",
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

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLog)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import RaisedCosineLogConv
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = RaisedCosineLogConv(n_basis_funcs, window_size)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLogConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = RaisedCosineLogConv(10, window_size=100)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", RaisedCosineBasisLog)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLogConv
        >>> from nemos.glm import GLM
        >>> basis = RaisedCosineLogConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.RaisedCosineLogConv(5, 10)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class OrthExponentialEval(EvalBasisMixin, OrthExponentialBasis):
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
    >>> from nemos.basis import OrthExponentialEval
    >>> X = np.random.normal(size=(1000, 1))
    >>> n_basis_funcs = 5
    >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # sample decay rates
    >>> window_size = 10
    >>> ortho_basis = OrthExponentialEval(n_basis_funcs, decay_rates)
    >>> ortho_basis
    OrthExponentialEval(n_basis_funcs=5)
    >>> sample_points = linspace(0, 1, 100)
    >>> # evaluate the basis
    >>> features = ortho_basis.compute_features(sample_points)

    """

    def __init__(
        self,
        n_basis_funcs: int,
        decay_rates: NDArray,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "OrthExponentialEval",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        OrthExponentialBasis.__init__(
            self,
            n_basis_funcs,
            decay_rates=decay_rates,
            mode="eval",
            label=label,
        )

    @add_docstring("evaluate_on_grid", OrthExponentialBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import OrthExponentialEval
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = OrthExponentialEval(n_basis_funcs, decay_rates=decay_rates)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples=n_samples)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import OrthExponentialEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = OrthExponentialEval(10, decay_rates=np.arange(1, 11))
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", OrthExponentialBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import OrthExponentialEval
        >>> from nemos.glm import GLM
        >>> # Define an additive basis
        >>> basis = OrthExponentialEval(n_basis_funcs=5, decay_rates=np.arange(1, 6), label="feature")
        >>> # Generate a sample input array and compute features
        >>> inp = np.random.randn(20)
        >>> X = basis.compute_features(inp)
        >>> # Split the feature matrix along axis 1
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        feature: shape (20, 5)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.OrthExponentialEval(5, decay_rates=np.arange(1, 6))
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class OrthExponentialConv(ConvBasisMixin, OrthExponentialBasis):
    """Set of 1D basis decaying exponential functions numerically orthogonalized.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size :
        The window size for convolution in number of samples.
    decay_rates :
        Decay rates of the exponentials, shape ``(n_basis_funcs,)``.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import OrthExponentialConv
    >>> X = np.random.normal(size=(1000, 1))
    >>> n_basis_funcs = 5
    >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # sample decay rates
    >>> window_size = 10
    >>> ortho_basis = OrthExponentialConv(n_basis_funcs, window_size, decay_rates)
    >>> ortho_basis
    OrthExponentialConv(n_basis_funcs=5, window_size=10)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = ortho_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        window_size: int,
        decay_rates: NDArray,
        label: Optional[str] = "OrthExponentialConv",
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
        # re-check window size because n_basis_funcs is not set yet when the
        # property setter runs the first check.
        self._check_window_size(self.window_size)

    @add_docstring("evaluate_on_grid", OrthExponentialBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import OrthExponentialConv
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = OrthExponentialConv(n_basis_funcs, window_size, decay_rates=decay_rates)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import OrthExponentialConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = OrthExponentialConv(10, window_size=100, decay_rates=np.arange(1, 11))
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", OrthExponentialBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import OrthExponentialConv
        >>> from nemos.glm import GLM
        >>> basis = OrthExponentialConv(
        ...     n_basis_funcs=6,
        ...     decay_rates=np.arange(1, 7),
        ...     window_size=10,
        ...     label="two_inputs"
        ... )
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.OrthExponentialConv(5, window_size=10, decay_rates=np.arange(1, 6))
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)

    def _check_window_size(self, window_size: int):
        """OrthExponentialBasis specific window size check."""
        super()._check_window_size(window_size)
        # if n_basis_funcs is not yet initialized, skip check
        n_basis = getattr(self, "n_basis_funcs", None)
        if n_basis and window_size < n_basis:
            raise ValueError(
                "OrthExponentialConv basis requires at least a window_size larger then the number "
                f"of basis functions. window_size is {window_size}, n_basis_funcs while"
                f"is {self.n_basis_funcs}."
            )


class IdentityEval(EvalBasisMixin, IdentityBasis):
    """
    Identity basis function.

    This function includes the samples themselves as predictor reshaped as
    a 2D array. It is intended to be used for including a task variable directly as
    a predictor.

    Parameters
    ----------
    bounds :
        The bounds for the basis domain. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    """

    def __init__(
        self,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "IdentityEval",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        IdentityBasis.__init__(
            self,
            n_basis_funcs=1,
            label=label,
        )

    @add_docstring("evaluate_on_grid", OrthExponentialBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import IdentityEval
        >>> basis = IdentityEval()
        >>> sample_points, basis_values = basis.evaluate_on_grid(100)

        """
        return super().evaluate_on_grid(n_samples=n_samples)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import IdentityEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = IdentityEval()
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 1)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", IdentityBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import IdentityEval
        >>> from nemos.glm import GLM
        >>> # Define an additive basis
        >>> basis = IdentityEval(label="feature")
        >>> # Generate a sample input array and compute features
        >>> inp = np.random.randn(20)
        >>> X = basis.compute_features(inp)
        >>> # Split the feature matrix along axis 1
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        feature: shape (20, 1)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.IdentityEval()
        >>> # Configure with an integer input:
        >>> _ = basis.set_input_shape(3)
        >>> basis.n_output_features
        3
        >>> # Configure with a tuple:
        >>> _ = basis.set_input_shape((4, 5))
        >>> basis.n_output_features
        20
        >>> # Configure with an array:
        >>> x = np.ones((10, 4, 5))
        >>> _ = basis.set_input_shape(x)
        >>> basis.n_output_features
        20

        """
        return AtomicBasisMixin.set_input_shape(self, xi)


class HistoryConv(ConvBasisMixin, HistoryBasis):
    """Basis for history effects.

    This function includes the history of the samples as predictor reshaped as
    a 2D array. It is intended to be used for including a raw history as predictor.

    Parameters
    ----------
    window_size:
        History window as the number of samples.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    conv_kwargs:
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor`;
        These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.
    """

    def __init__(
        self,
        window_size: int,
        label: Optional[str] = "HistoryConv",
        conv_kwargs: Optional[dict] = None,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size, conv_kwargs=conv_kwargs)
        HistoryBasis.__init__(
            self,
            n_basis_funcs=window_size,
            label=label,
        )

    @add_docstring("evaluate_on_grid", HistoryBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import HistoryConv
        >>> window_size=100
        >>> basis = HistoryConv(window_size=window_size)
        >>> sample_points, basis_values = basis.evaluate_on_grid(window_size)

        """
        return super().evaluate_on_grid(n_samples)

    @add_docstring("_compute_features", ConvBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import HistoryConv

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = HistoryConv(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return super().compute_features(xi)

    @add_docstring("split_by_feature", IdentityBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import HistoryConv
        >>> from nemos.glm import GLM
        >>> # Define an additive basis
        >>> basis = HistoryConv(5, label="feature")
        >>> # Generate a sample input array and compute features
        >>> inp = np.random.randn(20)
        >>> X = basis.compute_features(inp)
        >>> # Split the feature matrix along axis 1
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        feature: shape (20, 5)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.HistoryConv(5)
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

        """
        return AtomicBasisMixin.set_input_shape(self, xi)

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int) -> None:
        self._check_window_size(window_size)
        self._window_size = window_size
        self._n_basis_funcs = window_size
