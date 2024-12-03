"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import ArrayLike, NDArray

from ..typing import FeatureMatrix
from ._basis_mixin import BasisTransformerMixin, ConvBasisMixin, EvalBasisMixin
from ._decaying_exponential import OrthExponentialBasis, add_orth_exp_decay_docstring
from ._raised_cosine_basis import (
    RaisedCosineBasisLinear,
    RaisedCosineBasisLog,
    add_raised_cosine_linear_docstring,
    add_raised_cosine_log_docstring,
)
from ._spline_basis import (
    BSplineBasis,
    CyclicBSplineBasis,
    MSplineBasis,
    add_docstrings_bspline,
    add_docstrings_cyclic_bspline,
    add_docstrings_mspline,
)
from ._transformer_basis import TransformerBasis

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
    "TransformerBasis",
]


def __dir__() -> list[str]:
    return __all__


class EvalBSpline(EvalBasisMixin, BSplineBasis):
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

    Attributes
    ----------
    order :
        Spline order.


    References
    ----------
    .. [1] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline
        Techniques. Mathematics and Visualization. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-662-04919-8_5

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import EvalBSpline
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> bspline_basis = EvalBSpline(n_basis_funcs, order=order)
    >>> sample_points = linspace(0, 1, 100)
    >>> basis_functions = bspline_basis.compute_features(sample_points)
    """

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

    @add_docstrings_bspline("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalBSpline
        >>> from nemos.glm import GLM
        >>> basis = EvalBSpline(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 1, 6)

        """
        return BSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstrings_bspline("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalBSpline

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = EvalBSpline(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return BSplineBasis.compute_features(self, xi)

    @add_docstrings_bspline("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import EvalBSpline
        >>> bspline_basis = EvalBSpline(n_basis_funcs=4, order=3)
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
        return BSplineBasis.evaluate_on_grid(self, n_samples)


class ConvBSpline(ConvBasisMixin, BSplineBasis):
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

    Attributes
    ----------
    order :
        Spline order.


    References
    ----------
    .. [1] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In:
        Bézier and B-Spline Techniques. Mathematics and Visualization. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-662-04919-8_5

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import ConvBSpline
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> bspline_basis = ConvBSpline(n_basis_funcs, order=order, window_size=10)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = bspline_basis.compute_features(sample_points)
    """

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

    @add_docstrings_bspline("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvBSpline
        >>> from nemos.glm import GLM
        >>> basis = ConvBSpline(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return BSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstrings_bspline("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvBSpline

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvBSpline(10, window_size=11)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return BSplineBasis.compute_features(self, xi)

    @add_docstrings_bspline("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import ConvBSpline
        >>> bspline_basis = ConvBSpline(n_basis_funcs=4, order=3, window_size=10)
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
        return BSplineBasis.evaluate_on_grid(self, n_samples)


class EvalCyclicBSpline(EvalBasisMixin, CyclicBSplineBasis):
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

    Attributes
    ----------
    n_basis_funcs :
        Number of basis functions, int.
    order :
        Order of the splines used in basis functions, int.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import EvalCyclicBSpline
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> cyclic_bspline_basis = EvalCyclicBSpline(n_basis_funcs, order=order)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = cyclic_bspline_basis.compute_features(sample_points)
    """

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

    @add_docstrings_cyclic_bspline("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalCyclicBSpline
        >>> from nemos.glm import GLM
        >>> basis = EvalCyclicBSpline(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 1, 6)

        """
        return CyclicBSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstrings_cyclic_bspline("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalCyclicBSpline

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = EvalCyclicBSpline(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return CyclicBSplineBasis.compute_features(self, xi)

    @add_docstrings_cyclic_bspline("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 Cyclic B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import EvalCyclicBSpline
        >>> cbspline_basis = EvalCyclicBSpline(n_basis_funcs=4, order=3)
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
        return CyclicBSplineBasis.evaluate_on_grid(self, n_samples)


class ConvCyclicBSpline(ConvBasisMixin, CyclicBSplineBasis):
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

    Attributes
    ----------
    n_basis_funcs :
        Number of basis functions, int.
    order :
        Order of the splines used in basis functions, int.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import ConvCyclicBSpline
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> cyclic_bspline_basis = ConvCyclicBSpline(n_basis_funcs, order=order, window_size=10)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = cyclic_bspline_basis.compute_features(sample_points)
    """

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

    @add_docstrings_cyclic_bspline("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvCyclicBSpline
        >>> from nemos.glm import GLM
        >>> basis = ConvCyclicBSpline(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return CyclicBSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstrings_cyclic_bspline("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvCyclicBSpline

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvCyclicBSpline(10, window_size=11)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return CyclicBSplineBasis.compute_features(self, xi)

    @add_docstrings_cyclic_bspline("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 Cyclic B-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import ConvCyclicBSpline
        >>> cbspline_basis = ConvCyclicBSpline(n_basis_funcs=4, order=3, window_size=10)
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
        return CyclicBSplineBasis.evaluate_on_grid(self, n_samples)


class EvalMSpline(EvalBasisMixin, MSplineBasis):
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
    >>> from nemos.basis import EvalMSpline
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> mspline_basis = EvalMSpline(n_basis_funcs, order=order)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = mspline_basis.compute_features(sample_points)
    """

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

    @add_docstrings_mspline("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalMSpline
        >>> from nemos.glm import GLM
        >>> basis = EvalMSpline(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 1, 6)

        """
        return MSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstrings_mspline("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalMSpline

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = EvalMSpline(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return MSplineBasis.compute_features(self, xi)

    @add_docstrings_mspline("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import EvalMSpline
        >>> mspline_basis = EvalMSpline(n_basis_funcs=4, order=3)
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
        return MSplineBasis.evaluate_on_grid(self, n_samples)


class ConvMSpline(ConvBasisMixin, MSplineBasis):
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
    >>> from nemos.basis import ConvMSpline
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> mspline_basis = ConvMSpline(n_basis_funcs, order=order, window_size=10)
    >>> sample_points = linspace(0, 1, 100)
    >>> features = mspline_basis.compute_features(sample_points)
    """

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

    @add_docstrings_mspline("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvMSpline
        >>> from nemos.glm import GLM
        >>> basis = ConvMSpline(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return MSplineBasis.split_by_feature(self, x, axis=axis)

    @add_docstrings_mspline("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvMSpline

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvMSpline(10, window_size=11)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return MSplineBasis.compute_features(self, xi)

    @add_docstrings_mspline("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import ConvMSpline
        >>> mspline_basis = ConvMSpline(n_basis_funcs=4, order=3, window_size=10)
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
        return MSplineBasis.evaluate_on_grid(self, n_samples)


class EvalRaisedCosineLinear(
    EvalBasisMixin, RaisedCosineBasisLinear, BasisTransformerMixin
):
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
    >>> from nemos.basis import EvalRaisedCosineLinear
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = EvalRaisedCosineLinear(n_basis_funcs)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = raised_cosine_basis.compute_features(sample_points)
    """

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

    @add_raised_cosine_linear_docstring("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import EvalRaisedCosineLinear
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = EvalRaisedCosineLinear(n_basis_funcs)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return RaisedCosineBasisLinear.evaluate_on_grid(self, n_samples)

    @add_raised_cosine_linear_docstring("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalRaisedCosineLinear

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = EvalRaisedCosineLinear(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return RaisedCosineBasisLinear.compute_features(self, xi)

    @add_raised_cosine_linear_docstring("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import EvalRaisedCosineLinear
        >>> from nemos.glm import GLM
        >>> basis = EvalRaisedCosineLinear(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 1, 6)

        """
        return RaisedCosineBasisLinear.split_by_feature(self, x, axis=axis)


class ConvRaisedCosineLinear(
    ConvBasisMixin, RaisedCosineBasisLinear, BasisTransformerMixin
):
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

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
        C. E. (2005). Prediction and decoding of retinal ganglion cell responses
        with a probabilistic spiking model. Journal of Neuroscience, 25(47),
        11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import ConvRaisedCosineLinear
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = ConvRaisedCosineLinear(n_basis_funcs, window_size=10)
    >>> sample_points = np.random.randn(100)
    >>> # convolve the basis
    >>> features = raised_cosine_basis.compute_features(sample_points)
    """

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

    @add_raised_cosine_linear_docstring("evaluate_on_grid")
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import ConvRaisedCosineLinear
        >>> n_basis_funcs = 5
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
        >>> window_size=10
        >>> ortho_basis = ConvRaisedCosineLinear(n_basis_funcs, window_size)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)

        """
        return RaisedCosineBasisLinear.evaluate_on_grid(self, n_samples)

    @add_raised_cosine_linear_docstring("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvRaisedCosineLinear

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = ConvRaisedCosineLinear(10, window_size=100)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 10)

        """
        return RaisedCosineBasisLinear.compute_features(self, xi)

    @add_raised_cosine_linear_docstring("split_by_feature")
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import ConvRaisedCosineLinear
        >>> from nemos.glm import GLM
        >>> basis = ConvRaisedCosineLinear(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> X_multi = basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        two_inputs, shape (20, 2, 6)

        """
        return RaisedCosineBasisLinear.split_by_feature(self, x, axis=axis)


class EvalRaisedCosineLog(EvalBasisMixin, RaisedCosineBasisLog):
    """Represent log-spaced raised cosine basis functions.

    Similar to ``EvalRaisedCosineLinear`` but the basis functions are log-spaced.
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

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import EvalRaisedCosineLog
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = EvalRaisedCosineLog(n_basis_funcs)
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
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
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
        return RaisedCosineBasisLog.compute_features(self, xi)

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
        >>> basis = EvalRaisedCosineLog(n_basis_funcs=6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 1, 6)

        """
        return RaisedCosineBasisLog.split_by_feature(self, x, axis=axis)


class ConvRaisedCosineLog(ConvBasisMixin, RaisedCosineBasisLog):
    """Represent log-spaced raised cosine basis functions.

    Similar to ``ConvRaisedCosineLinear`` but the basis functions are log-spaced.
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

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import ConvRaisedCosineLog
    >>> n_basis_funcs = 5
    >>> raised_cosine_basis = ConvRaisedCosineLog(n_basis_funcs, window_size=10)
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
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
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
        return RaisedCosineBasisLog.compute_features(self, xi)

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
    >>> from nemos.basis import EvalOrthExponential
    >>> X = np.random.normal(size=(1000, 1))
    >>> n_basis_funcs = 5
    >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # sample decay rates
    >>> window_size = 10
    >>> ortho_basis = EvalOrthExponential(n_basis_funcs, decay_rates)
    >>> sample_points = linspace(0, 1, 100)
    >>> # evaluate the basis
    >>> features = ortho_basis.compute_features(sample_points)

    """

    def __init__(
        self,
        n_basis_funcs: int,
        decay_rates: NDArray,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "EvalOrthExponential",
    ):
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
        return OrthExponentialBasis.evaluate_on_grid(self, n_samples=n_samples)

    @add_orth_exp_decay_docstring("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
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
        return OrthExponentialBasis.compute_features(self, xi)

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
        >>> basis = EvalOrthExponential(n_basis_funcs=5, decay_rates=np.arange(1, 6), label="feature")
        >>> # Generate a sample input array and compute features
        >>> x = np.random.randn(20)
        >>> X = basis.compute_features(x)
        >>> # Split the feature matrix along axis 1
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        feature: shape (20, 1, 5)

        """
        return OrthExponentialBasis.split_by_feature(self, x, axis=axis)


class ConvOrthExponential(ConvBasisMixin, OrthExponentialBasis):
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
    >>> features = ortho_basis.compute_features(sample_points)
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
        return OrthExponentialBasis.evaluate_on_grid(self, n_samples)

    @add_orth_exp_decay_docstring("compute_features")
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
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
        return OrthExponentialBasis.compute_features(self, xi)

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
        >>> basis = ConvOrthExponential(
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
        return OrthExponentialBasis.split_by_feature(self, x, axis=axis)
