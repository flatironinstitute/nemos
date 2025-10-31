"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from numbers import Number
from typing import List, Literal, Optional, Sequence, Tuple

import jax
from numpy.typing import ArrayLike, NDArray

from ..type_casting import is_numpy_array_like
from ..typing import FeatureMatrix
from ._basis_mixin import AtomicBasisMixin, BasisMixin, ConvBasisMixin, EvalBasisMixin
from ._composition_utils import add_docstring
from ._decaying_exponential import OrthExponentialBasis
from ._fourier_basis import FourierBasis
from ._identity import HistoryBasis, IdentityBasis
from ._raised_cosine_basis import RaisedCosineBasisLinear, RaisedCosineBasisLog
from ._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis


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
    .. [2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline
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
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", BSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: B-Spline

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
            >>> plt.show()
        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", BSplineBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineEval
        >>> bspline_basis = BSplineEval(n_basis_funcs=4, order=3)
        >>> out = bspline_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
    .. [2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In:
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", BSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: B-Spline

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
            >>> plt.show()
        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", BSplineBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineConv
        >>> bspline_basis = BSplineConv(n_basis_funcs=4, window_size=20, order=3)
        >>> out = bspline_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", CyclicBSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 Cyclic B-spline basis functions of order 3:

        .. plot::
            :include-source: True
            :caption: Cyclic B-Spline

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
            >>> plt.show()
        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", CyclicBSplineBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CyclicBSplineEval
        >>> cbspline_basis = CyclicBSplineEval(n_basis_funcs=4, order=3)
        >>> out = cbspline_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", CyclicBSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 Cyclic B-spline basis functions of order 3:

        .. plot::
            :include-source: True
            :caption: Cyclic B-Spline

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
            >>> plt.show()
        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", CyclicBSplineBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import CyclicBSplineConv
        >>> cbspline_basis = CyclicBSplineConv(n_basis_funcs=4, window_size=20, order=3)
        >>> out = cbspline_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
    .. [2] Ramsay, J. O. (1988). Monotone regression splines in action. Statistical science,
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", MSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        .. plot::
            :include-source: True
            :caption: M-Spline

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
        >>> plt.show()
        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", MSplineBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineEval
        >>> mspline_basis = MSplineEval(n_basis_funcs=4, order=3)
        >>> out = mspline_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
    .. [2] Ramsay, J. O. (1988). Monotone regression splines in action. Statistical science,
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("evaluate_on_grid", MSplineBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        Evaluate and visualize 4 M-spline basis functions of order 3:

        .. plot::
            :include-source: True
            :caption: M-Spline

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
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", MSplineBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import MSplineConv
        >>> mspline_basis = MSplineConv(n_basis_funcs=4, window_size=20, order=3)
        >>> out = mspline_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
    .. [2] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
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
            label=label,
        )

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLinear)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: Linearly Spaced Raised-Cosine

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import RaisedCosineLinearEval
            >>> n_basis_funcs = 5
            >>> window_size=10
            >>> raised_cos_basis = RaisedCosineLinearEval(n_basis_funcs)
            >>> sample_points, basis_values = raised_cos_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", RaisedCosineBasisLinear)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLinearEval
        >>> raised_cos = RaisedCosineLinearEval(n_basis_funcs=4)
        >>> out = raised_cos.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
    .. [2] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
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
            width=width,
            label=label,
        )

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLinear)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: Linearly Spaced Raised-Cosine

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import RaisedCosineLinearConv
            >>> n_basis_funcs = 5
            >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
            >>> window_size=10
            >>> ortho_basis = RaisedCosineLinearConv(n_basis_funcs, window_size)
            >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", RaisedCosineBasisLinear)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLinearConv
        >>> raised_cos = RaisedCosineLinearConv(n_basis_funcs=4, window_size=20)
        >>> out = raised_cos.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
    .. [2] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
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
            label=label,
        )

    @add_docstring("evaluate_on_grid", RaisedCosineBasisLog)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: Log Spaced Raised-Cosine

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import RaisedCosineLogEval
            >>> n_basis_funcs = 5
            >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
            >>> window_size=10
            >>> ortho_basis = RaisedCosineLogEval(n_basis_funcs)
            >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", RaisedCosineBasisLog)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLogEval
        >>> raised_cos = RaisedCosineLogEval(n_basis_funcs=4)
        >>> out = raised_cos.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
    .. [2] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
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
        .. plot::
            :include-source: True
            :caption: Log Spaced Raised-Cosine

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import RaisedCosineLogConv
            >>> n_basis_funcs = 5
            >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
            >>> window_size=10
            >>> ortho_basis = RaisedCosineLogConv(n_basis_funcs, window_size)
            >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", RaisedCosineBasisLog)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import RaisedCosineLogConv
        >>> raised_cos = RaisedCosineLogConv(n_basis_funcs=4, window_size=20)
        >>> out = raised_cos.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 4)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
            label=label,
        )

    @add_docstring("evaluate_on_grid", OrthExponentialBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: Orthogonalized Exponential Decays

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import OrthExponentialEval
            >>> n_basis_funcs = 5
            >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
            >>> window_size=10
            >>> ortho_basis = OrthExponentialEval(n_basis_funcs, decay_rates=decay_rates)
            >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples=n_samples)

    @add_docstring("evaluate", OrthExponentialBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import OrthExponentialEval
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        >>> ortho_basis = OrthExponentialEval(n_basis_funcs=5, decay_rates=decay_rates)
        >>> out = ortho_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 5)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        .. plot::
            :include-source: True
            :caption: Orthogonalized Exponential Decays

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import OrthExponentialConv
            >>> n_basis_funcs = 5
            >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
            >>> window_size=10
            >>> ortho_basis = OrthExponentialConv(n_basis_funcs, window_size, decay_rates=decay_rates)
            >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", OrthExponentialBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import OrthExponentialConv
        >>> decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        >>> ortho_basis = OrthExponentialConv(n_basis_funcs=5, window_size=20, decay_rates=decay_rates)
        >>> out = ortho_basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 5)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
        return AtomicBasisMixin.set_input_shape(self, xi)

    def _check_window_size(self, window_size: int):
        """Specific window size check for ``Orthexponentialbasis``."""
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

    @add_docstring("evaluate_on_grid", IdentityBasis)
    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: Identity

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import IdentityEval
            >>> basis = IdentityEval()
            >>> sample_points, basis_values = basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples=n_samples)

    @add_docstring("evaluate", IdentityBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import IdentityEval
        >>> basis = IdentityEval()
        >>> out = basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 1)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        .. plot::
            :include-source: True
            :caption: History

            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import HistoryConv
            >>> window_size=100
            >>> basis = HistoryConv(window_size=window_size)
            >>> sample_points, basis_values = basis.evaluate_on_grid(window_size)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()

        """
        # ruff: noqa: D205, D400
        return super().evaluate_on_grid(n_samples)

    @add_docstring("evaluate", HistoryBasis)
    def evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import HistoryConv
        >>> basis = HistoryConv(window_size=20)
        >>> # evaluate for HistoryConv require a 1d input
        >>> out = basis.evaluate(np.random.randn(100, ))
        >>> out.shape
        (100, 20)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(sample_pts)

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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D205, D400
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
        # ruff: noqa: D400, D205
        return AtomicBasisMixin.set_input_shape(self, xi)

    @property
    def window_size(self):
        """Window size for the history basis."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int) -> None:
        self._check_window_size(window_size)
        self._window_size = window_size
        self._n_basis_funcs = window_size


class FourierEval(EvalBasisMixin, FourierBasis):
    """
    N-dimensional Fourier basis for feature expansion.

    This class generates a set of sine and cosine basis functions defined over
    an ``n``-dimensional input space. The basis functions are constructed from
    a Cartesian product of frequencies specified for each input dimension.
    Each selected frequency combination contributes two basis functions
    (cosine and sine), except for the all-zero frequency (DC component),
    which contributes only a cosine term.

    The class supports flexible frequency specification (integers, ranges, or
    arrays per dimension) and optional masking to include or exclude specific
    frequency combinations.

    Parameters
    ----------
    frequencies :
        Frequency specification(s).

        Single specification (broadcasted to all dimensions when ``ndim > 1``):

            * :class:`int`: An integer ``k`` with ``k >= 0``.

            * :class:`tuple`: ``(low, high)``, a 2-element tuple of integers with ``0 <= low < high``.

            * :class:`~numpy.ndarray`: 1-D NumPy array of non-negative integers. If not sorted ascending,
              a ``UserWarning`` is issue for non-sorted arrays.

        Per-dimension container:

            * A :class:`list` of length ``ndim`` whose elements are each a valid single specification.
              For ``ndim == 1``, a length-1 :class:`list` is also accepted.

    ndim :
        Dimensionality of the basis. Default is 1.

    bounds :
        Domain bounds for each dimension.

        * :class:`tuple`: ``(low, high)`` of floats: applies to all dimensions.
        * :class:`list` of :class:`tuple`: ``[(low, high), ...]``, one tuple per dimension, length must match ``ndim``.
        * :class:`None <NoneType>`: the domain is inferred from the input data (maximum to minimum values).

        In all cases, ``low`` must be strictly less than ``high``, and values must be convertible to floats.

    frequency_mask :
        Optional mask specifying which frequency components to include.
        Can be:

        * :class:`~typing.Literal`: either ``"no-intercept"`` - default - which drops
          the 0-frequency DC term, or ``"all"`` which keeps all the frequencies -
          equivalent to :class:`None <NoneType>`. The default excludes the intercept
          because these basis objects are most commonly used to generate design matrices
          for NeMoS GLMs, which already include an intercept term by default, making an
          additional intercept in the design matrix redundant.

        * Array-like of integers {0, 1} or booleans: Selects frequencies to
          keep (1/True) or exclude (0/False). Shape must match the number of
          available frequencies for each dimension.

        * :class:`~typing.Callable`: A function applied to each frequency index (one index
          per dimension), returning a single boolean or {0, 1} indicating whether
          to keep that frequency.

        * :class:`None <NoneType>`: All frequencies are kept.

        Values must be 0/1 or boolean. Callables must return a single boolean or
        {0, 1} value for each frequency coordinate.

    label :
        Descriptive label for the basis (e.g., to use in plots or summaries).

    Notes
    -----
    - If ``frequency_mask`` is provided, only the selected frequency
      combinations are used to build the basis.
    - The output of ``compute_features`` contains both cosine and sine components for
      each active frequency combination, except that the all-zero frequency
      includes only a cosine term.
    - When a :class:`tuple` is provided as a frequency, it is interpreted
      as a single range specification. Tuples that are not exactly a 2-element
      tuple of non-negative integers are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import FourierEval
    >>> rng = np.random.default_rng(0)

    **1D: basic usage**

    >>> n_freq = 5
    >>> fourier_1d = FourierEval(n_freq)
    >>> # cos at 0..4 (5) + sin at 1..4 (4) = 9
    >>> fourier_1d.n_basis_funcs
    8
    >>> x = rng.normal(size=8)
    >>> X = fourier_1d.compute_features(x)
    >>> X.shape  # (n_samples, n_basis_funcs)
    (8, 8)

    **2D: unmasked grid of frequency pairs**

    >>> fourier_2d = FourierEval(n_freq, ndim=2)
    >>> # (5*5 frequency pairs) * 2 (cos+sin) - 1 (no sine at DC) = 49
    >>> fourier_2d.n_basis_funcs
    48
    >>> x, y = rng.normal(size=(2, 6))
    >>> X = fourier_2d.compute_features(x, y)
    >>> X.shape
    (6, 48)

    **2D: masking with an array (drop 3 pairs)**

    >>> mask = np.ones((5, 5))
    >>> # drop 3 frequency pairs, including DC term (0,0)
    >>> mask[[0, 0, 1], [0, 1, 2]] = 0
    >>> fourier_2d_masked = FourierEval(
    ...     n_freq,
    ...     ndim=2,
    ...     frequency_mask=mask
    ... )
    >>> # (5*5-3 frequency pairs) * 2 (cos+sin) = 44
    >>> fourier_2d_masked.n_basis_funcs
    44

    **2D: masking with a callable**

    >>> # keep pairs inside a circle of radius 3.5 in frequency space
    >>> keep_circle = lambda fx, fy: (fx**2 + fy**2) ** 0.5 < 3.5
    >>> fourier_2d_funcmask = FourierEval(
    ...     n_freq,
    ...     ndim=2,
    ...     frequency_mask=keep_circle
    ... )
    >>> fourier_2d_funcmask.n_basis_funcs
    25

    **Explicit frequency specifications**

    >>> # mix forms per-dimension: an explicit array
    >>> # and an inclusive tuple (low, high)
    >>> fourier_mixed = FourierEval(
    ...     frequencies=[np.arange(3), (1, 4)],
    ...     ndim=2
    ... )
    >>> # (3*3 frequency pairs) * 2 (cos+sin) = 18; no DC term (0, 0)
    >>> fourier_mixed.n_basis_funcs
    18

    """

    def __init__(
        self,
        frequencies: (
            int
            | Tuple[int, int]
            | List[int]
            | List[Tuple[int, int]]
            | NDArray
            | List[NDArray]
        ),
        ndim: int = 1,
        bounds: Optional[Tuple[float, float] | Tuple[Tuple[float, float]]] = None,
        frequency_mask: (
            Literal["all", "no-intercept"] | NDArray[bool] | None
        ) = "no-intercept",
        label: Optional[str] = "FourierEval",
    ):
        FourierBasis.__init__(
            self,
            frequencies=frequencies,
            label=label,
            frequency_mask=frequency_mask,
            ndim=ndim,
        )
        EvalBasisMixin.__init__(self, bounds=bounds)

    @add_docstring("evaluate_on_grid", FourierBasis)
    def evaluate_on_grid(self, *n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: FourierEval

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import FourierEval
            >>> n_frequencies = 5
            >>> fourier_basis = FourierEval(n_frequencies)
            >>> sample_points, basis_values = fourier_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()
        """
        return super().evaluate_on_grid(*n_samples)

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples, ))  # raw time series
        >>> basis = FourierEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 18)

        """
        return super().compute_features(*xi)

    @add_docstring("split_by_feature", FourierBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierEval
        >>> from nemos.glm import GLM
        >>> basis = FourierEval(6, label="one_input")
        >>> X = basis.compute_features(np.random.randn(20,))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 10)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, *xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.FourierEval(5)
        >>> # Configure with an integer input:
        >>> _ = basis.set_input_shape(3)
        >>> basis.n_output_features
        24
        >>> # Configure with a tuple:
        >>> _ = basis.set_input_shape((4, 5))
        >>> basis.n_output_features
        160
        >>> # Configure with an array:
        >>> x = np.ones((10, 4, 5))
        >>> _ = basis.set_input_shape(x)
        >>> basis.n_output_features
        160

        """
        return BasisMixin.set_input_shape(self, *xi)

    @add_docstring("evaluate", FourierBasis)
    def evaluate(self, *sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierEval
        >>> basis = FourierEval(4)
        >>> out = basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 6)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(*sample_pts)

    @property
    def bounds(self) -> Tuple[Tuple[float, float]] | None:
        """Bounds.

        Tuple of bounds, one per dimension or None if no bounds are
        provided.
        """
        return self._bounds

    @bounds.setter
    def bounds(
        self, values: Tuple[float, float] | Sequence[Tuple[float, float]] | None
    ):
        if values is None:
            self._bounds = None
            return

        def _is_leaf(x):
            return isinstance(x, Sequence) and all(
                isinstance(xi, Number)
                or xi is None
                or isinstance(xi, jax.numpy.generic)  # NumPy/JAX numpy scalar types
                or (is_numpy_array_like(xi)[1] and xi.ndim == 0)  # 0-D arrays
                for xi in x
            )

        values = jax.tree_util.tree_leaves(values, is_leaf=_is_leaf)

        if len(values) == 1:
            values = self._format_bounds(values[0])
            values = (values,) * self.ndim

        elif len(values) != self.ndim:
            raise TypeError(
                f"Invalid bounds ``{values}`` provided. "
                "When provided, the bounds should be one or multiple tuples containing pair of floats.\n"
                "If multiple tuples are provided, one must provide a tuple per each dimension"
                "of the basis. "
            )
        else:
            values = jax.tree_util.tree_map(
                self._format_bounds, values, is_leaf=_is_leaf
            )
            values = tuple(vals for vals in values)

        self._bounds = values
