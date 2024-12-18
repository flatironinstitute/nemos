# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import copy
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor
from scipy.interpolate import splev

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin


class SplineBasis(Basis, AtomicBasisMixin, abc.ABC):
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
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    Attributes
    ----------
    order : int
        Spline order.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        order: int = 2,
        label: Optional[str] = None,
        mode: Literal["conv", "eval"] = "eval",
    ) -> None:
        self.order = order
        AtomicBasisMixin.__init__(self, n_basis_funcs=n_basis_funcs)
        super().__init__(
            label=label,
            mode=mode,
        )

        self._n_input_dimensionality = 1

    @property
    def order(self):
        """
        Spline order.

        Spline order, i.e. the polynomial degree of the spline plus one.
        """
        return self._order

    @order.setter
    def order(self, value):
        """Setter for the order parameter."""
        if value != int(value):
            raise ValueError(
                f"Spline order must be an integer! Order {value} provided."
            )
        value = int(value)
        if value < 1:
            raise ValueError(f"Spline order must be positive! Order {value} provided.")

        # Set to None only the first time the setter is called.
        orig_order = copy.deepcopy(getattr(self, "_order", None))

        # Set the order
        self._order = value

        # If the order was already initialized, re-check basis
        if orig_order is not None:
            try:
                self._check_n_basis_min()
            except ValueError as e:
                self._order = orig_order
                raise e

    def _generate_knots(
        self,
        is_cyclic: bool = False,
    ) -> NDArray:
        """
        Generate knots locations for spline basis functions.

        Parameters
        ----------
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


class MSplineBasis(SplineBasis, abc.ABC):
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
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    order :
        The order of the splines used in basis functions. Must be between [1,
        n_basis_funcs]. Default is 2. Higher order splines have more continuous
        derivatives at each interior knot, resulting in smoother basis functions.
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
    (x-axis) of an MSpline basis is expanded by a factor of :math:`\alpha`, the values on the co-domain (y-axis) values
    will shrink by a factor of :math:`1/\alpha`.
    For example, over the standard bounds of (0, 1), the maximum value of the MSpline is 18.
    If we set the bounds to (0, 2), the maximum value will be 9, i.e., 18 / 2.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import MSplineEval
    >>> n_basis_funcs = 5
    >>> order = 3
    >>> mspline_basis = MSplineEval(n_basis_funcs, order=order)
    >>> sample_points = linspace(0, 1, 100)
    >>> basis_functions = mspline_basis.compute_features(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode: Literal["eval", "conv"] = "eval",
        order: int = 2,
        label: Optional[str] = "MSplineEval",
    ) -> None:
        super().__init__(
            n_basis_funcs,
            mode=mode,
            order=order,
            label=label,
        )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        """
        Evaluate the M-spline basis functions at given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the M-spline is evaluated.
            `sample_pts` is a n-dimensional (n >= 1) array with first axis being the samples, i.e.
            `sample_pts.shape[0] == n_samples`.

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
        sample_pts, scaling = min_max_rescale_samples(
            sample_pts, getattr(self, "bounds", None)
        )
        # add knots if not passed
        knot_locs = self._generate_knots(is_cyclic=False)

        # get the original shape
        shape = sample_pts.shape
        X = np.stack(
            [
                mspline(
                    sample_pts.reshape(
                        -1,
                    ),
                    self.order,
                    i,
                    knot_locs,
                )
                for i in range(self.n_basis_funcs)
            ],
            axis=1,
        )
        X = X.reshape(*shape, X.shape[1])
        # re-normalize so that it integrates to 1 over the range.
        X /= scaling[..., None]

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
            Shape: ``(n_samples,)``.
        Y : NDArray
            A 2D array where each row corresponds to the evaluated M-spline basis
            function values at the points in X. Shape: ``(n_samples, n_basis_funcs)``.
        """
        return super().evaluate_on_grid(n_samples)


class BSplineBasis(SplineBasis, abc.ABC):
    """
    B-spline 1-dimensional basis functions.

    Implementation of the one-dimensional BSpline basis [1]_.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    mode :
        The mode of operation. ``'eval'`` for evaluation at sample points,
        'conv' for convolutional operation.
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
    .. [1] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
        Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        order: int = 4,
        label: Optional[str] = "BSplineBasis",
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            order=order,
            label=label,
        )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the B-spline is evaluated, shape (n_samples,).
            `sample_pts` is a n-dimensional (n >= 1) array with first axis being the samples, i.e.
            `sample_pts.shape[0] == n_samples`.

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
        The evaluation is performed by looping over each element and using ``splev``
        from SciPy to compute the basis values.
        """
        sample_pts, _ = min_max_rescale_samples(
            sample_pts, getattr(self, "bounds", None)
        )
        # add knots
        knot_locs = self._generate_knots(is_cyclic=False)

        # reshape to flat and store original shape
        shape = sample_pts.shape
        sample_pts = sample_pts.reshape(
            -1,
        )

        basis_eval = bspline(
            sample_pts, knot_locs, order=self.order, der=0, outer_ok=False
        )
        basis_eval = basis_eval.reshape(*shape, basis_eval.shape[1])
        return basis_eval

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the B-spline basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of points in the uniformly spaced grid. A higher number of
            samples will result in a more detailed visualization of the basis functions.

        Returns
        -------
        X :
            Array of shape ``(n_samples,)`` containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Raised cosine basis functions, shape ``(n_samples, n_basis_funcs)``

        Notes
        -----
        The evaluation is performed by looping over each element and using ``splev`` from
        SciPy to compute the basis values.
        """
        return super().evaluate_on_grid(n_samples)


class CyclicBSplineBasis(SplineBasis, abc.ABC):
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
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    Attributes
    ----------
    n_basis_funcs :
        Number of basis functions, int.
    order :
        Order of the splines used in basis functions, int.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        order: int = 4,
        label: Optional[str] = "CyclicBSplineBasis",
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            order=order,
            label=label,
        )
        if self.order < 2:
            raise ValueError(
                f"Order >= 2 required for cyclic B-spline, "
                f"order {self.order} specified instead!"
            )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(
        self,
        sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor,
    ) -> FeatureMatrix:
        """Evaluate the Cyclic B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the cyclic B-spline is evaluated. Samples must be stored in a
            multi-dimensional array with first axis being the samples, i.e. `sample_pts.shape[0] == n_samples`.

        Returns
        -------
        basis_funcs :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        Notes
        -----
        The evaluation is performed by looping over each element and using ``splev`` from
        SciPy to compute the basis values.

        """
        sample_pts, _ = min_max_rescale_samples(
            sample_pts, getattr(self, "bounds", None)
        )
        knot_locs = self._generate_knots(is_cyclic=True)

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

        # reshape to flat and store original shape
        shape = sample_pts.shape
        sample_pts = sample_pts.reshape(
            -1,
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

        basis_eval = basis_eval.reshape(*shape, basis_eval.shape[1])
        return basis_eval

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the Cyclic B-spline basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of points in the uniformly spaced grid. A higher number of
            samples will result in a more detailed visualization of the basis functions.

        Returns
        -------
        X :
            Array of shape ``(n_samples,)`` containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Raised cosine basis functions, shape ``(n_samples, n_basis_funcs)``

        Notes
        -----
        The evaluation is performed by looping over each element and using ``splev`` from
        SciPy to compute the basis values.
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

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import linspace
    >>> from nemos.basis._spline_basis import mspline
    >>> sample_points = linspace(0, 1, 100)
    >>> mspline_eval = mspline(x=sample_points, k=3, i=2, T=np.random.rand(7)) # define a cubic M-spline
    >>> mspline_eval.shape
    (100,)
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

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import linspace
    >>> from nemos.basis._spline_basis import bspline
    >>> sample_points = linspace(0, 1, 100)
    >>> knots = np.array([0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1])
    >>> bspline_eval = bspline(sample_points, knots) # define a cubic B-spline
    >>> bspline_eval.shape
    (100, 10)
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
