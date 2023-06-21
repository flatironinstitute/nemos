"""Bases classes.
"""
# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations

import abc
import warnings
from typing import Tuple

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.interpolate import splev

from utils import rowWiseKron


class Basis:
    """
    Generic class for basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    GB_limit : float, optional
        Limit in GB for the model matrix size. Default is 16.0.

    Attributes
    ----------
    _n_basis_funcs : int
        Number of basis functions.
    _GB_limit : float
        Limit in GB for the model matrix size.
    _n_input_samples : int
        Number of inputs that the evaluate method requires.

    Methods
    -------
    evaluate(*xi)
        Evaluate the basis at the input samples x1,...,xn

    check_input_number(x)
        Check the number of input samples.

    check_samples_consistency(x)
        Check the consistency of sample sizes.

    check_full_model_matrix_size(n_samples, dtype=np.float64)
        Check the size of the full model matrix.

    __add__(other)
        Add two Basis objects together.

    __mul__(other)
        Multiply two Basis objects together. Returns a Basis of 'multiply' type, which can be used to model
        multi-dimensional response functions.
    """

    def __init__(self, n_basis_funcs: int, GB_limit: float = 16.0):
        self._n_basis_funcs = n_basis_funcs
        self._GB_limit = GB_limit
        self._n_input_samples = 0

    def _evaluate(self, x: tuple[NDArray]):
        """
        Evaluate the basis set at the given samples x1,...,xn using the subclass-specific "_evaluate" method.

        Parameters
        ----------
        x1,...,xn : tuple
            The input samples.

        Returns
        -------
        NDArray
            The basis function evaluated at the samples (Time points x number of basis).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the _evaluate method.
        """
        subclass_name = type(self).__name__
        raise NotImplementedError(f"{subclass_name} must implement _evaluate method!")

    def _get_samples(self, n_samples: tuple[int]):
        """
        Evaluate the basis set at the given samples x1,...,xn using the subclass-specific "_gen_basis" method.

        Parameters
        ----------
        n_samples[0],...,n_samples[n] : int
            The number of samples in each axis of the grid.

        Returns
        -------
        tuple[NDArray]
            The  equi-spaced samples covering the basis domain.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the _gen_basis method.
        """
        subclass_name = type(self).__name__
        raise NotImplementedError(f"{subclass_name} must implement _evaluate method!")

    def evaluate(self, *xi: NDArray):
        """
        Evaluate the basis set at the given samples x1,...,xn using the subclass-specific "_evaluate" method.

        Parameters
        ----------
        x1,...,xn : NDArray
            The input samples.

        Returns
        -------
        NDArray
            The generated basis functions.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the _evaluate method.
        """
        # checks on input and outputs
        self._check_samples_consistency(xi)
        self._check_full_model_matrix_size(xi[0].shape[0])
        self._check_input_number(xi)

        return self._evaluate(xi)

    def gen_basis(self, *n_samples: int):
        """
        Evaluate the basis set on a grid of equi-spaced sample points. The i-th axis of the grid will be sampled
        with n_samples[i] equi-spaced points.

        Parameters
        ----------
        n_samples[0],...,n_samples[n] : int
            The number of samples in each axis of the grid.

        Returns
        -------
        NDArray
            The basis function evaluated at the samples :math:`\prod_{i} \text{n_samples}_i \times x \text{n_basis}`.

        """
        self._check_input_number(n_samples)
        self._check_full_model_matrix_size(np.prod(n_samples) * self._n_basis_funcs)

        # get the samples
        sample_tuple = self._get_samples(n_samples)
        Xs = np.meshgrid(*sample_tuple,indexing='ij')

        # call evaluate to evaluate the basis on a flat NDArray and reshape to match meshgrid output
        Y = self.evaluate(*tuple(grid_axis.flatten() for grid_axis in Xs)).reshape(
            (self._n_basis_funcs,) + n_samples
        )

        return *Xs, Y

    def _check_input_number(self, x: tuple):
        """
        Check that the number of inputs provided by the user matches the number of inputs that the Basis object requires.

        Parameters
        ----------
        x : tuple
            The input samples.

        Raises
        ------
        ValueError
            If the number of inputs doesn't match what the Basis object requires.
        """
        if len(x) != self._n_input_samples:
            raise ValueError(
                f"Input number mismatch. Basis requires {self._n_input_samples} input samples, {len(x)} inputs provided instead."
            )

    def _check_samples_consistency(self, x: tuple[NDArray]):
        """
        Check that each input provided to the Basis object has the same number of time points.

        Parameters
        ----------
        x : tuple of NDArray
            The input samples.

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs.
        """
        sample_sizes = [samp.shape[0] for samp in x]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(
                "Sample size mismatch. Input elements have inconsistent sample sizes."
            )

    def _check_full_model_matrix_size(self, n_samples, dtype=np.float64):
        """
        Check the size in GB of the full model matrix is <= self._GB_limit.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        dtype : type, optional
            Data type of the model matrix. Default is np.float64.

        Raises
        ------
        MemoryError
            If the size of the model matrix exceeds the specified memory limit.
        """
        size_in_bytes = np.dtype(dtype).itemsize * n_samples * self._n_basis_funcs
        if size_in_bytes > self._GB_limit * 10**9:
            raise MemoryError(f"Model matrix size exceeds {self._GB_limit} GB.")

    def __add__(self, other):
        """
        Add two Basis objects together.

        Parameters
        ----------
        other : Basis
            The other Basis object to add.

        Returns
        -------
        Basis
            The resulting Basis object.
        """
        return addBasis(self, other)

    def __mul__(self, other):
        """
        Multiply two Basis objects together.

        Parameters
        ----------
        other : Basis
            The other Basis object to multiply.

        Returns
        -------
        Basis
            The resulting Basis object.
        """
        return mulBasis(self, other)


class addBasis(Basis):
    """
    Class representing the addition of two Basis objects.

    Parameters
    ----------
    basis1 : Basis
        First basis object to add.
    basis2 : Basis
        Second basis object to add.

    Attributes
    ----------
    _n_basis_funcs : int
        Number of basis functions.
    _n_input_samples : int
        Number of input samples.
    _basis1 : Basis
        First basis object.
    _basis2 : Basis
        Second basis object.

    Methods
    -------
    _evaluate(x_tuple)
        Evaluate t

    """

    def __init__(self, basis1, basis2):
        self._n_basis_funcs = basis1._n_basis_funcs + basis2._n_basis_funcs
        super().__init__(self._n_basis_funcs, GB_limit=basis1._GB_limit)
        self._n_input_samples = basis1._n_input_samples + basis2._n_input_samples
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _evaluate(self, x_tuple: tuple[NDArray]):
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        x_tuple : tuple
            Tuple of input samples.

        Returns
        -------
        NDArray
            The basis function evaluated at the samples (Time points x number of basis)
        """
        return np.vstack(
            (
                self._basis1._evaluate(x_tuple[: self._basis1._n_input_samples]),
                self._basis2._evaluate(x_tuple[self._basis1._n_input_samples :]),
            )
        )

    def _get_samples(self, n_samples: tuple[int]):
        """
        Get equi-spaced samples for all the input dimensions. This will be used to evaluate
        the basis on a grid of points derived by the samples

        Parameters
        ----------
        n_samples[0],...,n_samples[n] : int
            The number of samples in each axis of the grid.

        Returns
        -------
        tuple[NDArray]
            The equi-spaced sample locations for each coordinate.

        """
        sample_1 = self._basis1._get_samples(n_samples[: self._basis1._n_input_samples])
        sample_2 = self._basis2._get_samples(n_samples[self._basis1._n_input_samples:])
        return sample_1 + sample_2


class mulBasis(Basis):
    """
    Class representing the multiplication (external product) of two Basis objects.

    Parameters
    ----------
    basis1 : Basis
        First basis object to multiply.
    basis2 : Basis
        Second basis object to multiply.

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.
    n_input_samples : int
        Number of input samples.
    basis1 : Basis
        First basis object.
    basis2 : Basis
        Second basis object.

    Methods
    -------
    _evaluate(x_tuple)
        Evaluates the basis function at the samples x_tuple[0],..,x_tuple[n]
    """

    def __init__(self, basis1, basis2):
        self._n_basis_funcs = basis1._n_basis_funcs * basis2._n_basis_funcs
        super().__init__(self._n_basis_funcs, GB_limit=basis1._GB_limit)
        self._n_input_samples = basis1._n_input_samples + basis2._n_input_samples
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _evaluate(self, x_tuple: tuple[NDArray]):
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        x_tuple : tuple
            Tuple of input samples.

        Returns
        -------
        NDArray
            The basis function evaluated at the samples (Time points x number of basis)
        """
        return rowWiseKron(
            self._basis1._evaluate(x_tuple[: self._basis1._n_input_samples]),
            self._basis2._evaluate(x_tuple[self._basis1._n_input_samples :]),
            transpose=True,
        )

    def _get_samples(self, n_samples: tuple[int]):
        """
        Get equi-spaced samples for all the input dimensions. This will be used to evaluate
        the basis on a grid of points derived by the samples

        Parameters
        ----------
        n_samples[0],...,n_samples[n] : int
            The number of samples in each axis of the grid.

        Returns
        -------
        tuple[NDArray]
            The equi-spaced sample locations for each coordinate.

        """

        sample_1 = self._basis1._get_samples(n_samples[: self._basis1._n_input_samples])
        sample_2 = self._basis2._get_samples(n_samples[self._basis1._n_input_samples:])
        return sample_1 + sample_2


class SplineBasis(Basis):
    """
    SplineBasis class inherits from the Basis class and represents spline basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    order : int, optional
        Spline order. Default is 2.

    Attributes
    ----------
    _order : int
        Spline order.
    _n_input_samples : int
        Number of input samples.

    Methods
    -------
    _generate_knots(sample_pts, perc_low, perc_high, is_cyclic=False)
        Generate knot locations for spline basis functions.

    """

    def __init__(self, n_basis_funcs: int, order: int = 2):
        super().__init__(n_basis_funcs)
        self._order = order
        self._n_input_samples = 1
        if self._order < 1:
            raise ValueError("Spline order must be positive!")

    def _generate_knots(
        self,
        sample_pts: NDArray,
        perc_low: float,
        perc_high: float,
        is_cyclic: bool = False,
    ) -> NDArray:
        """
        Generate knot locations for spline basis functions.

        Parameters
        ----------
        sample_pts : NDArray
            The sample points.
        perc_low : float
            The low percentile value.
        perc_high : float
            The high percentile value.
        is_cyclic : bool, optional
            Whether the spline is cyclic. Default is False.

        Returns
        -------
        NDArray
            The knot locations for the spline basis functions.

        Raises
        ------
        ValueError
            If the spline order is larger than the number of basis functions.
        AssertionError
            If the percentiles or order values are not within the valid range.
        """
        # Determine number of interior knots.
        num_interior_knots = self._n_basis_funcs - self._order
        if is_cyclic:
            num_interior_knots += self._order - 1

        # Check hyperparameters.
        if num_interior_knots < 0:
            raise ValueError(
                "Spline `order` parameter cannot be larger "
                "than `n_basis_funcs` parameter."
            )

        assert 0 <= perc_low <= 1, "Specify `perc_low` as a float between 0 and 1."
        assert 0 <= perc_high <= 1, "Specify `perc_high` as a float between 0 and 1."
        assert perc_low < perc_high, "perc_low must be less than perc_high."

        # clip to avoid numerical errors in case of percentile numerical precision close to 0 and 1
        mn = np.nanpercentile(sample_pts, np.clip(perc_low * 100, 0, 100))
        mx = np.nanpercentile(sample_pts, np.clip(perc_high * 100, 0, 100))

        self.knot_locs = np.concatenate(
            (
                mn * np.ones(self._order - 1),
                np.linspace(mn, mx, num_interior_knots + 2),
                mx * np.ones(self._order - 1),
            )
        )
        return self.knot_locs

    def _get_samples(self, n_samples: tuple[int]):
        """
        Generate the basis functions on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples : tuple of int
           The number of samples in each axis of the grid.

        Returns
        -------
        tuple[NDArray]
           The equi-spaced sample location.
        """
        return (np.linspace(0, 1, n_samples[0]),)


class BSplineBasis(SplineBasis):
    """
    B-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    order : int, optional
        Order of the splines used in basis functions. Must lie within [1, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.

    Attributes
    ----------
    _order : int
        Spline order.
    _n_input_samples : int
        Number of input samples.

    Methods
    -------
    _evaluate(x_tuple)
       Evaluate the basis function at the samples x_tuple[0]. x_tuple must be of length 1 in order to pass the checks
       of super().evaluate

    References
    ----------
    [2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
    Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5

    """

    def __init__(self, n_basis_funcs: int, order: int = 2):
        super().__init__(n_basis_funcs, order=order)

    def _evaluate(
        self, sample_pts: tuple[NDArray], outer_ok: bool = False, der: int = 0
    ) -> NDArray:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts : NDArray
            The sample points at which the B-spline is evaluated.
        outer_ok : bool, optional
            If True, accepts samples outside the knots range. If False, raises a ValueError.
        der : int, optional
            Order of the derivative of the B-spline (default is 0, e.g., B-spline evaluation).

        Returns
        -------
        NDArray
            The basis function evaluated at the samples (Time points x number of basis)

        Raises
        ------
        AssertionError
            If the sample points are not within the B-spline knots range unless `outer_ok=True`.

        Notes
        -----
        This method evaluates the B-spline basis functions at the given sample points. It requires the knots to be defined
        through the `_generate_knots` method. Knots will be flushed at the end of the call.

        The evaluation is performed by looping over each element and using `splev` from SciPy to compute the basis values.
        """
        # sample_points is a tuple of length 1 if passed the checks in Basis.evaluate
        sample_pts = sample_pts[0]

        # add knots
        self._generate_knots(sample_pts, 0.0, 1.0)

        # sort the knots in case user passed
        knots = self.knot_locs
        knots.sort()
        nk = knots.shape[0]

        # check for out of range points (in cyclic b-spline need_outer must be set to False)
        need_outer = any(sample_pts < knots[self._order - 1]) or any(
            sample_pts > knots[nk - self._order]
        )
        assert (
            not need_outer
        ) | outer_ok, 'sample points must lie within the B-spline knots range unless "outer_ok==True".'

        # select knots that are within the knots range (this takes care of eventual NaNs)
        in_sample = (sample_pts >= knots[0]) & (sample_pts <= knots[-1])

        if need_outer:
            reps = self._order - 1
            knots = np.hstack(
                (np.ones(reps) * knots[0], knots, np.ones(reps) * knots[-1])
            )
            nk = knots.shape[0]
        else:
            reps = 0

        # number of basis elements
        n_basis = nk - self._order

        # initialize the basis element container
        basis_eval = np.zeros((n_basis - 2 * reps, sample_pts.shape[0]))

        # loop one element at the time and evaluate the basis using splev
        id_basis = np.eye(n_basis, nk, dtype=np.int8)
        for i in range(reps, len(knots) - self._order - reps):
            basis_eval[i - reps, in_sample] = splev(
                sample_pts[in_sample], (knots, id_basis[i], self._order - 1), der=der
            )

        delattr(self, "knot_locs")
        return basis_eval



class Cyclic_BSplineBasis(BSplineBasis):
    """
    B-spline 1-dimensional basis functions for cyclic splines.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    order : int, optional
        Order of the splines used in basis functions. Must lie within [1, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.

    Attributes
    ----------
    _n_basis_funcs : int
        Number of basis functions.
    _order : int
        Order of the splines used in basis functions.

    Methods
    -------
    _evaluate(sample_pts, der)
        Evaluate the B-spline basis functions with given sample points.
    """

    def __init__(self, n_basis_funcs: int, order: int = 2):
        super().__init__(n_basis_funcs, order=order)
        assert (
            self._order >= 2
        ), f"Order >= 2 required for cyclic B-spline, order {self._order} specified instead!"
        assert (
            self._n_basis_funcs >= order + 2
        ), "n_basis_funcs >= order + 2 required for cyclic B-spline"
        assert (
            self._n_basis_funcs >= 2 * order - 2
        ), "n_basis_funcs >= 2*(order - 1) required for cyclic B-spline"

    def _evaluate(self, sample_pts: tuple[NDArray], der: int = 0) -> NDArray:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts : tuple
            The sample points at which the B-spline is evaluated. Must be a tuple of length 1.
        der : int, optional
            Order of the derivative of the B-spline (default is 0, e.g., B-spline evaluation).

        Returns
        -------
        NDArray
            The basis function evaluated at the samples (Time points x number of basis)

        Raises
        ------
        AssertionError
            If the sample points are not within the B-spline knots range unless `outer_ok=True`.

        Notes
        -----
        This method evaluates the B-spline basis functions at the given sample points. It requires the knots to be defined
        through the `_generate_knots` method. Knots will be flushed at the end of the call.

        The evaluation is performed by looping over each element and using `splev` from SciPy to compute the basis values.
        """
        # sample points is a tuple of length 1 if the checks in Basis.evaluate did not raise an exception
        sample_pts = sample_pts[0]

        self._generate_knots(sample_pts, 0.0, 1.0, is_cyclic=True)

        # for cyclic, do not repeat knots
        self.knot_locs = np.unique(self.knot_locs)

        knots_orig = self.knot_locs.copy()

        nk = knots_orig.shape[0]

        # make sure knots are sorted
        knots_orig.sort()
        xc = knots_orig[nk - 2 * self._order + 1]
        knots = np.hstack(
            (
                self.knot_locs[0]
                - self.knot_locs[-1]
                + self.knot_locs[nk - self._order : nk - 1],
                self.knot_locs,
            )
        )
        ind = sample_pts > xc

        # temporarily set the extended knots as attribute
        self.knot_locs = knots
        basis_eval = super()._evaluate((sample_pts,), outer_ok=True, der=der)
        sample_pts[ind] = sample_pts[ind] - knots.max() + knots_orig[0]
        if np.sum(ind):
            X2 = super()._evaluate((sample_pts[ind],), outer_ok=True, der=der)
            basis_eval[:, ind] = basis_eval[:, ind] + X2
        # restore points
        sample_pts[ind] = sample_pts[ind] + knots.max() - knots_orig[0]
        return basis_eval


class MSplineBasis(SplineBasis):
    """M-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.
    order
        Order of the splines used in basis functions. Must lie within [1,
        n_basis_funcs]. The m-splines have ``order-2`` continuous derivatives
        at each interior knot. The higher this number, the smoother the basis
        representation will be.


    References
    ----------
    .. [1] Ramsay, J. O. (1988). Monotone regression splines in action.
       Statistical science, 3(4), 425-441.

    """

    def __init__(self, n_basis_funcs: int,  order: int = 2):
        super().__init__(n_basis_funcs, order)


    def _evaluate(self, sample_pts: tuple[NDArray]) -> NDArray:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts : (n_pts,)
            Spacing for basis functions, holding elements on the interval [0,
            window_size). A good default is np.arange(window_size).

        Returns
        -------
        basis_funcs : (n_basis_funcs, n_pts)
            Evaluated spline basis functions.

        """

        sample_pts = sample_pts[0]

        # add knots if not passed
        self._generate_knots(sample_pts, 0.0, 1.0, is_cyclic=True)


        return np.stack(
            [mspline(sample_pts, self._order, i, self.knot_locs) for i in range(self._n_basis_funcs)],
            axis=0
        )

def mspline(x: NDArray, k: int, i: int, T: NDArray):
    """Compute M-spline basis function.

    Parameters
    ----------
    x : (n_pts,)
        Spacing for basis functions, holding elements on the interval [0,
        window_size). If None, use a grid (``np.arange(self.window_size)``).
    k
        Order of the spline basis.
    i
        Number of the spline basis.
    T : (k + n_basis_funcs,)
        knot locations. should lie in interval [0, 1].


    Returns
    -------
    spline : (n_pts,)
        M-spline basis function.
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
        return k * (
                (x - T[i]) * mspline(x, k - 1, i, T)
                + (T[i + k] - x) * mspline(x, k - 1, i + 1, T)
        ) / ((k - 1) * (T[i + k] - T[i]))




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    samples = np.random.normal(size=100)
    basis1 = BSplineBasis(15, order=4)
    basis2 = BSplineBasis(15, order=4)
    basis_add = basis1 + basis2

    basis_add_add = basis_add + basis2
    basis_add_add_add = basis_add_add + basis_add

    print(basis_add.evaluate(samples, samples).shape)
    print(basis_add_add.evaluate(samples, samples, samples).shape)
    print(basis_add_add_add.evaluate(samples, samples, samples, samples, samples).shape)

    basis1 = BSplineBasis(15, order=4)
    basis2 = MSplineBasis(15, order=4)
    mulbase = basis1 * basis2
    X, Y, Z = mulbase.gen_basis(100, 110)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Z = np.array(Z)
    Z[Z == 0] = np.nan
    ax.plot_surface(X, Y, Z[50], cmap="viridis", alpha=0.8)
    ax.plot_surface(X, Y, Z[100], cmap="rainbow", alpha=0.8)
    ax.plot_surface(X, Y, Z[200], cmap="inferno", alpha=0.8)

    # Customize the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Overlapped Surfaces")

    print("multiply and additive base with a evaluate type base")
    basis1 = BSplineBasis(6, order=4)
    basis2 = MSplineBasis(7, order=4)
    basis3 = Cyclic_BSplineBasis(8, order=4)
    base_res = (basis1 + basis2) * basis3
    X = base_res.evaluate(
        np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    )
    print(X.shape, (6 + 7) * 8)



    basis1 = BSplineBasis(6, order=4)
    basis2 = BSplineBasis(7, order=4)
    basis3 = MSplineBasis(8, order=4)

    multb = basis1 + basis2 * basis3
    X, Y, W, Z = multb.gen_basis(10, 11, 12)