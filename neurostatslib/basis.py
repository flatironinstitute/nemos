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
    """Generic class for basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    base_type : str
        Type of basis functions ('evaluate', 'convolve', 'add', 'multiply').
    basis1 : Basis, optional
        First basis function for 'add' or 'multiply' base_type. Default is None.
    basis2 : Basis, optional
        Second basis function for 'add' or 'multiply' base_type. Default is None.
    GB_limit : float, optional
        Limit in GB for the model matrix size. Default is 16.0.

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.
    base_type : str
        Type of basis functions ('evaluate', 'convolve', 'add', 'multiply').
    GB_limit : float
        Limit in GB for the model matrix size.

    Methods
    -------
    evaluate(samples, check_support=False)
        Evaluate the basis functions at given samples.
    convolve(samples)
        Convolve the basis functions with given samples.
    gen_basis(*x)
        Generate the basis matrix using provided input samples.
    check_input_number(x)
        Check the number of input samples.
    check_samples_consistency(x)
        Check the consistency of sample sizes.
    check_full_model_matrix_size(n_samples, dtype=np.float64)
        Check the size of the full model matrix.
    __add__(other)
        Add two Basis objects together.
    __mul__(other)
        Multiply two Basis objects element-wise.

    """

    def __init__(self, n_basis_funcs: int, base_type: str, basis1=None, basis2=None, GB_limit: float =16.):

        if not base_type in ['evaluate', 'convolve', 'add', 'multiply']:
            raise ValueError(f"base_type must be 'evaluate', 'convolve', 'add', or 'multiply'. {base_type} provided instead.")

        self.n_basis_funcs = n_basis_funcs
        self.base_type = base_type
        self.GB_limit = GB_limit


        if base_type == 'evaluate':
            self._gen_basis = lambda x_list : self.evaluate(x_list[0])
            self.n_input_samples = 1

        if base_type == 'convolve':
            self._gen_basis = lambda x_list : self.convolve(x_list[0])
            self.n_input_samples = 1

        if base_type == 'add':
            self.n_input_samples = basis1.n_input_samples + basis2.n_input_samples
            self._gen_basis = lambda x_list: np.vstack((basis1._gen_basis(x_list[:basis1.n_input_samples]),
                                                        basis2._gen_basis(x_list[basis1.n_input_samples:])))

        if base_type == 'multiply':
            self.n_input_samples = basis1.n_input_samples + basis2.n_input_samples
            self._gen_basis = lambda x_list: rowWiseKron(basis1._gen_basis(x_list[:basis1.n_input_samples]),
                                                         basis2._gen_basis(x_list[basis1.n_input_samples:]), transpose=True)


        return


    def evaluate(self, samples, check_support=False):
        """
        Evaluate the basis functions at the given samples.
        (This is a placeholder, evaluate is basis type specific).

        Parameters
        ----------
        samples : numpy.ndarray or List[numpy.ndarray]
            The input samples.
        check_support : bool, optional
            Whether to perform support checks. Default is False.

        Returns
        -------
        numpy.ndarray
            The basis function values evaluated at the given samples.
        """
        # perform checks and evaluate basis
        return

    def convolve(self, samples):
        """
        Perform convolution of the basis functions with the given samples.
        (This is a placeholder, convolve is basis type specific).

        Parameters
        ----------
        samples : numpy.ndarray
            The input samples.

        Returns
        -------
        numpy.ndarray
            The convolution result of the basis functions with the given samples.
        """
        # Perform convolution and return the result
        return

    def gen_basis(self, *x: NDArray):
        """
        Generate the basis functions for the given input samples.

        Parameters
        ----------
        *x : numpy.ndarray
            The input samples.

        Returns
        -------
        numpy.ndarray
            The generated basis functions.
        """
        x = list(x)

        # checks on input and outputs
        self.check_samples_consistency(x)
        self.check_full_model_matrix_size(x[0].shape[0])
        self.check_input_number(x)

        return self._gen_basis(x)





    def check_input_number(self,x: list):
        """
        Check the consistency of sample sizes.

        Parameters
        ----------
        x : List[numpy.ndarray]
            The input samples.

        Raises
        ------
        ValueError
            If the input sample sizes are inconsistent.
        """
        if len(x) != self.n_input_samples:
            raise ValueError(f'input number mismatch. Basis requires {self.n_input_samples} input samples, {len(x)} inputs provided instead.')
    def check_samples_consistency(self, x: list):
        """
        Check the consistency of sample sizes.

        Parameters
        ----------
        x : list
           The input samples.

        Raises
        ------
        ValueError
           If the input sample sizes are inconsistent.
        """
        sample_sizes = [samp.shape[0] for samp in x]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(f'sample size mismatch. Input elements have inconsistent sample sizes.')
    def check_full_model_matrix_size(self, n_samples, dtype: type = np.float64):
        """
        Check the size of the full model matrix.

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
        size_in_bytes = np.dtype(dtype).itemsize * n_samples * self.n_basis_funcs
        if size_in_bytes > self.GB_limit * 10**9:
            raise MemoryError(f'Model matrix size exceeds {self.GB_limit} GB.')
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
        return Basis(self.n_basis_funcs + self.n_basis_funcs, base_type='add', basis1=self, basis2=other)

    def __mul__(self, other):
        """
        Multiply two Basis objects together. Returns a Basis of 'multiply' type, which can be used to model
        multi-dimensional response functions.

        Parameters
        ----------
        other : Basis
            The other Basis object to multiply.

        Returns
        -------
        Basis
            The resulting Basis object.

        """
        return Basis(self.n_basis_funcs * self.n_basis_funcs, base_type='multiply', basis1=self, basis2=other)



class SplineBasis(Basis):
    """
    SplineBasis class inherits from the Basis class and represents spline basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    base_type : str
        Type of basis functions. Must be one of ['evaluate', 'convolve', 'add', 'multiply'].
    basis1 : Basis, optional
        The first basis function object. Required if base_type is 'add' or 'multiply'.
    basis2 : Basis, optional
        The second basis function object. Required if base_type is 'add' or 'multiply'.
    order : int, optional
        Spline order. Default is 2.

    """
    def __init__(self, n_basis_funcs: int, base_type: str, basis1=None, basis2=None, order: int = 2):
        super().__init__(n_basis_funcs, base_type, basis1=basis1, basis2=basis2)
        self.order = order
        if self.order < 1:
            raise ValueError("Spline order must be positive!")

    def generate_knots(
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
        sample_pts : numpy.ndarray
            The sample points.
        perc_low : float
            The low percentile value.
        perc_high : float
            The high percentile value.
        is_cyclic : bool, optional
            Whether the spline is cyclic. Default is False.

        Returns
        -------
        numpy.ndarray
            The knot locations for the spline basis functions.

        Raises
        ------
        ValueError
            If the spline order is larger than the number of basis functions.
        AssertionError
            If the percentiles or order values are not within the valid range.

        """

        # Determine number of interior knots.
        num_interior_knots = self.n_basis_funcs - self.order
        if is_cyclic:
            num_interior_knots += self.order - 1

        # Check hyperparameters.
        if num_interior_knots < 0:
            raise ValueError(
                "Spline `order` parameter cannot be larger "
                "than `n_basis_funcs` parameter."
            )

        assert (perc_low >= 0) & (
            perc_high <= 1
        ), "Specify low and high percentile (perc_low, perc_high) as float between 0 and 1"
        assert perc_low < perc_high, "perc_low must be < perc_high. "

        # clip to avoid numerical errors in case of percentile numerical precision close to 0 and 1
        mn = np.nanpercentile(sample_pts, np.clip(perc_low * 100, 0, 100))
        mx = np.nanpercentile(sample_pts, np.clip(perc_high * 100, 0, 100))

        self.knot_locs = np.concatenate(
            (
                mn * np.ones(self.order - 1),
                np.linspace(mn, mx, num_interior_knots + 2),
                mx * np.ones(self.order - 1),
            )
        )

class BSplineBasis(SplineBasis):
    """
    B-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    base_type : str
        Type of basis functions. Must be one of ['evaluate', 'convolve', 'add', 'multiply'].
    basis1 : Basis, optional
        The first basis function object. Required if base_type is 'add' or 'multiply'.
    basis2 : Basis, optional
        The second basis function object. Required if base_type is 'add' or 'multiply'.
    order : int, optional
        Order of the splines used in basis functions. Must lie within [1, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.

    References
    ----------
    [2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
    Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5

    """

    def __init__(self, n_basis_funcs: int, base_type: str, basis1=None, basis2=None, order: int = 2):
        super().__init__(n_basis_funcs, base_type, basis1=basis1, basis2=basis2, order=order)

    def evaluate(
        self, sample_pts: NDArray, outer_ok: bool = False, der: int = 0
    ) -> NDArray:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts : NDArray
            The sample points at which the B-spline is evaluated.
        outer_ok : bool, optional
            If True, accepts samples outside the knots range. If False, raises a value error.
        der : int, optional
            Order of the derivative of the B-spline (default is 0, e.g., B-spline evaluation).

        Returns
        -------
        NDArray
            The evaluated basis functions.

        Raises
        ------
        AssertionError
            If the sample points are not within the B-spline knots range unless `outer_ok=True`.

        Notes
        -----
        This method evaluates the B-spline basis functions at the given sample points. It requires the knots to be defined,
        through the `generate_knots` method. Knots will be flushed at the end of the call.

        The evaluation is performed by looping over each element and using `splev` from SciPy to compute the basis values.

        """
        super().evaluate(sample_pts, check_support=False)
        # add knots
        self.generate_knots(sample_pts, 0.0, 1.0)

        # sort the knots in case user passed

        knots = self.knot_locs
        knots.sort()
        nk = knots.shape[0]

        # check for out of range points (in cyclic b-spline need_outer must be set to False)
        need_outer = any(sample_pts < knots[self.order - 1]) or any(
            sample_pts > knots[nk - self.order]
        )
        assert (
            not need_outer
        ) | outer_ok, 'sample points must lie within the B-spline knots range unless "outer_ok==True".'

        # select knots that are within the knots range (this takes care of eventual NaNs)
        in_sample = (sample_pts >= knots[0]) & (sample_pts <= knots[-1])

        if need_outer:
            reps = self.order - 1
            knots = np.hstack(
                (np.ones(reps) * knots[0], knots, np.ones(reps) * knots[-1])
            )
            nk = knots.shape[0]
        else:
            reps = 0

        # number of basis elements
        n_basis = nk - self.order

        # initialize the basis element container
        basis_eval = np.zeros((n_basis - 2 * reps, sample_pts.shape[0]))

        # loop one element at the time and evaluate the basis using splev
        id_basis = np.eye(n_basis, nk, dtype=np.int8)
        for i in range(reps, len(knots) - self.order - reps):
            basis_eval[i - reps, in_sample] = splev(
                sample_pts[in_sample], (knots, id_basis[i], self.order - 1), der=der
            )

        delattr(self, 'knot_locs')
        # # check sum equal 1 (B-spline are supposed to sum to 1)
        # assert(np.abs(basis_eval.sum(axis=0) - 1).max() < 1e-6)
        return basis_eval

class Cyclic_BSplineBasis(BSplineBasis):
    """
    Generate basis functions with given spacing, calls scipy.interpolate.splev which is a wrapper to fortran.
    Comes with the additional bonus of evaluating the derivatives of b-spline, needed for smoothing penalization.

    Parameters
    ----------
    sample_pts : ndarray
        Spacing for basis functions, holding elements on the interval [0,
        window_size). A good default is np.arange(window_size).
    der : int, optional
        Order of the derivative of the B-spline (default is 0, e.g. Bspline eval).

    Returns
    -------
    basis_eval : ndarray
        Basis function evaluation results.
    """

    def __init__(self, n_basis_funcs: int, base_type: str, basis1=None, basis2=None, order: int = 2):
        super().__init__(n_basis_funcs, base_type, basis1=basis1, basis2=basis2, order=order)
        assert (
            self.order >= 2
        ), f"Order >= 2 required for cyclic B-spline, order {self.order} specified instead!"
        assert (
            self.n_basis_funcs >= order + 2
        ), "n_basis_funcs >= order + 2 required for cyclic B-spline"
        assert (
            self.n_basis_funcs >= 2 * order - 2
        ), "n_basis_funcs >= 2*(order - 1) required for cyclic B-spline"

    def evaluate(self, sample_pts: NDArray, der: int = 0) -> NDArray:
        """
        Generate basis functions with given spacing, calls scipy.interpolate.splev which is a wrapper to fortran.
        Comes with the additional bonus of evaluating the derivatives of b-spline, needed for smoothing penalization.
        Parameters
        ----------
        sample_pts: (n_pts,)
            Spacing for basis functions, holding elements on the interval [0,
            window_size). A good default is np.arange(window_size).

        outer_ok: bool
            if True, accepts samples outside knots range
            if False, raise value error

        der: int
            order of the derivative of the B-spline (default is 0, e.g. Bspline eval).

        Returns
        -------

        """

        # add knots if not passed
        if not hasattr(self, "knot_locs"):
            self.generate_knots(sample_pts, 0.0, 1.0, is_cyclic=True)

        # for cyclic, do not repeat knots
        self.knot_locs = np.unique(self.knot_locs)

        knots_orig = self.knot_locs.copy()

        nk = knots_orig.shape[0]

        # make sure knots are sorted
        knots_orig.sort()
        xc = knots_orig[nk - 2 * self.order + 1]
        knots = np.hstack(
            (
                self.knot_locs[0]
                - self.knot_locs[-1]
                + self.knot_locs[nk - self.order : nk - 1],
                self.knot_locs,
            )
        )
        ind = sample_pts > xc

        # temporarily set the extended knots as attribute
        self.knot_locs = knots
        basis_eval = super().evaluate(sample_pts, outer_ok=True, der=der)
        sample_pts[ind] = sample_pts[ind] - knots.max() + knots_orig[0]
        if np.sum(ind):
            X2 = super().evaluate(sample_pts[ind], outer_ok=True, der=der)
            basis_eval[:, ind] = basis_eval[:, ind] + X2
        # restore points
        sample_pts[ind] = sample_pts[ind] + knots.max() - knots_orig[0]
        # restore the original knots
        self.knot_loc = knots_orig
        return basis_eval

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator



    samples = np.random.normal(size=100)
    basis1 = BSplineBasis(15, 'evaluate', basis1=None, basis2=None, order=4)
    basis2 = BSplineBasis(15, 'evaluate', basis1=None, basis2=None, order=4)
    basis_add = basis1 + basis2
    basis_add_add = basis_add + basis2
    basis_add_add_add = basis_add_add + basis_add



    print(basis_add.gen_basis(samples, samples).shape)
    print(basis_add_add.gen_basis(samples, samples, samples).shape)
    print(basis_add_add_add.gen_basis(samples, samples, samples, samples, samples).shape)


    basis1 = BSplineBasis(15, 'evaluate', basis1=None, basis2=None, order=4)
    basis2 = BSplineBasis(15, 'evaluate', basis1=None, basis2=None, order=4)
    mulbase = basis1 * basis2
    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z = mulbase.gen_basis(X.flatten(), Y.flatten())
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Z = np.array(Z)
    Z[Z==0] = np.nan
    ax.plot_surface(X, Y, Z[50].reshape(X.shape), cmap='viridis', alpha=0.8)
    ax.plot_surface(X, Y, Z[100].reshape(X.shape), cmap='rainbow', alpha=0.8)
    ax.plot_surface(X, Y, Z[200].reshape(X.shape), cmap='inferno', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Overlapped Surfaces')

    print('multiply and additive base with a evaluate type base')
    basis1 = BSplineBasis(6, 'evaluate', basis1=None, basis2=None, order=4)
    basis2 = BSplineBasis(7, 'evaluate', basis1=None, basis2=None, order=4)
    basis3 = Cyclic_BSplineBasis(8, 'evaluate', basis1=None, basis2=None, order=4)
    base_res = (basis1 + basis2) * basis3
    X = base_res.gen_basis(np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    print(X.shape, (6+7) * 8)
