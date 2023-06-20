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
    n_basis_funcs : int
        Number of basis functions.
    GB_limit : float
        Limit in GB for the model matrix size.

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

    def __init__(self, n_basis_funcs: int, GB_limit: float =16.):


        self._n_basis_funcs = n_basis_funcs
        self.GB_limit = GB_limit

        return

    def _evaluate(self, x: tuple[NDArray]):
        subclass_name = type(self).__name__
        raise NotImplementedError(f"{subclass_name} must implement _evaluate method!")

    def evaluate(self, *xi: NDArray):
        """
        This method evaluate the basis set at the given samples x1,...,xn using the subclass-specific "_evaluate" method.
        The specific functionality of _evaluate should be implemented in the subclasses.

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



    def _check_input_number(self,x: tuple[NDArray]):
        """
        Check that the number of inputs provided by the user matches the number of inputs that the Basis object requires.

        Parameters
        ----------
        x : List[numpy.ndarray]
            The input samples.

        Raises
        ------
        ValueError
            If the number of inputs doesn't match what the Basis object requires.
        """
        if len(x) != self._n_input_samples:
            raise ValueError(f'input number mismatch. Basis requires {self._n_input_samples} input samples, {len(x)} inputs provided instead.')

    def _check_samples_consistency(self, x: tuple[NDArray]):
        """
        Check that each input provided to the Basis object has the same number of time points.

        Parameters
        ----------
        x : list
           The input samples.

        Raises
        ------
        ValueError
           If the time point number is inconsistent between inputs.
        """
        sample_sizes = [samp.shape[0] for samp in x]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(f'sample size mismatch. Input elements have inconsistent sample sizes.')

    def _check_full_model_matrix_size(self, n_samples, dtype: type = np.float64):
        """
        Check the size in GB of the full model matrix is <= self.GB_limit

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
        Generate the model matrix using provided input samples.

    """
    def __init__(self, basis1, basis2):
        """
        Initialize the addBasis object.

        Parameters
        ----------
        basis1 : Basis
            First basis object to add.
        basis2 : Basis
            Second basis object to add.
        """
        self._n_basis_funcs = basis1._n_basis_funcs + basis2._n_basis_funcs
        super().__init__(self._n_basis_funcs, GB_limit=basis1.GB_limit)
        self._n_input_samples = basis1._n_input_samples + basis2._n_input_samples
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _evaluate(self, x_tuple: tuple[NDArray]):
        """
        Generate the model matrix using provided input samples.

        Parameters
        ----------
        x_tuple : tuple
            List of input samples.

        Returns
        -------
        numpy.ndarray
            The generated model matrix.
        """
        return np.vstack((self._basis1._evaluate(x_tuple[:self._basis1._n_input_samples]),
                   self._basis2._evaluate(x_tuple[self._basis1._n_input_samples:])))

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
        Generate the model matrix using provided input samples.
    """
    def __init__(self, basis1, basis2):
        """
        Initialize the mulBasis object.

        Parameters
        ----------
        basis1 : Basis
            First basis object to multiply.
        basis2 : Basis
            Second basis object to multiply.
        """
        self._n_basis_funcs = basis1._n_basis_funcs * basis2._n_basis_funcs
        super().__init__(self._n_basis_funcs, GB_limit=basis1.GB_limit)
        self._n_input_samples = basis1._n_input_samples + basis2._n_input_samples
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _evaluate(self, x_tuple: tuple[NDArray]):
        """
        Generate the model matrix given the provided input samples.

        Parameters
        ----------
        x_tuple : tuple
            List of input samples.

        Returns
        -------
        numpy.ndarray
            The generated model matrix.
        """
        return rowWiseKron(self._basis1._evaluate(x_tuple[:self._basis1._n_input_samples]),
                           self._basis2._evaluate(x_tuple[self._basis1._n_input_samples:]), transpose=True)

class SplineBasis(Basis):
    """
    SplineBasis class inherits from the Basis class and represents spline basis functions.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    order : int, optional
        Spline order. Default is 2.
    eval_type : str, optional
        Type of basis functions. Must be one of ['evaluate', 'convolve']. Default is 'evaluate'.

    Attributes
    ----------
    order : int
        Spline order.
    n_input_samples : int
        Number of input samples.

    Methods
    -------
    _generate_knots(sample_pts, perc_low, perc_high, is_cyclic=False)
        Generate knot locations for spline basis functions.

    """

    def __init__(self, n_basis_funcs: int, order: int = 2, eval_type: str = 'evaluate'):
        """
        Initialize the SplineBasis object.

        Parameters
        ----------
        n_basis_funcs : int
            Number of basis functions.
        order : int, optional
            Spline order. Default is 2.
        eval_type : str, optional
            Type of basis functions. Must be one of ['evaluate', 'convolve']. Default is 'evaluate'.
        """
        super().__init__(n_basis_funcs)
        self._order = order
        self._n_input_samples = 1
        if self._order < 1:
            raise ValueError("Spline order must be positive!")
        if eval_type not in ['evaluate', 'convolve']:
            raise ValueError(
                f"eval_type must be 'evaluate' or 'convolve'. '{eval_type}' provided instead."
            )

    def _generate_knots(
        self,
        sample_pts: np.ndarray,
        perc_low: float,
        perc_high: float,
        is_cyclic: bool = False,
    ) -> np.ndarray:
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
    eval_type : str, optional
        Type of basis functions. Must be one of ['evaluate', 'convolve']. Default is 'evaluate'.

    References
    ----------
    [2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
    Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5

    """

    def __init__(self, n_basis_funcs: int, order: int = 2, eval_type: str = 'evaluate'):
        """
        Initialize the BSplineBasis object.

        Parameters
        ----------
        n_basis_funcs : int
            Number of basis functions.
        order : int, optional
            Order of the splines used in basis functions. Must lie within [1, n_basis_funcs].
            The B-splines have (order-2) continuous derivatives at each interior knot.
            The higher this number, the smoother the basis representation will be.
        eval_type : str, optional
            Type of basis functions. Must be one of ['evaluate', 'convolve']. Default is 'evaluate'.
        """
        super().__init__(n_basis_funcs, order=order, eval_type=eval_type)

    def _evaluate(
        self, sample_pts: tuple[NDArray], outer_ok: bool = False, der: int = 0
    ) -> np.ndarray:
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
            The evaluated basis functions.

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

        delattr(self, 'knot_locs')
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

    """

    def __init__(self, n_basis_funcs: int, order: int = 2, eval_type: str = 'evaluate'):

        super().__init__(n_basis_funcs, order=order, eval_type=eval_type)
        assert (
            self._order >= 2
        ), f"Order >= 2 required for cyclic B-spline, order {self._order} specified instead!"
        assert (
            self._n_basis_funcs >= order + 2
        ), "n_basis_funcs >= order + 2 required for cyclic B-spline"
        assert (
            self._n_basis_funcs >= 2 * order - 2
        ), "n_basis_funcs >= 2*(order - 1) required for cyclic B-spline"

    def _evaluate(self, sample_pts: tuple[NDArray], der: int = 0) -> np.ndarray:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts : tuple
            The sample points at which the B-spline is evaluated.
        der : int, optional
            Order of the derivative of the B-spline (default is 0, e.g., B-spline evaluation).

        Returns
        -------
        np.ndarray
            The evaluated basis functions.

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator



    samples = np.random.normal(size=100)
    basis1 = BSplineBasis(15, eval_type='evaluate', order=4)
    basis2 = BSplineBasis(15, eval_type='evaluate', order=4)
    basis_add = basis1 + basis2
    basis_add_add = basis_add + basis2
    basis_add_add_add = basis_add_add + basis_add



    print(basis_add.evaluate(samples, samples).shape)
    print(basis_add_add.evaluate(samples, samples, samples).shape)
    print(basis_add_add_add.evaluate(samples, samples, samples, samples, samples).shape)


    basis1 = BSplineBasis(15, eval_type='evaluate', order=4)
    basis2 = BSplineBasis(15, eval_type='evaluate', order=4)
    mulbase = basis1 * basis2
    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z = mulbase.evaluate(X.flatten(), Y.flatten())
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
    basis1 = BSplineBasis(6, eval_type='evaluate', order=4)
    basis2 = BSplineBasis(7, eval_type='evaluate', order=4)
    basis3 = Cyclic_BSplineBasis(8, eval_type='evaluate', order=4)
    base_res = (basis1 + basis2) * basis3
    X = base_res.evaluate(np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    print(X.shape, (6+7) * 8)
