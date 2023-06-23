"""Bases classes.
"""
# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations

import abc
import warnings
from typing import Tuple, Any

import numpy as np
from numpy.typing import NDArray

from neurostatslib.utils import rowWiseKron


class Basis(abc.ABC):
    """
    Abstract class for basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    GB_limit : optional
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

    def __init__(self, n_basis_funcs: int, GB_limit: float = 16.0) -> None:
        self._n_basis_funcs = n_basis_funcs
        self._GB_limit = GB_limit
        self._n_input_samples = 0

    @abc.abstractmethod
    def _evaluate(self, *x: NDArray) -> NDArray:
        """
        Evaluate the basis set at the given samples x1,...,xn using the subclass-specific "_evaluate" method.

        Parameters
        ----------
        x[0],...,x[n] : (number of samples, )
            The input samples.
        """
        pass

    @abc.abstractmethod
    def _get_samples(self, *n_samples: int) -> Tuple[NDArray, ...]:
        """
        Get equi-spaced samples for all the input dimensions. This will be used to evaluate
        the basis on a grid of points derived by the samples

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
            The equi-spaced samples covering the basis domain.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the _gen_basis method.
        """
        pass

    def evaluate(self, *xi: NDArray) -> NDArray:
        """
        Evaluate the basis set at the given samples x[0],...,x[n] using the subclass-specific "_evaluate" method.

        Parameters
        ----------
        xi[0],...,xi[n] : (number of samples, )
            The input samples.

        Returns
        -------
            The generated basis functions.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the _evaluate method.
        """
        # checks on input and outputs
        self._check_samples_consistency(*xi)
        self._check_full_model_matrix_size(xi[0].shape[0])
        self._check_input_number(xi)

        eval_basis = self._evaluate(*xi)

        # checks on the evaluated basis
        self._check_enough_samples(eval_basis) # move to the GLM model

        # check the conditioning
        conditioning = np.linalg.cond(eval_basis) # model should do that
        if np.isinf(conditioning):
            if any(eval_basis.sum(axis=1) == 0):
                warnings.warn("eval_basis has an empty row. No samples in the input domain "
                              "of at least one basis function. Try to reduce the number of basis or increase the sample"
                              "size")
            else:
                warnings.warn("Linearly dependent columns in eval_basis. Check for perfect collinearity in the inputs"
                              "or insufficient sample size.")

        print(f'Conditioning of the evaluated basis function: {conditioning}')
        return eval_basis

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[NDArray, ...]:
        """
        Evaluate the basis set on a grid of equi-spaced sample points. The i-th axis of the grid will be sampled
        with n_samples[i] equi-spaced points.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
            Xs[1], ..., Xs[n] : (n_samples[0], ... , n_samples[n])
            A tuple containing the meshgrid values, one element for each of the n dimension of the grid, where n equals
            to the number of inputs.
            Y : (number of basis, n_samples[0], ... , n_samples[n])
            the basis function evaluated at the samples.

        """
        self._check_input_number(n_samples)
        self._check_full_model_matrix_size(np.prod(n_samples) * self._n_basis_funcs)

        # get the samples
        sample_tuple = self._get_samples(n_samples)
        Xs = np.meshgrid(*sample_tuple, indexing='ij')

        # call evaluate to evaluate the basis on a flat NDArray and reshape to match meshgrid output
        Y = self.evaluate(*tuple(grid_axis.flatten() for grid_axis in Xs)).reshape(
            (self._n_basis_funcs,) + n_samples
        )

        return *Xs, Y

    def _check_enough_samples(self, eval_basis: NDArray) -> None:
        """
        Checks if there are enough samples for evaluation.

        Parameters
        ----------
        eval_basis : (number of basis, number of samples)
            The basis evaluated at the samples

        Raises
        ------
        UserWarning
            If the number of basis sets exceeds the number of samples.
        """
        if eval_basis.shape[0] > eval_basis.shape[1]:
            warnings.warn('Basis set number exceeds the number of samples! '
                          'Reduce the number of basis or increase sample size')

    def _check_input_number(self, xi: Tuple) -> None:
        """
        Check that the number of inputs provided by the user matches the number of inputs that the Basis object requires.

        Parameters
        ----------
        xi[0], ..., xi[n] : (number of samples, )
            The input samples.

        Raises
        ------
        ValueError
            If the number of inputs doesn't match what the Basis object requires.
        """
        if len(xi) != self._n_input_samples:
            raise ValueError(
                f"Input number mismatch. Basis requires {self._n_input_samples} input samples, {len(xi)} inputs provided instead."
            )

    def _check_samples_consistency(self, *xi: NDArray) -> None:
        """
        Check that each input provided to the Basis object has the same number of time points.

        Parameters
        ----------
        xi[0], ..., xi[n] : (number of samples, )
            The input samples.

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs.
        """
        sample_sizes = [samp.shape[0] for samp in xi]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(
                "Sample size mismatch. Input elements have inconsistent sample sizes."
            )

    def _check_full_model_matrix_size(self, n_samples: int, dtype: type = np.float64) -> None:
        """
        Check the size in GB of the full model matrix is <= self._GB_limit.

        Parameters
        ----------
        n_samples
            Number of samples.
        dtype : optional
            Data type of the model matrix. Default is np.float64.

        Raises
        ------
        MemoryError
            If the size of the model matrix exceeds the specified memory limit.
        """
        size_in_bytes = np.dtype(dtype).itemsize * n_samples * self._n_basis_funcs
        if size_in_bytes > self._GB_limit * 10 ** 9:
            raise MemoryError(f"Model matrix size exceeds {self._GB_limit} GB.")

    def __add__(self, other: Basis) -> Basis:
        """
        Add two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to add.

        Returns
        -------
        The resulting Basis object.
        """
        return addBasis(self, other)

    def __mul__(self, other) -> Basis:
        """
        Multiply two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to multiply.

        Returns
        -------
        The resulting Basis object.
        """
        return mulBasis(self, other)


class addBasis(Basis, abc.ABC):
    """
    Class representing the addition of two Basis objects.

    Parameters
    ----------
    basis1 :
        First basis object to add.
    basis2 :
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

    def __init__(self, basis1, basis2) -> None:
        self._n_basis_funcs = basis1._n_basis_funcs + basis2._n_basis_funcs
        super().__init__(self._n_basis_funcs, GB_limit=basis1._GB_limit)
        self._n_input_samples = basis1._n_input_samples + basis2._n_input_samples
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _evaluate(self, *xi: NDArray) -> NDArray:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (number of samples, )
            Tuple of input samples.

        Returns
        -------
            The basis function evaluated at the samples (number of samples x number of basis)
        """
        return np.vstack(
            (
                self._basis1._evaluate(*xi[: self._basis1._n_input_samples]),
                self._basis2._evaluate(*xi[self._basis1._n_input_samples:]),
            )
        )

    def _get_samples(self, *n_samples: int) -> Tuple[NDArray, ...]:
        """
        Get equi-spaced samples for all the input dimensions. This will be used to evaluate
        the basis on a grid of points derived by the samples

        Parameters
        ----------
        n_samples[0],...,n_samples[n] : int
            The number of samples in each axis of the grid.

        Returns
            The equi-spaced sample locations for each coordinate.

        """
        sample_1 = self._basis1._get_samples(n_samples[: self._basis1._n_input_samples])
        sample_2 = self._basis2._get_samples(n_samples[self._basis1._n_input_samples:])
        return *sample_1,  *sample_2


class mulBasis(Basis,abc.ABC):
    """
    Class representing the multiplication (external product) of two Basis objects.

    Parameters
    ----------
    basis1 :
        First basis object to multiply.
    basis2 :
        Second basis object to multiply.

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
        Evaluates the basis function at the samples x_tuple[0],..,x_tuple[n]
    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self._n_basis_funcs = basis1._n_basis_funcs * basis2._n_basis_funcs
        super().__init__(self._n_basis_funcs, GB_limit=basis1._GB_limit)
        self._n_input_samples = basis1._n_input_samples + basis2._n_input_samples
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _evaluate(self, *xi: NDArray) -> NDArray: # try * all _evaluate
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (number of samples, )
            Tuple of input samples.

        Returns
        -------
            The basis function evaluated at the samples (number of samples x number of basis)
        """
        return np.array(rowWiseKron(
            self._basis1._evaluate(*xi[: self._basis1._n_input_samples]),
            self._basis2._evaluate(*xi[self._basis1._n_input_samples:]),
            transpose=True,
        ))

    def _get_samples(self, *n_samples: int) -> Tuple[NDArray, ...]:
        """
        Get equi-spaced samples for all the input dimensions. This will be used to evaluate
        the basis on a grid of points derived by the samples

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
            The equi-spaced sample locations for each coordinate.

        """

        sample_1 = self._basis1._get_samples(n_samples[: self._basis1._n_input_samples])
        sample_2 = self._basis2._get_samples(n_samples[self._basis1._n_input_samples:])
        return *sample_1, *sample_2


class SplineBasis(Basis, abc.ABC):
    """
    SplineBasis class inherits from the Basis class and represents spline basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    order : optional
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

    def __init__(self, n_basis_funcs: int, order: int = 2) -> None:
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
        sample_pts : (number of samples, )
            The sample points.
        perc_low
            The low percentile value.
        perc_high
            The high percentile value.
        is_cyclic : optional
            Whether the spline is cyclic. Default is False.

        Returns
        -------
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

    def _get_samples(self, n_samples: int) -> NDArray:
        """
        Generate the basis functions on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples
           The number of samples in each axis of the grid.

        Returns
        -------
           The equi-spaced sample location.
        """
        return np.linspace(0, 1, n_samples[0])


class MSplineBasis(SplineBasis):
    """M-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    window_size :
        Size of basis functions.
    order :
        Order of the splines used in basis functions. Must lie within [1,
        n_basis_funcs]. The m-splines have ``order-2`` continuous derivatives
        at each interior knot. The higher this number, the smoother the basis
        representation will be.


    References
    ----------
    .. [1] Ramsay, J. O. (1988). Monotone regression splines in action.
       Statistical science, 3(4), 425-441.

    """

    def __init__(self, n_basis_funcs: int, order: int = 2) -> None:
        super().__init__(n_basis_funcs, order)

    def _evaluate(self, sample_pts: NDArray) -> NDArray:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts : (number of samples, )
            Spacing for basis functions, holding elements on the interval [0,
            window_size). A good default is np.arange(window_size).

        Returns
        -------
        basis_funcs : (number of basis, number of samples)
            Evaluated spline basis functions.

        """

        #sample_pts = sample_pts[0]

        # add knots if not passed
        self._generate_knots(sample_pts, 0.0, 1.0, is_cyclic=True)

        return np.stack(
            [mspline(sample_pts, self._order, i, self.knot_locs) for i in range(self._n_basis_funcs)],
            axis=0
        )


class RaisedCosineBasis(Basis, abc.ABC):
    def __init__(self, n_basis_funcs: int) -> None:
        super().__init__(n_basis_funcs)
        self._n_input_samples = 1

    @abc.abstractmethod
    def _transform_samples(self, sample_pts: NDArray) -> NDArray:
        """
        Abstract method for transforming sample points.

        Parameters
        ----------
        sample_pts : (number of samples, )
           The sample points to be transformed.
        """
        pass

    def _evaluate(self, sample_pts: NDArray) -> NDArray:
        """Generate basis functions with given samples.

        Parameters
        ----------
        sample_pts : (number of samples,)
            Spacing for basis functions, holding elements on interval [0,
            1). A good default is
            ``nsl.sample_points.raised_cosine_log`` for log spacing (as used in
            [2]_) or ``nsl.sample_points.raised_cosine_linear`` for linear
            spacing.

        Returns
        -------
        basis_funcs : (number of basis, number of samples)
            Raised cosine basis functions

        """
        #sample_pts = sample_pts[0]

        if any(sample_pts < -1E-12) or any(sample_pts > 1 + 1E-12): # check for a better way to control precision
            raise ValueError(f"Sample points for RaisedCosine basis must lie in [0,1]!")

        # transform to the proper domain
        sample_pts = self._transform_samples(sample_pts)

        shifted_sample_pts = sample_pts[None, :] - (np.pi * np.arange(self._n_basis_funcs))[:, None]
        basis_funcs = .5 * (np.cos(np.clip(shifted_sample_pts, -np.pi, np.pi)) + 1)

        return basis_funcs

    def _get_samples(self, n_samples: int) -> NDArray:
        """
        Generate an array equi-spaced sample points.

        Parameters
        ----------
        n_samples
           The number of samples in each axis of the grid.

        Returns
        -------
           The equi-spaced sample location.
        """
        return np.linspace(0, 1, n_samples[0])


class RaisedCosineBasisLinear(RaisedCosineBasis):
    """Linearly-spaced raised cosine basis functions used by Pillow et al. [2]_.

    These are "cosine bumps" that uniformly tile the space.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.

    References
    ----------
    .. [2] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013. http://dx.doi.org/10.1523/jneurosci.3305-05.2005

    """

    def __init__(self, n_basis_funcs: int) -> None:
        super().__init__(n_basis_funcs)

    def _transform_samples(self, sample_pts: NDArray) -> NDArray:
        """
        Linearly map the samples from [0,1] to the the [0, (number of basis - 1) * pi]
        Parameters
        ----------
        sample_pts : (number of samples, )
            The sample points used for evaluating the splines

        Returns
        -------
        NDArray : (number of samples, )
            A transformed version of the sample points that matches the Raised Cosine basis domain.
        """
        return sample_pts * np.pi * (self._n_basis_funcs - 1)


class RaisedCosineBasisLog(RaisedCosineBasis):
    """Log-spaced raised cosine basis functions used by Pillow et al. [2]_.
    These are "cosine bumps" that uniformly tile the space.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.

    References
    ----------
    .. [2] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013. http://dx.doi.org/10.1523/jneurosci.3305-05.2005

    """

    def __init__(self, n_basis_funcs: int) -> None:
        super().__init__(n_basis_funcs)

    def _transform_samples(self, sample_pts: NDArray) -> NDArray:
        """
        Maps the equi-spaced samples from [0,1] to log equi-spaced samples [0, (number of basis - 1) * pi]

        Parameters
        ----------
        sample_pts : (number of samples, )
            The sample points used for evaluating the splines

        Returns
        -------
        NDArray : (number of samples, )
            A transformed version of the sample points that matches the Raised Cosine basis domain.
        """
        return np.power(10, -(np.log10((self._n_basis_funcs - 1) * np.pi) + 1) * sample_pts +
                        np.log10((self._n_basis_funcs - 1) * np.pi)) - 0.1


def mspline(x: NDArray, k: int, i: int, T: NDArray) -> NDArray:
    """Compute M-spline basis function.

    Parameters
    ----------
    x : (number of samples, )
        Spacing for basis functions, holding elements on the interval [0,
        window_size). If None, use a grid (``np.arange(self.window_size)``).
    k
        Order of the spline basis.
    i
        Number of the spline basis.
    T : (k + number of basis,)
        knot locations. should lie in interval [0, 1].


    Returns
    -------
    spline : (number of samples, )
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

    plt.close('all')
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    samples = np.random.normal(size=100)
    basis1 = MSplineBasis(15, order=4)
    basis2 = MSplineBasis(15, order=4)
    basis_add = basis1 + basis2

    basis_add_add = basis_add + basis2

    basis_add_add_add = basis_add_add + basis_add

    print(basis_add.evaluate(samples, samples).shape)
    print(basis_add_add.evaluate(samples, samples, samples).shape)
    print(basis_add_add_add.evaluate(samples, samples, samples, samples, samples).shape)

    basis1 = RaisedCosineBasisLog(15)
    basis2 = MSplineBasis(15, order=4)
    mulbase = basis1 * basis2
    X, Y, Z = mulbase.evaluate_on_grid(100, 110) # maybe create a visualize basis class.

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Z = np.array(Z)
    Z[Z == 0] = np.nan
    ax.plot_surface(X, Y, Z[50], cmap="viridis", alpha=0.8)
    ax.plot_surface(X, Y, Z[100], cmap="rainbow", alpha=0.8)
    ax.plot_surface(X, Y, Z[200], cmap="inferno", alpha=0.8)
    #
    # # Customize the plot
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Overlapped Surfaces")
    #
    # print("multiply and additive base with a evaluate type base")
    # basis1 = LogRaisedCosineBasis(6)
    # basis2 = MSplineBasis(7, order=4)
    # basis3 = LinearRaisedCosineBasis(8)
    # base_res = (basis1 + basis2) * basis3
    # X = base_res.evaluate(
    #     np.linspace(0, 1, 64), np.linspace(0, 1, 64), np.linspace(0, 1, 64)
    # )
    # print(X.shape, (6 + 7) * 8)

    #
    #
    # basis1 = MSplineBasis(6, order=4)
    # basis2 = MSplineBasis(7, order=4)
    # basis3 = MSplineBasis(8, order=4)
    #
    # multb = basis1 + basis2 * basis3
    # X, Y, W, Z = multb.gen_basis(10, 11, 12)
