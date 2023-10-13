"""Bases classes."""

# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations

import abc
from typing import Generator, Tuple

import jax.numpy
import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import splev

from neurostatslib.utils import row_wise_kron

__all__ = [
    "MSplineBasis",
    "BSplineBasis",
    "CyclicBSplineBasis",
    "RaisedCosineBasisLinear",
    "RaisedCosineBasisLog",
    "OrthExponentialBasis",
    "AdditiveBasis",
    "MultiplicativeBasis",
    "FourierBasis",
]


def __dir__() -> list[str]:
    return __all__


class Basis(abc.ABC):
    """
    Abstract class for basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.

    """

    def __init__(self, n_basis_funcs: int) -> None:
        self.n_basis_funcs = n_basis_funcs
        self._n_input_dimensionality = 0
        self._check_n_basis_min()

    @abc.abstractmethod
    def _evaluate(self, *xi: NDArray) -> NDArray:
        """
        Evaluate the basis set at the given samples x1,...,xn using the subclass-specific "_evaluate" method.

        Parameters
        ----------
        *xi: (number of samples, )
            The input samples xi[0],...,xi[n] .
        """
        pass

    @staticmethod
    def _get_samples(*n_samples: int) -> Generator[NDArray, None, None]:
        """Get equi-spaced samples for all the input dimensions.

        This will be used to evaluate the basis on a grid of
        points derived by the samples.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
        :
            A generator yielding numpy arrays of linspaces from 0 to 1 of sizes specified by `n_samples`.
        """
        return (np.linspace(0, 1, n_samples[k]) for k in range(len(n_samples)))

    def evaluate(self, *xi: ArrayLike) -> NDArray:
        """
        Evaluate the basis set at the given samples x[0],...,x[n] using the subclass-specific "_evaluate" method.

        Parameters
        ----------
        xi[0],...,xi[n] :
            The input samples, each  with shape (number of samples, ).

        Returns
        -------
        :
            The generated basis functions.

        Raises
        ------
        ValueError
            - If the time point number is inconsistent between inputs.
            - If the number of inputs doesn't match what the Basis object requires.
            - At least one of the samples is empty.
        """
        # check that the input is array-like
        if any(
            not isinstance(x, (list, tuple, np.ndarray, jax.numpy.ndarray)) for x in xi
        ):
            raise TypeError("Input samples must be array-like!")

        # convert to numpy.array of floats
        xi = tuple(np.asarray(x, dtype=float) for x in xi)

        # check for non-empty samples
        if self._has_zero_samples(tuple(len(x) for x in xi)):
            raise ValueError("All sample provided must be non empty.")

        # checks on input and outputs
        self._check_samples_consistency(*xi)
        self._check_input_dimensionality(xi)

        eval_basis = self._evaluate(*xi)

        return eval_basis

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        The i-th axis of the grid will be sampled with n_samples[i] equi-spaced points.
        The method uses numpy.meshgrid with `indexing="ij"`, returning matrix indexing
        instead of the default cartesian indexing, see Notes.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
        *Xs :
            A tuple of arrays containing the meshgrid values, one element for each of the n dimension of the grid,
            where n equals to the number of inputs.
            The size of Xs[i] is (n_samples[0], ... , n_samples[n]).
        Y :
            The basis function evaluated at the samples,
            shape (n_samples[0], ... , n_samples[n], number of basis).

        Raises
        ------
        ValueError
            - If the time point number is inconsistent between inputs or if the number of inputs doesn't match what
            the Basis object requires.
            - If one of the n_samples is <= 0.

        Notes
        -----
        Setting "indexing = 'ij'" returns a meshgrid with matrix indexing. In the N-D case with inputs of size
        $M_1,...,M_N$, outputs are of shape $(M_1, M_2, M_3, ....,M_N)$.
        This differs from the numpy.meshgrid default, which uses Cartesian indexing.
        For the same input, Cartesian indexing would return an output of shape $(M_2, M_1, M_3, ....,M_N)$.

        """
        self._check_input_dimensionality(n_samples)

        if self._has_zero_samples(n_samples):
            raise ValueError("All sample counts provided must be greater than zero.")

        # get the samples
        sample_tuple = self._get_samples(*n_samples)
        Xs = np.meshgrid(*sample_tuple, indexing="ij")

        # call evaluate to evaluate the basis on a flat NDArray and reshape to match meshgrid output
        Y = self.evaluate(*tuple(grid_axis.flatten() for grid_axis in Xs)).reshape(
            (*n_samples, self.n_basis_funcs)
        )

        return *Xs, Y

    @staticmethod
    def _has_zero_samples(n_samples: Tuple[int]) -> bool:
        return any([n <= 0 for n in n_samples])

    def _check_input_dimensionality(self, xi: Tuple) -> None:
        """
        Check that the number of inputs provided by the user matches the number of inputs required.

        Parameters
        ----------
        xi[0], ..., xi[n] :
            The input samples, shape (number of samples, ).

        Raises
        ------
        ValueError
            If the number of inputs doesn't match what the Basis object requires.
        """
        if len(xi) != self._n_input_dimensionality:
            raise ValueError(
                f"Input dimensionality mismatch. This basis evaluation requires {self._n_input_dimensionality} inputs, "
                f"{len(xi)} inputs provided instead."
            )

    @staticmethod
    def _check_samples_consistency(*xi: NDArray) -> None:
        """
        Check that each input provided to the Basis object has the same number of time points.

        Parameters
        ----------
        xi[0], ..., xi[n] :
            The input samples, shape (number of samples, ).

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs.
        """
        sample_sizes = [sample.shape[0] for sample in xi]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(
                "Sample size mismatch. Input elements have inconsistent sample sizes."
            )

    @abc.abstractmethod
    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Most of the basis work with at least 1 element, but some
        such as the RaisedCosineBasisLog requires a minimum of 2 basis to be well defined.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        pass

    def __add__(self, other: Basis) -> AdditiveBasis:
        """
        Add two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to add.

        Returns
        -------
        : AdditiveBasis
            The resulting Basis object.
        """
        return AdditiveBasis(self, other)

    def __mul__(self, other: Basis) -> MultiplicativeBasis:
        """
        Multiply two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to multiply.

        Returns
        -------
        :
            The resulting Basis object.
        """
        return MultiplicativeBasis(self, other)

    def __pow__(self, exponent: int) -> MultiplicativeBasis:
        """Exponentiation of a Basis object.

        Define the power of a basis by repeatedly applying the method __multiply__.
        The exponent must be a positive integer.

        Parameters
        ----------
        exponent :
            Positive integer exponent

        Returns
        -------
        :
            The product of the basis with itself "exponent" times. Equivalent to self * self * ... * self.

        Raises
        ------
        TypeError
            If the provided exponent is not an integer.
        ValueError
            If the integer is zero or negative.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent should be an integer!")

        if exponent <= 0:
            raise ValueError("Exponent should be a non-negative integer!")

        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result


class AdditiveBasis(Basis):
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
    n_basis_funcs : int
        Number of basis functions.


    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self.n_basis_funcs = basis1.n_basis_funcs + basis2.n_basis_funcs
        super().__init__(self.n_basis_funcs)
        self._n_input_dimensionality = (
            basis1._n_input_dimensionality + basis2._n_input_dimensionality
        )
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _check_n_basis_min(self) -> None:
        pass

    def _evaluate(self, *xi: NDArray) -> NDArray:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (number of samples, )
            Tuple of input samples.

        Returns
        -------
        :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)
        """
        return np.hstack(
            (
                self._basis1._evaluate(*xi[: self._basis1._n_input_dimensionality]),
                self._basis2._evaluate(*xi[self._basis1._n_input_dimensionality :]),
            )
        )


class MultiplicativeBasis(Basis):
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
    n_basis_funcs : int
        Number of basis functions.

    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self.n_basis_funcs = basis1.n_basis_funcs * basis2.n_basis_funcs
        super().__init__(self.n_basis_funcs)
        self._n_input_dimensionality = (
            basis1._n_input_dimensionality + basis2._n_input_dimensionality
        )
        self._basis1 = basis1
        self._basis2 = basis2
        return

    def _check_n_basis_min(self) -> None:
        pass

    def _evaluate(self, *xi: NDArray) -> NDArray:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (number of samples, )
            Tuple of input samples.

        Returns
        -------
        :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)
        """
        return np.array(
            row_wise_kron(
                self._basis1._evaluate(*xi[: self._basis1._n_input_dimensionality]),
                self._basis2._evaluate(*xi[self._basis1._n_input_dimensionality :]),
                transpose=False,
            )
        )


class SplineBasis(Basis, abc.ABC):
    """
    SplineBasis class inherits from the Basis class and represents spline basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    order : optional
        Spline order.

    Attributes
    ----------
    order : int
        Spline order.

    """

    def __init__(self, n_basis_funcs: int, order: int = 2) -> None:
        self.order = order
        super().__init__(n_basis_funcs)
        self._n_input_dimensionality = 1
        if self.order < 1:
            raise ValueError("Spline order must be positive!")

    def _generate_knots(
        self,
        sample_pts: NDArray,
        perc_low: float = 0.0,
        perc_high: float = 1.0,
        is_cyclic: bool = False,
    ) -> NDArray:
        """
        Generate knot locations for spline basis functions.

        Parameters
        ----------
        sample_pts : (number of samples, )
            The sample points.
        perc_low
            The low percentile value, between [0,1).
        perc_high
            The high percentile value, between (0,1].
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

        assert 0 <= perc_low <= 1, "Specify `perc_low` as a float between 0 and 1."
        assert 0 <= perc_high <= 1, "Specify `perc_high` as a float between 0 and 1."
        assert perc_low < perc_high, "perc_low must be less than perc_high."

        # clip to avoid numerical errors in case of percentile numerical precision close to 0 and 1
        # Spline basis have support on the semi-open [a, b)  interval, we add a small epsilon
        # to mx so that the so that basis_element(max(samples)) != 0
        mn = np.nanpercentile(sample_pts, np.clip(perc_low * 100, 0, 100))
        mx = np.nanpercentile(sample_pts, np.clip(perc_high * 100, 0, 100)) + 10**-8

        knot_locs = np.concatenate(
            (
                mn * np.ones(self.order - 1),
                np.linspace(mn, mx, num_interior_knots + 2),
                mx * np.ones(self.order - 1),
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


class MSplineBasis(SplineBasis):
    """M-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
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
        sample_pts :
            Spacing for basis functions, holding elements on the interval [min(sample_pts),
            max(sample_pts)], shape (number of samples, )

        Returns
        -------
        basis_funcs :
            Evaluated spline basis functions, shape (n_samples, n_basis_funcs).

        """
        # add knots if not passed
        knot_locs = self._generate_knots(
            sample_pts, perc_low=0.0, perc_high=1.0, is_cyclic=False
        )

        return np.stack(
            [
                mspline(sample_pts, self.order, i, knot_locs)
                for i in range(self.n_basis_funcs)
            ],
            axis=1,
        )


class BSplineBasis(SplineBasis):
    """
    B-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions.
    order :
        Order of the splines used in basis functions. Must lie within [1, n_basis_funcs].
        The B-splines have (order-2) continuous derivatives at each interior knot.
        The higher this number, the smoother the basis representation will be.

    Attributes
    ----------
    order :
        Spline order.


    References
    ----------
    ..[2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
    Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5

    """

    def __init__(self, n_basis_funcs: int, order: int = 2):
        super().__init__(n_basis_funcs, order=order)

    def _evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the B-spline is evaluated.

        Returns
        -------
        NDArray
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        Raises
        ------
        AssertionError
            If the sample points are not within the B-spline knots range unless `outer_ok=True`.

        Notes
        -----
        The evaluation is performed by looping over each element and using `splev`
        from SciPy to compute the basis values.
        """
        # add knots
        knot_locs = self._generate_knots(sample_pts, 0.0, 1.0)

        basis_eval = bspline(
            sample_pts, knot_locs, order=self.order, der=0, outer_ok=False
        )

        return basis_eval


class CyclicBSplineBasis(SplineBasis):
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

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.
    order : int
        Order of the splines used in basis functions.
    """

    def __init__(self, n_basis_funcs: int, order: int = 2):
        super().__init__(n_basis_funcs, order=order)
        if self.order < 2:
            raise ValueError(
                f"Order >= 2 required for cyclic B-spline, "
                f"order {self.order} specified instead!"
            )

    def _evaluate(self, sample_pts: NDArray) -> NDArray:
        """
        Evaluate the B-spline basis functions with given sample points.

        Parameters
        ----------
        sample_pts :
            The sample points at which the B-spline is evaluated. Must be a tuple of length 1.

        Returns
        -------
        NDArray
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        Raises
        ------
        AssertionError
            If the sample points are not within the B-spline knots range unless `outer_ok=True`.

        Notes
        -----
        The evaluation is performed by looping over each element and using `splev` from
        SciPy to compute the basis values.
        """
        knot_locs = self._generate_knots(sample_pts, 0.0, 1.0, is_cyclic=True)

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
        ind = sample_pts > xc

        basis_eval = bspline(sample_pts, knots, order=self.order, der=0, outer_ok=True)
        sample_pts[ind] = sample_pts[ind] - knots.max() + knot_locs[0]

        if np.sum(ind):
            basis_eval[ind] = basis_eval[ind] + bspline(
                sample_pts[ind], knots, order=self.order, outer_ok=True, der=0
            )
        # restore points
        sample_pts[ind] = sample_pts[ind] + knots.max() - knot_locs[0]

        return basis_eval


class RaisedCosineBasis(Basis, abc.ABC):
    def __init__(self, n_basis_funcs: int) -> None:
        super().__init__(n_basis_funcs)
        self._n_input_dimensionality = 1

    @abc.abstractmethod
    def _transform_samples(self, sample_pts: NDArray) -> NDArray:
        """
        Abstract method for transforming sample points.

        Parameters
        ----------
        sample_pts :
           The sample points to be transformed, shape (number of samples, ).
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
        basis_funcs :
            Raised cosine basis functions, shape (n_samples, n_basis_funcs).

        Raises
        ------
        ValueError
            If the sample provided do not lie in [0,1].
        """
        if any(sample_pts < 0) or any(sample_pts > 1):
            raise ValueError("Sample points for RaisedCosine basis must lie in [0,1]!")

        # transform to the proper domain
        transform_sample_pts = self._transform_samples(sample_pts)

        shifted_sample_pts = (
            transform_sample_pts[:, None]
            - (np.pi * np.arange(self.n_basis_funcs))[None, :]
        )
        basis_funcs = 0.5 * (np.cos(np.clip(shifted_sample_pts, -np.pi, np.pi)) + 1)

        return basis_funcs


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
        Linearly map the samples from [0,1] to the the [0, (n_basis_funcs - 1) * pi].

        Parameters
        ----------
        sample_pts :
            The sample points used for evaluating the splines, shape (number of samples, )

        Returns
        -------
        :
            A transformed version of the sample points that matches the Raised Cosine basis domain,
            shape (number of samples, ).
        """
        return sample_pts * np.pi * (self.n_basis_funcs - 1)

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Check that the number of basis is at least 1.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < 1:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 1 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )


class RaisedCosineBasisLog(RaisedCosineBasis):
    """Log-spaced raised cosine basis functions used by Pillow et al. [2]_.

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
        """Map the sample domain to log-space.

        Map the equi-spaced samples from [0,1] to log equi-spaced samples [0, (n_basis_funcs - 1) * pi].

        Parameters
        ----------
        sample_pts :
            The sample points used for evaluating the splines, shape (number of samples, ).

        Returns
        -------
        :
            A transformed version of the sample points that matches the Raised Cosine basis domain,
            shape (n_sample_points, ).
        """
        return (
            np.power(
                10,
                -(np.log10((self.n_basis_funcs - 1) * np.pi) + 1) * sample_pts
                + np.log10((self.n_basis_funcs - 1) * np.pi),
            )
            - 0.1
        )

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Checks that the number of basis is at least 2.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < 2:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 2 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )


class OrthExponentialBasis(Basis):
    """Set of 1D basis decaying exponential functions numerically orthogonalized.

    Parameters
    ----------
    n_basis_funcs
            Number of basis functions.
    decay_rates :
            Decay rates of the exponentials, shape (n_basis_funcs,).
    """

    def __init__(self, n_basis_funcs: int, decay_rates: NDArray[np.floating]):
        super().__init__(n_basis_funcs=n_basis_funcs)
        self._decay_rates = np.asarray(decay_rates)
        if self._decay_rates.shape[0] != n_basis_funcs:
            raise ValueError(
                f"The number of basis functions must match the number of decay rates provided. "
                f"Number of basis functions provided: {n_basis_funcs}, "
                f"Number of decay rates provided: {self._decay_rates.shape[0]}"
            )

        self._check_rates()
        self._n_input_dimensionality = 1

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Checks that the number of basis is at least 1.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < 1:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 1 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )

    def _check_rates(self):
        """
        Check if the decay rates list has duplicate entries.

        Raises
        ------
        ValueError
            If two or more decay rates are repeated, which would result in a linearly
            dependent set of functions for the basis.
        """
        if len(set(self._decay_rates)) != len(self._decay_rates):
            raise ValueError(
                "Two or more rate are repeated! Repeating rate will result in a "
                "linearly dependent set of function for the basis."
            )

    @staticmethod
    def _check_sample_range(sample_pts: NDArray):
        """
        Check if the sample points are all positive.

        Parameters
        ----------
        sample_pts
            Sample points to check.

        Raises
        ------
        ValueError
            If any of the sample points are negative, as OrthExponentialBasis requires
            positive samples.
        """
        if any(sample_pts < 0):
            raise ValueError(
                "OrthExponentialBasis requires positive samples. Negative values provided instead!"
            )

    def _check_sample_size(self, *sample_pts: NDArray):
        """Check that the sample size is greater than the number of basis.

        This is necessary for the orthogonalization procedure,
        that otherwise will return (sample_size, ) basis elements instead of the expected number.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval [0, inf).

        Raises
        ------
        ValueError
            If the number of basis element is less than the number of samples.
        """
        if sample_pts[0].size < self.n_basis_funcs:
            raise ValueError(
                "OrthExponentialBasis requires at least as many samples as basis functions!\n"
                f"Class instantiated with {self.n_basis_funcs} basis functions "
                f"but only {sample_pts[0].size} samples provided!"
            )

    def _evaluate(self, sample_pts: NDArray) -> NDArray:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval [0, inf), shape (n_pts,).

        Returns
        -------
        basis_funcs
            Evaluated exponentially decaying basis functions,
            numerically orthogonalized, shape (number of basis, number of samples).
        """
        self._check_sample_range(sample_pts)
        self._check_sample_size(sample_pts)
        # because of how scipy.linalg.orth works, have to create a matrix of
        # shape (n_pts, n_basis_funcs) and then transpose, rather than
        # directly computing orth on the matrix of shape (n_basis_funcs,
        # n_pts)
        return scipy.linalg.orth(
            np.stack([np.exp(-lam * sample_pts) for lam in self._decay_rates], axis=1)
        )


class FourierBasis(Basis):
    """Set of 1D Fourier basis.

    Parameters
    ----------
    n_freqs
            Number of frequencies. The number of basis function will be 2*n_freqs - 1.
    """

    def __init__(self, n_freqs: int):
        super().__init__(n_basis_funcs=2 * n_freqs - 1)

        self._frequencies = np.arange(n_freqs, dtype=np.float32)
        self._n_input_dimensionality = 1

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Checks that the number of basis is at least 1.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < 1:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 1 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )

    def _evaluate(self, sample_pts: NDArray) -> NDArray:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions.

        Returns
        -------
        basis_funcs
            Evaluated Fourier basis, shape (n_samples, n_basis_funcs).

        Notes
        -----
            If the frequencies provided are np.arange(n_freq), convolving a signal
            of length n_samples with this basis is equivalent, but slower,
            then computing the FFT truncated to the first n_freq components.

            Therefore, convolving a signal with this basis is equivalent
            to compute the FFT over sliding window.

            >>> import neurostatslib as nsl
            >>> import numpy as np
            >>> n_samples, n_freqs = 1000, 10
            >>> basis = nsl.basis.FourierBasis(n_freqs*2)
            >>> eval_basis = basis.evaluate(np.linspace(0, 1, n_samples))
            >>> sinusoid = np.cos(3 * np.arange(0, 1000) * np.pi * 2 / 1000.)
            >>> conv = [np.convolve(eval_basis[::-1, k], sinusoid, mode='valid')[0] for k in range(2*n_freqs-1)]
            >>> fft = np.fft.fft(sinusoid)
            >>> print('FFT power:   ', np.round(np.real(fft[:10]), 4))
            >>> print('Convolution: ', np.round(conv[:10], 4))
        """
        # assumes equi-spaced samples.
        if sample_pts.shape[0] / np.max(self._frequencies) < 2:
            raise ValueError("Not enough samples, aliasing likely to occur!")

        # rescale to [0, 2pi)
        mn, mx = np.nanmin(sample_pts), np.nanmax(sample_pts)
        # first sample in 0, last sample in 2 pi - 2 pi / n_samples.
        sample_pts = (
            2
            * np.pi
            * (sample_pts - mn)
            / (mx - mn)
            * (1.0 - 1.0 / sample_pts.shape[0])
        )
        # create the basis
        angles = np.einsum("i,j->ij", sample_pts, self._frequencies)
        return np.concatenate([np.cos(angles), -np.sin(angles[:, 1:])], axis=1)


def mspline(x: NDArray, k: int, i: int, T: NDArray):
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
):
    """
    Calculate and return the evaluation of B-spline basis.

    This function evaluates B-spline basis for given sample points. It checks for
    out of range points and optionally handles them. It also handles the NaNs if present.

    Parameters
    ----------
    sample_pts :
        An array containing sample points for which B-spline basis needs to be evaluated.
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
    basis_eval = np.zeros((n_basis - 2 * reps, sample_pts.shape[0]))

    # loop one element at the time and evaluate the basis using splev
    id_basis = np.eye(n_basis, nk, dtype=np.int8)
    for i in range(reps, len(knots) - order - reps):
        basis_eval[i - reps, in_sample] = splev(
            sample_pts[in_sample], (knots, id_basis[i], order - 1), der=der
        )

    return basis_eval.T
