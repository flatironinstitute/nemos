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


class Basis:
    """Generic class for basis functions

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.
    support
        Possible domain of the basis functions.

    """

    def __init__(self, n_basis_funcs: int, window_size: int, support: Tuple[int, int]):
        self.n_basis_funcs = n_basis_funcs
        self.window_size = window_size
        self.support = support
        # display string when showing support
        self._support_display = f"[{self.support[0]}, {self.support[1]})"

    def _check_array(self, x: NDArray, ndim: int = 1):
        """Check whether x is array with given number of dims.

        We check whether it's an array implicitly, by checking its ``ndim``
        attribute (thus, works with both numpy and jax.numpy arrays).

        Parameters
        ----------
        x
            Input array to check.

        Raises
        ------
        ValueError
            If x does not have the appropriate number of dimensions.
        TypeError
            If x does not have an ndim attribute (and thus is not an array).

        """
        try:
            if x.ndim != ndim:
                raise ValueError(
                    f"Input must have {ndim} dimensions but has {x.ndim} instead!"
                )
        except AttributeError:
            raise TypeError("Input is not an array!")

    def _check_num_sample_pts(self, sample_pts: NDArray):
        """Raise warning if number of sample points looks off.

        It's not necessarily *wrong* if ``len(sample_pts) !=
        self.window_size``, but that's generally what we want, so raise a
        warning if so.

        """
        if len(sample_pts) != self.window_size:
            warnings.warn(
                "sample_pts is not the same length as the Basis "
                "window_size -- are you sure that's what you want?"
            )

    def check_in_support(self, x: NDArray):
        """Check whether x lies within support.

        Parameters
        ----------
        x
            Input array to check.

        Raises
        ------
        ValueError
            If x lies outside the interval [self.support[0], self.support[1]),
            i.e., we check ``x<support[0]`` and ``x>=support[1]``

        """
        if (x.min() < self.support[0]) or (x.max() >= self.support[1]):
            raise ValueError(f"Input must lie within support {self._support_display}!")

    @abc.abstractmethod
    def gen_basis_funcs(
        self, sample_pts: NDArray, check_support: bool = True
    ) -> NDArray:
        """Generate basis functions with given spacing.

        Note input must be 1d and output 2d. Children classes can call this as
        convenient way to check input.

        Parameters
        ----------
        sample_pts : (n_pts,)
            Spacing for basis functions. *RECOMMEND A GOOD DEFAULT*

        Returns
        -------
        basis_funcs : (n_basis_funcs, n_pts)
            basis functions

        """
        self._check_array(sample_pts)
        # better not be strict, in many case the tail of the data distribution can be quite high
        # and it might be the case that the basis covers only within certain percentiles
        if check_support:
            self.check_in_support(sample_pts)
        self._check_num_sample_pts(sample_pts)


class RaisedCosineBasis(Basis):
    """Raised cosine basis functions used by Pillow et al. [2]_.

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

    def __init__(self, n_basis_funcs: int, window_size: int):
        super().__init__(n_basis_funcs, window_size, (0, window_size))

    def gen_basis_funcs(self, sample_pts: NDArray) -> NDArray:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts : (n_pts,)
            Spacing for basis functions, holding elements on interval [0,
            window_size). A good default is
            ``nsl.sample_points.raised_cosine_log`` for log spacing (as used in
            [2]_) or ``nsl.sample_points.raised_cosine_linear`` for linear
            spacing.

        Returns
        -------
        basis_funcs : (n_basis_funcs, n_pts)
            Raised cosine basis functions

        """
        super().gen_basis_funcs(sample_pts)
        # this has shape (n_basis_funcs, n_pts) and just consists of shifted
        # copies of the input.
        shifted_sample_pts = (
            sample_pts[None, :] - (np.pi * np.arange(self.n_basis_funcs))[:, None]
        )
        basis_funcs = 0.5 * (np.cos(np.clip(shifted_sample_pts, -np.pi, np.pi)) + 1)
        if (abs(basis_funcs.sum(0) - 1) > 1e-12).any():
            raise ValueError(
                "sample_pts was generated with too large an n_basis_funcs arg, "
                "our generated basis functions do not uniformly tile the space!"
            )
        if (basis_funcs == 0).all(1).any():
            raise ValueError(
                "sample_pts was generated with too small an n_basis_funcs arg, "
                "at least one of our generated basis functions is 0 everywhere!"
            )
        return basis_funcs


class OrthExponentialBasis(Basis):
    """Set of 1D basis functions that are decaying exponentials numerically
    orthogonalized.

    Parameters
    ----------
    decay_rates : (n_basis_funcs,)
        Decay rates of the basis functions
    window_size
        Size of basis functions.

    """

    def __init__(self, decay_rates: NDArray[np.floating], window_size: int):
        super().__init__(len(decay_rates), window_size, (0, window_size))
        self.decay_rates = decay_rates

    def gen_basis_funcs(self, sample_pts: NDArray) -> NDArray:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts : (n_pts,)
            Spacing for basis functions, holding elements on the interval [0,
            window_size). A good default is np.arange(window_size).

        Returns
        -------
        basis_funcs : (n_basis_funcs, n_pts)
            Evaluated spline basis functions

        """
        super().gen_basis_funcs(sample_pts)
        # because of how scipy.linalg.orth works, have to create a matrix of
        # shape (n_pts, n_basis_funcs) and then transpose, rather than
        # directly computing orth on the matrix of shape (n_basis_funcs,
        # n_pts)
        return scipy.linalg.orth(
            np.stack([np.exp(-lam * sample_pts) for lam in self.decay_rates], axis=1)
        ).T


class SplineBasis(Basis):
    def __init__(self, n_basis_funcs: int, window_size: int, order: int = 2):
        super().__init__(n_basis_funcs, window_size, (0, window_size))
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
        # Set of spline knots. We need to add extra knots to
        # the end to handle boundary conditions for higher-order
        # spline bases. See Ramsay (1988) cited above.
        #
        # Note - this is poorly explained on most corners of the
        # internet that I've found.
        #
        # TODO : allow users to specify the knot locations if
        # they want.... but this could be the default.

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

    def __init__(self, n_basis_funcs: int, window_size: int, order: int = 2):
        super().__init__(n_basis_funcs, window_size, order)

    def gen_basis_funcs(self, sample_pts: NDArray) -> NDArray:
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

        super().gen_basis_funcs(sample_pts)

        sample_pts = sample_pts / self.window_size

        # add knots if not passed
        if not hasattr(self, "knot_locs"):
            self.generate_knots(sample_pts, 0.0, 1.0)

        return np.stack(
            [
                mspline(sample_pts, self.order, i, self.knot_locs)
                for i in range(self.n_basis_funcs)
            ],
            axis=0,
        )


class BSplineBasis(SplineBasis):
    """B-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.
    order
        Order of the splines used in basis functions. Must lie within [1,
        n_basis_funcs]. The B-splines have ``order-2`` continuous derivatives
        at each interior knot. The higher this number, the smoother the basis
        representation will be.


    References
    ----------
    .. [2] Prautzsch, H., Boehm, W., Paluszny, M. (2002). B-spline representation. In: Bézier and B-Spline Techniques.
    Mathematics and Visualization. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-04919-8_5


    """

    def __init__(self, n_basis_funcs: int, window_size: int, order: int = 2):
        super().__init__(n_basis_funcs, window_size, order)

    def gen_basis_funcs(
        self, sample_pts: NDArray, outer_ok: bool = False, der: int = 0
    ) -> NDArray:
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

        knots : None or (number of knots,)
            add the possibility of passing a different knots vector, used for cyclic splines

        Returns
        -------

        """
        super().gen_basis_funcs(sample_pts, check_support=False)
        # add knots if not passed
        if not hasattr(self, "knot_locs"):
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

        # # check sum equal 1 (B-spline are supposed to sum to 1)
        # assert(np.abs(basis_eval.sum(axis=0) - 1).max() < 1e-6)
        return basis_eval


class Cyclic_BSplineBasis(BSplineBasis):
    """Cyclic B-spline 1-dimensional basis functions.

    Parameters
    ----------
    n_basis_funcs
        Number of basis functions.
    window_size
        Size of basis functions.
    order
        Order of the splines used in basis functions. Must lie within [1,
        n_basis_funcs]. The B-splines have ``order-2`` continuous derivatives
        at each interior knot. The higher this number, the smoother the basis
        representation will be.

    """

    def __init__(self, n_basis_funcs: int, window_size: int, order: int = 2):
        super().__init__(n_basis_funcs, window_size, order)
        assert (
            self.order >= 2
        ), f"Order >= 2 required for cyclic B-spline, order {self.order} specified instead!"
        assert (
            self.n_basis_funcs >= order + 2
        ), "n_basis_funcs >= order + 2 required for cyclic B-spline"
        assert (
            self.n_basis_funcs >= 2 * order - 2
        ), "n_basis_funcs >= 2*(order - 1) required for cyclic B-spline"

    def gen_basis_funcs(self, sample_pts: NDArray, der: int = 0) -> NDArray:
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
        basis_eval = super().gen_basis_funcs(sample_pts, outer_ok=True, der=der)
        sample_pts[ind] = sample_pts[ind] - knots.max() + knots_orig[0]
        if np.sum(ind):
            X2 = super().gen_basis_funcs(sample_pts[ind], outer_ok=True, der=der)
            basis_eval[:, ind] = basis_eval[:, ind] + X2
        # restore points
        sample_pts[ind] = sample_pts[ind] + knots.max() - knots_orig[0]
        # restore the original knots
        self.knot_loc = knots_orig
        return basis_eval


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
        return (
            k
            * (
                (x - T[i]) * mspline(x, k - 1, i, T)
                + (T[i + k] - x) * mspline(x, k - 1, i + 1, T)
            )
            / ((k - 1) * (T[i + k] - T[i]))
        )


# Short test
if __name__ == "__main__":
    # For plotting.
    import matplotlib.pyplot as plt

    # # Create figure and grid of evaluation points.
    # fig, axes = plt.subplots(1, 5, sharey=True)
    # # Iterate over axes to plot.
    # for k, ax in enumerate(axes):
    #     # Create spline object.
    #     spline = MSplineBasis(n_basis_funcs=6, window_size=1000, order=(k + 1))
    #     # Transform and plot spline bases.
    #     ax.plot(spline.transform().T)
    #     ax.set_yticks([])
    #     ax.set_title(f"order-{k + 1}")
    # fig.tight_layout()
    # plt.show()
    # # Test for orthogonalized exponentials
    # basis = OrthExponentialBasis(
    #     decay_rates=np.logspace(-1, 0, 5),
    #     window_size=100
    # )
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(basis.transform().T)
    # plt.show()
    # Test for raised cosines
    basis = RaisedCosineBasis(n_basis_funcs=5, window_size=1000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(basis.transform().T)
    plt.show()

    # from utils import convolve_1d_basis
    # nt = 1000
    # spikes = np.random.RandomState(123).randn(nt) > 2.5
    # fig, ax = plt.subplots(1, 1, sharex=True)
    # X = convolve_1d_basis(
    #     basis.transform(), spikes
    # )
    # ax.plot(spikes, color='k')
    # ax.plot(
    #     np.arange(basis.window_size - 1, nt),
    #     X[0].T
    # )
    # plt.show()
