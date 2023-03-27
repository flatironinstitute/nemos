"""Bases classes.
"""
# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations
import numpy as np
import abc
import scipy.linalg
from typing import Tuple, Optional
from numpy.typing import ArrayLike, NDArray


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
    def __init__(self, n_basis_funcs: int,
                 window_size: int,
                 support: Tuple[int, int]):
        self.n_basis_funcs = n_basis_funcs
        self.window_size = window_size
        self.support = support
        # display string when showing support
        self._support_display = f'[{self.support[0]}, {self.support[1]})'

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
                raise ValueError(f"Input must have {ndim} dimensions but has {x.ndim} instead!")
        except AttributeError:
            raise TypeError("Input is not an array!")


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
    def gen_basis_funcs(self, sample_pts: NDArray) -> NDArray:
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
        self.check_in_support(sample_pts)


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
    def __init__(self, n_basis_funcs: int,
                 window_size: int):
        super().__init__(n_basis_funcs, window_size,
                         (0, window_size))

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
        shifted_sample_pts = sample_pts[None, :] - (np.pi * np.arange(self.n_basis_funcs))[:, None]
        return .5 * (np.cos(np.clip(shifted_sample_pts, -np.pi, np.pi)) + 1)


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

    def __init__(self, decay_rates: NDArray[np.floating],
                 window_size: int):
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
            np.stack([np.exp(-lam * sample_pts) for lam in self.decay_rates],
                     axis=1)
        ).T


class MSplineBasis(Basis):
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

    def __init__(self, n_basis_funcs: int, window_size:int,
                 order: int = 2):
        super().__init__(n_basis_funcs, window_size, (0, window_size))
        self.order = order

        # Determine number of interior knots.
        num_interior_knots = n_basis_funcs - order

        if order < 1:
            raise ValueError('Spline order must be positive!')
        # Check hyperparameters.
        if num_interior_knots < 0:
            raise ValueError(
                "Spline `order` parameter cannot be larger "
                "than `n_basis_funcs` parameter."
            )

        # Set of spline knots. We need to add extra knots to
        # the end to handle boundary conditions for higher-order
        # spline bases. See Ramsay (1988) cited above.
        #
        # Note - this is poorly explained on most corners of the
        # internet that I've found.
        #
        # TODO : allow users to specify the knot locations if
        # they want.... but this could be the default.
        self.knot_locs = np.concatenate((
            np.zeros(order - 1),
            np.linspace(0, 1, num_interior_knots + 2),
            np.ones(order - 1),
        ))

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

        return np.stack(
            [mspline(sample_pts, self.order, i, self.knot_locs) for i in range(self.n_basis_funcs)],
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
        ) / ((k-1) * (T[i + k] - T[i]))


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
    basis = RaisedCosineBasis(
        n_basis_funcs=5,
        window_size=1000
    )
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



