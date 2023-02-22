"""
Python code to generate M-splines.

References
----------
Ramsay, J. O. (1988). Monotone regression splines in action.
Statistical science, 3(4), 425-441.
"""
import numpy as np
import scipy.linalg


class Basis:
    def __init__(self, num_basis_funcs, support):
        """
        n : int
            Number of basis functions.
        support:
            Domain of the basis functions.
        """
        self.num_basis_funcs = num_basis_funcs
        self.support = support

    @property
    def ndim(self):
        return len(self.shape)

    def check_in_support(self, x):
        raise NotImplementedError() # TODO


class OrthExponentials(Basis):
    """
    Set of 1D basis functions that are decaying exponentials
    numerically orthogonalized.
    """

    def __init__(self, *, decay_rates, window_size):
        super().__init__(len(decay_rates), (0, window_size))
        self.decay_rates = decay_rates
        self.window_size = window_size

    def transform(self, x=None):
        """
        Parameters
        ----------
        x : ndarray
            1d array, shape = (num_pts,), holding elements on the
            interval [0, window_size).

        Returns
        -------
        vals : ndarray
            2d array, shape = (num_basis_funcs, num_pts), holding
            evaluated spline basis functions.
        """

        # Default is to use a grid.
        if x is None:
            x = np.arange(self.window_size)

        # because of how scipy.linalg.orth works, have to create a matrix of
        # shape (num_pts, num_basis_funcs) and then transpose, rather than
        # directly computing orth on the matrix of shape (num_basis_funcs,
        # num_pts)
        return scipy.linalg.orth(
            np.stack([np.exp(-lam * x) for lam in self.decay_rates],
                     axis=1)
        ).T


class MSpline(Basis):
    """
    M-spline 1-dimensional basis functions.

    References
    ----------
    Ramsay, J. O. (1988). Monotone regression splines in action.
    Statistical science, 3(4), 425-441.
    """

    def __init__(self, *, num_basis_funcs, window_size, order=2):
        super().__init__(num_basis_funcs, (0, window_size))
        self.order = order
        self.window_size = window_size

        # Determine number of interior knots.
        self.num_interior_knots = nik = num_basis_funcs - order

        # Check hyperparameters.
        if nik < 0:
            raise ValueError(
                "Spline `order` parameter cannot be larger "
                "than `num_basis_funcs` parameter."
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
            np.linspace(0, 1, nik + 2),
            np.ones(order - 1),
        ))

    def transform(self, x=None):
        """
        Parameters
        ----------
        x : ndarray
            1d array, shape = (num_pts,), holding elements on the
            interval [0, window_size).

        Returns
        -------
        vals : ndarray
            2d array, shape = (num_basis_funcs, num_pts), holding
            evaluated spline basis functions.
        """

        # Default is to use a grid.
        if x is None:
            x = np.arange(self.window_size)

        x = x / self.window_size
        assert x.min() >= 0
        assert x.max() < 1

        return np.stack(
            [mspline(x, self.order, i, self.knot_locs) for i in range(self.num_basis_funcs)],
            axis=0
        )



def mspline(x, k, i, T):
    """
    Compute M-spline basis function `i` at points `x` for a spline
    basis of order-`k` with knots `T`.
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

    # Create figure and grid of evaluation points.
    fig, axes = plt.subplots(1, 5, sharey=True)

    # Iterate over axes to plot.
    for k, ax in enumerate(axes):
        
        # Create spline object.
        spline = MSpline(num_basis_funcs=6, window_size=1000, order=(k + 1))
        
        # Transform and plot spline bases.
        ax.plot(spline.transform().T)
        ax.set_yticks([])
        ax.set_title(f"order-{k + 1}")

    fig.tight_layout()
    plt.show()

    # Test for orthogonalized exponentials
    basis = OrthExponentials(
        decay_rates=np.logspace(-1, 0, 5),
        window_size=100
    )
    fig, ax = plt.subplots(1, 1)
    ax.plot(basis.transform().T)
    plt.show()

    from utils import convolve_1d_basis
    nt = 1000
    spikes = np.random.RandomState(123).randn(nt) > 2.5
    fig, ax = plt.subplots(1, 1, sharex=True)
    X = convolve_1d_basis(
        basis.transform(), spikes
    )
    ax.plot(spikes, color='k')
    ax.plot(
        np.arange(basis.window_size - 1, nt),
        X[0].T
    )
    plt.show()



