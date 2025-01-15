from typing import Optional, Tuple

import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_fraction_valid_samples, check_transform_input
from ._basis_mixin import AtomicBasisMixin


class IdentityBasis(Basis, AtomicBasisMixin):
    """
    Base class for the Identity basis.

    This basis should be used when one wants to incorporate raw inputs
    as part of a composite basis.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        label: Optional[str] = None,
    ) -> None:
        AtomicBasisMixin.__init__(self, n_basis_funcs=n_basis_funcs)
        super().__init__(
            label=label,
            mode="eval",
        )
        self._n_input_dimensionality = 1

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        """
        Returns the samples as a 2D array.

        Parameters
        ----------
        sample_pts:
            Array of samples. The shape can be arbitrary, as long as the first axis is the
            sample axis.

        Returns
        -------
        :
            The samples with an extra axis, the n_basis_funcs axis which is = 1.

        """
        vmin = np.nanmin(sample_pts, axis=0) if self.bounds is None else self.bounds[0]
        vmax = np.nanmax(sample_pts, axis=0) if self.bounds is None else self.bounds[1]
        sample_pts[(sample_pts < vmin) | (sample_pts > vmax)] = np.nan
        check_fraction_valid_samples(
            sample_pts,
            err_msg="All the samples lie outside the [vmin, vmax] range.",
            warn_msg="More than 90% of the samples lie outside the [vmin, vmax] range.",
        )
        return sample_pts[..., np.newaxis]

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of points in the uniformly spaced grid. A higher number of
            samples will result in a more detailed visualization of the basis functions.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            The equi-spaced sample points X as a 2D array (n_samples, 1).
        """
        return super().evaluate_on_grid(n_samples)

    def _check_n_basis_min(self):
        """No checks necessary."""
        pass

    @property
    def n_basis_funcs(self) -> tuple | None:
        """Number of basis functions (read-only)."""
        return super().n_basis_funcs


class HistoryBasis(Basis, AtomicBasisMixin):
    """
    Base class for history effects.

    This basis should be used when one wants to use input history
    as a predictor.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions (which corresponds to the length of the history window).
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        label: Optional[str] = None,
    ) -> None:
        AtomicBasisMixin.__init__(self, n_basis_funcs=n_basis_funcs)
        super().__init__(
            label=label,
            mode="conv",
        )
        self._n_input_dimensionality = 1

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        """
        Returns an identity matrix of size len(samples).

        The output is the convolutional kernels for spike history.

        Parameters
        ----------
        sample_pts:
            Array of samples.

        Returns
        -------
        :
            np.eye(*samples.shape, n_basis_funcs).

        """
        sample_pts = np.squeeze(np.asarray(sample_pts))
        if sample_pts.ndim != 1:
            raise ValueError("`_evaluate` for HistoryBasis allows 1D input only.")
        # this is called by set kernel.
        return np.eye(np.asarray(sample_pts).shape[0], self.n_basis_funcs)

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of points used to construct the identity matrix.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing a equi-spaced samples between 0 and 1.
        basis_funcs :
            The identity matrix of shape, i.e. ``np.eye(window_size, n_samples)``.
        """
        self._check_input_dimensionality((n_samples,))
        if self._has_zero_samples((n_samples,)):
            raise ValueError("All sample counts provided must be greater than zero.")
        return np.linspace(0, 1, n_samples), np.eye(n_samples, self.n_basis_funcs)

    def _check_n_basis_min(self):
        """No checks necessary."""
        pass

    @property
    def n_basis_funcs(self) -> tuple | None:
        """Read-only property for history basis."""
        return super().n_basis_funcs
