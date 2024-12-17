from typing import Optional, Tuple

import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin


class IdentityBasis(Basis, AtomicBasisMixin):
    """
    Base class for the Identity basis.

    This basis should be used when one wants to use input samples
    as they are as predictors (IndentityEval).

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
            The samples reshaped into a 2D array.

        """
        sample_pts, _ = min_max_rescale_samples(
            np.copy(sample_pts), getattr(self, "bounds", None)
        )
        return sample_pts.reshape(sample_pts.shape[0], -1)

    def evaluate_on_grid(self, n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
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
            The linspace as a 2D array (n_samples, 1).
        """
        return super().evaluate_on_grid(n_samples)

    def _check_n_basis_min(self):
        """No checks are necessary."""
        pass


class HistoryBasis(Basis, AtomicBasisMixin):
    """
    Base class for history effects.

    This basis should be used when one wants to use input samples history
    as a predictors.

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
            The identity matrix of size len(samples).

        """
        # this is called by set kernel.
        return np.eye(len(sample_pts))

    def evaluate_on_grid(self, n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of sample points used to construct the identity matrix.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing a equispaced samples between 0 and 1.
        basis_funcs :
            The identity matrix of shape, i.e. ``np.eye(window_size, n_samples)``.
        """
        pass

    def _check_n_basis_min(self):
        """No checks are necessary."""
        pass
