"""Zero basis used for variable selection."""

from typing import Optional, Tuple

import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input
from ._basis_mixin import AtomicBasisMixin


class ZeroBasis(AtomicBasisMixin, Basis):
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
        label: Optional[str] = None,
    ) -> None:
        AtomicBasisMixin.__init__(self, n_basis_funcs=0, label=label)
        Basis.__init__(
            self,
        )
        self._n_input_dimensionality = 1

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def evaluate(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        """
        Evaluate the Zero basis functions.

        Define an empty array with shape ``(*sample_pts.shape, 0)``.

        Parameters
        ----------
        sample_pts:
            Array of samples. The shape can be arbitrary, as long as the first axis is the
            sample axis.

        Returns
        -------
        :
            The samples with an extra axis, the n_basis_funcs axis which is = -1.

        """
        shape = sample_pts.shape
        return np.empty((*shape, 0), dtype=float)

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
