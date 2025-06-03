import abc
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin


class FourierBasis(Basis, AtomicBasisMixin, abc.ABC):
    """Fourier Basis.

    Parameters
    ----------
    n_frequencies :
        The number of frequencies for the Fourier basis.
    include_constant:
        Include the constant term, which corresponds to 0 frequency. Default is False.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    """

    def __init__(
        self,
        n_frequencies: int,
        include_constant: bool = False,
        label: Optional[str] = "RaisedCosineBasisLinear",
    ) -> None:

        self.include_constant = include_constant
        self._n_input_dimensionality = 1

        # this sets the _n_basis_funcs too
        self.n_frequencies = n_frequencies

        AtomicBasisMixin.__init__(self, n_basis_funcs=self._n_basis_funcs, label=label)
        super().__init__()

    @property
    def include_constant(self):
        return bool(self._include_constant)

    @include_constant.setter
    def include_constant(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                f"`include_constant` must be a boolean. {value} provided instead!"
            )
        # store as int (used in slicing)
        self._include_constant = int(value)

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def evaluate(  # call these _evaluate
        self,
        sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor,
    ) -> FeatureMatrix:
        """Generate basis functions with given samples.

        Parameters
        ----------
        sample_pts :
            Spacing for basis functions, holding elements on interval [0, 1].
            `sample_pts` is a n-dimensional (n >= 1) array with first axis being the samples, i.e.
            `sample_pts.shape[0] == n_samples`.

        Raises
        ------
        ValueError
            If the sample provided do not lie in [0,1].

        """

        # scale to 0, 1
        sample_pts: NDArray = min_max_rescale_samples(
            np.copy(sample_pts), getattr(self, "bounds", None)
        )[0]
        # first sample in 0, last sample in 2 pi - 2 pi / n_samples.
        sample_pts *= 2 * np.pi * (1.0 - 1.0 / sample_pts.shape[0])

        # reshape samples
        shape = sample_pts.shape
        sample_pts = sample_pts.reshape(
            -1,
        )

        # compute angles
        angles = np.outer(sample_pts, self._frequencies)
        out = np.concatenate(
            [np.cos(angles), -np.sin(angles[:, int(self._include_constant) :])], axis=1
        )
        return out.reshape(*shape, out.shape[1])

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
            Fourier basis, shape (n_samples, n_basis_funcs)
        """
        return super().evaluate_on_grid(n_samples)

    @property
    def n_basis_funcs(self) -> tuple | None:
        """Read-only property for Fourier basis."""
        return super().n_basis_funcs

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies for the basis."""
        return int(self._frequencies[-1])

    @n_frequencies.setter
    def n_frequencies(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"`value` must be an integer. {value} provided instead!")
        if value <= 0:
            raise ValueError(
                f"`value` must be a positive integer. {value} provided instead!"
            )
        self._n_basis_funcs = 2 * value + self._include_constant
        self._frequencies = np.arange(
            1 - self._include_constant, value + 1, dtype=float
        )

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Checks that the number of basis is at least 1.

        Raises
        ------
        ValueError
            If a negative number of basis is provided.
        """
        if self.n_basis_funcs < 0:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 1 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )
