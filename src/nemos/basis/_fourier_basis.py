from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin
import abc
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pynapple import Tsd, TsdFrame, TsdTensor
from ..type_casting import support_pynapple
from ..typing import FeatureMatrix


class Fourier(Basis, AtomicBasisMixin, abc.ABC):
    """Fourier Basis.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        label: Optional[str] = "RaisedCosineBasisLinear",
    ) -> None:
        AtomicBasisMixin.__init__(self, n_basis_funcs=n_basis_funcs)
        super().__init__(
            mode=mode,
            label=label,
        )
        self._n_input_dimensionality = 1
        # for these linear raised-cosine basis functions,
        # the samples must be rescaled to 0 and 1.
        self._rescale_samples = True



    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(  # call these _evaluate
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
        pass

    def _compute_peaks(self) -> NDArray:
        """
        Compute the location of raised cosine peaks.

        Returns
        -------
            Peak locations of each basis element.
        """
        return np.linspace(0, 1, self.n_basis_funcs)

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
            Raised cosine basis functions, shape (n_samples, n_basis_funcs)
        """
        return super().evaluate_on_grid(n_samples)

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Check that the number of basis is at least 2.

        Raises
        ------
        ValueError
            If n_basis_funcs < 2.
        """
        if self.n_basis_funcs < 2:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 2 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )

