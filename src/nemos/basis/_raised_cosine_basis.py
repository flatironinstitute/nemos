# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin


class RaisedCosineBasisLinear(Basis, AtomicBasisMixin, abc.ABC):
    """Represent linearly-spaced raised cosine basis functions.

    This implementation is based on the cosine bumps used by Pillow et al. [1]_
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    width :
        Width of the raised cosine. By default, it's set to 2.0.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
        C. E. (2005). Prediction and decoding of retinal ganglion cell responses
        with a probabilistic spiking model. Journal of Neuroscience, 25(47),
        11003–11013.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        width: float = 2.0,
        label: Optional[str] = "RaisedCosineBasisLinear",
    ) -> None:
        AtomicBasisMixin.__init__(self, n_basis_funcs=n_basis_funcs)
        super().__init__(
            mode=mode,
            label=label,
        )
        self._n_input_dimensionality = 1
        self._check_width(width)
        self._width = width
        # for these linear raised-cosine basis functions,
        # the samples must be rescaled to 0 and 1.
        self._rescale_samples = True

    @property
    def width(self):
        """Return width of the raised cosine."""
        return self._width

    @width.setter
    def width(self, width: float):
        self._check_width(width)
        self._width = width

    @staticmethod
    def _check_width(width: float) -> None:
        """Validate the width value.

        Parameters
        ----------
        width :
            The width value to validate.

        Raises
        ------
        ValueError
            If width <= 1 or 2*width is not a positive integer. Values that do not match
            this constraint will result in:
            - No overlap between bumps (width < 1).
            - Oscillatory behavior when summing the basis elements (2*width not integer).
        """
        if width <= 1 or (not np.isclose(width * 2, round(2 * width))):
            raise ValueError(
                f"Invalid raised cosine width. "
                f"2*width must be a positive integer, 2*width = {2 * width} instead!"
            )

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
        if self._rescale_samples:
            # note that sample points is converted to NDArray
            # with the decorator.
            # copy is necessary otherwise:
            # basis1 = nmo.basis.RaisedCosineBasisLinear(5)
            # basis2 = nmo.basis.RaisedCosineBasisLog(5)
            # additive_basis = basis1 + basis2
            # additive_basis._evaluate(*([x] * 2)) would modify both inputs
            sample_pts, _ = min_max_rescale_samples(
                np.copy(sample_pts), getattr(self, "bounds", None)
            )

        peaks = self._compute_peaks()
        delta = peaks[1] - peaks[0]

        # reshape samples
        shape = sample_pts.shape
        sample_pts = sample_pts.reshape(
            -1,
        )

        # generate a set of shifted cosines, and constrain them to be non-zero
        # over a single period, then enforce the codomain to be [0,1], by adding 1
        # and then multiply by 0.5
        basis_funcs = 0.5 * (
            np.cos(
                np.clip(
                    np.pi * (sample_pts[:, None] - peaks[None]) / (delta * self.width),
                    -np.pi,
                    np.pi,
                )
            )
            + 1
        )
        basis_funcs = basis_funcs.reshape(*shape, basis_funcs.shape[1])
        return basis_funcs

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


class RaisedCosineBasisLog(RaisedCosineBasisLinear, abc.ABC):
    """Represent log-spaced raised cosine basis functions.

    Similar to ``RaisedCosineBasisLinear`` but the basis functions are log-spaced.
    This implementation is based on the cosine bumps used by Pillow et al. [1]_
    to uniformly tile the internal points of the domain.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    width :
        Width of the raised cosine.
    time_scaling :
        Non-negative hyper-parameter controlling the logarithmic stretch magnitude, with
        larger values resulting in more stretching. As this approaches 0, the
        transformation becomes linear.
    enforce_decay_to_zero:
        If set to True, the algorithm first constructs a basis with ``n_basis_funcs + ceil(width)`` elements
        and subsequently trims off the extra basis elements. This ensures that the final basis element
        decays to 0.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    References
    ----------
    .. [1] Pillow, J. W., Paninski, L., Uzzel, V. J., Simoncelli, E. P., & J.,
       C. E. (2005). Prediction and decoding of retinal ganglion cell responses
       with a probabilistic spiking model. Journal of Neuroscience, 25(47),
       11003–11013.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode="eval",
        width: float = 2.0,
        time_scaling: float = None,
        enforce_decay_to_zero: bool = True,
        label: Optional[str] = "RaisedCosineBasisLog",
    ) -> None:
        super().__init__(
            n_basis_funcs,
            mode=mode,
            width=width,
            label=label,
        )
        # The samples are scaled appropriately in the self._transform_samples which scales
        # and applies the log-stretch, no additional transform is needed.
        self._rescale_samples = False
        if time_scaling is None:
            time_scaling = 50.0

        self.time_scaling = time_scaling
        self.enforce_decay_to_zero = enforce_decay_to_zero

    @property
    def time_scaling(self):
        """Time scaling.

        Time scaling parameter regulating the logarithmic stretch of the basis.
        """
        return self._time_scaling

    @time_scaling.setter
    def time_scaling(self, time_scaling):
        """Setter property for time_scaling."""
        self._check_time_scaling(time_scaling)
        self._time_scaling = time_scaling

    @staticmethod
    def _check_time_scaling(time_scaling: float) -> None:
        if time_scaling <= 0:
            raise ValueError(
                f"Only strictly positive time_scaling are allowed, {time_scaling} provided instead."
            )

    def _transform_samples(
        self,
        sample_pts: ArrayLike,
    ) -> NDArray:
        """
        Map the sample domain to log-space.

        Parameters
        ----------
        sample_pts :
            Sample points used for evaluating the splines,
            shape (n_samples, ).

        Returns
        -------
            Transformed version of the sample points that matches the Raised Cosine basis domain,
            shape (n_samples, ).
        """
        # rescale to [0,1]
        # copy is necessary to avoid unwanted rescaling in additive/multiplicative basis.
        sample_pts, _ = min_max_rescale_samples(
            np.copy(sample_pts), getattr(self, "bounds", None)
        )
        # This log-stretching of the sample axis has the following effect:
        # - as the time_scaling tends to 0, the points will be linearly spaced across the whole domain.
        # - as the time_scaling tends to inf, basis will be small and dense around 0 and
        # progressively larger and less dense towards 1.
        log_spaced_pts = np.log(self.time_scaling * sample_pts + 1) / np.log(
            self.time_scaling + 1
        )
        return log_spaced_pts

    def _compute_peaks(self) -> NDArray:
        """
        Peak location of each log-spaced cosine basis element.

        Compute the peak location for the log-spaced raised cosine basis.
        Enforcing that the last basis decays to zero is equivalent to
        setting the last peak to a value smaller than 1.

        Returns
        -------
            Peak locations of each basis element.

        """
        if self.enforce_decay_to_zero:
            # compute the last peak location such that the last
            # basis element decays to zero at the last sample.
            last_peak = 1 - self.width / (self.n_basis_funcs + self.width - 1)
        else:
            last_peak = 1
        return np.linspace(0, last_peak, self.n_basis_funcs)

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def _evaluate(
        self,
        sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor,
    ) -> FeatureMatrix:
        """Generate log-spaced raised cosine basis with given samples.

        Parameters
        ----------
        sample_pts :
            Spacing for basis functions. Samples will be rescaled to the interval [0, 1].
            `sample_pts` is a n-dimensional (n >= 1) array with first axis being the samples, i.e.
            `sample_pts.shape[0] == n_samples`.

        Returns
        -------
        basis_funcs :
            Log-raised cosine basis functions, shape (n_samples, n_basis_funcs).

        Raises
        ------
        ValueError
            If the sample provided do not lie in [0,1].
        """
        return super()._evaluate(self._transform_samples(sample_pts))
