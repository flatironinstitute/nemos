"""Decaying exponential basis."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
from functools import partial
from typing import Optional, Tuple

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import (
    Basis,
    add_docstring,
    check_one_dimensional,
    check_transform_input,
    min_max_rescale_samples,
)


class OrthExponentialBasis(Basis, abc.ABC):
    """Set of 1D basis decaying exponential functions numerically orthogonalized.

    Parameters
    ----------
    n_basis_funcs
            Number of basis functions.
    decay_rates :
            Decay rates of the exponentials, shape ``(n_basis_funcs,)``.
    mode :
        The mode of operation. ``'eval'`` for evaluation at sample points,
        ``'conv'`` for convolutional operation.
    window_size :
        The window size for convolution. Required if mode is ``'conv'``.
    bounds :
        The bounds for the basis domain in ``mode="eval"``. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    **kwargs :
        Additional keyword arguments passed to :func:`nemos.convolve.create_convolutional_predictor` when
        ``mode='conv'``; These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.
    """

    def __init__(
        self,
        n_basis_funcs: int,
        decay_rates: NDArray[np.floating],
        mode="eval",
        label: Optional[str] = "OrthExponentialBasis",
        **kwargs,
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            label=label,
            **kwargs,
        )
        self.decay_rates = decay_rates
        self._check_rates()
        self._n_input_dimensionality = 1

    @property
    def decay_rates(self):
        r"""Decay rate.

        The rate of decay of the exponential functions. If :math:`f_i(t) = e^{-\alpha_i t}` is the i-th decay
        exponential before orthogonalization, :math:`\alpha_i` is the i-th element of the ``decay_rate`` vector.
        """
        return self._decay_rates

    @decay_rates.setter
    def decay_rates(self, value: NDArray):
        """Decay rate setter."""
        value = np.asarray(value, dtype=float)
        if value.shape[0] != self.n_basis_funcs:
            raise ValueError(
                f"The number of basis functions must match the number of decay rates provided. "
                f"Number of basis functions provided: {self.n_basis_funcs}, "
                f"Number of decay rates provided: {value.shape[0]}"
            )
        self._decay_rates = value

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

    def _check_rates(self) -> None:
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

    def _check_sample_size(self, *sample_pts: NDArray) -> None:
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

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(
        self,
        sample_pts: NDArray,
    ) -> FeatureMatrix:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval :math:`[0,inf)`,
            shape ``(n_samples,)``.

        Returns
        -------
        basis_funcs
            Evaluated exponentially decaying basis functions, numerically
            orthogonalized, shape ``(n_samples, n_basis_funcs)``.

        """
        self._check_sample_size(sample_pts)
        sample_pts, _ = min_max_rescale_samples(
            sample_pts, getattr(self, "bounds", None)
        )
        valid_idx = ~np.isnan(sample_pts)
        # because of how scipy.linalg.orth works, have to create a matrix of
        # shape (n_pts, n_basis_funcs) and then transpose, rather than
        # directly computing orth on the matrix of shape (n_basis_funcs,
        # n_pts)
        exp_decay_eval = np.stack(
            [np.exp(-lam * sample_pts[valid_idx]) for lam in self._decay_rates], axis=1
        )
        # count the linear independent components (could be lower than n_basis_funcs for num precision).
        n_independent_component = np.linalg.matrix_rank(exp_decay_eval)
        # initialize output to nan
        basis_funcs = np.full(
            shape=(sample_pts.shape[0], n_independent_component), fill_value=np.nan
        )
        # orthonormalize on valid points
        basis_funcs[valid_idx] = scipy.linalg.orth(exp_decay_eval)
        return basis_funcs

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
            OrthExponential basis functions, shape (n_samples, n_basis_funcs).
        """
        return super().evaluate_on_grid(n_samples)


add_orth_exp_decay_docstring = partial(add_docstring, cls=OrthExponentialBasis)
