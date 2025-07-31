"""Module for ND fourier basis class."""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy._typing import NDArray
from numpy.typing import ArrayLike
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin


def rowwise_outer_sum_flatten(N, M):
    """
    N: array of shape (T, n)
    M: array of shape (T, m)
    Returns: array of shape (T, n * m)
    """
    T, n = N.shape
    _, m = M.shape

    # Expand N and M to enable broadcasting
    # N: (T, n, 1)
    # M: (T, 1, m)
    N_exp = N[:, :, jnp.newaxis]  # (T, n, 1)
    M_exp = M[:, jnp.newaxis, :]  # (T, 1, m)

    # Compute pairwise sums for each row: (T, n, m)
    pairwise_sum = N_exp + M_exp

    # Flatten the last two axes: (T, n * m)
    result = pairwise_sum.reshape(T, n * m)

    return result


def make_arange(arg):
    if isinstance(arg, int):
        if arg <= 0:
            raise ValueError("Integer frequencies must be > 0.")
        return jnp.arange(arg, dtype=float)
    elif isinstance(arg, tuple) and len(arg) == 2:
        start, stop = arg
        if not (isinstance(start, int) and isinstance(stop, int)):
            raise TypeError("Tuple frequencies must be integers.")
        if start < 0 or stop <= start:
            raise ValueError("Tuple frequencies must satisfy 0 <= start < stop.")
        return jnp.arange(start, stop, dtype=float)
    else:
        raise TypeError("Each frequency must be an int or a 2-element tuple of ints.")


class FourierBasis(AtomicBasisMixin, Basis):

    def __init__(
        self,
        frequencies: int | Tuple[int, int] | List[int] | List[Tuple[int, int]],
        frequency_mask: jnp.ndarray | None = None,
        label: Optional[str] = None,
    ) -> None:
        self.frequencies = frequencies
        self.frequency_mask = frequency_mask

        AtomicBasisMixin.__init__(
            self,
            n_basis_funcs=self.n_basis_funcs,
            label=label,
        )

    @property
    def frequency_mask(self) -> jnp.ndarray | None:
        return self._frequency_mask

    @frequency_mask.setter
    def frequency_mask(self, values: ArrayLike | jnp.ndarray | None) -> None:
        if values is None:
            self._frequency_mask = None

            # cache all frequencies
            grids = jnp.meshgrid(
                *[jnp.arange(len(freqs)) for freqs in self._frequencies], indexing="ij"
            )
            idxs = jnp.stack([g.reshape(-1) for g in grids], axis=1)
            self._eval_freq = jnp.stack(
                [freqs[idxs[:, d]] for d, freqs in enumerate(self._frequencies)], axis=0
            )
            return
        try:
            values = jnp.asarray(values)
        except Exception:
            raise ValueError(
                "``frequency_mask`` cannot be converted to a jax array of boolean."
            )

        if not jnp.all((values == 0) | (values == 1)):
            raise ValueError("Frequency mask must be an array-like of 0s and 1s.")

        values = values.astype(bool)

        if not values.ndim == self._n_input_dimensionality:
            ndim = self._n_input_dimensionality
            raise ValueError(
                f"The frequency mask for an  {ndim}-dimensional Fourier basis "
                f"must be an {ndim}-dimensional array of 0s and 1s. "
                f"The provided mask is an {ndim}-dimensional array instead."
            )

        if tuple(len(freqs) for freqs in self._frequencies) != values.shape:
            expected_shape = tuple(len(freqs) for freqs in self._frequencies)
            raise ValueError(
                "Invalid shape for ``frequency_mask``. "
                f"Expected shape {expected_shape}, "
                f"but got {values.shape} instead. The mask must have one entry (0 or 1) for each "
                "frequency along every dimension of the Fourier basis."
            )

        self._frequency_mask = values
        self._n_basis_funcs = 2 * int(jnp.sum(self._frequency_mask)) - int(
            self._frequency_mask[(0,) * values.ndim]
        )

        idxs = jnp.stack(jnp.where(self._frequency_mask), axis=0)
        self._eval_freq = jnp.stack(
            [freqs[idxs[d]] for d, freqs in enumerate(self._frequencies)]
        )

    @property
    def frequencies(self) -> Tuple[jnp.ndarray, ...]:
        return self._frequencies

    @frequencies.setter
    def frequencies(
        self, frequencies: int | tuple[int, int] | list[int] | list[tuple[int, int]]
    ) -> None:

        if isinstance(frequencies, int):
            self._frequencies = tuple(
                make_arange(frequencies) for _ in range(self._n_input_dimensionality)
            )

        elif isinstance(frequencies, tuple):
            self._frequencies = tuple(
                make_arange(frequencies) for _ in range(self._n_input_dimensionality)
            )

        elif isinstance(frequencies, list):
            if len(frequencies) != self._n_input_dimensionality:
                raise ValueError(
                    "Length of frequencies list must match input dimensionality."
                )
            self._frequencies = tuple(make_arange(f) for f in frequencies)

        else:
            raise TypeError(
                "Frequencies must be one of:\n"
                "  - int\n"
                "  - tuple[int, int]\n"
                "  - list[int]\n"
                "  - list[tuple[int, int]]\n"
                f"If a list is provided, the list should be of length ``{self.ndim}``, "
                f"one entry for each dimension of the Fourier basis."
            )

    @property
    def ndim(self):
        return self._n_input_dimensionality

    @ndim.setter
    def ndim(self, ndim: int) -> None:
        # TODO: Add an exception handling
        self._n_input_dimensionality = ndim

    @property
    def n_basis_funcs(self) -> int | None:
        if self._frequency_mask is None:
            return 2 * len(self._frequencies) - 1 * (0 in self._frequencies)
        else:
            return 2 * int(jnp.sum(self._frequency_mask)) - int(
                self._frequency_mask[(0,) * self._frequency_mask.ndim]
            )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def evaluate(  # call these _evaluate
        self,
        *sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor,
    ) -> FeatureMatrix:
        """Evaluate the Fourier basis at the sample points.

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
        shape = sample_pts[0].shape

        bounds = getattr(self, "bounds", None)
        if bounds is None:
            bounds = (None,) * self._n_input_dimensionality

        # min/max rescale to [0,1]:
        # The function does so over the time axis (each extra dim is
        # normalized independently)
        def _flat_samples_to_angles(xs):
            scaled_samples = jax.tree_util.tree_map(
                lambda x, b: 2
                * jnp.pi
                * self._shift_angles(min_max_rescale_samples(x, b)[0].reshape(-1)),
                xs,
                bounds,
            )
            return jnp.stack(scaled_samples, axis=-1)

        sample_pts = _flat_samples_to_angles(sample_pts)
        angles = sample_pts @ self._eval_freq
        out = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=1)
        return out.reshape(*shape, out.shape[-1])

    def _shift_angles(self, sample_pts: ArrayLike) -> ArrayLike:
        """
        Shift angles.

        Reimplemented for ``FourierConv``, shifting the angles to
        match the Fourier coefficients when the basis is used for convolutions.
        This shift must not be applied for ``FourierEval`` basis, therefore the
        super-class implements an identity function.

        Parameters
        ----------
        sample_pts :
            The samples.

        Returns
        -------
        sample_pts :
            The samples as provided, identity function.
        """
        return sample_pts
