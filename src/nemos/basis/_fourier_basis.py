"""Module for ND fourier basis class."""

import warnings
from numbers import Number
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy._typing import NDArray
from numpy.typing import ArrayLike
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import is_numpy_array_like, support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin

FREQUENCY_ERROR_MSGS = {
    "has_incorrect_length": "The length of provided tuple of frequencies doesn't match the basis dimensionality.\n"
    "The basis dimensionality is {expected}, while {provided} frequency arrays "
    "were provided instead.",
    "has_negative_values": "The provided frequencies contain negative values. Valid frequencies must be"
    "non-negative integers.",
    "not_int_like": "Some frequency values are not integers. Valid frequencies must be"
    "non-negative integers.",
}


def _to_tuple_of_arange(frequencies: int | Tuple[int, int] | Tuple[int], ndim: int):
    if hasattr(frequencies, "__len__") and len(frequencies) == 1:
        frequencies = frequencies[0]
    return tuple(make_arange(frequencies) for _ in range(ndim))


def _check_array_properties(*arrays: ArrayLike, ndim: int = 1):
    has_correct_length = len(arrays) == ndim
    is_int_like = all(np.all(f == np.asarray(f, dtype=int)) for f in arrays)
    is_positive = all(np.all(np.asarray(f) >= 0) for f in arrays)
    return has_correct_length, is_int_like, is_positive


def _format_error_message(
    ndim: int,
    frequencies: Tuple[ArrayLike],
    has_correct_length: bool,
    is_positive: bool,
    is_int_like: bool,
):
    msg = "Invalid frequencies provided. The following issues were detected:\n\n"

    if not has_correct_length:
        msg += (
            "\t- "
            + FREQUENCY_ERROR_MSGS["has_incorrect_length"]
            .format(expected=ndim, provided=len(frequencies))
            .replace("\n", "\n\t  ")
            + "\n\n"
        )
    if not is_positive:
        msg += (
            "\t- "
            + FREQUENCY_ERROR_MSGS["has_negative_values"].replace("\n", "\n\t  ")
            + "\n\n"
        )
    if not is_int_like:
        msg += (
            "\t- "
            + FREQUENCY_ERROR_MSGS["not_int_like"].replace("\n", "\n\t  ")
            + "\n\n"
        )

    return msg


def _check_and_sort_frequencies(*frequencies, ndim: int = 1):
    has_correct_length, is_int_like, is_positive = _check_array_properties(
        *frequencies, ndim=ndim
    )

    if has_correct_length and is_int_like and is_positive:
        sorted_freqs = tuple(np.sort(f) for f in frequencies)
        if any(not np.array_equal(f1, f2) for f1, f2 in zip(frequencies, sorted_freqs)):
            warnings.warn("Unsorted frequencies provided! Frequencies will be sorted.")
            return sorted_freqs
        return frequencies

    raise ValueError(
        _format_error_message(
            ndim, frequencies, has_correct_length, is_positive, is_int_like
        )
    )


def make_arange(arg: NDArray | int | Tuple[int, int]):

    if is_numpy_array_like(arg):
        return arg

    if isinstance(arg, int):
        if arg <= 0:
            raise ValueError(
                f"Integer frequencies must be > 0, {arg} provided instead."
            )
        return jnp.arange(arg, dtype=float)
    elif isinstance(arg, tuple) and len(arg) == 2:
        start, stop = arg
        if not (isinstance(start, int) and isinstance(stop, int)):
            raise TypeError("Tuple frequencies must be integers.")
        if start < 0 or stop <= start:
            raise ValueError(
                f"Tuple frequencies must satisfy 0 <= start < stop. "
                f"Start is ``{start}`` and stop is ``{stop}`` instead."
            )
        return jnp.arange(start, stop, dtype=float)
    else:
        raise TypeError("Each frequency must be an int or a 2-element tuple of ints.")


class FourierBasis(AtomicBasisMixin, Basis):

    _is_complex = True

    def __init__(
        self,
        frequencies: (
            int | Tuple[int, int] | List[int] | List[Tuple[int, int]] | ArrayLike
        ),
        frequency_mask: jnp.ndarray | None = None,
        label: Optional[str] = None,
    ) -> None:
        """
        N-dimensional Fourier basis for feature expansion.

        This class generates a set of sine and cosine basis functions defined over
        an ``n``-dimensional input space. The basis functions are constructed from
        a Cartesian product of frequencies specified for each input dimension.
        Each selected frequency combination contributes two basis functions
        (cosine and sine), except for the all-zero frequency (DC component),
        which contributes only a cosine term.

        The class supports flexible frequency specification (integers, ranges, or
        lists per dimension) and optional masking to include or exclude specific
        frequency combinations.

        Parameters
        ----------
        frequencies : int | tuple[int, int] | list[int] | list[tuple[int, int]]
            Frequencies to use along each input dimension.

            - ``int``: creates a range ``[0, 1, ..., n-1]`` for **all dimensions**.
            - ``tuple[int, int]``: creates a range ``[start, ..., stop-1]`` for **all dimensions**.
            - ``list[int]``: list of integers giving the number of frequencies per dimension.
            - ``list[tuple[int, int]]``: list of (start, stop) tuples specifying ranges per dimension.

        frequency_mask : array-like of {0, 1}, optional
            Boolean mask specifying which frequency combinations to include.
            If ``None``, all possible frequency combinations are used. The mask
            must have shape ``(len(frequencies[0]), len(frequencies[1]), ...)``.

        label : str, optional
            Descriptive label for the basis (e.g., to use in plots or summaries).

        Notes
        -----
        - If ``frequency_mask`` is provided, only the selected frequency
          combinations are used to build the basis.
        - The output of ``compute_features`` contains both cosine and sine components for
          each active frequency combination, except that the all-zero frequency
          includes only a cosine term.
        """
        self.frequencies = frequencies
        self.frequency_mask = frequency_mask

        AtomicBasisMixin.__init__(
            self,
            n_basis_funcs=self.n_basis_funcs,
            label=label,
        )

    @property
    def frequency_mask(self) -> jnp.ndarray | None:
        """Get or set the frequency mask for the Fourier basis.

        The frequency mask is a boolean array (or array-like of 0s and 1s)
        that specifies which frequencies to include when evaluating the basis.
        Its shape must match the number of frequencies along each input dimension.

        - If ``None``, all possible frequency combinations are included.
        - If provided, entries set to 1 (``True``) enable the corresponding frequency
          combination, while 0 (``False``) disables it.

        Returns
        -------
        :
            A boolean JAX array indicating the selected frequencies, or ``None``
            if no mask is applied.
        """
        # safe get when getter is called at init initialization
        return getattr(self, "_frequency_mask", None)

    @frequency_mask.setter
    def frequency_mask(self, values: ArrayLike | jnp.ndarray | None) -> None:
        """Set the frequency mask for the Fourier basis.

        Parameters
        ----------
        values :
            A boolean array (or array-like of 0s and 1s) specifying which
            frequency combinations to include. Must have shape
            ``(len(frequencies[0]), len(frequencies[1]), ...)``.

            - If ``None``, all frequency combinations are included.
            - If provided, each entry set to 1 includes the corresponding
              frequency combination; entries set to 0 exclude it.

        Raises
        ------
        ValueError
            If the array contains values other than 0 or 1, or if the shape
            does not match the expected number of frequencies.
        TypeError
            If ``values`` cannot be converted to a JAX array.

        Notes
        -----
        Setting this property also updates:

        - ``self._frequency_mask``: stores the boolean mask.
        - ``self._n_basis_funcs``: number of active basis functions.
        - ``self._eval_freq``: array of selected frequencies for evaluation.
        """
        if values is None:
            self._frequency_mask = None

            # cache all frequencies
            grids = jnp.meshgrid(
                *[jnp.arange(len(freqs)) for freqs in self._frequencies], indexing="ij"
            )
            idxs = jnp.stack([g.reshape(-1) for g in grids])

        else:
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

            idxs = jnp.stack(jnp.where(self._frequency_mask))

        self._eval_freq = jnp.stack(
            [freqs[idxs[d]] for d, freqs in enumerate(self._frequencies)]
        )
        # used to drop or not the zero phase
        self._has_zero_phase = int(jnp.all(self._eval_freq[:, 0] == 0))

    @property
    def frequencies(self) -> Tuple[jnp.ndarray, ...]:
        return self._frequencies

    @frequencies.setter
    def frequencies(
        self,
        frequencies: (
            int | tuple[int, int] | list[int] | list[tuple[int, int]] | tuple[NDArray]
        ),
    ) -> None:
        ndim = self._n_input_dimensionality

        if isinstance(frequencies, Number) and (frequencies == int(frequencies)):
            frequencies = _to_tuple_of_arange(int(frequencies), ndim)

        elif (
            isinstance(frequencies, tuple)
            and len(frequencies) in [ndim, 2]
            and all(isinstance(f, Number) and (f == int(f)) for f in frequencies)
        ):
            frequencies = _to_tuple_of_arange(tuple(frequencies), ndim)

        elif is_numpy_array_like(frequencies):
            frequencies = _check_and_sort_frequencies(
                *(frequencies for _ in range(ndim)), ndim=ndim
            )

        elif isinstance(frequencies, (tuple, list)) and all(
            is_numpy_array_like(f) for f in frequencies
        ):
            frequencies = _check_and_sort_frequencies(*frequencies, ndim=ndim)

        elif isinstance(frequencies, list):

            if len(frequencies) != self._n_input_dimensionality:
                raise ValueError(
                    "Length of frequencies list must match input dimensionality."
                )

            frequencies = tuple(make_arange(f) for f in frequencies)

        else:
            if isinstance(frequencies, np.ndarray) and ~np.issubdtype(
                frequencies.dtype, np.integer
            ):
                type_string = f"NDArray[{frequencies.dtype}]"
            else:
                type_string = repr(type(frequencies))
            raise TypeError(
                f"Unrecognized type {type_string} for the ``frequencies`` parameter. ``frequencies`` "
                "must be one of:\n\n"
                "  - int\n"
                "  - tuple[int, int]\n"
                "  - list[int]\n"
                "  - list[tuple[int, int]]\n"
                "  - tuple[NDArray[int]]\n"
                "  - NDArray[int]\n\n"
                f"If a list is provided, the list should be of length ``{ndim}``, "
                "one entry for each dimension of the Fourier basis."
            )

        # Skip update if same as current
        current_freqs = getattr(self, "_frequencies", [])
        if len(frequencies) == len(current_freqs) and all(
            np.array_equal(f1, f2) for f1, f2 in zip(current_freqs, frequencies)
        ):
            return

        self._frequencies = frequencies
        if self.frequency_mask is not None:
            warnings.warn(
                "Resetting ``frequency_mask`` to None (all frequencies will be included).\n"
                "To sub-select frequencies, please provide a new ``frequency_mask``.",
                UserWarning,
            )
            self.frequency_mask = None

    @property
    def ndim(self):
        return self._n_input_dimensionality

    @ndim.setter
    def ndim(self, ndim: int) -> None:
        # TODO: Add an exception handling
        self._n_input_dimensionality = ndim

    @property
    def n_basis_funcs(self) -> int | None:
        return 2 * self._eval_freq.size - self._has_zero_phase

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
        out = jnp.concatenate(
            [jnp.cos(angles), jnp.sin(angles[..., self._has_zero_phase :])], axis=1
        )
        return out.reshape(*shape, out.shape[-1])

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
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
            Fourier basis functions, shape (n_samples, n_basis_funcs)
        """
        return super().evaluate_on_grid(*n_samples)

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

    def set_params(self, **params: Any):
        """Set params handling correctly the frequencies and their mask."""
        # if both frequencies and mask are set ignore warning
        if "frequencies" in params and "frequency_mask" in params:
            freq = params.pop("frequencies")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Resetting ``frequency_mask``.*",
                )
                # set first frequencies
                self.frequencies = freq
                # then set everything else (so that the mask is
                # checked against the new frequencies)
                super().set_params(**params)
        else:
            super().set_params(**params)
