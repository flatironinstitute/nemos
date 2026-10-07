"""Module for ND fourier basis class."""

from __future__ import annotations

import math
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
from numpy._typing import NDArray
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import is_at_least_1d_numpy_array_like, support_pynapple
from ..typing import Array, FeatureMatrix
from ..utils import format_repr
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin, BoundedEvalBasisMixin
from ._composition_utils import add_docstring

# A collection of frequency arrays only needs to be indexable/sliceable and
# iterable, hence ``Sequence``; each element is a NumPy or JAX ``Array``.
FreqArrays = Sequence[Array]

FREQUENCY_ERROR_MSGS = {
    "has_negative_values": "The provided frequencies contain negative values. Valid frequencies must be"
    "non-negative integers.",
    "not_int_like": "Some frequency values are not integers. Valid frequencies must be"
    "non-negative integers.",
}


def _check_array_properties(*arrays: ArrayLike):
    is_int_like = all(jnp.all(f == jnp.asarray(f, dtype=int)) for f in arrays)
    is_positive = all(jnp.all(jnp.asarray(f) >= 0) for f in arrays)
    return is_int_like, is_positive


def _format_error_message(
    is_positive: bool,
    is_int_like: bool,
):
    msg = "Invalid frequencies provided. The following issues were detected:\n\n"

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


def _check_and_sort_frequencies(*frequencies) -> List[jnp.ndarray]:
    is_int_like, is_positive = _check_array_properties(*frequencies)

    if is_int_like and is_positive:
        sorted_freqs = [jnp.sort(f).astype(float) for f in frequencies]
        if any(
            not jnp.array_equal(f1, f2) for f1, f2 in zip(frequencies, sorted_freqs)
        ):
            warnings.warn("Unsorted frequencies provided! Frequencies will be sorted.")
        return sorted_freqs

    raise ValueError(_format_error_message(is_positive, is_int_like))


def arange_constructor(arg: NDArray | int | Tuple[int, int]) -> jnp.ndarray:
    """
    Create an array of frequencies from different types of input.

    This function accepts either:
    - An array-like object of frequencies (returned after sorting and checking),
    - A single positive integer (returns an array from 0 to `arg - 1`),
    - A tuple of two integers `(start, stop)` (returns an array from `start` to `stop - 1`).

    Parameters
    ----------
    arg :
        Specifies how to create the frequency array:
        * If array-like, it is validated and sorted before returning.
        * If int, must be > 0, and the result is `jnp.arange(arg, dtype=float)`.
        * If tuple of two ints `(start, stop)`, must satisfy `0 <= start < stop`,
        and the result is `jnp.arange(start, stop, dtype=float)`.

    Returns
    -------
    :
        A 1D array of frequencies as floats.

    Raises
    ------
    ValueError
        If the integer input is not positive, or if the tuple does not satisfy
        `0 <= start < stop`.
    TypeError
        If the tuple elements are not integers or if the input is of an unsupported type.

    Notes
    -----
    If an array-like input is provided, it is first validated and sorted using
    `_check_and_sort_frequencies`.

    Examples
    --------
    >>> arange_constructor(3)
    Array([0., 1., 2.], dtype=float32)

    >>> arange_constructor((2, 5))
    Array([2., 3., 4.], dtype=float32)

    >>> arange_constructor(jnp.array([3, 1, 2]))
    Array([1., 2., 3.], dtype=float32)
    """
    if is_at_least_1d_numpy_array_like(arg):
        arg = _check_and_sort_frequencies(arg)
        return arg[0]

    if isinstance(arg, int):
        if arg < 0:
            raise ValueError(
                f"Integer frequencies must be >= 0, {arg} provided instead."
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


def _process_tuple_frequencies(
    frequencies: Tuple[int, int], ndim: int
) -> List[jnp.ndarray]:
    """
    Process a tuple of frequencies and return the corresponding frequency arrays.

    The tuple must be a 2-element tuple of integers, i.e. both elements must be
    numeric and equal to their integer representation. It is interpreted as a
    single range specification and broadcast to all ``ndim`` dimensions.

    Parameters
    ----------
    frequencies :
        The input tuple specifying frequencies. It must be a 2-element tuple of integers.

    ndim :
        The number of input dimensions the range is broadcast to.

    Returns
    -------
    :
        A list of ``ndim`` frequency arrays, one per input dimension.

    Raises
    ------
    ValueError
        If the tuple does not match the expected format.

    Examples
    --------
    >>> _process_tuple_frequencies((2, 5), 1)
    [Array([2., 3., 4.], dtype=float32)]

    >>> _process_tuple_frequencies((2, 5), 2)
    [Array([2., 3., 4.], dtype=float32), Array([2., 3., 4.], dtype=float32)]

    """
    if len(frequencies) == 2 and all(
        isinstance(f, Number) and (f == int(f)) for f in frequencies
    ):
        return [arange_constructor(frequencies)] * ndim

    raise ValueError(
        "Invalid frequencies specification. If ``frequencies`` are provided as a tuple, "
        f"it must be a 2-element tuple of non-negative integers. Tuple {frequencies} provided instead."
    )


def _get_all_frequency_pairs(frequencies):
    grids = jnp.meshgrid(*[freqs for freqs in frequencies], indexing="ij")
    return jnp.stack([g.reshape(-1) for g in grids])


def _get_frequency_pairs_from_callable(
    frequency_mask: Callable[..., bool], frequencies: List[jnp.ndarray]
):
    """
    Apply the callable assigned to `frequency_mask` to all frequency tuples.

    Parameters
    ----------
    frequency_mask :
        A function with signature: frequency_mask(*freqs) -> bool (or 0/1).
    frequencies :
        Non-negative 1D frequency arrays (one per dimension).

    Returns
    -------
    :
        Shape (D, K): columns are the selected frequency tuples, taken from the
        half-space combinations (DC first when kept).
    """
    combinations = _combinator_builder(frequencies)  # (D, K), DC first

    selected = []
    for j in range(combinations.shape[1]):
        freqs = np.asarray(combinations[:, j]).tolist()
        try:
            include = frequency_mask(*freqs)
        except Exception as e:
            raise TypeError(
                "Error while applying the callable assigned to `frequency_mask`.\n"
                "Expected signature: frequency_mask(*frequencies) -> bool.\n"
                f"Failed at index {j} with frequencies={freqs!r}."
            ) from e

        # Normalize/validate the result to a single boolean
        is_valid_number = isinstance(include, (Number, np.bool_)) and include in (0, 1)
        is_valid_array = (
            isinstance(include, (jnp.ndarray, np.ndarray))
            and include.size == 1
            and include in (0, 1)
        )
        if is_valid_number or is_valid_array:
            include = bool(include)
        else:
            string_vals = ", ".join(str(f) for f in freqs)
            raise ValueError(
                "`frequency_mask(*freqs)` must return a single boolean or 0/1.\n"
                f"``frequency_mask({string_vals})`` returned {include!r} "
                f"of type {type(include).__name__}."
            )

        if include:
            selected.append(j)

    if selected:
        return combinations[:, jnp.asarray(selected)]
    return jnp.zeros((len(frequencies), 0), dtype=int)


def _signed_box_axes(freq_arrays: FreqArrays) -> List[Array]:
    """Mirror every axis except the first to include negative frequencies.

    The first axis stays non-negative; every other axis is extended symmetrically
    with the negatives of its positive entries (the zero/DC entry, if present, is
    not duplicated). Keeping the first axis non-negative is what lets
    :func:`_half_space_selection` pick exactly one of each ``{J, -J}`` pair later.

    Parameters
    ----------
    freq_arrays:
        Non-negative, ascending frequency arrays, one per dimension.

    Returns
    -------
    :
        Per-axis arrays defining the signed bounding box of the half-space. The
        first axis is unchanged; each remaining axis runs from ``-max`` to ``max``.
    """
    return [freq_arrays[0]] + [
        np.concatenate([-f[f > 0][::-1], f]) for f in freq_arrays[1:]
    ]


def _half_space_selection(grid: Array) -> NDArray:
    """Keep one frequency from each redundant ``{J, -J}`` pair.

    A frequency multi-index ``J`` and its negation ``-J`` produce the same pair
    of Fourier features (cosine is even, sine is odd), so only one of each pair
    is kept: the origin, or whichever of ``J``/``-J`` has its first non-zero
    coordinate positive (scanning coordinates left to right). For example, of
    ``(0, 1)`` and ``(0, -1)`` it keeps ``(0, 1)``; of ``(1, -2)`` and
    ``(-1, 2)`` it keeps ``(1, -2)``. The retained set is known formally as the
    lexicographically positive half-space.

    Parameters
    ----------
    grid:
        Array of shape ``(ndim, n_combinations)`` whose columns are frequency
        multi-indices.

    Returns
    -------
    :
        Boolean array of shape ``(n_combinations,)``, ``True`` for the columns to
        keep.
    """
    non_zero = grid != 0
    first_nonzero = np.argmax(non_zero, axis=0)
    first_val = grid[first_nonzero, np.arange(grid.shape[1])]
    return (~non_zero.any(axis=0)) | (first_val > 0)


def _move_dc_first(freq_combinations: Array) -> Array:
    """Move the all-zero (DC) combination to the first column when present.

    ``evaluate`` and ``_has_zero_phase`` assume the DC term is the first column,
    because its sine contribution is identically zero and is dropped.

    Parameters
    ----------
    freq_combinations:
        Array of shape ``(ndim, n_combinations)``.

    Returns
    -------
    :
        The same columns with the DC column first, when present.
    """
    is_dc = np.all(freq_combinations == 0, axis=0)
    if is_dc.any():
        order = np.concatenate([np.flatnonzero(is_dc), np.flatnonzero(~is_dc)])
        freq_combinations = freq_combinations[:, order]
    return freq_combinations


def _combinator_builder(freq_arrays: FreqArrays) -> Array:
    """Generate the half-space frequency combinations from non-negative axes.

    Mirrors every axis but the first to a signed bounding box, takes the full
    Cartesian product, and keeps one representative of each redundant ``J``/``-J``
    pair (see :func:`_half_space_selection`). The DC term, if present, is placed
    first.

    Parameters
    ----------
    freq_arrays:
        Non-negative, ascending frequency arrays, one per dimension.

    Returns
    -------
    :
        Array of shape ``(ndim, n_combinations)`` whose columns are the retained
        frequency multi-indices.
    """
    grid = _get_all_frequency_pairs(_signed_box_axes(freq_arrays))
    combinations = grid[:, _half_space_selection(grid)]
    return _move_dc_first(combinations)


class FourierBasis(AtomicBasisMixin, Basis):
    _is_complex = True

    def __init__(
        self,
        ndim: int,
        freq_combinations: jnp.ndarray,
        weights: Optional[ArrayLike] = None,
        label: Optional[str] = None,
    ) -> None:
        self._n_inputs = self._check_ndim(ndim)
        self._set_fourier_params(freq_combinations, weights)
        Basis.__init__(
            self,
        )
        AtomicBasisMixin.__init__(self, n_basis_funcs=self.n_basis_funcs, label=label)

    @property
    def ndim(self):
        """The dimensionality of the basis."""
        return self._n_inputs

    @staticmethod
    def _check_ndim(ndim: int) -> int:
        try:
            is_int = int(ndim) == ndim
        except Exception as e:
            raise TypeError(f"Cannot convert ndim {ndim!r} to type int.") from e
        is_positive = ndim > 0
        if not is_int or not is_positive:
            raise ValueError(
                f"ndim must be a positive integer. {ndim!r} provided instead."
            )
        return int(ndim)

    @property
    def weights(self) -> Optional[jnp.ndarray]:
        """Per-output-column weights applied inside :meth:`evaluate`.

        ``None`` when no weights were provided at construction, in which case no
        reweighting is applied.
        """
        return self._weights

    def _set_fourier_params(
        self, freq_combinations: jnp.ndarray, weights: Optional[ArrayLike]
    ) -> None:
        freq_combinations = jnp.asarray(freq_combinations, dtype=float)
        if freq_combinations.ndim != 2:
            raise ValueError(
                "``freq_combinations`` must be 2D with shape "
                f"({self._n_inputs}, n_combinations); "
                f"got an array with {freq_combinations.ndim} axes instead."
            )
        if freq_combinations.shape[0] != self._n_inputs:
            raise ValueError(
                "``freq_combinations`` must have the same number of rows as the input dimension "
                f"(``ndim`` = {self._n_inputs}); "
                f"got {freq_combinations.shape[0]} instead."
            )
        if (freq_combinations.shape[-1] > 0) and (
            jnp.all(freq_combinations[:, 0] == 0)
        ):
            zero_phase_flag = 1
        else:
            zero_phase_flag = 0
        n_basis_functions = 2 * freq_combinations.shape[-1] - zero_phase_flag
        if weights is not None:
            weights = jnp.asarray(weights, dtype=float)
            if weights.ndim != 1:
                raise ValueError(
                    "``weights`` must be 1D with shape "
                    f"({n_basis_functions}, ); "
                    f"got an array with {weights.ndim} axes instead."
                )
            if weights.shape[0] != n_basis_functions:
                raise ValueError(
                    "``weights`` must have one entry per basis function "
                    f"(``n_basis_funcs`` = {n_basis_functions}); "
                    f"got {weights.shape[0]} instead."
                )
        self._freq_combinations = freq_combinations
        self._has_zero_phase = zero_phase_flag
        self._weights = weights
        self._n_basis_funcs = n_basis_functions

    @property
    def freq_combinations(self) -> jnp.ndarray:
        return self._freq_combinations

    @property
    def has_zero_phase(self) -> int:
        return self._has_zero_phase

    @property
    def n_basis_funcs(self) -> int:
        return 2 * self._freq_combinations.shape[-1] - self._has_zero_phase

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
        bounds = self._get_bounds_per_dim()

        # min/max rescale to [0,1]:
        # The function does so over the time axis (each extra dim is
        # normalized independently)
        def _flat_samples_to_angles(xs):
            scaled_samples = jax.tree_util.tree_map(
                lambda x, b: (
                    2
                    * jnp.pi
                    * self._shift_angles(min_max_rescale_samples(x, b)[0].reshape(-1))
                ),
                xs,
                bounds,
            )
            return jnp.stack(scaled_samples, axis=-1)

        sample_pts = _flat_samples_to_angles(list(sample_pts))
        angles = sample_pts @ self._freq_combinations
        out = jnp.concatenate(
            [jnp.cos(angles), jnp.sin(angles[..., self._has_zero_phase :])], axis=1
        )
        if self._weights is not None:
            out = out * self._weights
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

    def __repr__(self):
        return format_repr(self, exclude_keys=["fill_value"])


class FourierEval(BoundedEvalBasisMixin, FourierBasis):
    """
    N-dimensional Fourier basis for feature expansion.

    This class generates a set of sine and cosine basis functions defined over
    an ``n``-dimensional input space. The basis functions are constructed from
    a Cartesian product of frequencies specified for each input dimension.
    Each selected frequency combination contributes two basis functions
    (cosine and sine), except for the all-zero frequency (DC component),
    which contributes only a cosine term.

    The class supports flexible frequency specification (integers, ranges, or
    arrays per dimension) and optional masking to include or exclude specific
    frequency combinations.

    Parameters
    ----------
    frequencies :
        Frequency specification(s).

        Single specification (broadcasted to all dimensions when ``ndim > 1``):

            * :class:`int`: An integer ``k`` with ``k >= 0``.

            * :class:`tuple`: ``(low, high)``, a 2-element tuple of integers with ``0 <= low < high``.

            * :class:`~numpy.ndarray`: 1-D NumPy array of non-negative integers. If not sorted ascending,
              a ``UserWarning`` is issue for non-sorted arrays.

        Per-dimension container:

            * A :class:`list` of length ``ndim`` whose elements are each a valid single specification.
              For ``ndim == 1``, a length-1 :class:`list` is also accepted.

    ndim :
        Dimensionality of the basis. Default is 1.

    bounds :
        Period bounds for each dimension. Unlike other basis classes where bounds define
        a valid domain (with out-of-bounds samples filled with NaN), for the Fourier basis
        the bounds define the period of the basis functions. Samples outside these bounds
        are still valid and will be evaluated using the periodic nature of the basis.

        * :class:`tuple`: ``(low, high)`` of floats: applies to all dimensions.
        * :class:`list` of :class:`tuple`: ``[(low, high), ...]``, one tuple per dimension,
        length must match ``ndim``.
        * :class:`None <NoneType>`: the period is inferred from the input data (minimum to maximum values).

        In all cases, ``low`` must be strictly less than ``high``, and values must be convertible to floats.

    frequency_mask :
        Optional mask specifying which frequency components to include.
        Can be:

        * :class:`~typing.Literal`: either ``"no-intercept"`` - default - which drops
          the 0-frequency DC term, or ``"all"`` which keeps all the frequencies -
          equivalent to :class:`None <NoneType>`. The default excludes the intercept
          because these basis objects are most commonly used to generate design matrices
          for NeMoS GLMs, which already include an intercept term by default, making an
          additional intercept in the design matrix redundant.

        * Array-like of integers {0, 1} or booleans: A 1D mask with one entry
          per column of ``masked_frequencies``, keeping (1/True) or dropping
          (0/False) that frequency combination. At construction it filters the
          combinations left by the default ``"no-intercept"`` selection; print
          ``masked_frequencies`` to see the combinations in column order.

        * :class:`~typing.Callable`: A function applied to each retained frequency
          combination (one scalar per dimension, signed), returning a single
          boolean or {0, 1} indicating whether to keep that frequency.

        * :class:`None <NoneType>`: All frequencies are kept.

        Values must be 0/1 or boolean. Callables must return a single boolean or
        {0, 1} value for each frequency coordinate.

    weights :
        Optional per-output-column weights, with one entry per basis function
        (``n_basis_funcs``), applied inside ``evaluate``. The columns are ordered
        as the cosine terms followed by the sine terms. When ``None`` (default),
        no reweighting is applied.

    label :
        Descriptive label for the basis (e.g., to use in plots or summaries).

    Notes
    -----
    - If ``frequency_mask`` is provided, only the selected frequency
      combinations are used to build the basis.
    - The output of ``compute_features`` contains both cosine and sine components for
      each active frequency combination, except that the all-zero frequency
      includes only a cosine term.
    - When a :class:`tuple` is provided as a frequency, it is interpreted
      as a single range specification. Tuples that are not exactly a 2-element
      tuple of non-negative integers are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import FourierEval
    >>> rng = np.random.default_rng(0)

    **1D: basic usage**

    >>> n_freq = 5
    >>> fourier_1d = FourierEval(n_freq)
    >>> # cos at 0..4 (5) + sin at 1..4 (4) = 9
    >>> fourier_1d.n_basis_funcs
    8
    >>> x = rng.normal(size=8)
    >>> X = fourier_1d.compute_features(x)
    >>> X.shape  # (n_samples, n_basis_funcs)
    (8, 8)

    **2D: unmasked grid of frequency pairs**

    >>> fourier_2d = FourierEval(n_freq, ndim=2)
    >>> # half-space of the 5x5 grid has 41 pairs (incl. DC); DC dropped -> 40; *2
    >>> fourier_2d.n_basis_funcs
    80
    >>> x, y = rng.normal(size=(2, 6))
    >>> X = fourier_2d.compute_features(x, y)
    >>> X.shape
    (6, 80)

    **2D: masking with an array (drop 3 pairs)**

    >>> # one mask entry per retained pair: the default "no-intercept"
    >>> # selection drops the DC and leaves 40 pairs
    >>> fourier_2d.masked_frequencies.shape
    (2, 40)
    >>> mask = np.ones(40)
    >>> mask[:3] = 0  # drop the first 3 pairs
    >>> fourier_2d_masked = FourierEval(n_freq, ndim=2, frequency_mask=mask)
    >>> # (40 pairs - 3 dropped) * 2 (cos+sin) = 74
    >>> fourier_2d_masked.n_basis_funcs
    74

    **2D: masking with a callable**

    >>> # keep pairs inside a circle of radius 3.5 in frequency space
    >>> keep_circle = lambda fx, fy: (fx**2 + fy**2) ** 0.5 < 3.5
    >>> fourier_2d_funcmask = FourierEval(n_freq, ndim=2, frequency_mask=keep_circle)
    >>> fourier_2d_funcmask.n_basis_funcs
    37

    **Explicit frequency specifications**

    >>> # mix forms per-dimension: an explicit array
    >>> # and an inclusive tuple (low, high)
    >>> fourier_mixed = FourierEval(frequencies=[np.arange(3), (1, 4)], ndim=2)
    >>> # 15 half-space pairs (no DC, since the y-axis omits 0) -> 2*15 = 30
    >>> fourier_mixed.n_basis_funcs
    30

    """

    # Fourier basis is defined over the entire real line; out-of-bounds
    # samples should not be filled with a sentinel value.
    _apply_bounds_fill = False

    def __init__(
        self,
        frequencies: (
            int
            | Tuple[int, int]
            | List[int]
            | List[Tuple[int, int]]
            | NDArray
            | List[NDArray]
        ),
        ndim: int = 1,
        bounds: Optional[Tuple[float, float] | Tuple[Tuple[float, float]]] = None,
        frequency_mask: (
            Literal["all", "no-intercept"] | jnp.ndarray | None
        ) = "no-intercept",
        weights: Optional[ArrayLike] = None,
        label: Optional[str] = "FourierEval",
    ) -> None:
        self._n_inputs = super()._check_ndim(ndim)
        self.frequencies = frequencies
        self.frequency_mask = frequency_mask
        self.weights = weights
        FourierBasis.__init__(
            self,
            ndim=ndim,
            freq_combinations=self.masked_frequencies,
            weights=self.weights,
            label=label,
        )
        BoundedEvalBasisMixin.__init__(self, bounds=bounds)

    @property
    def frequency_mask(self) -> Callable | jnp.ndarray | Literal["all", "no-intercept"]:
        """Get the frequency mask for the Fourier basis.

        The frequency mask can be either:

        - a 1D boolean array with one entry per column of the ``masked_frequencies``
          it was applied to, or
        - a callable with signature ``frequency_mask(*freqs) -> bool`` (or 0/1) applied
          to each frequency tuple, or
        - a string, either ``"all"``, if all possible frequency combinations are included, or
          ``"no-intercept"``, if the intercept term is dropped (DC component).

        Returns
        -------
        :
            The string, callable, or boolean JAX array mask that was last
            assigned (``None`` is stored as ``"all"``).
        """
        # safe get when getter is called at init initialization
        return getattr(self, "_frequency_mask", "no-intercept")

    @frequency_mask.setter
    def frequency_mask(
        self,
        values: (
            Literal["all", "no-intercept"]
            | ArrayLike
            | jnp.ndarray
            | Callable[..., bool]
            | None
        ),
    ) -> None:
        """Set the frequency mask for the Fourier basis.

        Parameters
        ----------
        values :
            One of:
            - :class:`Literal <typing.Literal>`: either `"no-intercept"`` - default - which drops
              the 0-frequency DC term, or ``"all"`` which keeps all the frequencies -
              equivalent to :class:`None <NoneType>`.
            - **Array / array-like (bool or 0/1)**: a 1D mask with one entry per
              column of the current ``masked_frequencies``; the retained columns
              become the new ``masked_frequencies``.
            - :class:`callable`: A function with signature
              ``frequency_mask(*freqs) -> bool`` (or 0/1). It is applied to each
              frequency tuple ``(f1, f2, ..., f_n)`` to build the mask. The callable is
              **not required to be vectorized**; ``n`` is the input dimensionality.
            - :class:`None <NoneType>`: Include all frequency combinations.

        Raises
        ------
        ValueError
            If an array mask has values other than {0, 1}/booleans, or if its
            length does not match the current number of frequency combinations.
        TypeError
            If ``values`` is neither array-like, callable, nor ``None``; or if
            the callable returns a value that is not a single boolean or 0/1.

        Notes
        -----
        - Setting this property updates ``frequency_mask`` and
          ``masked_frequencies`` (and, through the latter, ``n_basis_funcs``).
        - An array mask filters the *current* ``masked_frequencies``: assigning a
          second array mask filters the already-filtered combinations. Assign
          ``"all"`` or ``"no-intercept"`` to start from the full half-space again.
        """
        if isinstance(values, str) and values == "no-intercept":
            mask = "no-intercept"
            combinations = _combinator_builder(self._frequencies)
            # the DC term, if present, is the first column; drop it.
            if combinations.shape[1] and jnp.all(combinations[:, 0] == 0):
                combinations = combinations[:, 1:]

        elif values is None or isinstance(values, str) and values == "all":
            mask = "all"
            combinations = _combinator_builder(self._frequencies)

        elif callable(values):
            combinations = _get_frequency_pairs_from_callable(values, self._frequencies)
            mask = values
        else:
            mask, combinations = self._set_array_frequency_mask(values)

        if getattr(self, "_weights", None) is not None:
            warnings.warn(
                "Resetting ``weights`` to ``None``. \n"
                "To re-weight chosen frequencies, please provide new ``weights``.",
                UserWarning,
            )
        self._set_fourier_params(combinations, None)
        self._frequency_mask = mask

    def _set_array_frequency_mask(self, values: ArrayLike):
        """Validate a boolean array mask and filter the frequency combinations.

        The mask is checked for 0/1 values and for shape ``(K,)``, with ``K``
        the current number of columns of ``masked_frequencies``. The columns
        where the mask is ``True`` become the new ``_freq_combinations``, and
        the boolean mask is stored as ``_frequency_mask``.
        """
        try:
            values = jnp.asarray(values)
        except Exception as e:
            raise ValueError(
                f"``frequency_mask`` {values} cannot be converted to a jax array of boolean."
            ) from e

        if not jnp.all((values == 0) | (values == 1)):
            raise ValueError("Frequency mask must be an array-like of 0s and 1s.")

        values = values.astype(bool)
        expected_len = self.masked_frequencies.shape[1]
        if not values.shape == (expected_len,):
            raise ValueError(
                f"Mis-shaped ``frequency_mask``. An array mask must have shape "
                f"``({expected_len},)``, one entry per frequency combination: "
                f"entry ``i`` keeps (1/True) or drops (0/False) the combination "
                f"``masked_frequencies[:, i]``. Print the ``masked_frequencies`` "
                "attribute to see the combinations currently included."
            )

        mask = values
        combinations = self._freq_combinations[:, values]
        return mask, combinations

    @property
    def frequencies(self) -> List[jnp.ndarray]:
        """Frequencies for the basis.

        Returns
        -------
        :
            A tuple of arrays with the fourier frequencies, one per
            dimension of the basis.
        """
        return self._frequencies

    @frequencies.setter
    def frequencies(
        self,
        frequencies: int | tuple[int, int] | list[int] | list[tuple[int, int]],
    ) -> None:
        ndim = self._n_inputs

        if isinstance(frequencies, Number) and (frequencies == int(frequencies)):
            frequencies = [arange_constructor(frequencies)] * ndim

        elif isinstance(frequencies, tuple):
            frequencies = _process_tuple_frequencies(frequencies, self._n_inputs)

        elif is_at_least_1d_numpy_array_like(frequencies):
            frequencies = _check_and_sort_frequencies(*([frequencies] * ndim))

        elif isinstance(frequencies, list):
            if len(frequencies) != self._n_inputs:
                raise ValueError(
                    "Length of frequencies list must match input dimensionality."
                )
            frequencies = [arange_constructor(f) for f in frequencies]

        else:
            if isinstance(frequencies, (np.ndarray, jnp.ndarray)) and not np.issubdtype(
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
                "  - NDArray[int]\n"
                "  - list[int | tuple[int, int] | NDArray[int] | NDArray[int]]\n\n"
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

        if not isinstance(self.frequency_mask, str):
            warnings.warn(
                "Resetting ``frequency_mask`` to ``'no-intercept'`` (all frequencies "
                "except the intercept - DC term - will be included).\n"
                "To sub-select frequencies, please provide a new ``frequency_mask``.",
                UserWarning,
            )
            self.frequency_mask = "no-intercept"
        else:
            # call the setter to re-calculate frequency pairs
            self.frequency_mask = self.frequency_mask

    @property
    def masked_frequencies(self) -> jnp.ndarray:
        """
        The retained frequency combinations.

        Returns
        -------
        :
            The retained frequency combinations, shape
            ``(ndim, n_frequency_combinations)``. Column ``i`` is the frequency
            multi-index of the i-th combination; each column contributes a
            cosine and a sine feature (cosine only for the DC term). Read-only:
            assign ``frequency_mask`` to change it.

        """
        return self._freq_combinations

    @FourierBasis.weights.setter
    def weights(self, values) -> None:
        self._set_fourier_params(self._freq_combinations, values)

    def set_params(self, **params: Any):
        """Set params handling correctly the frequencies and their mask."""
        has_weights = "weights" in params
        weights = params.pop("weights", None)
        with warnings.catch_warnings():
            # if both frequencies and mask are set ignore warning
            if "frequencies" in params and "frequency_mask" in params:
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Resetting ``frequency_mask``.*",
                )
            if has_weights:
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message="Resetting ``weights``.*"
                )
            # check for frequencies first
            if "frequencies" in params:
                self.frequencies = params.pop("frequencies")
            # then set everything else (so that the mask is
            # checked against the new frequencies)
            super().set_params(**params)
        # only set weights once everything else has been set
        if has_weights:
            self.weights = weights
        return self

    @add_docstring("evaluate_on_grid", FourierBasis)
    def evaluate_on_grid(self, *n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: FourierEval

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import FourierEval
            >>> n_frequencies = 5
            >>> fourier_basis = FourierEval(n_frequencies)
            >>> sample_points, basis_values = fourier_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()
        """
        return super().evaluate_on_grid(*n_samples)

    @add_docstring("_compute_features", BoundedEvalBasisMixin)
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierEval

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.normal(size=(num_samples,))  # raw time series
        >>> basis = FourierEval(10)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 18)

        """
        return super().compute_features(*xi)

    @add_docstring("split_by_feature", FourierBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierEval
        >>> from nemos.glm import GLM
        >>> basis = FourierEval(6, label="one_input")
        >>> X = basis.compute_features(
        ...     np.random.randn(
        ...         20,
        ...     )
        ... )
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...     print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 10)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, *xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.FourierEval(5)
        >>> # Configure with an integer input:
        >>> _ = basis.set_input_shape(3)
        >>> basis.n_output_features
        24
        >>> # Configure with a tuple:
        >>> _ = basis.set_input_shape((4, 5))
        >>> basis.n_output_features
        160
        >>> # Configure with an array:
        >>> x = np.ones((10, 4, 5))
        >>> _ = basis.set_input_shape(x)
        >>> basis.n_output_features
        160

        """
        return super().set_input_shape(*xi)

    @add_docstring("evaluate", FourierBasis)
    def evaluate(self, *sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierEval
        >>> basis = FourierEval(4)
        >>> out = basis.evaluate(np.random.randn(100, 5, 2))
        >>> out.shape
        (100, 5, 2, 6)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(*sample_pts)


def _get_nodes_weights(
    lengthscale: float, variance: float, eps: float, L: float
) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """Find the nodes of the equispaced quadrature in Fourier domain.

    Operates on the discretized inverse Fourier transform for a 1-D
    squared-exponential kernel. This involves finding the spacing between
    nodes ``h`` and the number of nodes ``2m + 1``. This is done from a
    formula in [1]_.

    Returns the non-negative frequency grid ``xi_j = j * h`` for
    ``j = 0, 1, ..., m``, the weights corresponding to each column
    (also prior standard deviations from the GP perspective) in
    ``[cos columns, sin columns]`` so that an i.i.d. ``N(0, 1)`` prior
    on coefficients gives an approximate squared-exponential covariance
    kernel. also returns the frequency spacing ``h``, and the number of
    non-negative frequencies, ``m``.

    Parameters
    ----------
    lengthscale, variance :
        se kernel hyperparameters.
    eps :
        error tolerance on kernel approximation
    L :
        time domain length ``t1 - t0``.

    Returns
    -------
    xis :
        non-negative frequencies of shape ``(m + 1,)``.
    weights :
        weights of shape ``(2 * m + 1,)`` corresponding to the columns
        ``[cos columns, sin columns]``: cos columns first
        (``j = 0, 1, ..., m``), then sin columns (``j = 1, ..., m``).
    h :
        spacing between frequencies
    m :
        number of positive frequencies

    References
    ----------
    .. [1] Barnett, A. H., Greengard, P., & Rachh, M. (2024). Uniform
        approximation of common Gaussian process kernels using equispaced Fourier
        grids. Applied and Computational Harmonic Analysis, 71, 101640.
    """
    lengthscale = float(lengthscale)
    var = float(variance)
    eps_use = float(eps) / var

    # Heuristic for h and m
    h = 1.0 / (L + lengthscale * math.sqrt(2.0 * math.log(12.0 / eps_use)))
    m = math.ceil(math.sqrt(math.log(16.0 / eps_use) / 2.0) / math.pi / lengthscale / h)

    j = jnp.arange(m + 1, dtype=float)
    xis = j * h

    # 1d se spectral density: S(xi) = var * sqrt(2*pi*l^2) * exp(-2*pi^2*l^2*xi^2)
    prefactor = var * math.sqrt(2.0 * math.pi * lengthscale**2)
    S = prefactor * jnp.exp(-2.0 * (math.pi * lengthscale) ** 2 * xis**2)
    w = jnp.sqrt(S * h)  # efgp per-mode weight, shape (m + 1,)

    # convert standard efgp xis with +/- modes into a positive-only modes.
    sqrt2 = math.sqrt(2.0)
    w_cos = w.at[1:].multiply(sqrt2)  # cos columns: w_0, sqrt(2)*w_1, ...
    w_sin = sqrt2 * w[1:]  # sin columns: sqrt(2)*w_1, ...
    weights = jnp.concatenate([w_cos, w_sin])
    return xis, weights, h, m


def _grid_params_from_nodes(xis: ArrayLike) -> Tuple[float, int]:
    """Recover the grid spacing ``h`` and max index ``m`` from the node grid.

    The nodes are the non-negative, equispaced frequencies ``xi_j = j * h`` for
    ``j = 0, 1, ..., m`` produced by :func:`_get_nodes_weights`, so the spacing
    is ``h = xis[1] - xis[0]`` and the number of positive frequencies is
    ``m = len(xis) - 1``.

    Parameters
    ----------
    xis :
        Non-negative, equispaced frequency grid, shape ``(m + 1,)``.

    Returns
    -------
    h :
        Spacing between consecutive frequencies.
    m :
        Number of positive frequencies.
    """
    xis = jnp.asarray(xis)
    m = int(xis.shape[0] - 1)
    h = float(xis[1] - xis[0])
    return h, m


class FourierGP(BoundedEvalBasisMixin, FourierBasis):
    """1d Fourier basis with an approximate squared-exponential GP prior.

    Generates ``cos`` and ``sin`` basis functions on a domain ``[t0, t1]``
    whose frequencies and per-column weights are picked so that an i.i.d.
    ``N(0, 1)`` prior on the basis coefficients corresponds to a Gaussian
    process with approximately squared-exponential (SE) covariance:

    .. code-block:: text

        k(r) = variance * exp(-r^2 / (2 * lengthscale^2))

    The equispaced frequency grid for Gaussian processes is from [1]_. Given
    the error tolerance ``eps``, the basis has ``2 * m + 1`` functions: cosines
    at frequencies ``j * h`` for ``j = 0, ..., m`` and sines at ``j * h`` for
    ``j = 1, ..., m``; the spacing ``h`` and node count ``m`` follow the kernel
    approximation bounds in [2]_.

    This basis reuses :class:`FourierBasis`: it is built on the integer
    harmonics ``j = 0, ..., m`` whose effective frequency is rescaled to
    ``j * h`` (see :meth:`_shift_angles`), and the SE spectral density enters
    through the inherited per-column ``weights``.

    Parameters
    ----------
    lengthscale :
        SE kernel lengthscale, in the same units as ``bounds``.
    bounds :
        Pair ``(t0, t1)`` with ``t0 < t1`` defining the construction domain.
    eps :
        kernel approximation error tolerance.
    variance :
        SE kernel variance (prefactor). Default is ``1.0``.
    label :
        descriptive label for the basis. Defaults to the class name.

    References
    ----------
    .. [1] Greengard, P., Rachh, M., & Barnett, A. H. (2025). Equispaced Fourier
        representations for efficient Gaussian process regression from a billion
        data points. SIAM/ASA Journal on Uncertainty Quantification, 13(1).
    .. [2] Barnett, A. H., Greengard, P., & Rachh, M. (2024). Uniform
        approximation of common Gaussian process kernels using equispaced Fourier
        grids. Applied and Computational Harmonic Analysis, 71, 101640.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.basis import FourierGP
    >>> basis = FourierGP(lengthscale=0.2, bounds=(0.0, 1.0), eps=1e-4, variance=2.0)
    >>> basis.n_frequencies
    8
    >>> basis.n_basis_funcs
    17
    >>> x = np.linspace(0, 1, 100)
    >>> X = basis.compute_features(x)
    >>> X.shape  # (n_samples, n_basis_funcs)
    (100, 17)
    """

    def __init__(
        self,
        lengthscale: float,
        bounds: Tuple[float, float],
        eps: float,
        variance: float = 1.0,
        label: Optional[str] = "FourierGP",
    ) -> None:
        ndim = 1
        self._n_inputs = self._check_ndim(ndim)
        self._rebuild_grid(lengthscale, variance, eps, bounds)
        FourierBasis.__init__(
            self,
            ndim=ndim,
            freq_combinations=self.freq_combinations,
            weights=self.weights,
            label=label,
        )
        BoundedEvalBasisMixin.__init__(self, bounds=self.bounds)

    def _rebuild_grid(self, lengthscale, variance, eps, bounds):
        if lengthscale <= 0:
            raise ValueError(f"``lengthscale`` must be positive, got {lengthscale}.")
        if variance <= 0:
            raise ValueError(f"``variance`` must be positive, got {variance}.")
        if eps <= 0:
            raise ValueError(f"``eps`` must be positive, got {eps}.")
        if bounds is None:
            raise ValueError("``bounds`` must not be ``None``.")
        BoundedEvalBasisMixin.bounds.fset(self, bounds)
        ((t0, t1),) = self._get_bounds_per_dim()
        xis, weights, _, _ = _get_nodes_weights(lengthscale, variance, eps, t1 - t0)
        self._set_fourier_params(xis.reshape(self._n_inputs, -1), weights)
        self._lengthscale = lengthscale
        self._variance = variance
        self._eps = eps
        self._xis = xis

    @property
    def lengthscale(self) -> float:
        """SE kernel lengthscale."""
        return self._lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        value = float(value)
        self._rebuild_grid(value, self._variance, self._eps, self._bounds)

    @property
    def variance(self) -> float:
        """SE kernel variance (prefactor)."""
        return self._variance

    @variance.setter
    def variance(self, value):
        value = float(value)
        self._rebuild_grid(self._lengthscale, value, self._eps, self._bounds)

    @property
    def eps(self) -> float:
        """Kernel error approximation tolerance."""
        return self._eps

    @eps.setter
    def eps(self, value):
        value = float(value)
        self._rebuild_grid(self._lengthscale, self._variance, value, self._bounds)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, values):
        self._rebuild_grid(self._lengthscale, self._variance, self._eps, values)

    @property
    def xis(self) -> jnp.ndarray:
        """Non-negative frequencies ``j * h`` for ``j = 0, ..., m``."""
        return self._xis

    @property
    def frequency_spacing(self) -> float:
        """Frequency spacing, recovered from :attr:`xis`."""
        return _grid_params_from_nodes(self.xis)[0]

    @property
    def n_frequencies(self) -> int:
        """Number of positive frequencies ``m``.

        The underlying grid :attr:`xis` holds ``m + 1`` non-negative
        frequencies ``j * h`` for ``j = 0, ..., m``; this excludes the
        ``j = 0`` (DC) term, so ``n_frequencies == len(xis) - 1``.
        The basis has ``2 * n_frequencies + 1`` functions.
        """
        return len(self.xis) - 1

    def _shift_angles(self, sample_pts: ArrayLike) -> ArrayLike:
        """Rescale ``[0, 1]`` samples so frequency ``j`` evaluates at ``j * h``.

        :meth:`FourierBasis.evaluate` maps samples to
        ``2 * pi * _shift_angles(x_scaled)`` and multiplies by the frequency
        ``xi``.
        """
        t0, t1 = self.bounds
        return sample_pts * (t1 - t0)

    def _get_samples(self, *n_samples: int) -> Generator[NDArray, None, None]:
        """Produce equispaced samples over the construction domain."""
        t0, t1 = self.bounds
        return (np.linspace(t0, t1, n_samples[0]),)

    @add_docstring("evaluate_on_grid", FourierBasis)
    def evaluate_on_grid(self, *n_samples: int) -> Tuple[NDArray, NDArray]:
        """
        Examples
        --------
        .. plot::
            :include-source: True
            :caption: FourierGP

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from nemos.basis import FourierGP
            >>> gp_basis = FourierGP(lengthscale=0.2, bounds=(0.0, 1.0), eps=1e-4)
            >>> sample_points, basis_values = gp_basis.evaluate_on_grid(100)
            >>> plt.plot(sample_points, basis_values)
            [<matplotlib.lines.Line2D object at ...
            >>> plt.show()
        """
        return super().evaluate_on_grid(*n_samples)

    @add_docstring("_compute_features", BoundedEvalBasisMixin)
    def compute_features(self, *xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierGP

        >>> # Generate data
        >>> num_samples = 1000
        >>> X = np.random.uniform(size=(num_samples,))  # raw time series
        >>> basis = FourierGP(lengthscale=0.2, bounds=(0.0, 1.0), eps=1e-4)
        >>> features = basis.compute_features(X)  # basis transformed time series
        >>> features.shape
        (1000, 17)

        """
        return super().compute_features(*xi)

    @add_docstring("split_by_feature", FourierBasis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierGP
        >>> basis = FourierGP(
        ...     lengthscale=0.2, bounds=(0.0, 1.0), eps=1e-4, label="one_input"
        ... )
        >>> X = basis.compute_features(np.random.uniform(size=(20,)))
        >>> split_features_multi = basis.split_by_feature(X, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...     print(f"{feature}, shape {sub_dict.shape}")
        one_input, shape (20, 17)

        """
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, *xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.FourierGP(lengthscale=0.2, bounds=(0.0, 1.0), eps=1e-4)
        >>> # Configure with an integer input:
        >>> _ = basis.set_input_shape(3)
        >>> basis.n_output_features
        51
        >>> # Configure with a tuple:
        >>> _ = basis.set_input_shape((4, 5))
        >>> basis.n_output_features
        340
        >>> # Configure with an array:
        >>> x = np.ones((10, 4, 5))
        >>> _ = basis.set_input_shape(x)
        >>> basis.n_output_features
        340

        """
        return super().set_input_shape(*xi)

    @add_docstring("evaluate", FourierBasis)
    def evaluate(self, *sample_pts: NDArray) -> NDArray:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import FourierGP
        >>> basis = FourierGP(lengthscale=0.2, bounds=(0.0, 1.0), eps=1e-4)
        >>> out = basis.evaluate(np.random.uniform(size=(100, 5, 2)))
        >>> out.shape
        (100, 5, 2, 17)
        """
        # ruff: noqa: D205, D400
        return super().evaluate(*sample_pts)
