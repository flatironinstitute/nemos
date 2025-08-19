"""Module for ND fourier basis class."""

import itertools
import warnings
from numbers import Number
from typing import Any, Callable, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy._typing import NDArray
from numpy.typing import ArrayLike
from pynapple import Tsd, TsdFrame, TsdTensor

from ..type_casting import is_at_least_1d_numpy_array_like, support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis, check_transform_input, min_max_rescale_samples
from ._basis_mixin import AtomicBasisMixin

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


def arange_constructor(arg: NDArray | int | Tuple[int, int]):
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


def _process_tuple_frequencies(frequencies: tuple, ndim: int):
    """
    Process a tuple of frequencies and return the corresponding frequency arrays.

    This function handles two valid cases for specifying frequencies as a tuple:

    - A 2-element tuple of integers:
       - Both elements must be numeric and equal to their integer representation.
       - Returns a single-element tuple containing the result of `arange_constructor(frequencies)`.


    Parameters
    ----------
    frequencies :
        The input tuple specifying frequencies. It must be a 2-element tuple of integers.

    ndim :
        The expected number of input dimensions. Used to validate the tuple length
        in the second case.

    Returns
    -------
    :
        A tuple of frequency arrays processed according to the input.

    Raises
    ------
    ValueError
        If the tuple does not match one of the expected formats.

    Examples
    --------
    >>> _process_tuple_frequencies((2, 5), 1)
    [Array([2., 3., 4.], dtype=float32)]

    """
    if len(frequencies) == 2 and all(
        isinstance(f, Number) and (f == int(f)) for f in frequencies
    ):
        return [arange_constructor(frequencies)]

    raise ValueError(
        "Invalid frequencies specification. If ``frequencies`` are provided as a tuple, "
        f"it must be a 2-element tuple of non-negative integers. Tuple {frequencies} provided instead."
    )


def _get_all_frequency_pairs(frequencies):
    grids = jnp.meshgrid(*[freqs for freqs in frequencies], indexing="ij")
    return jnp.stack([g.reshape(-1) for g in grids])


def _get_frequency_pairs_from_callable(
    frequency_mask: Callable[..., bool], frequencies: Tuple[jnp.ndarray, ...]
):
    """
    Apply the callable assigned to `frequency_mask` to all frequency tuples.

    Parameters
    ----------
    frequency_mask :
        A function with signature: frequency_mask(*freqs) -> bool (or 0/1).
    frequencies :
        1D arrays of frequencies (one per dimension).

    Returns
    -------
    :
        Shape (D, K): columns are the selected frequency tuples.
    """
    all_pairs = itertools.product(*frequencies)  # shape (D, N)

    if not callable(frequency_mask):
        raise TypeError(
            "`frequency_mask` must be a callable like frequency_mask(*freqs) -> bool."
        )

    selected = []
    for j, freqs in enumerate(all_pairs):
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
            selected.append(jnp.asarray(freqs))

    return (
        jnp.stack(selected, axis=1)
        if selected
        else jnp.zeros((len(frequencies), 0), dtype=int)
    )


class FourierBasis(AtomicBasisMixin, Basis):
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
    frequencies :
        Frequency specification(s).

        **Single specification** (broadcasted to all dimensions when ``ndim > 1``):
          * ``int k`` with ``k >= 0``.
          * ``(low, high)``: a 2-element tuple of integers with ``0 <= low < high``.
          * 1-D NumPy ``ndarray`` of non-negative integers. If not sorted ascending,
            a ``UserWarning`` is issued.

        **Per-dimension container**:
          * A **list** of length ``ndim`` whose elements are each a valid single specification.
            For ``ndim == 1``, a length-1 list is also accepted.

    ndim :
        Dimensionality of the basis. Default is 1.

    frequency_mask :
        Frequency selection mask. Used to filter the evaluated frequencies after
        construction.

        Accepted forms:
            * ``no-intercept``: drop the all-zero frequency (DC component).
            * ``"all"``: Keep all frequencies
            * **None** : Keep all frequencies.
            * **array_like** of {0, 1} or booleans : Shape must match the number
              of possible frequency combinations for the given ``frequencies``.
              - In 1D: shape = (n_freq,).
              - In nD: shape = (n_freq_dim1, n_freq_dim2, ..., n_freq_dimN).
            * **callable** : A function applied to each tuple of frequency
              coordinates, returning a single boolean or {0, 1}. For example:
              ``lambda f1, f2, ...: condition``.
                - Must return a scalar boolean or integer {0, 1}.
                - Returning arrays, lists, or non-boolean values raises a
                  ``ValueError``.

        Validation rules:
            * Array values must be exactly 0 or 1 (floats are allowed if equal to
              0.0 or 1.0).
            * Strings or non-numeric values raise ``ValueError``.
            * Callable return values must be a single boolean or {0, 1}; anything
              else raises ``ValueError``.
            * Errors raised inside the callable are propagated as
              ``TypeError`` with a descriptive message.

    label : str, optional
        Descriptive label for the basis (e.g., to use in plots or summaries).

    Notes
    -----
    - If ``frequency_mask`` is provided, only the selected frequency
      combinations are used to build the basis.
    - The output of ``compute_features`` contains both cosine and sine components for
      each active frequency combination, except that the all-zero frequency
      includes only a cosine term.
    - When a **tuple** is provided as a frequency, it is interpreted
      as a single range specification. Tuples that are not exactly a 2-element
      tuple of non-negative integers are invalid.
    """

    _is_complex = True

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
        ndim: int,
        frequency_mask: (
            Literal["all", "no-intercept"] | jnp.ndarray | None
        ) = "no-intercept",
        label: Optional[str] = None,
    ) -> None:
        self._n_input_dimensionality = self._check_ndim(ndim)
        self.frequencies = frequencies
        self.frequency_mask = frequency_mask

        AtomicBasisMixin.__init__(
            self,
            n_basis_funcs=self.n_basis_funcs,
            label=label,
        )

    @property
    def frequency_mask(self) -> Callable | jnp.ndarray | Literal["all", "no-intercept"]:
        """Get the frequency mask for the Fourier basis.

        The frequency mask can be either:

        - a boolean array (or array-like of 0s and 1s) whose shape matches the number
          of frequencies along each input dimension, or
        - a callable with signature ``frequency_mask(*freqs) -> bool`` (or 0/1) applied
          to each frequency tuple, or
        - a string, either ``"all"``, if all possible frequency combinations are included, or
          ``"no-intercept"``, if the intercept term is dropped (DC component).

        Returns
        -------
        :
            The callable used to build the mask, the boolean JAX array mask,
            or ``None`` if no mask is applied.
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
            - **Array / array-like (bool or 0/1)**: Explicit mask over the
              frequency grid. Must have shape
              ``(len(frequencies[0]), len(frequencies[1]), ...)`` and contain
              only booleans or the integers {0, 1}.
            - :class:`callable`: A function with signature
              ``frequency_mask(*freqs) -> bool`` (or 0/1). It is applied to each
              frequency tuple ``(f1, f2, ..., f_n)`` to build the mask. The callable is
              **not required to be vectorized**; ``n`` is the input dimensionality.
            - :class:`None <NoneType>`: Include all frequency combinations.

        Raises
        ------
        ValueError
            If an array mask has values other than {0, 1}/booleans, or if its
            shape does not match the expected grid shape.
        TypeError
            If ``values`` is neither array-like, callable, nor ``None``; or if
            the callable returns a value that is not a single boolean or 0/1.

        Notes
        -----
        - Setting this property updates:
          ``self._frequency_mask`` (callable or boolean mask),
          ``self._n_basis_funcs`` (number of active basis functions),
          and ``self._eval_freq`` (selected frequencies for evaluation).
        """

        if isinstance(values, str) and values == "no-intercept":
            self._frequency_mask = "no-intercept"
            self._freq_combinations = _get_all_frequency_pairs(self._frequencies)
            if jnp.all(self._freq_combinations[..., 0] == 0):
                self._freq_combinations = self._freq_combinations[..., 1:]

        elif values is None or isinstance(values, str) and values == "all":
            self._frequency_mask = "all"
            self._freq_combinations = _get_all_frequency_pairs(self._frequencies)

        elif callable(values):
            self._freq_combinations = _get_frequency_pairs_from_callable(
                values, self._frequencies
            )
            self._frequency_mask = values
        else:
            try:
                values = jnp.asarray(values)
            except Exception as e:
                raise ValueError(
                    f"``frequency_mask`` {values} cannot be converted to a jax array of boolean."
                ) from e

            if not jnp.all((values == 0) | (values == 1)):
                raise ValueError("Frequency mask must be an array-like of 0s and 1s.")

            values = values.astype(bool)

            if not values.ndim == self._n_input_dimensionality:
                ndim = self._n_input_dimensionality
                raise ValueError(
                    f"The frequency mask for a {ndim}-dimensional Fourier basis "
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
            self._freq_combinations = jnp.stack(
                [freqs[idxs[d]] for d, freqs in enumerate(self._frequencies)]
            )

        # used to drop or not the zero phase
        self._has_zero_phase = (
            0
            if self._freq_combinations.size == 0
            else int(jnp.all(self._freq_combinations[:, 0] == 0))
        )

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
        ndim = self._n_input_dimensionality

        if isinstance(frequencies, Number) and (frequencies == int(frequencies)):
            frequencies = [arange_constructor(frequencies)] * ndim

        elif isinstance(frequencies, tuple):
            frequencies = _process_tuple_frequencies(
                frequencies, self._n_input_dimensionality
            )

        elif is_at_least_1d_numpy_array_like(frequencies):
            frequencies = _check_and_sort_frequencies(*([frequencies] * ndim))

        elif isinstance(frequencies, list):
            if len(frequencies) != self._n_input_dimensionality:
                raise ValueError(
                    "Length of frequencies list must match input dimensionality."
                )
            frequencies = [arange_constructor(f) for f in frequencies]

        else:
            if isinstance(frequencies, (np.ndarray, jnp.ndarray)) and ~np.issubdtype(
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
        The frequencies after the masking is applied.

        Returns
        -------
            The masked frequencies, shape ``(ndim, n_frequency_combinations)``.
            ``masked_frequencies[:, i]`` is the frequency combination for the
            i-th basis function.

        """
        return self._freq_combinations

    @property
    def ndim(self):
        """The dimensionality of the basis."""
        return self._n_input_dimensionality

    @staticmethod
    def _check_ndim(ndim: int) -> int:
        try:
            is_int = int(ndim) == ndim
        except Exception as e:
            raise TypeError(f"Cannot convert ndim {ndim!r} to type int.") from e
        is_positive = ndim > 0
        if not is_int or not is_positive:
            raise ValueError(
                f"ndim must be a non-negative integer. {ndim!r} provided instead."
            )
        return int(ndim)

    @property
    def n_basis_funcs(self) -> int | None:
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
        angles = sample_pts @ self._freq_combinations
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
                return super().set_params(**params)
        else:
            return super().set_params(**params)
