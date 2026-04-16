"""Validation mixin class for HMM-based models."""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import pynapple as nap
from pynapple import Tsd, TsdFrame
from ..type_casting import is_pynapple_tsd
from ..typing import DESIGN_INPUT_TYPE

from .. import validation
from ..base_validator import RegressorValidator
from ..typing import ArrayLike
from .params import HMMParams, HMMUserParams
from .utils import (
    initialize_is_new_session,
    compute_is_new_session,
    shift_nan_is_new_session,
)

from .params import HMMModelParamsT, HMMUserProvidedParamsT


def has_nans_only_at_border(arr):
    """Check if NaNs appear only at the start and end along axis=0."""
    # Check which rows have any NaN values
    is_nan = jnp.any(jnp.isnan(arr.reshape(arr.shape[0], -1)), axis=1)

    # If no NaNs, it's valid
    if not jnp.any(is_nan):
        return True

    # If all NaNs, it's valid
    if jnp.all(is_nan):
        return True

    # Find first and last non-NaN positions
    non_nan_indices = jnp.where(~is_nan)[0]
    first_valid = non_nan_indices[0]
    last_valid = non_nan_indices[-1]

    # Check if there are any NaNs between first and last valid values
    return not jnp.any(is_nan[first_valid : last_valid + 1])


def to_hmm_params(user_params: HMMUserParams) -> HMMParams:
    """Map from HMMUserParams to HMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return HMMParams(*(jnp.log(p) for p in user_params))


def from_hmm_params(params: HMMParams) -> HMMUserParams:
    """Map from HMMParams to HMMUserParams.

    Converts internal model parameters (log_scale and log probabilities)
    to user-facing parameters (scale and probabilities in regular space).
    """
    # exponentiate and re-normalize
    initial_prob = jnp.exp(params.log_initial_prob)
    initial_prob /= initial_prob.sum()
    transition_prob = jnp.exp(params.log_transition_prob)
    transition_prob /= transition_prob.sum(axis=1, keepdims=True)
    return (
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class HMMValidator(RegressorValidator[HMMUserProvidedParamsT, HMMModelParamsT]):
    """Validate HMM parameters. Meant to be used as a mixin class for models that use HMMs."""

    n_states: int = field(kw_only=True)  # keyword only and required.
    model_param_names: Tuple[str] = ("initial_prob", "transition_prob")
    model_class: str = "HMM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        ("check_init_and_transition_prob_shape", None),
        ("check_init_and_transition_prob_sum_to_1", None),
    )

    def check_array_dimensions(
        self,
        params: HMMUserProvidedParamsT,
        err_msg: Optional[str] = None,
        err_message_format: str = None,
    ) -> HMMUserProvidedParamsT:
        """
        Check array dimensions with custom error formatting for HMM-based model parameters.

        Overrides the base implementation to provide model-specific error messages
        that include the actual shapes of the provided parameters. The expected shapes of
        additional model parameters and error message should be set in the child class (e.g
        see GLMHMMValidator for an example).

        Parameters
        ----------
        params :
            User-provided parameters as a tuple.
        err_msg :
            Custom error message (unused, overridden by err_message_format).
        err_message_format :
            Format string for error message that takes two shape arguments.

        Returns
        -------
        :
            The validated parameters.

        Raises
        ------
        ValueError
            If arrays have incorrect dimensionality.
        """
        wrapped = self.wrap_user_params(params)
        shapes = tuple(jax.tree_util.tree_map(lambda x: x.shape, p) for p in wrapped)
        err_msg = err_message_format.format(*shapes)
        return super().check_array_dimensions(params, err_msg=err_msg)

    def check_user_params_structure(
        self, params: HMMUserProvidedParamsT, **kwargs
    ) -> HMMUserProvidedParamsT:
        """
        Validate that user parameters are a two-element structure.

        Parameters
        ----------
        params :
            User-provided parameters (should be a tuple/list of length 2).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        :
            The validated parameters.

        Raises
        ------
        ValueError
            If parameters do not have length two.
        """
        validation.check_length(
            params,
            len(self.model_param_names),
            f"Params must have length {len(self.model_param_names)}: "
            f"({', '.join(self.model_param_names)}).",
        )
        if not isinstance(params, (tuple, list)):
            raise TypeError(
                f"{self.model_class} params must be a tuple/list of length {len(self.model_param_names)}, "
                f"({', '.join(self.model_param_names)})."
            )
        return params

    def check_init_and_transition_prob_shape(
        self, params: HMMUserProvidedParamsT
    ) -> HMMUserProvidedParamsT:
        """Check initial and transition probabilities shape."""
        initial_prob, transition_prob = self.wrap_user_params(params)[-2:]
        if initial_prob.shape != (self.n_states,):
            raise ValueError(
                f"initial_prob must be a 1-dimensional array of shape ``({self.n_states},)``. "
                f"Provided initial_prob is of shape ``{initial_prob.shape}`` instead."
            )
        if transition_prob.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"transition_prob must be a 2-dimensional array of shape ``({self.n_states}, {self.n_states})``."
                f"Provided transition_prob is of shape ``{transition_prob.shape}`` instead."
            )
        return params

    def check_init_and_transition_prob_sum_to_1(
        self, params: HMMUserProvidedParamsT
    ) -> HMMUserProvidedParamsT:
        """Check that initial and transition probability sum to 1."""
        initial_prob, transition_prob = self.wrap_user_params(params)[-2:]

        if not jnp.allclose(initial_prob.sum(), 1):
            raise ValueError(
                f"initial_prob must sum to 1, but got sum = {initial_prob.sum()}. "
            )
        if not jnp.allclose(jnp.sum(transition_prob, axis=1), 1):
            row_sums = jnp.sum(transition_prob, axis=1)
            raise ValueError(
                f"transition_prob matrix rows must sum to 1 over columns, but got sum = {row_sums}. "
                f"Each row i represents the probability distribution of transitioning from state i"
                f"and must sum to 1. "
            )
        return params

    def validate_inputs(
        self,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray | Tsd | TsdFrame] = None,
    ):
        """Validate inputs for HMM model."""
        super().validate_inputs(X, y)

        # Additional checks due to the time-series structure.
        # (the forward-backward implementation assumes no nans in the inputs)
        # Skip NaN border check if y is None (e.g., during simulation)
        if y is None:
            if X is not None and not has_nans_only_at_border(X):
                raise ValueError(
                    "HMM requires continuous time-series data. NaN values must only "
                    "appear at the beginning or end of the data, not in the middle."
                )
            return

        if is_pynapple_tsd(X):
            # loop over epochs and check that nans are all at the border
            epoch_slices = [
                X.get_slice(ep.start[0], ep.end[0]) for ep in X.time_support
            ]
            y_array = jnp.asarray(y)
            is_continuous = all(
                has_nans_only_at_border(X.d[s]) and has_nans_only_at_border(y_array[s])
                for s in epoch_slices
            )
        elif is_pynapple_tsd(y):
            # loop over epochs and check that nans are all at the border
            epoch_slices = [
                y.get_slice(ep.start[0], ep.end[0]) for ep in y.time_support
            ]
            is_continuous = all(
                has_nans_only_at_border(X[s]) and has_nans_only_at_border(y.d[s])
                for s in epoch_slices
            )
        else:
            # check nans at the border
            is_continuous = has_nans_only_at_border(X) and has_nans_only_at_border(y)
        if not is_continuous:
            raise ValueError(
                "HMM requires continuous time-series data. NaN values must only "
                "appear at the beginning or end of the data, not in the middle. "
                "Found NaN values within the time series, which would break the "
                "forward-backward algorithm. Please ensure your data is continuous "
                "or split it into separate epochs at the gaps."
            )

    def validate_and_cast_is_new_session(
        self, X, y, is_new_session: Optional[ArrayLike | nap.IntervalSet] = None
    ) -> jnp.ndarray:
        """Validate and cast is_new_session to a binary array of shape (n_samples,)."""
        if is_new_session is None:
            if is_pynapple_tsd(y):
                is_new_session = y.time_support
            elif is_pynapple_tsd(X):
                is_new_session = X.time_support
            # return initialize_new_session(n_samples, is_new_session)
        if isinstance(is_new_session, nap.IntervalSet):
            is_new_session = compute_is_new_session(X, y, is_new_session)
        else:
            n_samples = X.shape[0]
            is_new_session = initialize_is_new_session(n_samples, is_new_session)

        # shift any True values that fall on NaN samples to the next valid sample
        nan_x = jnp.any(jnp.isnan(jnp.asarray(X)).reshape(X.shape[0], -1), axis=1)
        nan_y = jnp.any(jnp.isnan(jnp.asarray(y)).reshape(y.shape[0], -1), axis=1)
        return shift_nan_is_new_session(is_new_session, nan_x | nan_y)

    def get_empty_params(self, X, y) -> HMMModelParamsT:
        """Return the param shape given the input data."""
        pass
