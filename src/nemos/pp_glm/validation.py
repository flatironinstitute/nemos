"""Validation classes for PPGLM and PopulationPPGLM models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from pynapple import IntervalSet, Tsd

from ..base_validator import RegressorValidator
from ..glm.params import GLMParams, GLMUserParams
from ..glm.validation import GLMValidator, from_glm_params, to_glm_params
from .data import PredictorsPPGLM, SpikesPPGLM
from .params import PPGLMParamsWithKey


def to_pp_glm_params_with_key(params: GLMParams, random_key: jnp.array):
    """Map from PPGLMParams to PPGLMParamsWithKey.

     Map from PPGLMParams to PPGLMParamsWithKey by appending a jax random key.
    The key is converted from uint32 to float to avoid solver initialization error"""
    return PPGLMParamsWithKey(params, random_key.astype(params.coef.dtype))


@dataclass(frozen=True, repr=False)
class PPGLMValidator(GLMValidator):
    """
    Validator for a single-neuron PP-GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for single-neuron PP-GLMs. Single-neuron PP-GLMs have:
    - 1D coefficients: shape (n_features,) or dict of (n_features,) arrays
    - 1D intercept: shape (1, )
    - 2D input X: shape (2, n_events) or pytree of same
    - 2D output y: shape (3, n_events)

    """

    n_basis_funcs: int = field(kw_only=True)
    # random_key: jnp.array = field(kw_only=True)
    expected_param_dims: Tuple[int] = (
        1,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    to_model_params: Callable[[GLMUserParams], GLMParams] = to_glm_params
    from_model_params: Callable[[GLMParams], GLMUserParams] = from_glm_params
    model_class: str = "PPGLM"
    X_dimensionality: int = 2
    y_dimensionality: int = 2
    _glm_validator: GLMValidator = GLMValidator()
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. coef must be an array "
                "or nemos.pytree.FeaturePytree with array leafs of shape "
                "(n_features, ). intercept must be of shape (1,)."
                "\nThe provided coef, intercept and random_key have shapes ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
        ("validate_intercept_shape", None),
        ("validate_random_key", None),
    )

    def validate_random_key(self, random_key, dtype: type = jnp.uint32):
        """Validate random key dtype and shape.

        Parameters
        ----------
        random_key :
            The random key to validate.
        dtype :
            Expected dtype — ``jnp.uint32`` for user-facing keys,
            ``jnp.float64`` for solver-internal keys.
        """
        key = jnp.asarray(random_key)
        if key.dtype != dtype or key.shape != (2,):
            raise ValueError(
                f"random_key must be a {dtype} array with shape (2,). "
                f"Got shape {key.shape}, dtype {key.dtype}."
            )

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[PredictorsPPGLM] = None,
        y: Optional[SpikesPPGLM] = None,
    ):
        """
        Validate consistency between parameters and inputs for PP-GLM.

        For single-neuron PP-GLM, only validates feature consistency with X.
        Does not validate y since it's 1D (single neuron, no neuron axis to check).
        """
        if X is not None:
            n_features = jax.tree_util.tree_map(lambda p: p.shape[0], params.coef)
            n_predictors_params = int(n_features / self.n_basis_funcs)
            predictors_X = jnp.unique(X.ids)

            if n_predictors_params != predictors_X.size:
                raise ValueError(
                    "Inconsistent number of features. "
                    f"Model coefficients assume {n_predictors_params} groups and {self.n_basis_funcs} basis functions, "
                    f"X has {predictors_X.size} groups instead!"
                )

            if not all(predictors_X == jnp.arange(predictors_X.size)):
                raise ValueError(
                    "Inconsistent predictor IDs. "
                    f"Predictor IDs must be consecutive integers from 0 to {predictors_X.size - 1}."
                )

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        n_features = int(jnp.unique(X.ids).size * self.n_basis_funcs)
        empty_coef = jnp.zeros(n_features)
        empty_intercept = jnp.empty((1,))
        return to_glm_params((empty_coef, empty_intercept))

    def validate_span(self, X: Tsd, y: Tsd, recording_time: IntervalSet, tol=1.0):
        """Check that X, y, and recording_time are mutually consistent in time support."""
        span = recording_time.time_span()

        for name, obj in (("X", X), ("y", y)):
            # check data doesn't fall outside recording_time
            outside = obj.time_support.time_span().set_diff(recording_time)
            if len(outside) > 0:
                raise ValueError(
                    f"{name} has time support outside recording_time: {outside}."
                )
            # check recording_time span doesn't extend far beyond data
            uncovered = span.set_diff(obj.time_support.time_span())
            if uncovered.tot_length() > tol:
                raise ValueError(
                    f"recording_time span extends more than {tol}s beyond "
                    f"{name} time support. Uncovered duration: {uncovered.tot_length():.3f}s. "
                    f"recording_time span should match the time support of your data."
                )


@dataclass(frozen=True, repr=False)
class PopulationPPGLMValidator(PPGLMValidator):
    """
    Validator for population (multi-neuron) PP-GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for population PP-GLMs. Population PP-GLMs have:
    - 2D coefficients: shape (n_features, n_neurons) or dict of (n_features, n_neurons) arrays
    - 1D intercept: shape (n_neurons,)
    - 2D input X: shape (2, n_events) or pytree of same
    - 2D output y: shape (3, n_events)

    """

    expected_param_dims: Tuple[int] = (
        2,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    model_class: str = "PopulationPPGLM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. "
                "coef must be an array or pytree "
                "with array leaves of shape (n_features, n_neurons). "
                "intercept must be of shape (n_neurons,)."
                "\nThe provided coef, intercept and random_key have shapes ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
        ("validate_random_key", None),
    )

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[PredictorsPPGLM] = None,
        y: Optional[SpikesPPGLM] = None,
    ):

        # First validate X consistency (features) using parent implementation
        super().validate_consistency(params, X=X, y=None)

        # Then validate y consistency (neurons) - specific to population GLM
        if y is not None:
            n_neurons_coef = jax.tree_util.tree_map(lambda p: p.shape[1], params.coef)
            y_neurons = jnp.unique(y.ids)
            if n_neurons_coef != y_neurons.size:
                raise ValueError(
                    "Inconsistent number of neurons. "
                    f"Model coefficients assume "
                    f"{n_neurons_coef} neurons, "
                    f"y has {y_neurons.size} neurons instead!"
                )

            if not jnp.allclose(jnp.arange(y_neurons.size), y_neurons):
                raise ValueError(
                    "Inconsistent neuron IDs. "
                    f"Neuron IDs must be consecutive integers from 0 to {y_neurons.size - 1}."
                )

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        n_features = int(jnp.unique(X.ids).size * self.n_basis_funcs)
        n_neurons = jnp.unique(y.ids).size
        empty_coef = jnp.zeros((n_features, n_neurons))
        empty_intercept = jnp.empty((n_neurons,))
        return to_glm_params((empty_coef, empty_intercept))
