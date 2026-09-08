"""Tests for GLMHMMValidator and its HMMValidator base methods in isolation.

No model construction needed — validator methods are called directly on
``GLMHMMValidator(n_states=3)``.  EM correctness lives in
``test_glm_hmm_algorithms.py``; fit-path wiring lives in
``test_glm_hmm_class.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nemos.glm_hmm.params import GLMHMMParams
from nemos.glm_hmm.validation import GLMHMMValidator

# ---------------------------------------------------------------------------
# Constants and shared fixtures
# ---------------------------------------------------------------------------

N_STATES = 3
N_FEATURES = 2


@pytest.fixture
def validator():
    return GLMHMMValidator(n_states=N_STATES)


@pytest.fixture
def valid_user_params():
    """Valid 5-tuple of user params for n_states=3, n_features=2."""
    return (
        jnp.zeros((N_FEATURES, N_STATES)),
        jnp.zeros((N_STATES,)),
        jnp.ones((N_STATES,)),
        jnp.ones(N_STATES) / N_STATES,
        jnp.ones((N_STATES, N_STATES)) / N_STATES,
    )


# ---------------------------------------------------------------------------
# check_model_params_shape
# ---------------------------------------------------------------------------


class TestCheckModelParamsShape:
    """Full validate_and_cast_params pipeline; exercises GLMHMM-specific shape checks."""

    def test_valid_params_no_error(self, validator, valid_user_params):
        validator.validate_and_cast_params(valid_user_params)

    @pytest.mark.parametrize("dim", [0, 1, 3])
    def test_coef_wrong_ndim_raises(self, validator, valid_user_params, dim):
        if dim == 0:
            coef = jnp.array(1.0)
        elif dim == 1:
            coef = jnp.zeros((N_FEATURES,))
        else:
            coef = jnp.zeros((N_FEATURES, N_STATES, 1))
        params = (coef, *valid_user_params[1:])
        with pytest.raises(ValueError, match="dimensionality"):
            validator.validate_and_cast_params(params)

    @pytest.mark.parametrize("dim", [0, 2, 3])
    def test_intercept_wrong_ndim_raises(self, validator, valid_user_params, dim):
        if dim == 0:
            intercept = jnp.array(1.0)
        elif dim == 2:
            intercept = jnp.zeros((N_STATES, N_STATES))
        else:
            intercept = jnp.zeros((N_STATES, 1, 1))
        params = (valid_user_params[0], intercept, *valid_user_params[2:])
        with pytest.raises(ValueError, match="Unexpected array dimensionality"):
            validator.validate_and_cast_params(params)

    @pytest.mark.parametrize(
        "bad_input",
        [
            (),
            (jnp.zeros((2, 3)),),
            (jnp.zeros((2, 3)), jnp.zeros((3,))),
            (jnp.zeros((2, 3)), jnp.zeros((3,)), jnp.ones((3,))),
            (jnp.zeros((2, 3)), jnp.zeros((3,)), jnp.ones((3,)), jnp.ones(3) / 3),
        ],
    )
    def test_wrong_tuple_length_raises(self, validator, bad_input):
        with pytest.raises(ValueError, match="Params must have length 5"):
            validator.validate_and_cast_params(bad_input)

    @pytest.mark.parametrize("bad_input", [0, {0, 1}])
    def test_non_tuple_input_raises(self, validator, bad_input):
        with pytest.raises(ValueError, match="Params must have length 5"):
            validator.validate_and_cast_params(bad_input)

    def test_coef_wrong_n_states_raises(self, validator, valid_user_params):
        # Right ndim (2) but wrong last axis — hits check_model_params_shape, not check_array_dimensions
        coef = jnp.zeros((N_FEATURES, N_STATES + 1))
        params = (coef, *valid_user_params[1:])
        with pytest.raises(ValueError, match="GLM coef must be of shape"):
            validator.validate_and_cast_params(params)

    def test_intercept_wrong_n_states_raises(self, validator, valid_user_params):
        # Right ndim (1) but wrong size — hits check_model_params_shape, not check_array_dimensions
        intercept = jnp.zeros((N_STATES + 1,))
        params = (valid_user_params[0], intercept, *valid_user_params[2:])
        with pytest.raises(ValueError, match="GLM intercept must be of shape"):
            validator.validate_and_cast_params(params)

    def test_string_coef_raises_type_error(self, validator, valid_user_params):
        params = ("bad_coef", *valid_user_params[1:])
        with pytest.raises(TypeError):
            validator.validate_and_cast_params(params)

    def test_string_intercept_raises_type_error(self, validator, valid_user_params):
        params = (valid_user_params[0], "bad_intercept", *valid_user_params[2:])
        with pytest.raises(TypeError):
            validator.validate_and_cast_params(params)


# ---------------------------------------------------------------------------
# check_init_and_transition_prob_shape
# ---------------------------------------------------------------------------


class TestCheckInitAndTransitionProbShape:
    """Shape checks on initial and transition probability arrays."""

    def test_valid_shapes_no_error(self, validator, valid_user_params):
        validator.check_init_and_transition_prob_shape(valid_user_params)

    @pytest.mark.parametrize(
        "bad_init_prob",
        [
            jnp.ones(N_STATES + 1) / (N_STATES + 1),
            jnp.ones(N_STATES - 1) / (N_STATES - 1),
            jnp.ones((N_STATES, N_STATES)) / N_STATES,
        ],
    )
    def test_init_prob_wrong_shape_raises(
        self, validator, valid_user_params, bad_init_prob
    ):
        params = (*valid_user_params[:3], bad_init_prob, valid_user_params[4])
        with pytest.raises(ValueError):
            validator.check_init_and_transition_prob_shape(params)

    @pytest.mark.parametrize(
        "bad_trans_prob",
        [
            jnp.ones((N_STATES + 1, N_STATES + 1)) / (N_STATES + 1),
            jnp.ones((N_STATES, N_STATES + 1)) / (N_STATES + 1),
            jnp.ones(N_STATES) / N_STATES,
        ],
    )
    def test_transition_prob_wrong_shape_raises(
        self, validator, valid_user_params, bad_trans_prob
    ):
        params = (*valid_user_params[:4], bad_trans_prob)
        with pytest.raises(ValueError):
            validator.check_init_and_transition_prob_shape(params)


# ---------------------------------------------------------------------------
# check_init_and_transition_prob_sum_to_1
# ---------------------------------------------------------------------------


class TestCheckInitAndTransitionProbSumTo1:
    """Probability normalization checks."""

    def test_valid_normalized_probs_no_error(self, validator, valid_user_params):
        validator.check_init_and_transition_prob_sum_to_1(valid_user_params)

    def test_init_prob_not_normalized_raises(self, validator, valid_user_params):
        bad_init = jnp.array([0.5, 0.5, 0.5])  # sums to 1.5
        params = (*valid_user_params[:3], bad_init, valid_user_params[4])
        with pytest.raises(ValueError):
            validator.check_init_and_transition_prob_sum_to_1(params)

    def test_transition_prob_row_not_normalized_raises(
        self, validator, valid_user_params
    ):
        bad_trans = jnp.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],  # row sums to 1.5
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
            ]
        )
        params = (*valid_user_params[:4], bad_trans)
        with pytest.raises(ValueError):
            validator.check_init_and_transition_prob_sum_to_1(params)


# ---------------------------------------------------------------------------
# validate_and_cast_session_starts
# ---------------------------------------------------------------------------


class TestValidateAndCastIsNewSession:
    """Session boundary array validation and casting."""

    @pytest.fixture
    def simple_data(self):
        n = 10
        return np.ones((n, 1)), np.zeros(n)

    def test_none_returns_default(self, validator, simple_data):
        X, y = simple_data
        result = validator.validate_and_cast_session_starts(X, y, session_starts=None)
        assert result.shape == (len(y),)
        assert result[0]  # first sample always starts a session

    def test_bool_array_correct_length_no_error(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=bool)
        is_ns[0] = True
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result.shape == (n,)
        assert result[0]

    def test_bool_array_marks_session_boundary(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=bool)
        is_ns[0] = True
        is_ns[n // 2] = True
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result[n // 2]

    def test_bool_array_wrong_length_raises(self, validator, simple_data):
        X, y = simple_data
        bad_ns = np.zeros(len(y) + 1, dtype=bool)
        with pytest.raises(ValueError, match="Boolean session_starts must have shape"):
            validator.validate_and_cast_session_starts(X, y, session_starts=bad_ns)

    def test_int_array_cast_succeeds(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=int)
        is_ns[0] = 1
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result.shape == (n,)
        assert np.issubdtype(result.dtype, bool)

    def test_int_index_array_marks_sessions(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        # Integer array interpreted as indices of session starts (not a 0/1 mask)
        is_ns = np.array([0, n // 2], dtype=int)
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result.shape == (n,)
        assert result[0]
        assert result[n // 2]
        assert not result[1]

    @pytest.mark.parametrize("delta", [(0, 5), (-5, 0), (-1, 1)])
    def test_int_index_array_out_of_range_raises(self, validator, simple_data, delta):
        X, y = simple_data
        n = len(y)
        is_ns = np.array([delta[0], n + delta[1]], dtype=int)  # max >= n_samples
        with pytest.raises(
            ValueError, match="Integer session_starts values must be between"
        ):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)

    def test_2d_bool_array_raises(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros((n, 1), dtype=bool)
        with pytest.raises(ValueError, match="Boolean session_starts must have shape"):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)

    def test_float_dtype_raises(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=float)
        with pytest.raises(
            TypeError, match="session_starts must be a boolean or integer array"
        ):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)

    def test_unsupported_type_raises(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = [0] * n  # plain list has no .dtype
        with pytest.raises(
            TypeError, match="session_starts must be a boolean or integer array"
        ):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)


# ---------------------------------------------------------------------------
# validate_and_cast_feature_mask
# ---------------------------------------------------------------------------


class TestValidateAndCastFeatureMask:
    """Feature mask validation and casting (delegates to GLMValidator)."""

    def test_none_returns_none(self, validator):
        assert validator.validate_and_cast_feature_mask(None) is None

    def test_valid_binary_array_no_error(self, validator):
        mask = np.array([[1, 0], [0, 1]], dtype=float)
        result = validator.validate_and_cast_feature_mask(mask)
        assert result is not None

    def test_non_binary_values_raises(self, validator):
        mask = np.array([[1, 2], [0, 1]], dtype=float)
        with pytest.raises(
            ValueError, match="feature_mask.*must contain only 0s and 1s"
        ):
            validator.validate_and_cast_feature_mask(mask)

    def test_valid_dict_mask_no_error(self, validator):
        mask = {"a": np.array([[1, 0]]), "b": np.array([[0, 1]])}
        result = validator.validate_and_cast_feature_mask(mask)
        assert result is not None

    def test_invalid_dict_mask_raises(self, validator):
        mask = {"a": np.array([[1, 0]]), "b": np.array([[0, 3]])}
        with pytest.raises(
            ValueError, match="feature_mask.*must contain only 0s and 1s"
        ):
            validator.validate_and_cast_feature_mask(mask)


# ---------------------------------------------------------------------------
# validate_consistency
# ---------------------------------------------------------------------------


class TestValidateConsistency:
    """Consistency checks between model params and inputs."""

    @pytest.fixture
    def model_params(self, validator, valid_user_params):
        return validator.validate_and_cast_params(valid_user_params)

    @pytest.fixture
    def X(self):
        return np.ones((20, N_FEATURES))

    @pytest.fixture
    def y(self):
        return np.zeros(20)

    def test_consistent_params_no_error(self, validator, model_params, X, y):
        validator.validate_consistency(model_params, X=X, y=y)

    def test_n_features_mismatch_raises(self, validator, valid_user_params, y):
        bad_coef = jnp.zeros((N_FEATURES + 1, N_STATES))
        params = (bad_coef, *valid_user_params[1:])
        model_params = validator.validate_and_cast_params(params)
        X_wrong = np.ones((20, N_FEATURES))
        with pytest.raises(ValueError, match="Inconsistent number of features"):
            validator.validate_consistency(model_params, X=X_wrong, y=y)

    def test_scale_shape_mismatch_raises(self, validator, valid_user_params):
        bad_scale = jnp.ones((N_STATES + 1,))
        params = (*valid_user_params[:2], bad_scale, *valid_user_params[3:])
        model_params = validator.validate_and_cast_params(params)
        with pytest.raises(ValueError):
            validator.validate_consistency(model_params, X=None, y=None)


# ---------------------------------------------------------------------------
# to_model_params / from_model_params round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip through to_model_params / from_model_params."""

    def test_coef_intercept_preserved(self, validator, valid_user_params):
        internal = validator.to_model_params(valid_user_params)
        result = validator.from_model_params(internal)
        assert jnp.allclose(result[0], valid_user_params[0])
        assert jnp.allclose(result[1], valid_user_params[1])

    def test_probabilities_preserved(self, validator, valid_user_params):
        internal = validator.to_model_params(valid_user_params)
        result = validator.from_model_params(internal)
        # from_model_params re-normalises; original probs are already normalised
        assert jnp.allclose(result[3], valid_user_params[3], atol=1e-5)
        assert jnp.allclose(result[4], valid_user_params[4], atol=1e-5)

    def test_scale_transform(self, validator, valid_user_params):
        # to_model_params stores scale directly as log_scale;
        # from_model_params returns exp(log_scale).
        internal = validator.to_model_params(valid_user_params)
        result = validator.from_model_params(internal)
        expected_scale = jnp.exp(valid_user_params[2])
        assert jnp.allclose(result[2], expected_scale, atol=1e-5)


# ---------------------------------------------------------------------------
# get_empty_params
# ---------------------------------------------------------------------------


class TestGetEmptyParams:
    """get_empty_params returns a GLMHMMParams with correct shapes derived from X, y, and n_states."""

    @pytest.fixture
    def empty(self, validator):
        X = np.ones((20, N_FEATURES))
        y = np.zeros(20)
        return validator.get_empty_params(X, y)

    def test_returns_glm_hmm_params(self, empty):
        assert isinstance(empty, GLMHMMParams)

    def test_coef_shape(self, empty):
        assert empty.model_params.coef.shape == (N_FEATURES, N_STATES)

    def test_intercept_shape(self, empty):
        assert empty.model_params.intercept.shape == (N_STATES,)

    def test_log_scale_shape(self, empty):
        assert empty.model_params.log_scale.shape == (N_STATES,)

    def test_log_initial_prob_shape(self, empty):
        assert empty.hmm_params.log_initial_prob.shape == (N_STATES,)

    def test_log_transition_prob_shape(self, empty):
        assert empty.hmm_params.log_transition_prob.shape == (N_STATES, N_STATES)
