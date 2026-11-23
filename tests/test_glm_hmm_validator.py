"""Tests for GLMHMMValidator and its HMMValidator base methods in isolation.

No model construction needed — validator methods are called directly on
``GLMHMMValidator(n_states=3)``.  EM correctness lives in
``test_glm_hmm_algorithms.py``; fit-path wiring lives in
``test_glm_hmm_class.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nemos._observation_model_builder import (
    AVAILABLE_OBSERVATION_MODELS,
    instantiate_observation_model,
)
from nemos.glm.params import FromGLMParams, ToGLMParams
from nemos.glm_hmm.params import FromGLMHMMParams, ToGLMHMMParams
from nemos.glm_hmm.validation import GLMHMMValidator

# ---------------------------------------------------------------------------
# Constants and shared fixtures
# ---------------------------------------------------------------------------

N_STATES = 3
N_FEATURES = 2


def get_obs_model_instances():
    return [instantiate_observation_model(o) for o in AVAILABLE_OBSERVATION_MODELS]


@pytest.fixture
def validator(request):
    obs_model = request.param
    to_glm_hmm_params = ToGLMHMMParams(ToGLMParams(obs_model.map_to))
    from_glm_hmm_params = FromGLMHMMParams(FromGLMParams(obs_model.map_from))
    commons = GLMHMMValidator.model_param_names
    param_names = frozenset({*commons, *obs_model.extra_params.keys()})
    return GLMHMMValidator(
        n_states=N_STATES,
        to_model_params=to_glm_hmm_params,
        from_model_params=from_glm_hmm_params,
        model_param_names=param_names,
    )


@pytest.fixture
def valid_user_params():
    """Valid 5-tuple of user params for n_states=3, n_features=2."""
    return {
        "coef": jnp.zeros((N_FEATURES, N_STATES)),
        "intercept": jnp.zeros((N_STATES,)),
        "scale": jnp.ones((N_STATES,)),
        "dispersion": jnp.ones((N_STATES,)),
        "initial_prob": jnp.ones(N_STATES) / N_STATES,
        "transition_prob": jnp.ones((N_STATES, N_STATES)) / N_STATES,
    }


def filter_params(params: dict, extra_params):
    commons = ["coef", "intercept", "initial_prob", "transition_prob"]
    all_params = commons + list(extra_params)
    return {k: v for k, v in params.items() if k in all_params}


# ---------------------------------------------------------------------------
# check_model_params_shape
# ---------------------------------------------------------------------------


class TestCheckModelParamsShape:
    """Full validate_and_cast_params pipeline; exercises GLMHMM-specific shape checks."""

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_valid_params_no_error(self, validator, valid_user_params, request):
        validator.validate_and_cast_params(
            filter_params(valid_user_params, validator.model_param_names)
        )

    @pytest.mark.parametrize("dim", [0, 1, 3])
    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_coef_wrong_ndim_raises(self, validator, valid_user_params, dim):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        if dim == 0:
            coef = jnp.array(1.0)
        elif dim == 1:
            coef = jnp.zeros((N_FEATURES,))
        else:
            coef = jnp.zeros((N_FEATURES, N_STATES, 1))
        valid_user_params["coef"] = coef
        with pytest.raises(ValueError, match="dimensionality"):
            validator.validate_and_cast_params(valid_user_params)

    @pytest.mark.parametrize("dim", [0, 2, 3])
    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_intercept_wrong_ndim_raises(self, validator, valid_user_params, dim):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        if dim == 0:
            intercept = jnp.array(1.0)
        elif dim == 2:
            intercept = jnp.zeros((N_STATES, N_STATES))
        else:
            intercept = jnp.zeros((N_STATES, 1, 1))
        valid_user_params["intercept"] = intercept
        with pytest.raises(ValueError, match="Unexpected array dimensionality"):
            validator.validate_and_cast_params(valid_user_params)

    @pytest.mark.parametrize(
        "bad_input",
        [
            {"coefs": jnp.zeros((2, 3))},
            {"coef": jnp.zeros((2, 3)), "intercepts": jnp.zeros((3,))},
            {
                "coef": jnp.zeros((2, 3)),
                "intercept": jnp.zeros((3,)),
                "initials_prob": jnp.ones((3,)),
            },
            {
                "coef": jnp.zeros((2, 3)),
                "intercept": jnp.zeros((3,)),
                "initialprob": jnp.ones(3) / 3,
            },
        ],
    )
    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_invalid_key_raises(self, validator, bad_input):
        with pytest.raises(KeyError, match="Parameters must have key"):
            validator.validate_and_cast_params(bad_input)

    @pytest.mark.parametrize("bad_input", [0, {0, 1}])
    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_non_dict_input_raises(self, validator, bad_input):
        with pytest.raises(TypeError, match="Parameters must of dictionary type"):
            validator.validate_and_cast_params(bad_input)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_coef_wrong_n_states_raises(self, validator, valid_user_params):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        # Right ndim (2) but wrong last axis — hits check_model_params_shape, not check_array_dimensions
        coef = jnp.zeros((N_FEATURES, N_STATES + 1))
        valid_user_params["coef"] = coef
        with pytest.raises(ValueError, match="GLM coef must be of shape"):
            validator.validate_and_cast_params(valid_user_params)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_intercept_wrong_n_states_raises(self, validator, valid_user_params):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        # Right ndim (1) but wrong size — hits check_model_params_shape, not check_array_dimensions
        intercept = jnp.zeros((N_STATES + 1,))
        valid_user_params["intercept"] = intercept
        with pytest.raises(ValueError, match="GLM intercept must be of shape"):
            validator.validate_and_cast_params(valid_user_params)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_string_coef_raises_type_error(self, validator, valid_user_params):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        valid_user_params["coef"] = "bad_coef"
        with pytest.raises(TypeError):
            validator.validate_and_cast_params(valid_user_params)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_string_intercept_raises_type_error(self, validator, valid_user_params):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        valid_user_params["intercept"] = "bad_intercept"
        with pytest.raises(TypeError):
            validator.validate_and_cast_params(valid_user_params)


# ---------------------------------------------------------------------------
# check_init_and_transition_prob_shape
# ---------------------------------------------------------------------------


class TestCheckInitAndTransitionProbShape:
    """Shape checks on initial and transition probability arrays."""

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_valid_shapes_no_error(self, validator, valid_user_params):
        validator.check_init_and_transition_prob_shape(
            filter_params(valid_user_params, validator.model_param_names)
        )

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
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
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        valid_user_params["initial_prob"] = bad_init_prob
        with pytest.raises(ValueError):
            validator.check_init_and_transition_prob_shape(valid_user_params)

    @pytest.mark.parametrize(
        "bad_trans_prob",
        [
            jnp.ones((N_STATES + 1, N_STATES + 1)) / (N_STATES + 1),
            jnp.ones((N_STATES, N_STATES + 1)) / (N_STATES + 1),
            jnp.ones(N_STATES) / N_STATES,
        ],
    )
    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_transition_prob_wrong_shape_raises(
        self, validator, valid_user_params, bad_trans_prob
    ):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        valid_user_params["transition_prob"] = bad_trans_prob
        with pytest.raises(ValueError):
            validator.check_init_and_transition_prob_shape(valid_user_params)


# ---------------------------------------------------------------------------
# check_init_and_transition_prob_sum_to_1
# ---------------------------------------------------------------------------


class TestCheckInitAndTransitionProbSumTo1:
    """Probability normalization checks."""

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_valid_normalized_probs_no_error(self, validator, valid_user_params):
        validator.check_init_and_transition_prob_sum_to_1(
            filter_params(valid_user_params, validator.model_param_names)
        )

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_init_prob_not_normalized_raises(self, validator, valid_user_params):
        bad_init = jnp.array([0.5, 0.5, 0.5])  # sums to 1.5
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        valid_user_params["initial_prob"] = bad_init
        with pytest.raises(ValueError, match="must sum to 1"):
            validator.check_init_and_transition_prob_sum_to_1(valid_user_params)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
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
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        valid_user_params["transition_prob"] = bad_trans
        with pytest.raises(ValueError, match="must sum to 1"):
            validator.check_init_and_transition_prob_sum_to_1(valid_user_params)


# ---------------------------------------------------------------------------
# validate_and_cast_session_starts
# ---------------------------------------------------------------------------


class TestValidateAndCastIsNewSession:
    """Session boundary array validation and casting."""

    @pytest.fixture
    def simple_data(self):
        n = 10
        return np.ones((n, 1)), np.zeros(n)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_none_returns_default(self, validator, simple_data):
        X, y = simple_data
        result = validator.validate_and_cast_session_starts(X, y, session_starts=None)
        assert result.shape == (len(y),)
        assert result[0]  # first sample always starts a session

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_bool_array_correct_length_no_error(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=bool)
        is_ns[0] = True
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result.shape == (n,)
        assert result[0]

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_bool_array_marks_session_boundary(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=bool)
        is_ns[0] = True
        is_ns[n // 2] = True
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result[n // 2]

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_bool_array_wrong_length_raises(self, validator, simple_data):
        X, y = simple_data
        bad_ns = np.zeros(len(y) + 1, dtype=bool)
        with pytest.raises(ValueError, match="Boolean session_starts must have shape"):
            validator.validate_and_cast_session_starts(X, y, session_starts=bad_ns)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_int_array_cast_succeeds(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=int)
        is_ns[0] = 1
        result = validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)
        assert result.shape == (n,)
        assert np.issubdtype(result.dtype, bool)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
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
    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_int_index_array_out_of_range_raises(self, validator, simple_data, delta):
        X, y = simple_data
        n = len(y)
        is_ns = np.array([delta[0], n + delta[1]], dtype=int)  # max >= n_samples
        with pytest.raises(
            ValueError, match="Integer session_starts values must be between"
        ):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_2d_bool_array_raises(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros((n, 1), dtype=bool)
        with pytest.raises(ValueError, match="Boolean session_starts must have shape"):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_float_dtype_raises(self, validator, simple_data):
        X, y = simple_data
        n = len(y)
        is_ns = np.zeros(n, dtype=float)
        with pytest.raises(
            TypeError, match="session_starts must be a boolean or integer array"
        ):
            validator.validate_and_cast_session_starts(X, y, session_starts=is_ns)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
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

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_none_returns_none(self, validator):
        assert validator.validate_and_cast_feature_mask(None) is None

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_valid_binary_array_no_error(self, validator):
        mask = np.array([[1, 0], [0, 1]], dtype=float)
        result = validator.validate_and_cast_feature_mask(mask)
        assert result is not None

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_non_binary_values_raises(self, validator):
        mask = np.array([[1, 2], [0, 1]], dtype=float)
        with pytest.raises(
            ValueError, match="feature_mask.*must contain only 0s and 1s"
        ):
            validator.validate_and_cast_feature_mask(mask)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_valid_dict_mask_no_error(self, validator):
        mask = {"a": np.array([[1, 0]]), "b": np.array([[0, 1]])}
        result = validator.validate_and_cast_feature_mask(mask)
        assert result is not None

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
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
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        return validator.validate_and_cast_params(valid_user_params)

    @pytest.fixture
    def X(self):
        return np.ones((20, N_FEATURES))

    @pytest.fixture
    def y(self):
        return np.zeros(20)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_consistent_params_no_error(self, validator, model_params, X, y):
        validator.validate_consistency(model_params, X=X, y=y)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_n_features_mismatch_raises(self, validator, valid_user_params, y):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        bad_coef = jnp.zeros((N_FEATURES + 1, N_STATES))
        valid_user_params["coef"] = bad_coef
        model_params = validator.validate_and_cast_params(valid_user_params)
        X_wrong = np.ones((20, N_FEATURES))
        with pytest.raises(ValueError, match="Inconsistent number of features"):
            validator.validate_consistency(model_params, X=X_wrong, y=y)

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_extras_shape_mismatch_raises(self, validator, valid_user_params):
        extra_names = (
            validator.model_param_names - validator.__class__.model_param_names
        )
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        if len(extra_names) == 0:
            pytest.skip("No extra params for the validator.")
        bad_extras = jnp.ones((N_STATES + 1,))
        for k in extra_names:
            valid_user_params[k] = bad_extras
        model_params = validator.validate_and_cast_params(valid_user_params)
        with pytest.raises(ValueError, match="The extra parameters are of shape"):
            validator.validate_consistency(model_params, X=None, y=None)


# ---------------------------------------------------------------------------
# to_model_params / from_model_params round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip through to_model_params / from_model_params."""

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    def test_coef_intercept_preserved(self, validator, valid_user_params):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        internal = validator.to_model_params(valid_user_params)
        result = validator.from_model_params(internal)
        assert jnp.allclose(result["coef"], valid_user_params["coef"])
        assert jnp.allclose(result["intercept"], valid_user_params["intercept"])

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    @pytest.mark.requires_x64
    def test_probabilities_preserved(self, validator, valid_user_params):
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        internal = validator.to_model_params(valid_user_params)
        result = validator.from_model_params(internal)
        # from_model_params re-normalises; original probs are already normalised
        assert jnp.allclose(
            result["transition_prob"], valid_user_params["transition_prob"], atol=1e-12
        )
        assert jnp.allclose(
            result["initial_prob"], valid_user_params["initial_prob"], atol=1e-12
        )

    @pytest.mark.parametrize("validator", get_obs_model_instances(), indirect=True)
    @pytest.mark.requires_x64
    def test_extra_transform(self, validator, valid_user_params):
        extra_names = (
            validator.model_param_names - validator.__class__.model_param_names
        )
        if len(extra_names) == 0:
            pytest.skip("No extra params for the validator.")
        valid_user_params = filter_params(
            valid_user_params, validator.model_param_names
        )
        # to_model_params stores the obs params in log space;
        # from_model_params maps them back with exp.
        internal = validator.to_model_params(valid_user_params)
        result = validator.from_model_params(internal)
        for k in extra_names:
            assert jnp.allclose(result[k], valid_user_params[k], atol=1e-12)
