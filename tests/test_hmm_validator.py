from contextlib import nullcontext as does_not_raise

import numpy as np
import pynapple as nap
import pytest

from conftest import MockHMM
from nemos.hmm.validation import HMMValidator


def all_subclasses(cls):
    seen = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        stack.extend(sub.__subclasses__())
    return seen


class TestHMMValidator:
    """Test suite for input validation logic in HMMValidator."""

    def test_user_param_order(self) -> None:
        """Meta-test.

        Tests that any subclasses of HMMValidator have the correct user parameter order
        """
        import importlib
        import pkgutil

        import nemos

        # Import every submodule so all HMMValidator subclasses get registered.
        for _, modname, _ in pkgutil.walk_packages(nemos.__path__, prefix="nemos."):
            importlib.import_module(modname)

        # Filter the classes that are subclasses of 'SuperClass'.
        subclasses = all_subclasses(HMMValidator)

        for validator in subclasses:
            n_params = len(validator.model_param_names)
            user_par = [0.0] * (n_params - 2) + [1.0, 1.0]
            params = validator.to_model_params(user_par)
            assert np.all(params.hmm_params.log_initial_prob == 0.0)
            assert np.all(params.hmm_params.log_transition_prob == 0.0)

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            (
                np.random.rand(10, 2),
                np.random.rand(10),
                does_not_raise(),
            ),
            (
                np.random.rand(10, 2),
                np.random.rand(9),
                pytest.raises(ValueError, match="X and y must have"),
            ),
            (
                nap.TsdFrame(
                    t=np.arange(10),
                    d=np.random.rand(10, 2),
                ),
                nap.Tsd(
                    t=np.arange(10) + 1,
                    d=np.random.rand(10),
                ),
                pytest.raises(ValueError, match="Time axis mismatch"),
            ),
        ],
    )
    def test_validate_inputs(self, X, y, expectation):
        """Test that validate_inputs correctly validates X and y."""
        model = MockHMM(n_states=3)
        with expectation:
            model._validator.validate_inputs(X, y)

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            # nan border y
            (
                np.ones((5, 1)),
                np.array([np.nan, 1, 2, 3, np.nan]),
                does_not_raise(),
            ),
            # nan border x
            (
                np.array([[np.nan], [2], [3], [np.nan]]),
                np.array([0, 1, 3, 4]),
                does_not_raise(),
            ),
            # nan middle y
            (
                np.ones((5, 1)),
                np.array([np.nan, 1, np.nan, 2, 3]),
                pytest.raises(ValueError, match="HMM requires continuous"),
            ),
            # nan middle x
            (
                np.array([[np.nan], [2], [np.nan], [3]]),
                np.array([0, 1, 3, 4]),
                pytest.raises(ValueError, match="HMM requires continuous"),
            ),
        ],
    )
    def test_nans_only_at_border(self, X, y, expectation):
        """Test that validate_inputs allows NaNs only at the borders of the data."""
        model = MockHMM(n_states=3)
        with expectation:
            model._validator.validate_inputs(X, y)

    @pytest.mark.parametrize(
        "X_ndim, expectation",
        [
            (1, pytest.raises(ValueError, match="X must be 2-dimensional")),
            (2, does_not_raise()),
            (3, pytest.raises(ValueError, match="X must be 2-dimensional")),
        ],
    )
    def test_X_wrong_ndim_raises(self, X_ndim, expectation):
        validator = MockHMM(n_states=3)._validator
        n = 5
        shape = {1: (n,), 2: (n, 1), 3: (n, 1, 1)}[X_ndim]
        X = np.ones(shape)
        y = np.zeros(n)
        with expectation:
            validator.validate_inputs(X, y)

    @pytest.mark.parametrize(
        "y_ndim, expectation",
        [
            (2, pytest.raises(ValueError, match="y must be 1-dimensional")),
            (1, does_not_raise()),
        ],
    )
    def test_y_wrong_ndim_raises(self, y_ndim, expectation):
        validator = MockHMM(n_states=3)._validator
        n = 5
        X = np.ones((n, 1))
        y = np.zeros((n, 1)) if y_ndim == 2 else np.zeros(n)
        with expectation:
            validator.validate_inputs(X, y)

    def test_X_y_sample_mismatch_raises(self):
        validator = MockHMM(n_states=3)._validator
        with pytest.raises(ValueError, match="same number of samples"):
            validator.validate_inputs(np.ones((5, 1)), np.zeros(6))

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            # NaN at start/end of array — allowed (epoch boundary)
            (np.array([[np.nan], [0]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [np.nan]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([np.nan, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([0, np.nan]), does_not_raise()),
            # NaN in the middle — rejected
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, 1, 2]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            (
                np.array([[0], [0], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Pynapple: NaN inside an epoch — rejected
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [np.nan], [0], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Pynapple: NaN at epoch boundary — allowed
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                does_not_raise(),
            ),
            # Pynapple y: NaN inside epoch — rejected
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, np.nan, 2, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Pynapple y: NaN at epoch boundary — allowed
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, 1, np.nan, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs in the middle — rejected
            (
                np.array([[0], [np.nan], [np.nan], [0]]),
                np.array([0, 1, 2, 3]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Multiple consecutive NaNs at start — allowed
            (
                np.array([[np.nan], [np.nan], [0]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs at end — allowed
            (
                np.array([[0], [np.nan], [np.nan]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # All NaN — rejected (caught by parent validation)
            (
                np.array([[np.nan], [np.nan]]),
                np.array([np.nan, np.nan]),
                pytest.raises(ValueError),
            ),
            # No NaN — allowed
            (np.array([[0], [1]]), np.array([0, 1]), does_not_raise()),
            # Pynapple: NaN at start of second epoch — allowed
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 2], [1.9, 5]),
                ),
                np.zeros(5),
                does_not_raise(),
            ),
            # Both X and y NaN in middle at different positions — rejected
            (
                np.array([[0], [np.nan], [0], [0]]),
                np.array([0, 1, np.nan, 3]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Both X and y NaN in middle at same position — rejected
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
        ],
    )
    def test_nan_at_boundary_allowed_in_middle_rejected(self, X, y, expectation):
        validator = MockHMM(n_states=3)._validator
        with expectation:
            validator.validate_inputs(X, y)
