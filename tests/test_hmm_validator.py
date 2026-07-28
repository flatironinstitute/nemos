from contextlib import nullcontext as does_not_raise

import numpy as np
import pynapple as nap
import pytest

from conftest import MockHMM, all_subclasses
from nemos.hmm.validation import HMMValidator, has_interior_nans


def _check_continuity(model, X, y, session_starts=None):
    """Cast the session boundaries and run the continuity check on them, as models do."""
    session_starts = model._validator.validate_and_cast_session_starts(
        X, y, session_starts=session_starts
    )
    model._validator.check_is_continuous(X, y, session_starts)


def _validate_and_check_continuity(model, X, y, session_starts=None):
    """Run the validation sequence models apply to their inputs.

    Mirrors ``BaseHMM._validate_and_prepare_inputs``: shape and NaN checks first, then
    the session boundaries, then continuity within each session.
    """
    model._validator.validate_inputs(X, y)
    _check_continuity(model, X, y, session_starts)


def _nan_design(nan_at, n_samples=6):
    """A ``(n_samples, 1)`` design matrix holding NaNs at the given sample indices."""
    X = np.zeros((n_samples, 1))
    X[list(nan_at)] = np.nan
    return X


def _reference_has_interior_nans(is_nan, session_starts):
    """Naive per-session scan, as a reference for the run-counting implementation."""
    starts = np.flatnonzero(session_starts)
    for start, end in zip(starts, np.append(starts[1:], len(is_nan))):
        block = is_nan[start:end]
        valid = np.flatnonzero(~block)
        if valid.size and block[valid[0] : valid[-1] + 1].any():
            return True
    return False


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
        """Test that the validation sequence allows NaNs only at the borders of the data."""
        model = MockHMM(n_states=3)
        with expectation:
            _validate_and_check_continuity(model, X, y)

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
        model = MockHMM(n_states=3)
        with expectation:
            _validate_and_check_continuity(model, X, y)


class TestCheckIsContinuous:
    """NaNs may sit at the borders of a session, never between two of its valid samples."""

    @pytest.mark.parametrize(
        "nan_at, expectation",
        [
            ((), does_not_raise()),  # no NaN, nothing to check
            ((0,), does_not_raise()),  # head
            ((0, 1), does_not_raise()),
            ((5,), does_not_raise()),  # tail
            ((4, 5), does_not_raise()),
            ((0, 5), does_not_raise()),  # both borders
            ((0, 1, 2, 3, 4, 5), does_not_raise()),  # entirely NaN
            ((3,), pytest.raises(ValueError, match="requires continuous")),
            ((2, 3), pytest.raises(ValueError, match="requires continuous")),
            ((0, 3, 5), pytest.raises(ValueError, match="requires continuous")),
        ],
    )
    def test_single_session(self, nan_at, expectation):
        """Without boundaries the whole recording is one session."""
        model = MockHMM(n_states=3)
        with expectation:
            _check_continuity(model, _nan_design(nan_at), np.zeros(6))

    @pytest.mark.parametrize(
        "session_starts, expectation",
        [
            (None, pytest.raises(ValueError, match="requires continuous")),
            (np.array([0, 3]), does_not_raise()),  # the NaN heads the second session
            (np.array([0, 4]), does_not_raise()),  # the NaN tails the first session
            (np.array([0, 2]), pytest.raises(ValueError, match="requires continuous")),
        ],
    )
    def test_boundaries_decide_the_verdict(self, session_starts, expectation):
        """The same NaN is interior or at a border depending on where sessions start."""
        model = MockHMM(n_states=3)
        with expectation:
            _check_continuity(
                model, _nan_design((3,)), np.zeros(6), session_starts=session_starts
            )

    @pytest.mark.parametrize(
        "session_starts",
        [
            np.array([0, 3]),  # indices
            np.array([True, False, False, True, False, False]),  # per-sample indicator
            np.array([1, 0, 0, 1, 0, 0]),  # integer 0/1 indicator
        ],
    )
    def test_boundary_formats_agree(self, session_starts):
        """Equivalent boundaries give the same verdict whatever format they come in."""
        model = MockHMM(n_states=3)
        _check_continuity(model, _nan_design((3,)), np.zeros(6), session_starts)
        with pytest.raises(ValueError, match="requires continuous"):
            _check_continuity(model, _nan_design((4,)), np.zeros(6), session_starts)

    @pytest.mark.parametrize(
        "nan_at, expectation",
        [
            ((0,), does_not_raise()),
            ((5,), does_not_raise()),
            ((3,), pytest.raises(ValueError, match="requires continuous")),
        ],
    )
    def test_y_is_none(self, nan_at, expectation):
        """During simulation only the feedforward input is available to check."""
        model = MockHMM(n_states=3)
        with expectation:
            _check_continuity(model, _nan_design(nan_at), None)

    @pytest.mark.parametrize(
        "nan_at, expectation",
        [
            ((0,), does_not_raise()),
            ((5,), does_not_raise()),
            ((3,), pytest.raises(ValueError, match="requires continuous")),
        ],
    )
    def test_nan_in_y(self, nan_at, expectation):
        """NaNs in the observations are as disqualifying as NaNs in the design matrix."""
        model = MockHMM(n_states=3)
        y = np.zeros(6)
        y[list(nan_at)] = np.nan
        with expectation:
            _check_continuity(model, np.zeros((6, 1)), y)

    def test_single_sample_sessions(self):
        """A session of one sample can hold no NaN between two valid samples."""
        model = MockHMM(n_states=3)
        _check_continuity(
            model,
            _nan_design((1, 3)),
            np.zeros(6),
            session_starts=np.ones(6, dtype=bool),
        )

    @pytest.mark.parametrize("nan_at", [(0, 1, 2), (3, 4, 5)])
    def test_entirely_nan_session_allowed(self, nan_at):
        """A session with no valid sample cannot break the recursion."""
        model = MockHMM(n_states=3)
        _check_continuity(
            model, _nan_design(nan_at), np.zeros(6), session_starts=np.array([0, 3])
        )

    @pytest.mark.parametrize(
        "nan_a, nan_b, expectation",
        [
            ((0,), (5,), does_not_raise()),  # each leaf keeps its NaNs at a border
            ((3,), (), pytest.raises(ValueError, match="requires continuous")),
            # NaNs in different leaves combine into a single interior gap
            ((2,), (3,), pytest.raises(ValueError, match="requires continuous")),
        ],
    )
    def test_pytree_leaves_combine(self, nan_a, nan_b, expectation):
        """A sample is missing when any leaf of the pytree is missing it."""
        model = MockHMM(n_states=3)
        X = {"a": _nan_design(nan_a), "b": _nan_design(nan_b)}
        with expectation:
            _check_continuity(model, X, np.zeros(6))

    @pytest.mark.parametrize(
        "session_starts, expectation",
        [
            (None, does_not_raise()),  # epochs of the time support delimit the sessions
            (np.array([0]), pytest.raises(ValueError, match="requires continuous")),
        ],
    )
    def test_pynapple_time_support_vs_explicit_boundaries(
        self, session_starts, expectation
    ):
        """Explicit boundaries take over from the time support, and can invalidate it."""
        model = MockHMM(n_states=3)
        X = nap.TsdFrame(
            t=np.arange(5),
            d=np.array([[0], [0], [np.nan], [0], [0]], dtype=float),
            time_support=nap.IntervalSet([0, 3], [2, 5]),
        )
        with expectation:
            _check_continuity(model, X, np.zeros(5), session_starts=session_starts)


class TestHasInteriorNans:
    """Run-counting identity behind the continuity check."""

    @pytest.mark.parametrize(
        "is_nan, session_starts, expected",
        [
            # single session
            ([0, 0, 0], [1, 0, 0], False),
            ([1, 0, 0], [1, 0, 0], False),  # head
            ([0, 0, 1], [1, 0, 0], False),  # tail
            ([1, 1, 1], [1, 0, 0], False),  # entirely NaN
            ([0, 1, 0], [1, 0, 0], True),  # interior
            ([0, 1, 1, 0], [1, 0, 0, 0], True),  # consecutive interior NaNs
            ([1, 0, 1, 0, 1], [1, 0, 0, 0, 0], True),  # borders plus an interior NaN
            # the same NaN pattern, cut into two sessions at the NaN
            ([0, 1, 0], [1, 1, 0], False),  # NaN heads the second session
            ([0, 1, 0], [1, 0, 1], False),  # NaN tails the first session
            # NaNs at both borders of the second session
            ([1, 0, 1, 0, 1], [1, 0, 1, 0, 0], False),
            # a session with no valid sample alongside a valid one
            ([1, 1, 0, 0], [1, 0, 1, 0], False),
            # every sample its own session
            ([0, 1, 0], [1, 1, 1], False),
            # a valid run spanning a boundary, with the next session split
            ([0, 0, 0, 1, 0], [1, 0, 1, 0, 0], True),
            # interior NaN in the second of two sessions
            ([0, 0, 1, 0], [1, 0, 0, 0], True),
            ([0, 1, 0, 0], [1, 0, 0, 1], True),
        ],
    )
    def test_truth_table(self, is_nan, session_starts, expected):
        is_nan = np.asarray(is_nan, dtype=bool)
        session_starts = np.asarray(session_starts, dtype=bool)
        assert has_interior_nans(is_nan, session_starts) == expected

    def test_matches_per_session_reference(self):
        """Cross-check the counting identity against a naive per-session scan."""
        rng = np.random.default_rng(0)
        for _ in range(500):
            n_samples = int(rng.integers(1, 20))
            is_nan = rng.random(n_samples) < rng.choice([0.0, 0.2, 0.5, 0.9, 1.0])
            session_starts = rng.random(n_samples) < rng.choice([0.0, 0.3, 0.8])
            session_starts[0] = True
            assert has_interior_nans(is_nan, session_starts) == (
                _reference_has_interior_nans(is_nan, session_starts)
            ), f"is_nan={is_nan.astype(int)}, session_starts={session_starts.astype(int)}"
