from contextlib import nullcontext as does_not_raise

import pytest

from nemos.solvers import AbstractSolver, list_available_solvers
from nemos.solvers._validation import (
    _assert_step_result,
    _check_all_signatures_match,
    _check_required_methods_exist,
    _validate_method_signature,
    validate_solver_class,
)

pytestmark = pytest.mark.solver_related


class GoodSolver:
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params=None,
        **solver_init_kwargs,
    ):
        pass

    def init_state(self, init_params, *args):
        return None

    def update(self, params, state, *args):
        return None

    def run(self, init_params, *args):
        return None

    @classmethod
    def get_accepted_arguments(cls):
        return set()

    def get_optim_info(self, state):
        return None


class ExtraInitParamSolver(GoodSolver):
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params=None,
        extra=None,
        **solver_init_kwargs,
    ):
        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            has_aux,
            init_params=init_params,
            **solver_init_kwargs,
        )


class BadSignatureSolver(GoodSolver):
    def update(self, params, state2, *args):
        return None

    def run(self, init_params, extra, *args):
        return None


class NoRunSolver:
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params=None,
        **solver_init_kwargs,
    ):
        pass

    def init_state(self, init_params, *args):
        return None

    def update(self, params, state, *args):
        return None

    @classmethod
    def get_accepted_arguments(cls):
        return set()

    def get_optim_info(self, state):
        return None


@pytest.mark.parametrize("method_name", list(AbstractSolver.__abstractmethods__))
def test_validate_method_signature_matches_all_methods(method_name):
    ok, error = _validate_method_signature(GoodSolver, method_name)
    assert ok is True
    assert error is None


def test_validate_method_signature_detects_mismatch():
    ok, error = _validate_method_signature(BadSignatureSolver, "update")
    assert ok is False
    assert "Incompatible signature for update" in error
    assert "state2" in error


def test_validate_method_signature_init_allows_extra_args():
    ok, error = _validate_method_signature(ExtraInitParamSolver, "__init__")
    assert ok is True
    assert error is None


@pytest.mark.parametrize(
    "solver_class, expectation",
    [
        (GoodSolver, does_not_raise()),
        (
            NoRunSolver,
            pytest.raises(
                AttributeError,
                match=r"NoRunSolver\.run does not exist\. Please implement it\.",
            ),
        ),
    ],
)
def test_check_required_methods_exist(solver_class, expectation):
    with expectation:
        _check_required_methods_exist(solver_class)


@pytest.mark.parametrize(
    "solver_class, expectation",
    [
        (GoodSolver, does_not_raise()),
        (ExtraInitParamSolver, does_not_raise()),
    ],
)
def test_check_all_signatures_match(solver_class, expectation):
    with expectation:
        _check_all_signatures_match(solver_class)


def test_check_all_signatures_match_aggregates_errors():
    with pytest.raises(ValueError) as excinfo:
        _check_all_signatures_match(BadSignatureSolver)

    message = str(excinfo.value)
    assert "Incompatible signature for update" in message
    assert "Incompatible signature for run" in message


@pytest.mark.parametrize(
    "solver_class,expectation",
    [
        (GoodSolver, does_not_raise()),
        (ExtraInitParamSolver, does_not_raise()),
        (BadSignatureSolver, pytest.raises(ValueError, match="Incompatible signature")),
        (NoRunSolver, pytest.raises(AttributeError, match=".run does not exist.")),
    ],
)
def test_validate_solver_class_without_ridge(solver_class, expectation):
    with expectation:
        validate_solver_class(solver_class, test_ridge=False, loss_has_aux=False)


@pytest.mark.parametrize("test_ridge", [True, False])
@pytest.mark.parametrize("loss_has_aux", [True, False])
def test_all_nemos_solvers_pass_validation(test_ridge, loss_has_aux):
    for spec in list_available_solvers():
        validate_solver_class(
            spec.implementation, test_ridge=test_ridge, loss_has_aux=loss_has_aux
        )


def test_assert_step_result_accepts_3_tuple():
    step_result = ("params", "state", "aux")
    assert _assert_step_result(step_result, "run") == step_result


@pytest.mark.parametrize("bad_result", [None, [], {}, "abc", 1])
def test_assert_step_result_rejects_non_tuple(bad_result):
    with pytest.raises(TypeError, match=r"run must return a tuple"):
        _assert_step_result(bad_result, "run")


@pytest.mark.parametrize("bad_tuple, expected_len", [((1, 2), 2), ((1, 2, 3, 4), 4)])
def test_assert_step_result_rejects_wrong_tuple_length(bad_tuple, expected_len):
    with pytest.raises(TypeError, match=rf"got a tuple of length {expected_len}"):
        _assert_step_result(bad_tuple, "update")


def test_assert_step_result_error_mentions_method_name():
    with pytest.raises(TypeError) as excinfo:
        _assert_step_result([], "custom_step")
    assert "custom_step must return a tuple" in str(excinfo.value)
