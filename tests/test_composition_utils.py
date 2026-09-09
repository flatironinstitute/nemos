"""
Test corner case handling in nemos.basis._composition_utils.py
"""

import asyncio
import threading
from contextlib import nullcontext as does_not_raise

import pytest

import nemos.basis._composition_utils as compose_utils
from nemos.basis import BSplineEval


@pytest.fixture(scope="module")
def mock_class(request):
    class Mock:
        def __init__(self, label=None):
            if label == "no-default":
                return
            elif label:
                self.label = label
            else:
                self.label = self.__class__.__name__

        def compute_features(self, x, y, *args, z=10):
            pass

    return Mock(request.param)


@pytest.fixture(scope="module")
def atomic_basis(request):
    return BSplineEval(5, label=request.param)


@pytest.mark.parametrize("mock_class", ["custom", "no-default"], indirect=True)
def test_external_class_has_default_label(mock_class):
    if hasattr(mock_class, "label"):
        assert compose_utils._has_default_label(mock_class) is None
    else:
        assert compose_utils._has_default_label(mock_class) is not None


@pytest.mark.parametrize(
    "atomic_basis, new_label, expectation",
    [
        ("label", "valid", does_not_raise()),
        ("label", 1, pytest.raises(TypeError, match="'label' must be a string")),
    ],
    indirect=["atomic_basis"],
)
def test_composition_basis_setter_label_type(atomic_basis, new_label, expectation):
    with expectation:
        exception = compose_utils._atomic_basis_label_setter_logic(
            atomic_basis, new_label
        )
        if exception:
            raise exception


@pytest.mark.parametrize("mock_class", ["custom"], indirect=True)
def test_infer_input_dimensionality(mock_class):
    assert compose_utils.infer_input_dimensionality(mock_class) == 2


class TestShallowConstruction:
    """Unit tests for the shallow-construction context variable."""

    def test_default_is_false(self):
        assert compose_utils.is_shallow_construction() is False

    def test_sets_and_resets(self):
        with compose_utils.shallow_construction():
            assert compose_utils.is_shallow_construction() is True
        assert compose_utils.is_shallow_construction() is False

    def test_resets_on_exception(self):
        with pytest.raises(RuntimeError):
            with compose_utils.shallow_construction():
                raise RuntimeError("boom")
        assert compose_utils.is_shallow_construction() is False

    def test_nesting(self):
        with compose_utils.shallow_construction():
            with compose_utils.shallow_construction():
                assert compose_utils.is_shallow_construction() is True
            # inner exit must not clear the outer context
            assert compose_utils.is_shallow_construction() is True
        assert compose_utils.is_shallow_construction() is False

    @pytest.mark.parametrize("enabled, expected", [(True, True), (False, False)])
    def test_enabled_flag(self, enabled, expected):
        with compose_utils.shallow_construction(enabled):
            assert compose_utils.is_shallow_construction() is expected
        assert compose_utils.is_shallow_construction() is False

    def test_deepcopy_outside_context(self):
        bas = BSplineEval(5)
        add = bas + bas
        # default: components are independent deep copies
        assert add.basis1 is not bas
        assert add.basis2 is not bas

    def test_shallow_inside_context(self):
        bas = BSplineEval(5)
        with compose_utils.shallow_construction():
            add = bas + bas
        # inside the context the constructor stores the components by reference
        assert add.basis1 is bas
        assert add.basis2 is bas

    def test_thread_isolation(self):
        # A holds the context open while B checks concurrently on another thread.
        a_is_inside = threading.Event()
        b_has_checked = threading.Event()
        res = {}

        def worker_A():
            with compose_utils.shallow_construction():
                res["A_inside"] = compose_utils.is_shallow_construction()
                a_is_inside.set()
                b_has_checked.wait()  # hold the context open
            res["A_after"] = compose_utils.is_shallow_construction()

        def worker_B():
            a_is_inside.wait()  # ensure A is inside its context
            res["B_while_A_inside"] = compose_utils.is_shallow_construction()
            b_has_checked.set()

        ta, tb = threading.Thread(target=worker_A), threading.Thread(target=worker_B)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert res["A_inside"] is True
        assert res["A_after"] is False
        # B must not see A's flag: threads are isolated
        assert res["B_while_A_inside"] is False

    def test_async_isolation(self):
        # Two tasks on one thread, interleaved at ``await``.
        log = {}

        async def task_A():
            with compose_utils.shallow_construction():
                log["A_inside"] = compose_utils.is_shallow_construction()
                await asyncio.sleep(0.01)  # yield to B mid-block
                log["A_resumed"] = compose_utils.is_shallow_construction()

        async def task_B():
            await asyncio.sleep(0.005)  # run during A's await
            log["B_during_A"] = compose_utils.is_shallow_construction()

        async def main():
            await asyncio.gather(task_A(), task_B())

        asyncio.run(main())

        assert log["A_inside"] is True
        assert log["A_resumed"] is True  # A's context survives the await
        # B must not see A's flag mid-flight: async tasks are isolated
        assert log["B_during_A"] is False
