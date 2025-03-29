"""
Test corner case handling in nemos.basis._composition_utils.py
"""

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
