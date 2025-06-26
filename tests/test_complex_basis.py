import itertools

import numpy as np
import pytest
from conftest import instantiate_atomic_basis

import nemos as nmo

_ONE_DIM_BASIS_COMBINATIONS = list(
    itertools.product(
        [[nmo.basis.FourierEval], [nmo.basis.FourierConv]], [[True], [False]]
    )
)
_TWO_DIM_BASIS_COMBINATIONS = list(
    itertools.product(
        [
            [nmo.basis.FourierEval, nmo.basis.FourierEval],
            [nmo.basis.FourierEval, nmo.basis.FourierConv],
            [nmo.basis.FourierEval, nmo.basis.BSplineEval],
            [nmo.basis.BSplineEval, nmo.basis.FourierEval],
            [nmo.basis.FourierConv, nmo.basis.FourierConv],
        ],
        list(itertools.product([True, False], [True, False])),
    )
)
_THREE_DIM_BASIS_COMBINATIONS = list(
    itertools.product(
        [[nmo.basis.FourierEval, nmo.basis.FourierEval, nmo.basis.FourierEval]],
        list(itertools.product([True, False], [True, False], [True, False])),
    )
)
_MULT_BASIS_COMBINATIONS = (
    _ONE_DIM_BASIS_COMBINATIONS
    + _TWO_DIM_BASIS_COMBINATIONS
    + _THREE_DIM_BASIS_COMBINATIONS
)


@pytest.fixture(scope="module")
def basis_to_add(request):
    basis_cls = request.param
    return instantiate_atomic_basis(
        basis_cls, window_size=20, n_basis_funcs=5, n_frequencies=5
    )


@pytest.fixture(scope="module")
def mult_basis(request):
    basis_cls_list, include_intercept_list = request.param
    basis_obj = [
        instantiate_atomic_basis(
            b, window_size=20, n_basis_funcs=5, n_frequencies=5, include_intecept=ii
        )
        for b, ii in zip(basis_cls_list, include_intercept_list)
    ]
    out = basis_obj[0]
    for b in basis_obj[1:]:
        out = out * b
    return out


@pytest.fixture(scope="module")
def input_var(request):
    extra_input_shape = request.param
    return np.random.randn(20, *extra_input_shape)


@pytest.mark.parametrize(
    "basis_to_add",
    [
        nmo.basis.BSplineEval,
    ],
    indirect=True,
)
@pytest.mark.parametrize("mult_basis", _MULT_BASIS_COMBINATIONS, indirect=True)
@pytest.mark.parametrize("add_side", ["left", "right"])
@pytest.mark.parametrize(
    "input_var", [(1,), (1, 2), (2, 1), (2, 2), (2, 3, 4)], indirect=True
)
@pytest.mark.parametrize("drop_intercept", [True, False])
def test_expected_num_features(
    input_var, add_side, mult_basis, basis_to_add, drop_intercept
):
    if add_side == "left":
        bas = basis_to_add + mult_basis
    else:
        bas = mult_basis + basis_to_add

    basis_to_add.set_input_shape(*[input_var] * basis_to_add._n_input_dimensionality)
    mult_basis.set_input_shape(*[input_var] * mult_basis._n_input_dimensionality)

    if mult_basis._include_constant:
        # formula for col dropping (assuming the added basis is handled correctly)
        # hardcoded the num basis function
        n_features = basis_to_add.n_output_features + 2 * 5 * np.prod(
            input_var.shape[1:]
        )

    else:
        n_features = (
            basis_to_add.n_output_features
            + 2
            * (5 * np.prod(input_var.shape[1:])) ** mult_basis._n_input_dimensionality
            - np.prod(input_var.shape[1:]) * mult_basis._n_input_dimensionality
        )
    print("\n")
    print(n_features)
    print("\n")
