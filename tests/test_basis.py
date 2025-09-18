import inspect
import itertools
import re
from contextlib import nullcontext as does_not_raise
from functools import partial
from unittest.mock import patch

import jax.numpy
import numpy as np
import pynapple as nap
import pytest
from conftest import (
    DEFAULT_KWARGS,
    BasisFuncsTesting,
    CombinedBasis,
    SizeTerminal,
    custom_basis,
    list_all_basis_classes,
    list_all_real_basis_classes,
)

import nemos._inspect_utils as inspect_utils
import nemos.basis.basis as basis
import nemos.convolve as convolve
from nemos.basis import (
    CustomBasis,
    FourierEval,
    HistoryConv,
    IdentityEval,
    TransformerBasis,
)
from nemos.basis._basis import (
    AdditiveBasis,
    MultiplicativeBasis,
    add_docstring,
)
from nemos.basis._composition_utils import generate_basis_label_pair, set_input_shape
from nemos.basis._decaying_exponential import OrthExponentialBasis
from nemos.basis._fourier_basis import FourierBasis
from nemos.basis._identity import HistoryBasis, IdentityBasis
from nemos.basis._raised_cosine_basis import (
    RaisedCosineBasisLinear,
    RaisedCosineBasisLog,
)
from nemos.basis._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from nemos.utils import pynapple_concatenate_numpy


def compare_bounds(bas, bounds):
    if isinstance(bas, FourierBasis) and bas.bounds is not None:
        return all(b == bounds for b in bas.bounds)
    return bounds == bas.bounds if bounds else bas.bounds is None


def instantiate_atomic_basis(cls, **kwargs):
    if cls == CustomBasis:
        return custom_basis(**kwargs)
    names = cls._get_param_names()

    all_defaults = DEFAULT_KWARGS.copy()

    for name in DEFAULT_KWARGS:
        if name not in names:
            all_defaults.pop(name)
        elif name not in kwargs:
            kwargs[name] = all_defaults.pop(name)

    new_kwargs = kwargs.copy()
    for key in kwargs:
        if key not in names:
            new_kwargs.pop(key)
    return cls(**new_kwargs)


def set_basis_attr(bas, n_basis):
    if isinstance(bas, FourierBasis):
        # set frequencies to match the requested n_basis
        bas.frequencies = np.arange((n_basis + 1) % 2, 1 + (n_basis - n_basis % 2) // 2)
    elif isinstance(bas, CustomBasis):
        bas.basis_kwargs = {"n_basis_funcs": n_basis}
    else:
        bas.n_basis_funcs = n_basis


def get_basis_attr(bas):
    if isinstance(bas, CustomBasis):
        return bas.basis_kwargs.get("n_basis_funcs", -1)
    else:
        return bas.n_basis_funcs


def extra_kwargs(cls, n_basis):
    name = cls.__name__
    if "OrthExp" in name:
        return dict(decay_rates=np.arange(1, n_basis + 1))
    elif "Fourier" in name:
        return dict(frequencies=(1, 1 + n_basis // 2))
    return {}


def filter_attributes(obj, exclude_keys):
    return {key: val for key, val in obj.__dict__.items() if key not in exclude_keys}


def compare_basis(b1, b2):
    assert id(b1) != id(b2)
    assert b1.__class__.__name__ == b2.__class__.__name__
    par1 = b1.__dict__.get("_parent", None)
    par2 = b2.__dict__.get("_parent", None)
    if par1 is None:
        assert par2 is None
    elif par2 is None:
        assert par1 is None

    # root and all child are checked recursively
    if hasattr(b1, "basis1"):
        compare_basis(b1.basis1, b2.basis1)
        compare_basis(b1.basis2, b2.basis2)
        # add all params that are not parent or basis1,basis2
        d1 = filter_attributes(b1, exclude_keys=["_basis1", "_basis2", "_parent"])
        d2 = filter_attributes(b2, exclude_keys=["_basis1", "_basis2", "_parent"])
        assert d1 == d2
    else:
        decay_rates_b1 = b1.__dict__.get("_decay_rates", -1)
        decay_rates_b2 = b2.__dict__.get("_decay_rates", -1)
        assert np.array_equal(decay_rates_b1, decay_rates_b2)
        freqs1 = b1.__dict__.get("_frequencies", [-1])
        freqs2 = b2.__dict__.get("_frequencies", [-1])
        assert all(np.all(fi == fj) for fi, fj in zip(freqs1, freqs2))
        freqs1 = b1.__dict__.get("_freq_combinations", -1)
        freqs2 = b2.__dict__.get("_freq_combinations", -1)
        assert np.all(freqs1 == freqs2)
        f1, f2 = b1.__dict__.pop("_funcs", [True]), b2.__dict__.pop("_funcs", [True])
        assert all(fi == fj for fi, fj in zip(f1, f2))
        d1 = filter_attributes(
            b1,
            exclude_keys=[
                "_decay_rates",
                "_parent",
                "_frequencies",
                "_freq_combinations",
            ],
        )
        d2 = filter_attributes(
            b2,
            exclude_keys=[
                "_decay_rates",
                "_parent",
                "_frequencies",
                "_freq_combinations",
            ],
        )
        assert d1 == d2


def create_atomic_basis_pairs(full_list):
    """
    Define basis pairs preventing combinatorial explosion.

    Returns a list of basis class tuple paring all bases with
    a conv and eval basis. Pairing order matters.
    """
    sub_list = []
    cnt_eval, cnt_conv = 0, 0
    for cls in full_list:
        if cls.__name__.endswith("Eval") and cnt_eval < 1:
            sub_list.append(cls)
            cnt_eval += 1
        elif cls.__name__.endswith("Conv") and cnt_conv < 1:
            sub_list.append(cls)
            cnt_conv += 1
        if cnt_eval and cnt_conv:
            break
    return [
        *itertools.product(sub_list, full_list),
        *itertools.product(full_list, sub_list),
    ]


def test_all_basis_are_tested() -> None:
    """Meta-test.

    Ensure that all concrete classes in the 'basis' module are tested.
    """
    # Get all classes from the current module.
    all_classes = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isclass
    )

    # Filter the classes that are subclasses of 'SuperClass'.
    subclasses = [
        cls
        for _, cls in all_classes
        if issubclass(cls, BasisFuncsTesting) and cls != BasisFuncsTesting
    ]

    # Create the set of basis function objects that are tested using the cls definition
    tested_bases = {
        test_cls.cls[mode]
        for mode in ["eval", "conv"]
        for test_cls in subclasses
        if test_cls != CombinedBasis
        if mode in test_cls.cls
    }
    # add CustomBasis, since it is tested in tests/test_custom_basis.py
    tested_bases = tested_bases.union({CustomBasis})

    # Create the set of all the concrete basis classes
    all_bases = set(list_all_basis_classes())

    if all_bases != all_bases.intersection(tested_bases):
        raise ValueError(
            "Test should be implemented for each of the concrete classes in the basis module.\n"
            f"The following classes are not tested: {[bas.__qualname__ for bas in all_bases.difference(tested_bases)]}"
        )

    pytest_marks = getattr(TestSharedMethods, "pytestmark", [])

    # Find the parametrize mark for TestSharedMethods
    out = None
    for mark in pytest_marks:
        if mark.name == "parametrize":
            # Return the arguments of the parametrize mark
            out = mark.args[1]  # The second argument contains the list

    if out is None:
        raise ValueError("cannot fine parametrization.")

    basis_tested_in_shared_methods = set(out)
    all_one_dim_basis = set(
        list_all_basis_classes("Eval") + list_all_basis_classes("Conv") + [CustomBasis]
    )
    assert basis_tested_in_shared_methods == all_one_dim_basis


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
@pytest.mark.parametrize(
    "method_name, descr_match",
    [
        (
            "evaluate_on_grid",
            "The number of points in the uniformly spaced grid|The number of points used to construct",
        ),
        (
            "compute_features",
            "Apply the basis transformation to the input data|Convolve basis functions with input "
            "time series|Evaluate basis at sample points",
        ),
        (
            "split_by_feature",
            "Decompose an array along a specified axis into sub-arrays",
        ),
        (
            "set_input_shape",
            "Set the expected input shape for the basis object",
        ),
        (
            "evaluate",
            "Evaluate the .+ sample points",
        ),
    ],
)
def test_example_docstrings_add(
    basis_cls, method_name, descr_match, basis_class_specific_params
):
    if (
        basis_cls.__name__ in ["HistoryConv", "IdentityEval"]
        and method_name == "evaluate"
    ):
        pytest.skip("History and Identity eval docstring is specific.")
    elif basis_cls.__name__ == "CustomBasis" and method_name == "evaluate_on_grid":
        pytest.skip("CustomBasis doesn't implement the evaluate_on_grid method.")

    basis_instance = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    method = getattr(basis_instance, method_name)
    doc = inspect.getdoc(method)  # strips uniform indentation and ensures full doc
    examp_delim = "\nExamples\n--------"

    assert examp_delim in doc
    doc_components = doc.split(examp_delim)
    assert len(doc_components) == 2
    assert len(doc_components[0].strip()) > 0
    assert re.search(descr_match, doc_components[0])

    # check that the basis name is in the example
    if basis_cls not in [AdditiveBasis, MultiplicativeBasis]:
        assert basis_cls.__name__ in doc_components[1]

    # check that no other basis name is in the example (except for additive and multiplicative)
    for basis_name in basis.__dir__():
        if basis_cls in [AdditiveBasis, MultiplicativeBasis]:
            continue
        if basis_name == basis_instance.__class__.__name__:
            continue
        assert f" {basis_name}" not in doc_components[1]


@pytest.mark.parametrize(
    "public_class, meth_super",
    [
        ("IdentityEval", "IdentityBasis"),
        ("HistoryConv", "HistoryBasis"),
        ("MSplineEval", "MSplineBasis"),
        ("MSplineConv", "MSplineBasis"),
        ("BSplineEval", "BSplineBasis"),
        ("BSplineConv", "BSplineBasis"),
        ("CyclicBSplineEval", "CyclicBSplineBasis"),
        ("CyclicBSplineConv", "CyclicBSplineBasis"),
        ("RaisedCosineLinearEval", "RaisedCosineBasisLinear"),
        ("RaisedCosineLinearConv", "RaisedCosineBasisLinear"),
        ("RaisedCosineLogEval", "RaisedCosineBasisLog"),
        ("RaisedCosineLogConv", "RaisedCosineBasisLog"),
        ("OrthExponentialEval", "OrthExponentialBasis"),
        ("OrthExponentialConv", "OrthExponentialBasis"),
        ("FourierEval", "FourierBasis"),
    ],
)
@pytest.mark.parametrize("method", ["evaluate", "split_by_feature", "evaluate_on_grid"])
def test_docstrings_decorator_superclass(public_class, meth_super, method):
    cls_pub = getattr(basis, public_class)
    cls_sup = getattr(basis, meth_super)
    meth_pub = getattr(cls_pub, method)
    meth_super = getattr(cls_sup, method)
    assert meth_pub.__doc__.startswith(meth_super.__doc__)


@pytest.mark.parametrize(
    "public_class",
    [
        "IdentityEval",
        "HistoryConv",
        "MSplineEval",
        "MSplineConv",
        "BSplineEval",
        "BSplineConv",
        "CyclicBSplineEval",
        "CyclicBSplineConv",
        "RaisedCosineLinearEval",
        "RaisedCosineLinearConv",
        "RaisedCosineLogEval",
        "RaisedCosineLogConv",
        "OrthExponentialEval",
        "OrthExponentialConv",
        "FourierEval",
    ],
)
@pytest.mark.parametrize(
    "method, mixin",
    [("set_input_shape", "AtomicBasisMixin"), ("compute_features", None)],
)
def test_docstrings_decorator_mixinclass(public_class, mixin, method):
    cls_pub = getattr(basis, public_class)
    if mixin is None:
        mixin = "EvalBasisMixin" if public_class.endswith("Eval") else "ConvBasisMixin"
        mixin_meth = getattr(getattr(basis, mixin), "_" + method)
    else:
        mixin_meth = getattr(getattr(basis, mixin), method)
    meth_pub = getattr(cls_pub, method)
    assert meth_pub.__doc__.startswith(mixin_meth.__doc__)


def test_add_docstring():

    class CustomClass:
        def method(self):
            """My extra text."""
            pass

    custom_add_docstring = partial(add_docstring, cls=CustomClass)

    class CustomSubClass(CustomClass):
        @custom_add_docstring("method")
        def method(self):
            """My custom method."""
            pass

    assert CustomSubClass().method.__doc__ == "My extra text.\nMy custom method."
    with pytest.raises(AttributeError, match="CustomClass has no attribute"):

        class CustomSubClass2(CustomClass):
            @custom_add_docstring("unknown", cls=CustomClass)
            def method(self):
                """My custom method."""
                pass

        CustomSubClass2()


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.BSplineEval(10), BSplineBasis),
        (basis.BSplineConv(10, window_size=11), BSplineBasis),
        (basis.CyclicBSplineEval(10), CyclicBSplineBasis),
        (basis.CyclicBSplineConv(10, window_size=11), CyclicBSplineBasis),
        (basis.MSplineEval(10), MSplineBasis),
        (basis.MSplineConv(10, window_size=11), MSplineBasis),
        (basis.RaisedCosineLinearEval(10), RaisedCosineBasisLinear),
        (basis.RaisedCosineLinearConv(10, window_size=11), RaisedCosineBasisLinear),
        (basis.RaisedCosineLogEval(10), RaisedCosineBasisLog),
        (basis.RaisedCosineLogConv(10, window_size=11), RaisedCosineBasisLog),
        (basis.OrthExponentialEval(10, np.arange(1, 11)), OrthExponentialBasis),
        (
            basis.OrthExponentialConv(10, decay_rates=np.arange(1, 11), window_size=12),
            OrthExponentialBasis,
        ),
        (basis.IdentityEval(), IdentityBasis),
        (basis.HistoryConv(11), HistoryBasis),
        (basis.FourierEval(11), FourierBasis),
    ],
)
def test_expected_output_eval_on_grid(basis_instance, super_class):
    x, y = super_class.evaluate_on_grid(basis_instance, 100)
    xx, yy = basis_instance.evaluate_on_grid(100)
    np.testing.assert_equal(xx, x)
    np.testing.assert_equal(np.asarray(yy), np.asarray(y))


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.BSplineEval(10), BSplineBasis),
        (basis.BSplineConv(10, window_size=11), BSplineBasis),
        (basis.CyclicBSplineEval(10), CyclicBSplineBasis),
        (basis.CyclicBSplineConv(10, window_size=11), CyclicBSplineBasis),
        (basis.MSplineEval(10), MSplineBasis),
        (basis.MSplineConv(10, window_size=11), MSplineBasis),
        (basis.RaisedCosineLinearEval(10), RaisedCosineBasisLinear),
        (basis.RaisedCosineLinearConv(10, window_size=11), RaisedCosineBasisLinear),
        (basis.RaisedCosineLogEval(10), RaisedCosineBasisLog),
        (basis.RaisedCosineLogConv(10, window_size=11), RaisedCosineBasisLog),
        (basis.OrthExponentialEval(10, np.arange(1, 11)), OrthExponentialBasis),
        (
            basis.OrthExponentialConv(10, decay_rates=np.arange(1, 11), window_size=12),
            OrthExponentialBasis,
        ),
        (basis.IdentityEval(), IdentityBasis),
        (basis.HistoryConv(11), HistoryBasis),
        (basis.FourierEval(10), FourierBasis),
    ],
)
def test_expected_output_compute_features(basis_instance, super_class):
    x = super_class.compute_features(basis_instance, np.linspace(0, 1, 100))
    xx = basis_instance.compute_features(np.linspace(0, 1, 100))
    nans = np.isnan(x.sum(axis=1))
    assert np.all(np.isnan(xx[nans]))
    np.testing.assert_array_equal(xx[~nans], x[~nans])


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.BSplineEval(10, label="label"), BSplineBasis),
        (basis.BSplineConv(10, window_size=11, label="label"), BSplineBasis),
        (basis.CyclicBSplineEval(10, label="label"), CyclicBSplineBasis),
        (
            basis.CyclicBSplineConv(10, window_size=11, label="label"),
            CyclicBSplineBasis,
        ),
        (basis.MSplineEval(10, label="label"), MSplineBasis),
        (basis.MSplineConv(10, window_size=11, label="label"), MSplineBasis),
        (basis.RaisedCosineLinearEval(10, label="label"), RaisedCosineBasisLinear),
        (
            basis.RaisedCosineLinearConv(10, window_size=11, label="label"),
            RaisedCosineBasisLinear,
        ),
        (basis.RaisedCosineLogEval(10, label="label"), RaisedCosineBasisLog),
        (
            basis.RaisedCosineLogConv(10, window_size=11, label="label"),
            RaisedCosineBasisLog,
        ),
        (
            basis.OrthExponentialEval(10, np.arange(1, 11), label="label"),
            OrthExponentialBasis,
        ),
        (
            basis.OrthExponentialConv(
                10, decay_rates=np.arange(1, 11), window_size=12, label="label"
            ),
            OrthExponentialBasis,
        ),
        (
            basis.OrthExponentialConv(
                10, decay_rates=np.arange(1, 11), window_size=12, label="a"
            )
            * basis.RaisedCosineLogConv(10, window_size=11, label="b"),
            OrthExponentialBasis,
        ),
        (
            basis.OrthExponentialConv(
                10, decay_rates=np.arange(1, 11), window_size=12, label="a"
            )
            + basis.RaisedCosineLogConv(10, window_size=11, label="b"),
            OrthExponentialBasis,
        ),
        (basis.IdentityEval(label="label"), IdentityBasis),
        (basis.HistoryConv(11, label="label"), HistoryBasis),
        (basis.FourierEval(11, label="label"), FourierBasis),
    ],
)
def test_expected_output_split_by_feature(basis_instance, super_class):
    inp = [np.linspace(0, 1, 100)] * basis_instance._n_input_dimensionality
    x = super_class.compute_features(basis_instance, *inp)
    xdict = super_class.split_by_feature(basis_instance, x)
    xxdict = basis_instance.split_by_feature(x)
    assert xdict.keys() == xxdict.keys()
    for k in xdict.keys():
        xx = xxdict[k]
        x = xdict[k]
        nans = np.isnan(x.sum(axis=(1,)))
        assert np.all(np.isnan(xx[nans]))
        np.testing.assert_array_equal(xx[~nans], x[~nans])


@pytest.mark.parametrize("label", [None, "", "default-behavior", "CoolFeature"])
def test_repr_label(label):
    with patch("os.get_terminal_size", return_value=SizeTerminal(80, 24)):
        if label == "default-behavior":
            bas = basis.RaisedCosineLinearEval(n_basis_funcs=5)
        else:
            bas = basis.RaisedCosineLinearEval(n_basis_funcs=5, label=label)
        if label in [None, "default-behavior"]:
            expected = "RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)"
        else:
            expected = f"'{label}': RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)"
        out = repr(bas)
        assert out == expected


@pytest.mark.parametrize(
    "input_shape_1",
    [
        (100,),
        (100, 10),
        (100, 10, 1),
        (100, 1, 10),
    ],
)
@pytest.mark.parametrize("input_shape_2", [(100,), (100, 1), (100, 1, 2), (100, 2, 1)])
def test_composite_split_by_feature(input_shape_1, input_shape_2):
    # by default, jax was sorting the dict we use in split_by_feature for the labels to
    # be alphabetical. thus, if the additive basis was made up of basis objects whose
    # n_basis_input values were different AND whose alphabetical sorting was the
    # different from their order in initialization, it would fail

    comp_basis = basis.RaisedCosineLogEval(10) + basis.CyclicBSplineEval(5)
    X = comp_basis.compute_features(
        np.random.rand(*input_shape_1), np.random.rand(*input_shape_2)
    )
    features = comp_basis.split_by_feature(X)
    # if the user only passes a 1d input, we append the second dim (number of inputs)

    split_shape_1 = tuple(i for i in input_shape_1 + (comp_basis.basis1.n_basis_funcs,))
    split_shape_2 = tuple(i for i in input_shape_2 + (comp_basis.basis2.n_basis_funcs,))
    assert features["RaisedCosineLogEval"].shape == split_shape_1
    assert features["CyclicBSplineEval"].shape == split_shape_2


@pytest.mark.parametrize(
    "input_shape",
    [
        (11,),
        (11, 10),
        (11, 10, 1),
        (11, 1, 10),
    ],
)
def test_composite_split_by_feature_multiply(input_shape):
    # by default, jax was sorting the dict we use in split_by_feature for the labels to
    # be alphabetical. thus, if the additive basis was made up of basis objects whose
    # n_basis_input values were different AND whose alphabetical sorting was the
    # different from their order in initialization, it would fail
    comp_basis = basis.RaisedCosineLogEval(10) * basis.CyclicBSplineEval(5)
    X = comp_basis.compute_features(
        np.random.rand(*input_shape), np.random.rand(*input_shape)
    )
    features = comp_basis.split_by_feature(X)
    # if the user only passes a 1d input, we append the second dim (number of inputs)
    # concatenation of shapes except for the last term which is the product of the num bases
    assert features["(RaisedCosineLogEval * CyclicBSplineEval)"].shape == (
        *input_shape,
        comp_basis.basis1.n_basis_funcs * comp_basis.basis2.n_basis_funcs,
    )


@pytest.mark.parametrize(
    "cls",
    [
        basis.RaisedCosineLogConv,
        basis.RaisedCosineLinearConv,
        basis.BSplineConv,
        basis.CyclicBSplineConv,
        basis.MSplineConv,
        basis.OrthExponentialConv,
        basis.HistoryConv,
    ],
)
class TestConvBasis:
    @pytest.mark.parametrize("n_basis", [5, 6])
    @pytest.mark.parametrize("ws", [10, 20])
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_sklearn_clone_conv(self, cls, n_basis, ws, inp_num):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=ws,
            **extra_kwargs(cls, n_basis),
        )
        bas.set_input_shape(inp_num)
        bas2 = bas.__sklearn_clone__()
        assert id(bas) != id(bas2)
        assert np.all(
            bas.__dict__.pop("decay_rates", True)
            == bas2.__dict__.pop("decay_rates", True)
        )
        f1, f2 = bas.__dict__.pop("_funcs", [True]), bas2.__dict__.pop("_funcs", [True])
        assert all(fi == fj for fi, fj in zip(f1, f2))
        assert bas.__dict__ == bas2.__dict__

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Empty array provided")),
            (1, does_not_raise()),
            (3, does_not_raise()),
        ],
    )
    def test_expected_input_number(self, n_input, expectation, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=10,
            **extra_kwargs(cls, 5),
        )
        x = np.random.randn(20, 2)
        bas.compute_features(x)
        with expectation:
            bas.compute_features(np.random.randn(30, n_input))

    @pytest.mark.parametrize(
        "conv_kwargs, expectation",
        [
            (dict(), does_not_raise()),
            (
                dict(axis=0),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (
                dict(axis=1),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (dict(shift=True), does_not_raise()),
            (
                dict(shift=True, axis=0),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (
                dict(shifts=True),
                pytest.raises(ValueError, match="Unrecognized keyword arguments"),
            ),
            (dict(shift=True, predictor_causality="causal"), does_not_raise()),
            (
                dict(shift=True, time_series=np.arange(10)),
                pytest.raises(ValueError, match="Unrecognized keyword arguments"),
            ),
        ],
    )
    def test_init_conv_kwargs(self, conv_kwargs, expectation, cls):
        with expectation:
            instantiate_atomic_basis(
                cls,
                n_basis_funcs=5,
                window_size=200,
                conv_kwargs=conv_kwargs,
                **extra_kwargs(cls, 5),
            )

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=10,
            **extra_kwargs(cls, 5),
        )
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize(
        "kwargs, input1_shape, expectation",
        [
            (dict(), (10,), does_not_raise()),
            (
                dict(axis=0),
                (10,),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (
                dict(axis=1),
                (2, 10),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
        ],
    )
    def test_compute_features_axis(self, kwargs, input1_shape, expectation, cls):
        with expectation:
            basis_obj = instantiate_atomic_basis(
                cls,
                n_basis_funcs=5,
                window_size=5,
                conv_kwargs=kwargs,
                **extra_kwargs(cls, 5),
            )
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [4, 5])
    @pytest.mark.parametrize("time_scaling", [50, 70])
    @pytest.mark.parametrize("enforce_decay", [True, False])
    @pytest.mark.parametrize("window_size", [10, 15])
    @pytest.mark.parametrize("order", [3, 4])
    @pytest.mark.parametrize("width", [2, 3])
    @pytest.mark.parametrize(
        "input_shape, expected_n_input",
        [
            ((20,), 1),
            ((20, 1), 1),
            ((20, 2), 2),
            ((20, 1, 2), 2),
            ((20, 2, 1), 2),
            ((20, 2, 2), 4),
        ],
    )
    def test_compute_features_conv_input(
        self,
        n_basis_funcs,
        time_scaling,
        enforce_decay,
        window_size,
        input_shape,
        expected_n_input,
        order,
        width,
        cls,
        basis_class_specific_params,
    ):
        x = np.ones(input_shape)

        kwargs = dict(
            n_basis_funcs=n_basis_funcs,
            decay_rates=np.arange(1, n_basis_funcs + 1),
            time_scaling=time_scaling,
            window_size=window_size,
            enforce_decay_to_zero=enforce_decay,
            order=order,
            width=width,
        )

        # figure out which kwargs needs to be removed
        kwargs = inspect_utils.trim_kwargs(cls, kwargs, basis_class_specific_params)

        basis_obj = instantiate_atomic_basis(cls, **kwargs)
        out = basis_obj.compute_features(x)
        assert out.shape[1] == expected_n_input * basis_obj.n_basis_funcs

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (
                None,
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
            (
                (0, 3),
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
            (
                (1, 4),
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
            (
                (1, 3),
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception, cls):
        extra_args = {"n_basis_funcs": 5}
        if cls == HistoryConv:
            extra_args = {}
        with exception:
            cls(
                **extra_args,
                window_size=10,
                bounds=bounds,
                **extra_kwargs(cls, 5),
            )

    def test_convolution_is_performed(self, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=10,
            **extra_kwargs(cls, 5),
        )
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_fit_kernel(self, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=30,
            **extra_kwargs(cls, 5),
        )
        bas._set_kernel()
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self, cls):
        n_basis = 5 if cls != HistoryConv else 30
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=30,
            **extra_kwargs(cls, n_basis),
        )
        bas._set_kernel()
        assert bas.kernel_.shape == (30, n_basis)

    def test_set_window_size(self, cls):
        kwargs = (
            {"window_size": 10, "n_basis_funcs": 10}
            if cls != HistoryConv
            else {"window_size": 10}
        )

        with does_not_raise():
            cls(**kwargs, **extra_kwargs(cls, 10))

        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=10,
            window_size=10,
            **extra_kwargs(cls, 10),
        )
        with pytest.raises(ValueError, match="You must provide a window_siz"):
            bas.set_params(window_size=None)

    def test_transform_fails(self, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=5,
            **extra_kwargs(cls, 5),
        )
        with pytest.raises(
            RuntimeError, match="You must call `setup_basis` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "ws, expectation",
        [
            (5, does_not_raise()),
            (
                -1,
                pytest.raises(
                    ValueError, match="`window_size` must be a positive integer"
                ),
            ),
            (
                None,
                pytest.raises(
                    ValueError,
                    match="You must provide a window_size",
                ),
            ),
            (
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
        ],
    )
    def test_init_window_size(self, ws, expectation, cls):
        extra = dict(n_basis_funcs=5) if cls != HistoryConv else {}
        with expectation:
            cls(**extra, window_size=ws, **extra_kwargs(cls, 5))

    def test_set_bounds(self, cls):
        kwargs = (
            {"bounds": (1, 2), "n_basis_funcs": 10}
            if cls != HistoryConv
            else {"bounds": (1, 2)}
        )
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'bounds'"
        ):
            cls(**kwargs, **extra_kwargs(cls, 10))

        kwargs = {"n_basis_funcs": 10} if cls != IdentityEval else {}
        bas = instantiate_atomic_basis(
            cls,
            **kwargs,
            window_size=20,
            **extra_kwargs(cls, 10),
        )
        with pytest.raises(
            ValueError, match="Invalid parameter 'bounds' for estimator"
        ):
            bas.set_params(bounds=(1, 2))


@pytest.mark.parametrize(
    "cls",
    [
        CustomBasis,
        basis.RaisedCosineLogEval,
        basis.RaisedCosineLinearEval,
        basis.BSplineEval,
        basis.CyclicBSplineEval,
        basis.MSplineEval,
        basis.OrthExponentialEval,
        basis.IdentityEval,
        basis.FourierEval,
    ],
)
class TestEvalBasis:
    @pytest.mark.parametrize(
        "samples, vmin, vmax, expectation",
        [
            (0.5, 0, 1, does_not_raise()),
            (
                -0.5,
                0,
                1,
                pytest.raises(ValueError, match="All the samples lie outside"),
            ),
            (np.linspace(-1, 1, 10), 0, 1, does_not_raise()),
            (
                np.linspace(-1, 0, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
            (
                np.linspace(1, 2, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
        ],
    )
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation, cls):
        if (
            "OrthExp" in cls.__name__ and not hasattr(samples, "shape")
        ) or cls == CustomBasis:
            pytest.skip(f"Skipping test_call_vmin_vmax for {cls.__name__}")
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=(vmin, vmax),
            **extra_kwargs(cls, 5),
        )
        with expectation:
            bas.evaluate(samples)

    @pytest.mark.parametrize("n_basis", [5, 6])
    @pytest.mark.parametrize("vmin, vmax", [(0, 1), (-1, 1)])
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_sklearn_clone_eval(self, cls, n_basis, vmin, vmax, inp_num):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            bounds=(vmin, vmax),
            **extra_kwargs(cls, n_basis),
        )
        bas.set_input_shape(inp_num)
        bas2 = bas.__sklearn_clone__()
        assert id(bas) != id(bas2)
        assert np.all(
            bas.__dict__.pop("decay_rates", True)
            == bas2.__dict__.pop("decay_rates", True)
        )
        f1, f2 = bas.__dict__.pop("_funcs", [True]), bas2.__dict__.pop("_funcs", [True])
        assert all(fi == fj for fi, fj in zip(f1, f2))
        f1, f2 = bas.__dict__.pop("_frequencies", [True]), bas2.__dict__.pop(
            "_frequencies", [True]
        )
        assert all(np.all(fi == fj) for fi, fj in zip(f1, f2))
        f1, f2 = bas.__dict__.pop("_frequency_mask", [True]), bas2.__dict__.pop(
            "_frequency_mask", [True]
        )
        if f1 is not None and f2 is not None:
            assert all(np.all(fi == fj) for fi, fj in zip(f1, f2))
        else:
            assert f1 is f2 is None
        f1, f2 = bas.__dict__.pop("_freq_combinations", [True]), bas2.__dict__.pop(
            "_freq_combinations", [True]
        )
        assert all(np.all(fi == fj) for fi, fj in zip(f1, f2))
        assert bas.__dict__ == bas2.__dict__

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(
        self, bounds, samples, nan_idx, mn, mx, cls
    ):
        if cls == CustomBasis:
            pytest.skip(
                f"Skipping test_vmin_vmax_eval_on_grid_affects_x for {cls.__name__}"
            )
        bas_no_range = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=None,
            **extra_kwargs(cls, 5),
        )
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=bounds,
            **extra_kwargs(cls, 5),
        )
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_eval_on_grid_no_effect_on_eval(
        self, vmin, vmax, samples, nan_idx, cls
    ):
        if cls == CustomBasis:
            pytest.skip(
                f"Skipping test_vmin_vmax_eval_on_grid_no_effect_on_eval for {cls.__name__}"
            )
        # MSPline integrates to 1 on domain so must be excluded from this check
        # Identity also returns the same array, so if the range changes so will
        # evaluate on grid output.
        if "MSpline" in cls.__name__ or "Identity" in cls.__name__:
            return
        bas_no_range = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=None,
            **extra_kwargs(cls, 5),
        )
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=(vmin, vmax),
            **extra_kwargs(cls, 5),
        )
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1, out2)

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            (
                (None, 3),
                pytest.raises(TypeError, match=r"Could not convert|Invalid bounds"),
            ),
            (
                (1, None),
                pytest.raises(TypeError, match=r"Could not convert|Invalid bounds"),
            ),
            ((1, 3), does_not_raise()),
            (
                ("a", 3),
                pytest.raises(TypeError, match="Could not convert|Invalid bounds"),
            ),
            (
                (1, "a"),
                pytest.raises(TypeError, match="Could not convert|Invalid bounds"),
            ),
            (
                ("a", "a"),
                pytest.raises(TypeError, match="Could not convert|Invalid bounds"),
            ),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_vmin_vmax_init for {cls.__name__}")
        with expectation:
            bas = instantiate_atomic_basis(
                cls,
                n_basis_funcs=5,
                bounds=bounds,
                **extra_kwargs(cls, 5),
            )
            assert compare_bounds(bas, bounds)

    @pytest.mark.parametrize(
        "samples, vmin, vmax, expectation",
        [
            (0.5, 0, 1, does_not_raise()),
            (
                -0.5,
                0,
                1,
                pytest.raises(ValueError, match="All the samples lie outside"),
            ),
            (np.linspace(-1, 1, 10), 0, 1, does_not_raise()),
            (
                np.linspace(-1, 0, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
            (
                np.linspace(1, 2, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
        ],
    )
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation, cls):
        if (
            "OrthExp" in cls.__name__ and not hasattr(samples, "shape")
        ) or cls == CustomBasis:
            pytest.skip(f"Skipping test_compute_features_vmin_vmax for {cls.__name__}")
        basis_obj = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=(vmin, vmax),
            **extra_kwargs(cls, 5),
        )
        with expectation:
            basis_obj.compute_features(samples)

    @pytest.mark.parametrize(
        "samples, expectation",
        [
            (np.array([0, 1, 2, 3, 4, 5]), does_not_raise()),
            (
                np.array(["a", "1", "2", "3", "4", "5"]),
                pytest.raises(TypeError, match="Input samples must"),
            ),
        ],
    )
    @pytest.mark.parametrize("n_basis", [6])
    def test_call_input_type(self, samples, expectation, n_basis, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            **extra_kwargs(cls, n_basis),
        )  # Only eval mode is relevant here
        with expectation:
            bas.compute_features(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_compute_features_input(self, eval_input, cls):
        # test only in eval because conv requires at least window_size samples
        # orth exp needs more inputs (orthogonalizaiton impossible otherwise)
        if "OrthExp" in cls.__name__:
            return
        basis_obj = instantiate_atomic_basis(cls, n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_vmin_vmax_range for {cls.__name__}")
        bounds = None if vmin is None else (vmin, vmax)
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=bounds,
            **extra_kwargs(cls, 5),
        )
        out = np.asarray(bas.compute_features(samples))
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            (
                (None, 3),
                pytest.raises(TypeError, match=r"Could not convert|Invalid bounds"),
            ),
            (
                (1, None),
                pytest.raises(TypeError, match=r"Could not convert|Invalid bounds"),
            ),
            ((1, 3), does_not_raise()),
            (
                ("a", 3),
                pytest.raises(TypeError, match="Could not convert|Invalid bounds"),
            ),
            (
                (1, "a"),
                pytest.raises(TypeError, match="Could not convert|Invalid bounds"),
            ),
            (
                ("a", "a"),
                pytest.raises(TypeError, match="Could not convert|Invalid bounds"),
            ),
            (
                (2, 1),
                pytest.raises(
                    ValueError,
                    match=r"Invalid bound \(2.0, 1.0\). Lower bound is greater",
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_vmin_vmax_setter for {cls.__name__}")
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=(1, 3),
            **extra_kwargs(cls, 5),
        )
        with expectation:
            bas.set_params(bounds=bounds)
            assert compare_bounds(bas, bounds)

    def test_conv_kwargs_error(self, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_conv_kwargs_error for {cls.__name__}")
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'test'"
        ):
            if cls in [IdentityEval, FourierEval]:
                extra = {}
            else:
                extra = dict(n_basis_funcs=5)
            cls(**extra, test="hi", **extra_kwargs(cls, 5))

    def test_set_window_size(self, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_set_window_size for {cls.__name__}")
        if cls not in [IdentityEval, FourierEval]:
            kwargs = {"window_size": 10, "n_basis_funcs": 10}
        else:
            kwargs = {"window_size": 10}

        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'window_size'"
        ):
            cls(**kwargs, **extra_kwargs(cls, 10))

        bas = instantiate_atomic_basis(cls, n_basis_funcs=10, **extra_kwargs(cls, 10))
        with pytest.raises(
            ValueError, match="Invalid parameter 'window_size' for estimator"
        ):
            bas.set_params(window_size=10)

    @pytest.mark.parametrize(
        "ws, expectation",
        [
            (
                None,
                pytest.raises(
                    TypeError,
                    match=r"got an unexpected keyword argument 'window_size'",
                ),
            ),
            (
                10,
                pytest.raises(
                    TypeError,
                    match=r"got an unexpected keyword argument 'window_size'",
                ),
            ),
        ],
    )
    def test_init_window_size(self, ws, expectation, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_init_window_size for {cls.__name__}")
        extra = dict(n_basis_funcs=5) if cls not in [IdentityEval, FourierEval] else {}
        with expectation:
            cls(**extra, window_size=ws, **extra_kwargs(cls, 5))

    def test_set_bounds(self, cls):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_set_bounds for {cls.__name__}")
        kwargs = (
            {"bounds": (1, 2), "n_basis_funcs": 10}
            if cls not in [IdentityEval, FourierEval]
            else {"bounds": (1, 2)}
        )
        with does_not_raise():
            cls(**kwargs, **extra_kwargs(cls, 10))


@pytest.mark.parametrize(
    "cls",
    [
        {"eval": basis.RaisedCosineLogEval, "conv": basis.RaisedCosineLogConv},
        {"eval": basis.RaisedCosineLinearEval, "conv": basis.RaisedCosineLinearConv},
        {"eval": basis.BSplineEval, "conv": basis.BSplineConv},
        {"eval": basis.CyclicBSplineEval, "conv": basis.CyclicBSplineConv},
        {"eval": basis.MSplineEval, "conv": basis.MSplineConv},
        {"eval": basis.OrthExponentialEval, "conv": basis.OrthExponentialConv},
        {"eval": basis.IdentityEval, "conv": basis.HistoryConv},
    ],
)
@pytest.mark.parametrize("n_basis", [6])
def test_call_equivalent_in_conv(n_basis, cls):
    # Identity and history have a different behavior
    if cls["eval"] is IdentityEval:
        return
    bas_con = instantiate_atomic_basis(
        cls["conv"],
        n_basis_funcs=n_basis,
        window_size=10,
        **extra_kwargs(cls["conv"], n_basis),
    )
    bas_eval = instantiate_atomic_basis(
        cls["eval"],
        n_basis_funcs=n_basis,
        **extra_kwargs(cls["eval"], n_basis),
    )
    x = np.linspace(0, 1, 10)
    assert np.all(bas_con.evaluate(x) == bas_eval.evaluate(x))


@pytest.mark.parametrize(
    "cls",
    [
        CustomBasis,
        basis.RaisedCosineLogEval,
        basis.RaisedCosineLogConv,
        basis.RaisedCosineLinearEval,
        basis.RaisedCosineLinearConv,
        basis.BSplineEval,
        basis.BSplineConv,
        basis.CyclicBSplineEval,
        basis.CyclicBSplineConv,
        basis.MSplineEval,
        basis.MSplineConv,
        basis.OrthExponentialEval,
        basis.OrthExponentialConv,
        basis.IdentityEval,
        basis.HistoryConv,
        basis.FourierEval,
    ],
)
class TestSharedMethods:

    @pytest.mark.parametrize(
        "expected_out",
        [
            {
                CustomBasis: "CustomBasis(\n    funcs=[partial(power_func, 1), ..., partial(power_func, 5)],\n    ndim_input=1,\n    pynapple_support=True,\n    is_complex=False\n)",
                basis.RaisedCosineLogEval: "RaisedCosineLogEval(n_basis_funcs=5, width=2.0, time_scaling=50.0, enforce_decay_to_zero=True, bounds=(1.0, 2.0))",
                basis.RaisedCosineLinearEval: "RaisedCosineLinearEval(n_basis_funcs=5, width=2.0, bounds=(1.0, 2.0))",
                basis.BSplineEval: "BSplineEval(n_basis_funcs=5, order=4, bounds=(1.0, 2.0))",
                basis.CyclicBSplineEval: "CyclicBSplineEval(n_basis_funcs=5, order=4, bounds=(1.0, 2.0))",
                basis.MSplineEval: "MSplineEval(n_basis_funcs=5, order=4, bounds=(1.0, 2.0))",
                basis.OrthExponentialEval: "OrthExponentialEval(n_basis_funcs=5, bounds=(1.0, 2.0))",
                basis.IdentityEval: "IdentityEval(bounds=(1.0, 2.0))",
                basis.RaisedCosineLogConv: "RaisedCosineLogConv(n_basis_funcs=5, window_size=10, width=2.0, time_scaling=50.0, enforce_decay_to_zero=True)",
                basis.RaisedCosineLinearConv: "RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0)",
                basis.BSplineConv: "BSplineConv(n_basis_funcs=5, window_size=10, order=4)",
                basis.CyclicBSplineConv: "CyclicBSplineConv(n_basis_funcs=5, window_size=10, order=4)",
                basis.MSplineConv: "MSplineConv(n_basis_funcs=5, window_size=10, order=4)",
                basis.OrthExponentialConv: "OrthExponentialConv(n_basis_funcs=5, window_size=10)",
                basis.HistoryConv: "HistoryConv(window_size=10)",
                basis.FourierEval: "FourierEval(frequencies=[Array([1., 2.], dtype=float32)], ndim=1, bounds=((1.0, 2.0),), frequency_mask='no-intercept')",
            }
        ],
    )
    def test_repr_out(self, cls, expected_out):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=(1, 2),
            window_size=10,
            **extra_kwargs(cls, 5),
        )
        out = repr(bas)
        assert out.startswith(expected_out.get(cls, ""))

    @pytest.mark.parametrize(
        "expected_out",
        [
            {
                CustomBasis: "'mylabel': CustomBasis(\n    funcs=[partial(power_func, 1), ..., partial(power_func, 5)",
                basis.RaisedCosineLogEval: "'mylabel': RaisedCosineLogEval(n_basis_funcs=5, width=2.0, time_scaling=50.0, enforce_decay_to_zero=True, bounds=(1.0, 2.0))",
                basis.RaisedCosineLinearEval: "'mylabel': RaisedCosineLinearEval(n_basis_funcs=5, width=2.0, bounds=(1.0, 2.0))",
                basis.BSplineEval: "'mylabel': BSplineEval(n_basis_funcs=5, order=4, bounds=(1.0, 2.0))",
                basis.CyclicBSplineEval: "'mylabel': CyclicBSplineEval(n_basis_funcs=5, order=4, bounds=(1.0, 2.0))",
                basis.MSplineEval: "'mylabel': MSplineEval(n_basis_funcs=5, order=4, bounds=(1.0, 2.0))",
                basis.OrthExponentialEval: "'mylabel': OrthExponentialEval(n_basis_funcs=5, bounds=(1.0, 2.0))",
                basis.IdentityEval: "'mylabel': IdentityEval(bounds=(1.0, 2.0))",
                basis.RaisedCosineLogConv: "'mylabel': RaisedCosineLogConv(n_basis_funcs=5, window_size=10, width=2.0, time_scaling=50.0, enforce_decay_to_zero=True)",
                basis.RaisedCosineLinearConv: "'mylabel': RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0)",
                basis.BSplineConv: "'mylabel': BSplineConv(n_basis_funcs=5, window_size=10, order=4)",
                basis.CyclicBSplineConv: "'mylabel': CyclicBSplineConv(n_basis_funcs=5, window_size=10, order=4)",
                basis.MSplineConv: "'mylabel': MSplineConv(n_basis_funcs=5, window_size=10, order=4)",
                basis.OrthExponentialConv: "'mylabel': OrthExponentialConv(n_basis_funcs=5, window_size=10)",
                basis.HistoryConv: "'mylabel': HistoryConv(window_size=10)",
                basis.FourierEval: "'mylabel': FourierEval(frequencies=[Array([1., 2.], dtype=float32)], ndim=1, bounds=((1.0, 2.0),), frequency_mask='no-intercept')",
            }
        ],
    )
    def test_repr_out_with_label(self, cls, expected_out):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            bounds=(1, 2),
            window_size=10,
            label="mylabel",
            **extra_kwargs(cls, 5),
        )
        out = repr(bas)
        assert out.startswith(expected_out.get(cls, ""))

    @pytest.mark.parametrize("n_basis", [5])
    @pytest.mark.parametrize("ws", [10])
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_len(self, cls, n_basis, ws, inp_num):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=ws,
            **extra_kwargs(cls, n_basis),
        )
        assert len(bas) == 1

    @pytest.mark.parametrize("n_basis", [5])
    @pytest.mark.parametrize("ws", [10])
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_iter(self, cls, n_basis, ws, inp_num):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=ws,
            **extra_kwargs(cls, n_basis),
        )
        for b in bas:
            assert id(bas) == id(b)

    @pytest.mark.parametrize(
        "attribute, value, expectation",
        [
            ("label", None, does_not_raise()),
            ("label", "label", does_not_raise()),
            (
                "n_output_features",
                5,
                pytest.raises(
                    AttributeError,
                    match=r"can't set attribute 'n_output_features'|property 'n_output_features' of '.+' object",
                ),
            ),
        ],
    )
    def test_attr_setter(self, attribute, value, cls, expectation):
        if cls == CustomBasis:
            pytest.skip(f"Skipping test_attr_setter for {cls.__name__}")
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            **extra_kwargs(cls, 5),
            window_size=10,
        )
        with expectation:
            setattr(bas, attribute, value)

        if expectation is does_not_raise():
            if value is not None:
                assert getattr(bas, attribute) == value

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            label=label,
            **extra_kwargs(cls, 5),
            window_size=10,
        )
        expected_label = str(label) if label is not None else cls.__name__
        assert bas.label == expected_label

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=10,
            **extra_kwargs(cls, 5),
        )

        ishape_prod = getattr(bas, "_input_shape_product", None)
        assert ishape_prod is None
        bas.compute_features(np.random.randn(20, n_input))
        ishape_prod = getattr(bas, "_input_shape_product", None)
        if ishape_prod:
            assert bas._input_shape_product == (n_input,)

    @pytest.mark.parametrize("n_basis", [6, 7])
    def test_call_basis_number(self, n_basis, cls):
        if cls is IdentityEval:
            n_basis = 1
        elif cls is HistoryConv:
            n_basis = 8
        elif issubclass(cls, FourierBasis):
            # In the instantiate_atomic_basis, the number of frequencies is set
            # to np.arange(1, 1 + n_basis // 2), so only even n_basis works for this
            # test.
            n_basis = n_basis + n_basis % 2

        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=8,
            **extra_kwargs(cls, n_basis),
        )
        x = np.linspace(0, 1, 10)
        assert bas.evaluate(x).shape[1] == n_basis

    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (
                0,
                pytest.raises(
                    TypeError,
                    match=r"missing 1 required positional argument|This basis requires \d+ input\(s\)",
                ),
            ),
            (1, does_not_raise()),
            (
                2,
                pytest.raises(
                    TypeError,
                    match=r"takes 2 positional arguments but 3 were given|This basis requires \d+ input\(s\)",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("n_basis", [6])
    def test_call_input_num(self, num_input, n_basis, expectation, cls):
        bas = instantiate_atomic_basis(
            cls,
            window_size=8,
            n_basis_funcs=n_basis,
            **extra_kwargs(cls, n_basis),
        )
        with expectation:
            bas.evaluate(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], does_not_raise()),
            (np.repeat(np.linspace(0, 1, 10), 10).reshape(10, 5, 2), does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("n_basis", [6])
    def test_call_input_shape(self, inp, expectation, n_basis, cls):
        if cls == CustomBasis:
            pytest.skip(
                f"Skipping test_call_input_shape for {cls.__name__}.\n"
                f"The `evaluate` call of custom basis concatenate the vectorized outputs."
            )
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=8,
            **extra_kwargs(cls, n_basis),
        )
        if isinstance(bas, IdentityEval):
            n_basis = 1
        elif isinstance(bas, HistoryConv):
            n_basis = 8
            if inp.ndim != 1:
                return
        with expectation:
            out = bas.evaluate(inp)
            assert out.shape == tuple((*inp.shape, n_basis))
            out2 = bas.evaluate_on_grid(inp.shape[0])[1]
            assert np.all((out.reshape(out.shape[0], -1, n_basis) - out2[:, None]) == 0)

    @pytest.mark.parametrize("n_basis", [6])
    def test_call_nan_location(self, n_basis, cls):
        if cls is HistoryConv or cls is CustomBasis:
            return
        if cls is IdentityEval:
            n_basis = 1
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=8,
            **extra_kwargs(cls, n_basis),
        )
        inp = np.random.randn(10, 2, 3)
        inp[2, 0, [0, 2]] = np.nan
        inp[4, 1, 1] = np.nan
        out = bas.evaluate(inp)
        assert np.all(np.isnan(out[2, 0, [0, 2]]))
        assert np.all(np.isnan(out[4, 1, 1]))
        assert np.isnan(out).sum() == 3 * n_basis

    def test_call_nan(self, cls):
        if cls is HistoryConv:
            # eval simply returns the evaluate...
            return
        elif cls is IdentityEval:
            n_basis = 1
        else:
            n_basis = 5
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=8,
            **extra_kwargs(cls, n_basis),
        )
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas.evaluate(x)[3]))

    @pytest.mark.parametrize("n_basis", [6, 7])
    def test_call_non_empty(self, n_basis, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=8,
            **extra_kwargs(cls, n_basis),
        )
        meth = bas.compute_features if cls == CustomBasis else bas.evaluate
        with pytest.raises(ValueError, match="All sample provided must"):
            meth(np.array([]))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    def test_call_sample_axis(self, time_axis_shape, cls):
        bas = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=8, **extra_kwargs(cls, 5)
        )
        assert (
            bas.evaluate(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape
        )

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
        ],
    )
    def test_call_sample_range(self, mn, mx, expectation, cls):
        bas = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=8, **extra_kwargs(cls, 5)
        )
        with expectation:
            bas.evaluate(np.linspace(mn, mx, 10))

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [6, 10, 13]],
    )
    def test_compute_features_returns_expected_number_of_basis(
        self, args, sample_size, cls
    ):
        args_copy = args.copy()
        if cls == IdentityEval:
            args_copy["n_basis_funcs"] = 1
        elif cls == HistoryConv:
            args_copy["n_basis_funcs"] = 30
        elif issubclass(cls, FourierBasis):
            args_copy["n_basis_funcs"] = (
                args_copy["n_basis_funcs"] + args_copy["n_basis_funcs"] % 2
            )
        basis_obj = instantiate_atomic_basis(
            cls,
            **args_copy,
            window_size=30,
            **extra_kwargs(cls, args_copy["n_basis_funcs"]),
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        assert eval_basis.shape[1] == args_copy["n_basis_funcs"], (
            "Dimensions do not agree: The number of basis should match the first dimension "
            f"of the evaluated basis. The number of basis is {args['n_basis_funcs']}, but the "
            f"evaluated basis has dimension {eval_basis.shape[1]}"
        )

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size, cls):
        if "OrthExp" in cls.__name__ or cls == CustomBasis:
            pytest.skip(f"Skipping test_evaluate_on_grid_basis_size for {cls.__name__}")
        basis_obj = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=8, **extra_kwargs(cls, 5)
        )
        if sample_size <= 0:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input, cls):
        if cls == CustomBasis:
            pytest.skip(
                f"Skipping test_evaluate_on_grid_input_number for {cls.__name__}"
            )
        basis_obj = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=5, **extra_kwargs(cls, 5)
        )
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r".*evaluate_on_grid\(\) missing 1 required positional argument| but 0 were provided",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r".*evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given|but 2 were provided",
            )
        else:
            expectation = does_not_raise()

        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size, cls):
        if "OrthExp" in cls.__name__ or cls == CustomBasis:
            pytest.skip(
                f"Skipping test_evaluate_on_grid_meshgrid_size for {cls.__name__}"
            )
        basis_obj = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=5, **extra_kwargs(cls, 5)
        )
        if sample_size <= 0:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("samples", [[], [0] * 10, [0] * 11])
    def test_non_empty_samples(self, samples, cls):
        if "OrthExp" in cls.__name__:
            pytest.skip(f"Skipping test_non_empty_samples for {cls.__name__}")
        if cls.__name__.endswith("Conv") and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                instantiate_atomic_basis(
                    cls,
                    n_basis_funcs=5,
                    window_size=5,
                    **extra_kwargs(cls, 5),
                ).compute_features(samples)
        else:
            instantiate_atomic_basis(
                cls,
                n_basis_funcs=5,
                window_size=5,
                **extra_kwargs(cls, 5),
            ).compute_features(samples)

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_compute_features(self, n_input, cls):
        basis_obj = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=6, **extra_kwargs(cls, 5)
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"missing 1 required positional argument|This basis requires \d+ input\(s\)",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"takes 2 positional arguments but \d were given|This basis requires \d+ input\(s\)",
            )
        else:
            expectation = does_not_raise()

        with expectation:
            basis_obj.compute_features(*inputs)

    def test_pynapple_support(self, cls):
        bas = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=6, **extra_kwargs(cls, 5)
        )
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas.evaluate(x)
        y_nap = bas.evaluate(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size, cls):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis,
            window_size=10,
            **extra_kwargs(cls, n_basis),
        ).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.array_equal(
            out.time_support.values, inp.time_support.values, equal_nan=True
        )

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [5, 10, 80])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, cls
    ):
        basis_obj = instantiate_atomic_basis(
            cls,
            n_basis_funcs=n_basis_funcs,
            window_size=90,
            **extra_kwargs(cls, n_basis_funcs),
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        assert eval_basis.shape[0] == sample_size, (
            f"Dimensions do not agree: The sample size of the output should match the input sample size. "
            f"Expected {sample_size}, but got {eval_basis.shape[0]}."
        )

    @pytest.mark.parametrize(
        "enforce_decay_to_zero, time_scaling, width, n_basis_funcs, bounds, decay_rates",
        [
            (False, 15, 4, 10, (1, 2), np.arange(1, 11)),
            (False, 15, 4, 10, None, np.arange(1, 11)),
        ],
    )
    @pytest.mark.parametrize(
        "order, conv_kwargs",
        [
            (10, dict(shift=True)),
        ],
    )
    def test_set_params(
        self,
        enforce_decay_to_zero,
        time_scaling,
        width,
        n_basis_funcs,
        bounds,
        order,
        decay_rates,
        conv_kwargs,
        cls,
        basis_class_specific_params,
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            enforce_decay_to_zero=enforce_decay_to_zero,
            time_scaling=time_scaling,
            width=width,
            window_size=10,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
            order=order,
            decay_rates=decay_rates,
            conv_kwargs=conv_kwargs,
        )
        pars = {
            key: value
            for key, value in pars.items()
            if key in basis_class_specific_params[cls.__name__]
        }

        keys = list(pars.keys())
        bas = instantiate_atomic_basis(cls, **pars)
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas = bas.set_params(**par_set)
                assert isinstance(bas, cls)

    def test_transformer_get_params(self, cls):
        bas = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=10,
            **extra_kwargs(cls, 5),
        )
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("basis")
        funcs_transf = (
            params_transf.pop("funcs") if hasattr(bas_transformer, "funcs") else None
        )
        params_basis = bas.get_params()
        funcs_orig = params_basis.pop("funcs") if hasattr(bas, "funcs") else None
        rates_1 = params_basis.pop("decay_rates", 1)
        rates_2 = params_transf.pop("decay_rates", 1)
        freqs_1 = params_basis.pop("frequencies", [1])
        freqs_2 = params_transf.pop("frequencies", [1])
        assert params_transf == params_basis
        assert np.all(rates_1 == rates_2)
        assert all(np.all(f1 == f2) for f1, f2 in zip(freqs_1, freqs_2))
        if funcs_orig:
            assert all(
                f1.keywords == f2.keywords for f1, f2 in zip(funcs_orig, funcs_transf)
            )

    @pytest.mark.parametrize(
        "x, inp_shape, expectation",
        [
            (np.ones((10,)), 1, does_not_raise()),
            (
                np.ones((10, 1)),
                1,
                does_not_raise(),
            ),
            (np.ones((10, 2)), 2, does_not_raise()),
            (
                np.ones((10, 1)),
                2,
                does_not_raise(),
            ),
            (
                np.ones((10, 2, 1)),
                2,
                does_not_raise(),
            ),
            (
                np.ones((10, 1, 2)),
                2,
                does_not_raise(),
            ),
            (np.ones((10, 1)), (1,), does_not_raise()),
            (np.ones((10,)), tuple(), does_not_raise()),
            (np.ones((10,)), np.zeros((12,)), does_not_raise()),
            (
                np.ones((10,)),
                (1,),
                does_not_raise(),
            ),
            (
                np.ones((10, 1)),
                (),
                does_not_raise(),
            ),
            (
                np.ones((10, 1)),
                np.zeros((12,)),
                does_not_raise(),
            ),
            (
                np.ones((10)),
                np.zeros((12, 1)),
                does_not_raise(),
            ),
        ],
    )
    def test_input_shape_validity(self, x, inp_shape, expectation, cls):
        bas = instantiate_atomic_basis(
            cls, n_basis_funcs=5, window_size=8, **extra_kwargs(cls, 5)
        )
        bas.set_input_shape(inp_shape)
        with expectation:
            bas.compute_features(x)

    @pytest.mark.parametrize(
        "inp_shape, expectation",
        [
            ((1, 1), does_not_raise()),
            (
                (1, 1.0),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
            (np.ones((1,)), does_not_raise()),
            (np.ones((1, 1)), does_not_raise()),
        ],
    )
    def test_set_input_value_types(self, inp_shape, expectation, cls):
        bas = instantiate_atomic_basis(
            cls,
            window_size=10,
            n_basis_funcs=5,
            **extra_kwargs(cls, 5),
        )
        with expectation:
            bas.set_input_shape(inp_shape)

    def test_iterate_over_component(self, cls):
        basis_obj = instantiate_atomic_basis(
            cls,
            n_basis_funcs=5,
            window_size=6,
            **extra_kwargs(cls, 5),
        )

        out = tuple(basis_obj._iterate_over_components())
        assert len(out) == 1
        assert id(out[0]) == id(basis_obj)


class TestIdentityBasis(BasisFuncsTesting):
    cls = {"eval": IdentityEval}

    def test_n_basis_not_settable(self):
        bas = IdentityEval()
        with pytest.raises(AttributeError):
            bas.n_basis_funcs = 11

    @pytest.mark.parametrize(
        "inp",
        [
            np.random.randn(10, 2),
            np.random.randn(10, 2, 3),
            np.random.randn(
                10,
            ),
        ],
    )
    def test_comp_feature_output(self, inp):
        bas = IdentityEval()
        np.testing.assert_array_equal(
            bas.compute_features(inp),
            inp.reshape(inp.shape[0], -1),
        )


class TestHistoryBasis(BasisFuncsTesting):
    cls = {"conv": HistoryConv}

    def test_n_basis_not_settable(self):
        bas = HistoryConv(window_size=8)
        with pytest.raises(AttributeError):
            bas.n_basis_funcs = 11

    @pytest.mark.parametrize(
        "ws, expectation",
        [
            (None, pytest.raises(ValueError, match="You must provide a window_size")),
            (7, does_not_raise()),
            (
                0.5,
                pytest.raises(
                    ValueError, match="`window_size` must be a positive integer"
                ),
            ),
        ],
    )
    def test_window_size_setter(self, ws, expectation):
        bas = HistoryConv(window_size=8)
        with expectation:
            bas.window_size = ws
        bas.window_size = 12
        assert bas.n_basis_funcs == 12

    @pytest.mark.parametrize(
        "inp",
        [
            np.random.randn(10, 2),
            np.random.randn(10, 2, 3),
            np.random.randn(
                10,
            ),
        ],
    )
    def test_comp_feature_output(self, inp):
        bas = IdentityEval()
        np.testing.assert_array_equal(
            bas.compute_features(inp),
            inp.reshape(inp.shape[0], -1),
        )


class TestRaisedCosineLogBasis(BasisFuncsTesting):
    cls = {"eval": basis.RaisedCosineLogEval, "conv": basis.RaisedCosineLogConv}

    @pytest.mark.parametrize("width", [1.5, 2, 2.5])
    def test_decay_to_zero_basis_number_match(self, width):
        n_basis_funcs = 10
        _, ev = self.cls["conv"](
            n_basis_funcs=n_basis_funcs,
            width=width,
            enforce_decay_to_zero=True,
            window_size=5,
        ).evaluate_on_grid(2)
        assert ev.shape[1] == n_basis_funcs, (
            "Basis function number mismatch. "
            f"Expected {n_basis_funcs}, got {ev.shape[1]} instead!"
        )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, kwargs
    ):
        if n_basis_funcs < 2:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls[mode].__name__} requires >= 2 basis elements.",
            ):
                self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)
        else:
            self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (10, does_not_raise()),
            (10.5, does_not_raise()),
            (
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width. Width must be strictly greater",
                ),
            ),
            (
                10.3,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width. Width must be strictly greater",
                ),
            ),
            (
                -10,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width. Width must be strictly greater",
                ),
            ),
            (None, pytest.raises(TypeError, match="'<=' not supported between")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_set_width(self, width, expectation, mode, kwargs):
        basis_obj = self.cls[mode](n_basis_funcs=5, **kwargs)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)

    def test_time_scaling_property(self):
        time_scaling = [0.1, 10, 100]
        n_basis_funcs = 5
        _, lin_ev = basis.RaisedCosineLinearEval(n_basis_funcs).evaluate_on_grid(100)
        corr = np.zeros(len(time_scaling))
        for idx, ts in enumerate(time_scaling):
            basis_log = self.cls["eval"](
                n_basis_funcs=n_basis_funcs,
                time_scaling=ts,
                enforce_decay_to_zero=False,
            )
            _, log_ev = basis_log.evaluate_on_grid(100)
            corr[idx] = (lin_ev.flatten() @ log_ev.flatten()) / (
                np.linalg.norm(lin_ev.flatten()) * np.linalg.norm(log_ev.flatten())
            )
        assert np.all(
            np.diff(corr) < 0
        ), "As time scales increases, deviation from linearity should increase!"

    @pytest.mark.parametrize(
        "time_scaling, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="Only strictly positive time_scaling are allowed"
                ),
            ),
            (
                0,
                pytest.raises(
                    ValueError, match="Only strictly positive time_scaling are allowed"
                ),
            ),
            (0.1, does_not_raise()),
            (10, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_time_scaling_values(self, time_scaling, expectation, mode, kwargs):
        with expectation:
            self.cls[mode](n_basis_funcs=5, time_scaling=time_scaling, **kwargs)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (-1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0.5, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1.5, does_not_raise()),
            (2, does_not_raise()),
            (2.1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_width_values(self, width, expectation, mode, kwargs):
        with expectation:
            self.cls[mode](n_basis_funcs=5, width=width, **kwargs)


class TestRaisedCosineLinearBasis(BasisFuncsTesting):
    cls = {"eval": basis.RaisedCosineLinearEval, "conv": basis.RaisedCosineLinearConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, kwargs
    ):
        if n_basis_funcs < 2:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls[mode].__name__} requires >= 2 basis elements.",
            ):
                self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)
        else:
            self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (10, does_not_raise()),
            (10.5, does_not_raise()),
            (
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width. Width must be strictly greater",
                ),
            ),
            (
                -10,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width. Width must be strictly greater",
                ),
            ),
        ],
    )
    def test_set_width(self, width, expectation):
        basis_obj = self.cls["eval"](n_basis_funcs=5)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (-1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0.5, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1.5, does_not_raise()),
            (2, does_not_raise()),
            (2.1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_width_values(self, width, expectation, mode, kwargs):
        """
        Test allowable widths: integer multiple of 1/2, greater than 1.
        This test validates the behavior of both `eval` and `conv` modes.
        """
        basis_obj = self.cls[mode](n_basis_funcs=5, **kwargs)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)


class TestMSplineBasis(BasisFuncsTesting):
    cls = {"eval": basis.MSplineEval, "conv": basis.MSplineConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, kwargs
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1)
        and order < #basis are enforced.
        """
        raise_exception = (order < 1) | (n_basis_funcs < 1) | (order > n_basis_funcs)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Spline order must be positive!|"
                rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls[mode](
                    n_basis_funcs=n_basis_funcs, order=order, **kwargs
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))

            # test the setter valuerror
            if (order > 1) & (n_basis_funcs > 1):
                basis_obj = self.cls[mode](n_basis_funcs=20, order=order, **kwargs)
                with pytest.raises(
                    ValueError,
                    match=rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
                ):
                    basis_obj.n_basis_funcs = n_basis_funcs
        else:
            basis_obj = self.cls[mode](
                n_basis_funcs=n_basis_funcs, order=order, **kwargs
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the order must be positive and less than or equal to the number of basis functions.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [5])
    @pytest.mark.parametrize(
        "order, expectation",
        [
            (1.5, pytest.raises(ValueError, match=r"Spline order must be an integer")),
            (-1, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (0, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                10,
                pytest.raises(
                    ValueError,
                    match=r"[a-z]+|[A-Z]+ `order` parameter cannot be larger",
                ),
            ),
        ],
    )
    def test_order_setter(self, n_basis_funcs, order, expectation):
        basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_compute_features_requirements(self, sample_range):
        """
        Verifies that the compute_features() method can handle input range.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, scaling",
        [
            (None, np.arange(5), [4], 1),
            ((1, 4), np.arange(5), [0], 3),
            ((1, 3), np.arange(5), [0, 4], 2),
        ],
    )
    def test_vmin_vmax_eval_on_grid_scaling_effect_on_eval(
        self, bounds, samples, nan_idx, scaling
    ):
        """
        Check that the MSpline has the expected scaling property.

        The MSpline must integrate to one. If the support is reduced, the height of the spline increases.
        """
        bas_no_range = self.cls["eval"](5, bounds=None)
        bas = self.cls["eval"](5, bounds=bounds)
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1 * scaling, out2)


class TestOrthExponentialBasis(BasisFuncsTesting):
    cls = {"eval": basis.OrthExponentialEval, "conv": basis.OrthExponentialConv}

    def test_check_window_size_after_init(self):
        decay_rates = np.asarray(np.arange(1, 5 + 1), dtype=float)
        expectation = pytest.raises(
            ValueError,
            match="OrthExponentialConv basis requires at least a window_size",
        )
        bas = self.cls["conv"](n_basis_funcs=5, decay_rates=decay_rates, window_size=10)
        with expectation:
            bas.window_size = 4

    @pytest.mark.parametrize(
        "window_size, n_basis, expectation",
        [
            (
                4,
                5,
                pytest.raises(
                    ValueError,
                    match="OrthExponentialConv basis requires at least a window_size",
                ),
            ),
            (5, 5, does_not_raise()),
        ],
    )
    def test_window_size_at_init(self, window_size, n_basis, expectation):
        decay_rates = np.asarray(np.arange(1, n_basis + 1), dtype=float)
        obj = self.cls["conv"](
            n_basis_funcs=n_basis, decay_rates=decay_rates, window_size=n_basis + 1
        )
        with expectation:
            obj.window_size = window_size

        with expectation:
            obj.set_params(window_size=window_size)

    @pytest.mark.parametrize(
        "decay_rates", [[1, 2, 3], [0.01, 0.02, 0.001], [2, 1, 1, 2.4]]
    )
    def test_decay_rate_repetition(self, decay_rates):
        """
        Tests whether the class instance correctly processes the decay rates without repetition.
        A repeated rate causes linear algebra issues, and should raise a ValueError exception.
        """
        decay_rates = np.asarray(decay_rates, dtype=float)
        raise_exception = len(set(decay_rates)) != len(decay_rates)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Two or more rates are repeated! Repeating rates will",
            ):
                self.cls["eval"](
                    n_basis_funcs=len(decay_rates), decay_rates=decay_rates
                )
        else:
            self.cls["eval"](n_basis_funcs=len(decay_rates), decay_rates=decay_rates)

    @pytest.mark.parametrize(
        "decay_rates", [[], [1], [1, 2, 3], [1, 0.01, 0.02, 0.001]]
    )
    @pytest.mark.parametrize("n_basis_funcs", [1, 2, 3, 4])
    def test_decay_rate_size_match_n_basis_funcs(self, decay_rates, n_basis_funcs):
        """
        Tests whether the size of decay rates matches the number of basis functions.
        """
        raise_exception = len(decay_rates) != n_basis_funcs
        decay_rates = np.asarray(decay_rates, dtype=float)
        if raise_exception:
            with pytest.raises(
                ValueError, match="The number of basis functions must match the"
            ):
                self.cls["eval"](n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)
        else:
            self.cls["eval"](n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 30})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, kwargs
    ):
        """
        Tests whether the class instance has a minimum number of basis functions.
        """
        raise_exception = n_basis_funcs < 1
        decay_rates = np.arange(1, 1 + n_basis_funcs) if n_basis_funcs > 0 else []
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls[mode].__name__} requires >= 1 basis elements.",
            ):
                self.cls[mode](
                    n_basis_funcs=n_basis_funcs,
                    decay_rates=decay_rates,
                    **kwargs,
                )
        else:
            self.cls[mode](
                n_basis_funcs=n_basis_funcs,
                decay_rates=decay_rates,
                **kwargs,
            )


class TestBSplineBasis(BasisFuncsTesting):
    cls = {"eval": basis.BSplineEval, "conv": basis.BSplineConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, kwargs
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order > n_basis_funcs
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls[mode](
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    **kwargs,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls[mode](
                n_basis_funcs=n_basis_funcs,
                order=order,
                **kwargs,
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [5])
    @pytest.mark.parametrize(
        "order, expectation",
        [
            (1.5, pytest.raises(ValueError, match=r"Spline order must be an integer")),
            (-1, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (0, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                10,
                pytest.raises(
                    ValueError,
                    match=r"[a-z]+|[A-Z]+ `order` parameter cannot be larger",
                ),
            ),
        ],
    )
    def test_order_setter(self, n_basis_funcs, order, expectation):
        basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_compute_features_requirements(
        self, sample_range: tuple
    ):
        """
        Verifies that the compute_features() method can handle input range.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))


class TestCyclicBSplineBasis(BasisFuncsTesting):
    cls = {"eval": basis.CyclicBSplineEval, "conv": basis.CyclicBSplineConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, kwargs
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1)
        and order < #basis are enforced.
        """
        raise_exception = order > n_basis_funcs
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls[mode](
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    **kwargs,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls[mode](
                n_basis_funcs=n_basis_funcs,
                order=order,
                **kwargs,
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_1_invalid(self, n_basis_funcs, order):
        """
        Verifies that order >= 2 is required for cyclic B-splines.
        """
        raise_exception = order == 1
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"Order >= 2 required for cyclic B-spline"
            ):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 2, 3])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the order is positive and < #basis.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [5])
    @pytest.mark.parametrize(
        "order, expectation",
        [
            (1.5, pytest.raises(ValueError, match=r"Spline order must be an integer")),
            (-1, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (0, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                10,
                pytest.raises(
                    ValueError,
                    match=r"[a-z]+|[A-Z]+ `order` parameter cannot be larger",
                ),
            ),
        ],
    )
    def test_order_setter(self, n_basis_funcs, order, expectation):
        """
        Verifies that setting `order` validates the value correctly.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_compute_features_requirements(
        self, sample_range: tuple
    ):
        """
        Verifies that the compute_features() method can handle input ranges.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))


class TestFourierBasis(BasisFuncsTesting):
    cls = {"eval": FourierEval}

    @pytest.mark.parametrize("mode", ["eval"])
    @pytest.mark.parametrize(
        "frequency_mask",
        [None, np.random.binomial(n=1, p=0.5, size=5), lambda x: x < 3],
    )
    def test_sklearn_clone_freq_mask(self, frequency_mask, mode):
        if frequency_mask is None:
            n_basis = 10
        elif callable(frequency_mask):
            n_basis = 3
        else:
            n_basis = frequency_mask.shape[0] * 2
        bas = instantiate_atomic_basis(
            self.cls[mode],
            **extra_kwargs(self.cls[mode], n_basis),
            frequency_mask=frequency_mask,
        )
        bas.set_input_shape(1)
        bas2 = bas.__sklearn_clone__()
        assert id(bas) != id(bas2)
        assert np.all(
            bas.__dict__.pop("decay_rates", True)
            == bas2.__dict__.pop("decay_rates", True)
        )
        f1, f2 = bas.__dict__.pop("_funcs", [True]), bas2.__dict__.pop("_funcs", [True])
        assert all(fi == fj for fi, fj in zip(f1, f2))
        f1, f2 = bas.__dict__.pop("_frequencies", [True]), bas2.__dict__.pop(
            "_frequencies", [True]
        )
        assert all(np.all(fi == fj) for fi, fj in zip(f1, f2))
        f1, f2 = bas.__dict__.pop("_frequency_mask", [True]), bas2.__dict__.pop(
            "_frequency_mask", [True]
        )
        if f1 is not None and f2 is not None and not callable(f1):
            assert all(np.all(fi == fj) for fi, fj in zip(f1, f2))
        elif callable(f1):
            assert f2 is f1
        else:
            assert f1 is f2 is None
        f1, f2 = bas.__dict__.pop("_freq_combinations", [True]), bas2.__dict__.pop(
            "_freq_combinations", [True]
        )
        assert all(np.all(fi == fj) for fi, fj in zip(f1, f2))
        assert bas.__dict__ == bas2.__dict__

    @pytest.mark.parametrize("mode", ["eval"])
    @pytest.mark.parametrize(
        "ndim, frequencies, expectation",
        [
            (1, 10, does_not_raise()),
            (1, (1, 10), does_not_raise()),
            (1, [1, 10], pytest.raises(ValueError, match="Length of frequencies list")),
            (1, np.arange(1, 10), does_not_raise()),
            (1, [10], does_not_raise()),
            (1, [(1, 10)], does_not_raise()),
            (1, [np.arange(1, 10)], does_not_raise()),
            (
                1,
                (np.arange(1, 10),),
                pytest.raises(
                    ValueError,
                    match="must be a 2-element tuple of non-negative integers",
                ),
            ),
            (2, 10, does_not_raise()),
            (2, (1, 10), does_not_raise()),
            (2, [1, 10], does_not_raise()),
            (2, np.arange(1, 10), does_not_raise()),
            (2, [10], pytest.raises(ValueError, match="Length of frequencies list")),
            (
                2,
                [(1, 10)],
                pytest.raises(ValueError, match="Length of frequencies list"),
            ),
            (
                2,
                [np.arange(1, 10)],
                pytest.raises(
                    ValueError,
                    match=r"Length of frequencies list",
                ),
            ),
            (
                2,
                (np.arange(1, 10),),
                pytest.raises(
                    ValueError,
                    match=r"must be a 2-element tuple of non-negative integers",
                ),
            ),
            (2, [10, 10], does_not_raise()),
            (2, [(1, 10), (1, 10)], does_not_raise()),
            (2, [np.arange(1, 10), np.arange(1, 10)], does_not_raise()),
            (
                2,
                (np.arange(1, 10), np.arange(1, 10)),
                pytest.raises(
                    ValueError,
                    match=r"must be a 2-element tuple of non-negative integers",
                ),
            ),
            (1, [(0, 1)], does_not_raise()),
            (
                1,
                [(10, 1)],
                pytest.raises(ValueError, match="Tuple frequencies must satisfy"),
            ),
            (
                2,
                [(1, 10), (10, 1)],
                pytest.raises(ValueError, match="Tuple frequencies must satisfy"),
            ),
            (
                1,
                -1,
                pytest.raises(ValueError, match="Integer frequencies must be >= 0"),
            ),
            (
                2,
                [1, -1],
                pytest.raises(ValueError, match="Integer frequencies must be >= 0"),
            ),
            (
                1,
                [(-1, 1)],
                pytest.raises(ValueError, match="Tuple frequencies must satisfy 0"),
            ),
            (
                1,
                [(0, 2.1)],
                pytest.raises(TypeError, match="Tuple frequencies must be integers"),
            ),
            (
                1,
                [np.array([1, 3, 2])],
                pytest.warns(UserWarning, match="Unsorted frequencies provided"),
            ),
            (
                1,
                [np.array([-1, 2, 3])],
                pytest.raises(ValueError, match="frequencies contain negative values"),
            ),
            (
                1,
                [np.array([0.5, 2, 3])],
                pytest.raises(ValueError, match="frequency values are not integers"),
            ),
            (
                2,
                [np.array([1, 2, 3]), np.array([-1, 2, 3])],
                pytest.raises(ValueError, match="frequencies contain negative values"),
            ),
            (
                2,
                [np.array([1, 2, 3]), np.array([0.5, 2, 3])],
                pytest.raises(ValueError, match="frequency values are not integers"),
            ),
            (
                2,
                [np.arange(-1, 3), (1, 3)],
                pytest.raises(
                    ValueError,
                ),
            ),
            (
                2,
                [np.arange(1, 3), -1],
                pytest.raises(ValueError, match="Integer frequencies must be >= 0"),
            ),
            (
                2,
                [np.arange(1, 3), (-1, 2)],
                pytest.raises(ValueError, match="Tuple frequencies must satisfy 0"),
            ),
            (3, [np.arange(1, 3), (1, 3), 10], does_not_raise()),
        ],
    )
    def test_frequencies_setter(self, frequencies, expectation, mode, ndim):
        bas = instantiate_atomic_basis(
            self.cls[mode],
            **extra_kwargs(self.cls[mode], 10),
            ndim=ndim,
            frequency_mask=None,
        )
        with expectation:
            bas.frequencies = frequencies
            assert all(np.issubdtype(f, np.floating) for f in bas.frequencies)
            assert isinstance(bas.frequencies, list)
            assert all(isinstance(f, jax.numpy.ndarray) for f in bas.frequencies)

    @pytest.mark.parametrize(
        "frequency_mask, expectation, output_pairs",
        [
            ("no-intercept", does_not_raise(), np.arange(1, 4).reshape(1, -1)),
            ("all", does_not_raise(), np.arange(1, 4).reshape(1, -1)),
            (None, does_not_raise(), np.arange(1, 4).reshape(1, -1)),
            (np.array([1, 0, 1]), does_not_raise(), np.array([[1, 3]])),
            (np.array([False, False, True]), does_not_raise(), np.array([[3]])),
            (lambda x: x == 1, does_not_raise(), np.array([[1.0]])),
            (np.array([1.0, 0.0, 1.0]), does_not_raise(), np.array([[1, 3]])),
            (
                "a",
                pytest.raises(
                    ValueError, match="cannot be converted to a jax array of"
                ),
                None,
            ),
            (
                np.array(["a", "b"]),
                pytest.raises(
                    ValueError, match="cannot be converted to a jax array of"
                ),
                None,
            ),
            (lambda x: float(x == 1), does_not_raise(), np.array([[1.0]])),
            (
                lambda x: 1.4,
                pytest.raises(ValueError, match="must return a single boolean"),
                None,
            ),
            (
                lambda x: np.array([True, True]),
                pytest.raises(ValueError, match="must return a single boolean"),
                None,
            ),
            (
                lambda x: np.array([1, 0]),
                pytest.raises(ValueError, match="must return a single boolean"),
                None,
            ),
            # force a raise
            (
                lambda x: np.array("a") ** 2,
                pytest.raises(
                    TypeError, match="Error while applying the callable assigned"
                ),
                None,
            ),
            (lambda *x: True, does_not_raise(), np.arange(1, 4).reshape(1, -1)),
            (lambda *x: False, does_not_raise(), np.array([[]])),
            (lambda *x: np.True_, does_not_raise(), np.arange(1, 4).reshape(1, -1)),
            (lambda *x: np.False_, does_not_raise(), np.array([[]])),
        ],
    )
    @pytest.mark.parametrize("mode", ["eval"])
    def test_frequency_mask_setter_1d(
        self, mode, frequency_mask, expectation, output_pairs
    ):
        with expectation:
            bas = instantiate_atomic_basis(
                self.cls[mode],
                **extra_kwargs(self.cls[mode], 6),
                ndim=1,
                frequency_mask=frequency_mask,
            )
            np.testing.assert_array_equal(bas._freq_combinations, output_pairs)

        bas = instantiate_atomic_basis(
            self.cls[mode],
            **extra_kwargs(self.cls[mode], 6),
            ndim=1,
            frequency_mask=None,
        )
        with expectation:
            # check setter directly
            bas.frequency_mask = frequency_mask
            np.testing.assert_array_equal(bas._freq_combinations, output_pairs)

    @pytest.mark.parametrize(
        "frequency_mask, expectation, output_pairs",
        [
            (
                None,
                does_not_raise(),
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                        ],
                        [
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                        ],
                    ]
                ),
            ),
            (
                np.array([[1, 0], [1, 1], [0, 1]]),
                does_not_raise(),
                np.array([[0, 1, 1, 2], [0, 0, 1, 1]]),
            ),
            (
                np.array([[False, True], [False, False], [True, True]]),
                does_not_raise(),
                np.array([[0, 2, 2], [1, 0, 1]]),
            ),
            (
                lambda *x: x[0] < 2 and x[1] == 1,
                does_not_raise(),
                np.array([[0, 1], [1, 1]]),
            ),
            (
                np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                does_not_raise(),
                np.array([[0, 1, 1, 2], [0, 0, 1, 1]]),
            ),
            (
                np.array([1.0, 0.0, 5.0]),
                pytest.raises(
                    ValueError,
                    match="Frequency mask must be an array-like of 0s and 1s",
                ),
                None,
            ),
            (
                "a",
                pytest.raises(
                    ValueError, match="cannot be converted to a jax array of"
                ),
                None,
            ),
            (
                np.array(["a", "b"]),
                pytest.raises(
                    ValueError, match="cannot be converted to a jax array of"
                ),
                None,
            ),
            (
                np.array([1, 0]),
                pytest.raises(
                    ValueError,
                    match="The frequency mask for a 2-dimensional Fourier basis must be",
                ),
                None,
            ),
            (
                lambda *x: all(xi == 1 for xi in x),
                does_not_raise(),
                np.array([[1], [1]]),
            ),
            (
                lambda *x: 1.4,
                pytest.raises(ValueError, match="must return a single boolean"),
                None,
            ),
            (
                lambda *x: np.array([True, True]),
                pytest.raises(ValueError, match="must return a single boolean or 0/1"),
                None,
            ),
            (
                lambda *x: np.array([1, 0]),
                pytest.raises(ValueError, match="must return a single boolean or 0/1"),
                None,
            ),
            (
                lambda *x: [1, 0],
                pytest.raises(ValueError, match="must return a single boolean or 0/1"),
                None,
            ),
            # force a raise
            (
                lambda *x: np.array("a") ** 2,
                pytest.raises(
                    TypeError, match="Error while applying the callable assigned"
                ),
                None,
            ),
            (
                lambda *x: True,
                does_not_raise(),
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                        ],
                        [
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                        ],
                    ]
                ),
            ),
            (lambda *x: False, does_not_raise(), np.array([[], []])),
            (
                lambda *x: np.True_,
                does_not_raise(),
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                        ],
                        [
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                        ],
                    ]
                ),
            ),
            (lambda *x: np.False_, does_not_raise(), np.array([[], []])),
        ],
    )
    @pytest.mark.parametrize("mode", ["eval"])
    def test_frequency_mask_setter_2d(
        self, mode, frequency_mask, expectation, output_pairs
    ):
        with expectation:
            bas = self.cls[mode](
                frequencies=[np.arange(3), np.arange(2)],
                ndim=2,
                frequency_mask=frequency_mask,
            )
            np.testing.assert_array_equal(bas._freq_combinations, output_pairs)

        bas = self.cls[mode](
            frequencies=[np.arange(3), np.arange(2)],
            ndim=2,
            frequency_mask=None,
        )
        with expectation:
            # check setter directly
            bas.frequency_mask = frequency_mask
            np.testing.assert_array_equal(bas._freq_combinations, output_pairs)

    @pytest.mark.parametrize("mode", ["eval"])
    @pytest.mark.parametrize(
        "frequency_mask, frequencies, expected_eval",
        [
            (None, 5, np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)),
            (
                [True, False, False, True, True],
                5,
                np.array([[0.0, 3.0, 4.0]], dtype=np.float32),
            ),
            (lambda x: x < 3, 5, np.array([[0.0, 1.0, 2.0]], dtype=np.float32)),
        ],
    )
    def test_joint_frequency_and_frequency_mask_set_params(
        self, frequency_mask, frequencies, expected_eval, mode
    ):
        bas = self.cls[mode](frequencies=10, ndim=1, frequency_mask=None)
        bas.set_params(frequencies=frequencies, frequency_mask=frequency_mask)
        np.testing.assert_array_equal(bas._freq_combinations, expected_eval)
        bas = self.cls[mode](frequencies=10, ndim=1, frequency_mask=None)
        bas.set_params(frequency_mask=frequency_mask, frequencies=frequencies)
        np.testing.assert_array_equal(bas._freq_combinations, expected_eval)

    @pytest.mark.parametrize("mode", ["eval"])
    @pytest.mark.parametrize(
        "frequency_mask, frequencies, new_frequencies, expected_eval, expectation",
        [
            (
                None,
                5,
                5,
                np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                does_not_raise(),
            ),
            (
                None,
                5,
                6,
                np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                does_not_raise(),
            ),
            (
                [True, False, False, True, True],
                5,
                5,
                np.array([[0.0, 3.0, 4.0]], dtype=np.float32),
                does_not_raise(),
            ),
            (
                [True, False, False, True, True],
                5,
                6,
                np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                pytest.warns(UserWarning, match="Resetting ``frequency_mask`` to "),
            ),
            (
                lambda x: x < 3,
                5,
                6,
                np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                pytest.warns(UserWarning, match="Resetting ``frequency_mask`` to "),
            ),
        ],
    )
    def test_frequency_mask_resetting_behavior(
        self,
        frequency_mask,
        frequencies,
        new_frequencies,
        expected_eval,
        expectation,
        mode,
    ):
        bas = self.cls[mode](
            frequencies=frequencies, ndim=1, frequency_mask=frequency_mask
        )
        with expectation:
            bas.frequencies = new_frequencies
            np.testing.assert_array_equal(bas._freq_combinations, expected_eval)
            if np.any(new_frequencies != frequencies):
                assert (
                    bas.frequency_mask == "no-intercept"
                    if frequency_mask is not None
                    else "all"
                )

    @pytest.mark.parametrize("mode", ["eval"])
    @pytest.mark.parametrize(
        "ndim, expectation",
        [
            (0, pytest.raises(ValueError, match="ndim must be a non-negative integer")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (2.0, does_not_raise()),
            (
                2.1,
                pytest.raises(ValueError, match="ndim must be a non-negative integer"),
            ),
            (np.array(2.0), does_not_raise()),
            (
                np.array(2.1),
                pytest.raises(ValueError, match="ndim must be a non-negative integer"),
            ),
            (
                "a",
                pytest.raises(TypeError, match="Cannot convert ndim 'a' to type int"),
            ),
            ({"a"}, pytest.raises(TypeError, match="Cannot convert ndim {'a'}")),
        ],
    )
    def test_ndim_checks(self, mode, ndim, expectation):
        with expectation:
            bas = self.cls[mode](frequencies=5, ndim=ndim)
            assert bas.ndim == ndim

    @pytest.mark.parametrize("mode", ["eval"])
    @pytest.mark.parametrize(
        "frequency_mask, ndim, expected_output_shape",
        [
            (None, 1, (1, 9)),  # 5 * 2 -1
            (None, 2, (1, 49)),  # 5 * 5 * 2 -1
            # drop tree frequencies
            (jax.numpy.ones(5).at[1:4].set(0), 1, (1, 3)),  # (5 - 3) * 2 - 1
            # drop tree elements, including 0
            (jax.numpy.ones(5).at[:3].set(0), 1, (1, 4)),  # (5 - 3) * 2
            (lambda x: x < 3.1, 1, (1, 7)),  # 4 * 2 - 1
            # # The lambda func below returns true for 11 values:
            # - (0, 0), (0, 1), (0, 2), (0, 3)
            # - (1, 0), (1, 1), (1, 2)
            # - (2, 0), (2, 1), (2, 2)
            # - (3, 0)
            (lambda x, y: np.sqrt(x**2 + y**2) < 3.1, 2, (1, 21)),  # 11 * 2 - 1
            # set 3 entries to 0 from the 5 x 5 mask
            (
                jax.numpy.ones((5, 5))
                .at[jax.numpy.array([1, 2, 3]), jax.numpy.array([2, 2, 4])]
                .set(0),
                2,
                (10, 43),  # (5 * 5 - 3) * 2 - 1
            ),
            # set 3 entries to 0 from the 5 x 5 mask, including (0, 0)
            (
                jax.numpy.ones((5, 5))
                .at[jax.numpy.array([0, 2, 3]), jax.numpy.array([0, 2, 4])]
                .set(0),
                2,
                (1, 44),  # (5 * 5 - 3) * 2
            ),
            (jax.numpy.zeros((5,)), 1, (1, 0)),
            (jax.numpy.zeros((5, 5)), 2, (1, 0)),
        ],
    )
    def test_n_basis_function_compute(
        self, frequency_mask, ndim, expected_output_shape, mode
    ):
        bas = self.cls[mode](frequencies=5, ndim=ndim, frequency_mask=frequency_mask)
        out = bas.compute_features(*np.ones((ndim, expected_output_shape[0])))
        assert out.shape == expected_output_shape
        assert bas.n_basis_funcs == expected_output_shape[-1]

    @pytest.mark.parametrize(
        "bounds, ndim, expectation",
        [
            (None, 1, does_not_raise()),
            (
                (None, np.pi),
                1,
                pytest.raises(TypeError, match="Could not convert `bounds` to float"),
            ),
            (
                (np.pi, None),
                1,
                pytest.raises(TypeError, match="Could not convert `bounds` to float"),
            ),
            ((np.pi / 2, np.pi), 1, does_not_raise()),
            # generic numpy scalar
            ((np.int64(1), np.int64(2)), 1, does_not_raise()),
            # 0-dim array
            ((np.array(1), np.array(2)), 1, does_not_raise()),
            ((jax.numpy.array(1), jax.numpy.array(2)), 1, does_not_raise()),
            (
                (np.pi, np.pi),
                1,
                pytest.raises(
                    ValueError, match=" Lower bound is greater or equal than the"
                ),
            ),
            ([(np.pi / 2, np.pi)], 1, does_not_raise()),
            (None, 2, does_not_raise()),
            (
                [None, None],
                2,
                pytest.raises(TypeError, match="Could not convert `bounds` to float"),
            ),
            (
                [(np.pi / 2, np.pi)] * 2,
                1,
                pytest.raises(
                    TypeError, match="When provided, the bounds should be one"
                ),
            ),
            ([(np.pi / 2, np.pi)] * 2, 2, does_not_raise()),
            (
                [(np.pi / 2, np.pi), (np.pi, np.pi)],
                2,
                pytest.raises(
                    ValueError, match=" Lower bound is greater or equal than the"
                ),
            ),
            (
                [(np.pi / 2, np.pi), ("a", np.pi)],
                2,
                pytest.raises(
                    TypeError,
                    match="the bounds should be one or multiple tuples containing pair of floats",
                ),
            ),
        ],
    )
    def test_bounds_setter(self, bounds, ndim, expectation):
        with expectation:
            self.cls["eval"](frequencies=5, bounds=bounds, ndim=ndim)

    def test_masked_frequencies_property(self):
        bas = self.cls["eval"](frequencies=np.arange(5), ndim=1, frequency_mask=None)
        np.testing.assert_array_equal(
            bas.masked_frequencies, np.arange(5).reshape(1, 5)
        )
        bas.frequency_mask = np.array([0, 1, 0, 1, 0])
        np.testing.assert_array_equal(bas.masked_frequencies, np.array([[1, 3]]))
        with pytest.raises(AttributeError, match="has no setter|can't set attribute"):
            bas.masked_frequencies = np.arange(5).reshape(1, 5)

    @pytest.mark.parametrize(
        "frequency_mask, freqs_input, expected_output",
        [
            ("all", np.arange(0, 4), np.array([0, 1, 2, 3])),
            ("no-intercept", np.arange(0, 4), np.array([1, 2, 3])),
            ("all", np.arange(1, 4), np.array([1, 2, 3])),
            ("no-intercept", np.arange(1, 4), np.array([1, 2, 3])),
        ],
    )
    def test_string_init(self, frequency_mask, freqs_input, expected_output):
        bas = self.cls["eval"](
            frequencies=freqs_input, ndim=1, frequency_mask=frequency_mask
        )
        np.array_equal(expected_output, bas.masked_frequencies)

    @pytest.mark.parametrize(
        "frequency_mask, freqs_input, expected_output",
        [
            ("all", np.arange(0, 2), np.array([[0, 0, 1, 1], [0, 1, 0, 1]])),
            ("no-intercept", np.arange(0, 2), np.array([[0, 1, 1], [1, 0, 1]])),
            ("all", np.arange(1, 2), np.array([[1], [1]])),
            ("no-intercept", np.arange(1, 2), np.array([[1], [1]])),
        ],
    )
    def test_string_init_2d(self, frequency_mask, freqs_input, expected_output):
        bas = self.cls["eval"](
            frequencies=freqs_input, ndim=2, frequency_mask=frequency_mask
        )
        np.array_equal(expected_output, bas.masked_frequencies)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_evaluate_on_grid_ndim(self, ndim):
        bas = self.cls["eval"](frequencies=3, ndim=ndim, frequency_mask=None)
        out = bas.evaluate_on_grid(*[10] * ndim)
        assert len(out) == 1 + ndim
        for i in range(ndim):
            assert out[i].shape == ((10,) * ndim)
        assert out[ndim].shape == (*(10,) * ndim, bas.n_basis_funcs)


class TestAdditiveBasis(CombinedBasis):
    cls = {"eval": AdditiveBasis, "conv": AdditiveBasis}

    @pytest.mark.parametrize(
        "bas_cls",
        list_all_basis_classes("Eval") + list_all_basis_classes("Conv") + [CustomBasis],
    )
    def test_mul_by_int_basis_with_label(self, bas_cls, basis_class_specific_params):
        basis_obj = self.instantiate_basis(
            5, bas_cls, basis_class_specific_params, window_size=10
        )
        _ = basis_obj * 2
        basis_obj.label = "x"
        with pytest.raises(ValueError, match="Cannot multiply by an integer"):
            _ = basis_obj * 2

    @pytest.mark.parametrize(
        "basis_a",
        list_all_basis_classes("Eval") + list_all_basis_classes("Conv") + [CustomBasis],
    )
    def test_add_label_using_class_name(self, basis_a, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        add = basis_a_obj + basis_a_obj + basis_a_obj
        with pytest.raises(ValueError, match="Cannot set basis label"):
            add.label = "MultiplicativeBasis"
        add.label = "AdditiveBasis"

    @pytest.mark.parametrize("bas", list_all_basis_classes())
    def test_inherit_setting(self, bas, basis_class_specific_params):
        basis_obj = self.instantiate_basis(
            5, bas, basis_class_specific_params, window_size=10
        )
        comp_bases = basis_obj + basis_obj.__sklearn_clone__().set_params(label="z")
        basis_update = basis.BSplineEval(5) + basis.BSplineEval(5, label="z")
        basis_update.set_params(z=comp_bases)
        assert basis_update.basis2.label != "z"

    def test_redundant_label_in_nested_basis(self):
        bas = (
            basis.BSplineEval(4)
            + basis.BSplineEval(5)
            + basis.BSplineEval(6)
            + basis.BSplineEval(7)
        )
        with pytest.raises(
            ValueError,
            match="All user-provided labels of basis elements must be distinct",
        ):
            bas.set_params(
                **{
                    "(BSplineEval + BSplineEval_1)": AdditiveBasis(
                        basis.BSplineEval(9), basis.BSplineEval(10), label="ciao"
                    ),
                    "((BSplineEval + BSplineEval_1) + BSplineEval_2)": AdditiveBasis(
                        basis.BSplineEval(9), basis.BSplineEval(10), label="ciao"
                    ),
                }
            )

    @pytest.mark.parametrize("basis_a", list_all_basis_classes("Eval"))
    def test_set_params_basis(self, basis_a, basis_class_specific_params):
        basis_b = basis_a.__name__.replace("Eval", "Conv")
        if not hasattr(basis, basis_b):
            return
        else:
            basis_b = getattr(basis, basis_b)
        cls_b_name = basis_b.__name__
        cls_a_name = basis_a.__name__
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        # check update label tag
        add_a_twice = basis_a_obj + basis_a_obj
        assert add_a_twice.basis2.label == f"{cls_a_name}_1"

        # set different classs and check refreshing labels
        add_a_twice.set_params(**{cls_a_name: basis_b_obj})
        assert add_a_twice.basis2.label == cls_a_name
        assert add_a_twice.basis1.label == cls_b_name

        # set basis label with tag
        add_a_twice.set_params(**{cls_b_name: basis_a_obj})
        add_a_twice.basis1.label = f"{cls_a_name}_1"
        assert add_a_twice.basis1.label == cls_a_name
        assert add_a_twice.basis2.label == f"{cls_a_name}_1"

        # assign both the same basis
        add_a_twice.set_params(
            **{f"{cls_a_name}_1": basis_b_obj, cls_a_name: basis_b_obj}
        )
        assert add_a_twice.basis1.label == f"{cls_b_name}"
        assert add_a_twice.basis2.label == f"{cls_b_name}_1"
        # revert order of basis
        add_a_twice.set_params(
            **{cls_b_name: basis_a_obj, f"{cls_b_name}_1": basis_a_obj}
        )
        assert add_a_twice.basis1.label == f"{cls_a_name}"
        assert add_a_twice.basis2.label == f"{cls_a_name}_1"

        # add a label and check that it is passed down correctly
        # and the other label is updated
        add_a_twice.basis1.label = "x"
        add_a_twice.set_params(x=basis_b_obj)
        assert add_a_twice.basis1.label == "x"
        assert add_a_twice.basis2.label == f"{cls_a_name}"

        # add a label and set a basis with a modified label
        add_a_twice.basis1.label = "x"
        new_basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        add_a_twice.set_params(
            **{"x": basis_b_obj.set_params(label="z"), cls_a_name: new_basis_b_obj}
        )
        assert add_a_twice.basis1.label == "z"
        assert add_a_twice.basis2.label == f"{cls_b_name}"

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_input_shape_product_init(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a_obj + basis_b_obj
        assert add._input_shape_product is None
        basis_a_obj.set_input_shape(())
        add = basis_a_obj + basis_b_obj
        assert add._input_shape_product is None
        basis_b_obj.set_input_shape(())
        add = basis_a_obj + basis_b_obj
        assert add._input_shape_product == (1, 1)
        basis_b_obj.set_input_shape((1, 2, 3))
        add = basis_a_obj + basis_b_obj
        assert add._input_shape_product == (1, 6)
        assert (add + add)._input_shape_product == (1, 6, 1, 6)

    @pytest.mark.parametrize(
        "bas",
        list_all_basis_classes("Eval") + list_all_basis_classes("Conv") + [CustomBasis],
    )
    def test_rmul_lmul(self, bas, basis_class_specific_params):
        basis_obj = self.instantiate_basis(
            5, bas, basis_class_specific_params, window_size=10
        )
        out = 10 * basis_obj
        assert isinstance(out, AdditiveBasis)
        assert sum((1 for _ in out._iterate_over_components())) == 10
        out = basis_obj * 10
        assert isinstance(out, AdditiveBasis)
        assert sum((1 for _ in out._iterate_over_components())) == 10

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_provide_label_at_init(self, basis_a, basis_b, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        basis_a_obj.label = "a"
        basis_b_obj.label = "b"
        add = AdditiveBasis(basis_a_obj, basis_b_obj, label="newlabel")
        assert add.label == "newlabel"
        add.label = None
        assert add.label == "(a + b)"

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    def test_len(self, basis_a, basis_b, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a_obj + basis_b_obj
        expected_len = (
            2
            + isinstance(basis_a_obj, AdditiveBasis)
            + isinstance(basis_b_obj, AdditiveBasis)
        )
        assert len(add) == expected_len

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    def test_iter(self, basis_a, basis_b, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a_obj + basis_b_obj
        # manually unpack basis
        bas_list = []
        if isinstance(add.basis1, AdditiveBasis):
            bas_list += [add.basis1.basis1, add.basis1.basis2]
        else:
            bas_list += [add.basis1]
        if isinstance(add.basis2, AdditiveBasis):
            bas_list += [add.basis2.basis1, add.basis2.basis2]
        else:
            bas_list += [add.basis2]

        for b, b1 in zip(add, bas_list):
            assert not isinstance(b, AdditiveBasis)
            assert id(b) == id(b1)

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    def test_iterate_over_component(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a_obj + basis_b_obj
        out = tuple(add._iterate_over_components())
        assert len(out) == add._n_input_dimensionality

        def get_ids(bas):

            if hasattr(bas, "basis1"):
                ids = get_ids(bas.basis1)
                ids += get_ids(bas.basis2)
            else:
                ids = [id(bas)]
            return ids

        id_list = get_ids(add)

        assert tuple(id(o) for o in out) == tuple(id_list)

    @pytest.mark.parametrize("samples", [[[0], []], [[], [0]], [[0, 0], [0, 0]]])
    @pytest.mark.parametrize(
        "base_cls", [basis.BSplineEval, basis.BSplineConv, CustomBasis]
    )
    def test_non_empty_samples(self, base_cls, samples, basis_class_specific_params):
        kwargs = {"window_size": 2, "n_basis_funcs": 5}
        kwargs = inspect_utils.trim_kwargs(
            base_cls, kwargs, basis_class_specific_params
        )
        if base_cls == CustomBasis:
            basis_obj = custom_basis(**kwargs) + custom_basis(**kwargs)
        else:
            basis_obj = base_cls(**kwargs) + base_cls(**kwargs)
        if any(tuple(len(s) == 0 for s in samples)):
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                basis_obj.compute_features(*samples)
        else:
            basis_obj.compute_features(*samples)

    @pytest.mark.parametrize(
        "eval_input",
        [
            [0, 0],
            [[0], [0]],
            [(0,), (0,)],
            [np.array([0]), [0]],
            [jax.numpy.array([0]), [0]],
        ],
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = basis.MSplineEval(5) + basis.MSplineEval(5)
        basis_obj.compute_features(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [6])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("vmin, vmax", [(-1, 1)])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_sklearn_clone(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        vmin,
        vmax,
        inp_num,
        basis_class_specific_params,
    ):
        """Recursively check cloning."""
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_a_obj = basis_a_obj.set_input_shape(
            *([inp_num] * basis_a_obj._n_input_dimensionality)
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=15
        )
        basis_b_obj = basis_b_obj.set_input_shape(
            *([inp_num] * basis_b_obj._n_input_dimensionality)
        )
        add = basis_a_obj + basis_b_obj

        add2 = add.__sklearn_clone__()
        compare_basis(add, add2)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("window_size", [10])
    def test_compute_features_returns_expected_number_of_basis(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the evaluation of the `AdditiveBasis` results in a number of basis
        that is the sum of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )

        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[1] != basis_a_obj.n_basis_funcs + basis_b_obj.n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {n_basis_a + n_basis_b}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("window_size", [10])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the output sample size from `AdditiveBasis` compute_features function matches input sample size.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the "
                f"output features basis."
                f"The window size is {sample_size}",
                f"The second dimension of the output features basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("window_size", [10])
    def test_number_of_required_inputs_compute_features(
        self,
        n_input,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the number of required inputs for the `compute_features` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match=r"This basis requires \d+ input\(s\)."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_meshgrid_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the resulting meshgrid size matches the sample size input.
        """
        if basis_a == CustomBasis or basis_b == CustomBasis:
            pytest.skip(
                f"Skipping test_evaluate_on_grid_meshgrid_size for {basis_a.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj + basis_b_obj
        res = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )
        for grid in res[:-1]:
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_basis_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number sample size output by evaluate_on_grid matches the sample size of the input.
        """
        if basis_a == CustomBasis or basis_b == CustomBasis:
            pytest.skip(
                f"Skipping test_evaluate_on_grid_basis_size for {basis_a.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )[-1]
        assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2, 5, 6, 11, 30])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_input_number(
        self,
        n_input,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number of inputs provided to `evaluate_on_grid` matches
        the sum of the number of input samples required from each of the basis objects.
        """
        if basis_a == CustomBasis or basis_b == CustomBasis:
            pytest.skip(
                f"Skipping test_evaluate_on_grid_input_number for {basis_a.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj + basis_b_obj
        inputs = [20] * n_input
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match=r"This basis requires \d+ input\(s\)."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    def test_pynapple_support_compute_features(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_class_specific_params,
    ):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        basis_add = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        ) + self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        # compute_features the basis over pynapple Tsd objects
        out = basis_add.compute_features(*([inp] * basis_add._n_input_dimensionality))
        # check type
        assert isinstance(out, nap.TsdFrame)
        # check value
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize(" window_size", [8])
    def test_call_input_num(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        num_input,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                TypeError, match=r"This basis requires \d+ input\(s\)"
            )
        with expectation:
            basis_obj.evaluate(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2])
    @pytest.mark.parametrize(" window_size", [8])
    def test_set_input_shape_input_num(
        self,
        n_basis_a,
        basis_a,
        num_input,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = basis_a_obj.__sklearn_clone__()
        basis_obj = basis_a_obj + basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(ValueError, match="set_input_shape expects")
        with expectation:
            basis_obj.set_input_shape(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_input_shape(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        inp,
        window_size,
        expectation,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        with expectation:
            basis_obj.evaluate(*([inp] * basis_obj._n_input_dimensionality))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_axis(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        time_axis_shape,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        inp = [np.linspace(0, 1, time_axis_shape)] * basis_obj._n_input_dimensionality
        assert basis_obj.evaluate(*inp).shape[0] == time_axis_shape

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_nan(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        if basis_a in (basis.OrthExponentialBasis, basis.HistoryConv) or basis_b in (
            basis.OrthExponentialBasis,
            basis.HistoryConv,
        ):
            pytest.skip(f"Skipping test_call_nan for {basis_a.__name__}")
        if basis_a == IdentityEval:
            n_basis_a = 1
        else:
            n_basis_b = 5
        if basis_b == IdentityEval:
            n_basis_b = 1
        else:
            n_basis_b = 5
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        inp = [np.linspace(0, 1, 10)] * basis_obj._n_input_dimensionality
        for x in inp:
            x[3] = np.nan
        assert all(np.isnan(basis_obj.evaluate(*inp)[3]))

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_equivalent_in_conv(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        if (
            basis_a == HistoryConv
            or basis_b == HistoryConv
            or basis_a == CustomBasis
            or basis_b == CustomBasis
        ):
            # evaluate returns identity
            pytest.skip(
                f"Skipping test_call_nan for {basis_a.__name__} and {basis_b.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=9
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=9
        )
        bas_eva = basis_a_obj + basis_b_obj

        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=8
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=8
        )
        bas_con = basis_a_obj + basis_b_obj

        x = [np.linspace(0, 1, 10)] * bas_con._n_input_dimensionality
        assert np.all(bas_con.evaluate(*x) == bas_eva.evaluate(*x))

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_pynapple_support(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        x = np.linspace(0, 1, 10)
        x_nap = [nap.Tsd(t=np.arange(10), d=x)] * bas._n_input_dimensionality
        x = [x] * bas._n_input_dimensionality
        y = bas.evaluate(*x)
        y_nap = bas.evaluate(*x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap[0].t)

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [6, 7])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_basis_number(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality

        assert (
            bas.evaluate(*x).shape[1]
            == basis_a_obj.n_basis_funcs + basis_b_obj.n_basis_funcs
        )

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_non_empty(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        with pytest.raises(ValueError, match="All sample provided must"):
            bas.compute_features(*([np.array([])] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
            (0.1, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_range(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        mn,
        mx,
        expectation,
        window_size,
        basis_class_specific_params,
    ):
        if basis_a == CustomBasis or basis_b == CustomBasis:
            pytest.skip(
                f"Skipping test_call_sample_range for {basis_a.__name__} and {basis_b.__name__}"
            )
        if expectation == "check":
            if (
                basis_a == basis.OrthExponentialBasis
                or basis_b == basis.OrthExponentialBasis
            ):
                expectation = pytest.raises(
                    ValueError, match="OrthExponentialBasis requires positive samples"
                )
            else:
                expectation = does_not_raise()
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        with expectation:
            bas.evaluate(*([np.linspace(mn, mx, 10)] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_fit_kernel(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj + basis_b_obj
        bas.setup_basis(*([np.ones(10)] * bas._n_input_dimensionality))

        def check_kernel(basis_obj):
            has_kern = []
            if hasattr(basis_obj, "basis1"):
                has_kern += check_kernel(basis_obj.basis1)
                has_kern += check_kernel(basis_obj.basis2)
            else:
                has_kern += [
                    (
                        basis_obj.kernel_ is not None
                        if basis_obj.__class__.__name__.endswith("Conv")
                        else True
                    )
                ]
            return has_kern

        assert all(check_kernel(bas))

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_transform_fails(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        if basis_a == CustomBasis or basis_b == CustomBasis:
            pytest.skip(
                f"Skipping test_transform_fails for {basis_a.__name__} and {basis_b.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj + basis_b_obj
        if "Eval" in basis_a.__name__ and "Eval" in basis_b.__name__:
            context = does_not_raise()
        else:
            context = pytest.raises(
                RuntimeError,
                match="You must call `setup_basis` before `_compute_features`",
            )
        with context:
            x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
            bas._compute_features(*x)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_output_features(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(11, window_size=10)
        bas_add = bas1 + bas2
        assert bas_add.n_output_features is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_output_features == (n_basis_input1 * 10 + n_basis_input2 * 11)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_basis_input(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas_add = bas1 + bas2
        assert bas_add._input_shape_product is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add._input_shape_product == (n_basis_input1, n_basis_input2)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (3, does_not_raise()),
            (0, pytest.raises(ValueError, match="Empty array provided")),
            (1, does_not_raise()),
            (4, does_not_raise()),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas = bas1 + bas2
        x = np.random.randn(20, 2), np.random.randn(20, 3)
        bas.compute_features(*x)
        with expectation:
            x = np.random.randn(30, 2), np.random.randn(30, n_input)
            bas.compute_features(*x)
            assert all(
                xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
                for ishape, xi in zip(bas.input_shape, x)
            )

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    @pytest.mark.parametrize("shape_a", [1, (), np.ones(3)])
    @pytest.mark.parametrize("shape_b", [1, (), np.ones(3)])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_1d_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, *add_shape_a)), np.ones((10, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        add.set_input_shape(shape_a, shape_b)
        add.compute_features(*x)
        assert all(
            xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
            for ishape, xi in zip(add.input_shape, x)
        )

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    @pytest.mark.parametrize("shape_a", [2, (2,), np.ones((3, 2))])
    @pytest.mark.parametrize("shape_b", [3, (3,), np.ones((3, 3))])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_2d_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, 2, *add_shape_a)), np.ones((10, 3, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        add.set_input_shape(shape_a, shape_b)
        add.compute_features(*x)
        assert all(
            xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
            for ishape, xi in zip(add.input_shape, x)
        )

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    @pytest.mark.parametrize("shape_a", [(2, 2), np.ones((3, 2, 2))])
    @pytest.mark.parametrize("shape_b", [(3, 1), np.ones((3, 3, 1))])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_nd_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, 2, 2, *add_shape_a)), np.ones((10, 3, 1, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        add.set_input_shape(shape_a, shape_b)
        add.compute_features(*x)
        assert all(
            xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
            for ishape, xi in zip(add.input_shape, x)
        )

    @pytest.mark.parametrize(
        "inp_shape, expectation",
        [
            (((1, 1), (1, 1)), does_not_raise()),
            (
                ((1, 1.0), (1, 1)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
            (
                ((1, 1), (1, 1.0)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_set_input_value_types(
        self, inp_shape, expectation, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b
        with expectation:
            add.set_input_shape(*inp_shape)

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_deep_copy_basis(self, basis_a, basis_b, basis_class_specific_params):

        if basis_a == HistoryConv:
            n_basis_a = 10
        elif basis_a == IdentityEval:
            n_basis_a = 1
        else:
            n_basis_a = 5
        if basis_b == HistoryConv:
            n_basis_b = 10
        elif basis_b == IdentityEval:
            n_basis_b = 1
        else:
            n_basis_b = 5
        basis_a = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b
        # test pointing to different objects
        assert id(add.basis1) != id(basis_a)
        assert id(add.basis1) != id(basis_b)
        assert id(add.basis2) != id(basis_a)
        assert id(add.basis2) != id(basis_b)

        if isinstance(basis_a, (HistoryConv, IdentityEval)) or isinstance(
            basis_b, (HistoryConv, IdentityEval)
        ):
            pytest.skip(
                f"Skipping test_call_sample_range for {basis_a.__class__.__name__} and {basis_b.__class__.__name__}"
            )
        # test attributes are not related
        set_basis_attr(basis_a, 10)
        assert get_basis_attr(add.basis1) != 10
        set_basis_attr(add.basis1, 6)
        assert basis_a.n_basis_funcs != 6

        set_basis_attr(basis_b, 10)
        assert get_basis_attr(add.basis2) != 10
        set_basis_attr(add.basis2, 6)
        assert basis_b.n_basis_funcs != 6

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_compute_n_basis_runtime(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        if basis_a == HistoryConv:
            n_basis_a = 10
        elif basis_a == IdentityEval:
            n_basis_a = 1
        else:
            n_basis_a = 5
        if basis_b == HistoryConv:
            n_basis_b = 10
        elif basis_b == IdentityEval:
            n_basis_b = 1
        else:
            n_basis_b = 5
        basis_a = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        if not isinstance(add.basis1, (HistoryConv, IdentityEval, CustomBasis)):
            set_basis_attr(add.basis1, 10)
            assert add.n_basis_funcs == 10 + n_basis_b
        if not isinstance(add.basis2, (HistoryConv, IdentityEval, CustomBasis)):
            set_basis_attr(add.basis2, 10)
            assert add.n_basis_funcs == 10 + add.basis1.n_basis_funcs

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_basis_classes())
    )
    def test_runtime_n_basis_out_compute(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_a.set_input_shape(
            *([1] * basis_a._n_input_dimensionality)
        ).to_transformer()
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        basis_b.set_input_shape(
            *([1] * basis_b._n_input_dimensionality)
        ).to_transformer()
        add = basis_a + basis_b
        inps_a = [2] * basis_a._n_input_dimensionality
        set_input_shape(add.basis1, *inps_a)
        new_out_num = inps_a[0] * add.basis1.n_basis_funcs
        assert add.n_output_features == new_out_num + add.basis2.n_basis_funcs
        inps_b = [3] * basis_b._n_input_dimensionality
        set_input_shape(add.basis2, *inps_b)
        new_out_num_b = inps_b[0] * add.basis2.n_basis_funcs
        assert add.n_output_features == new_out_num + new_out_num_b

    @pytest.mark.parametrize(
        "basis_a", [basis.BSplineEval, AdditiveBasis, MultiplicativeBasis]
    )
    @pytest.mark.parametrize("basis_b", [basis.MSplineEval])
    @pytest.mark.parametrize(
        "expected_out",
        [
            {
                basis.BSplineEval: "'(BSplineEval + MSplineEval)': AdditiveBasis(\n    basis1=BSplineEval(n_basis_funcs=5, order=4),\n    basis2=MSplineEval(n_basis_funcs=6, order=4),\n)",
                AdditiveBasis: "'((MSplineEval + RaisedCosineLinearConv) + MSplineEval_1)': AdditiveBasis(\n    basis1='(MSplineEval + RaisedCosineLinearConv)': AdditiveBasis(\n        basis1=MSplineEval(n_basis_funcs=5, order=4),\n        basis2=RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0),\n    ),\n    basis2='MSplineEval_1': MSplineEval(n_basis_funcs=6, order=4),\n)",
                MultiplicativeBasis: "'((MSplineEval * RaisedCosineLinearConv) + MSplineEval_1)': AdditiveBasis(\n    basis1='(MSplineEval * RaisedCosineLinearConv)': MultiplicativeBasis(\n        basis1=MSplineEval(n_basis_funcs=5, order=4),\n        basis2=RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0),\n    ),\n    basis2='MSplineEval_1': MSplineEval(n_basis_funcs=6, order=4),\n)",
            }
        ],
    )
    def test_repr_out(
        self, basis_a, basis_b, basis_class_specific_params, expected_out
    ):
        with patch("os.get_terminal_size", return_value=SizeTerminal(80, 24)):
            basis_a_obj = self.instantiate_basis(
                5, basis_a, basis_class_specific_params, window_size=10
            )
            basis_b_obj = self.instantiate_basis(
                6, basis_b, basis_class_specific_params, window_size=10
            )
            basis_obj = basis_a_obj + basis_b_obj
            assert repr(basis_obj) == expected_out[basis_a]

    @pytest.mark.parametrize("label", [None, "", "default-behavior", "CoolFeature"])
    def test_repr_label(self, label, basis_class_specific_params):
        with patch("os.get_terminal_size", return_value=SizeTerminal(80, 24)):
            if label == "default-behavior":
                bas = basis.RaisedCosineLinearEval(n_basis_funcs=5)
            else:
                bas = basis.RaisedCosineLinearEval(n_basis_funcs=5, label=label)

            if label in [None, "default-behavior"]:
                expected_a = "RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)"
                exp_name = "RaisedCosineLinearEval"
            else:
                expected_a = (
                    f"'{label}': RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)"
                )
                exp_name = label
            bas = bas + self.instantiate_basis(
                6, basis.MSplineEval, basis_class_specific_params
            )
            expected = f"'({exp_name} + MSplineEval)': AdditiveBasis(\n    basis1={expected_a},\n    basis2=MSplineEval(n_basis_funcs=6, order=4),\n)"
            out = repr(bas)
            assert out == expected

    @pytest.mark.parametrize(
        "real_cls",
        list_all_real_basis_classes("Eval")
        + list_all_real_basis_classes("Conv")
        + [CustomBasis],
    )
    @pytest.mark.parametrize("complex_cls", [basis.FourierEval])
    def test_multiply_complex(self, real_cls, complex_cls, basis_class_specific_params):
        basis_real = self.instantiate_basis(
            5, real_cls, basis_class_specific_params, window_size=10
        )
        basis_complex = self.instantiate_basis(
            5, complex_cls, basis_class_specific_params, window_size=10
        )
        new_complex = basis_real + basis_complex
        assert new_complex.is_complex

        with pytest.raises(
            ValueError, match="Invalid multiplication between two complex bases"
        ):
            new_complex * basis_complex

        with pytest.raises(
            ValueError, match="Invalid multiplication between two complex bases"
        ):
            basis_complex * basis_complex

        with pytest.raises(
            ValueError, match="Invalid multiplication between two complex bases"
        ):
            basis_complex * basis_real * basis_complex


class TestMultiplicativeBasis(CombinedBasis):
    cls = {"eval": MultiplicativeBasis, "conv": MultiplicativeBasis}

    @pytest.mark.parametrize(
        "bas_cls",
        list_all_real_basis_classes("Eval")
        + list_all_real_basis_classes("Conv")
        + [CustomBasis],
    )
    def test_pow_by_int_basis_with_label(self, bas_cls, basis_class_specific_params):
        basis_obj = self.instantiate_basis(
            5, bas_cls, basis_class_specific_params, window_size=10
        )
        _ = basis_obj**2
        basis_obj.label = "x"
        with pytest.raises(ValueError, match="Cannot calculate the power of a basis"):
            _ = basis_obj**2

    @pytest.mark.parametrize(
        "basis_a",
        list_all_real_basis_classes("Eval")
        + list_all_real_basis_classes("Conv")
        + [CustomBasis],
    )
    def test_add_label_using_class_name(self, basis_a, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        mul = basis_a_obj * basis_a_obj * basis_a_obj
        with pytest.raises(ValueError, match="Cannot set basis label"):
            mul.label = "AdditiveBasis"
        mul.label = "MultiplicativeBasis"

    def test_redundant_label_in_nested_basis(self):
        bas = (
            basis.BSplineEval(4) * basis.BSplineEval(5)
            + basis.BSplineEval(6)
            + basis.BSplineEval(7)
        )
        with pytest.raises(
            ValueError,
            match="All user-provided labels of basis elements must be distinct",
        ):
            bas.set_params(
                **{
                    "(BSplineEval * BSplineEval_1)": AdditiveBasis(
                        basis.BSplineEval(9), basis.BSplineEval(10), label="ciao"
                    ),
                    "((BSplineEval * BSplineEval_1) + BSplineEval_2)": AdditiveBasis(
                        basis.BSplineEval(9), basis.BSplineEval(10), label="ciao"
                    ),
                }
            )

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_real_basis_classes("Eval")
            + list_all_real_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_input_shape_product_init(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a_obj * basis_b_obj
        assert mul._input_shape_product is None
        basis_a_obj.set_input_shape(())
        mul = basis_a_obj * basis_b_obj
        assert mul._input_shape_product is None
        basis_b_obj.set_input_shape(())
        mul = basis_a_obj * basis_b_obj
        assert mul._input_shape_product == (1, 1)
        basis_b_obj.set_input_shape((1, 2, 3))
        with pytest.warns(
            UserWarning, match="Multiple different input shapes detected"
        ):
            mul = basis_a_obj * basis_b_obj
        # incompatible shape resets input shape
        assert mul._input_shape_product is None
        basis_a_obj.set_input_shape((1, 2, 3))
        mul = basis_a_obj * basis_b_obj
        assert mul._input_shape_product == (6, 6)
        assert (mul * mul)._input_shape_product == (6, 6, 6, 6)

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    def test_len(self, basis_a, basis_b, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a_obj * basis_b_obj
        assert len(mul) == 1

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    def test_iter(self, basis_a, basis_b, basis_class_specific_params):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a_obj * basis_b_obj
        for b in mul:
            assert id(b) == id(mul)

    @pytest.mark.parametrize(
        "samples", [[[0], []], [[], [0]], [[0], [0]], [[0, 0], [0, 0]]]
    )
    @pytest.mark.parametrize(" ws", [3])
    def test_non_empty_samples(self, samples, ws):
        basis_obj = basis.MSplineEval(5) * basis.RaisedCosineLinearEval(5)
        if any(tuple(len(s) == 0 for s in samples)):
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                basis_obj.compute_features(*samples)
        else:
            basis_obj.compute_features(*samples)

    @pytest.mark.parametrize(
        "eval_input",
        [
            [0, 0],
            [[0], [0]],
            [(0,), (0,)],
            [np.array([0]), [0]],
            [jax.numpy.array([0]), [0]],
        ],
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = basis.MSplineEval(5) * basis.MSplineEval(5)
        basis_obj.compute_features(*eval_input)

    @pytest.mark.parametrize(
        "basis_a", [basis.BSplineEval, AdditiveBasis, MultiplicativeBasis]
    )
    @pytest.mark.parametrize("basis_b", [basis.MSplineEval])
    @pytest.mark.parametrize(
        "expected_out",
        [
            {
                basis.BSplineEval: "'(BSplineEval * MSplineEval)': MultiplicativeBasis(\n    basis1=BSplineEval(n_basis_funcs=5, order=4),\n    basis2=MSplineEval(n_basis_funcs=6, order=4),\n)",
                AdditiveBasis: "'((MSplineEval + RaisedCosineLinearConv) * MSplineEval_1)': MultiplicativeBasis(\n    basis1='(MSplineEval + RaisedCosineLinearConv)': AdditiveBasis(\n        basis1=MSplineEval(n_basis_funcs=5, order=4),\n        basis2=RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0),\n    ),\n    basis2='MSplineEval_1': MSplineEval(n_basis_funcs=6, order=4),\n)",
                MultiplicativeBasis: "'((MSplineEval * RaisedCosineLinearConv) * MSplineEval_1)': MultiplicativeBasis(\n    basis1='(MSplineEval * RaisedCosineLinearConv)': MultiplicativeBasis(\n        basis1=MSplineEval(n_basis_funcs=5, order=4),\n        basis2=RaisedCosineLinearConv(n_basis_funcs=5, window_size=10, width=2.0),\n    ),\n    basis2='MSplineEval_1': MSplineEval(n_basis_funcs=6, order=4),\n)",
            }
        ],
    )
    def test_repr_out(
        self, basis_a, basis_b, basis_class_specific_params, expected_out
    ):
        with patch("os.get_terminal_size", return_value=SizeTerminal(80, 24)):
            basis_a_obj = self.instantiate_basis(
                5, basis_a, basis_class_specific_params, window_size=10
            )
            basis_b_obj = self.instantiate_basis(
                6, basis_b, basis_class_specific_params, window_size=10
            )
            basis_obj = basis_a_obj * basis_b_obj
            assert repr(basis_obj) == expected_out[basis_a]

    @pytest.mark.parametrize("label", [None, "", "default-behavior", "CoolFeature"])
    def test_repr_label(self, label, basis_class_specific_params):
        with patch("os.get_terminal_size", return_value=SizeTerminal(80, 24)):
            if label == "default-behavior":
                bas = basis.RaisedCosineLinearEval(n_basis_funcs=5)
            else:
                bas = basis.RaisedCosineLinearEval(n_basis_funcs=5, label=label)

            if label in [None, "default-behavior"]:
                expected_a = "RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)"
                exp_name = "RaisedCosineLinearEval"
            else:
                expected_a = (
                    f"'{label}': RaisedCosineLinearEval(n_basis_funcs=5, width=2.0)"
                )
                exp_name = label
            bas = bas * self.instantiate_basis(
                6, basis.MSplineEval, basis_class_specific_params
            )
            expected = f"'({exp_name} * MSplineEval)': MultiplicativeBasis(\n    basis1={expected_a},\n    basis2=MSplineEval(n_basis_funcs=6, order=4),\n)"
            out = repr(bas)
            assert out == expected

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("window_size", [10])
    def test_compute_features_returns_expected_number_of_basis(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the evaluation of the `MultiplicativeBasis` results in a number of basis
        that is the product of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )

        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[1] != basis_a_obj.n_basis_funcs * basis_b_obj.n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the "
                "fit_transformed basis."
                f"The number of basis is {n_basis_a * n_basis_b}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [12, 30, 35])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("window_size", [10])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the output sample size from the `MultiplicativeBasis` fit_transform function
        matches the input sample size.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("window_size", [10])
    def test_number_of_required_inputs_compute_features(
        self,
        n_input,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the number of required inputs for the `compute_features` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match=r"This basis requires \d+ input\(s\)."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_meshgrid_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the resulting meshgrid size matches the sample size input.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj * basis_b_obj
        res = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )
        for grid in res[:-1]:
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_basis_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number sample size output by evaluate_on_grid matches the sample size of the input.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )[-1]
        assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2, 5, 6, 11, 30])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_input_number(
        self,
        n_input,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number of inputs provided to `evaluate_on_grid` matches
        the sum of the number of input samples required from each of the basis objects.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj * basis_b_obj
        inputs = [20] * n_input
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match=r"This basis requires \d+ input\(s\)."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    @pytest.mark.parametrize("sample_size_a", [11, 12])
    @pytest.mark.parametrize("sample_size_b", [11, 12])
    def test_inconsistent_sample_sizes(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        sample_size_a,
        sample_size_b,
        basis_class_specific_params,
    ):
        """Test that the inputs of inconsistent sample sizes result in an exception when compute_features is called"""
        raise_exception = sample_size_a != sample_size_b
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        input_a = [
            np.linspace(0, 1, sample_size_a)
        ] * basis_a_obj._n_input_dimensionality
        input_b = [
            np.linspace(0, 1, sample_size_b)
        ] * basis_b_obj._n_input_dimensionality
        basis_obj = basis_a_obj * basis_b_obj
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Sample size mismatch\. Input elements have inconsistent",
            ):
                basis_obj.compute_features(*input_a, *input_b)
        else:
            basis_obj.compute_features(*input_a, *input_b)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    def test_pynapple_support_compute_features(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_class_specific_params,
    ):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        basis_prod = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        ) * self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        out = basis_prod.compute_features(*([inp] * basis_prod._n_input_dimensionality))
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize(" window_size", [8])
    def test_call_input_num(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        num_input,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                TypeError, match=r"This basis requires \d+ input\(s\)"
            )
        with expectation:
            basis_obj.evaluate(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize("basis_a", list_all_real_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2])
    @pytest.mark.parametrize(" window_size", [8])
    def test_set_input_shape_input_num(
        self,
        n_basis_a,
        basis_a,
        num_input,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = basis_a_obj.__sklearn_clone__()
        basis_obj = basis_a_obj * basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(ValueError, match="set_input_shape expects")
        with expectation:
            basis_obj.set_input_shape(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp",
        [
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 10)[:, None],
            np.random.randn(10, 2, 3),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_input_shape(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        inp,
        window_size,
        basis_class_specific_params,
    ):
        does_raise = (
            any(b == AdditiveBasis for b in (basis_a, basis_b)) and inp.ndim != 1
        )
        does_raise |= (
            any(b == basis.HistoryConv for b in (basis_a, basis_b)) and inp.ndim > 2
        )
        if does_raise:
            expectation = pytest.raises(
                ValueError,
                match="Input sample must be one dimensional|`evaluate` for HistoryBasis",
            )
        else:
            expectation = does_not_raise()
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        with expectation:
            out = basis_obj.evaluate(*([inp] * basis_obj._n_input_dimensionality))
            assert out.shape[:-1] == inp.shape
            assert out.shape[-1] == basis_obj.n_basis_funcs

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_axis(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        time_axis_shape,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        inp = [np.linspace(0, 1, time_axis_shape)] * basis_obj._n_input_dimensionality
        assert basis_obj.evaluate(*inp).shape[0] == time_axis_shape

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_nan(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        if (
            basis_a == basis.OrthExponentialBasis
            or basis_b == basis.OrthExponentialBasis
        ):
            pytest.skip(
                f"Skipping test_call_sample_range for {basis_a.__name__} and {basis_b.__name__}"
            )
        if basis_a is HistoryConv or basis_b is HistoryConv:
            pytest.skip(
                f"Skipping test_call_sample_range for {basis_a.__name__} and {basis_b.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        inp = [np.linspace(0, 1, 10)] * basis_obj._n_input_dimensionality
        for x in inp:
            x[3] = np.nan
        assert all(np.isnan(basis_obj.evaluate(*inp)[3]))

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_equivalent_in_conv(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        if (
            basis_a == HistoryConv
            or basis_b == HistoryConv
            or basis_a == CustomBasis
            or basis_b == CustomBasis
        ):
            # evaluate returns identity
            pytest.skip(
                f"Skipping test_call_nan for {basis_a.__name__} and {basis_b.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas_eva = basis_a_obj * basis_b_obj

        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=8
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=8
        )
        bas_con = basis_a_obj * basis_b_obj

        x = [np.linspace(0, 1, 10)] * bas_con._n_input_dimensionality
        assert np.all(bas_con.evaluate(*x) == bas_eva.evaluate(*x))

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_pynapple_support(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        x = np.linspace(0, 1, 10)
        x_nap = [nap.Tsd(t=np.arange(10), d=x)] * bas._n_input_dimensionality
        x = [x] * bas._n_input_dimensionality
        y = bas.evaluate(*x)
        y_nap = bas.evaluate(*x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap[0].t)

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [6, 7])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_basis_number(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
        assert (
            bas.evaluate(*x).shape[1]
            == basis_a_obj.n_basis_funcs * basis_b_obj.n_basis_funcs
        )

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_non_empty(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        with pytest.raises(ValueError, match="All sample provided must"):
            bas.evaluate(*([np.array([])] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
            (0.1, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_range(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        mn,
        mx,
        expectation,
        window_size,
        basis_class_specific_params,
    ):
        if expectation == "check":
            if (
                basis_a == basis.OrthExponentialBasis
                or basis_b == basis.OrthExponentialBasis
            ):
                expectation = pytest.raises(
                    ValueError, match="OrthExponentialBasis requires positive samples"
                )
            else:
                expectation = does_not_raise()
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        with expectation:
            bas.evaluate(*([np.linspace(mn, mx, 10)] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_fit_kernel(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj * basis_b_obj
        bas._set_input_independent_states()

        def check_kernel(basis_obj):
            has_kern = []
            if hasattr(basis_obj, "basis1"):
                has_kern += check_kernel(basis_obj.basis1)
                has_kern += check_kernel(basis_obj.basis2)
            else:
                has_kern += [
                    (
                        basis_obj.kernel_ is not None
                        if basis_obj.__class__.__name__.endswith("Conv")
                        else True
                    )
                ]
            return has_kern

        assert all(check_kernel(bas))

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_transform_fails(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        if basis_a == CustomBasis or basis_b == CustomBasis:
            pytest.skip(
                f"Skipping test_transform_fails for {basis_a.__name__} and {basis_b.__name__}"
            )
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj * basis_b_obj
        if "Eval" in basis_a.__name__ and "Eval" in basis_b.__name__:
            context = does_not_raise()
        else:
            context = pytest.raises(
                RuntimeError,
                match="You must call `setup_basis` before `_compute_features`",
            )
        with context:
            x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
            # needed for re-shaping krons
            bas.set_input_shape(*x)
            bas._compute_features(*x)

    @pytest.mark.parametrize("n_basis_input", [1, 2, 3])
    def test_set_num_output_features(self, n_basis_input):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(11, window_size=10)
        bas_mul = bas1 * bas2
        assert bas_mul.n_output_features is None
        bas_mul.compute_features(
            np.ones((20, n_basis_input)), np.ones((20, n_basis_input))
        )
        assert bas_mul.n_output_features == (n_basis_input * 10 * 11)

    @pytest.mark.parametrize("n_basis_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_basis_input):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas_mul = bas1 * bas2
        assert bas_mul._input_shape_product is None
        bas_mul.compute_features(
            np.ones((20, n_basis_input)), np.ones((20, n_basis_input))
        )
        assert bas_mul._input_shape_product == (n_basis_input, n_basis_input)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (
                3,
                pytest.raises(
                    ValueError,
                    match="MultiplicativeBasis requires all inputs to have",
                ),
            ),
            (
                0,
                pytest.raises(
                    ValueError,
                    match="MultiplicativeBasis requires all inputs to have",
                ),
            ),
            (2, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError,
                    match="MultiplicativeBasis requires all inputs to have",
                ),
            ),
            (
                4,
                pytest.raises(
                    ValueError,
                    match="MultiplicativeBasis requires all inputs to have",
                ),
            ),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas = bas1 * bas2
        x = np.random.randn(20, 2), np.random.randn(20, 2)
        bas.compute_features(*x)
        with expectation:
            x = np.random.randn(30, 2), np.random.randn(30, n_input)
            bas.compute_features(*x)

    @pytest.mark.parametrize("n_basis_input", [1, 2, 3])
    def test_input_shape_product(self, n_basis_input):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas_prod = bas1 * bas2
        bas_prod.compute_features(
            np.ones((20, n_basis_input)), np.ones((20, n_basis_input))
        )
        assert bas_prod._input_shape_product == (n_basis_input, n_basis_input)

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    @pytest.mark.parametrize("shape", [1, (), np.ones(3)])
    @pytest.mark.parametrize("add_shape", [(), (1,)])
    def test_set_input_shape_type_1d_arrays(
        self,
        basis_a,
        basis_b,
        shape,
        basis_class_specific_params,
        add_shape,
    ):
        x = (np.ones((10, *add_shape)), np.ones((10, *add_shape)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b

        mul.set_input_shape(shape, shape)
        mul.compute_features(*x)
        assert all(
            xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
            for ishape, xi in zip(mul.input_shape, x)
        )

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    @pytest.mark.parametrize("shape", [2, (2,), np.ones((3, 2))])
    @pytest.mark.parametrize("add_shape", [(), (1,)])
    def test_set_input_shape_type_2d_arrays(
        self,
        basis_a,
        basis_b,
        shape,
        basis_class_specific_params,
        add_shape,
    ):
        x = (np.ones((10, 1, *add_shape)), np.ones((10, 1, *add_shape)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b

        mul.set_input_shape(shape, shape)
        mul.compute_features(*x)
        assert all(
            xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
            for ishape, xi in zip(mul.input_shape, x)
        )

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    @pytest.mark.parametrize("shape", [(2, 2), np.ones((3, 2, 2))])
    @pytest.mark.parametrize("add_shape", [(), (1,)])
    def test_set_input_shape_type_nd_arrays(
        self,
        basis_a,
        basis_b,
        shape,
        basis_class_specific_params,
        add_shape,
    ):
        x = (np.ones((10, 3, 1, *add_shape)), np.ones((10, 3, 1, *add_shape)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b

        mul.set_input_shape(shape, shape)
        mul.compute_features(*x)
        assert all(
            xi.shape[1:] == ishape if xi.ndim != 1 else () == ishape
            for ishape, xi in zip(mul.input_shape, x)
        )

    @pytest.mark.parametrize(
        "inp_shape, expectation",
        [
            (((1, 1), (1, 1)), does_not_raise()),
            (
                ((1, 1.0), (1, 1)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
            (
                ((1, 1), (1, 1.0)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_set_input_value_types(
        self, inp_shape, expectation, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b
        with expectation:
            mul.set_input_shape(*inp_shape)

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_deep_copy_basis(self, basis_a, basis_b, basis_class_specific_params):

        if basis_a == HistoryConv:
            n_basis_a = 10
        elif basis_a == IdentityEval:
            n_basis_a = 1
        else:
            n_basis_a = 5
        if basis_b == HistoryConv:
            n_basis_b = 10
        elif basis_b == IdentityEval:
            n_basis_b = 1
        else:
            n_basis_b = 5

        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b
        # test pointing to different objects
        assert id(mul.basis1) != id(basis_a)
        assert id(mul.basis1) != id(basis_b)
        assert id(mul.basis2) != id(basis_a)
        assert id(mul.basis2) != id(basis_b)

        if isinstance(basis_a, (HistoryConv, IdentityEval, CustomBasis)) or isinstance(
            basis_b, (HistoryConv, IdentityEval, CustomBasis)
        ):
            return
        # test attributes are not related
        set_basis_attr(basis_a, 10)
        assert get_basis_attr(mul.basis1) != 10
        set_basis_attr(mul.basis1, 6)
        assert basis_a.n_basis_funcs != 6

        set_basis_attr(basis_b, 10)
        assert get_basis_attr(mul.basis2) != 10
        set_basis_attr(mul.basis2, 6)
        assert basis_b.n_basis_funcs != 6

    @pytest.mark.parametrize(
        "basis_a, basis_b",
        create_atomic_basis_pairs(
            list_all_basis_classes("Eval")
            + list_all_basis_classes("Conv")
            + [CustomBasis]
        ),
    )
    def test_compute_n_basis_runtime(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        if basis_a == HistoryConv:
            n_basis_a = 10
        elif basis_a == IdentityEval:
            n_basis_a = 1
        else:
            n_basis_a = 5
        if basis_b == HistoryConv:
            n_basis_b = 10
        elif basis_b == IdentityEval:
            n_basis_b = 1
        else:
            n_basis_b = 5
        basis_a = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )

        mul = basis_a * basis_b
        if not isinstance(mul.basis1, (HistoryConv, IdentityEval, CustomBasis)):
            set_basis_attr(mul.basis1, 10)
            assert mul.n_basis_funcs == 10 * n_basis_b
        if not isinstance(mul.basis2, (HistoryConv, IdentityEval, CustomBasis)):
            set_basis_attr(mul.basis2, 10)
            assert mul.n_basis_funcs == 10 * mul.basis1.n_basis_funcs

    @pytest.mark.parametrize(
        "basis_a, basis_b", create_atomic_basis_pairs(list_all_real_basis_classes())
    )
    def test_runtime_n_basis_out_compute(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_a.set_input_shape(
            *([1] * basis_a._n_input_dimensionality)
        ).to_transformer()
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        basis_b.set_input_shape(
            *([1] * basis_b._n_input_dimensionality)
        ).to_transformer()
        mul = basis_a * basis_b
        mul.set_input_shape(*[2] * mul._n_input_dimensionality)
        assert (
            mul.n_output_features
            == 2 * mul.basis1.n_basis_funcs * mul.basis2.n_basis_funcs
        )

    @pytest.mark.parametrize(
        "x_shape",
        [
            (10, 3),  # 2D inputs (1D vectorization)
            (10, 3, 4),  # 3D inputs (2D vectorization)
            (10, 2, 3, 4),  # 4D inputs (3D vectorization)
        ],
    )
    @pytest.mark.parametrize(
        "bas",
        list_all_real_basis_classes("Eval")
        + list_all_real_basis_classes("Conv")
        + [CustomBasis],
    )
    def test_vectorization_equivalence(self, bas, x_shape, basis_class_specific_params):
        """Test that vectorized computation equals explicit nested loops."""
        bas = (
            self.instantiate_basis(5, bas, basis_class_specific_params, window_size=10)
            ** 2
        )
        # Seed for reproducibility
        np.random.seed(42)

        # Create random inputs
        x = np.random.randn(*x_shape)
        y = np.random.randn(*x_shape)

        # Create basis (IdentityEval has 1 basis function)
        n_basis_funcs = bas.n_basis_funcs  # Should be 1 for IdentityEval ** 2

        # Get vectorized result
        vectorized_result = bas.compute_features(x, y)

        # Compute expected result with explicit loops
        n_samples = x_shape[0]
        vec_shape = x_shape[1:]  # vectorized dimensions

        # Initialize output array
        out = np.empty((*x.shape, n_basis_funcs))

        # Generate all combinations of vectorized indices
        vec_indices = itertools.product(*[range(dim) for dim in vec_shape])

        for indices in vec_indices:
            # Extract 1D slices for this combination of indices
            x_slice = x[(slice(None),) + indices]  # x[:, i, j, ...]
            y_slice = y[(slice(None),) + indices]  # y[:, i, j, ...]

            # Compute features for this slice
            slice_result = bas.compute_features(x_slice, y_slice)

            # Store in output array
            out[(slice(None),) + indices + (slice(None),)] = slice_result

            # Reshape to match expected output format: (n_samples, flattened_features)
            expected_result = out.reshape(n_samples, -1)

        # Verify equivalence
        np.testing.assert_array_equal(vectorized_result, expected_result)

        # Also verify shapes are correct
        expected_n_features = (
            np.prod(vec_shape) * n_basis_funcs if vec_shape else n_basis_funcs
        )
        assert vectorized_result.shape == (n_samples, expected_n_features)

    @pytest.mark.parametrize(
        "bas",
        list_all_real_basis_classes("Eval") + [CustomBasis],
    )
    @pytest.mark.parametrize("x, y", [(np.random.randn(10, 2), np.random.randn(10, 2))])
    def test_eval_and_compute_features_equivalence(
        self, x, y, bas, basis_class_specific_params
    ):
        """
        Test the evaluate/compute_features equivalence for Eval bases.

        This is not true for Conv bases, where the two methods perform different
        operations.
        """
        bas = (
            self.instantiate_basis(5, bas, basis_class_specific_params, window_size=10)
            ** 2
        )
        X = bas.compute_features(x, y).reshape(x.shape[0], -1, bas.n_basis_funcs)
        Y = bas.evaluate(x, y)
        np.testing.assert_array_equal(X, Y)

    @pytest.mark.parametrize(
        "real_cls",
        list_all_real_basis_classes("Eval")
        + list_all_real_basis_classes("Conv")
        + [CustomBasis],
    )
    @pytest.mark.parametrize("complex_cls", [basis.FourierEval])
    def test_multiply_complex(self, real_cls, complex_cls, basis_class_specific_params):
        basis_real = self.instantiate_basis(
            5, real_cls, basis_class_specific_params, window_size=10
        )
        basis_complex = self.instantiate_basis(
            5, complex_cls, basis_class_specific_params, window_size=10
        )
        new_complex = basis_real * basis_complex
        assert new_complex.is_complex

        with pytest.raises(
            ValueError, match="Invalid multiplication between two complex bases"
        ):
            new_complex * basis_complex

        with pytest.raises(
            ValueError, match="Invalid multiplication between two complex bases"
        ):
            basis_complex * basis_complex

        with pytest.raises(
            ValueError, match="Invalid multiplication between two complex bases"
        ):
            basis_complex * basis_real * basis_complex


@pytest.mark.parametrize(
    "exponent", [-1, 0, 0.5, basis.RaisedCosineLogEval(4), 1, 2, 3]
)
@pytest.mark.parametrize("basis_class", list_all_real_basis_classes())
def test_power_of_basis(exponent, basis_class, basis_class_specific_params):
    """Test if the power behaves as expected."""
    raise_exception_type = not type(exponent) is int

    if not raise_exception_type:
        raise_exception_value = exponent <= 0
    else:
        raise_exception_value = False

    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=5
    )

    if raise_exception_type:
        with pytest.raises(TypeError, match=r"Basis exponent should be an integer\!"):
            basis_obj**exponent
    elif raise_exception_value:
        with pytest.raises(
            ValueError, match=r"Basis exponent should be a non-negative integer\!"
        ):
            basis_obj**exponent
    else:
        basis_pow = basis_obj**exponent
        samples = np.linspace(0, 1, 10)
        eval_pow = basis_pow.compute_features(
            *[samples] * basis_pow._n_input_dimensionality
        )

        if exponent == 2:
            basis_obj = basis_obj * basis_obj
        elif exponent == 3:
            basis_obj = basis_obj * basis_obj * basis_obj

        non_nan = ~np.isnan(eval_pow)
        out = basis_obj.compute_features(*[samples] * basis_obj._n_input_dimensionality)
        assert np.allclose(
            eval_pow[non_nan],
            out[non_nan],
        )
        assert np.all(np.isnan(out[~non_nan]))


@pytest.mark.parametrize("basis_class", list_all_real_basis_classes())
def test_power_of_basis_repr(basis_class, basis_class_specific_params):
    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=5
    )
    pow_basis = basis_obj**3
    actual_labels = list(l for l, _ in generate_basis_label_pair(pow_basis))
    assert len(actual_labels) == len(set(actual_labels))
    cls_names = {b.__class__.__name__ for b in pow_basis._iterate_over_components()}
    count_cls = {c: 0 for c in cls_names}
    lab_additive_expected = []
    for b in pow_basis._iterate_over_components():
        k = count_cls[b.__class__.__name__]
        lab_additive_expected.append(
            b.__class__.__name__ + f"_{k}" if k > 0 else b.__class__.__name__
        )
        count_cls[b.__class__.__name__] += 1
    list_additive_actual = [b.label for b in pow_basis._iterate_over_components()]
    assert set(lab_additive_expected) == set(list_additive_actual)


@pytest.mark.parametrize("basis_class", list_all_real_basis_classes())
def test_mul_of_basis_repr(basis_class, basis_class_specific_params):
    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=5
    )
    mul_basis = basis_obj * 3
    actual_labels = list(l for l, _ in generate_basis_label_pair(mul_basis))
    assert len(actual_labels) == len(set(actual_labels))
    cls_names = {b.__class__.__name__ for b in mul_basis._iterate_over_components()}
    count_cls = {c: 0 for c in cls_names}
    lab_additive_expected = []
    for b in mul_basis._iterate_over_components():
        k = count_cls[b.__class__.__name__]
        lab_additive_expected.append(
            b.__class__.__name__ + f"_{k}" if k > 0 else b.__class__.__name__
        )
        count_cls[b.__class__.__name__] += 1
    list_additive_actual = [b.label for b in mul_basis._iterate_over_components()]
    assert set(lab_additive_expected) == set(list_additive_actual)


@pytest.mark.parametrize("mul", [-1, 0, 0.5, 1, 2, 3])
@pytest.mark.parametrize("basis_class", list_all_real_basis_classes())
def test_mul_of_basis_by_int(mul, basis_class, basis_class_specific_params):
    """Test if the power behaves as expected."""
    raise_exception_type = not isinstance(mul, int)

    if not raise_exception_type:
        raise_exception_value = mul <= 0
    else:
        raise_exception_value = False

    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=5
    )

    if raise_exception_type:
        with pytest.raises(TypeError, match=r"Basis multiplicative factor should be"):
            basis_obj * mul
    elif raise_exception_value:
        with pytest.raises(ValueError, match=r"Basis multiplication error"):
            basis_obj * mul
    else:

        for basis_mul in [basis_obj * mul, mul * basis_obj]:
            samples = np.linspace(0, 1, 10)
            eval_mul = basis_mul.compute_features(
                *[samples] * basis_mul._n_input_dimensionality
            )

            if mul == 2:
                basis_add = basis_obj + basis_obj
            elif mul == 3:
                basis_add = basis_obj + basis_obj + basis_obj
            else:
                basis_add = basis_obj
            non_nan = ~np.isnan(eval_mul)
            out = basis_add.compute_features(
                *[samples] * basis_add._n_input_dimensionality
            )
            assert np.allclose(
                eval_mul[non_nan],
                out[non_nan],
            )
            assert np.all(np.isnan(out[~non_nan]))


@pytest.mark.parametrize(
    "basis_class",
    list_all_real_basis_classes("Eval")
    + list_all_real_basis_classes("Conv")
    + [CustomBasis],
)
def test_mul_of_basis_from_nested(basis_class, basis_class_specific_params):
    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=5
    )
    add = basis_obj * 2
    b1 = add.basis1 * 1
    # use deep copy
    b2 = add.basis1
    with pytest.raises(AssertionError):
        compare_basis(b1, b2)
    b2._parent = None
    compare_basis(b1, b2)
    # nest one more
    add = basis_obj * 3
    b1 = add.basis1 * 1
    # use deep copy
    b2 = add.basis1
    with pytest.raises(AssertionError):
        compare_basis(b1, b2)
    b2._parent = None
    compare_basis(b1, b2)


@pytest.mark.parametrize(
    "basis_class",
    list_all_real_basis_classes("Eval")
    + list_all_real_basis_classes("Conv")
    + [CustomBasis],
)
def test_pow_of_basis_from_nested(basis_class, basis_class_specific_params):
    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=5
    )
    add = basis_obj * 2
    b1 = add.basis1**1
    # use deep copy
    b2 = add.basis1
    with pytest.raises(AssertionError):
        compare_basis(b1, b2)
    b2._parent = None
    compare_basis(b1, b2)
    # nest one more
    add = basis_obj * 3
    b1 = add.basis1**1
    # use deep copy
    b2 = add.basis1
    with pytest.raises(AssertionError):
        compare_basis(b1, b2)
    b2._parent = None
    compare_basis(b1, b2)


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_basis_to_transformer(basis_cls, basis_class_specific_params):
    n_basis_funcs = 5
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas = bas.set_input_shape(
        *([1] * bas._n_input_dimensionality)
    ).to_transformer()

    assert isinstance(trans_bas, TransformerBasis)

    # check that things like n_basis_funcs are the same as the original basis
    for k in bas.__dict__.keys():
        # skip for add and multiplicative.
        if basis_cls in [AdditiveBasis, MultiplicativeBasis]:
            continue
        if k in ["_funcs", "_frequencies"]:
            f1s, f2s = getattr(bas, k), getattr(trans_bas, k)
            assert np.all(f1 == f2 for f1, f2 in zip(f1s, f2s))
        else:
            assert np.all(getattr(bas, k) == getattr(trans_bas, k))


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(
            t=np.arange(100),
            d=np.arange(100),
            time_support=nap.IntervalSet(start=[0, 50], end=[20, 75]),
        )
    ],
)
@pytest.mark.parametrize(
    "window_size, shift, predictor_causality, nan_index",
    [
        (3, True, "causal", [0, 1, 2, 50, 51, 52]),
        (2, True, "causal", [0, 1, 50, 51]),
        (3, False, "causal", [0, 1, 50, 51]),
        (2, False, "causal", [0, 50]),
        (2, None, "causal", [0, 1, 50, 51]),
        (3, True, "anti-causal", [20, 19, 18, 75, 74, 73]),
        (2, True, "anti-causal", [20, 19, 75, 74]),
        (3, False, "anti-causal", [20, 19, 75, 74]),
        (2, False, "anti-causal", [20, 75]),
        (2, None, "anti-causal", [20, 19, 75, 74]),
        (3, False, "acausal", [0, 20, 50, 75]),
        (5, False, "acausal", [0, 1, 19, 20, 50, 51, 74, 75]),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv"),
)
def test_multi_epoch_pynapple_basis(
    basis_cls,
    tsd,
    window_size,
    shift,
    predictor_causality,
    nan_index,
    basis_class_specific_params,
):
    """Test nan location in multi-epoch pynapple tsd."""
    kwargs = dict(
        conv_kwargs=dict(shift=shift, predictor_causality=predictor_causality)
    )

    # require a ws of at least nbasis funcs.
    if "OrthExp" in basis_cls.__name__:
        nbasis = 2
    # splines requires at least 1 basis more than the order of the spline.
    else:
        nbasis = 5
    bas = CombinedBasis().instantiate_basis(
        nbasis,
        basis_cls,
        basis_class_specific_params,
        window_size=window_size,
        **kwargs,
    )

    n_input = bas._n_input_dimensionality

    res = bas.compute_features(*([tsd] * n_input))

    nan_index = np.sort(nan_index)
    times_nan_found = res[np.isnan(res.d[:, 0])].t
    assert len(times_nan_found) == len(nan_index)
    assert np.all(times_nan_found == np.array(nan_index))
    idx_nan = [np.where(res.t == k)[0][0] for k in nan_index]
    assert np.all(np.isnan(res.d[idx_nan]))


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(
            t=np.arange(100),
            d=np.arange(100),
            time_support=nap.IntervalSet(start=[0, 50], end=[20, 75]),
        )
    ],
)
@pytest.mark.parametrize(
    "window_size, shift, predictor_causality, nan_index",
    [
        (3, True, "causal", [0, 1, 2, 50, 51, 52]),
        (2, True, "causal", [0, 1, 50, 51]),
        (3, False, "causal", [0, 1, 50, 51]),
        (2, False, "causal", [0, 50]),
        (2, None, "causal", [0, 1, 50, 51]),
        (3, True, "anti-causal", [20, 19, 18, 75, 74, 73]),
        (2, True, "anti-causal", [20, 19, 75, 74]),
        (3, False, "anti-causal", [20, 19, 75, 74]),
        (2, False, "anti-causal", [20, 75]),
        (2, None, "anti-causal", [20, 19, 75, 74]),
        (3, False, "acausal", [0, 20, 50, 75]),
        (5, False, "acausal", [0, 1, 19, 20, 50, 51, 74, 75]),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_real_basis_classes("Conv"),
)
def test_multi_epoch_pynapple_basis_transformer(
    basis_cls,
    tsd,
    window_size,
    shift,
    predictor_causality,
    nan_index,
    basis_class_specific_params,
):
    """Test nan location in multi-epoch pynapple tsd."""
    kwargs = dict(
        conv_kwargs=dict(shift=shift, predictor_causality=predictor_causality)
    )
    # require a ws of at least nbasis funcs.
    if "OrthExp" in basis_cls.__name__:
        nbasis = 2
    # splines requires at least 1 basis more than the order of the spline.
    else:
        nbasis = 5

    bas = CombinedBasis().instantiate_basis(
        nbasis,
        basis_cls,
        basis_class_specific_params,
        window_size=window_size,
        **kwargs,
    )

    n_input = bas._n_input_dimensionality

    # concat input
    X = pynapple_concatenate_numpy([tsd[:, None]] * n_input, axis=1)

    # run convolutions
    # pass through transformer
    bas.set_input_shape(X)
    bas = TransformerBasis(bas)
    res = bas.fit_transform(X)

    # check nans
    nan_index = np.sort(nan_index)
    times_nan_found = res[np.isnan(res.d[:, 0])].t
    assert len(times_nan_found) == len(nan_index)
    assert np.all(times_nan_found == np.array(nan_index))
    idx_nan = [np.where(res.t == k)[0][0] for k in nan_index]
    assert np.all(np.isnan(res.d[idx_nan]))


@pytest.mark.parametrize(
    "bas1, bas2", create_atomic_basis_pairs(list_all_basis_classes())
)
@pytest.mark.parametrize(
    "operator1, operator2, compute_slice",
    [
        (
            "__add__",
            "__add__",
            lambda bas1, bas2, bas3: {
                "1": slice(0, bas1._input_shape_product[0] * bas1.n_basis_funcs),
                "2": slice(
                    bas1._input_shape_product[0] * bas1.n_basis_funcs,
                    bas1._input_shape_product[0] * bas1.n_basis_funcs
                    + bas2._input_shape_product[0] * bas2.n_basis_funcs,
                ),
                "3": slice(
                    bas1._input_shape_product[0] * bas1.n_basis_funcs
                    + bas2._input_shape_product[0] * bas2.n_basis_funcs,
                    bas1._input_shape_product[0] * bas1.n_basis_funcs
                    + bas2._input_shape_product[0] * bas2.n_basis_funcs
                    + bas3._input_shape_product[0] * bas3.n_basis_funcs,
                ),
            },
        ),
        (
            "__add__",
            "__mul__",
            lambda bas1, bas2, bas3: {
                "1": slice(0, bas1._input_shape_product[0] * bas1.n_basis_funcs),
                "(2 * 3)": slice(
                    bas1._input_shape_product[0] * bas1.n_basis_funcs,
                    bas1._input_shape_product[0] * bas1.n_basis_funcs
                    + bas2._input_shape_product[0]
                    * bas2.n_basis_funcs
                    * bas3.n_basis_funcs,
                ),
            },
        ),
        (
            "__mul__",
            "__add__",
            lambda bas1, bas2, bas3: {
                # note that it doesn't respect algebra order but execute right to left (first add then multiplies)
                "(1 * (2 + 3))": slice(
                    0,
                    bas1._input_shape_product[0]
                    * bas1.n_basis_funcs
                    * (bas2.n_basis_funcs + bas3.n_basis_funcs),
                ),
            },
        ),
        (
            "__mul__",
            "__mul__",
            lambda bas1, bas2, bas3: {
                "(1 * (2 * 3))": slice(
                    0,
                    bas1._input_shape_product[0]
                    * bas1.n_basis_funcs
                    * bas2.n_basis_funcs
                    * bas3.n_basis_funcs,
                ),
            },
        ),
    ],
)
def test__get_splitter(
    bas1, bas2, operator1, operator2, compute_slice, basis_class_specific_params
):
    # reduce the number of test
    bas3 = bas2
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, TransformerBasis)
        for bas in [bas1, bas2, bas3]
    ):
        return
    if any(o == "__mul__" for o in (operator1, operator2)) and any(
        issubclass(bas, FourierBasis) for bas in (bas1, bas2, bas3)
    ):
        pytest.skip("skip multiplication involving fourier basis.")
    # define the basis
    n_basis = [5, 6, 7]
    n_input_basis = [1, 2, 3]

    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        n_basis[0], bas1, basis_class_specific_params, window_size=10, label="1"
    )
    bas2_instance = combine_basis.instantiate_basis(
        n_basis[1], bas2, basis_class_specific_params, window_size=10, label="2"
    )
    bas3_instance = combine_basis.instantiate_basis(
        n_basis[2], bas3, basis_class_specific_params, window_size=10, label="3"
    )

    func1 = getattr(bas1_instance, operator1)
    func2 = getattr(bas2_instance, operator2)
    bas23 = func2(bas3_instance)
    bas123 = func1(bas23)
    if "__mul__" in [operator1, operator2]:
        inps = [np.zeros((1, 2)) for _ in n_input_basis]
    else:
        inps = [np.zeros((1, n)) if n > 1 else np.zeros((1,)) for n in n_input_basis]
    bas123.set_input_shape(*inps)
    for b, i in zip([bas1_instance, bas2_instance, bas3_instance], inps):
        b.set_input_shape(i)
    splitter_dict, _ = bas123._get_feature_slicing()
    exp_slices = compute_slice(bas1_instance, bas2_instance, bas3_instance)
    assert exp_slices == splitter_dict


@pytest.mark.parametrize("bas1", list_all_basis_classes())
def test_duplicate_keys(bas1, basis_class_specific_params):
    # skip nested
    if bas1 in (AdditiveBasis, MultiplicativeBasis, TransformerBasis):
        return

    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        5, bas1, basis_class_specific_params, window_size=10
    )
    bas2_instance = combine_basis.instantiate_basis(
        5, bas1, basis_class_specific_params, window_size=10
    )
    bas3_instance = combine_basis.instantiate_basis(
        5, bas1, basis_class_specific_params, window_size=10
    )
    bas_obj = bas1_instance + bas2_instance + bas3_instance

    inps = [np.zeros((1,)) for _ in range(3)]
    bas_obj.set_input_shape(*inps)
    slice_dict = bas_obj._get_feature_slicing()[0]
    expected_label = bas1_instance.__class__.__name__
    assert tuple(slice_dict.keys()) == (
        expected_label,
        expected_label + "_1",
        expected_label + "_2",
    )


@pytest.mark.parametrize(
    "bas1, bas2",
    create_atomic_basis_pairs(list_all_basis_classes()),
)
def test_label_uniqueness_enforcing(bas1, bas2, basis_class_specific_params):
    bas3 = bas2
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, TransformerBasis)
        for bas in [bas1, bas2, bas3]
    ):
        return

    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        5, bas1, basis_class_specific_params, window_size=10, label="x"
    )
    bas2_instance = combine_basis.instantiate_basis(
        5, bas2, basis_class_specific_params, window_size=10, label="x"
    )

    if sum([b.is_complex for b in (bas1_instance, bas2_instance)]) > 1:
        pytest.skip("Cannot multiply more than one complex basis.")

    err_msg = "All user-provided labels of basis elements must be distinct"
    with pytest.raises(ValueError, match=err_msg):
        bas1_instance + bas2_instance

    with pytest.raises(ValueError, match=err_msg):
        AdditiveBasis(bas1_instance, bas2_instance)

    with pytest.raises(ValueError, match=err_msg):
        bas1_instance * bas2_instance

    with pytest.raises(ValueError, match=err_msg):
        MultiplicativeBasis(bas1_instance, bas2_instance)

    bas2_instance.label = "y"
    add = bas1_instance + bas2_instance
    mul = bas1_instance * bas2_instance

    # check error when setting attr directly
    setter_error_msg = r"Label '[xyz]' is already in use. When user-provided"
    with pytest.raises(ValueError, match=setter_error_msg):
        add.basis1.label = "y"
    with pytest.raises(ValueError, match=setter_error_msg):
        add.basis2.label = "x"
    with pytest.raises(ValueError, match=setter_error_msg):
        mul.basis1.label = "y"
    with pytest.raises(ValueError, match=setter_error_msg):
        mul.basis2.label = "x"

    # check error when using set_params
    with pytest.raises(ValueError, match=setter_error_msg):
        add.set_params(x__label="y")
    with pytest.raises(ValueError, match=setter_error_msg):
        add.set_params(y__label="x")
    with pytest.raises(ValueError, match=setter_error_msg):
        mul.set_params(x__label="y")
    with pytest.raises(ValueError, match=setter_error_msg):
        mul.set_params(y__label="x")

    # add more nesting
    bas3_instance = combine_basis.instantiate_basis(
        5, bas3, basis_class_specific_params, window_size=10, label="x"
    )
    if sum([b.is_complex for b in (bas1_instance, bas2_instance, bas3_instance)]) > 1:
        pytest.skip("Cannot multiply more than one complex basis.")
    # add
    with pytest.raises(ValueError, match=err_msg):
        add + bas3_instance

    with pytest.raises(ValueError, match=err_msg):
        AdditiveBasis(add, bas3_instance)

    with pytest.raises(ValueError, match=err_msg):
        add * bas3_instance

    with pytest.raises(ValueError, match=err_msg):
        MultiplicativeBasis(add, bas3_instance)

    # mul
    with pytest.raises(ValueError, match=err_msg):
        mul + bas3_instance

    with pytest.raises(ValueError, match=err_msg):
        AdditiveBasis(mul, bas3_instance)

    with pytest.raises(ValueError, match=err_msg):
        mul * bas3_instance

    with pytest.raises(ValueError, match=err_msg):
        MultiplicativeBasis(mul, bas3_instance)

    bas3_instance.label = "z"
    for bas in [add, mul]:
        for meth in ["__add__", "__mul__"]:
            meth = getattr(bas, meth)
            comb = meth(bas3_instance)
            for lab1 in ["x", "y", "z"]:
                for lab2 in ["x", "y", "z"]:
                    if lab1 == lab2:
                        continue
                    with pytest.raises(ValueError, match=setter_error_msg):
                        comb.set_params(**{f"{lab1}__label": lab2})

                    with pytest.raises(ValueError, match=setter_error_msg):
                        comb[lab1].label = lab2

                with pytest.raises(ValueError, match=setter_error_msg):
                    comb.label = lab1
                with pytest.raises(ValueError, match=setter_error_msg):
                    comb.basis1.label = "z"
                with pytest.raises(ValueError, match=setter_error_msg):
                    comb[comb.basis1.label].label = lab2
                with pytest.raises(ValueError, match=setter_error_msg):
                    comb.set_params(**{f"{comb.basis1.label}__label": lab2})


@pytest.mark.parametrize("bas", list_all_basis_classes())
def test_dynamic_set_label_mul(bas, basis_class_specific_params):
    if bas in (AdditiveBasis, MultiplicativeBasis, TransformerBasis):
        return

    combine_basis = CombinedBasis()
    bas_instance = combine_basis.instantiate_basis(
        5,
        bas,
        basis_class_specific_params,
        window_size=10,
    )
    if bas_instance.is_complex:
        pytest.skip("Cannot multiply more than one complex basis.")

    assert bas_instance.label == bas_instance.__class__.__name__
    mul_12 = bas_instance * bas_instance
    assert mul_12.basis1.label == bas_instance.__class__.__name__
    assert mul_12.basis2.label == (bas_instance.__class__.__name__ + "_1")

    # check labels
    mul_123 = mul_12 * bas_instance
    mix_123 = mul_12 + mul_12

    assert mul_123.basis1.basis1.label == bas_instance.__class__.__name__
    assert mul_123.basis1.basis2.label == (bas_instance.__class__.__name__ + "_1")
    assert mul_123.basis2.label == (bas_instance.__class__.__name__ + "_2")

    assert mix_123.basis1.basis1.label == bas_instance.__class__.__name__
    assert mix_123.basis1.basis2.label == (bas_instance.__class__.__name__ + "_1")
    assert mix_123.basis2.basis1.label == (bas_instance.__class__.__name__ + "_2")
    assert mix_123.basis2.basis2.label == (bas_instance.__class__.__name__ + "_3")

    assert (
        mul_123.basis1.label
        == "("
        + bas_instance.__class__.__name__
        + " * "
        + bas_instance.__class__.__name__
        + "_1)"
    )
    assert (
        mix_123.basis1.label
        == "("
        + bas_instance.__class__.__name__
        + " * "
        + bas_instance.__class__.__name__
        + "_1)"
    )

    # change label leaves
    mul_123.basis1.basis1.label = "x"
    mix_123.basis1.basis1.label = "x"

    assert mul_123.basis1.basis1.label == "x"
    assert mul_123.basis1.basis2.label == bas_instance.__class__.__name__
    assert mul_123.basis2.label == (bas_instance.__class__.__name__ + "_1")

    assert mix_123.basis1.basis1.label == "x"
    assert mix_123.basis1.basis2.label == bas_instance.__class__.__name__
    assert mix_123.basis2.basis1.label == (bas_instance.__class__.__name__ + "_1")
    assert mix_123.basis2.basis2.label == (bas_instance.__class__.__name__ + "_2")

    assert (
        mul_123.basis1.label
        == "(" + "x" + " * " + bas_instance.__class__.__name__ + ")"
    )
    assert (
        mix_123.basis1.label
        == "(" + "x" + " * " + bas_instance.__class__.__name__ + ")"
    )

    assert (
        mul_123.label
        == f"((x * {bas_instance.__class__.__name__}) * {bas_instance.__class__.__name__}_1)"
    )
    assert (
        mix_123.label
        == f"((x * {bas_instance.__class__.__name__}) + ({bas_instance.__class__.__name__}_1 * {bas_instance.__class__.__name__}_2))"
    )

    # change composite label
    mul_123.basis1.label = "y"
    mix_123.basis1.label = "y"

    assert mul_123.basis1.label == "y"
    assert mix_123.basis1.label == "y"

    assert mul_123.basis1.basis1.label == "x"
    assert mix_123.basis1.basis1.label == "x"

    assert mul_123.basis1.basis2.label == bas_instance.__class__.__name__
    assert mix_123.basis1.basis2.label == bas_instance.__class__.__name__

    assert mul_123.label == f"(y * {bas_instance.__class__.__name__}_1)"
    assert (
        mix_123.label
        == f"(y + ({bas_instance.__class__.__name__}_1 * {bas_instance.__class__.__name__}_2))"
    )

    assert mul_123.basis2.label == f"{bas_instance.__class__.__name__}_1"
    assert (
        mix_123.basis2.label
        == f"({bas_instance.__class__.__name__}_1 * {bas_instance.__class__.__name__}_2)"
    )


@pytest.mark.parametrize("bas", list_all_basis_classes())
def test_dynamic_set_label_add(bas, basis_class_specific_params):
    if bas in (AdditiveBasis, MultiplicativeBasis, TransformerBasis):
        return

    combine_basis = CombinedBasis()
    bas_instance = combine_basis.instantiate_basis(
        5,
        bas,
        basis_class_specific_params,
        window_size=10,
    )
    assert bas_instance.label == bas_instance.__class__.__name__
    add_12 = bas_instance + bas_instance
    assert add_12.basis1.label == bas_instance.__class__.__name__
    assert add_12.basis2.label == (bas_instance.__class__.__name__ + "_1")

    # check labels
    add_123 = add_12 + bas_instance
    assert add_123.basis1.basis1.label == bas_instance.__class__.__name__
    assert add_123.basis1.basis2.label == (bas_instance.__class__.__name__ + "_1")
    assert add_123.basis2.label == (bas_instance.__class__.__name__ + "_2")

    assert (
        add_123.basis1.label
        == "("
        + bas_instance.__class__.__name__
        + " + "
        + bas_instance.__class__.__name__
        + "_1)"
    )

    # change label leaves
    add_123.basis1.basis1.label = "x"
    assert add_123.basis1.basis1.label == "x"
    assert add_123.basis1.basis2.label == bas_instance.__class__.__name__
    assert add_123.basis2.label == (bas_instance.__class__.__name__ + "_1")

    assert (
        add_123.basis1.label
        == "(" + "x" + " + " + bas_instance.__class__.__name__ + ")"
    )

    assert (
        add_123.label
        == f"((x + {bas_instance.__class__.__name__}) + {bas_instance.__class__.__name__}_1)"
    )

    # change composite label
    add_123.basis1.label = "y"

    assert add_123.basis1.label == "y"

    assert add_123.basis1.basis1.label == "x"

    assert add_123.basis1.basis2.label == bas_instance.__class__.__name__

    assert add_123.label == f"(y + {bas_instance.__class__.__name__}_1)"

    assert add_123.basis2.label == f"{bas_instance.__class__.__name__}_1"


@pytest.mark.parametrize("bas", list_all_basis_classes())
def test_add_left_and_right(bas, basis_class_specific_params):
    if bas in (AdditiveBasis, MultiplicativeBasis, TransformerBasis):
        return

    combine_basis = CombinedBasis()
    bas_instance = combine_basis.instantiate_basis(
        5,
        bas,
        basis_class_specific_params,
        window_size=10,
    )
    bas_instance.label = "x"
    bas2_instance = bas_instance.__sklearn_clone__()
    bas3_instance = bas_instance.__sklearn_clone__()
    bas2_instance.label = "y"
    bas3_instance.label = "z"
    add_left = (bas_instance + bas2_instance) + bas3_instance
    assert add_left.label == "((x + y) + z)"
    add_right = bas_instance + (bas2_instance + bas3_instance)
    assert add_right.label == "(x + (y + z))"


@pytest.mark.parametrize("bas", list_all_basis_classes())
def test_multiply_left_and_right(bas, basis_class_specific_params):
    if issubclass(
        bas, (AdditiveBasis, MultiplicativeBasis, TransformerBasis, basis.FourierBasis)
    ):
        pytest.skip("skip multiplicaiton for complex and non-atomic bases.")

    combine_basis = CombinedBasis()
    bas_instance = combine_basis.instantiate_basis(
        5,
        bas,
        basis_class_specific_params,
        window_size=10,
    )
    bas_instance.label = "x"
    bas2_instance = bas_instance.__sklearn_clone__()
    bas3_instance = bas_instance.__sklearn_clone__()
    bas2_instance.label = "y"
    bas3_instance.label = "z"
    add_left = (bas_instance * bas2_instance) * bas3_instance
    assert add_left.label == "((x * y) * z)"
    add_right = bas_instance * (bas2_instance * bas3_instance)
    assert add_right.label == "(x * (y * z))"


@pytest.mark.parametrize("bas", list_all_basis_classes())
def test_basis_protected_name(bas, basis_class_specific_params):
    if bas in (AdditiveBasis, MultiplicativeBasis, TransformerBasis):
        return

    combine_basis = CombinedBasis()
    bas_instance = combine_basis.instantiate_basis(
        5,
        bas,
        basis_class_specific_params,
        window_size=10,
    )
    name = bas.__name__
    # does not raise because the basis name is the same as the current.
    with does_not_raise():
        bas_instance.label = name
    # same behavior at initialization
    with does_not_raise():
        combine_basis.instantiate_basis(
            5,
            bas,
            basis_class_specific_params,
            window_size=10,
            label=name,
        )

    name += "_1"
    with does_not_raise():
        bas_instance.label = name
        # assert that name is not updated / "_1" is stripped
        assert bas_instance.label == bas.__name__
    # same behavior at initialization
    with does_not_raise():
        bas_instance_init = combine_basis.instantiate_basis(
            5,
            bas,
            basis_class_specific_params,
            window_size=10,
            label=name,
        )
        # assert that name is the default / "_1" is stripped
        assert bas_instance_init.label == bas.__name__

    # if stripping the last number is not a basis name, no error
    name += "_1"
    with does_not_raise():
        bas_instance.label = name
        # assert that name is updated
        assert bas_instance.label == name
    # same behavior at initialization
    with does_not_raise():
        bas_instance_init = combine_basis.instantiate_basis(
            5,
            bas,
            basis_class_specific_params,
            window_size=10,
            label=name,
        )
        # assert that name is correct
        assert bas_instance_init.label == name

    # cannot assign the name of a different basis
    invalid_label = "MSplineEval" if bas.__name__ != "MSplineEval" else "MSplineConv"
    with pytest.raises(
        ValueError, match=f"Cannot assign '{invalid_label}' to a basis of class"
    ):
        bas_instance.label = invalid_label
    # same behavior at initialization
    with pytest.raises(
        ValueError, match=f"Cannot assign '{invalid_label}' to a basis of class"
    ):
        combine_basis.instantiate_basis(
            5,
            bas,
            basis_class_specific_params,
            window_size=10,
            label=invalid_label,
        )

    # cannot assign the name of a different basis with number
    invalid_label += "_1"
    with pytest.raises(
        ValueError, match=f"Cannot assign '{invalid_label}' to a basis of class"
    ):
        bas_instance.label = invalid_label
    # same behavior at initialization
    with pytest.raises(
        ValueError, match=f"Cannot assign '{invalid_label}' to a basis of class"
    ):
        combine_basis.instantiate_basis(
            5,
            bas,
            basis_class_specific_params,
            window_size=10,
            label=invalid_label,
        )

    # no longer a protected name
    invalid_label += "_1"
    with does_not_raise():
        bas_instance.label = invalid_label
        # assert that name is updated
        assert bas_instance.label == invalid_label
    # same behavior at initialization
    with does_not_raise():
        bas_instance_init = combine_basis.instantiate_basis(
            5,
            bas,
            basis_class_specific_params,
            window_size=10,
            label=invalid_label,
        )
        # assert that name is correct
        assert bas_instance_init.label == invalid_label


@pytest.mark.parametrize("bas1", list_all_basis_classes())
@pytest.mark.parametrize("bas2", list_all_basis_classes())
def test_getitem(bas1, bas2, basis_class_specific_params):
    if any(
        issubclass(
            bas,
            (AdditiveBasis, MultiplicativeBasis, TransformerBasis, basis.FourierBasis),
        )
        for bas in (bas1, bas2)
    ):
        pytest.skip("skip multiplicaiton for complex and non-atomic bases.")

    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        5,
        bas1,
        basis_class_specific_params,
        window_size=10,
    )
    bas2_instance = combine_basis.instantiate_basis(
        6,
        bas2,
        basis_class_specific_params,
        window_size=10,
    )

    add_12 = bas1_instance + bas2_instance
    mul_12 = bas1_instance * bas2_instance
    add_123 = add_12 + bas1_instance
    mul_123 = mul_12 * bas1_instance
    mix_123 = add_12 * bas1_instance

    name1, name2 = bas1.__name__, bas2.__name__
    if name1 == name2:
        name2 += "_1"
        name3 = name1 + "_2"
    else:
        name3 = name1 + "_1"

    list_all_label = add_123._generate_subtree_labels("all")
    assert tuple(list_all_label) == (
        f"(({name1} + {name2}) + {name3})",
        f"({name1} + {name2})",
        f"{name1}",
        f"{name2}",
        f"{name3}",
    )
    assert add_123[f"(({name1} + {name2}) + {name3})"] is add_123
    assert add_123[f"({name1} + {name2})"] is add_123.basis1
    assert add_123[f"{name1}"] is add_123.basis1.basis1
    assert add_123[f"{name2}"] is add_123.basis1.basis2
    assert add_123[f"{name3}"] is add_123.basis2

    list_all_label = mul_123._generate_subtree_labels("all")
    assert tuple(list_all_label) == (
        f"(({name1} * {name2}) * {name3})",
        f"({name1} * {name2})",
        f"{name1}",
        f"{name2}",
        f"{name3}",
    )
    assert mul_123[f"(({name1} * {name2}) * {name3})"] is mul_123
    assert mul_123[f"({name1} * {name2})"] is mul_123.basis1
    assert mul_123[f"{name1}"] is mul_123.basis1.basis1
    assert mul_123[f"{name2}"] is mul_123.basis1.basis2
    assert mul_123[f"{name3}"] is mul_123.basis2

    list_all_label = mix_123._generate_subtree_labels("all")
    assert tuple(list_all_label) == (
        f"(({name1} + {name2}) * {name3})",
        f"({name1} + {name2})",
        f"{name1}",
        f"{name2}",
        f"{name3}",
    )
    assert mix_123[f"(({name1} + {name2}) * {name3})"] is mix_123
    assert mix_123[f"({name1} + {name2})"] is mix_123.basis1
    assert mix_123[f"{name1}"] is mix_123.basis1.basis1
    assert mix_123[f"{name2}"] is mix_123.basis1.basis2
    assert mix_123[f"{name3}"] is mix_123.basis2

    add_123.basis1.basis1.label = "x"
    add_123.basis1.basis2.label = "y"
    add_123.basis2.label = "z"
    mul_123.basis1.basis1.label = "x"
    mul_123.basis1.basis2.label = "y"
    mul_123.basis2.label = "z"
    mix_123.basis1.basis1.label = "x"
    mix_123.basis1.basis2.label = "y"
    mix_123.basis2.label = "z"

    list_all_label = add_123._generate_subtree_labels()
    assert tuple(list_all_label) == ("((x + y) + z)", "(x + y)", "x", "y", "z")
    assert add_123["((x + y) + z)"] is add_123
    assert add_123["(x + y)"] is add_123.basis1
    assert add_123["x"] is add_123.basis1.basis1
    assert add_123["y"] is add_123.basis1.basis2
    assert add_123["z"] is add_123.basis2

    list_all_label = mul_123._generate_subtree_labels("all")
    assert tuple(list_all_label) == ("((x * y) * z)", "(x * y)", "x", "y", "z")
    assert mul_123["((x * y) * z)"] is mul_123
    assert mul_123["(x * y)"] is mul_123.basis1
    assert mul_123["x"] is mul_123.basis1.basis1
    assert mul_123["y"] is mul_123.basis1.basis2
    assert mul_123["z"] is mul_123.basis2

    list_all_label = mix_123._generate_subtree_labels("all")
    assert tuple(list_all_label) == ("((x + y) * z)", "(x + y)", "x", "y", "z")
    assert mix_123["((x + y) * z)"] is mix_123
    assert mix_123["(x + y)"] is mix_123.basis1
    assert mix_123["x"] is mix_123.basis1.basis1
    assert mix_123["y"] is mix_123.basis1.basis2
    assert mix_123["z"] is mix_123.basis2

    with pytest.raises(IndexError, match=f"Basis label BSplineEval not found"):
        add_123["BSplineEval"]
    with pytest.raises(IndexError, match=f"Basis label BSplineEval not found"):
        mul_123["BSplineEval"]
    with pytest.raises(IndexError, match=f"Basis label BSplineEval not found"):
        mix_123["BSplineEval"]


@pytest.mark.parametrize(
    "bas1, bas2",
    list(itertools.product(*[list_all_basis_classes()] * 2)),
)
@pytest.mark.parametrize(
    "x, axis, expectation, exp_shapes",  # num output is 5*2 + 6*3 = 28
    [
        (np.ones((1, 28)), 1, does_not_raise(), [(1, 2, 5), (1, 3, 6)]),
        (np.ones((1, 28)), -1, does_not_raise(), [(1, 2, 5), (1, 3, 6)]),
        (np.ones((1, 28, 2)), -2, does_not_raise(), [(1, 2, 5, 2), (1, 3, 6, 2)]),
        (np.ones((28,)), 0, does_not_raise(), [(2, 5), (3, 6)]),
        (np.ones((2, 2, 28)), 2, does_not_raise(), [(2, 2, 2, 5), (2, 2, 3, 6)]),
        (
            np.ones((2, 2, 27)),
            2,
            pytest.raises(
                ValueError, match=r"`x.shape\[axis\]` does not match the expected"
            ),
            [(2, 2, 2, 5), (2, 2, 3, 6)],
        ),
    ],
)
def test_split_feature_axis(
    bas1, bas2, x, axis, expectation, exp_shapes, basis_class_specific_params
):
    # skip nested
    if any(
        bas
        in (
            AdditiveBasis,
            MultiplicativeBasis,
            TransformerBasis,
            IdentityEval,
            HistoryConv,
        )
        for bas in [bas1, bas2]
    ):
        return
    # define the basis
    n_basis = [5, 6]
    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        n_basis[0], bas1, basis_class_specific_params, window_size=10, label="1"
    )
    bas2_instance = combine_basis.instantiate_basis(
        n_basis[1], bas2, basis_class_specific_params, window_size=10, label="2"
    )

    bas = bas1_instance + bas2_instance
    bas.set_input_shape(np.zeros((1, 2)), np.zeros((1, 3)))
    with expectation:
        out = bas.split_by_feature(x, axis=axis)
        for i, itm in enumerate(out.items()):
            _, val = itm
            assert val.shape == exp_shapes[i]


def test_composite_basis_repr_wrapping():
    with patch("os.get_terminal_size", return_value=SizeTerminal(80, 24)):
        # check multi
        bas = basis.BSplineEval(10) ** 100
        out = repr(bas)
        assert out.startswith(
            "MultiplicativeBasis(\n    basis1=MultiplicativeBasis(\n        basis1=MultiplicativeBasis(\n "
        )
        assert out.endswith(
            "'BSplineEval_98': BSplineEval(n_basis_funcs=10, order=4),\n    ),\n    basis2='BSplineEval_99': BSplineEval(n_basis_funcs=10, order=4),\n)"
        )
        assert "    ...\n" in out

        bas = basis.MSplineEval(10, label="0")
        for k in range(1, 100):
            bas = bas + basis.MSplineEval(10, label=str(k))

        # large additive basis
        out = repr(bas)
        assert out.startswith(
            "AdditiveBasis(\n    basis1=AdditiveBasis(\n        basis1=AdditiveBasis(\n "
        )
        assert out.endswith(
            "        basis2='98': MSplineEval(n_basis_funcs=10, order=4),\n    ),\n    basis2='99': MSplineEval(n_basis_funcs=10, order=4),\n)"
        )
        assert "    ...\n" in out

        bas = basis.MSplineEval(10) * 100
        out = repr(bas)
        assert out.startswith(
            "AdditiveBasis(\n    basis1=AdditiveBasis(\n        basis1=AdditiveBasis(\n "
        )
        assert out.endswith(
            "'MSplineEval_98': MSplineEval(n_basis_funcs=10, order=4),\n    ),\n    basis2='MSplineEval_99': MSplineEval(n_basis_funcs=10, order=4),\n)"
        )
        assert "    ...\n" in out


def test_all_public_importable_bases_equal():
    import nemos.basis

    # this is the list of publicly available bases
    public_bases = set(dir(nemos.basis))
    # these are all the bases that are imported in the init file
    # Get all classes that are explicitly defined or imported into nemos.basis
    imported_bases = {
        name
        for name, obj in inspect.getmembers(nemos.basis, inspect.isclass)
        if issubclass(obj, nemos.basis._basis.Basis)
    }

    if public_bases.difference(imported_bases) != {"CustomBasis"}:
        raise ValueError(
            "nemos/basis/__init__.py imported basis objects does not match"
            " nemos/basis/_composition_utils.py's __PUBLIC_BASES__ list:\n"
            f"imported but not public: {imported_bases - public_bases}\n",
            f"public but not imported: {public_bases - imported_bases}",
        )
