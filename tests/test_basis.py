import abc
import inspect
import itertools
import pickle
import re
from contextlib import nullcontext as does_not_raise
from functools import partial
from typing import Literal

import jax.numpy
import numpy as np
import pynapple as nap
import pytest
import utils_testing
from sklearn.base import clone as sk_clone

import nemos.basis.basis as basis
import nemos.convolve as convolve
from nemos.basis._basis import AdditiveBasis, Basis, MultiplicativeBasis, add_docstring
from nemos.basis._decaying_exponential import OrthExponentialBasis
from nemos.basis._raised_cosine_basis import (
    RaisedCosineBasisLinear,
    RaisedCosineBasisLog,
)
from nemos.basis._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from nemos.utils import pynapple_concatenate_numpy


# automatic define user accessible basis and check the methods
def list_all_basis_classes() -> list[type]:
    """
    Return all the classes in nemos.basis which are a subclass of Basis,
    which should be all concrete classes except TransformerBasis.
    """
    return [
        class_obj
        for _, class_obj in utils_testing.get_non_abstract_classes(basis)
        if issubclass(class_obj, Basis)
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
    tested_bases = {test_cls.cls for test_cls in subclasses}

    # Create the set of all the concrete basis classes
    all_bases = set(list_all_basis_classes())

    if all_bases != all_bases.intersection(tested_bases):
        raise ValueError(
            "Test should be implemented for each of the concrete classes in the basis module.\n"
            f"The following classes are not tested: {[bas.__qualname__ for bas in all_bases.difference(tested_bases)]}"
        )


@pytest.mark.parametrize(
    "basis_instance",
    [
        basis.EvalBSpline(10),
        basis.ConvBSpline(10, window_size=11),
        basis.EvalCyclicBSpline(10),
        basis.ConvCyclicBSpline(10, window_size=11),
        basis.EvalMSpline(10),
        basis.ConvMSpline(10, window_size=11),
        basis.EvalRaisedCosineLinear(10),
        basis.ConvRaisedCosineLinear(10, window_size=11),
        basis.EvalRaisedCosineLog(10),
        basis.ConvRaisedCosineLog(10, window_size=11),
        basis.EvalOrthExponential(10, np.arange(1, 11)),
        basis.ConvOrthExponential(10, decay_rates=np.arange(1, 11), window_size=12),
    ],
)
@pytest.mark.parametrize(
    "method_name, descr_match",
    [
        ("evaluate_on_grid", "The number of points in the uniformly spaced grid"),
        (
            "compute_features",
            "Compute the basis functions and transform input data into model features",
        ),
        (
            "split_by_feature",
            "Decompose an array along a specified axis into sub-arrays",
        ),
    ],
)
def test_example_docstrings_add(basis_instance, method_name, descr_match):
    method = getattr(basis_instance, method_name)
    doc = method.__doc__
    examp_delim = "\n        Examples\n        --------"

    assert examp_delim in doc
    doc_components = doc.split(examp_delim)
    assert len(doc_components) == 2
    assert len(doc_components[0].strip()) > 0
    assert re.search(descr_match, doc_components[0])

    # check that the basis name is in the example
    assert basis_instance.__class__.__name__ in doc_components[1]

    # check that no other basis name is in the example
    for basis_name in basis.__dir__():
        if basis_name == basis_instance.__class__.__name__:
            continue
        assert basis_name not in doc_components[1]


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


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.EvalBSpline(10), BSplineBasis),
        (basis.ConvBSpline(10, window_size=11), BSplineBasis),
        (basis.EvalCyclicBSpline(10), CyclicBSplineBasis),
        (basis.ConvCyclicBSpline(10, window_size=11), CyclicBSplineBasis),
        (basis.EvalMSpline(10), MSplineBasis),
        (basis.ConvMSpline(10, window_size=11), MSplineBasis),
        (basis.EvalRaisedCosineLinear(10), RaisedCosineBasisLinear),
        (basis.ConvRaisedCosineLinear(10, window_size=11), RaisedCosineBasisLinear),
        (basis.EvalRaisedCosineLog(10), RaisedCosineBasisLog),
        (basis.ConvRaisedCosineLog(10, window_size=11), RaisedCosineBasisLog),
        (basis.EvalOrthExponential(10, np.arange(1, 11)), OrthExponentialBasis),
        (
            basis.ConvOrthExponential(10, decay_rates=np.arange(1, 11), window_size=12),
            OrthExponentialBasis,
        ),
    ],
)
def test_expected_output_eval_on_grid(basis_instance, super_class):
    x, y = super_class.evaluate_on_grid(basis_instance, 100)
    xx, yy = basis_instance.evaluate_on_grid(100)
    np.testing.assert_equal(xx, x)
    np.testing.assert_equal(yy, y)


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.EvalBSpline(10), BSplineBasis),
        (basis.ConvBSpline(10, window_size=11), BSplineBasis),
        (basis.EvalCyclicBSpline(10), CyclicBSplineBasis),
        (basis.ConvCyclicBSpline(10, window_size=11), CyclicBSplineBasis),
        (basis.EvalMSpline(10), MSplineBasis),
        (basis.ConvMSpline(10, window_size=11), MSplineBasis),
        (basis.EvalRaisedCosineLinear(10), RaisedCosineBasisLinear),
        (basis.ConvRaisedCosineLinear(10, window_size=11), RaisedCosineBasisLinear),
        (basis.EvalRaisedCosineLog(10), RaisedCosineBasisLog),
        (basis.ConvRaisedCosineLog(10, window_size=11), RaisedCosineBasisLog),
        (basis.EvalOrthExponential(10, np.arange(1, 11)), OrthExponentialBasis),
        (
            basis.ConvOrthExponential(10, decay_rates=np.arange(1, 11), window_size=12),
            OrthExponentialBasis,
        ),
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
        (basis.EvalBSpline(10, label="label"), BSplineBasis),
        (basis.ConvBSpline(10, window_size=11, label="label"), BSplineBasis),
        (basis.EvalCyclicBSpline(10, label="label"), CyclicBSplineBasis),
        (
            basis.ConvCyclicBSpline(10, window_size=11, label="label"),
            CyclicBSplineBasis,
        ),
        (basis.EvalMSpline(10, label="label"), MSplineBasis),
        (basis.ConvMSpline(10, window_size=11, label="label"), MSplineBasis),
        (basis.EvalRaisedCosineLinear(10, label="label"), RaisedCosineBasisLinear),
        (
            basis.ConvRaisedCosineLinear(10, window_size=11, label="label"),
            RaisedCosineBasisLinear,
        ),
        (basis.EvalRaisedCosineLog(10, label="label"), RaisedCosineBasisLog),
        (
            basis.ConvRaisedCosineLog(10, window_size=11, label="label"),
            RaisedCosineBasisLog,
        ),
        (
            basis.EvalOrthExponential(10, np.arange(1, 11), label="label"),
            OrthExponentialBasis,
        ),
        (
            basis.ConvOrthExponential(
                10, decay_rates=np.arange(1, 11), window_size=12, label="label"
            ),
            OrthExponentialBasis,
        ),
    ],
)
def test_expected_output_split_by_feature(basis_instance, super_class):
    x = super_class.compute_features(basis_instance, np.linspace(0, 1, 100))
    xdict = super_class.split_by_feature(basis_instance, x)
    xxdict = basis_instance.split_by_feature(x)
    assert xdict.keys() == xxdict.keys()
    xx = xxdict["label"]
    x = xdict["label"]
    nans = np.isnan(x.sum(axis=(1, 2)))
    assert np.all(np.isnan(xx[nans]))
    np.testing.assert_array_equal(xx[~nans], x[~nans])


class BasisFuncsTesting(abc.ABC):
    """
    An abstract base class that sets the foundation for individual basis function testing.
    This class requires an implementation of a 'cls' method, which is utilized by the meta-test
    that verifies if all basis functions are properly tested.
    """

    @abc.abstractmethod
    def cls(self):
        pass



class TestRaisedCosineLogBasis(BasisFuncsTesting):
    cls = {"eval": basis.EvalRaisedCosineLog, "conv": basis.ConvRaisedCosineLog}

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    @pytest.mark.parametrize("mode, kwargs", [("eval", {}), ("conv", {"window_size": 2})])
    def test_non_empty_samples(self, samples, mode, kwargs):
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls[mode](5, **kwargs).compute_features(
                    samples
                )
        else:
            self.cls[mode](5, **kwargs).compute_features(samples)

    @pytest.mark.parametrize("eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])])
    def test_compute_features_input(self, eval_input):
        basis_obj = self.cls["eval"](n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (10, does_not_raise()),
            (10.5, does_not_raise()),
            (
                    0.5,
                    pytest.raises(
                        ValueError,
                        match=r"Invalid raised cosine width\. 2\*width must be a positive",
                    ),
            ),
            (
                    10.3,
                    pytest.raises(
                        ValueError,
                        match=r"Invalid raised cosine width\. 2\*width must be a positive",
                    ),
            ),
            (
                    -10,
                    pytest.raises(
                        ValueError,
                        match=r"Invalid raised cosine width\. 2\*width must be a positive",
                    ),
            ),
            (None, pytest.raises(TypeError, match="'<=' not supported between")),
        ],
    )
    @pytest.mark.parametrize("cls, kwargs", [
        (basis.EvalRaisedCosineLog, {}),
        (basis.ConvRaisedCosineLog, {"window_size": 5}),
    ])
    def test_set_width(self, width, expectation, cls, kwargs):
        basis_obj = cls(n_basis_funcs=5, **kwargs)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)

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
    def test_compute_features_axis(self, kwargs, input1_shape, expectation):
        with expectation:
            basis_obj = self.cls["conv"](n_basis_funcs=5, window_size=5, conv_kwargs=kwargs)
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [4, 5])
    @pytest.mark.parametrize("time_scaling", [50, 70])
    @pytest.mark.parametrize("enforce_decay", [True, False])
    @pytest.mark.parametrize("window_size", [10, 15])
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
    ):
        x = np.ones(input_shape)
        bas = self.cls(
            n_basis_funcs=n_basis_funcs,
            time_scaling=time_scaling,
            mode="conv",
            window_size=window_size,
            enforce_decay_to_zero=enforce_decay,
        )
        out = bas.compute_features(x)
        assert out.shape[1] == expected_n_input * bas.n_basis_funcs

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [2, 10, 100]],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_compute_features_returns_expected_number_of_basis(
        self, args, mode, window_size, sample_size
    ):
        """
        Verifies the number of basis functions returned by the evaluate() method matches
        the expected number of basis functions.
        """
        basis_obj = self.cls(mode=mode, window_size=window_size, **args)
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != args["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {args['n_basis_funcs']}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 100])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, mode, window_size
    ):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

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
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, window_size
    ):
        """
        Verifies that the minimum number of basis functions required (i.e., 2) is enforced.
        """
        raise_exception = n_basis_funcs < 2
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls.__name__} "
                "requires >= 2 basis elements.",
            ):
                self.cls(
                    n_basis_funcs=n_basis_funcs, mode=mode, window_size=window_size
                )
        else:
            self.cls(n_basis_funcs=n_basis_funcs, mode=mode, window_size=window_size)

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, window_size
    ):
        """
        Confirms that the compute_features() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, mode=mode, window_size=window_size)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="Input dimensionality mismatch",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """
        Checks that the evaluate_on_grid() method returns a grid of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        """
        Ensures that the evaluate_on_grid() method returns basis functions of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        """
        Validates that the evaluate_on_grid() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize(
        "width ,expectation",
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
    def test_width_values(self, width, expectation):
        """Test allowable widths: integer multiple of 1/2, greater than 1."""
        with expectation:
            self.cls(n_basis_funcs=5, width=width)

    @pytest.mark.parametrize("width", [1.5, 2, 2.5])
    def test_decay_to_zero_basis_number_match(self, width):
        """Test that the number of basis is preserved."""
        n_basis_funcs = 10
        _, ev = self.cls(
            n_basis_funcs=n_basis_funcs, width=width, enforce_decay_to_zero=True
        ).evaluate_on_grid(2)
        assert ev.shape[1] == n_basis_funcs, (
            "Basis function number mismatch. "
            f"Expected {n_basis_funcs}, got {ev.shape[1]} instead!"
        )

    @pytest.mark.parametrize(
        "time_scaling ,expectation",
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
    def test_time_scaling_values(self, time_scaling, expectation):
        """Test that only positive time_scaling are allowed."""
        with expectation:
            self.cls(n_basis_funcs=5, time_scaling=time_scaling)

    def test_time_scaling_property(self):
        """Test that larger time_scaling results in larger departures from linearity."""
        time_scaling = [0.1, 10, 100]
        n_basis_funcs = 5
        _, lin_ev = basis.RaisedCosineBasisLinear(n_basis_funcs).evaluate_on_grid(100)
        corr = np.zeros(len(time_scaling))
        for idx, ts in enumerate(time_scaling):
            # set default decay to zero to get comparable basis
            basis_log = self.cls(
                n_basis_funcs=n_basis_funcs,
                time_scaling=ts,
                enforce_decay_to_zero=False,
            )
            _, log_ev = basis_log.evaluate_on_grid(100)
            # compute the correlation
            corr[idx] = (lin_ev.flatten() @ log_ev.flatten()) / (
                np.linalg.norm(lin_ev.flatten()) * np.linalg.norm(log_ev.flatten())
            )
        # check that the correlation decreases as time_scale increases
        assert np.all(
            np.diff(corr) < 0
        ), "As time scales increases, deviation from linearity should increase!"

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = self.cls(n_basis).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(self, num_input, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_shape(self, inp, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(inp)

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_axis(self, time_axis_shape, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        assert bas(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_nan(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas(x)[3]))

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
    def test_call_input_type(self, samples, expectation):
        bas = self.cls(5)
        with expectation:
            bas(samples)

    def test_call_equivalent_in_conv(self):
        bas_con = self.cls(5, mode="conv", window_size=10)
        bas_eva = self.cls(5, mode="eval")
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con(x) == bas_eva(x))

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_pynapple_support(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas(x)
        y_nap = bas(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_basis_number(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        assert bas(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_non_empty(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(np.array([]))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_range(self, mn, mx, expectation, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(np.linspace(mn, mx, 10))

    def test_fit_kernel(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_.shape == (3, 5)

    def test_transform_fails(self):
        bas = self.cls(5, mode="conv", window_size=3)
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="`mode` should be either 'conv' or 'eval'"
                ),
            ),
        ],
    )
    def test_init_mode(self, mode, expectation):
        window_size = None if mode == "eval" else 2
        with expectation:
            self.cls(5, mode=mode, window_size=window_size)

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label):
        bas = self.cls(5, label=label)
        assert bas.label == (str(label) if label is not None else self.cls.__name__)

    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("label", None),
            ("label", "label"),
            ("n_basis_input", 1),
            ("n_output_features", 5),
        ],
    )
    def test_attr_setter(self, attribute, value):
        bas = self.cls(5)
        with pytest.raises(
            AttributeError, match=rf"can't set attribute|property '{attribute}' of"
        ):
            setattr(bas, attribute, value)

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_basis_input is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_basis_input == (n_input,)
        assert bas._n_basis_input == (n_input,)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (3, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas = self.cls(5, mode="conv", window_size=10)
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
    def test_init_conv_kwargs(self, conv_kwargs, expectation):
        with expectation:
            self.cls(5, mode="conv", window_size=200, **conv_kwargs)

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 2, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "conv",
                None,
                pytest.raises(
                    ValueError,
                    match="If the basis is in `conv` mode, you must provide a ",
                ),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            ("eval", None, does_not_raise()),
            (
                "eval",
                10,
                pytest.raises(
                    ValueError,
                    match=r"If basis is in `mode=='eval'`, `window_size` should be None",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation):
        with expectation:
            self.cls(5, mode=mode, window_size=ws)

    @pytest.mark.parametrize(
        "enforce_decay_to_zero, time_scaling, width, window_size, n_basis_funcs, bounds, mode",
        [
            (False, 15, 4, None, 10, (1, 2), "eval"),
            (False, 15, 4, 10, 10, None, "conv"),
        ],
    )
    def test_set_params(
        self,
        enforce_decay_to_zero,
        time_scaling,
        width,
        window_size,
        n_basis_funcs,
        bounds,
        mode: Literal["eval", "conv"],
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            enforce_decay_to_zero=enforce_decay_to_zero,
            time_scaling=time_scaling,
            width=width,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
        )
        keys = list(pars.keys())
        bas = self.cls(
            enforce_decay_to_zero=enforce_decay_to_zero,
            time_scaling=time_scaling,
            width=width,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            mode=mode,
        )
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas = bas.set_params(**par_set)
                assert isinstance(bas, self.cls)

        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                with pytest.raises(
                    AttributeError,
                    match="can't set attribute 'mode'|property 'mode' of ",
                ):
                    par_set = {
                        keys[i]: pars[keys[i]],
                        keys[j]: pars[keys[j]],
                        "mode": mode,
                    }
                    bas.set_params(**par_set)

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", pytest.raises(ValueError, match="`bounds` should only be set")),
        ],
    )
    def test_set_bounds(self, mode, expectation):
        ws = dict(eval=None, conv=10)
        with expectation:
            self.cls(window_size=ws[mode], n_basis_funcs=10, mode=mode, bounds=(1, 2))

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv", bounds=None)
        with pytest.raises(ValueError, match="`bounds` should only be set"):
            bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            ("eval", pytest.raises(ValueError, match="If basis is in `mode=='eval'`")),
        ],
    )
    def test_set_window_size(self, mode, expectation):
        """Test window size set behavior."""
        with expectation:
            self.cls(window_size=10, n_basis_funcs=10, mode=mode)

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv")
        with pytest.raises(ValueError, match="If the basis is in `conv` mode"):
            bas.set_params(window_size=None)

        bas = self.cls(window_size=None, n_basis_funcs=10, mode="eval")
        with pytest.raises(ValueError, match="If basis is in `mode=='eval'`"):
            bas.set_params(window_size=10)

    def test_convolution_is_performed(self):
        bas = self.cls(5, mode="conv", window_size=10)
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_conv_kwargs_error(self):
        with pytest.raises(ValueError, match="kwargs should only be set"):
            self.cls(5, mode="eval", test="hi")

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation):
        with expectation:
            bas = self.cls(3, bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation):
        bas = self.cls(3, bounds=(1, 3))
        with expectation:
            bas.set_params(bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx):
        bounds = None if vmin is None else (vmin, vmax)
        bas = self.cls(3, mode="eval", bounds=bounds)
        out = bas.compute_features(samples)
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_eval_on_grid_no_effect_on_eval(
        self, vmin, vmax, samples, nan_idx
    ):
        bas_no_range = self.cls(3, mode="eval", bounds=None)
        bas = self.cls(3, mode="eval", bounds=(vmin, vmax))
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1, out2)

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(self, bounds, samples, nan_idx, mn, mx):
        bas_no_range = self.cls(3, mode="eval", bounds=None)
        bas = self.cls(3, mode="eval", bounds=bounds)
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (None, np.arange(5), does_not_raise()),
            ((0, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 4), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception):
        with exception:
            self.cls(3, mode="conv", window_size=10, bounds=bounds)

    def test_transformer_get_params(self):
        bas = self.cls(5)
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        params_basis = bas.get_params()
        assert params_transf == params_basis


class TestRaisedCosineLinearBasis(BasisFuncsTesting):
    cls = basis.RaisedCosineBasisLinear

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_non_empty_samples(self, samples, mode, window_size):
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5, mode=mode, window_size=window_size).compute_features(
                    samples
                )
        else:
            self.cls(5, mode=mode, window_size=window_size).compute_features(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (10, does_not_raise()),
            (10.5, does_not_raise()),
            (
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (
                10.3,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (
                -10,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (None, pytest.raises(TypeError, match="'<=' not supported between")),
        ],
    )
    def test_set_width(self, width, expectation):
        basis_obj = self.cls(n_basis_funcs=5)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)

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
    def test_compute_features_axis(self, kwargs, input1_shape, expectation):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        with expectation:
            basis_obj = self.cls(n_basis_funcs=5, mode="conv", window_size=5, **kwargs)
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [4, 5])
    @pytest.mark.parametrize("window_size", [10, 15])
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
        self, n_basis_funcs, window_size, input_shape, expected_n_input
    ):
        x = np.ones(input_shape)
        bas = self.cls(
            n_basis_funcs=n_basis_funcs,
            mode="conv",
            window_size=window_size,
        )
        out = bas.compute_features(x)
        assert out.shape[1] == expected_n_input * bas.n_basis_funcs

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [2, 10, 100]],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_compute_features_returns_expected_number_of_basis(
        self, args, mode, window_size, sample_size
    ):
        """
        Verifies that the compute_features() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(mode=mode, window_size=window_size, **args)
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != args["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {args['n_basis_funcs']}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 100])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, mode, window_size
    ):
        """
        Checks that the sample size of the output from the co ute_features() method matches the input sample size.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features basis is {eval_basis.shape[0]}",
            )

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
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, window_size
    ):
        """
        Verifies that the minimum number of basis functions required (i.e., 1) is enforced.
        """
        raise_exception = n_basis_funcs < 2
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls.__name__} "
                r"requires >= 2 basis elements\.",
            ):
                self.cls(
                    n_basis_funcs=n_basis_funcs, mode=mode, window_size=window_size
                )
        else:
            self.cls(n_basis_funcs=n_basis_funcs, mode=mode, window_size=window_size)

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, window_size
    ):
        """
        Confirms that the compute_features() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, mode=mode, window_size=window_size)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="Input dimensionality mismatch",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """
        Checks that the evaluate_on_grid() method returns a grid of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        """
        Ensures that the evaluate_on_grid() method returns basis functions of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        """
        Validates that the evaluate_on_grid() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize(
        "width ,expectation",
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
    def test_width_values(self, width, expectation):
        """Test allowable widths: integer multiple of 1/2, greater than 1."""
        with expectation:
            self.cls(n_basis_funcs=5, width=width)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = self.cls(n_basis).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(self, num_input, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_shape(self, inp, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(inp)

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_axis(self, time_axis_shape, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        assert bas(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_nan(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas(x)[3]))

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
    def test_call_input_type(self, samples, expectation):
        bas = self.cls(5)
        with expectation:
            bas(samples)

    def test_call_equivalent_in_conv(self):
        bas_con = self.cls(5, mode="conv", window_size=10)
        bas_eva = self.cls(5, mode="eval")
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con(x) == bas_eva(x))

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
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_pynapple_support(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas(x)
        y_nap = bas(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_basis_number(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        assert bas(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_non_empty(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(np.array([]))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_range(self, mn, mx, expectation, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(np.linspace(mn, mx, 10))

    def test_fit_kernel(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_.shape == (3, 5)

    def test_transform_fails(self):
        bas = self.cls(5, mode="conv", window_size=3)
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="`mode` should be either 'conv' or 'eval'"
                ),
            ),
        ],
    )
    def test_init_mode(self, mode, expectation):
        window_size = None if mode == "eval" else 2
        with expectation:
            self.cls(5, mode=mode, window_size=window_size)

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label):
        bas = self.cls(5, label=label)
        assert bas.label == (str(label) if label is not None else self.cls.__name__)

    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("label", None),
            ("label", "label"),
            ("n_basis_input", 1),
            ("n_output_features", 5),
        ],
    )
    def test_attr_setter(self, attribute, value):
        bas = self.cls(5)
        with pytest.raises(
            AttributeError, match=rf"can't set attribute|property '{attribute}' of"
        ):
            setattr(bas, attribute, value)

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_basis_input is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_basis_input == (n_input,)
        assert bas._n_basis_input == (n_input,)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (3, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas = self.cls(5, mode="conv", window_size=10)
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
    def test_init_conv_kwargs(self, conv_kwargs, expectation):
        with expectation:
            self.cls(5, mode="conv", window_size=200, **conv_kwargs)

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 2, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            ("eval", None, does_not_raise()),
            (
                "eval",
                10,
                pytest.raises(
                    ValueError,
                    match=r"If basis is in `mode=='eval'`, `window_size` should be None",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation):
        with expectation:
            self.cls(5, mode=mode, window_size=ws)

    @pytest.mark.parametrize(
        "width, window_size, n_basis_funcs, bounds, mode",
        [
            (4, None, 10, (1, 2), "eval"),
            (4, 10, 10, None, "conv"),
        ],
    )
    def test_set_params(
        self, width, window_size, n_basis_funcs, bounds, mode: Literal["eval", "conv"]
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            width=width,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
        )
        keys = list(pars.keys())
        bas = self.cls(
            width=width, window_size=window_size, n_basis_funcs=n_basis_funcs, mode=mode
        )
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas.set_params(**par_set)
                assert isinstance(bas, self.cls)

        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                with pytest.raises(
                    AttributeError,
                    match="can't set attribute 'mode'|property 'mode' of ",
                ):
                    par_set = {
                        keys[i]: pars[keys[i]],
                        keys[j]: pars[keys[j]],
                        "mode": mode,
                    }
                    bas.set_params(**par_set)

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", pytest.raises(ValueError, match="`bounds` should only be set")),
        ],
    )
    def test_set_bounds(self, mode, expectation):
        ws = dict(eval=None, conv=10)
        with expectation:
            self.cls(window_size=ws[mode], n_basis_funcs=10, mode=mode, bounds=(1, 2))

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv", bounds=None)
        with pytest.raises(ValueError, match="`bounds` should only be set"):
            bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            ("eval", pytest.raises(ValueError, match="If basis is in `mode=='eval'`")),
        ],
    )
    def test_set_window_size(self, mode, expectation):
        """Test window size set behavior."""
        with expectation:
            self.cls(window_size=10, n_basis_funcs=10, mode=mode)

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv")
        with pytest.raises(ValueError, match="If the basis is in `conv` mode"):
            bas.set_params(window_size=None)

        bas = self.cls(window_size=None, n_basis_funcs=10, mode="eval")
        with pytest.raises(ValueError, match="If basis is in `mode=='eval'`"):
            bas.set_params(window_size=10)

    def test_convolution_is_performed(self):
        bas = self.cls(5, mode="conv", window_size=10)
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_conv_kwargs_error(self):
        with pytest.raises(ValueError, match="kwargs should only be set"):
            self.cls(5, mode="eval", test="hi")

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation):
        with expectation:
            bas = self.cls(3, bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation):
        bas = self.cls(5, bounds=(1, 3))
        with expectation:
            bas.set_params(bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx):
        bounds = None if vmin is None else (vmin, vmax)
        bas = self.cls(3, mode="eval", bounds=bounds)
        out = bas.compute_features(samples)
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_eval_on_grid_no_effect_on_eval(
        self, vmin, vmax, samples, nan_idx
    ):
        bas_no_range = self.cls(3, mode="eval", bounds=None)
        bas = self.cls(3, mode="eval", bounds=(vmin, vmax))
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1, out2)

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(self, bounds, samples, nan_idx, mn, mx):
        bas_no_range = self.cls(3, mode="eval", bounds=None)
        bas = self.cls(3, mode="eval", bounds=bounds)
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (None, np.arange(5), does_not_raise()),
            ((0, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 4), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception):
        with exception:
            self.cls(3, mode="conv", window_size=10, bounds=bounds)

    def test_transformer_get_params(self):
        bas = self.cls(5)
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        params_basis = bas.get_params()
        assert params_transf == params_basis


class TestMSplineBasis(BasisFuncsTesting):
    cls = basis.EvalMSpline

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_non_empty_samples(self, samples, mode, window_size):
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5, mode=mode, window_size=window_size).compute_features(
                    samples
                )
        else:
            self.cls(5, mode=mode, window_size=window_size).compute_features(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

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
    def test_compute_features_axis(self, kwargs, input1_shape, expectation):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        with expectation:
            basis_obj = self.cls(n_basis_funcs=5, mode="conv", window_size=5, **kwargs)
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [2, 3])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("window_size", [10, 15])
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
        order,
        window_size,
        input_shape,
        expected_n_input,
    ):
        x = np.ones(input_shape)
        bas = self.cls(
            n_basis_funcs=n_basis_funcs,
            order=order,
            mode="conv",
            window_size=window_size,
        )
        out = bas.compute_features(x)
        assert out.shape[1] == expected_n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_basis_funcs", [6, 8, 10])
    @pytest.mark.parametrize("order", range(1, 6))
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_compute_features_returns_expected_number_of_basis(
        self, n_basis_funcs: int, order: int, mode, window_size
    ):
        """
        Verifies that the compute_features() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, order=order, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, 100))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [4, 10, 100])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, order, mode, window_size
    ):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, order=order, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features is {eval_basis.shape[0]}",
            )

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
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, window_size
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = (order < 1) | (n_basis_funcs < 1) | (order > n_basis_funcs)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Spline order must be positive!|"
                rf"{self.cls.__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls(
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    mode=mode,
                    window_size=window_size,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(
                n_basis_funcs=n_basis_funcs,
                order=order,
                mode=mode,
                window_size=window_size,
            )
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
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, window_size
    ):
        """
        Confirms that the compute_features() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(
            n_basis_funcs=5, order=3, mode=mode, window_size=window_size
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="Input dimensionality mismatch",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """
        Checks that the evaluate_on_grid() method returns a grid of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        """
        Ensures that the evaluate_on_grid() method returns basis functions of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        """
        Validates that the evaluate_on_grid() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = self.cls(n_basis).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(self, num_input, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_shape(self, inp, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(inp)

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_axis(self, time_axis_shape, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        assert bas(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_nan(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas(x)[3]))

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
    def test_call_input_type(self, samples, expectation):
        bas = self.cls(5)
        with expectation:
            bas(samples)

    def test_call_equivalent_in_conv(self):
        bas_con = self.cls(5, mode="conv", window_size=10)
        bas_eva = self.cls(5, mode="eval")
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con(x) == bas_eva(x))

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
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_pynapple_support(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas(x)
        y_nap = bas(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_basis_number(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        assert bas(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_non_empty(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(np.array([]))

    @pytest.mark.parametrize("mn, mx, expectation", [(0, 1, does_not_raise())])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_range(self, mn, mx, expectation, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(np.linspace(mn, mx, 10))

    def test_fit_kernel(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_.shape == (3, 5)

    def test_transform_fails(self):
        bas = self.cls(5, mode="conv", window_size=3)
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="`mode` should be either 'conv' or 'eval'"
                ),
            ),
        ],
    )
    def test_init_mode(self, mode, expectation):
        window_size = None if mode == "eval" else 2
        with expectation:
            self.cls(5, mode=mode, window_size=window_size)

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label):
        bas = self.cls(5, label=label)
        assert bas.label == (str(label) if label is not None else self.cls.__name__)

    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("label", None),
            ("label", "label"),
            ("n_basis_input", 1),
            ("n_output_features", 5),
        ],
    )
    def test_attr_setter(self, attribute, value):
        bas = self.cls(5)
        with pytest.raises(
            AttributeError, match=rf"can't set attribute|property '{attribute}' of"
        ):
            setattr(bas, attribute, value)

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_basis_input is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_basis_input == (n_input,)
        assert bas._n_basis_input == (n_input,)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (3, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas = self.cls(5, mode="conv", window_size=10)
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
    def test_init_conv_kwargs(self, conv_kwargs, expectation):
        with expectation:
            self.cls(5, mode="conv", window_size=200, **conv_kwargs)

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 2, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            ("eval", None, does_not_raise()),
            (
                "eval",
                10,
                pytest.raises(
                    ValueError,
                    match=r"If basis is in `mode=='eval'`, `window_size` should be None",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation):
        with expectation:
            self.cls(5, mode=mode, window_size=ws)

    @pytest.mark.parametrize(
        "order, window_size, n_basis_funcs, bounds, mode",
        [
            (4, None, 10, (1, 2), "eval"),
            (4, 10, 10, None, "conv"),
        ],
    )
    def test_set_params(
        self, order, window_size, n_basis_funcs, bounds, mode: Literal["eval", "conv"]
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            order=order,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
        )
        keys = list(pars.keys())
        bas = self.cls(
            order=order, window_size=window_size, n_basis_funcs=n_basis_funcs, mode=mode
        )
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas.set_params(**par_set)
                assert isinstance(bas, self.cls)

        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                with pytest.raises(
                    AttributeError,
                    match="can't set attribute 'mode'|property 'mode' of ",
                ):
                    par_set = {
                        keys[i]: pars[keys[i]],
                        keys[j]: pars[keys[j]],
                        "mode": mode,
                    }
                    bas.set_params(**par_set)

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
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
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
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", pytest.raises(ValueError, match="`bounds` should only be set")),
        ],
    )
    def test_set_bounds(self, mode, expectation):
        ws = dict(eval=None, conv=10)
        with expectation:
            self.cls(window_size=ws[mode], n_basis_funcs=10, mode=mode, bounds=(1, 2))

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv", bounds=None)
        with pytest.raises(ValueError, match="`bounds` should only be set"):
            bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            ("eval", pytest.raises(ValueError, match="If basis is in `mode=='eval'`")),
        ],
    )
    def test_set_window_size(self, mode, expectation):
        """Test window size set behavior."""
        with expectation:
            self.cls(window_size=10, n_basis_funcs=10, mode=mode)

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv")
        with pytest.raises(ValueError, match="If the basis is in `conv` mode"):
            bas.set_params(window_size=None)

        bas = self.cls(window_size=None, n_basis_funcs=10, mode="eval")
        with pytest.raises(ValueError, match="If basis is in `mode=='eval'`"):
            bas.set_params(window_size=10)

    def test_convolution_is_performed(self):
        bas = self.cls(5, mode="conv", window_size=10)
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_conv_kwargs_error(self):
        with pytest.raises(ValueError, match="kwargs should only be set"):
            self.cls(5, mode="eval", test="hi")

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation):
        with expectation:
            bas = self.cls(3, bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation):
        bas = self.cls(3, bounds=(1, 3))
        with expectation:
            bas.set_params(bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx):
        bounds = None if vmin is None else (vmin, vmax)
        bas = self.cls(3, mode="eval", bounds=bounds)
        out = bas.compute_features(samples)
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

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
        """Check that the MSpline has the expected scaling property."""
        bas_no_range = self.cls(3, mode="eval", bounds=None)
        bas = self.cls(3, mode="eval", bounds=bounds)
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        # multiply by scaling to get the invariance
        # mspline must integrate to one, if the support
        # is reduced, the height of the spline increases.
        assert np.allclose(out1 * scaling, out2)

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(self, bounds, samples, nan_idx, mn, mx):
        bas_no_range = self.cls(3, mode="eval", bounds=None)
        bas = self.cls(3, mode="eval", bounds=bounds)
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (None, np.arange(5), does_not_raise()),
            ((0, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 4), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception):
        with exception:
            self.cls(3, mode="conv", window_size=10, bounds=bounds)

    def test_transformer_get_params(self):
        bas = self.cls(5)
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        params_basis = bas.get_params()
        assert params_transf == params_basis


class TestOrthExponentialBasis(BasisFuncsTesting):
    cls = basis.OrthExponentialBasis

    # this class requires at leas `n_basis` samples
    @pytest.mark.parametrize("samples", [[], [0] * 30, [0] * 20])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_non_empty_samples(self, samples, mode, window_size):
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(
                    5, decay_rates=np.arange(1, 6), mode=mode, window_size=window_size
                ).compute_features(samples)
        else:
            self.cls(
                5, decay_rates=np.arange(1, 6), mode=mode, window_size=window_size
            ).compute_features(samples)

    @pytest.mark.parametrize(
        "eval_input",
        [0, [0] * 6, (0,) * 6, np.array([0] * 6), jax.numpy.array([0] * 6)],
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        if isinstance(eval_input, int):
            # OrthExponentialBasis is special -- cannot accept int input
            with pytest.raises(
                ValueError,
                match="OrthExponentialBasis requires at least as many samples",
            ):
                basis_obj.compute_features(eval_input)
        else:
            basis_obj.compute_features(eval_input)

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
    def test_compute_features_axis(self, kwargs, input1_shape, expectation):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        with expectation:
            basis_obj = self.cls(
                n_basis_funcs=5,
                mode="conv",
                window_size=5,
                decay_rates=np.arange(1, 6),
                **kwargs,
            )
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [2, 3])
    @pytest.mark.parametrize("window_size", [10, 15])
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
        window_size,
        input_shape,
        expected_n_input,
    ):
        x = np.ones(input_shape)
        bas = self.cls(
            n_basis_funcs=n_basis_funcs,
            mode="conv",
            window_size=window_size,
            decay_rates=0.1 * np.arange(1, n_basis_funcs + 1),
        )
        out = bas.compute_features(x)
        assert out.shape[1] == expected_n_input * n_basis_funcs

    @pytest.mark.parametrize("n_basis_funcs", [1, 2, 4, 8])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_compute_features_returns_expected_number_of_basis(
        self, n_basis_funcs, sample_size, mode, window_size
    ):
        """Tests whether the evaluate method returns the expected number of basis functions."""
        decay_rates = np.arange(1, 1 + n_basis_funcs)
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs,
            decay_rates=decay_rates,
            mode=mode,
            window_size=window_size,
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the output features basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 30)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, mode, window_size
    ):
        """Tests whether the sample size of the features result matches that of the input."""
        decay_rates = np.linspace(0.1, 20, n_basis_funcs)
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs,
            decay_rates=decay_rates,
            mode=mode,
            window_size=window_size,
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize(
        "samples, vmin, vmax, expectation",
        [
            (
                np.linspace(-0.5, -0.001, 7),
                0,
                1,
                pytest.raises(ValueError, match="All the samples lie outside"),
            ),
            (
                np.linspace(1.5, 2.0, 7),
                0,
                1,
                pytest.raises(ValueError, match="All the samples lie outside"),
            ),
            (
                [-0.5, -0.1, -0.01, 1.5, 2, 3],
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
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax), decay_rates=np.linspace(0.1, 1, 5))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 30)])
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, window_size
    ):
        """Tests whether the class instance has a minimum number of basis functions."""
        raise_exception = n_basis_funcs < 1
        decay_rates = np.arange(1, 1 + n_basis_funcs)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls.__name__} "
                r"requires >= 1 basis elements\.",
            ):
                self.cls(
                    n_basis_funcs=n_basis_funcs,
                    decay_rates=decay_rates,
                    mode=mode,
                    window_size=window_size,
                )
        else:
            self.cls(
                n_basis_funcs=n_basis_funcs,
                decay_rates=decay_rates,
                mode=mode,
                window_size=window_size,
            )

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, window_size
    ):
        """Tests whether the compute_features method correctly processes the number of required inputs."""
        basis_obj = self.cls(
            n_basis_funcs=5,
            decay_rates=np.arange(1, 6),
            mode=mode,
            window_size=window_size,
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="Input dimensionality mismatch",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 2, 3, 4, 5, 6, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """Tests whether the compute_features_on_grid method correctly outputs the grid mesh size."""
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        raise_exception = sample_size < 5
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls.__name__} requires at least as "
                r"many samples as basis functions\!|"
                r"All sample counts provided must be greater",
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        """Tests whether the evaluate_on_grid method correctly outputs the basis size."""
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        raise_exception = sample_size < 5
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"All sample counts provided must be greater|"
                rf"{self.cls.__name__} requires at least as many samples as basis",
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        """Tests whether the evaluate_on_grid method correctly processes the Input dimensionality."""
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize(
        "decay_rates", [[1, 2, 3], [0.01, 0.02, 0.001], [2, 1, 1, 2.4]]
    )
    def test_decay_rate_repetition(self, decay_rates):
        """
        Tests whether the class instance correctly processes the decay rates without repetition.
        A repeated rate causes linear algebra issues, and should raise a ValyeError exception.
        """
        decay_rates = np.asarray(decay_rates, dtype=float)
        # raise exception if any of the decay rate is repeated
        raise_exception = len(set(decay_rates)) != len(decay_rates)
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"Two or more rate are repeated\! Repeating rate will"
            ):
                self.cls(n_basis_funcs=len(decay_rates), decay_rates=decay_rates)
        else:
            self.cls(n_basis_funcs=len(decay_rates), decay_rates=decay_rates)

    @pytest.mark.parametrize(
        "decay_rates", [[], [1], [1, 2, 3], [1, 0.01, 0.02, 0.001]]
    )
    @pytest.mark.parametrize("n_basis_func", [1, 2, 3, 4])
    def test_decay_rate_size_match_n_basis_func(self, decay_rates, n_basis_func):
        """Tests whether the size of decay rates matches the number of basis functions."""
        raise_exception = len(decay_rates) != n_basis_func
        decay_rates = np.asarray(decay_rates, dtype=float)
        if raise_exception:
            with pytest.raises(
                ValueError, match="The number of basis functions must match the"
            ):
                self.cls(n_basis_funcs=n_basis_func, decay_rates=decay_rates)
        else:
            self.cls(n_basis_funcs=n_basis_func, decay_rates=decay_rates)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = self.cls(n_basis, np.arange(1, n_basis + 1)).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_input_num(self, num_input, mode, window_size, expectation):
        bas = self.cls(
            5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6)
        )
        with expectation:
            bas(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_input_shape(self, inp, mode, window_size, expectation):
        bas = self.cls(
            5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6)
        )
        with expectation:
            bas(inp)

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_sample_axis(self, time_axis_shape, mode, window_size):
        bas = self.cls(
            5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6)
        )
        assert bas(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_nan(self, mode, window_size):
        bas = self.cls(
            5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6)
        )
        x = np.linspace(0, 1, 15)
        x[13] = np.nan
        with does_not_raise():
            out = bas(x)
            assert np.all(np.isnan(out[13]))

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
    def test_call_input_type(self, samples, expectation):
        bas = self.cls(5, np.linspace(0.1, 1, 5))
        with expectation:
            bas(samples)

    def test_call_equivalent_in_conv(self):
        bas_con = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        bas_eva = self.cls(5, mode="eval", decay_rates=np.arange(1, 6))
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con(x) == bas_eva(x))

    @pytest.mark.parametrize(
        "samples, vmin, vmax, expectation",
        [
            (
                np.linspace(-1, -0.5, 10),
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
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, decay_rates=np.linspace(0, 1, 5), bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_pynapple_support(self, mode, window_size):
        bas = self.cls(
            5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6)
        )
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas(x)
        y_nap = bas(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_basis_number(self, n_basis, mode, window_size):
        bas = self.cls(
            n_basis,
            mode=mode,
            window_size=window_size,
            decay_rates=np.arange(1, n_basis + 1),
        )
        x = np.linspace(0, 1, 10)
        assert bas(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_non_empty(self, n_basis, mode, window_size):
        bas = self.cls(
            n_basis,
            mode=mode,
            window_size=window_size,
            decay_rates=np.arange(1, n_basis + 1),
        )
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(np.array([]))

    @pytest.mark.parametrize("mn, mx, expectation", [(0, 1, does_not_raise())])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_call_sample_range(self, mn, mx, expectation, mode, window_size):
        bas = self.cls(
            5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6)
        )
        with expectation:
            bas(np.linspace(mn, mx, 10))

    def test_fit_kernel(self):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        bas._set_kernel(None)
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        bas._set_kernel(None)
        assert bas.kernel_.shape == (10, 5)

    def test_transform_fails(self):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="`mode` should be either 'conv' or 'eval'"
                ),
            ),
        ],
    )
    def test_init_mode(self, mode, expectation):
        window_size = None if mode == "eval" else 10
        with expectation:
            self.cls(5, mode=mode, window_size=window_size, decay_rates=np.arange(1, 6))

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label):
        bas = self.cls(5, label=label, decay_rates=np.arange(1, 6))
        assert bas.label == (str(label) if label is not None else self.cls.__name__)

    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("label", None),
            ("label", "label"),
            ("n_basis_input", 1),
            ("n_output_features", 5),
        ],
    )
    def test_attr_setter(self, attribute, value):
        bas = self.cls(5, decay_rates=np.arange(1, 6))
        with pytest.raises(
            AttributeError, match=rf"can't set attribute|property '{attribute}' of"
        ):
            setattr(bas, attribute, value)

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        assert bas.n_basis_input is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_basis_input == (n_input,)
        assert bas._n_basis_input == (n_input,)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (3, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
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
    def test_init_conv_kwargs(self, conv_kwargs, expectation):
        with expectation:
            self.cls(
                5,
                mode="conv",
                window_size=200,
                decay_rates=np.arange(1, 6),
                **conv_kwargs,
            )

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 2, does_not_raise()),
            ("conv", 10, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            ("eval", None, does_not_raise()),
            (
                "eval",
                10,
                pytest.raises(
                    ValueError,
                    match=r"If basis is in `mode=='eval'`, `window_size` should be None",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation):
        with expectation:
            self.cls(5, mode=mode, window_size=ws, decay_rates=np.arange(1, 6))

    @pytest.mark.parametrize(
        "decay_rates, window_size, n_basis_funcs, bounds, mode",
        [
            (np.arange(1, 11), None, 10, (1, 2), "eval"),
            (np.arange(1, 11), 10, 10, None, "conv"),
        ],
    )
    def test_set_params(
        self,
        decay_rates,
        window_size,
        n_basis_funcs,
        bounds,
        mode: Literal["eval", "conv"],
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            decay_rates=decay_rates,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
        )
        keys = list(pars.keys())
        bas = self.cls(
            decay_rates=decay_rates,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            mode=mode,
        )
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas.set_params(**par_set)
                assert isinstance(bas, self.cls)

        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                with pytest.raises(
                    AttributeError,
                    match="can't set attribute 'mode'|property 'mode' of ",
                ):
                    par_set = {
                        keys[i]: pars[keys[i]],
                        keys[j]: pars[keys[j]],
                        "mode": mode,
                    }
                    bas.set_params(**par_set)

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", pytest.raises(ValueError, match="`bounds` should only be set")),
        ],
    )
    def test_set_bounds(self, mode, expectation):
        ws = dict(eval=None, conv=10)
        with expectation:
            self.cls(
                decay_rates=np.arange(1, 11),
                window_size=ws[mode],
                n_basis_funcs=10,
                mode=mode,
                bounds=(1, 2),
            )

        bas = self.cls(
            decay_rates=np.arange(1, 11),
            window_size=10,
            n_basis_funcs=10,
            mode="conv",
            bounds=None,
        )
        with pytest.raises(ValueError, match="`bounds` should only be set"):
            bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            ("eval", pytest.raises(ValueError, match="If basis is in `mode=='eval'`")),
        ],
    )
    def test_set_window_size(self, mode, expectation):
        """Test window size set behavior."""
        with expectation:
            self.cls(
                decay_rates=np.arange(1, 11),
                window_size=10,
                n_basis_funcs=10,
                mode=mode,
            )

        bas = self.cls(
            decay_rates=np.arange(1, 11), window_size=10, n_basis_funcs=10, mode="conv"
        )
        with pytest.raises(ValueError, match="If the basis is in `conv` mode"):
            bas.set_params(window_size=None)

        bas = self.cls(
            decay_rates=np.arange(1, 11),
            window_size=None,
            n_basis_funcs=10,
            mode="eval",
        )
        with pytest.raises(ValueError, match="If basis is in `mode=='eval'`"):
            bas.set_params(window_size=10)

    def test_convolution_is_performed(self):
        bas = self.cls(5, mode="conv", window_size=10, decay_rates=np.arange(1, 6))
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_conv_kwargs_error(self):
        with pytest.raises(ValueError, match="kwargs should only be set"):
            self.cls(5, decay_rates=[1, 2, 3, 4, 5], mode="eval", test="hi")

    def test_transformer_get_params(self):
        bas = self.cls(5, decay_rates=[1, 2, 3, 4, 5])
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        rates_transf = params_transf.pop("decay_rates")
        params_basis = bas.get_params()
        rates_basis = params_basis.pop("decay_rates")
        assert params_transf == params_basis
        assert np.all(rates_transf == rates_basis)


class TestBSplineBasis(BasisFuncsTesting):
    cls = basis.BSplineBasis

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_non_empty_samples(self, samples, mode, window_size):
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5, mode=mode, window_size=window_size).compute_features(
                    samples
                )
        else:
            self.cls(5, mode=mode, window_size=window_size).compute_features(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

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
    def test_compute_features_axis(self, kwargs, input1_shape, expectation):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        with expectation:
            basis_obj = self.cls(n_basis_funcs=5, mode="conv", window_size=5, **kwargs)
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [2, 3])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("window_size", [10, 15])
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
        order,
        window_size,
        input_shape,
        expected_n_input,
    ):
        x = np.ones(input_shape)
        bas = self.cls(
            n_basis_funcs=n_basis_funcs,
            order=order,
            mode="conv",
            window_size=window_size,
        )
        out = bas.compute_features(x)
        assert out.shape[1] == expected_n_input * n_basis_funcs

    @pytest.mark.parametrize("n_basis_funcs", [6, 8, 10])
    @pytest.mark.parametrize("order", range(1, 6))
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_compute_features_returns_expected_number_of_basis(
        self, n_basis_funcs: int, order: int, mode, window_size
    ):
        """
        Verifies that the compute_features() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, order=order, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, 100))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [4, 10, 100])
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, order, mode, window_size
    ):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, order=order, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features is {eval_basis.shape[0]}",
            )

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
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, window_size
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order > n_basis_funcs
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls.__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls(
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    mode=mode,
                    window_size=window_size,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(
                n_basis_funcs=n_basis_funcs,
                order=order,
                mode=mode,
                window_size=window_size,
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
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
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
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=4)
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
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, window_size
    ):
        """
        Confirms that the compute_features() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(
            n_basis_funcs=5, order=3, mode=mode, window_size=window_size
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="Input dimensionality mismatch",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """
        Checks that the evaluate_on_grid() method returns a grid of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Invalid input data|"
                r"All sample counts provided must be greater",
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        """
        Ensures that the evaluate_on_grid() method returns basis functions of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"All sample counts provided must be greater|"
                r"Invalid input data",
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        """
        Validates that the evaluate_on_grid() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = self.cls(n_basis).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(self, num_input, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_shape(self, inp, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(inp)

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_axis(self, time_axis_shape, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        assert bas(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_nan(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas(x)[3]))

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
    def test_call_input_type(self, samples, expectation):
        bas = self.cls(5)
        with expectation:
            bas(samples)

    def test_call_equivalent_in_conv(self):
        bas_con = self.cls(5, mode="conv", window_size=10)
        bas_eva = self.cls(5, mode="eval")
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con(x) == bas_eva(x))

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
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_pynapple_support(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas(x)
        y_nap = bas(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("n_basis", [6, 7])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_basis_number(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        assert bas(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [6, 7])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_non_empty(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(np.array([]))

    @pytest.mark.parametrize(
        "mn, mx, expectation", [(0, 1, does_not_raise()), (-2, 2, does_not_raise())]
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_range(self, mn, mx, expectation, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(np.linspace(mn, mx, 10))

    def test_fit_kernel(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_.shape == (3, 5)

    def test_transform_fails(self):
        bas = self.cls(5, mode="conv", window_size=3)
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="`mode` should be either 'conv' or 'eval'"
                ),
            ),
        ],
    )
    def test_init_mode(self, mode, expectation):
        window_size = None if mode == "eval" else 2
        with expectation:
            self.cls(5, mode=mode, window_size=window_size)

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label):
        bas = self.cls(5, label=label)
        assert bas.label == (str(label) if label is not None else self.cls.__name__)

    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("label", None),
            ("label", "label"),
            ("n_basis_input", 1),
            ("n_output_features", 5),
        ],
    )
    def test_attr_setter(self, attribute, value):
        bas = self.cls(5)
        with pytest.raises(
            AttributeError, match=rf"can't set attribute|property '{attribute}' of"
        ):
            setattr(bas, attribute, value)

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input):
        bas = self.cls(5, mode="conv", window_size=10)
        assert bas.n_basis_input is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_basis_input == (n_input,)
        assert bas._n_basis_input == (n_input,)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (3, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas = self.cls(5, mode="conv", window_size=10)
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
    def test_init_conv_kwargs(self, conv_kwargs, expectation):
        with expectation:
            self.cls(5, mode="conv", window_size=200, **conv_kwargs)

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 2, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            ("eval", None, does_not_raise()),
            (
                "eval",
                10,
                pytest.raises(
                    ValueError,
                    match=r"If basis is in `mode=='eval'`, `window_size` should be None",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation):
        with expectation:
            self.cls(5, mode=mode, window_size=ws)

    @pytest.mark.parametrize(
        "order, window_size, n_basis_funcs, bounds, mode",
        [
            (3, None, 10, (1, 2), "eval"),
            (3, 10, 10, None, "conv"),
        ],
    )
    def test_set_params(
        self, order, window_size, n_basis_funcs, bounds, mode: Literal["eval", "conv"]
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            order=order,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
        )
        keys = list(pars.keys())
        bas = self.cls(
            order=order, window_size=window_size, n_basis_funcs=n_basis_funcs, mode=mode
        )
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas.set_params(**par_set)
                assert isinstance(bas, self.cls)

        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                with pytest.raises(
                    AttributeError,
                    match="can't set attribute 'mode'|property 'mode' of ",
                ):
                    par_set = {
                        keys[i]: pars[keys[i]],
                        keys[j]: pars[keys[j]],
                        "mode": mode,
                    }
                    bas.set_params(**par_set)

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", pytest.raises(ValueError, match="`bounds` should only be set")),
        ],
    )
    def test_set_bounds(self, mode, expectation):
        ws = dict(eval=None, conv=10)
        with expectation:
            self.cls(window_size=ws[mode], n_basis_funcs=10, mode=mode, bounds=(1, 2))

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv", bounds=None)
        with pytest.raises(ValueError, match="`bounds` should only be set"):
            bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            ("eval", pytest.raises(ValueError, match="If basis is in `mode=='eval'`")),
        ],
    )
    def test_set_window_size(self, mode, expectation):
        """Test window size set behavior."""
        with expectation:
            self.cls(window_size=10, n_basis_funcs=10, mode=mode)

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv")
        with pytest.raises(ValueError, match="If the basis is in `conv` mode"):
            bas.set_params(window_size=None)

        bas = self.cls(window_size=None, n_basis_funcs=10, mode="eval")
        with pytest.raises(ValueError, match="If basis is in `mode=='eval'`"):
            bas.set_params(window_size=10)

    def test_convolution_is_performed(self):
        bas = self.cls(5, mode="conv", window_size=10)
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_conv_kwargs_error(self):
        with pytest.raises(ValueError, match="kwargs should only be set"):
            self.cls(5, mode="eval", test="hi")

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation):
        with expectation:
            bas = self.cls(5, bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation):
        bas = self.cls(5, bounds=(1, 3))
        with expectation:
            bas.set_params(bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx):
        bounds = None if vmin is None else (vmin, vmax)
        bas = self.cls(5, mode="eval", bounds=bounds)
        out = bas.compute_features(samples)
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_eval_on_grid_no_effect_on_eval(
        self, vmin, vmax, samples, nan_idx
    ):
        bas_no_range = self.cls(5, mode="eval", bounds=None)
        bas = self.cls(5, mode="eval", bounds=(vmin, vmax))
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1, out2)

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(self, bounds, samples, nan_idx, mn, mx):
        bas_no_range = self.cls(5, mode="eval", bounds=None)
        bas = self.cls(5, mode="eval", bounds=bounds)
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (None, np.arange(5), does_not_raise()),
            ((0, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 4), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception):
        with exception:
            self.cls(5, mode="conv", window_size=10, bounds=bounds)

    def test_transformer_get_params(self):
        bas = self.cls(5)
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        params_basis = bas.get_params()
        assert params_transf == params_basis


class TestCyclicBSplineBasis(BasisFuncsTesting):
    cls = basis.CyclicBSplineBasis

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_non_empty_samples(self, samples, mode, window_size):
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5, mode=mode, window_size=window_size).compute_features(
                    samples
                )
        else:
            self.cls(5, mode=mode, window_size=window_size).compute_features(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

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
    def test_compute_features_axis(self, kwargs, input1_shape, expectation):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        with expectation:
            basis_obj = self.cls(n_basis_funcs=5, mode="conv", window_size=5, **kwargs)
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [4, 5])
    @pytest.mark.parametrize("order", [3, 2])
    @pytest.mark.parametrize("window_size", [10, 15])
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
        order,
        window_size,
        input_shape,
        expected_n_input,
    ):
        x = np.ones(input_shape)
        bas = self.cls(
            n_basis_funcs=n_basis_funcs,
            order=order,
            mode="conv",
            window_size=window_size,
        )
        out = bas.compute_features(x)
        assert out.shape[1] == expected_n_input * n_basis_funcs

    @pytest.mark.parametrize("n_basis_funcs", [8, 10])
    @pytest.mark.parametrize("order", range(2, 6))
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_compute_features_returns_expected_number_of_basis(
        self, n_basis_funcs: int, order: int, mode, window_size
    ):
        """
        Verifies that the compute_features() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, order=order, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, 100))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the output features is {eval_basis.shape[0]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [8, 10, 100])
    @pytest.mark.parametrize("order", [2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, order, mode, window_size
    ):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = self.cls(
            n_basis_funcs=n_basis_funcs, order=order, mode=mode, window_size=window_size
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features is {eval_basis.shape[1]}",
            )

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
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 2)])
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, window_size
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order > n_basis_funcs
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls.__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls(
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    mode=mode,
                    window_size=window_size,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(
                n_basis_funcs=n_basis_funcs,
                order=order,
                mode=mode,
                window_size=window_size,
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 2, 3])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
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
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_1_invalid(self, n_basis_funcs, order):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order == 1
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"Order >= 2 required for cyclic B-spline"
            ):
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
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
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, window_size
    ):
        """
        Confirms that the compute_features() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(
            n_basis_funcs=5, order=3, mode=mode, window_size=window_size
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="Input dimensionality mismatch",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """
        Checks that the evaluate_on_grid() method returns a grid of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Empty sample array provided\. At least one sample is required|"
                "All sample counts provided must be greater",
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        """
        Ensures that the evaluate_on_grid() method returns basis functions of the expected size.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        raise_exception = sample_size <= 0
        if raise_exception:
            with pytest.raises(
                ValueError,
                match="All sample counts provided must be greater|"
                r"Empty sample array provided\. At least one sample is required for",
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        """
        Validates that the evaluate_on_grid() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = self.cls(n_basis).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(self, num_input, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_shape(self, inp, mode, window_size, expectation):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(inp)

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_axis(self, time_axis_shape, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        assert bas(np.linspace(0, 1, time_axis_shape)).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_nan(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas(x)[3]))

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
    def test_call_input_type(self, samples, expectation):
        bas = self.cls(5)
        with expectation:
            bas(samples)

    def test_call_equivalent_in_conv(self):
        bas_con = self.cls(5, mode="conv", window_size=10)
        bas_eva = self.cls(5, mode="eval")
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con(x) == bas_eva(x))

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
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation):
        bas = self.cls(5, bounds=(vmin, vmax))
        with expectation:
            bas(samples)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_pynapple_support(self, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas(x)
        y_nap = bas(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("n_basis", [6, 7])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_basis_number(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        x = np.linspace(0, 1, 10)
        assert bas(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [6, 7])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_non_empty(self, n_basis, mode, window_size):
        bas = self.cls(n_basis, mode=mode, window_size=window_size)
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(np.array([]))

    @pytest.mark.parametrize(
        "mn, mx, expectation", [(0, 1, does_not_raise()), (-2, 2, does_not_raise())]
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_sample_range(self, mn, mx, expectation, mode, window_size):
        bas = self.cls(5, mode=mode, window_size=window_size)
        with expectation:
            bas(np.linspace(mn, mx, 10))

    def test_fit_kernel(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self):
        bas = self.cls(5, mode="conv", window_size=3)
        bas._set_kernel(None)
        assert bas.kernel_.shape == (3, 5)

    def test_transform_fails(self):
        bas = self.cls(5, mode="conv", window_size=3)
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 2, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            ("eval", None, does_not_raise()),
            (
                "eval",
                10,
                pytest.raises(
                    ValueError,
                    match=r"If basis is in `mode=='eval'`, `window_size` should be None",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation):
        with expectation:
            self.cls(5, mode=mode, window_size=ws)

    @pytest.mark.parametrize(
        "order, window_size, n_basis_funcs, bounds, mode",
        [
            (3, None, 10, (1, 2), "eval"),
            (3, 10, 10, None, "conv"),
        ],
    )
    def test_set_params(
        self, order, window_size, n_basis_funcs, bounds, mode: Literal["eval", "conv"]
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            order=order,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
        )
        keys = list(pars.keys())
        bas = self.cls(
            order=order, window_size=window_size, n_basis_funcs=n_basis_funcs, mode=mode
        )
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas.set_params(**par_set)
                assert isinstance(bas, self.cls)

        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                with pytest.raises(
                    AttributeError,
                    match="can't set attribute 'mode'|property 'mode' of ",
                ):
                    par_set = {
                        keys[i]: pars[keys[i]],
                        keys[j]: pars[keys[j]],
                        "mode": mode,
                    }
                    bas.set_params(**par_set)

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            ("conv", pytest.raises(ValueError, match="`bounds` should only be set")),
        ],
    )
    def test_set_bounds(self, mode, expectation):
        ws = dict(eval=None, conv=10)
        with expectation:
            self.cls(window_size=ws[mode], n_basis_funcs=10, mode=mode, bounds=(1, 2))

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv", bounds=None)
        with pytest.raises(ValueError, match="`bounds` should only be set"):
            bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            ("eval", pytest.raises(ValueError, match="If basis is in `mode=='eval'`")),
        ],
    )
    def test_set_window_size(self, mode, expectation):
        """Test window size set behavior."""
        with expectation:
            self.cls(window_size=10, n_basis_funcs=10, mode=mode)

        bas = self.cls(window_size=10, n_basis_funcs=10, mode="conv")
        with pytest.raises(ValueError, match="If the basis is in `conv` mode"):
            bas.set_params(window_size=None)

        bas = self.cls(window_size=None, n_basis_funcs=10, mode="eval")
        with pytest.raises(ValueError, match="If basis is in `mode=='eval'`"):
            bas.set_params(window_size=10)

    def test_convolution_is_performed(self):
        bas = self.cls(5, mode="conv", window_size=10)
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    def test_conv_kwargs_error(self):
        with pytest.raises(ValueError, match="kwargs should only be set"):
            self.cls(5, mode="eval", test="hi")

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation):
        with expectation:
            bas = self.cls(5, bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation):
        bas = self.cls(5, bounds=(1, 3))
        with expectation:
            bas.set_params(bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx):
        bounds = None if vmin is None else (vmin, vmax)
        bas = self.cls(5, mode="eval", bounds=bounds)
        out = bas.compute_features(samples)
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_eval_on_grid_no_effect_on_eval(
        self, vmin, vmax, samples, nan_idx
    ):
        bas_no_range = self.cls(5, mode="eval", bounds=None)
        bas = self.cls(5, mode="eval", bounds=(vmin, vmax))
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1, out2)

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(self, bounds, samples, nan_idx, mn, mx):
        bas_no_range = self.cls(5, mode="eval", bounds=None)
        bas = self.cls(5, mode="eval", bounds=bounds)
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (None, np.arange(5), does_not_raise()),
            ((0, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 4), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
            ((1, 3), np.arange(5), pytest.raises(ValueError, match="`bounds` should")),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception):
        with exception:
            self.cls(5, mode="conv", window_size=10, bounds=bounds)

    def test_transformer_get_params(self):
        bas = self.cls(5)
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        params_basis = bas.get_params()
        assert params_transf == params_basis


class CombinedBasis(BasisFuncsTesting):
    """
    This class is used to run tests on combination operations (e.g., addition, multiplication) among Basis functions.

    Properties:
    - cls: Class (default = None)
    """

    cls = None

    @staticmethod
    def instantiate_basis(n_basis, basis_class, mode="eval", window_size=10):
        """Instantiate and return two basis of the type specified."""

        if mode == "eval":
            window_size = None

        if basis_class == basis.EvalMSpline:
            basis_obj = basis_class(
                n_basis_funcs=n_basis, order=4, mode=mode, window_size=window_size
            )
        elif basis_class in [basis.RaisedCosineBasisLinear, basis.RaisedCosineBasisLog]:
            basis_obj = basis_class(
                n_basis_funcs=n_basis, mode=mode, window_size=window_size
            )
        elif basis_class == basis.OrthExponentialBasis:
            basis_obj = basis_class(
                n_basis_funcs=n_basis,
                decay_rates=np.arange(1, 1 + n_basis),
                mode=mode,
                window_size=window_size,
            )
        elif basis_class == basis.BSplineBasis:
            basis_obj = basis_class(
                n_basis_funcs=n_basis, order=3, mode=mode, window_size=window_size
            )
        elif basis_class == basis.CyclicBSplineBasis:
            basis_obj = basis_class(
                n_basis_funcs=n_basis, order=3, mode=mode, window_size=window_size
            )
        elif basis_class == AdditiveBasis:
            b1 = basis.EvalMSpline(
                n_basis_funcs=n_basis, order=2, mode=mode, window_size=window_size
            )
            b2 = basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis + 1)
            basis_obj = b1 + b2
        elif basis_class == MultiplicativeBasis:
            b1 = basis.EvalMSpline(
                n_basis_funcs=n_basis, order=2, mode=mode, window_size=window_size
            )
            b2 = basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis + 1)
            basis_obj = b1 * b2
        else:
            raise ValueError(
                f"Test for basis addition not implemented for basis of type {basis_class}!"
            )
        return basis_obj


class TestAdditiveBasis(CombinedBasis):
    cls = AdditiveBasis

    @pytest.mark.parametrize(
        "samples", [[[0], []], [[], [0]], [[0], [0]], [[0, 0], [0, 0]]]
    )
    @pytest.mark.parametrize("mode, ws", [("conv", 2), ("eval", None)])
    def test_non_empty_samples(self, samples, mode, ws):
        if mode == "conv" and len(samples[0]) < 2:
            return
        basis_obj = basis.EvalMSpline(5, mode=mode, window_size=ws) + basis.EvalMSpline(
            5, mode=mode, window_size=ws
        )
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
        basis_obj = basis.EvalMSpline(5) + basis.EvalMSpline(5)
        basis_obj.compute_features(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_compute_features_returns_expected_number_of_basis(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b, mode, window_size
    ):
        """
        Test whether the evaluation of the `AdditiveBasis` results in a number of basis
        that is the sum of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
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
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b, mode, window_size
    ):
        """
        Test whether the output sample size from `AdditiveBasis` compute_features function matches input sample size.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
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

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        """
        Test whether the number of required inputs for the `compute_features` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_meshgrid_size(
        self, sample_size, n_basis_a, n_basis_b, basis_a, basis_b
    ):
        """
        Test whether the resulting meshgrid size matches the sample size input.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj + basis_b_obj
        res = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )
        for grid in res[:-1]:
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_basis_size(
        self, sample_size, n_basis_a, n_basis_b, basis_a, basis_b
    ):
        """
        Test whether the number sample size output by evaluate_on_grid matches the sample size of the input.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )[-1]
        assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2, 5, 6, 11, 30])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_input_number(
        self, n_input, basis_a, basis_b, n_basis_a, n_basis_b
    ):
        """
        Test whether the number of inputs provided to `evaluate_on_grid` matches
        the sum of the number of input samples required from each of the basis objects.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj + basis_b_obj
        inputs = [20] * n_input
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_pynapple_support_compute_features(
        self, basis_a, basis_b, n_basis_a, n_basis_b, sample_size
    ):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        basis_add = self.instantiate_basis(n_basis_a, basis_a) + self.instantiate_basis(
            n_basis_b, basis_b
        )
        # compute_features the basis over pynapple Tsd objects
        out = basis_add.compute_features(*([inp] * basis_add._n_input_dimensionality))
        # check type
        assert isinstance(out, nap.TsdFrame)
        # check value
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(
        self, n_basis_a, n_basis_b, basis_a, basis_b, num_input, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch"
            )
        with expectation:
            basis_obj(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_input_shape(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        inp,
        mode,
        window_size,
        expectation,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        with expectation:
            basis_obj(*([inp] * basis_obj._n_input_dimensionality))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_axis(
        self, n_basis_a, n_basis_b, basis_a, basis_b, time_axis_shape, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        inp = [np.linspace(0, 1, time_axis_shape)] * basis_obj._n_input_dimensionality
        assert basis_obj(*inp).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_nan(self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size):
        if (
            basis_a == basis.OrthExponentialBasis
            or basis_b == basis.OrthExponentialBasis
        ):
            return
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        inp = [np.linspace(0, 1, 10)] * basis_obj._n_input_dimensionality
        for x in inp:
            x[3] = np.nan
        assert all(np.isnan(basis_obj(*inp)[3]))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_equivalent_in_conv(self, n_basis_a, n_basis_b, basis_a, basis_b):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="eval", window_size=None
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="eval", window_size=None
        )
        bas_eva = basis_a_obj + basis_b_obj

        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="conv", window_size=8
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="conv", window_size=8
        )
        bas_con = basis_a_obj + basis_b_obj

        x = [np.linspace(0, 1, 10)] * bas_con._n_input_dimensionality
        assert np.all(bas_con(*x) == bas_eva(*x))

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_pynapple_support(
        self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        x = np.linspace(0, 1, 10)
        x_nap = [nap.Tsd(t=np.arange(10), d=x)] * bas._n_input_dimensionality
        x = [x] * bas._n_input_dimensionality
        y = bas(*x)
        y_nap = bas(*x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap[0].t)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [6, 7])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_basis_number(
        self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
        assert bas(*x).shape[1] == basis_a_obj.n_basis_funcs + basis_b_obj.n_basis_funcs

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_non_empty(
        self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(*([np.array([])] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
            (0.1, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
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
        mode,
        window_size,
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
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        with expectation:
            bas(*([np.linspace(mn, mx, 10)] * bas._n_input_dimensionality))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_fit_kernel(self, n_basis_a, n_basis_b, basis_a, basis_b):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="conv", window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="conv", window_size=10
        )
        bas = basis_a_obj + basis_b_obj
        bas._set_kernel(None)

        def check_kernel(basis_obj):
            has_kern = []
            if hasattr(basis_obj, "_basis1"):
                has_kern += check_kernel(basis_obj._basis1)
                has_kern += check_kernel(basis_obj._basis2)
            else:
                has_kern += [
                    basis_obj.kernel_ is not None if basis_obj.mode == "conv" else True
                ]
            return has_kern

        assert all(check_kernel(bas))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_transform_fails(self, n_basis_a, n_basis_b, basis_a, basis_b):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="conv", window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="conv", window_size=10
        )
        bas = basis_a_obj + basis_b_obj
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
            bas._compute_features(*x)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_output_features(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(11, mode="conv", window_size=10)
        bas_add = bas1 + bas2
        assert bas_add.n_output_features is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_output_features == (n_basis_input1 * 10 + n_basis_input2 * 11)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_basis_input(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(10, mode="conv", window_size=10)
        bas_add = bas1 + bas2
        assert bas_add.n_basis_input is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_basis_input == (n_basis_input1, n_basis_input2)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (3, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (4, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(10, mode="conv", window_size=10)
        bas = bas1 + bas2
        x = np.random.randn(20, 2), np.random.randn(20, 3)
        bas.compute_features(*x)
        with expectation:
            bas.compute_features(np.random.randn(30, 2), np.random.randn(30, n_input))


class TestMultiplicativeBasis(CombinedBasis):
    cls = MultiplicativeBasis

    @pytest.mark.parametrize(
        "samples", [[[0], []], [[], [0]], [[0], [0]], [[0, 0], [0, 0]]]
    )
    @pytest.mark.parametrize("mode, ws", [("conv", 2), ("eval", None)])
    def test_non_empty_samples(self, samples, mode, ws):
        if mode == "conv" and len(samples[0]) < 2:
            return
        basis_obj = basis.EvalMSpline(5, mode=mode, window_size=ws) * basis.EvalMSpline(
            5, mode=mode, window_size=ws
        )
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
        basis_obj = basis.EvalMSpline(5) * basis.EvalMSpline(5)
        basis_obj.compute_features(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_compute_features_returns_expected_number_of_basis(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b, mode, window_size
    ):
        """
        Test whether the evaluation of the `MultiplicativeBasis` results in a number of basis
        that is the product of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
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
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b, mode, window_size
    ):
        """
        Test whether the output sample size from the `MultiplicativeBasis` fit_transform function
        matches the input sample size.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
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

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 10)])
    def test_number_of_required_inputs_compute_features(
        self, n_input, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        """
        Test whether the number of required inputs for the `compute_features` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_meshgrid_size(
        self, sample_size, n_basis_a, n_basis_b, basis_a, basis_b
    ):
        """
        Test whether the resulting meshgrid size matches the sample size input.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj * basis_b_obj
        res = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )
        for grid in res[:-1]:
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_basis_size(
        self, sample_size, n_basis_a, n_basis_b, basis_a, basis_b
    ):
        """
        Test whether the number sample size output by evaluate_on_grid matches the sample size of the input.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )[-1]
        assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2, 5, 6, 11, 30])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_input_number(
        self, n_input, basis_a, basis_b, n_basis_a, n_basis_b
    ):
        """
        Test whether the number of inputs provided to `evaluate_on_grid` matches
        the sum of the number of input samples required from each of the basis objects.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj * basis_b_obj
        inputs = [20] * n_input
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("basis_a", [basis.EvalMSpline])
    @pytest.mark.parametrize("basis_b", [basis.OrthExponentialBasis])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    @pytest.mark.parametrize("sample_size_a", [11, 12])
    @pytest.mark.parametrize("sample_size_b", [11, 12])
    def test_inconsistent_sample_sizes(
        self, basis_a, basis_b, n_basis_a, n_basis_b, sample_size_a, sample_size_b
    ):
        """Test that the inputs of inconsistent sample sizes result in an exception when compute_features is called"""
        raise_exception = sample_size_a != sample_size_b
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj * basis_b_obj
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Sample size mismatch\. Input elements have inconsistent",
            ):
                basis_obj.compute_features(
                    np.linspace(0, 1, sample_size_a), np.linspace(0, 1, sample_size_b)
                )
        else:
            basis_obj.compute_features(
                np.linspace(0, 1, sample_size_a), np.linspace(0, 1, sample_size_b)
            )

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_pynapple_support_compute_features(
        self, basis_a, basis_b, n_basis_a, n_basis_b, sample_size
    ):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        basis_prod = self.instantiate_basis(
            n_basis_a, basis_a
        ) * self.instantiate_basis(n_basis_b, basis_b)
        out = basis_prod.compute_features(*([inp] * basis_prod._n_input_dimensionality))
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    def test_call_input_num(
        self, n_basis_a, n_basis_b, basis_a, basis_b, num_input, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch"
            )
        with expectation:
            basis_obj(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_input_shape(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        inp,
        mode,
        window_size,
        expectation,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        with expectation:
            basis_obj(*([inp] * basis_obj._n_input_dimensionality))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_axis(
        self, n_basis_a, n_basis_b, basis_a, basis_b, time_axis_shape, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        inp = [np.linspace(0, 1, time_axis_shape)] * basis_obj._n_input_dimensionality
        assert basis_obj(*inp).shape[0] == time_axis_shape

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_nan(self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size):
        if (
            basis_a == basis.OrthExponentialBasis
            or basis_b == basis.OrthExponentialBasis
        ):
            return
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        inp = [np.linspace(0, 1, 10)] * basis_obj._n_input_dimensionality
        for x in inp:
            x[3] = np.nan
        assert all(np.isnan(basis_obj(*inp)[3]))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_equivalent_in_conv(self, n_basis_a, n_basis_b, basis_a, basis_b):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="eval", window_size=None
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="eval", window_size=None
        )
        bas_eva = basis_a_obj * basis_b_obj

        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="conv", window_size=8
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="conv", window_size=8
        )
        bas_con = basis_a_obj * basis_b_obj

        x = [np.linspace(0, 1, 10)] * bas_con._n_input_dimensionality
        assert np.all(bas_con(*x) == bas_eva(*x))

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_pynapple_support(
        self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        x = np.linspace(0, 1, 10)
        x_nap = [nap.Tsd(t=np.arange(10), d=x)] * bas._n_input_dimensionality
        x = [x] * bas._n_input_dimensionality
        y = bas(*x)
        y_nap = bas(*x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap[0].t)

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [6, 7])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_basis_number(
        self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
        assert bas(*x).shape[1] == basis_a_obj.n_basis_funcs * basis_b_obj.n_basis_funcs

    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_non_empty(
        self, n_basis_a, n_basis_b, basis_a, basis_b, mode, window_size
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        with pytest.raises(ValueError, match="All sample provided must"):
            bas(*([np.array([])] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
            (0.1, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("mode, window_size", [("eval", None), ("conv", 3)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
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
        mode,
        window_size,
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
            n_basis_a, basis_a, mode=mode, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode=mode, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        with expectation:
            bas(*([np.linspace(mn, mx, 10)] * bas._n_input_dimensionality))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_fit_kernel(self, n_basis_a, n_basis_b, basis_a, basis_b):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="conv", window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="conv", window_size=10
        )
        bas = basis_a_obj * basis_b_obj
        bas._set_kernel(None)

        def check_kernel(basis_obj):
            has_kern = []
            if hasattr(basis_obj, "_basis1"):
                has_kern += check_kernel(basis_obj._basis1)
                has_kern += check_kernel(basis_obj._basis2)
            else:
                has_kern += [
                    basis_obj.kernel_ is not None if basis_obj.mode == "conv" else True
                ]
            return has_kern

        assert all(check_kernel(bas))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_transform_fails(self, n_basis_a, n_basis_b, basis_a, basis_b):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, mode="conv", window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, mode="conv", window_size=10
        )
        bas = basis_a_obj * basis_b_obj
        with pytest.raises(
            ValueError, match="You must call `_set_kernel` before `_compute_features`"
        ):
            x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
            bas._compute_features(*x)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_output_features(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(11, mode="conv", window_size=10)
        bas_add = bas1 * bas2
        assert bas_add.n_output_features is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_output_features == (n_basis_input1 * 10 * n_basis_input2 * 11)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_basis_input(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(10, mode="conv", window_size=10)
        bas_add = bas1 * bas2
        assert bas_add.n_basis_input is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_basis_input == (n_basis_input1, n_basis_input2)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (3, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (4, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(10, mode="conv", window_size=10)
        bas = bas1 * bas2
        x = np.random.randn(20, 2), np.random.randn(20, 3)
        bas.compute_features(*x)
        with expectation:
            bas.compute_features(np.random.randn(30, 2), np.random.randn(30, n_input))

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_n_basis_input(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineBasisLinear(10, mode="conv", window_size=10)
        bas2 = basis.BSplineBasis(10, mode="conv", window_size=10)
        bas_prod = bas1 * bas2
        bas_prod.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_prod.n_basis_input == (n_basis_input1, n_basis_input2)


@pytest.mark.parametrize(
    "exponent", [-1, 0, 0.5, basis.EvalRaisedCosineLog(4), 1, 2, 3]
)
@pytest.mark.parametrize("basis_class", list_all_basis_classes())
def test_power_of_basis(exponent, basis_class):
    """Test if the power behaves as expected."""
    raise_exception_type = not type(exponent) is int

    if not raise_exception_type:
        raise_exception_value = exponent <= 0
    else:
        raise_exception_value = False

    basis_obj = CombinedBasis.instantiate_basis(5, basis_class)

    if raise_exception_type:
        with pytest.raises(TypeError, match=r"Exponent should be an integer\!"):
            basis_obj**exponent
    elif raise_exception_value:
        with pytest.raises(
            ValueError, match=r"Exponent should be a non-negative integer\!"
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

        assert np.allclose(
            eval_pow,
            basis_obj.compute_features(*[samples] * basis_obj._n_input_dimensionality),
        )


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_basis_to_transformer(basis_cls):
    n_basis_funcs = 5
    bas = basis_cls(n_basis_funcs)

    trans_bas = bas.to_transformer()

    assert isinstance(trans_bas, basis.TransformerBasis)

    # check that things like n_basis_funcs are the same as the original basis
    for k in bas.__dict__.keys():
        assert getattr(bas, k) == getattr(trans_bas, k)


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformer_has_the_same_public_attributes_as_basis(basis_cls):
    n_basis_funcs = 5
    bas = basis_cls(n_basis_funcs)

    public_attrs_basis = {attr for attr in dir(bas) if not attr.startswith("_")}
    public_attrs_transformerbasis = {
        attr for attr in dir(bas.to_transformer()) if not attr.startswith("_")
    }

    assert public_attrs_transformerbasis - public_attrs_basis == {
        "fit",
        "fit_transform",
        "transform",
    }

    assert public_attrs_basis - public_attrs_transformerbasis == set()


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_to_transformer_and_constructor_are_equivalent(basis_cls):
    n_basis_funcs = 5
    bas = basis_cls(n_basis_funcs)

    trans_bas_a = bas.to_transformer()
    trans_bas_b = basis.TransformerBasis(bas)

    # they both just have a _basis
    assert (
        list(trans_bas_a.__dict__.keys())
        == list(trans_bas_b.__dict__.keys())
        == ["_basis"]
    )
    # and those bases are the same
    assert trans_bas_a._basis.__dict__ == trans_bas_b._basis.__dict__


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_basis_to_transformer_makes_a_copy(basis_cls):
    bas_a = basis_cls(5)
    trans_bas_a = bas_a.to_transformer()

    # changing an attribute in bas should not change trans_bas
    bas_a.n_basis_funcs = 10
    assert trans_bas_a.n_basis_funcs == 5

    # changing an attribute in the transformerbasis should not change the original
    bas_b = basis_cls(5)
    trans_bas_b = bas_b.to_transformer()
    trans_bas_b.n_basis_funcs = 100
    assert bas_b.n_basis_funcs == 5


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
@pytest.mark.parametrize("n_basis_funcs", [5, 10, 20])
def test_transformerbasis_getattr(basis_cls, n_basis_funcs):
    trans_basis = basis.TransformerBasis(basis_cls(n_basis_funcs))
    assert trans_basis.n_basis_funcs == n_basis_funcs


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
@pytest.mark.parametrize("n_basis_funcs_init", [5])
@pytest.mark.parametrize("n_basis_funcs_new", [6, 10, 20])
def test_transformerbasis_set_params(basis_cls, n_basis_funcs_init, n_basis_funcs_new):
    trans_basis = basis.TransformerBasis(basis_cls(n_basis_funcs_init))
    trans_basis.set_params(n_basis_funcs=n_basis_funcs_new)

    assert trans_basis.n_basis_funcs == n_basis_funcs_new
    assert trans_basis._basis.n_basis_funcs == n_basis_funcs_new


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_setattr_basis(basis_cls):
    # setting the _basis attribute should change it
    trans_bas = basis.TransformerBasis(basis_cls(10))
    trans_bas._basis = basis_cls(20)

    assert trans_bas.n_basis_funcs == 20
    assert trans_bas._basis.n_basis_funcs == 20
    assert isinstance(trans_bas._basis, basis_cls)


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_setattr_basis_attribute(basis_cls):
    # setting an attribute that is an attribute of the underlying _basis
    # should propagate setting it on _basis itself
    trans_bas = basis.TransformerBasis(basis_cls(10))
    trans_bas.n_basis_funcs = 20

    assert trans_bas.n_basis_funcs == 20
    assert trans_bas._basis.n_basis_funcs == 20
    assert isinstance(trans_bas._basis, basis_cls)


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_copy_basis_on_contsruct(basis_cls):
    # modifying the transformerbasis's attributes shouldn't
    # touch the original basis that was used to create it
    orig_bas = basis_cls(10)
    trans_bas = basis.TransformerBasis(orig_bas)
    trans_bas.n_basis_funcs = 20

    assert orig_bas.n_basis_funcs == 10
    assert trans_bas._basis.n_basis_funcs == 20
    assert trans_bas._basis.n_basis_funcs == 20
    assert isinstance(trans_bas._basis, basis_cls)


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_setattr_illegal_attribute(basis_cls):
    # changing an attribute that is not _basis or an attribute of _basis
    # is not allowed
    trans_bas = basis.TransformerBasis(basis_cls(10))

    with pytest.raises(
        ValueError,
        match="Only setting _basis or existing attributes of _basis is allowed.",
    ):
        trans_bas.random_attr = "random value"


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_addition(basis_cls):
    n_basis_funcs_a = 5
    n_basis_funcs_b = n_basis_funcs_a * 2
    trans_bas_a = basis.TransformerBasis(basis_cls(n_basis_funcs_a))
    trans_bas_b = basis.TransformerBasis(basis_cls(n_basis_funcs_b))
    trans_bas_sum = trans_bas_a + trans_bas_b
    assert isinstance(trans_bas_sum, basis.TransformerBasis)
    assert isinstance(trans_bas_sum._basis, AdditiveBasis)
    assert (
        trans_bas_sum.n_basis_funcs
        == trans_bas_a.n_basis_funcs + trans_bas_b.n_basis_funcs
    )
    assert (
        trans_bas_sum._n_input_dimensionality
        == trans_bas_a._n_input_dimensionality + trans_bas_b._n_input_dimensionality
    )
    assert trans_bas_sum._basis1.n_basis_funcs == n_basis_funcs_a
    assert trans_bas_sum._basis2.n_basis_funcs == n_basis_funcs_b


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_multiplication(basis_cls):
    n_basis_funcs_a = 5
    n_basis_funcs_b = n_basis_funcs_a * 2
    trans_bas_a = basis.TransformerBasis(basis_cls(n_basis_funcs_a))
    trans_bas_b = basis.TransformerBasis(basis_cls(n_basis_funcs_b))
    trans_bas_prod = trans_bas_a * trans_bas_b
    assert isinstance(trans_bas_prod, basis.TransformerBasis)
    assert isinstance(trans_bas_prod._basis, MultiplicativeBasis)
    assert (
        trans_bas_prod.n_basis_funcs
        == trans_bas_a.n_basis_funcs * trans_bas_b.n_basis_funcs
    )
    assert (
        trans_bas_prod._n_input_dimensionality
        == trans_bas_a._n_input_dimensionality + trans_bas_b._n_input_dimensionality
    )
    assert trans_bas_prod._basis1.n_basis_funcs == n_basis_funcs_a
    assert trans_bas_prod._basis2.n_basis_funcs == n_basis_funcs_b


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
@pytest.mark.parametrize(
    "exponent, error_type, error_message",
    [
        (2, does_not_raise, None),
        (5, does_not_raise, None),
        (0.5, TypeError, "Exponent should be an integer"),
        (-1, ValueError, "Exponent should be a non-negative integer"),
    ],
)
def test_transformerbasis_exponentiation(
    basis_cls, exponent: int, error_type, error_message
):
    trans_bas = basis.TransformerBasis(basis_cls(5))

    if not isinstance(exponent, int):
        with pytest.raises(error_type, match=error_message):
            trans_bas_exp = trans_bas**exponent
            assert isinstance(trans_bas_exp, basis.TransformerBasis)
            assert isinstance(trans_bas_exp._basis, MultiplicativeBasis)


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_dir(basis_cls):
    trans_bas = basis.TransformerBasis(basis_cls(5))
    for attr_name in (
        "fit",
        "transform",
        "fit_transform",
        "n_basis_funcs",
        "mode",
        "window_size",
    ):
        assert attr_name in dir(trans_bas)


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_transformerbasis_sk_clone_kernel_noned(basis_cls):
    orig_bas = basis_cls(10, mode="conv", window_size=5)
    trans_bas = basis.TransformerBasis(orig_bas)

    # kernel should be saved in the object after fit
    trans_bas.fit(np.random.randn(100, 20))
    assert isinstance(trans_bas.kernel_, np.ndarray)

    # cloning should set kernel_ to None
    trans_bas_clone = sk_clone(trans_bas)

    # the original object should still have kernel_
    assert isinstance(trans_bas.kernel_, np.ndarray)
    # but the clone should not have one
    assert trans_bas_clone.kernel_ is None


@pytest.mark.parametrize(
    "basis_cls",
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
@pytest.mark.parametrize("n_basis_funcs", [5])
def test_transformerbasis_pickle(tmpdir, basis_cls, n_basis_funcs):
    # the test that tries cross-validation with n_jobs = 2 already should test this
    trans_bas = basis.TransformerBasis(basis_cls(n_basis_funcs))
    filepath = tmpdir / "transformerbasis.pickle"
    with open(filepath, "wb") as f:
        pickle.dump(trans_bas, f)
    with open(filepath, "rb") as f:
        trans_bas2 = pickle.load(f)

    assert isinstance(trans_bas2, basis.TransformerBasis)
    assert trans_bas2.n_basis_funcs == n_basis_funcs


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
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
        AdditiveBasis,
        MultiplicativeBasis,
    ],
)
def test_multi_epoch_pynapple_basis(
    basis_cls, tsd, window_size, shift, predictor_causality, nan_index
):
    """Test nan location in multi-epoch pynapple tsd."""
    if basis_cls == AdditiveBasis:
        bas = basis.BSplineBasis(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )
        bas = bas + basis.RaisedCosineBasisLinear(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )
    elif basis_cls == MultiplicativeBasis:
        bas = basis.RaisedCosineBasisLog(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )
        bas = basis.EvalMSpline(3) * bas
    else:
        bas = basis_cls(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
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
    [
        basis.EvalMSpline,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
        AdditiveBasis,
        MultiplicativeBasis,
    ],
)
def test_multi_epoch_pynapple_basis_transformer(
    basis_cls, tsd, window_size, shift, predictor_causality, nan_index
):
    """Test nan location in multi-epoch pynapple tsd."""
    if basis_cls == AdditiveBasis:
        bas = basis.BSplineBasis(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )
        bas = bas + basis.RaisedCosineBasisLinear(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )
    elif basis_cls == MultiplicativeBasis:
        bas = basis.RaisedCosineBasisLog(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )
        bas = basis.EvalMSpline(3) * bas
    else:
        bas = basis_cls(
            5,
            mode="conv",
            window_size=window_size,
            predictor_causality=predictor_causality,
            shift=shift,
        )

    n_input = bas._n_input_dimensionality

    # pass through transformer
    bas = basis.TransformerBasis(bas)

    # concat input
    X = pynapple_concatenate_numpy([tsd[:, None]] * n_input, axis=1)

    # run convolutions
    res = bas.fit_transform(X)

    # check nans
    nan_index = np.sort(nan_index)
    times_nan_found = res[np.isnan(res.d[:, 0])].t
    assert len(times_nan_found) == len(nan_index)
    assert np.all(times_nan_found == np.array(nan_index))
    idx_nan = [np.where(res.t == k)[0][0] for k in nan_index]
    assert np.all(np.isnan(res.d[idx_nan]))


@pytest.mark.parametrize(
    "bas1, bas2, bas3",
    list(
        itertools.product(
            *[tuple((getattr(basis, basis_name) for basis_name in dir(basis)))] * 3
        )
    ),
)
@pytest.mark.parametrize(
    "mode1, mode2, mode3",
    list(itertools.product(["eval", "conv"], ["eval", "conv"], ["eval", "conv"])),
)
@pytest.mark.parametrize(
    "operator1, operator2, compute_slice",
    [
        (
            "__add__",
            "__add__",
            lambda bas1, bas2, bas3: {
                "1": slice(0, bas1._n_basis_input[0] * bas1.n_basis_funcs),
                "2": slice(
                    bas1._n_basis_input[0] * bas1.n_basis_funcs,
                    bas1._n_basis_input[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input[0] * bas2.n_basis_funcs,
                ),
                "3": slice(
                    bas1._n_basis_input[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input[0] * bas2.n_basis_funcs,
                    bas1._n_basis_input[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input[0] * bas2.n_basis_funcs
                    + bas3._n_basis_input[0] * bas3.n_basis_funcs,
                ),
            },
        ),
        (
            "__add__",
            "__mul__",
            lambda bas1, bas2, bas3: {
                "1": slice(0, bas1._n_basis_input[0] * bas1.n_basis_funcs),
                "(2 * 3)": slice(
                    bas1._n_basis_input[0] * bas1.n_basis_funcs,
                    bas1._n_basis_input[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input[0]
                    * bas2.n_basis_funcs
                    * bas3._n_basis_input[0]
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
                    bas1._n_basis_input[0]
                    * bas1.n_basis_funcs
                    * (
                        bas2._n_basis_input[0] * bas2.n_basis_funcs
                        + bas3._n_basis_input[0] * bas3.n_basis_funcs
                    ),
                ),
            },
        ),
        (
            "__mul__",
            "__mul__",
            lambda bas1, bas2, bas3: {
                "(1 * (2 * 3))": slice(
                    0,
                    bas1._n_basis_input[0]
                    * bas1.n_basis_funcs
                    * bas2._n_basis_input[0]
                    * bas2.n_basis_funcs
                    * bas3._n_basis_input[0]
                    * bas3.n_basis_funcs,
                ),
            },
        ),
    ],
)
def test__get_splitter(
    mode1, mode2, mode3, bas1, bas2, bas3, operator1, operator2, compute_slice
):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2, bas3]
    ):
        return
    # define the basis
    n_basis = [5, 6, 7]
    n_input_basis = [1, 2, 3]
    extra_kwargs = (
        {"decay_rates": np.arange(1, n_basis[0] + 1), "window_size": 5},
        {"decay_rates": np.arange(1, n_basis[1] + 1), "window_size": 5},
        {"decay_rates": np.arange(1, n_basis[2] + 1), "window_size": 5},
    )
    for i, val in enumerate(
        zip([bas1, bas2, bas3], [mode1, mode2, mode3], extra_kwargs)
    ):
        bas, mode, kwrgs = val
        if bas != basis.OrthExponentialBasis:
            kwrgs.pop("decay_rates")
        if mode == "eval":
            n_input_basis[i] = 1
            kwrgs.pop("window_size")

    bas1_instance = bas1(
        n_basis[0],
        mode=mode1,
        **extra_kwargs[0],
        label="1",
    )
    bas2_instance = bas2(
        n_basis[1],
        mode=mode2,
        **extra_kwargs[1],
        label="2",
    )
    bas3_instance = bas3(
        n_basis[2],
        mode=mode3,
        **extra_kwargs[2],
        label="3",
    )

    func1 = getattr(bas1_instance, operator1)
    func2 = getattr(bas2_instance, operator2)
    bas23 = func2(bas3_instance)
    bas123 = func1(bas23)
    inps = [np.zeros((1, n)) if n > 1 else np.zeros((1,)) for n in n_input_basis]
    bas123._set_num_output_features(*inps)
    splitter_dict, _ = bas123._get_feature_slicing(split_by_input=False)
    exp_slices = compute_slice(bas1_instance, bas2_instance, bas3_instance)
    assert exp_slices == splitter_dict


@pytest.mark.parametrize(
    "bas1, bas2",
    list(
        itertools.product(
            *[tuple((getattr(basis, basis_name) for basis_name in dir(basis)))] * 2
        )
    ),
)
@pytest.mark.parametrize(
    "operator, n_input_basis_1, n_input_basis_2, compute_slice",
    [
        (
            "__add__",
            1,
            1,
            lambda bas1, bas2: {
                "1": slice(0, bas1._n_basis_input[0] * bas1.n_basis_funcs),
                "2": slice(
                    bas1._n_basis_input[0] * bas1.n_basis_funcs,
                    bas1._n_basis_input[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input[0] * bas2.n_basis_funcs,
                ),
            },
        ),
        (
            "__mul__",
            1,
            1,
            lambda bas1, bas2: {
                "(1 * 2)": slice(
                    0,
                    bas1._n_basis_input[0]
                    * bas1.n_basis_funcs
                    * bas2._n_basis_input[0]
                    * bas2.n_basis_funcs,
                )
            },
        ),
        (
            "__add__",
            2,
            1,
            lambda bas1, bas2: {
                "1": {
                    "0": slice(0, bas1.n_basis_funcs),
                    "1": slice(bas1.n_basis_funcs, 2 * bas1.n_basis_funcs),
                },
                "2": slice(
                    2 * bas1.n_basis_funcs, 2 * bas1.n_basis_funcs + bas2.n_basis_funcs
                ),
            },
        ),
        (
            "__mul__",
            2,
            1,
            lambda bas1, bas2: {
                "(1 * 2)": slice(
                    0, bas1._n_basis_input[0] * bas1.n_basis_funcs * bas2.n_basis_funcs
                )
            },
        ),
        (
            "__add__",
            1,
            2,
            lambda bas1, bas2: {
                "1": slice(0, bas1.n_basis_funcs),
                "2": {
                    "0": slice(
                        bas1.n_basis_funcs, bas1.n_basis_funcs + bas2.n_basis_funcs
                    ),
                    "1": slice(
                        bas1.n_basis_funcs + bas2.n_basis_funcs,
                        bas1.n_basis_funcs + 2 * bas2.n_basis_funcs,
                    ),
                },
            },
        ),
        (
            "__mul__",
            1,
            2,
            lambda bas1, bas2: {
                "(1 * 2)": slice(
                    0, bas2._n_basis_input[0] * bas1.n_basis_funcs * bas2.n_basis_funcs
                )
            },
        ),
        (
            "__add__",
            2,
            2,
            lambda bas1, bas2: {
                "1": {
                    "0": slice(0, bas1.n_basis_funcs),
                    "1": slice(bas1.n_basis_funcs, 2 * bas1.n_basis_funcs),
                },
                "2": {
                    "0": slice(
                        2 * bas1.n_basis_funcs,
                        2 * bas1.n_basis_funcs + bas2.n_basis_funcs,
                    ),
                    "1": slice(
                        2 * bas1.n_basis_funcs + bas2.n_basis_funcs,
                        2 * bas1.n_basis_funcs + 2 * bas2.n_basis_funcs,
                    ),
                },
            },
        ),
        (
            "__mul__",
            2,
            2,
            lambda bas1, bas2: {
                "(1 * 2)": slice(0, 2 * bas1.n_basis_funcs * 2 * bas2.n_basis_funcs)
            },
        ),
    ],
)
def test__get_splitter_split_by_input(
    bas1, bas2, operator, n_input_basis_1, n_input_basis_2, compute_slice
):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2]
    ):
        return
    # define the basis
    n_basis = [5, 6]
    mode = "conv"
    extra_kwargs = (
        {"decay_rates": np.arange(1, n_basis[0] + 1), "window_size": 5},
        {"decay_rates": np.arange(1, n_basis[1] + 1), "window_size": 5},
    )
    for i, val in enumerate(zip([bas1, bas2], extra_kwargs)):
        bas, kwrgs = val
        if bas != basis.OrthExponentialBasis:
            kwrgs.pop("decay_rates")

    bas1_instance = bas1(
        n_basis[0],
        mode=mode,
        **extra_kwargs[0],
        label="1",
    )
    bas2_instance = bas2(
        n_basis[1],
        mode=mode,
        **extra_kwargs[1],
        label="2",
    )

    func1 = getattr(bas1_instance, operator)
    bas12 = func1(bas2_instance)

    inps = [
        np.zeros((1, n)) if n > 1 else np.zeros((1,))
        for n in (n_input_basis_1, n_input_basis_2)
    ]
    bas12._set_num_output_features(*inps)
    splitter_dict, _ = bas12._get_feature_slicing()
    exp_slices = compute_slice(bas1_instance, bas2_instance)
    assert exp_slices == splitter_dict


@pytest.mark.parametrize(
    "bas1, bas2, bas3",
    list(
        itertools.product(
            *[tuple((getattr(basis, basis_name) for basis_name in dir(basis)))] * 3
        )
    ),
)
def test_duplicate_keys(bas1, bas2, bas3):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2, bas3]
    ):
        return

    extra_kwargs = (
        {"decay_rates": np.arange(1, 5 + 1)},
        {"decay_rates": np.arange(1, 5 + 1)},
        {"decay_rates": np.arange(1, 5 + 1)},
    )
    for bas, kwrgs in zip((bas1, bas2, bas3), extra_kwargs):
        if bas != basis.OrthExponentialBasis:
            kwrgs.pop("decay_rates")

    bas_obj = (
        bas1(5, **extra_kwargs[0], label="label")
        + bas2(5, **extra_kwargs[1], label="label")
        + bas3(5, **extra_kwargs[2], label="label")
    )
    inps = [np.zeros((1,)) for n in range(3)]
    bas_obj._set_num_output_features(*inps)
    slice_dict = bas_obj._get_feature_slicing()[0]
    assert tuple(slice_dict.keys()) == ("label", "label-1", "label-2")


@pytest.mark.parametrize(
    "bas1, bas2",
    list(
        itertools.product(
            *[tuple((getattr(basis, basis_name) for basis_name in dir(basis)))] * 2
        )
    ),
)
@pytest.mark.parametrize(
    "x, axis, expectation, exp_shapes",  # num output is 5*2 + 6*3 = 28
    [
        (np.ones((1, 28)), 1, does_not_raise(), [(1, 2, 5), (1, 3, 6)]),
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
def test_split_feature_axis(bas1, bas2, x, axis, expectation, exp_shapes):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2]
    ):
        return
    # define the basis
    n_basis = [5, 6]
    mode = "conv"
    extra_kwargs = (
        {"decay_rates": np.arange(1, n_basis[0] + 1), "window_size": 5},
        {"decay_rates": np.arange(1, n_basis[1] + 1), "window_size": 5},
    )
    for i, val in enumerate(zip([bas1, bas2], extra_kwargs)):
        bas, kwrgs = val
        if bas != basis.OrthExponentialBasis:
            kwrgs.pop("decay_rates")

    bas1_instance = bas1(
        n_basis[0],
        mode=mode,
        **extra_kwargs[0],
        label="1",
    )
    bas2_instance = bas2(
        n_basis[1],
        mode=mode,
        **extra_kwargs[1],
        label="2",
    )
    bas = bas1_instance + bas2_instance
    bas._set_num_output_features(np.zeros((1, 2)), np.zeros((1, 3)))
    with expectation:
        out = bas.split_by_feature(x, axis=axis)
        for i, itm in enumerate(out.items()):
            _, val = itm
            assert val.shape == exp_shapes[i]
