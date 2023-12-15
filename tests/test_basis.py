import abc
import inspect

import jax.numpy
import numpy as np
import pytest
import utils_testing

import nemos.basis as basis
from contextlib import nullcontext as does_not_raise

# automatic define user accessible basis and check the methods


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
    all_bases = {
        class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)
    }

    if all_bases != all_bases.intersection(tested_bases):
        raise ValueError(
            "Test should be implemented for each of the concrete classes in the basis module.\n"
            f"The following classes are not tested: {[bas.__qualname__ for bas in all_bases.difference(tested_bases)]}"
        )


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
    cls = basis.RaisedCosineBasisLog

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    def test_non_empty_samples(self, samples):
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5).evaluate(samples)
        else:
            self.cls(5).evaluate(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.evaluate(eval_input)

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [2, 10, 100]],
    )
    def test_evaluate_returns_expected_number_of_basis(self, args, sample_size):
        """
        Verifies the number of basis functions returned by the evaluate() method matches
        the expected number of basis functions.
        """
        basis_obj = self.cls(**args)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != args["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {args['n_basis_funcs']}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 100])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size
    ):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs):
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
                self.cls(n_basis_funcs=n_basis_funcs)
        else:
            self.cls(n_basis_funcs=n_basis_funcs)

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        """
        Ensures that the evaluate() method correctly handles sample range inputs that are outside of its required bounds (0, 1).
        """
        raise_exception = (sample_range[0] < 0) | (sample_range[1] > 1)
        basis_obj = self.cls(n_basis_funcs=5)
        if raise_exception:
            with pytest.raises(
                ValueError, match="Sample points for RaisedCosine basis must lie in"
            ):
                basis_obj.evaluate(np.linspace(*sample_range, 100))
        else:
            basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        """
        Confirms that the evaluate() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="evaluate\(\) missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate(*inputs)

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
                match="evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
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
        assert np.all(np.diff(corr) < 0), "As time scales increases, deviation from linearity should increase!"


class TestRaisedCosineLinearBasis(BasisFuncsTesting):
    cls = basis.RaisedCosineBasisLinear

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    def test_non_empty_samples(self, samples):
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5).evaluate(samples)
        else:
            self.cls(5).evaluate(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.evaluate(eval_input)

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [2, 10, 100]],
    )
    def test_evaluate_returns_expected_number_of_basis(self, args, sample_size):
        """
        Verifies that the evaluate() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(**args)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != args["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {args['n_basis_funcs']}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 100])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size
    ):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs):
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
                self.cls(n_basis_funcs=n_basis_funcs)
        else:
            self.cls(n_basis_funcs=n_basis_funcs)

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        """
        Ensures that the evaluate() method correctly handles sample range inputs that are outside of its required bounds (0, 1).
        """
        raise_exception = (sample_range[0] < 0) | (sample_range[1] > 1)
        basis_obj = self.cls(n_basis_funcs=5)
        if raise_exception:
            with pytest.raises(
                ValueError, match="Sample points for RaisedCosine basis must lie in"
            ):
                basis_obj.evaluate(np.linspace(*sample_range, 100))
        else:
            basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        """
        Confirms that the evaluate() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="evaluate\(\) missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate(*inputs)

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
                match="evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
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


class TestMSplineBasis(BasisFuncsTesting):
    cls = basis.MSplineBasis

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    def test_non_empty_samples(self, samples):
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5).evaluate(samples)
        else:
            self.cls(5).evaluate(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.evaluate(eval_input)

    @pytest.mark.parametrize("n_basis_funcs", [6, 8, 10])
    @pytest.mark.parametrize("order", range(1, 6))
    def test_evaluate_returns_expected_number_of_basis(
        self, n_basis_funcs: int, order: int
    ):
        """
        Verifies that the evaluate() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, 100))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [4, 10, 100])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size, order
    ):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2, 3, 4, 5])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs, order):
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
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.evaluate(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.evaluate(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        """
        Verifies that the evaluate() method can handle input range.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        """
        Confirms that the evaluate() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="evaluate\(\) missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate(*inputs)

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
                match="evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)


class TestOrthExponentialBasis(BasisFuncsTesting):
    cls = basis.OrthExponentialBasis

    # this class requires at leas `n_basis` samples
    @pytest.mark.parametrize("samples", [[], [0] * 6, [0] * 7])
    def test_non_empty_samples(self, samples):
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5, decay_rates=np.arange(1, 6)).evaluate(samples)
        else:
            self.cls(5, decay_rates=np.arange(1, 6)).evaluate(samples)

    @pytest.mark.parametrize(
        "eval_input",
        [0, [0] * 6, (0,) * 6, np.array([0] * 6), jax.numpy.array([0] * 6)],
    )
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        if isinstance(eval_input, int):
            # OrthExponentialBasis is special -- cannot accept int input
            with pytest.raises(
                ValueError,
                match="OrthExponentialBasis requires at least as many samples",
            ):
                basis_obj.evaluate(eval_input)
        else:
            basis_obj.evaluate(eval_input)

    @pytest.mark.parametrize("n_basis_funcs", [1, 2, 4, 8])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    def test_evaluate_returns_expected_number_of_basis(
        self, n_basis_funcs, sample_size
    ):
        """Tests whether the evaluate method returns the expected number of basis functions."""
        decay_rates = np.arange(1, 1 + n_basis_funcs)
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 20])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size
    ):
        """Tests whether the sample size of the evaluated result matches that of the input."""
        decay_rates = np.arange(1, 1 + n_basis_funcs)
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs):
        """Tests whether the class instance has a minimum number of basis functions."""
        raise_exception = n_basis_funcs < 1
        decay_rates = np.arange(1, 1 + n_basis_funcs)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls.__name__} "
                r"requires >= 1 basis elements\.",
            ):
                self.cls(n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)
        else:
            self.cls(n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        """
        Tests whether the evaluate method correctly processes the given sample range.
        Raises an exception for negative samples
        """
        raise_exception = sample_range[0] < 0
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls.__name__} requires positive samples\. "
                r"Negative values provided instead\!",
            ):
                basis_obj.evaluate(np.linspace(*sample_range, 100))
        else:
            basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        """Tests whether the evaluate method correctly processes the number of required inputs."""
        basis_obj = self.cls(n_basis_funcs=5, decay_rates=np.arange(1, 6))
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="evaluate\(\) missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 2, 3, 4, 5, 6, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        """Tests whether the evaluate_on_grid method correctly outputs the grid mesh size."""
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
                match="evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
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


class TestBSplineBasis(BasisFuncsTesting):
    cls = basis.BSplineBasis

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    def test_non_empty_samples(self, samples):
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5).evaluate(samples)
        else:
            self.cls(5).evaluate(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.evaluate(eval_input)

    @pytest.mark.parametrize("n_basis_funcs", [6, 8, 10])
    @pytest.mark.parametrize("order", range(1, 6))
    def test_evaluate_returns_expected_number_of_basis(
        self, n_basis_funcs: int, order: int
    ):
        """
        Verifies that the evaluate() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, 100))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [4, 10, 100])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size, order
    ):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs, order):
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
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.evaluate(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.evaluate(np.linspace(0, 1, 10))

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
                basis_obj.evaluate(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.evaluate(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        """
        Verifies that the evaluate() method can handle input range.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        """
        Confirms that the evaluate() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="evaluate\(\) missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate(*inputs)

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
                rf"All sample counts provided must be greater",
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
                match="evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)


class TestCyclicBSplineBasis(BasisFuncsTesting):
    cls = basis.CyclicBSplineBasis

    @pytest.mark.parametrize("samples", [[], [0], [0, 0]])
    def test_non_empty_samples(self, samples):
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                self.cls(5).evaluate(samples)
        else:
            self.cls(5).evaluate(samples)

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=5)
        basis_obj.evaluate(eval_input)

    @pytest.mark.parametrize("n_basis_funcs", [8, 10])
    @pytest.mark.parametrize("order", range(2, 6))
    def test_evaluate_returns_expected_number_of_basis(
        self, n_basis_funcs: int, order: int
    ):
        """
        Verifies that the evaluate() method returns the expected number of basis functions.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, 100))
        if eval_basis.shape[1] != n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {n_basis_funcs}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[0]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [8, 10, 100])
    @pytest.mark.parametrize("order", [2, 3])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size, order
    ):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs, order):
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
                basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.evaluate(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.evaluate(np.linspace(0, 1, 10))

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
                basis_obj.evaluate(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.evaluate(np.linspace(0, 1, 10))

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
                basis_obj.evaluate(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls(n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.evaluate(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        """
        Verifies that the evaluate() method can handle input range.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        """
        Confirms that the evaluate() method correctly handles the number of input samples that are provided.
        """
        basis_obj = self.cls(n_basis_funcs=5, order=3)
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="evaluate\(\) missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate(*inputs)

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
                match="evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match="evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)


class CombinedBasis(BasisFuncsTesting):
    """
    This class is used to run tests on combination operations (e.g., addition, multiplication) among Basis functions.

    Properties:
    - cls: Class (default = None)
    """

    cls = None

    @staticmethod
    def instantiate_basis(n_basis, basis_class):
        """Instantiate and return two basis of the type specified."""
        if basis_class == basis.MSplineBasis:
            basis_obj = basis_class(n_basis_funcs=n_basis, order=4)
        elif basis_class in [basis.RaisedCosineBasisLinear, basis.RaisedCosineBasisLog]:
            basis_obj = basis_class(n_basis_funcs=n_basis)
        elif basis_class == basis.OrthExponentialBasis:
            basis_obj = basis_class(
                n_basis_funcs=n_basis, decay_rates=np.arange(1, 1 + n_basis)
            )
        elif basis_class == basis.BSplineBasis:
            basis_obj = basis_class(n_basis_funcs=n_basis, order=3)
        elif basis_class == basis.CyclicBSplineBasis:
            basis_obj = basis_class(n_basis_funcs=n_basis, order=3)
        elif basis_class == basis.AdditiveBasis:
            b1 = basis.MSplineBasis(n_basis_funcs=n_basis, order=2)
            b2 = basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis + 1)
            basis_obj = b1 + b2
        elif basis_class == basis.MultiplicativeBasis:
            b1 = basis.MSplineBasis(n_basis_funcs=n_basis, order=2)
            b2 = basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis + 1)
            basis_obj = b1 * b2
        else:
            raise ValueError(
                f"Test for basis addition not implemented for basis of type {basis_class}!"
            )
        return basis_obj


class TestAdditiveBasis(CombinedBasis):
    cls = basis.AdditiveBasis

    @pytest.mark.parametrize(
        "samples", [[[0], []], [[], [0]], [[0], [0]], [[0, 0], [0, 0]]]
    )
    def test_non_empty_samples(self, samples):
        basis_obj = basis.MSplineBasis(5) + basis.MSplineBasis(5)
        if any(tuple(len(s) == 0 for s in samples)):
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                basis_obj.evaluate(*samples)
        else:
            basis_obj.evaluate(*samples)

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
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = basis.MSplineBasis(5) + basis.MSplineBasis(5)
        basis_obj.evaluate(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    def test_evaluate_returns_expected_number_of_basis(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b
    ):
        """
        Test whether the evaluation of the `AdditiveBasis` results in a number of basis
        that is the sum of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)

        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.evaluate(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[1] != basis_a_obj.n_basis_funcs + basis_b_obj.n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {n_basis_a + n_basis_b}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b
    ):
        """
        Test whether the output sample size from the `AdditiveBasis` evaluate function matches the input sample size.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.evaluate(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    def test_number_of_required_inputs_evaluate(
        self, n_input, n_basis_a, n_basis_b, basis_a, basis_b
    ):
        """
        Test whether the number of required inputs for the `evaluate` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
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
            basis_obj.evaluate(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
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
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
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
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
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


class TestMultiplicativeBasis(CombinedBasis):
    cls = basis.MultiplicativeBasis

    @pytest.mark.parametrize(
        "samples", [[[0], []], [[], [0]], [[0], [0]], [[0, 0], [0, 0]]]
    )
    def test_non_empty_samples(self, samples):
        basis_obj = basis.MSplineBasis(5) * basis.MSplineBasis(5)
        if any(tuple(len(s) == 0 for s in samples)):
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                basis_obj.evaluate(*samples)
        else:
            basis_obj.evaluate(*samples)

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
    def test_evaluate_input(self, eval_input):
        """
        Checks that the sample size of the output from the evaluate() method matches the input sample size.
        """
        basis_obj = basis.MSplineBasis(5) * basis.MSplineBasis(5)
        basis_obj.evaluate(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    def test_evaluate_returns_expected_number_of_basis(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b
    ):
        """
        Test whether the evaluation of the `MultiplicativeBasis` results in a number of basis
        that is the product of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)

        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.evaluate(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[1] != basis_a_obj.n_basis_funcs * basis_b_obj.n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the evaluated basis."
                f"The number of basis is {n_basis_a * n_basis_b}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [12, 30, 35])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_a, n_basis_b, sample_size, basis_a, basis_b
    ):
        """
        Test whether the output sample size from the `MultiplicativeBasis` evaluate function matches the input sample size.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.evaluate(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    def test_number_of_required_inputs_evaluate(
        self, n_input, n_basis_a, n_basis_b, basis_a, basis_b
    ):
        """
        Test whether the number of required inputs for the `evaluate` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
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
            basis_obj.evaluate(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
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
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
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
    @pytest.mark.parametrize(
        "basis_a",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
    @pytest.mark.parametrize(
        "basis_b",
        [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
    )
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

    @pytest.mark.parametrize("basis_a", [basis.MSplineBasis])
    @pytest.mark.parametrize("basis_b", [basis.OrthExponentialBasis])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    @pytest.mark.parametrize("sample_size_a", [11, 12])
    @pytest.mark.parametrize("sample_size_b", [11, 12])
    def test_inconsistent_sample_sizes(
        self, basis_a, basis_b, n_basis_a, n_basis_b, sample_size_a, sample_size_b
    ):
        """Test that the inputs of inconsistent sample sizes result in an exception when evaluate is called"""
        raise_exception = sample_size_a != sample_size_b
        basis_a_obj = self.instantiate_basis(n_basis_a, basis_a)
        basis_b_obj = self.instantiate_basis(n_basis_b, basis_b)
        basis_obj = basis_a_obj * basis_b_obj
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Sample size mismatch\. Input elements have inconsistent",
            ):
                basis_obj.evaluate(
                    np.linspace(0, 1, sample_size_a), np.linspace(0, 1, sample_size_b)
                )
        else:
            basis_obj.evaluate(
                np.linspace(0, 1, sample_size_a), np.linspace(0, 1, sample_size_b)
            )


@pytest.mark.parametrize(
    "exponent", [-1, 0, 0.5, basis.RaisedCosineBasisLog(4), 1, 2, 3]
)
@pytest.mark.parametrize(
    "basis_class",
    [class_obj for _, class_obj in utils_testing.get_non_abstract_classes(basis)],
)
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
        eval_pow = basis_pow.evaluate(*[samples] * basis_pow._n_input_dimensionality)

        if exponent == 2:
            basis_obj = basis_obj * basis_obj
        elif exponent == 3:
            basis_obj = basis_obj * basis_obj * basis_obj

        assert np.allclose(
            eval_pow, basis_obj.evaluate(*[samples] * basis_obj._n_input_dimensionality)
        )
