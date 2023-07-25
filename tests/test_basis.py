import abc

import numpy as np
import pytest
import utils_testing

import neurostatslib.basis as basis


# automatic define user accessible basis and check the methods


def test_all_basis_are_parametrized() -> None:
    """
    Check that all the basis initialization are tested by inspecting the basis module and make sure that all
    the non-abstract classes, except additive and multiplicative are listed in the params for TestInitAndEvaluate.
    If this test fails, it means that you need to add some newly implemented basis function.

    Returns
    -------
    None
    """
    cls = TestInitAndEvaluate
    for class_name, class_obj in utils_testing.get_non_abstract_classes(basis):
        print(f'\n-> Testing "{class_name}"')
        # if class_name in ["AdditiveBasis", "MultiplicativeBasis"]:
        #     continue
        for test_name in cls.params:
            implemented_class = {
                cls.params[test_name][cc]["pars"]["class_name"]
                for cc in range(len(cls.params[test_name]))
            }
            assert class_name in implemented_class, (
                f"{class_name} not in the init_basis_parameter_grid " f"fixture keys!"
            )


class BasisFuncsTesting(abc.ABC):
    """
    Abstract class for individual basis testing. This clarifies the requirement of a cls method,
    which is going to be used by the meta-test that checks that all the basis are tested.
    """

    @abc.abstractmethod
    def cls(self):
        pass


class TestRaisedCosineLogBasis(BasisFuncsTesting):
    """Test class for Raised Cosine"""

    cls = basis.RaisedCosineBasisLog

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [2, 10, 100]],
    )
    def test_evaluate_returns_expected_number_of_basis(self, args, sample_size):
        basis_obj = self.cls(**args)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != args["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimensiton of the evaluated basis."
                f"The number of basis is {args['n_basis_funcs']}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[0]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 100])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size
    ):
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimensiton of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs):
        raise_exception = n_basis_funcs < 2
        if raise_exception:
            with pytest.raises(ValueError):
                self.cls(n_basis_funcs=n_basis_funcs)
        else:
            self.cls(n_basis_funcs=n_basis_funcs)

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        raise_exception = (sample_range[0] < 0) | (sample_range[1] > 1)
        basis_obj = self.cls(n_basis_funcs=5)
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate(np.linspace(*sample_range, 100))
        else:
            basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = n_input != basis_obj._n_input_samples
        inputs = [np.linspace(0, 1, 20)] * n_input
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate(*inputs)
        else:
            basis_obj.evaluate(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size < 0
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size < 0
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[1] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        basis_obj = self.cls(n_basis_funcs=5)
        inputs = [10] * n_input
        raise_exception = n_input != basis_obj._n_input_samples
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate_on_grid(*inputs)
        else:
            basis_obj.evaluate_on_grid(*inputs)


class TestRaisedCosineLinearBasis(BasisFuncsTesting):
    """Test class for Raised Cosine"""

    cls = basis.RaisedCosineBasisLinear

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [1, 2, 10, 100]],
    )
    def test_evaluate_returns_expected_number_of_basis(self, args, sample_size):
        basis_obj = self.cls(**args)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[0] != args["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimensiton of the evaluated basis."
                f"The number of basis is {args['n_basis_funcs']}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[0]}",
            )
        return

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [2, 10, 100])
    def test_sample_size_of_evaluate_matches_that_of_input(
        self, n_basis_funcs, sample_size
    ):
        basis_obj = self.cls(n_basis_funcs=n_basis_funcs)
        eval_basis = basis_obj.evaluate(np.linspace(0, 1, sample_size))
        if eval_basis.shape[1] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimensiton of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    def test_minimum_number_of_basis_required_is_matched(self, n_basis_funcs):
        raise_exception = n_basis_funcs < 1
        if raise_exception:
            with pytest.raises(ValueError):
                self.cls(n_basis_funcs=n_basis_funcs)
        else:
            self.cls(n_basis_funcs=n_basis_funcs)

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_evaluate_requirements(self, sample_range: tuple):
        raise_exception = (sample_range[0] < 0) | (sample_range[1] > 1)
        basis_obj = self.cls(n_basis_funcs=5)
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate(np.linspace(*sample_range, 100))
        else:
            basis_obj.evaluate(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    def test_number_of_required_inputs_evaluate(self, n_input):
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = n_input != basis_obj._n_input_samples
        inputs = [np.linspace(0, 1, 20)] * n_input
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate(*inputs)
        else:
            basis_obj.evaluate(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_meshgrid_size(self, sample_size):
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size < 0
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    def test_evaluate_on_grid_basis_size(self, sample_size):
        basis_obj = self.cls(n_basis_funcs=5)
        raise_exception = sample_size < 0
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[1] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    def test_evaluate_on_grid_input_number(self, n_input):
        basis_obj = self.cls(n_basis_funcs=5)
        inputs = [10] * n_input
        raise_exception = n_input != basis_obj._n_input_samples
        if raise_exception:
            with pytest.raises(ValueError):
                basis_obj.evaluate_on_grid(*inputs)
        else:
            basis_obj.evaluate_on_grid(*inputs)


class TestInitAndEvaluate:
    test_input = [
        {
            "pars": {
                "class_name": name,
                "sample_size": sample_size,
                "args": {"n_basis_funcs": nbasis, "order": order},
            }
        }
        for name in ["MSplineBasis"]
        for sample_size in [50, 80, 100]
        for nbasis in range(6, 10)
        for order in range(1, 5)
    ]
    test_input += [
        {
            "pars": {
                "class_name": "RaisedCosineBasisLinear",
                "sample_size": sample_size,
                "args": {"n_basis_funcs": nbasis},
            }
        }
        for sample_size in [50, 80, 100]
        for nbasis in range(2, 10)
    ]
    test_input += [
        {
            "pars": {
                "class_name": "RaisedCosineBasisLog",
                "sample_size": sample_size,
                "args": {"n_basis_funcs": nbasis},
            }
        }
        for sample_size in [50, 80, 100]
        for nbasis in range(2, 10)
    ]
    test_input += [
        {
            "pars": {
                "class_name": "OrthExponentialBasis",
                "sample_size": sample_size,
                "args": {
                    "n_basis_funcs": nbasis,
                    "decay_rates": np.linspace(10, nbasis * 10, nbasis),
                },
            }
        }
        for sample_size in [50, 80, 100]
        for nbasis in range(6, 10)
    ]

    min_basis = []
    for spline in ["MSplineBasis"]:
        for order in [-1, 0, 1, 2, 3, 4]:
            for n_basis in [-1, 0, 1, 3, 10, 20]:
                if spline == "CyclicBSplineBasis":
                    raise_exception = (n_basis < max(order * 2 - 2, order + 2)) or (
                        order < 2
                    ) | n_basis <= 0
                elif spline == "BSplineBasis":
                    raise_exception = n_basis < order + 2 | n_basis > 0
                elif spline == "MSplineBasis":
                    raise_exception = (order < 1) | (n_basis < 1)

                min_basis.append(
                    {
                        "pars": {
                            "class_name": spline,
                            "args": {"order": order, "n_basis_funcs": n_basis},
                            "raise_exception": raise_exception,
                        }
                    }
                )

    for n_basis in [-1, 0, 1, 3, 10, 20]:
        min_basis.append(
            {
                "pars": {
                    "class_name": "RaisedCosineBasisLinear",
                    "args": {"n_basis_funcs": n_basis},
                    "raise_exception": n_basis < 1,
                }
            }
        )
        min_basis.append(
            {
                "pars": {
                    "class_name": "RaisedCosineBasisLog",
                    "args": {"n_basis_funcs": n_basis},
                    "raise_exception": n_basis < 2,
                }
            }
        )
        min_basis.append(
            {
                "pars": {
                    "class_name": "OrthExponentialBasis",
                    "args": {
                        "n_basis_funcs": n_basis,
                        "decay_rates": np.linspace(0, 1, max(1, n_basis)),
                    },
                    "raise_exception": n_basis < 1,
                }
            }
        )

    params = {
        "test_nbasis": test_input,
        "test_sample_size": test_input,
        "test_min_basis": min_basis,
    }

    def test_nbasis(self, pars: dict, capfd: pytest.fixture):
        """
        Test initialization and evaluation of basis classes:
        Check:
            - does evaluation works and returns an NDArray with the expected number of basis?

        Parameters:
        -----------
        - pars:
            A dictionary containing basis names as keys and their initialization arguments as values.
        - capfd
            pytest fixture for capturing stdout and stderr.

        Raises:
        -------
        - ValueError
            If the dimensions of the evaluated basis do not match the expected dimensions.

        Returns:
        - None
        """
        basis_name = pars["class_name"]
        sample_size = pars["sample_size"]
        basis_class = getattr(basis, basis_name)
        basis_instance = basis_class(**pars["args"])
        eval_basis = basis_instance.evaluate(np.linspace(0, 1, sample_size))
        # capfd.readouterr()
        if eval_basis.shape[0] != pars["args"]["n_basis_funcs"]:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimensiton of the evaluated basis."
                f"The number of basis is {pars['args']['n_basis_funcs']}",
                f"The first dimension of the evaluated basis is {eval_basis.shape[0]}",
            )

    def test_sample_size(self, pars: dict, capfd: pytest.fixture):
        """
        Test initialization and evaluation of basis classes:
        Check:
            - does evaluation works and returns an NDArray with the expected number of samples?

        Parameters:
        -----------
        - pars:
            A dictionary containing basis names as keys and their initialization arguments as values.
        - capfd
            pytest fixture for capturing stdout and stderr.

        Raises:
        -------
        - ValueError
            If the dimensions of the evaluated basis do not match the expected dimensions.

        Returns:
        - None
        """
        basis_name = pars["class_name"]
        sample_size = pars["sample_size"]
        basis_class = getattr(basis, basis_name)
        basis_instance = basis_class(**pars["args"])
        eval_basis = basis_instance.evaluate(np.linspace(0, 1, sample_size))
        # capfd.readouterr()
        if eval_basis.shape[1] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimensiton of the evaluated basis."
                f"The window size is {sample_size}",
                f"The second dimension of the evaluated basis is {eval_basis.shape[1]}",
            )

    def test_min_basis(self, pars):
        """
        Check that the expected minimum number of basis is appropriately matched and a ValueError exception is raised
        otherwise.

        Parameters
        ----------
        min_basis_funcs : pytest.fixture
            Fixture containing a dictionary with the following keys:
                'args': ndarray
                    The basis function initialization arguments.
                'raise_exception': bool
                    True if the argument would result in an exception being raised, False otherwise.

        Returns
        -------
        None
        """
        class_name = pars["class_name"]
        basis_obj = getattr(basis, class_name)

        # params that should not raise exception
        if not pars["raise_exception"]:
            basis_obj(**pars["args"])
        else:
            with pytest.raises(ValueError):
                basis_obj(**pars["args"])


@pytest.mark.parametrize("basis_type", ["add2", "mul2", "add3"])
def test_basis_sample_consistency_check(
    basis_sample_consistency_check: pytest.fixture,
    capfd: pytest.fixture,
    basis_type: str,
) -> None:
    """
    Check that the expected minimum number of basis is appropriately matched and a ValueError exception is raised
    otherwise.

    Parameters
    ----------
    min_basis_funcs
        Fixture containing a dictionary with the following keys:
            "args" : NDArray
                The basis function initialization arguments.
            "raise_exception" : bool
                True if the argument would result in an exception being raised, False otherwise.

    Returns
    -------
    None
    """
    pars = basis_sample_consistency_check[basis_type]
    basis_obj = pars["basis_obj"]
    n_input = pars["n_input"]

    inputs = [np.linspace(0, 1, 100 + k) for k in range(n_input)]
    with pytest.raises(ValueError):
        basis_obj.evaluate(*inputs)


# Use pytest.mark.parametrize to run the test for each basis separately.
@pytest.mark.parametrize(
    "class_name, delta_input",
    [
        (cname, deltai)
        for cname in [
            "MSplineBasis",
            "RaisedCosineBasisLinear",
            "RaisedCosineBasisLog",
            "OrthExponentialBasis",
            "add2",
            "mul2",
            "add3",
        ]
        for deltai in [0, 1, -1]
    ],
)
def test_basis_eval_checks(
    evaluate_basis_object: pytest.fixture,
    capfd: pytest.fixture,
    class_name: str,
    delta_input: list,
):
    """
    Test if the basis function object can be evaluated, and check that the appropriate exceptions are raised
    if the input does not conform with the requirements.

    Parameters
    ----------
    evaluate_basis_object
        Fixture containing a dictionary with the following keys:
            "basis_obj" : basis object
                The basis function object to test.
            "n_input" : int
                The number of input samples.

    capfd
        Fixture for capturing stdout and stderr.

    class_name : str
        The name of the basis class to be tested.

    Returns
    -------
    None
    """
    basis_obj = evaluate_basis_object[class_name]["basis_obj"]
    n_input = evaluate_basis_object[class_name]["n_input"]

    inputs = [np.linspace(0, 1, 20)] * n_input
    basis_obj.evaluate(*inputs)
    inputs = [20] * (n_input + delta_input)

    if delta_input == 0:
        basis_obj.evaluate_on_grid(*inputs)
    else:
        with pytest.raises(ValueError):
            inputs = [np.linspace(0, 1, 10)] * (
                n_input + delta_input
            )  # wrong number of inputs passed
            basis_obj.evaluate(*inputs)
