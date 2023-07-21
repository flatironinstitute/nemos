import pytest
import neurostatslib.basis as basis
import utils_testing
import numpy as np

from numpy.typing import NDArray
# automatic define user accessible basis and check the methods

def test_basis_abstract_method_compliance() -> None:
    """
    Check that each non-abstract class implements all abstract methods of super-classes.

    Raises
    ------
    ValueError
        If any of the non-abstract classes doesn't re-implement at least one of the abstract methods it inherits.
    """
    utils_testing.check_all_abstract_methods_compliance(basis)
    return

def test_all_basis_are_in_fixture(init_basis_parameter_grid: pytest.fixture,
                                  min_basis_funcs: pytest.fixture,
                                  evaluate_basis_object: pytest.fixture) -> None:
    """
    Check that all the basis initialization are tested by inspecting the basis module and make sure that all
    the non-abstract classes, except additive and multiplicative are listed in the fixture. If this test fails,
    it means that you need to add some newly implemented basis function to the fixtures.

    Parameters
    ----------
    init_basis_parameter_grid
        Fixture containing the initialization arguments for the basis classes.

    min_basis_funcs
        Fixture containing a dictionary specifying the minimum number of basis functions allowed.

    evaluate_basis_object
        Fixture containing a dictionary specifying the basis objects used for evaluation.

    Returns
    -------
    None
    """
    for class_name, class_obj in utils_testing.get_non_abstract_classes(basis):
        print(f"\n-> Testing \"{class_name}\"")
        if class_name in ["AdditiveBasis", "MultiplicativeBasis"]:
            continue
        assert class_name in init_basis_parameter_grid.keys(), f"{class_name} not in the init_basis_parameter_grid " \
                                                               f"fixture keys!"
        assert class_name in min_basis_funcs.keys(), f"{class_name} not in the min_basis_funcs fixture keys!"
        assert class_name in evaluate_basis_object.keys(), f"{class_name} not in the evaluate_basis_object " \
                                                           f"fixture keys!"

def test_init_and_evaluate_basis(init_basis_parameter_grid: pytest.fixture, capfd: pytest.fixture) -> None:
    """
    Test initialization and evaluation of basis classes:
    Checks:
        - does the initialization accepts the expected inputs?
        - does evaluation works and returns an NDArray with the expected dimensions?

    Parameters:
    -----------
    - initialize_basis:
        A dictionary containing basis names as keys and their initialization arguments as values. This is defined
        as a pytest.fixture in conftest.py
    - capfd
        pytest fixture for capturing stdout and stderr.

    Raises:
    -------
    - ValueError
        If the dimensions of the evaluated basis do not match the expected dimensions.

    Returns:
    - None
    """
    for basis_name in init_basis_parameter_grid:
        basis_class = getattr(basis, basis_name)
        with capfd.disabled():
            disp_str = f"Testing class {basis_name}\n"
            disp_str += '-' * (len(disp_str) - 1)
            print(disp_str)
        for args in init_basis_parameter_grid[basis_name]:
            basis_instance = basis_class(*args)
            with capfd.disabled():
                print(f"num basis: {args[0]}")
            for window_size in [50, 80, 100]:
                eval_basis = basis_instance.evaluate(np.linspace(0, 1, window_size))
                capfd.readouterr()
                if eval_basis.shape[0] != args[0]:
                    raise ValueError(f"Dimensions do not agree: The number of basis should match the first dimensiton of the evaluated basis."
                                     f"The number of basis is {args[0]}",
                                     f"The first dimension of the evaluated basis is {eval_basis.shape[0]}")

                if eval_basis.shape[1] != window_size:
                    raise ValueError(
                        f"Dimensions do not agree: The window size should match the second dimensiton of the evaluated basis."
                        f"The window size is {window_size}",
                        f"The second dimension of the evaluated basis is {eval_basis.shape[1]}")

def test_min_basis_number(min_basis_funcs: pytest.fixture) -> None:
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
    for class_name in min_basis_funcs:
        print(f"\n-> Testing \"{class_name}\"")
        basis_obj = getattr(basis, class_name)
        # params that should not raise exception
        if not min_basis_funcs[class_name]['raise_exception']:
            basis_obj(**min_basis_funcs[class_name]['args'])
        else:
            with pytest.raises(ValueError):
                basis_obj(**min_basis_funcs[class_name]['args'])


def test_basis_sample_consistency_check(basis_sample_consistency_check: pytest.fixture, capfd: pytest.fixture) -> None:
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
    for pars in basis_sample_consistency_check:
        basis_obj = pars['basis_obj']
        n_input = pars['n_input']
        # check that consistent samples do not raise an error
        with capfd.disabled():
            print(f' -> Testing \"{basis_obj.__class__.__name__}\" with {basis_obj._n_input_samples} components')

        inputs = [np.linspace(0, 1, 100 + k) for k in range(n_input)]
        with pytest.raises(ValueError):
            capfd.readouterr()
            basis_obj.evaluate(*inputs)

# Use pytest.mark.parametrize to run the test for each basis separately.
@pytest.mark.parametrize("class_name", [
    'MSplineBasis',
    'RaisedCosineBasisLinear',
    'RaisedCosineBasisLog',
    'OrthExponentialBasis',
    'add2',
    'mul2',
    'add3'
])
def test_basis_eval_checks(evaluate_basis_object: pytest.fixture, capfd: pytest.fixture, class_name):
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
    basis_obj = evaluate_basis_object[class_name]['basis_obj']
    n_input = evaluate_basis_object[class_name]['n_input']
    # check that the correct input does not raise an error
    with capfd.disabled():
        print(f" -> Testing \"{basis_obj.__class__.__name__}\"")
    inputs = [np.linspace(0, 1, 20)] * n_input
    basis_obj.evaluate(*inputs)
    inputs = [20] * n_input
    basis_obj.evaluate_on_grid(*inputs)
    # hide print conditioning number
    capfd.readouterr()
    # check that incorrect input number raises value error
    for delta_input in [-1, 1]:
        with pytest.raises(ValueError):
            inputs = [np.linspace(0, 1, 10)] * (n_input + delta_input)  # wrong number of inputs passed
            basis_obj.evaluate(*inputs)

    for delta_input in [-1, 1]:
        with pytest.raises(ValueError):
            inputs = [10] * (n_input + delta_input)  # wrong number of inputs passed
            basis_obj.evaluate_on_grid(*inputs)
