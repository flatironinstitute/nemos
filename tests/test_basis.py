import pytest
import neurostatslib.basis as basis
import utils_testing
import inspect

# automatic define user accessible basis and check the methods

# @pytest.parametrize('basis_func',)
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




def get_subclass_methods_from_instance(class_instance):
    """
    Retrieves methods that are specific to the instantiated class (excludes inherited methods).

    Parameters:
    class_instance: object
        An instantiated class object.

    Returns:
    List[Tuple[str, method]]
        A list of tuples representing the methods that are specific to the instantiated class.
        Each tuple contains the method name (str) and the corresponding method object.

    Raises:
    ValueError: If the provided argument is not an instantiated class.
    """

    if not isinstance(class_instance, class_instance.__class__):
        raise ValueError('Must provide an instantiated class!')

    return inspect.getmembers(class_instance, predicate=inspect.ismethod)


def tmp_test_annotation(class_instance):
    """
    Check that the annotation in the scripts agree with the following rules:
        1. All functions and methods are fully annotated (input and output)
        2. Non-abstract method specific annotation rules:
            - evaluate annotated output is  NDArray
            - _evaluate annotated output is NDArray
            - _get_samples annotated output is  tuple[NDArray, ...] for Basis, addBasis and mulBasis, tuple[NDArray] otherwise
            - gen_basis annotated output is tuple[tuple[NDArray, ...], NDArray]
            - _check* annotated output is None
            - __add__ and __mul__ annotated output is Basis
            - __init__ annotated output is None
            - _generate_knots annotated output is NDArray
            - _transform_samples annotated output is NDArray
            - mspline annotated output is NDArray



    Parameters
    ----------
    class_instance

    Returns
    -------

    """
    pass
    # expected_annotation
    # expected_return_type = list[np.array]
    # annotation_return_type = inspect.signature(func).return_annotation
    #
    # # Evaluate the function
    # output = func(np.linspace(0,1,100))
    #
    # # Check if the output matches the expected return type
    # assert isinstance(output, expected_return_type), "Output does not conform to the expected return type."
    # print("Return type test passed.")

