import abc
import inspect
from typing import List, Tuple


def reimplements_method(
    class_obj: type, base_class_obj: type, method_name: str
) -> bool:
    """
    Check if a class has a specific method defined in the subclass, not inherited from the specified base class.

    Parameters
    ----------
    class_obj :
        The class object to check.
    base_class_obj :
        The base class object against which the check is performed.
    method_name : str
        The name of the method to check for.

    Returns
    -------
    bool
        True if the method is defined in the subclass and not inherited from the base class, False otherwise.
    """

    subclass_method = getattr(class_obj, method_name, None)
    superclass_method = getattr(base_class_obj, method_name, None)
    return subclass_method != superclass_method


def get_subclass_methods(class_obj: type) -> List[Tuple[str, type]]:
    """
    Get the subclass-only methods.

    Parameters
    ----------
    class_obj :
        The class object to inspect.

    Returns
    -------
    :
        A list of tuples representing the methods that are specific to the subclass.
        Each tuple contains the method name (str) and the corresponding method object.
    """
    class_methods = inspect.getmembers(class_obj, predicate=inspect.isfunction)

    # Retrieve the superclass methods
    superclass_methods = []
    if class_obj.__bases__:
        superclass = class_obj.__bases__[0]
        superclass_methods = inspect.getmembers(
            superclass, predicate=inspect.isfunction
        )

    # Filter out the methods defined in the superclass
    subclass_methods = [
        method for method in class_methods if method not in superclass_methods
    ]

    return subclass_methods


def list_abstract_methods(class_obj: type) -> List[Tuple[str, type]]:
    """
    Retrieve the names of abstract methods from a class object.

    Parameters
    ----------
    class_obj :
        The class object to inspect.

    Returns
    -------
        A list of tuples representing the abstract methods in the class.
        Each tuple contains the method name (str) and the corresponding method object.
    """
    class_methods = get_subclass_methods(class_obj)
    abstract_methods = [
        (method_name, method)
        for method_name, method in class_methods
        if getattr(method, "__isabstractmethod__", False)
    ]
    return abstract_methods


def is_abstract(class_obj: type) -> bool:
    """
    Check if a type object is an abstract class.

    Parameters
    ----------
    class_obj : type
        The type object to check.

    Returns
    -------
    bool
        True if the type object is an abstract class, False otherwise.
    """
    return abc.ABC in getattr(class_obj, "__bases__", [])


def get_non_abstract_classes(module) -> List[Tuple[str, type]]:
    """
    List all non-abstract classes in a module.

    Parameters
    ----------
    module : module
        The module object to inspect.

    Returns
    -------
    :
        A list of tuples representing the non-abstract classes in the module.
        Each tuple contains the class name (str) and the corresponding class object.
    """
    basis_classes = inspect.getmembers(
        module, lambda obj: inspect.isclass(obj) and obj.__module__ == module.__name__
    )
    return [
        (name, class_obj)
        for name, class_obj in basis_classes
        if not is_abstract(class_obj)
    ]


def get_abstract_classes(module) -> List[Tuple[str, type]]:
    """
    List all abstract classes in a module.

    Parameters
    ----------
    module : module
        The module object to inspect.

    Returns
    -------
    List[Tuple[str, type]]
        A list of tuples representing the abstract classes in the module.
        Each tuple contains the class name (str) and the corresponding class object.
    """
    basis_classes = inspect.getmembers(
        module, lambda obj: inspect.isclass(obj) and obj.__module__ == module.__name__
    )
    return [
        (name, class_obj) for name, class_obj in basis_classes if is_abstract(class_obj)
    ]


def get_superclass_abstract_methods(class_obj: type) -> List[Tuple[str, type, type]]:
    """
    Retrieve the abstract methods defined in the superclass.

    Parameters
    ----------
    class_obj :
        The class object to inspect.

    Returns
    -------
    List[Tuple[str, type, type]]
        A list of tuples representing the abstract methods defined in the superclass.
        Each tuple contains the method name (str), method object, and superclass object.
    """
    super_class = class_obj.__base__

    if super_class:
        super_class_abstract_methods = get_superclass_abstract_methods(super_class)
        super_class_abstract_methods += [
            (method_name, method, super_class)
            for method_name, method in list_abstract_methods(super_class)
            if getattr(method, "__isabstractmethod__", False)
        ]
    else:
        super_class_abstract_methods = []

    return super_class_abstract_methods


def check_all_abstract_methods_compliance(module) -> None:
    """
    Check if all classes in a module properly implement abstract methods defined in their respective base classes.

    Parameters
    ----------
    module :
        The module object to inspect.

    Raises
    ------
    ValueError
        If any abstract method is not implemented in the subclass.
    """
    non_abstract_classes = get_non_abstract_classes(module)
    for _, base_class in non_abstract_classes:
        string = f"\nAnalyzing class {base_class.__name__}:\n"
        string += "-" * len(string)
        print(string)
        for method_name, method, super_class in get_superclass_abstract_methods(
            base_class
        ):
            boolean = reimplements_method(base_class, super_class, method_name)
            print(f'Is method "{method_name}" re-instantiated? {boolean}')
            if not boolean:
                raise ValueError(
                    f"Abstract method {method} not implemented in {base_class} sub-class!"
                )


def trim_kwargs(cls: type, kwargs: dict, class_specific_params: dict):
    """
    Filter a dictionary of keyword arguments to include only those specific to a given class.

    Parameters
    ----------
    cls :
        The class object for which the keyword arguments are filtered.
    kwargs :
        A dictionary of keyword arguments to be filtered.
    class_specific_params :
        A mapping of class names to sets or lists of allowed parameter names.

    Returns
    -------
    :
        A dictionary containing only the keyword arguments specific to the given class.

    Example
    -------
    >>> class_specific_params = {
    ...     'MyClass': {'param1', 'param2'},
    ...     'OtherClass': {'param3', 'param4'}
    ... }
    >>> kwargs = {'param1': 10, 'param3': 20, 'param5': 30}
    >>> class MyClass:
    ...     pass
    >>> trim_kwargs(MyClass, kwargs, class_specific_params)
    {'param1': 10}
    """
    return {
        key: value
        for key, value in kwargs.items()
        if key in class_specific_params[cls.__name__]
    }
