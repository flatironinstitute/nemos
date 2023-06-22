import inspect

import abc
def inherits(module, super_class):
    """
    Check if classes in a module inherit from a given super class.

    Parameters
    ----------
    module : module
        The module object.
    module_name : str
        The name of the module.
    super_class_name : str
        The name of the super class to check for inheritance.

    Returns
    -------
    dict
        A dictionary mapping class names to a boolean indicating if they inherit from the super class.
    """
    basis_classes = inspect.getmembers(
        module,
        lambda obj: inspect.isclass(obj) and obj.__module__ == module.__name__
    )
    inherits_dict = {}
    for class_name, class_obj in basis_classes:
        inheritance = [base_class.__name__ for base_class in inspect.getmro(class_obj)[1:]]
        inherits_dict[class_name] = super_class.__name__ in inheritance
    return inherits_dict


def has_method(class_obj, base_class_obj, method_name):
    """
    Check if a class has a specific method defined in the subclass, not inherited from the specified base class.

    Parameters
    ----------
    class_obj : class
        The class object to check.
    base_class_obj : class
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



def list_abstract_methods(class_obj):
    """
    Retrieve the names of abstract methods from a class object.

    Parameters
    ----------
    class_obj : class
        The class object to inspect.

    Returns
    -------
    list
        A list of abstract method names.
    """
    class_methods = inspect.getmembers(class_obj, predicate=inspect.isfunction)
    abstract_methods = [
        method_name for method_name, method in class_methods
        if getattr(method, "__isabstractmethod__", False)
    ]
    return abstract_methods


def abstract_methods_compliance(module, base_class):
    """
    Check if classes in a module properly implement abstract methods defined in a base class.

    Parameters
    ----------
    module : module
        The module object.
    base_class : class
        The base class object containing abstract methods to check.

    Raises
    ------
    ValueError
        If any subclass does not properly implement an abstract method defined in the base class.
    """
    # Get all classes defined in the math module
    basis_classes = inspect.getmembers(
        module,
        lambda obj: inspect.isclass(obj) and obj.__module__ == module.__name__
    )

    inherits_dict = inherits(module, base_class)
    abstract_methods = list_abstract_methods(base_class)

    # Print class names and their inheritance
    for class_name, class_obj in basis_classes:
        if inherits_dict[class_name]:
            for method in abstract_methods:
                boolean = has_method(class_obj, base_class, method)
                print(f"{class_name} has method {method}? {boolean}")
                if not boolean:
                    raise ValueError(f'Abstract method {method} not implemented in {class_obj} sub-class!')
    return

def check_all_abstract_methods_compliance(module):
    """
    Check if all classes in a module properly implement abstract methods defined in their respective base classes.

    Parameters
    ----------
    module : module
        The module object to inspect.
    """
    base_classes = inspect.getmembers(
        module,
        lambda obj: inspect.isclass(obj) and obj.__module__ == module.__name__
    )
    for _, base_class in base_classes:
        string = f'\nAnalyzing base class {base_class.__name__}:\n'
        string += '-'*len(string)
        print(string)
        abstract_methods_compliance(module, base_class)
    return