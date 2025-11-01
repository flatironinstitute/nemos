"""
Utility function for composite basis.

Collection of functions that transverse the composite basis tree
with no to minimal re
"""

import re
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .._inspect_utils.inspect_utils import count_positional_and_var_args

if TYPE_CHECKING:
    from ._basis_mixin import AtomicBasisMixin, BasisMixin
    from ._custom_basis import CustomBasis

__PUBLIC_BASES__ = [
    "IdentityEval",
    "HistoryConv",
    "MSplineEval",
    "MSplineConv",
    "BSplineEval",
    "BSplineConv",
    "CyclicBSplineEval",
    "CyclicBSplineConv",
    "FourierEval",
    "RaisedCosineLinearEval",
    "RaisedCosineLinearConv",
    "RaisedCosineLogEval",
    "RaisedCosineLogConv",
    "OrthExponentialEval",
    "OrthExponentialConv",
    "AdditiveBasis",
    "MultiplicativeBasis",
    "CustomBasis",
]


def add_docstring(method_name, cls):
    """Prepend super-class docstrings."""
    attr = getattr(cls, method_name, None)
    if attr is None:
        raise AttributeError(f"{cls.__name__} has no attribute {method_name}!")
    doc = attr.__doc__

    # Decorator to add the docstring
    def wrapper(func):
        func.__doc__ = "\n".join([doc, func.__doc__])  # Combine docstrings
        return func

    return wrapper


def _iterate_over_components(basis: "BasisMixin"):
    components = (
        basis._iterate_over_components()
        if hasattr(basis, "_iterate_over_components")
        else [basis]
    )
    yield from components


def _get_root(bas: "BasisMixin"):
    """Get the basis root."""

    parent = bas
    while hasattr(parent, "_parent") and parent._parent is not None:
        parent = parent._parent
    return parent


def _has_default_label(bas: "BasisMixin"):
    """Check for default label.

    The check either use the property (if it is a nemos basis), or the class name
    (if it is a custom user defined basis).
    """
    if hasattr(bas, "_has_default_label"):
        return bas._has_default_label
    else:
        label = getattr(bas, "label", bas.__class__.__name__)
        return re.match(rf"^{bas.__class__.__name__}(_\d+)?$", label)


def _recompute_class_default_labels(
    bas: "BasisMixin",
):
    """
    Recompute all labels matching default for self.

    Parameters
    ----------
    bas:
        Basis component calling the method
    """
    class_name = bas.__class__.__name__
    pattern = re.compile(rf"^{class_name}(_\d+)?$")
    root = _get_root(bas)
    bas_id = 0
    # if root is one of our bases it will have the iteration method, if custom from user
    # I assume it is atomic
    for comp_bas in _iterate_over_components(root):
        if re.match(pattern, comp_bas._label):
            comp_bas._label = f"{class_name}_{bas_id}" if bas_id else class_name
            bas_id += 1


def _recompute_all_default_labels(root: "BasisMixin") -> "BasisMixin":
    """Recompute default all labels."""
    updated = []
    for bas in _iterate_over_components(root):
        if _has_default_label(bas) and bas.__class__.__name__ not in updated:
            _recompute_class_default_labels(bas)
            updated.append(bas.__class__.__name__)
    return root


def _update_label_from_root(bas: "BasisMixin", class_name: str, class_label: str):
    """
    Subtract 1 to each matching default label with higher ID then current.

    Parameters
    ----------
    class_name : str
        Class name of the component that is setting a new label.
    class_label : str
        Current component label.
    """
    pattern = re.compile(rf"^{class_name}(_\d+)?$")
    match = re.match(pattern, class_label)
    if match is None:
        return
    # get the "ID" of the label
    current_id = int(match.group(1)[1:]) if match.group(1) else 0
    # subtract one to the ID of any other default label with ID > current_id
    root = _get_root(bas)
    for bas in _iterate_over_components(root):
        match = re.match(pattern, bas._label)
        if match:
            bas_id = int(match.group(1)[1:]) if match.group(1) else 0
            bas_id = bas_id - 1 if bas_id > current_id else bas_id
            bas._label = f"{class_name}_{bas_id}" if bas_id else class_name


def _composite_basis_setter_logic(new: "BasisMixin", current: "BasisMixin"):
    """Setter logic for composite basis."""
    # Carry-on label if possible
    if _has_default_label(new) and not _has_default_label(current):
        try:
            new.label = getattr(current, "label", current.__class__.__name__)
        except ValueError:
            pass  # If label is in use, ignore

    # add a parent to the new basis
    new._parent = getattr(current, "_parent", None)

    # Carry-on input shape info if dimensions match
    for attr in ("_input_shape_product", "_input_shape_"):
        if getattr(new, attr, None) is None and getattr(
            new, "_n_input_dimensionality", None
        ) == getattr(current, "_n_input_dimensionality", None):
            setattr(new, attr, getattr(current, attr, None))
    return new


def _atomic_basis_label_setter_logic(
    bas: "AtomicBasisMixin | CustomBasis", new_label: str
) -> Exception | None:
    """Setter logic for atomic basis."""
    # check default cases
    current_label = getattr(bas, "_label", None)
    if new_label == current_label:
        return

    elif new_label is None:
        # check if already default
        bas._label = bas.__class__.__name__
        _recompute_class_default_labels(bas)
        return

        # raise error in case label is not string.
    elif not isinstance(new_label, str):
        return TypeError(
            f"'label' must be a string. Type {type(new_label)} was provided instead."
        )

    else:
        # check if label matches class-name plus identifier
        match = re.match(r"(.+)?_\d+$", new_label)
        check_string = match.group(1) if match else None
        check_string = check_string if check_string in __PUBLIC_BASES__ else new_label
        if check_string == bas.__class__.__name__:
            bas._label = check_string
            _recompute_class_default_labels(bas)
            return

    root = _get_root(bas)
    current_labels = (
        root._generate_subtree_labels() if root is not bas else [current_label]
    )
    if (check_string in current_labels) or (check_string in __PUBLIC_BASES__):
        if check_string in __PUBLIC_BASES__:
            msg = f"Cannot assign '{new_label}' to a basis of class {bas.__class__.__name__}."
        else:
            msg = f"Label '{new_label}' is already in use. When user-provided, label must be unique."
        return ValueError(msg)
    else:
        # check if current is the default label
        # if that's true, since the new label is not a default label
        # update all other default label names.
        _update_label_from_root(
            bas, bas.__class__.__name__, getattr(bas, "_label", bas.label)
        )
        bas._label = new_label
    return


def infer_input_dimensionality(bas: "BasisMixin") -> int:
    """Infer input dimensionality from compute_features signature.

    If `_n_input_dimensionality` return the attribute, otherwise return
    the number of fixed arguments in `compute_features`.
    """
    n_input_dim = getattr(bas, "_n_input_dimensionality", None)
    if n_input_dim is None:
        # infer from compute_features (facilitate custom basis compatibility).
        # assume compute_features is always implemented.
        if hasattr(bas, "funcs"):
            funcs = bas.funcs
            dims = [count_positional_and_var_args(f)[0] for f in funcs]
            if len(set(dims)) != 1:
                raise ValueError(
                    "Each function provided to ``funcs`` in ``CustomBasis`` must accept the same "
                    "number of input time series."
                )
            n_input_dim = dims[0]
        else:
            funcs = [bas.compute_features] if hasattr(bas, "compute_features") else []
            n_input_dim = sum(count_positional_and_var_args(f)[0] for f in funcs)
    return n_input_dim


def generate_basis_label_pair(bas: "BasisMixin"):
    """Generate all labels and basis in a basis tree."""
    if hasattr(bas, "basis1") and hasattr(bas, "basis2"):
        for label, sub_bas in generate_basis_label_pair(bas.basis1):
            yield label, sub_bas
        for label, sub_bas in generate_basis_label_pair(bas.basis2):
            yield label, sub_bas
    yield getattr(bas, "label", bas.__class__.__name__), bas


def generate_composite_basis_labels(bas: "BasisMixin", type_label: str) -> str:
    """Generate all labels in a basis tree."""
    if hasattr(bas, "basis1") and hasattr(bas, "basis2"):
        if type_label == "all" or bas._label:
            yield bas.label

        # generator for sub-basis
        def generate_labels(sub_basis):
            generate_subtree = getattr(sub_basis, "_generate_subtree_labels", None)
            if generate_subtree:
                yield from generate_subtree(type_label)
            elif type_label == "all" or not _has_default_label(sub_basis):
                yield sub_basis.label

        yield from generate_labels(bas.basis1)
        yield from generate_labels(bas.basis2)

    else:
        if type_label == "all" or (not _has_default_label(bas)):
            yield getattr(bas, "label", bas.__class__.__name__)


def label_setter(bas: "BasisMixin", label: str | None) -> None | ValueError:
    """Label setter logic for any basis."""
    if not hasattr(bas, "basis1") or not hasattr(bas, "basis2"):
        return _atomic_basis_label_setter_logic(bas, label)

    # composite basis setter logic.
    reset = (
        (label is None)
        or (label == bas._generate_label())
        or (label == bas.__class__.__name__)
    )
    error = None
    if reset:
        label = None
    else:
        label = str(label)
        if label in _get_root(bas)._generate_subtree_labels():
            error = ValueError(
                f"Label '{label}' is already in use. When user-provided, label must be unique."
            )
        elif label in __PUBLIC_BASES__ and label != bas.__class__.__name__:
            error = ValueError(
                f"Cannot set basis label '{label}' for basis of type {type(bas)}."
            )
    if not error:
        bas._label = label
    return error


def _check_valid_shape_tuple(*shapes):
    for shape in shapes:
        if shape is not None and not all(isinstance(i, int) for i in shape):
            raise ValueError(
                f"The tuple provided contains non integer values. Tuple: {shape}."
            )


def transform_to_shape(xi):
    if isinstance(xi, tuple):
        shape = xi
    elif isinstance(xi, int):
        shape = () if xi == 1 else (xi,)
    else:
        shape = xi if xi is None else xi.shape[1:]
    return shape


def set_input_shape_atomic(
    bas: "AtomicBasisMixin | CustomBasis", *xis: int | tuple[int, ...] | NDArray | None
) -> "AtomicBasisMixin":
    """Set input shape attributes for atomic basis."""
    # reset all to none
    if all(xi is None for xi in xis):
        bas._input_shape_ = None
        bas._input_shape_product = None
        return bas
    # otherwise apply shape
    shapes = []
    n_inputs = ()
    for xi in xis:
        n_inputs = (*n_inputs, int(np.prod(xi)))
        shapes.append(xi)

    bas._input_shape_ = shapes

    # total number of input time series. Used  for slicing and reshaping
    bas._input_shape_product = n_inputs
    return bas


def _have_unique_shapes(inputs: List[NDArray] | List[Tuple]):
    if inputs is None:
        return True

    # Extract shapes whether inputs are arrays or shape tuples
    if hasattr(inputs[0], "shape"):
        shapes = {x.shape for x in inputs}
        context = "was evaluated with"
    else:
        shapes = set(inputs)
        context = "was configured with"
    return len(shapes) == 1, shapes, context


def _check_unique_shapes(inputs: List[NDArray] | List[Tuple], basis: "BasisMixin"):
    """Check that all inputs have the same shape.

    Parameters
    ----------
    inputs: Either a list of shape tuples or actual arrays
    basis: The basis object for error context

    Raises
    ------
    ValueError:
        If all the arrays have the same shapes or all the tuples are the same.
    """
    have_unique_shape, shapes, context = _have_unique_shapes(inputs)
    if not have_unique_shape:
        raise ValueError(
            f"{basis.__class__.__name__} requires all inputs to have the same shape. "
            f"The basis {context} input shapes: {shapes}.\n{basis}"
        )


def set_input_shape(bas, *xi, allow_inputs_of_different_shape=True):
    """Set input shape.

    Set input shape logic, compatible with all bases (composite, atomic, and custom).
    """

    # use 1 as default or number of non-variable args
    n_args = (
        count_positional_and_var_args(bas.compute_features)[0]
        if hasattr(bas, "compute_features")
        else 1
    )
    # get the attribute if available
    n_input_dim = getattr(bas, "_n_input_dimensionality", n_args)

    if len(xi) == 1 and xi[0] is None:
        xi = (None,) * n_input_dim

    elif len(xi) != n_input_dim:
        expected_inputs = getattr(bas, "_n_input_dimensionality", 1)
        raise ValueError(
            f"set_input_shape expects {expected_inputs} input"
            f"{'s' if expected_inputs > 1 else ''}, but {len(xi)} were provided."
        )

    original_xi = xi
    try:
        xi = [transform_to_shape(x) for x in xi]
    except Exception as e:
        raise ValueError(
            f"Cannot convert inputs ``{original_xi}`` to shape tuple."
        ) from e

    if not allow_inputs_of_different_shape:
        _check_unique_shapes(xi, bas)

    _check_valid_shape_tuple(*xi)

    if not hasattr(bas, "basis1"):
        return set_input_shape_atomic(bas, *xi)

    # here we can assume it is a composite basis
    # grab the input dimensionality
    n_args_1, _ = (
        count_positional_and_var_args(bas.basis1.compute_features)
        if hasattr(bas.basis1, "compute_features")
        else 1
    )
    n_input_dim_1 = getattr(bas.basis1, "_n_input_dimensionality", n_args_1)

    out1 = set_input_shape(bas.basis1, *xi[:n_input_dim_1])
    out2 = set_input_shape(bas.basis2, *xi[n_input_dim_1:])

    # out1 and out2 will have an _input_shape_product set by the "set_input_shape_atomic" method.
    # here is safe to use the attribute.
    if out1._input_shape_product is not None and out2._input_shape_product is not None:
        bas._input_shape_product = (
            *out1._input_shape_product,
            *out2._input_shape_product,
        )
    else:
        bas._input_shape_product = None
    return bas


def unpack_shapes(basis) -> Tuple:
    """Generate input shapes for any basis type.

    Generate input shape independently of the basis kind (i.e. works
    for composite, custom, and atomic bases).
    """
    if hasattr(basis, "_input_shape_") and hasattr(basis, "input_shape"):
        if basis._input_shape_ is None:
            yield None
        else:
            yield from basis._input_shape_
    elif hasattr(basis, "input_shape"):
        yield basis.input_shape
    else:
        yield from [None] * infer_input_dimensionality(basis)


def list_shapes(basis) -> List[Tuple | None]:
    """List input shape for all components."""
    if basis is None:
        return [None]
    components = getattr(basis, "_iterate_over_components", lambda: [basis])()
    return [shape for comp in components for shape in unpack_shapes(comp)]


def get_input_shape(bas: "BasisMixin") -> List[Tuple | None]:
    """Get the input shape of a composite basis.

    Get input shape from composition of basis, including user defined ones.
    The function treats any bases without a `_iterate_over_components` methods as an
    atomic basis.

    The input shape for user-defined bases is retrieved with _input_shape_ property that
    is set at runtime when `compute_features` or `set_input_shape` is called. If the
    number of inputs is not set yet, it returns a list of None.
    """
    input_dim = infer_input_dimensionality(bas)
    if input_dim == 1:
        ishape = getattr(bas, "_input_shape_", [None])
        return ishape if ishape is not None else [None]

    elif not hasattr(bas, "basis1") and hasattr(bas, "_input_shape_"):
        return [None] * input_dim if bas._input_shape_ is None else bas._input_shape_

    basis1 = getattr(bas, "_basis1", None)
    basis2 = getattr(bas, "_basis2", None)

    # Handle cases where one or both bases are missing
    if basis1 is None and basis2 is None:
        return [None, None]
    if basis1 is None:
        return [None] + list_shapes(basis2)
    if basis2 is None:
        return list_shapes(basis1) + [None]

    # If both bases exist, return combined shapes
    return list_shapes(basis1) + list_shapes(basis2)


def is_basis_like(putative_basis: Any, sklearn_compatibility=False) -> bool:
    """Check if putative_basis respect nemos basis API."""
    is_basis = hasattr(putative_basis, "compute_features")
    if sklearn_compatibility:
        is_basis &= hasattr(putative_basis, "get_params") and hasattr(
            putative_basis, "set_params"
        )
    return is_basis


def multiply_basis_by_integer(bas: "BasisMixin", mul: int) -> "BasisMixin":
    """Multiplication by integer logic for bases."""
    if mul <= 0:
        raise ValueError(
            "Basis multiplication error. Integer multiplicative factor must be positive, "
            f"{mul} provided instead."
        )
    elif not all(_has_default_label(b) for _, b in generate_basis_label_pair(bas)):
        raise ValueError(
            "Cannot multiply by an integer a basis including a user-defined labels "
            "(because then they won't be unique). Set labels after multiplication."
        )
    # default case
    if mul == 1:
        # __sklearn_clone__ reset the parent to None in case bas.basis1 * 1
        # (deepcopy would not)
        copy_ = getattr(bas.__class__, "__sklearn_clone__", deepcopy)
        bas = copy_(bas)

        # if deepcopy was called (custom basis used in composition)
        # reset _parent if it exists and is not None
        if hasattr(bas, "_parent") and bas._parent is not None:
            bas._parent = None
        _recompute_all_default_labels(bas)
        return bas

    # parent is set to None at init for add and updated for self.
    add = bas + bas
    with add._set_shallow_copy(True):
        for _ in range(2, mul):
            add += deepcopy(bas)
    _recompute_all_default_labels(add)
    return add


def raise_basis_to_power(bas: "BasisMixin", exponent: int) -> "BasisMixin":
    """Power of basis by integer."""
    if not isinstance(exponent, int):
        raise TypeError("Basis exponent should be an integer!")

    if exponent <= 0:
        raise ValueError("Basis exponent should be a non-negative integer!")
    elif not all(_has_default_label(b) for _, b in generate_basis_label_pair(bas)):
        raise ValueError(
            "Cannot calculate the power of a basis including a user-defined labels "
            "(because then they won't be unique). Set labels after exponentiation."
        )

    # default case
    if exponent == 1:
        # __sklearn_clone__ reset the parent to None in case bas.basis1 ** 1
        # (deepcopy would not)
        copy_ = getattr(bas.__class__, "__sklearn_clone__", deepcopy)
        bas = copy_(bas)

        # if deepcopy was called (custom basis used in composition)
        # reset _parent if it exists and it is not None
        if hasattr(bas, "_parent") and bas._parent is not None:
            bas._parent = None

        _recompute_all_default_labels(bas)
        return bas

    mul = bas * bas
    with mul._set_shallow_copy(True):
        for _ in range(2, exponent):
            mul *= deepcopy(bas)
    _recompute_all_default_labels(mul)
    return mul


def is_transformer(bas: Any):
    """Check if it conforms to transformer API."""
    return hasattr(bas, "basis") and hasattr(bas, "fit_transform")


def promote_to_transformer(method):
    """Apply operations to basis within transformer and transform output."""

    @wraps(method)
    def wrapper(*args, **kwargs):
        # if it looks like a transformer, pull out basis
        args_bas = (b.basis if is_transformer(b) else b for b in args)
        kwargs_bas = {k: b.basis if is_transformer(b) else b for k, b in kwargs.items()}
        out = method(*args_bas, **kwargs_bas)
        any_transformer = any(
            (
                *(is_transformer(a) for a in args),
                *(is_transformer(b) for b in kwargs.values()),
            )
        )
        if any_transformer and hasattr(out, "to_transformer"):
            return out.to_transformer()
        return out

    return wrapper
