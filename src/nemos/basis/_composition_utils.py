"""
Utility function for composite basis.

Collection of functions that transverse the composite basis tree
with no to minimal re
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._basis import Basis
    from ._basis_mixin import AtomicBasisMixin, CompositeBasisMixin


__PUBLIC_BASES__ = [
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
    "AdditiveBasis",
    "MultiplicativeBasis",
]


def _get_root(bas: "AtomicBasisMixin | CompositeBasisMixin"):
    """Get the basis root"""
    parent = bas
    while hasattr(parent, "_parent") and parent._parent is not None:
        parent = parent._parent
    return parent


def _call_parent_method(bas: "AtomicBasisMixin", method: str, *args, **kwargs):
    """Call a parent's basis method, if available."""
    parent = getattr(bas, "_parent", None)
    if parent is not None:
        method = getattr(parent, method, None)
        return method(*args, **kwargs)


def _has_default_label(bas: "Basis"):
    """Check for default label.

    The check either use the property (if it is a nemos basis), or the class name
    (if it is a custom user defined basis).
    """
    if hasattr(bas, "_has_default_label"):
        return bas._has_default_label
    else:
        label = getattr(bas, "label", bas.__class__.__name__)
        return re.match(rf"^{bas.__class__.__name__}(_\d+)?$", label)


def _recompute_class_default_labels(bas: "AtomicBasisMixin | CompositeBasisMixin"):
    """
    Recompute all labels matching default for self.

    Parameters
    ----------
    bas:
        Basis component calling the method
    cls_name : str
        Class name of the component that is setting a new label.
    """
    cls_name = bas.__class__.__name__
    pattern = re.compile(rf"^{cls_name}(_\d+)?$")
    root = _get_root(bas)
    bas_id = 0
    # if root is one of our bases it will have the iteration method, if custom from user
    # I assume it is atomic
    components = (
        root._iterate_over_components()
        if hasattr(root, "_iterate_over_components")
        else [root]
    )
    for comp_bas in components:
        if re.match(pattern, comp_bas._label):
            comp_bas._label = f"{cls_name}_{bas_id}" if bas_id else cls_name
            bas_id += 1


def _recompute_all_default_labels(root: "Basis") -> "Basis":
    """Recompute default all labels."""
    updated = []
    components = (
        root._iterate_over_components()
        if hasattr(root, "_iterate_over_components")
        else [root]
    )
    for bas in components:
        if _has_default_label(bas) and bas.__class__.__name__ not in updated:
            _recompute_class_default_labels(bas)
            updated.append(bas.__class__.__name__)
    return root


def _update_label_from_root(
    bas: "AtomicBasisMixin | CompositeBasisMixin", cls_name: str, cls_label: str
):
    """
    Subtract 1 to each matching default label with higher ID then current.

    Parameters
    ----------
    cls_name : str
        Class name of the component that is setting a new label.
    cls_label : str
        Current component label.
    """
    pattern = re.compile(rf"^{cls_name}(_\d+)?$")
    match = re.match(pattern, cls_label)
    if match is None:
        return
    # get the "ID" of the label
    current_id = int(match.group(1)[1:]) if match.group(1) else 0
    # subtract one to the ID of any other default label with ID > current_id
    root = _get_root(bas)
    components = (
        root._iterate_over_components()
        if hasattr(root, "_iterate_over_components")
        else [root]
    )
    for bas in components:
        match = re.match(pattern, bas._label)
        if match:
            bas_id = int(match.group(1)[1:]) if match.group(1) else 0
            bas_id = bas_id - 1 if bas_id > current_id else bas_id
            bas._label = f"{cls_name}_{bas_id}" if bas_id else cls_name


def _composite_basis_setter_logic(new: "Basis", current: "Basis"):
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
    bas: "AtomicBasisMixin", new_label: str
) -> Exception | None:
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


def _check_input_consistency(bas, *x):
    """Check input consistency across calls."""
    # remove sample axis and squeeze
    shape = x.shape[1:]

    initialized = bas._input_shape_ is not None
    is_shape_match = bas._input_shape_[0] == shape
    if initialized and not is_shape_match:
        expected_shape_str = "(n_samples, " + f"{bas._input_shape_[0]}"[1:]
        expected_shape_str = expected_shape_str.replace(",)", ")")
        raise ValueError(
            f"Input shape mismatch detected.\n\n"
            f"The basis `{bas.__class__.__name__}` with label '{bas.label}' expects inputs with "
            f"a consistent shape (excluding the sample axis). Specifically, the shape should be:\n"
            f"  Expected: {expected_shape_str}\n"
            f"  But got:  {x.shape}.\n\n"
            "Note: The number of samples (`n_samples`) can vary between calls of `compute_features`, "
            "but all other dimensions must remain the same. If you need to process inputs with a "
            "different shape, please create a new basis instance, or set a new input shape by calling "
            "`set_input_shape`."
        )
