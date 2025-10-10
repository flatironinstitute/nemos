"""Utility functions for creating regularizer object."""

from .regularizer import ElasticNet, GroupLasso, Lasso, Ridge, UnRegularized

AVAILABLE_REGULARIZERS = ["UnRegularized", "Ridge", "Lasso", "GroupLasso", "ElasticNet"]

# Mapping for O(1) lookup
_REGULARIZER_MAP = {
    "UnRegularized": UnRegularized,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "GroupLasso": GroupLasso,
    "ElasticNet": ElasticNet,
    None: UnRegularized,  # Handle None case
}


def instantiate_regularizer(name: str | None, **kwargs):
    """
    Create a regularizer from a given name.

    Parameters
    ----------
    name :
        The string name of the regularizer to create.
        Must be one of: 'UnRegularized', 'Ridge', 'Lasso','GroupLasso', None.
        If set to None, it will behave as 'Unregularized'.
    **kwargs :
        Additional keyword arguments are passed to the regularizer constructor.

    Returns
    -------
    :
        The regularizer instance with default parameters.

    Raises
    ------
    ValueError
        If the `name` provided does not match to any available regularizer.
    """
    # If the name contains a module path, extract the class name
    if name:
        name = name.split("nemos.regularizer.")[-1]

    if name in _REGULARIZER_MAP:
        return _REGULARIZER_MAP[name](**kwargs)
    else:
        raise ValueError(
            f"Unknown regularizer: {name}. "
            f"Regularizer must be one of {AVAILABLE_REGULARIZERS} or their full module path."
        )
