"""Utility functions for creating regularizer object."""

AVAILABLE_REGULARIZERS = ["unregularized", "ridge", "lasso", "group_lasso"]


def create_regularizer(name: str):
    """
    Create a regularizer from a given name.

    Parameters
    ----------
    name :
        The string name of the regularizer to create. Must be one of: 'unregularized', 'ridge', 'lasso', 'group_lasso'.

    Returns
    -------
    :
        The regularizer instance with default parameters.

    Raises
    ------
    ValueError
        If the `name` provided does not match to any available regularizer.
    """
    match name:
        case "unregularized":
            from .regularizer import UnRegularized

            return UnRegularized()
        case "ridge":
            from .regularizer import Ridge

            return Ridge()
        case "lasso":
            from .regularizer import Lasso

            return Lasso()
        case "group_lasso":
            from .regularizer import GroupLasso

            return GroupLasso()

    raise ValueError(
        f"Unknown regularizer: {name}. Regularizer must be one of {AVAILABLE_REGULARIZERS}"
    )
