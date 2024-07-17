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
    if name == "unregularized":
        from .regularizer import UnRegularized

        return UnRegularized()
    elif name == "ridge":
        from .regularizer import Ridge

        return Ridge()
    elif name == "lasso":
        from .regularizer import Lasso

        return Lasso()
    elif name == "group_lasso":
        from .regularizer import GroupLasso

        return GroupLasso()
    else:
        raise ValueError(
            f"Unknown regularizer: {name}. "
            f"Regularizer must be one of {AVAILABLE_REGULARIZERS}"
        )
