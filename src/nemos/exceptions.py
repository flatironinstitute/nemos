"""Model specific exceptions."""


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from nemos.glm import GLM
    >>> from nemos.exceptions import NotFittedError
    >>> try:
    ...     GLM().predict([[[1, 2], [2, 3], [3, 4]]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This GLM instance is not fitted yet. Call 'fit' with appropriate arguments.")
    """
