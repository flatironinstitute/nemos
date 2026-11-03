from .inverse_link_function_utils import (
    exp,
    expit,
    identity,
    log_softmax,
    logistic,
    norm_cdf,
    softplus,
)
from .utils import one_over_x

LINK_NAME_TO_FUNC = {
    "exp": exp,
    "expit": expit,
    "jax._src.lax.lax.logistic": expit,
    "jax._src.nn.functions.softplus": softplus,
    "jax._src.nn.functions.log_softmax": log_softmax,
    "jax._src.numpy.ufuncs.exp": exp,
    "jax._src.scipy.special.expit": expit,
    "jax._src.scipy.stats.norm.cdf": norm_cdf,
    "jax._src.scipy.stats.norm.norm_cdf": norm_cdf,
    "jax.lax.logistic": logistic,
    "jax._src.scipy.special.logistic": logistic,
    "jax.nn.softplus": softplus,
    "jax.nn.log_softmax": log_softmax,
    "jax.numpy.exp": exp,
    "jax.scipy.special.expit": expit,
    "jax.scipy.stats.norm.cdf": norm_cdf,
    "logistic": logistic,
    "nemos.inverse_link_function_utils.exp": exp,
    "nemos.inverse_link_function_utils.expit": expit,
    "nemos.inverse_link_function_utils.logistic": logistic,
    "nemos.inverse_link_function_utils.norm_cdf": norm_cdf,
    "nemos.inverse_link_function_utils.one_over_x": one_over_x,
    "nemos.inverse_link_function_utils.softplus": softplus,
    "nemos.inverse_link_function_utils.log_softmax": log_softmax,
    "nemos.utils.one_over_x": one_over_x,
    "norm.cdf": norm_cdf,
    "norm_cdf": norm_cdf,
    "one_over_x": one_over_x,
    "softplus": softplus,
    "log_softmax": log_softmax,
    "identity": identity,
    "nemos.inverse_link_function_utils.identity": identity,
}

AVAILABLE_INVERSE_LINK_FUNCTIONS = list(LINK_NAME_TO_FUNC.keys())


def instantiate_inverse_link_function(inverse_link_function: str):
    """
    Create an inverse link function instance based on the provided name or path.

    Parameters
    ----------
    inverse_link_function:
        The link function or function path as a string.

    Returns
    -------
    :
        An instance of the specified inverse link function.

    Raises
    ------
    ValueError
        If the provided `inverse_link_function` name does not match any available link functions.
    """
    if inverse_link_function in LINK_NAME_TO_FUNC:
        return LINK_NAME_TO_FUNC[inverse_link_function]
    raise ValueError(
        f"Unknown link function: {inverse_link_function}. "
        f"Link function must be one of {AVAILABLE_INVERSE_LINK_FUNCTIONS}. "
        "If you want to use a custom link function, please provide it as a Callable. "
    )
