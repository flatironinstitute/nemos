"""Utility functions for creating parameter container objects."""

from .glm.params import GLMParams
from .glm_hmm.params import GLMHMMModelParams, GLMHMMParams
from .hmm.params import HMMParams

AVAILABLE_PARAM_CONTAINERS = [
    "GLMParams",
    "HMMParams",
    "GLMHMMParams",
    "GLMHMMModelParams",
]

# Mapping for O(1) lookup
_PARAM_CONTAINER_MAP = {
    "GLMParams": GLMParams,
    "HMMParams": HMMParams,
    "GLMHMMParams": GLMHMMParams,
    "GLMHMMModelParams": GLMHMMModelParams,
}


def instantiate_param_container(name: str, **kwargs):
    """
    Create a parameter container from a given name.

    Parameter containers are equinox modules holding a model's parameters, e.g.
    ``GLMParams(coef, intercept)``. They reach saved files as the values of other
    parameters — a structured ``GroupLasso`` mask, for instance — and are rebuilt from
    their class name at load time. Only the containers listed in
    ``AVAILABLE_PARAM_CONTAINERS`` can be built, so loading never calls an arbitrary
    constructor.

    Parameters
    ----------
    name :
        The string name of the container to create, either bare or as a full module
        path. Must name one of ``AVAILABLE_PARAM_CONTAINERS``.
    **kwargs :
        Additional keyword arguments are passed to the container constructor, one per
        field of the container.

    Returns
    -------
    :
        The parameter container instance.

    Raises
    ------
    ValueError
        If the ``name`` provided does not match any available container.
    """
    basename = name.split(".")[-1]
    if basename in _PARAM_CONTAINER_MAP:
        return _PARAM_CONTAINER_MAP[basename](**kwargs)

    raise ValueError(
        f"Unknown parameter container: {name}. "
        f"Container must be one of {AVAILABLE_PARAM_CONTAINERS} or their full module path."
    )
