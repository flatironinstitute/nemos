from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import jax

from ..observation_models import PoissonObservations
from ..regularizer import Ridge
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)

if TYPE_CHECKING:
    from ..glm import GLM, PopulationGLM


def glm_compute_optimal_stepsize_configs(
    model: Union[GLM, PopulationGLM]
) -> Tuple[Optional[Callable], Optional[Callable], Optional[float]]:
    """
    Compute configuration functions for optimal step size selection based on the model.

    This function returns a tuple of three elements that are used for configuring the
    optimal step size and batch size for variance reduced gradient (SVRG and
    ProxSVRG) algorithms. If the model is configured with specific solver names,
    the appropriate computation functions are returned. Additionally, it determines the
    smoothness and strong convexity constants based on the model's observation and regularizer.

    Parameters
    ----------
    model :
        The generalized linear model object for which the optimal step size and batch
        configuration need to be computed.

    Returns
    -------
    compute_optimal_params :
        A function to compute the optimal batch size and step size if the model
        is configured with the SVRG or ProxSVRG solver, None otherwise.

    compute_smoothness :
        A function to compute the smoothness constant of the loss function if the
        observation model uses a softplus inverse link function and is a Poisson
        observation model, None otherwise.

    strong_convexity :
        The strong convexity constant of the loss function if the model has a
        Ridge regularizer. If the model does not have a Ridge regularizer, this
        value will be None.

    """
    # initialize funcs and strong convexity constant
    compute_optimal_params = None
    compute_smoothness = None
    strong_convexity = (
        None if not isinstance(model.regularizer, Ridge) else model.regularizer_strength
    )

    # look-up table for selecting the optimal step and batch
    if model.solver_name in ("SVRG", "ProxSVRG"):
        compute_optimal_params = svrg_optimal_batch_and_stepsize

    # get the smoothness parameter compute function
    if model.observation_model.inverse_link_function is jax.nn.softplus and isinstance(
        model.observation_model, PoissonObservations
    ):
        compute_smoothness = glm_softplus_poisson_l_max_and_l

    return compute_optimal_params, compute_smoothness, strong_convexity
