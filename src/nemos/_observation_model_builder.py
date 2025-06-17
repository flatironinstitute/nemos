from .observation_models import (
    BernoulliObservations,
    GammaObservations,
    PoissonObservations,
)

AVAILABLE_OBSERVATION_MODELS = ["Bernoulli", "Gamma", "Poisson"]


def instantiate_observation_model(
    observation_model: str, inverse_link_function: str = None
):
    """
    Create an observation model from a given name.

    Parameters
    ----------
    observation_model:
        The observation model as a string.
    inverse_link_function:
        The inverse link function to use for the observation model.
        If not provided, the default inverse link function of the observation model will be used.

    Returns
    -------
        The observation model instance with default parameters.

    Raises
    ------
    ValueError
        If the `observation_model` provided does not match to any available observation models.
    """
    if "." in observation_model:
        # extract the observation class if parsed as a full path
        observation_model = observation_model.split(".")[-1]

    # Remove "Observations" suffix if present
    observation_model = observation_model.removesuffix("Observations")

    # if inverse_link_function, pass it as a keyword argument, if not, use the default
    kwargs = (
        {"inverse_link_function": inverse_link_function}
        if inverse_link_function
        else {}
    )

    if observation_model == "Poisson":
        return PoissonObservations(**kwargs)
    elif observation_model == "Gamma":
        return GammaObservations(**kwargs)
    elif observation_model == "Bernoulli":
        return BernoulliObservations(**kwargs)
    else:
        raise ValueError(
            f"Unknown observation model: {observation_model}. "
            f"Observation model must be one of {AVAILABLE_OBSERVATION_MODELS}"
        )
