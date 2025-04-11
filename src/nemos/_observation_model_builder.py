from .observation_models import (
    BernoulliObservations,
    GammaObservations,
    PoissonObservations,
)

AVAILABLE_OBSERVATION_MODELS = ["Bernoulli", "Gamma", "Poisson"]


def instantiate_observation_model(observation_model: str):
    """
    Create an observation model from a given name.

    Parameters
    ----------
    observation_model:
        The observation model as a string.

    Returns
    -------
        The observation model instance with default parameters.

    Raises
    ------
    ValueError
        If the `observation_model` provided does not match to any available observation models.
    """
    if observation_model == "Poisson":
        return PoissonObservations()
    elif observation_model == "Gamma":
        return GammaObservations()
    elif observation_model == "Bernoulli":
        return BernoulliObservations()
    else:
        raise ValueError(
            f"Unknown observation model: {observation_model}. "
            f"Observation model must be one of {AVAILABLE_OBSERVATION_MODELS}"
        )
