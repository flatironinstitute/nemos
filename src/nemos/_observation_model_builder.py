from .observation_models import GammaObservations, PoissonObservations

AVAILABLE_OBSERVATION_MODELS = ["Poisson", "Gamma"]


def instantiate_observation_model(observation_model: str):
    """
    Create an observation model from a given name.

    Parameters
    ----------
    observation_model:
        The observation model as a string.

    Returns
    -------
        The instantiated observation model.

    """
    if observation_model == "Poisson":
        return PoissonObservations()
    elif observation_model == "Gamma":
        return GammaObservations()
    else:
        raise ValueError(
            f"Unknown observation model: {observation_model}. "
            f"Observation model must be one of {AVAILABLE_OBSERVATION_MODELS}"
        )
