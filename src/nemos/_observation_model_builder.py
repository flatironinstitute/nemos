from .observation_models import (
    BernoulliObservations,
    GammaObservations,
    NegativeBinomialObservations,
    PoissonObservations,
)

AVAILABLE_OBSERVATION_MODELS = ["Bernoulli", "NegativeBinomial", "Gamma", "Poisson"]


def instantiate_observation_model(observation_model: str, **kwargs):
    """
    Create an observation model instance based on the provided name and optional parameters.

    Parameters
    ----------
    observation_model:
        The observation model as a string.
    **kwargs :
        Arbitrary keyword arguments to pass to the constructor of the observation model.

    Returns
    -------
    :
        An instance of the specified observation model class, initialized with the provided parameters.

    Raises
    ------
    ValueError
        If the provided `observation_model` name does not match any available observation models.
    """
    if "." in observation_model:
        # extract the observation class if parsed as a full path
        observation_model = observation_model.split(".")[-1]

    # Remove "Observations" suffix if present
    observation_model = observation_model.removesuffix("Observations")

    if observation_model == "Poisson":
        return PoissonObservations(**kwargs)
    elif observation_model == "Gamma":
        return GammaObservations(**kwargs)
    elif observation_model == "Bernoulli":
        return BernoulliObservations(**kwargs)
    elif observation_model == "NegativeBinomial":
        return NegativeBinomialObservations(**kwargs)
    else:
        raise ValueError(
            f"Unknown observation model: {observation_model}. "
            f"Observation model must be one of {AVAILABLE_OBSERVATION_MODELS}"
        )
