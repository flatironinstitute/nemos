"""GLM modeling module."""

from .classifier_glm import ClassifierGLM, ClassifierPopulationGLM
from .glm import GLM, PopulationGLM


def __dir__():
    return glm.__all__ + classifier_glm.__all__
