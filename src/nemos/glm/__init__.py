"""GLM modeling module."""

from .classifier_glm import ClassifierGLM, ClassifierPopulationGLM
from .glm import GLM, PopulationGLM

__all__ = ["GLM", "PopulationGLM", "ClassifierGLM", "ClassifierPopulationGLM"]


def __dir__():
    return __all__
