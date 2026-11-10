"""Parameter containers for GPGLM models."""

from typing import Any

from ..params import ModelParams


class GPParams(ModelParams):
    """Parameter container for GPGLM models."""

    mapping_params: Any

    @staticmethod
    def regularizable_subtrees():
        """Filter regularizable subtrees."""
        return [lambda p: p.mapping_params]
