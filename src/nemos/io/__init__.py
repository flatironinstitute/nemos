"""core tools for I/O OPERATIONS."""

from .io import load_model

__all__ = ["load_model"]


def __dir__() -> list[str]:
    return __all__
