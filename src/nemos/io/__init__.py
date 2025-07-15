"""core tools for I/O OPERATIONS."""

from .io import inspect_npz, load_model

__all__ = ["load_model", "inspect_npz"]


def __dir__() -> list[str]:
    return __all__
