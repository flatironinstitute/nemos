"""Public available modules."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

import lazy_loader as _lazy

# All submodules are lazy-loaded for faster import times
__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


try:
    __version__ = _get_version("nemos")
except _PackageNotFoundError:
    # package is not installed
    pass
